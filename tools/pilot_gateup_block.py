"""Bounded native IQ gate/up pilot with learned global block scales.

The objective is the complete captured routed MoE sum against BF16 gate/up
and retained ablated down weights. This is a mixed block reference and a
GSQ-inspired scale adaptation, not full GSQ, upstream RCO, or a BF16 model.
Only selected expert prototypes are saved; this script never assembles a model.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf
from tools.read_gateup_capture import sha256, verify_calibration_manifest, verify_capture_linkage

FOLDER = ROOT / "data/optimization-final-gateup"
BIN = ROOT / "third_party/llama.cpp/build-ninja-release/bin"
CONTROL = Path("models/flash-next-uncensored-gsq-rco-models/Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf")
FORMATS = ("IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_XXS")
PARITY_RMSE_LIMIT = .03
SCALE_RATIO_BOUND = 2.0
SCALE_LEARNING_RATE = .003


def write_json(path: Path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class NativeQuantizer:
    def __init__(self):
        self.directory = os.add_dll_directory(str(BIN))
        self.library = ctypes.CDLL(str(BIN / "ggml-base.dll"))
        self.fn = self.library.ggml_quantize_chunk
        self.fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.fn.restype = ctypes.c_size_t

    def quantize(self, matrix: np.ndarray, fmt: str, importance: np.ndarray):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        importance = np.ascontiguousarray(importance, dtype=np.float32)
        if matrix.shape != (640, 2560) or importance.shape != (2560,):
            raise ValueError("Native gate/up quantizer dimensions mismatch")
        if not np.isfinite(matrix).all() or not np.isfinite(importance).all() or np.any(importance <= 0):
            raise ValueError("Nonfinite weights or invalid native importance")
        kind = getattr(gguf.GGMLQuantizationType, fmt)
        block, size = gguf.GGML_QUANT_SIZES[kind]
        if block != 256:
            raise ValueError("Only verified 256-element IQ global-scale layouts are supported")
        blob = np.empty(matrix.size // block * size, dtype=np.uint8)
        length = self.fn(int(kind), matrix.ctypes.data, blob.ctypes.data, 0,
                         matrix.shape[0], matrix.shape[1], importance.ctypes.data)
        if length != blob.nbytes:
            raise ValueError("Incorrect native packed byte count")
        return blob


def decode(blob, fmt: str, shape=(640, 2560)):
    blob = np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1)
    kind = getattr(gguf.GGMLQuantizationType, fmt)
    block, size = gguf.GGML_QUANT_SIZES[kind]
    if shape[1] % block or blob.size != math.prod(shape) // block * size:
        raise ValueError("Packed expert dimensions/byte count mismatch")
    if fmt == "Q2_0":
        from tools.pilot_gumbel_formats import unpack
        result = unpack(blob, fmt, shape)
    else:
        result = gguf.quants.dequantize(blob.reshape(shape[0], shape[1] // block * size), kind).reshape(shape)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite packed reconstruction")
    return result


def scale_components(blob, fmt: str):
    if fmt not in FORMATS:
        raise ValueError("No global-scale adapter for requested codebook")
    _, block_bytes = gguf.GGML_QUANT_SIZES[getattr(gguf.GGMLQuantizationType, fmt)]
    blocks = np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1, block_bytes)
    scales = blocks[:, :2].copy().view("<f2").reshape(640, 10).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("Nonfinite native IQ global scale")
    unit = blocks.copy()
    unit[:, :2] = np.full((len(unit), 1), 1., dtype="<f2").view(np.uint8)
    basis = decode(unit, fmt)
    expected = basis.reshape(640, 10, 256) * scales[:, :, None]
    actual = decode(blocks, fmt)
    if not np.allclose(expected.reshape(640, 2560), actual, rtol=2e-6, atol=1e-8):
        raise ValueError("IQ codebook/global-scale decomposition does not roundtrip")
    return basis, scales


def pack_scales(original, scales, fmt: str):
    _, size = gguf.GGML_QUANT_SIZES[getattr(gguf.GGMLQuantizationType, fmt)]
    blocks = np.ascontiguousarray(original, dtype=np.uint8).reshape(-1, size).copy()
    scales = np.asarray(scales, dtype="<f2").reshape(-1, 1)
    if len(scales) != len(blocks) or not np.isfinite(scales).all():
        raise ValueError("Incorrect learned global scale count or nonfinite values")
    old_codes = hashlib.sha256(blocks[:, 2:].tobytes()).hexdigest()
    blocks[:, :2] = scales.view(np.uint8)
    if hashlib.sha256(blocks[:, 2:].tobytes()).hexdigest() != old_codes:
        raise ValueError("Scale adaptation changed fixed native codebook assignments")
    blob = blocks.reshape(-1)
    decode(blob, fmt)
    return blob


def contribution(x, gate, up, down, probability):
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T * probability[:, None]


def numpy_contribution(x, gate, up, down, probability, device):
    with torch.no_grad():
        operands = [torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=torch.float32)
                    for value in (x, gate, up, down, probability)]
        result = contribution(*operands).cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite nonlinear expert output")
    return result


def refine_scales(gate_blob, up_blob, fmt, x, probability, down,
                  residual_base, teacher, steps, batch_size, device, seed,
                  check_deadline=None):
    basis_g, initial_g = scale_components(gate_blob, fmt)
    basis_u, initial_u = scale_components(up_blob, fmt)
    basis = torch.as_tensor(np.stack([basis_g, basis_u]).reshape(2, 640, 10, 256), device=device)
    initial = torch.as_tensor(np.stack([initial_g, initial_u]), device=device)
    ratios = torch.nn.Parameter(torch.zeros_like(initial))
    optimiser = torch.optim.Adam([ratios], lr=SCALE_LEARNING_RATE)
    data = [torch.as_tensor(np.ascontiguousarray(value), device=device, dtype=torch.float32)
            for value in (x, probability, down, residual_base, teacher)]
    tx, tp, td, residual, target = data
    denominator = target.square().mean().clamp_min(1e-12)
    rng = np.random.default_rng(seed)
    losses = []
    bound = math.log(SCALE_RATIO_BOUND)
    def training_checkpoint(step):
        with torch.no_grad():
            packed_scales = (initial * ratios.exp()).half().float()
            matrix = (basis * packed_scales[:, :, :, None]).reshape(2, 640, 2560)
            prediction = residual + contribution(tx, matrix[0], matrix[1], td, tp)
            error = float((prediction - target).square().mean() / denominator)
            if not math.isfinite(error):
                raise ValueError("Nonfinite hard-rounded training checkpoint")
            losses.append({"step": step, "normalised_full_training_loss": error})
            return error, packed_scales.half().cpu().numpy()
    best_loss, final = training_checkpoint(0)
    selected_step = 0
    for step in range(steps):
        if check_deadline is not None:
            check_deadline()
        indices = torch.as_tensor(rng.choice(len(x), min(batch_size, len(x)), replace=False), device=device)
        optimiser.zero_grad(set_to_none=True)
        scales = initial * ratios.exp()
        # STE preserves the packed FP16 scale's forward value while keeping a
        # derivative for the bounded log-scale parameter.
        rounded = scales.half().float()
        scales = scales + (rounded - scales).detach()
        matrix = (basis * scales[:, :, :, None]).reshape(2, 640, 2560)
        prediction = residual[indices] + contribution(tx[indices], matrix[0], matrix[1], td, tp[indices])
        loss = (prediction - target[indices]).square().mean() / denominator
        loss = loss + 1e-4 * ratios.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite nonlinear block objective")
        loss.backward()
        if ratios.grad is None or not torch.isfinite(ratios.grad).all():
            raise ValueError("Invalid global-scale gradient")
        optimiser.step()
        with torch.no_grad():
            ratios.clamp_(-bound, bound)
        if (step + 1) % 8 == 0 or step + 1 == steps:
            error, checkpoint = training_checkpoint(step + 1)
            if error < best_loss:
                best_loss, final = error, checkpoint
                selected_step = step + 1
    blobs = [pack_scales(gate_blob, final[0], fmt), pack_scales(up_blob, final[1], fmt)]
    for blob, base, scale in zip(blobs, (gate_blob, up_blob), final):
        decoded = decode(blob, fmt)
        unit, _ = scale_components(base, fmt)
        expected = (unit.reshape(640, 10, 256) * scale.astype(np.float32)[:, :, None]).reshape(640, 2560)
        if not np.allclose(decoded, expected, rtol=2e-6, atol=1e-8):
            raise ValueError("Hard-packed learned-scale forward mismatch")
    return blobs, {"steps": steps, "batch_size": batch_size, "ratio_bound": SCALE_RATIO_BOUND,
                   "learning_rate": SCALE_LEARNING_RATE, "selected_training_checkpoint": selected_step,
                   "checkpoint_rule": "Best complete training-mixture error among native step0 and fixed8-step checkpoints; no validation data used for optimisation/checkpoint choice",
                   "fixed_codebook_assignments": True, "fp16_scale_forward": True,
                   "training_trace": losses}


def routes(capture, expert):
    rows, ranks = np.nonzero(capture.ids == expert)
    return rows, capture.weights[rows, ranks]


def select_experts(train, validation, count):
    nt, mass = train.coverage()
    nv, _ = validation.coverage()
    eligible = [int(expert) for expert in np.argsort(-mass, kind="stable") if nt[expert] >= 32 and nv[expert] >= 16]
    if len(eligible) < count or count % 3:
        raise ValueError(f"Need {count} covered experts divisible by3; only{len(eligible)} meet32/16 coverage")
    thirds = np.array_split(np.array(eligible), 3)
    each = count // 3
    selected = []
    for band, group in zip(("high", "middle", "lower"), thirds):
        if len(group) < each:
            raise ValueError("Insufficient coverage in route-mass band")
        indices = np.linspace(0, len(group) - 1, each).round().astype(int)
        for expert in group[indices]:
            selected.append({"expert": int(expert), "band": band,
                             "train_occurrences": int(nt[expert]), "validation_occurrences": int(nv[expert]),
                             "train_route_mass": float(mass[expert])})
    if len({item["expert"] for item in selected}) != count:
        raise ValueError("Expert selection duplicated an expert")
    return selected


def tensor_expert(tensor, expert, expected_shape):
    if tuple(tensor.shape.tolist()) != expected_shape:
        raise ValueError(f"Unexpected GGML tensor layout: {tensor.name}")
    rows, width = expected_shape[1], expected_shape[0]
    raw = np.ascontiguousarray(tensor.data[expert], dtype=np.uint8).reshape(-1)
    return decode(raw, tensor.tensor_type.name, (rows, width)), raw


def load_imatrix(path, layers, linkage):
    reader = gguf.GGUFReader(path)
    datasets = reader.get_field("imatrix.datasets")
    chunks = reader.get_field("imatrix.chunk_count")
    chunk_size = reader.get_field("imatrix.chunk_size")
    if datasets is None or chunks is None or chunk_size is None:
        raise ValueError("Training importance matrix lacks capture metadata")
    if int(chunks.contents()) != linkage["chunks"] or int(chunk_size.contents()) != 512:
        raise ValueError("Training importance matrix chunk metadata differs from linked capture")
    if [Path(value).resolve() for value in datasets.contents()] != [Path(linkage["corpus"]).resolve()]:
        raise ValueError("Training importance matrix dataset differs from linked frozen corpus")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    result = {}
    for layer in layers:
        for half in ("gate", "up"):
            name = f"blk.{layer}.ffn_{half}_exps.weight"
            value = tensors[name + ".in_sum2"].data
            counts = tensors[name + ".counts"].data.reshape(-1)
            if value.shape != (512, 2560) or counts.shape != (512,):
                raise ValueError("Calibration importance shape mismatch")
            if not np.isfinite(value).all() or not np.isfinite(counts).all() or np.any(value < 0) or np.any(counts < 0):
                raise ValueError("Nonfinite or negative training importance statistics")
            if float(counts.sum()) != linkage["chunks"] * 512 * 10:
                raise ValueError("Gate/up importance routing count differs from linked capture")
            result[(layer, half)] = (np.array(value, dtype=np.float32, copy=True), np.array(counts, copy=True))
    return result


def metrics(prediction, target):
    sse = float(np.square(prediction - target, dtype=np.float64).sum())
    power = float(np.square(target, dtype=np.float64).sum())
    return {"output_sse": sse, "output_power": power,
            "relative_mse": sse / max(power, 1e-20), "relative_rmse": math.sqrt(sse / max(power, 1e-20))}


def select_training_choice(control_training_score, candidate_scores):
    """Choose a coordinate update using training only; validation is diagnostic.

    A single pure selector also makes the boundary auditable: candidate_scores
    may contain independent validation diagnostics, but only train_full_mixture
    is accessed. Ties retain the earlier candidate, beginning with control.
    """
    best = "retained_control"
    best_sse = float(control_training_score["output_sse"])
    if not math.isfinite(best_sse) or best_sse < 0:
        raise ValueError("Invalid control training-mixture score")
    for method, record in candidate_scores.items():
        sse = float(record["train_full_mixture"]["output_sse"])
        if not math.isfinite(sse) or sse < 0:
            raise ValueError("Invalid candidate training-mixture score")
        if sse < best_sse:
            best, best_sse = method, sse
    return best, best_sse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-inputs", type=Path, default=FOLDER / "capture-train")
    parser.add_argument("--validation-inputs", type=Path, default=FOLDER / "capture-validation")
    parser.add_argument("--sources", type=Path, default=FOLDER / "sources")
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, default=FOLDER / "pilot")
    parser.add_argument("--layers", default="8,28,47")
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=["IQ2_S", "IQ2_XS", "IQ3_XXS"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--imatrix", type=Path)
    args = parser.parse_args()
    if not (1 <= args.steps <= 64 and 1 <= args.batch_size <= 32 and 3 <= args.experts <= 12 and args.experts % 3 == 0):
        parser.error("Pilot bounds are64 steps/batch32/12covered experts; expert count divisible by3")
    layers = [int(item) for item in args.layers.split(",")]
    if set(layers) - {8, 28, 47} or len(set(layers)) != len(layers):
        parser.error("Only predeclared pilot layers8,28,47")
    budget = json.loads((FOLDER / "budget.json").read_text(encoding="utf-8-sig"))
    deadline = datetime.fromisoformat(budget["pilot_deadline_utc"].replace("Z", "+00:00"))
    def check_deadline():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Gate/up pilot deadline reached")
    check_deadline()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently switch pilot runtime")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("Completed pilot exists; verify its provenance instead of rerunning")
    calibration = verify_calibration_manifest(FOLDER / "calibration/manifest.json")
    preflight = json.loads((FOLDER / "source-preflight.json").read_text(encoding="utf-8-sig"))
    identity = preflight["control_identity"]
    if args.control.resolve() != Path(identity["path"]).resolve() or args.control.stat().st_size != identity["bytes"] or args.control.stat().st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("Control changed since full checksum receipt")
    train_captures, train_linkage = verify_capture_linkage(args.train_inputs, "train", FOLDER / "calibration/manifest.json", identity, layers)
    validation_captures, validation_linkage = verify_capture_linkage(args.validation_inputs, "validation", FOLDER / "calibration/manifest.json", identity, layers)
    for key in ("calibration_cpu_moe", "production_cpu_moe", "calibration_load_mode", "engine_sha256", "collector_sha256"):
        if train_linkage[key] != validation_linkage[key]:
            raise ValueError(f"Training and validation capture settings disagree: {key}")
    source_index = json.loads((ROOT / "data/source/model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    source_records = {}
    for layer in layers:
        check_deadline()
        row = next(row for row in preflight["layers"] if row["layer"] == layer)
        path = args.sources / row["hf_file"]
        if path.stat().st_size != row["pinned_file"]["size"]:
            raise ValueError("BF16 source length differs from pinned shard")
        # Verify the complete pinned pilot source set before accepting it.
        digest = sha256(path)
        check_deadline()
        if digest != row["pinned_file"]["lfs_sha256"]:
            raise ValueError("BF16 source checksum differs from pinned revision")
        name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        if source_index[name] != path.name:
            raise ValueError("BF16 source index mismatch")
        with safe_open(path, framework="pt", device="cpu") as handle:
            source_slice = handle.get_slice(name)
            if source_slice.get_shape() != [512, 1280, 2560] or source_slice.get_dtype() != "BF16":
                raise ValueError("Unexpected BF16 fused expert layout")
        source_records[layer] = {"path": str(path), "sha256": digest, "tensor": name,
                                 "source_revision": preflight["source_revision"]}
    reader = gguf.GGUFReader(args.control)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    imatrix_path = args.imatrix or args.train_inputs / "imatrix.gguf"
    if imatrix_path.resolve() != Path(train_linkage["imatrix_path"]).resolve() or sha256(imatrix_path) != train_linkage["imatrix_sha256"]:
        raise ValueError("Pilot importance matrix is not the linked training capture's output")
    importance = load_imatrix(imatrix_path, layers, train_linkage)
    native = NativeQuantizer()
    provenance = {"method": "Native IQ2 initializers and GSQ-inspired fixed-codebook global-scale learning against routed nonlinear MoE output",
                  "full_GSQ_RCO": False, "reference": "BF16 gate/up + retained quantized ablated down, fixed control routing and inputs; mixed block teacher",
                  "capture_collector_sha256": sha256(ROOT / "tools/gateup_capture.h"),
                  "builder_sha256": sha256(Path(__file__)), "reader_sha256": sha256(ROOT / "tools/read_gateup_capture.py"),
                  "quantizer_sha256": sha256(BIN / "ggml-base.dll"), "imatrix_sha256": sha256(imatrix_path),
                  "control": identity, "sources": source_records, "calibration": calibration,
                  "capture_linkage": {"train": train_linkage, "validation": validation_linkage},
                  "parity_relative_rmse_limit": PARITY_RMSE_LIMIT, "runtime_device": args.device,
                  "steps": args.steps, "batch_size": args.batch_size, "scale_ratio_bound": SCALE_RATIO_BOUND,
                  "scale_learning_rate": SCALE_LEARNING_RATE,
                  "coordinate_selection": "Complete training-mixture SSE only, including retained-control fallback; independent validation is reporting only and never feeds the next training residual",
                  "prototype_scope": "Only selected experts change; all remaining experts retain control. Mixed-format experts within a fused GGUF tensor cannot be emitted as this prototype. Whole-format deployment must evaluate every expert.",
                  "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    write_json(args.output / "manifest.json", provenance)
    started = time.monotonic()
    reports = []
    for layer in layers:
        check_deadline()
        train = train_captures[layer]
        valid = validation_captures[layer]
        if train.path.resolve() == valid.path.resolve() or train.sha256 == valid.sha256 or train.seed == valid.seed:
            raise ValueError("Training/validation captures are not independently generated")
        selected = select_experts(train, valid, args.experts)
        captures = (train, valid)
        control_mix = [np.zeros_like(capture.y) for capture in captures]
        teacher_mix = [np.zeros_like(capture.y) for capture in captures]
        name = source_records[layer]["tensor"]
        selected_cache = {}
        layer_started = time.monotonic()
        with safe_open(source_records[layer]["path"], framework="pt", device="cpu") as source:
            source_slice = source.get_slice(name)
            for expert in range(512):
                check_deadline()
                routing = [routes(capture, expert) for capture in captures]
                if not any(len(rows) for rows, _ in routing):
                    continue
                bf = source_slice[expert].float().numpy()
                if bf.shape != (1280, 2560) or not np.isfinite(bf).all():
                    raise ValueError("Nonfinite BF16 gate/up expert")
                gate, gate_raw = tensor_expert(tensors[f"blk.{layer}.ffn_gate_exps.weight"], expert, (2560, 640, 512))
                up, up_raw = tensor_expert(tensors[f"blk.{layer}.ffn_up_exps.weight"], expert, (2560, 640, 512))
                down, down_raw = tensor_expert(tensors[f"blk.{layer}.ffn_down_exps.weight"], expert, (640, 2560, 512))
                for index, (capture, (rows, probability)) in enumerate(zip(captures, routing)):
                    if not len(rows):
                        continue
                    x = capture.x[rows]
                    control_mix[index][rows] += numpy_contribution(x, gate, up, down, probability, args.device)
                    teacher_mix[index][rows] += numpy_contribution(x, bf[:640], bf[640:], down, probability, args.device)
                if any(item["expert"] == expert for item in selected):
                    selected_cache[expert] = {"bf": bf.copy(), "gate": gate, "up": up, "down": down,
                                             "gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
                                             "up_sha256": hashlib.sha256(up_raw).hexdigest(),
                                             "down_sha256": hashlib.sha256(down_raw).hexdigest()}
                if expert % 64 == 0:
                    write_json(args.output / "status.json", {"status": "running", "stage": "reconstructing captured control and mixed reference",
                               "layer": layer, "expert": expert, "elapsed_seconds": time.monotonic() - started})
                    print(json.dumps({"stage": "reference", "layer": layer, "expert": expert}), flush=True)
        parity = [metrics(prediction, capture.y) for prediction, capture in zip(control_mix, captures)]
        if any(item["relative_rmse"] > PARITY_RMSE_LIMIT for item in parity):
            write_json(args.output / f"layer{layer}-parity-failure.json", {"parity": parity, "limit": PARITY_RMSE_LIMIT})
            raise ValueError("Decoded control mixture fails predeclared3% captured-output parity")
        reference_path = args.output / f"layer{layer}-mixed-reference.npz"
        np.savez(reference_path, control_train=control_mix[0], control_validation=control_mix[1],
                 teacher_train=teacher_mix[0], teacher_validation=teacher_mix[1])
        layer_report = {"layer": layer, "train_capture": train.receipt(), "validation_capture": valid.receipt(),
                        "reference_arrays": {"path": str(reference_path), "sha256": sha256(reference_path)},
                        "selected_experts": selected, "control_parity": parity,
                        "mixed_reference_control_error": [metrics(prediction, target) for prediction, target in zip(control_mix, teacher_mix)],
                        "reference_seconds": time.monotonic() - layer_started, "scenarios": []}
        for fmt in args.formats:
            # Each format is a separate coordinate experiment. Every coordinate
            # choice uses training only. Validation observes the chosen updates
            # but cannot choose a payload or alter the next training residual.
            current = [array.copy() for array in control_mix]
            scenario = {"format": fmt, "experts": []}
            for item in selected:
                check_deadline()
                expert = item["expert"]
                cached = selected_cache[expert]
                routing = [routes(capture, expert) for capture in captures]
                old_outputs = [numpy_contribution(capture.x[rows], cached["gate"], cached["up"], cached["down"], probability, args.device)
                               for capture, (rows, probability) in zip(captures, routing)]
                base = [prediction[rows] - old for prediction, (rows, _), old in zip(current, routing, old_outputs)]
                choices = {}
                timings = {}
                expert_started = time.monotonic()
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                for method in ("native", "native_calibrated"):
                    timestamp = time.monotonic()
                    factors = []
                    for half in ("gate", "up"):
                        values, counts = importance[(layer, half)]
                        if counts[expert] <= 0:
                            raise ValueError("Selected expert missing calibration importance")
                        factor = values[expert] / float(counts[expert]) if method == "native_calibrated" else np.ones(2560, np.float32)
                        factors.append(np.clip(factor, 1e-8, 1e8).astype(np.float32))
                    blobs = [native.quantize(cached["bf"][:640], fmt, factors[0]), native.quantize(cached["bf"][640:], fmt, factors[1])]
                    choices[method] = (blobs, {"initializer": method})
                    timings[method] = time.monotonic() - timestamp
                    rows, probability = routing[0]
                    timestamp = time.monotonic()
                    refined, trace = refine_scales(*blobs, fmt, train.x[rows], probability, cached["down"], base[0],
                                                  teacher_mix[0][rows], args.steps, args.batch_size, args.device,
                                                  1729 + layer * 512 + expert,
                                                  check_deadline=check_deadline)
                    choices[method + "_scale_refined"] = (refined, {"initializer": method, **trace})
                    timings[method + "_scale_refined"] = time.monotonic() - timestamp
                scores = {}
                candidate_outputs = {}
                control_training_score = metrics(current[0], teacher_mix[0])
                control_validation_score = metrics(current[1], teacher_mix[1])
                for method, (blobs, details) in choices.items():
                    gate, up = [decode(blob, fmt) for blob in blobs]
                    outputs = [numpy_contribution(capture.x[rows], gate, up, cached["down"], probability, args.device)
                               for capture, (rows, probability) in zip(captures, routing)]
                    trial_valid = current[1].copy()
                    trial_valid[routing[1][0]] = base[1] + outputs[1]
                    valid_score = metrics(trial_valid, teacher_mix[1])
                    train_score = metrics(base[0] + outputs[0], teacher_mix[0][routing[0][0]])
                    trial_train = current[0].copy()
                    trial_train[routing[0][0]] = base[0] + outputs[0]
                    train_full_score = metrics(trial_train, teacher_mix[0])
                    payloads = []
                    for half, blob in zip(("gate", "up"), blobs):
                        path = args.output / f"layer{layer}-expert{expert}-{fmt}-{method}-{half}.bin"
                        blob.tofile(path)
                        payloads.append({"path": str(path), "bytes": blob.nbytes, "sha256": sha256(path),
                                         "shape": [640, 2560], "format": fmt})
                    scores[method] = {"train_routed_rows": train_score, "train_full_mixture": train_full_score,
                                      "validation_full_mixture": valid_score,
                                      "payloads": payloads, "details": details, "seconds": timings[method]}
                    candidate_outputs[method] = outputs
                best, best_training_sse = select_training_choice(control_training_score, scores)
                if best != "retained_control":
                    for index in range(2):
                        current[index][routing[index][0]] = base[index] + candidate_outputs[best][index]
                result = {**item, "selected": best, "source_bf16_sha256": hashlib.sha256(cached["bf"].tobytes()).hexdigest(),
                          "retained_control_packed_hashes": {half: cached[half + "_sha256"] for half in ("gate", "up", "down")},
                          "selection_reference": "Complete training-mixture SSE only; validation metrics never select coordinate updates",
                          "before_training": control_training_score, "before_validation": control_validation_score, "scores": scores,
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                          "seconds": time.monotonic() - expert_started}
                scenario["experts"].append(result)
                write_json(args.output / f"layer{layer}-{fmt}-expert{expert}.json", result)
                write_json(args.output / "status.json", {"status": "running", "stage": "native and learned-scale coordinate pilot",
                           "layer": layer, "format": fmt, "expert": expert, "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"stage": "scale-pilot", "layer": layer, "format": fmt, "expert": expert,
                                  "selected": best,
                                  "training_relative_rmse": math.sqrt(best_training_sse / max(control_training_score["output_power"], 1e-20)),
                                  "validation_relative_rmse": (control_validation_score if best == "retained_control" else scores[best]["validation_full_mixture"])["relative_rmse"],
                                  "seconds": result["seconds"]}), flush=True)
            scenario["after_mixed_reference_error"] = [metrics(prediction, target) for prediction, target in zip(current, teacher_mix)]
            scenario["selected_experts_training_improved"] = sum(item["selected"] != "retained_control" for item in scenario["experts"])
            scenario["selected_updates_validation_improved"] = sum(
                item["selected"] != "retained_control" and
                item["scores"][item["selected"]]["validation_full_mixture"]["output_sse"] < item["before_validation"]["output_sse"]
                for item in scenario["experts"])
            scenario["extrapolation_warning"] = "Other500 experts remain control in this prototype. Cannot infer whole-layer/model quality or construct a GGUF with mixed expert formats inside one tensor."
            layer_report["scenarios"].append(scenario)
        layer_report["seconds"] = time.monotonic() - layer_started
        write_json(args.output / f"layer{layer}-summary.json", layer_report)
        reports.append(layer_report)
        del selected_cache
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    projections = {}
    for fmt in args.formats:
        scenarios = [scenario for layer_report in reports for scenario in layer_report["scenarios"] if scenario["format"] == fmt]
        samples = [expert["seconds"] for scenario in scenarios for expert in scenario["experts"]]
        average = float(np.mean(samples))
        projections[fmt] = {"measured_experts": len(samples), "mean_coordinate_seconds": average,
                            "all48layers_512experts_coordinate_seconds_if_same_cost": average * 48 * 512,
                            "reference_reconstruction_seconds_if_same_cost": float(np.mean([record["reference_seconds"] for record in reports])) * 48,
                            "assumption": "All experts trained with both uniform and calibrated initializers at fixed64 steps; poorly covered experts would stay conservative. Small sample and different source I/O limit extrapolation.",
                            "excluded": ["remaining BF16 source download", "whole-model assembly", "runtime conversion", "development and final confirmation", "coding/behavior/memory checks"]}
    summary = {**provenance, "status": "complete", "layers": reports,
               "elapsed_seconds": time.monotonic() - started,
               "completion_projection": projections,
               "projection": "Reference and12-expert timing measured per layer; extrapolating512-expert learned-scale work multiplies coordinate time by512/12 and excludes full source download/model assembly/end-to-end validation.",
               "completed_utc": datetime.now(timezone.utc).isoformat(), "models_assembled": 0,
               "confirmation_used": False, "public_acceptance": False}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "status.json", {"status": "complete", "elapsed_seconds": summary["elapsed_seconds"], "summary": str(args.output / "summary.json")})
    print(json.dumps({"status": "complete", "summary": str(args.output / "summary.json"), "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
