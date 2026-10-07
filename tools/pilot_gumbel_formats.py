"""Bounded BF16 -> GGUF format pilot for edited Flash Next tensors.

This adapts GSQ's Gumbel assignment and learned scales to the actual Q2_0
and IQ4_NL codebooks.  Its uniform reconstruction loss is a feasibility
test; it is not a calibrated whole-model GSQ run or a release builder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402


CODEBOOKS = {
    "Q2_0": [-1, 0, 1, 2],
    "IQ4_NL": [-127, -104, -83, -65, -49, -35, -22, -10,
               1, 13, 25, 38, 53, 69, 89, 113],
}
BLOCK_SIZES = {"Q2_0": 64, "IQ4_NL": 32}


def source_matrix(source: Path, tensor_name: str, expert: int | None, rows: int) -> torch.Tensor:
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    filename = index["weight_map"].get(tensor_name)
    if filename is None:
        raise ValueError(f"Source index does not contain {tensor_name}")
    path = source / filename
    with safe_open(path, framework="pt", device="cpu") as handle:
        slice_ = handle.get_slice(tensor_name)
        shape = slice_.get_shape()
        if len(shape) == 3:
            if expert is None or not 0 <= expert < shape[0]:
                raise ValueError(f"Choose an expert index in 0..{shape[0]-1}")
            matrix = slice_[expert]
        elif len(shape) == 2:
            if expert is not None:
                raise ValueError("--expert is only for fused expert matrices")
            matrix = slice_[:]
        else:
            raise ValueError(f"Expected a rank-2 or rank-3 tensor, got {shape}")
    return matrix[:rows].float().contiguous() if rows else matrix.float().contiguous()


def pack(codes: np.ndarray, scales: np.ndarray, qtype: str) -> np.ndarray:
    block = BLOCK_SIZES[qtype]
    codes = codes.reshape(-1, block).astype(np.uint8)
    scale_bytes = scales.astype("<f2").view(np.uint8).reshape(-1, 2)
    if qtype == "Q2_0":
        quant_bytes = codes[:, 0::4] | (codes[:, 1::4] << 2)
        quant_bytes |= (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)
    else:
        quant_bytes = codes[:, :16] | (codes[:, 16:] << 4)
    return np.concatenate((scale_bytes, quant_bytes), axis=1).reshape(-1)


def unpack(packed: np.ndarray, qtype: str, shape: tuple[int, int]) -> np.ndarray:
    if qtype == "IQ4_NL":
        row_bytes = shape[1] // 32 * 18
        return gguf.quants.dequantize(
            packed.reshape(shape[0], row_bytes), gguf.GGMLQuantizationType.IQ4_NL
        ).reshape(shape)
    blocks = packed.reshape(-1, 18)
    scale = blocks[:, :2].copy().view("<f2").astype(np.float32)
    byte = blocks[:, 2:]
    codes = np.stack([(byte >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1)
    return ((codes.reshape(-1, 64).astype(np.float32) - 1) * scale).reshape(shape)


def train_one(weight: torch.Tensor, qtype: str, steps: int, device: str,
              importance: torch.Tensor | None = None) -> tuple[dict[str, object], np.ndarray]:
    block = BLOCK_SIZES[qtype]
    if weight.shape[1] % block:
        raise ValueError(f"{qtype} requires row width divisible by {block}")
    start = time.monotonic()
    w = weight.to(device=device).reshape(-1, block)
    factor = (importance.to(device=device).reshape(-1, block).clamp_min(0)
              if importance is not None else None)
    grid = torch.tensor(CODEBOOKS[qtype], dtype=torch.float32, device=device)
    peak = w.abs().amax(dim=1).clamp_min(1e-9)
    initial_scale = peak / max(abs(grid.min().item()), abs(grid.max().item()))
    normalized = w / initial_scale[:, None]
    distance = (normalized[:, :, None] - grid[None, None, :]).square()
    initial_codes = distance.argmin(dim=-1)
    initial_quant = grid[initial_codes] * initial_scale[:, None].half().float()
    initial_squared = (initial_quant - w).square()
    initial_error = initial_squared.sum() / w.square().sum()
    initial_weighted = ((initial_squared * factor).sum() if factor is not None
                        else initial_squared.sum())

    logits = torch.nn.Parameter((-distance / 6).detach())
    log_scale = torch.nn.Parameter(initial_scale.log().detach())
    optimizer = torch.optim.Adam([
        {"params": logits, "lr": 0.04},
        {"params": log_scale, "lr": 0.01},
    ])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = 1.0 - 0.8 * step / max(1, steps - 1)
        probabilities = functional.gumbel_softmax(logits, tau=temperature, dim=-1)
        approximation = (probabilities * grid).sum(dim=-1) * log_scale.exp()[:, None]
        squared = (approximation - w).square()
        loss = (squared * factor).mean() / factor.mean().clamp_min(1e-8) if factor is not None else squared.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(-25, 10)

    with torch.no_grad():
        learned_codes = logits.argmax(dim=-1)
        learned_scales = log_scale.exp().half()
        learned = grid[learned_codes] * learned_scales.float()[:, None]
        learned_squared = (learned - w).square()
        learned_error = learned_squared.sum() / w.square().sum()
        learned_weighted = ((learned_squared * factor).sum() if factor is not None
                            else learned_squared.sum())
        use_learned = bool(learned_weighted < initial_weighted)
        codes = (learned_codes if use_learned else initial_codes).cpu().numpy()
        scales = (learned_scales if use_learned else initial_scale.half()).cpu().numpy()
        blob = pack(codes, scales, qtype)
        roundtrip = unpack(blob, qtype, tuple(weight.shape))
        source = weight.numpy()
        source_power = float(np.square(source, dtype=np.float64).sum())
        selected_sse = float(np.square(roundtrip - source, dtype=np.float64).sum())
        roundtrip_error = selected_sse / source_power
        factor_np = factor.cpu().numpy().reshape(source.shape) if factor is not None else None
        weighted_sse = (float((np.square(roundtrip - source, dtype=np.float64) * factor_np).sum())
                        if factor_np is not None else selected_sse)
        weighted_power = (float((np.square(source, dtype=np.float64) * factor_np).sum())
                          if factor_np is not None else source_power)
        expected_error = float((learned_error if use_learned else initial_error).item())
        if not np.isclose(roundtrip_error, expected_error, atol=1e-5, rtol=1e-4):
            raise ValueError(f"{qtype} packed roundtrip mismatch: {roundtrip_error} vs {expected_error}")
    result = {
        "format": qtype,
        "source_shape": list(weight.shape),
        "packed_bytes": int(blob.nbytes),
        "initial_relative_mse": float(initial_error.item()),
        "learned_relative_mse": float(learned_error.item()),
        "selected_relative_mse": roundtrip_error,
        "selected_sse": selected_sse,
        "source_power": source_power,
        "weighted_sse": weighted_sse,
        "weighted_power": weighted_power,
        "learned_selected": use_learned,
        "steps": steps,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
    }
    return result, blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source"))
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--expert", type=int)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--format", choices=tuple(CODEBOOKS), nargs="+", default=list(CODEBOOKS))
    parser.add_argument("--output", type=Path, default=Path("data/pilot"))
    args = parser.parse_args()
    torch.manual_seed(1729)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix = source_matrix(args.source, args.tensor, args.expert, args.rows)
    if not torch.isfinite(matrix).all():
        raise ValueError("Non-finite BF16 source weights")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"tensor": args.tensor, "expert": args.expert, "device": device,
              "source_shape": list(matrix.shape), "results": []}
    stem = args.tensor.replace(".", "_") + (f"_expert{args.expert}" if args.expert is not None else "")
    for qtype in args.format:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result, blob = train_one(matrix, qtype, args.steps, device)
        blob.tofile(args.output / f"{stem}_{qtype}.bin")
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    path = args.output / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Pilot report: {path}", flush=True)


if __name__ == "__main__":
    main()
