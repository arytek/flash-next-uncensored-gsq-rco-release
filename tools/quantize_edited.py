"""Resumably quantize only the 146 ablation-edited BF16 text tensors.

Fused expert-down tensors get Q2_0 and IQ4_NL variants with the bounded
Gumbel/scale adapter.  Other edited text tensors get Q8_0.  The published
GSQ-RCO packed tensors are assembled separately, without requantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

import gguf
import numpy as np
import torch
from safetensors import safe_open

from tools.assemble_hybrid import edited_names
from tools.download_ablated_text import selected_shards
from tools.pilot_gumbel_formats import train_one


REPO_REVISION = "deb02632504bb214702bc28b0381a93d3112f500"
CALIBRATION_REVISION = "8178d96379feb4c484ad5de8cb3ae61101ae8dfb"
QTYPE_LAYOUT = {"Q2_0": (64, 18), "IQ4_NL": (32, 18), "Q8_0": (32, 34)}


def source_name(gguf_name: str) -> str:
    prefix = "model.language_model."
    if gguf_name == "token_embd.weight":
        return prefix + "embed_tokens.weight"
    pieces = gguf_name.split(".")
    if len(pieces) < 4 or pieces[0] != "blk":
        raise ValueError(f"Unrecognized edited GGUF tensor: {gguf_name}")
    layer = int(pieces[1])
    stem = prefix + f"layers.{layer}."
    suffix = ".".join(pieces[2:])
    conversions = {
        "ple_value.weight": "ple.value_proj.weight",
        "ffn_down_exps.weight": "mlp.experts.down_proj",
        "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
    }
    if suffix not in conversions:
        raise ValueError(f"Unrecognized edited GGUF tensor: {gguf_name}")
    return stem + conversions[suffix]


def read_imatrix(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    reader = gguf.GGUFReader(path)
    field = reader.get_field("imatrix.chunk_count")
    if field is None or field.contents() < 128:
        raise ValueError("Calibration importance matrix must contain at least 128 chunks")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    result = {}
    for layer in range(48):
        name = f"blk.{layer}.ffn_down_exps.weight"
        values = tensors.get(name + ".in_sum2")
        counts = tensors.get(name + ".counts")
        if values is None or counts is None:
            raise ValueError(f"Missing calibration statistics for {name}")
        if values.data.shape != (512, 640) or counts.data.shape != (512, 1):
            raise ValueError(f"Unexpected calibration shape for {name}")
        result[name] = (values.data, counts.data.reshape(-1))
    return result, int(field.contents())


def checked_source(source: Path, required: set[str]) -> dict[str, str]:
    names, files = selected_shards(source)
    weight_map = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    translated = {name: source_name(name) for name in edited_names()}
    if set(translated.values()) != set(names):
        raise ValueError("BF16 ablation manifest and GGUF edited tensor names disagree")
    if required - translated.keys():
        raise ValueError("Requested GGUF tensor is outside the edited BF16 set")
    chosen = {name: weight_map[translated[name]] for name in required}
    missing = sorted({filename for filename in chosen.values() if not (source / filename).is_file()})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} BF16 shards for selected layers, first: {missing[0]}")
    return chosen


def open_state(path: Path, imatrix: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS chunks (
        name TEXT NOT NULL, qtype TEXT NOT NULL, part INTEGER NOT NULL,
        offset INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
        metrics TEXT NOT NULL, PRIMARY KEY(name, qtype, part))""")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    version = {
        "source_revision": REPO_REVISION,
        "calibration_revision": CALIBRATION_REVISION,
        "imatrix_sha256": hashlib.sha256(imatrix.read_bytes()).hexdigest(),
        "method": "Gumbel assignment/scale adapter with deterministic per-part seeds, Q8_0 GGUF reference quantizer",
        "ssm_layout": "llama.cpp grouped-to-tiled V heads v1",
    }
    for key, value in version.items():
        old = connection.execute("SELECT value FROM metadata WHERE name=?", (key,)).fetchone()
        if key == 'ssm_layout' and old is None and connection.execute('SELECT 1 FROM chunks LIMIT 1').fetchone():
            raise ValueError('Existing state predates the SSM layout fix; repair it or choose a new output directory')
        if old is not None and old[0] != value:
            raise ValueError(f"Existing variant state uses another {key}; choose a new output directory")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, value))
    connection.execute(
        "INSERT OR IGNORE INTO metadata VALUES (?,?)",
        ("build_started_utc", datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return connection


def quantize_dense(chunk: torch.Tensor) -> tuple[dict[str, object], np.ndarray]:
    if not torch.isfinite(chunk).all():
        raise ValueError("Non-finite BF16 edited tensor")
    matrix = chunk.float().numpy()
    packed = gguf.quants.quantize(matrix, gguf.GGMLQuantizationType.Q8_0)
    decoded = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
    sse = float(np.square(decoded - matrix, dtype=np.float64).sum())
    power = float(np.square(matrix, dtype=np.float64).sum())
    if power <= 0 or not np.isfinite(sse):
        raise ValueError("Invalid dense-tensor reconstruction metrics")
    return {"selected_sse": sse, "source_power": power,
            "weighted_sse": sse, "weighted_power": power,
            "selected_relative_mse": sse / power}, packed.ravel()


def convert_dense_layout(chunk: torch.Tensor, name: str, config: dict) -> torch.Tensor:
    """Match llama.cpp's grouped-to-tiled linear-attention V-head columns."""
    if not name.endswith('.ssm_out.weight'):
        return chunk
    hp = config.get('text_config', config)
    keys = int(hp['linear_num_key_heads'])
    values = int(hp['linear_num_value_heads'])
    width = int(hp['linear_value_head_dim'])
    if values % keys or chunk.shape[1] != values * width:
        raise ValueError('Invalid linear-attention output projection layout')
    return chunk.reshape(chunk.shape[0], keys, values // keys, width).transpose(1, 2).contiguous().reshape(chunk.shape)


def run_tensor(connection: sqlite3.Connection, source: Path, filename: str,
               gguf_name: str, qtype: str, output: Path, rows_per_part: int,
               steps: int, importance: tuple[np.ndarray, np.ndarray] | None,
               deadline: datetime) -> None:
    hf_name = source_name(gguf_name)
    with safe_open(source / filename, framework="pt", device="cpu") as handle:
        sliced = handle.get_slice(hf_name)
        shape = sliced.get_shape()
        if qtype in ("Q2_0", "IQ4_NL"):
            if shape != [512, 2560, 640] or importance is None:
                raise ValueError(f"Unexpected fused expert shape or missing calibration: {gguf_name} {shape}")
            units, rows_per_unit, columns = shape
        else:
            if len(shape) != 2:
                raise ValueError(f"Expected 2-D edited tensor: {gguf_name} {shape}")
            units, columns = shape
            rows_per_unit = 1
        block, block_bytes = QTYPE_LAYOUT[qtype]
        if columns % block:
            raise ValueError(f"{gguf_name} width {columns} is incompatible with {qtype}")
        total_rows = units * rows_per_unit
        row_bytes = columns // block * block_bytes
        expected_bytes = total_rows * row_bytes
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("r+b" if output.exists() else "w+b") as destination:
            destination.truncate(expected_bytes)
            part = 0
            for first_row in range(0, total_rows, rows_per_part):
                if datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError("42-hour build allocation expired; state is resumable")
                last_row = min(total_rows, first_row + rows_per_part)
                if rows_per_unit > 1 and (first_row % rows_per_unit or last_row % rows_per_unit):
                    raise ValueError("Fused expert chunks must contain complete experts")
                offset = first_row * row_bytes
                size = (last_row - first_row) * row_bytes
                saved = connection.execute(
                    "SELECT size, sha256, metrics FROM chunks WHERE name=? AND qtype=? AND part=?",
                    (gguf_name, qtype, part),
                ).fetchone()
                if saved is not None and saved[0] == size:
                    destination.seek(offset)
                    if hashlib.sha256(destination.read(size)).hexdigest() == saved[1]:
                        part += 1
                        continue
                if rows_per_unit > 1:
                    begin_expert, end_expert = first_row // rows_per_unit, last_row // rows_per_unit
                    chunk = sliced[begin_expert:end_expert].reshape(-1, columns).float()
                    if not torch.isfinite(chunk).all():
                        raise ValueError(f"Non-finite BF16 edited tensor: {gguf_name}")
                    values, counts = importance
                    weights = values[begin_expert:end_expert].copy()
                    relevant = counts[begin_expert:end_expert]
                    for index, count in enumerate(relevant):
                        if count > 0:
                            weights[index] /= count
                        else:
                            weights[index].fill(1.0)
                    weights = np.nan_to_num(weights, nan=1.0, posinf=100.0, neginf=1.0)
                    weights = np.clip(weights, 1e-5, 100.0)
                    weights = np.repeat(weights, rows_per_unit, axis=0)
                    seed = int.from_bytes(hashlib.sha256(
                        f"{gguf_name}|{qtype}|{part}".encode("utf-8")
                    ).digest()[:8], "little")
                    torch.manual_seed(seed)
                    torch.cuda.reset_peak_memory_stats()
                    metrics, payload = train_one(chunk, qtype, steps, "cuda", torch.from_numpy(weights))
                    metrics["seed"] = seed
                    metrics["experts_with_no_calibration"] = int(np.count_nonzero(relevant == 0))
                    metrics["experts_with_under_8_samples"] = int(np.count_nonzero(relevant < 8))
                else:
                    chunk = sliced[first_row:last_row]
                    chunk = convert_dense_layout(chunk, gguf_name,
                        json.loads((source / 'config.json').read_text(encoding='utf-8')))
                    metrics, payload = quantize_dense(chunk)
                if payload.nbytes != size:
                    raise ValueError(f"Packed size mismatch for {gguf_name}: {payload.nbytes} vs {size}")
                digest = hashlib.sha256(payload).hexdigest()
                destination.seek(offset)
                destination.write(payload.tobytes())
                destination.flush()
                connection.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (gguf_name, qtype, part, offset, size, digest, json.dumps(metrics, sort_keys=True)),
                )
                connection.commit()
                if part % 32 == 0 or last_row == total_rows:
                    print(f"{gguf_name} {qtype}: {last_row}/{total_rows} rows", flush=True)
                part += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source"))
    parser.add_argument("--imatrix", type=Path, default=Path("data/calibration/selected/imatrix-hybrid-128.gguf"))
    parser.add_argument("--output", type=Path, default=Path("data/variants"))
    parser.add_argument("--layers", default="0-47", help="Range, for example 0-47 or 0,23,47")
    parser.add_argument("--down", action="store_true", help="Produce fused expert-down variants")
    parser.add_argument("--dense", action="store_true", help="Produce Q8_0 variants for other edited tensors")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--experts-per-part", type=int, default=2)
    parser.add_argument("--dense-rows-per-part", type=int, default=1024)
    parser.add_argument("--max-build-hours", type=float, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.experts_per_part < 1 or 512 % args.experts_per_part:
        parser.error("steps must be positive and experts-per-part must divide 512")
    if args.max_build_hours <= 0:
        parser.error("max-build-hours must be positive")
    if not args.down and not args.dense:
        args.down = args.dense = True
    if "-" in args.layers:
        low, high = [int(item) for item in args.layers.split("-", 1)]
        layers = list(range(low, high + 1))
    else:
        layers = [int(item) for item in args.layers.split(",")]
    if not layers or any(layer < 0 or layer >= 48 for layer in layers):
        parser.error("layers must be within 0..47")

    torch.manual_seed(1729)
    if args.down and not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the bounded Gumbel quantizer")
    selected = {
        name for name in edited_names()
        if (not name.startswith("blk.") or int(name.split(".")[1]) in layers)
        and (args.down if name.endswith("ffn_down_exps.weight") else args.dense)
    }
    source_files = checked_source(args.source, selected)
    imatrix, chunk_count = read_imatrix(args.imatrix)
    args.output.mkdir(parents=True, exist_ok=True)
    connection = open_state(args.output / "state.sqlite", args.imatrix)
    started = datetime.fromisoformat(connection.execute(
        "SELECT value FROM metadata WHERE name='build_started_utc'"
    ).fetchone()[0])
    deadline = started + timedelta(hours=args.max_build_hours)
    print(f"Using {chunk_count} calibration chunks and BF16 revision {REPO_REVISION}", flush=True)
    print(f"Build deadline: {deadline.isoformat()}", flush=True)
    for name in sorted(selected):
        is_down = name.endswith("ffn_down_exps.weight")
        formats = ("Q2_0", "IQ4_NL") if is_down else ("Q8_0",)
        for qtype in formats:
            safe_name = name.replace(".", "_")
            output = args.output / f"{safe_name}.{qtype}.bin"
            rows = args.experts_per_part * 2560 if is_down else args.dense_rows_per_part
            run_tensor(connection, args.source, source_files[name], name, qtype,
                       output, rows, args.steps, imatrix[name] if is_down else None,
                       deadline)
    connection.close()


if __name__ == "__main__":
    main()
