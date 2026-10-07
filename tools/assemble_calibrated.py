"""Assemble a local GSQ-RCO/BF16-ablation mixed GGUF under an exact byte budget.

This retains unchanged packed tensors from a published ISTA variant and uses
freshly quantized BF16 edited tensors.  The targeted allocation minimizes a
calibration-weighted reconstruction proxy; it is not the upstream full-model
RCO task-loss search.  Output is local only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

import gguf
import numpy as np

from tools.assemble_hybrid import copy_metadata, edited_names, index_tensors


BASE_PATHS = {
    "IQ3_S": Path("data/base/IQ3_S"),
    "IQ3_XXS": Path("models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
    "Q2_0": Path("data/base/Q2_0"),
}


def base_shards(base_name: str) -> list[Path]:
    folder = BASE_PATHS[base_name]
    return [folder / f"Qwen3.8-Flash-Next-GSQ-RCO-{base_name}-{index:05d}-of-00002.gguf"
            for index in (1, 2)]


def variant_path(folder: Path, name: str, qtype: str) -> Path:
    return folder / f"{name.replace('.', '_')}.{qtype}.bin"


def variant_summary(connection: sqlite3.Connection, folder: Path, name: str,
                    qtype: str, expected_bytes: int) -> dict[str, float]:
    path = variant_path(folder, name, qtype)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"Missing or wrong-size variant: {path}")
    rows = connection.execute(
        "SELECT offset, size, sha256, metrics FROM chunks WHERE name=? AND qtype=? ORDER BY offset",
        (name, qtype),
    ).fetchall()
    if not rows:
        raise ValueError(f"No verified chunks for {name} {qtype}")
    position = 0
    summary = {"weighted_sse": 0.0, "source_power": 0.0,
               "under_8_samples": 0.0, "uncovered_experts": 0.0}
    with path.open("rb") as payload:
        for offset, size, digest, encoded in rows:
            if offset != position:
                raise ValueError(f"Incomplete chunk sequence for {name} {qtype}")
            if hashlib.sha256(payload.read(size)).hexdigest() != digest:
                raise ValueError(f"Variant chunk checksum mismatch: {name} {qtype} offset {offset}")
            position += size
            metrics = json.loads(encoded)
            summary["weighted_sse"] += metrics["weighted_sse"]
            summary["source_power"] += metrics["source_power"]
            summary["under_8_samples"] += metrics.get("experts_with_under_8_samples", 0)
            summary["uncovered_experts"] += metrics.get("experts_with_no_calibration", 0)
    if position != expected_bytes:
        raise ValueError(f"Incomplete variant {name} {qtype}")
    return summary


def allocate(base: list[gguf.GGUFReader], base_name: str, folder: Path, budget_gb: float,
             connection: sqlite3.Connection) -> dict[str, object]:
    base_tensors = index_tensors(base)
    selected = edited_names()
    if selected - base_tensors.keys():
        raise ValueError("Published base is missing edited tensor names")
    first = {tensor.name: tensor for tensor in base[0].tensors}
    unchanged_bytes = sum(tensor.n_bytes for name, tensor in first.items() if name not in selected)
    fixed_bytes = unchanged_bytes
    variant_sizes = {}
    reports = {}
    down_names = []
    for name in sorted(selected):
        tensor = base_tensors[name]
        dimensions = tuple(int(value) for value in reversed(tensor.shape))
        columns = dimensions[-1]
        elements = math.prod(dimensions)
        if name.endswith("ffn_down_exps.weight"):
            if dimensions != (512, 2560, 640):
                raise ValueError(f"Unexpected fused expert shape for {name}: {dimensions}")
            down_names.append(name)
            formats = ("Q2_0", "IQ4_NL")
        else:
            formats = ("Q8_0",)
        for qtype in formats:
            block, block_bytes = {"Q2_0": (64, 18), "IQ4_NL": (32, 18), "Q8_0": (32, 34)}[qtype]
            if columns % block:
                raise ValueError(f"{name} width {columns} incompatible with {qtype}")
            expected = elements // block * block_bytes
            variant_sizes[(name, qtype)] = expected
            reports[(name, qtype)] = variant_summary(connection, folder, name, qtype, expected)
        fixed_bytes += variant_sizes[(name, "Q2_0" if name in down_names else "Q8_0")]

    # GGUF headers, metadata and tensor alignment are excluded from packed sizes.
    # Reserve 16 MB, exceeding the current first shard's measured overhead.
    packed_budget = int(budget_gb * 1_000_000_000) - 16_000_000
    if fixed_bytes > packed_budget:
        raise ValueError(f"Minimum candidate exceeds budget by {fixed_bytes - packed_budget} bytes")
    increments = {name: variant_sizes[(name, "IQ4_NL")] - variant_sizes[(name, "Q2_0")]
                  for name in down_names}
    if len(set(increments.values())) != 1:
        raise ValueError("Allocation assumes identical fused expert tensor dimensions")
    extra = next(iter(increments.values()))
    high_count = min(len(down_names), (packed_budget - fixed_bytes) // extra)
    ranked = sorted(down_names, key=lambda name: (
        reports[(name, "Q2_0")]["under_8_samples"] > 0,
        reports[(name, "Q2_0")]["weighted_sse"] - reports[(name, "IQ4_NL")]["weighted_sse"],
    ), reverse=True)
    high = set(ranked[:high_count])
    low_coverage = [name for name in down_names
                    if reports[(name, "Q2_0")]["under_8_samples"] > 0]
    if any(name not in high for name in low_coverage):
        raise ValueError("Byte budget cannot keep every poorly covered fused expert tensor in IQ4_NL")
    types = {name: ("IQ4_NL" if name in high else "Q2_0") for name in down_names}
    types.update({name: "Q8_0" for name in selected if name not in down_names})
    predicted = fixed_bytes + high_count * extra
    return {
        "base": base_name,
        "budget_gb": budget_gb,
        "first_shard_packed_bytes": predicted,
        "first_shard_budget_bytes": int(budget_gb * 1_000_000_000),
        "unchanged_first_shard_bytes": unchanged_bytes,
        "edited_text_tensors": len(selected),
        "down_q2_layers": len(down_names) - high_count,
        "down_iq4_layers": high_count,
        "low_coverage_layers": low_coverage,
        "tensor_types": dict(sorted(types.items())),
        "objective": "128-chunk proxy-activation-weighted reconstruction error under packed byte budget",
        "source_revision": "deb02632504bb214702bc28b0381a93d3112f500",
        "ablation_tensor_source": "BF16 safetensors; 146 edited text tensors",
        "retained_source": "published GSQ-RCO variant; tensors byte-identical",
    }


def assemble(base: list[gguf.GGUFReader], folder: Path, plan: dict[str, object],
             output: Path, variant: str) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, "qwen4exp", split_max_tensors=len(base[0].tensors))
    description = (
        f"Local abliterated Flash Next; unchanged {variant} GSQ-RCO tensors retained; "
        "146 BF16-edited text tensors freshly quantized to calibrated Q2_0, IQ4_NL or Q8_0. "
        "Targeted reconstruction-budget allocation, not a full upstream GSQ-RCO rerun."
    )
    copy_metadata(writer, base[0], variant + " Calibrated Mixed", description)
    writer.data_alignment = int(base[0].alignment)
    for reader in base:
        for tensor in reader.tensors:
            qtype = plan["tensor_types"].get(tensor.name)
            if qtype is None:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
                continue
            shape = tuple(int(value) for value in reversed(tensor.shape))
            block, block_bytes = {"Q2_0": (64, 18), "IQ4_NL": (32, 18), "Q8_0": (32, 34)}[qtype]
            payload = np.memmap(variant_path(folder, tensor.name, qtype), dtype=np.uint8,
                                mode="r", shape=(*shape[:-1], shape[-1] // block * block_bytes))
            writer.add_tensor(tensor.name, payload,
                              raw_dtype=gguf.GGMLQuantizationType[qtype])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    paths = writer.format_shard_names(output)
    if paths[0].stat().st_size > int(plan["first_shard_budget_bytes"]):
        raise ValueError("Completed first shard exceeds its declared byte budget")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=tuple(BASE_PATHS), default="IQ3_S")
    parser.add_argument("--budget-gb", type=float, required=True)
    parser.add_argument("--variants", type=Path, default=Path("data/variants"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.budget_gb <= 0:
        parser.error("budget must be positive")
    paths = base_shards(args.base)
    base = [gguf.GGUFReader(path) for path in paths]
    connection = sqlite3.connect(args.variants / "state.sqlite")
    plan = allocate(base, args.base, args.variants, args.budget_gb, connection)
    connection.close()
    if args.write:
        paths = assemble(base, args.variants, plan, args.output, args.base)
        plan["output_shards"] = [str(path) for path in paths]
        plan["output_bytes"] = [path.stat().st_size for path in paths]
        report_path = args.output.with_suffix(".allocation.json")
        report_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {report_path}")
    else:
        summary = {key: value for key, value in plan.items() if key != "tensor_types"}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
