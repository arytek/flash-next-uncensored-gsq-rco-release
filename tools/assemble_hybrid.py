"""Assemble a transparently labeled GSQ-RCO / abliterated GGUF hybrid.

This copies packed tensors. It does not claim to run GSQ or RCO on the
abliterated weights. The output must be evaluated before publication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import gguf


def edited_names(layer_count: int = 48) -> set[str]:
    names = {"token_embd.weight", "blk.1.ple_value.weight"}
    for layer in range(layer_count):
        prefix = f"blk.{layer}."
        names.add(prefix + "ffn_down_exps.weight")
        names.add(prefix + "ffn_down_shexp.weight")
        names.add(prefix + ("attn_output.weight" if layer % 4 == 3 else "ssm_out.weight"))
    return names


def metadata(reader: gguf.GGUFReader, name: str):
    field = reader.get_field(name)
    if field is None:
        raise ValueError(f"Missing GGUF metadata: {name}")
    return field.contents()


def index_tensors(readers: list[gguf.GGUFReader]):
    result = {}
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in result:
                raise ValueError(f"Duplicate tensor: {tensor.name}")
            result[tensor.name] = tensor
    return result


def audit(base: list[gguf.GGUFReader], donor: gguf.GGUFReader):
    if len(base) != 2:
        raise ValueError("Expected two published GSQ-RCO shards")
    for reader in [base[0], donor]:
        if metadata(reader, "general.architecture") != "qwen4exp":
            raise ValueError("All inputs must be qwen4exp GGUFs")
        if metadata(reader, "qwen4exp.block_count") != 48:
            raise ValueError("Expected 48 decoder layers")
        expected_ratios = [4 if layer % 4 == 3 else 0 for layer in range(48)]
        if list(metadata(reader, "qwen4exp.attention.compress_ratios")) != expected_ratios:
            raise ValueError("Sparse-attention compression ratios are missing or incorrect")
    base_tensors = index_tensors(base)
    donor_tensors = index_tensors([donor])
    target = edited_names()
    missing_base = sorted(target - base_tensors.keys())
    missing_donor = sorted(target - donor_tensors.keys())
    if missing_base or missing_donor:
        raise ValueError(f"Missing edited tensors: base={missing_base}, donor={missing_donor}")
    mismatched = [
        name for name in sorted(target)
        if tuple(base_tensors[name].shape) != tuple(donor_tensors[name].shape)
    ]
    if mismatched:
        raise ValueError(f"Shape mismatch: {mismatched}")
    if "per_layer_token_embd.weight" not in {t.name for t in base[1].tensors}:
        raise ValueError("Base shard 2 must contain the n-gram table")
    base_bytes = sum(t.n_bytes for t in base_tensors.values())
    replacement_bytes = sum(donor_tensors[name].n_bytes - base_tensors[name].n_bytes for name in target)
    report = {
        "method": "packed-tensor transplant; GSQ-RCO tensors retained only where unedited",
        "base_tensors": len(base_tensors),
        "transplanted_tensors": len(target),
        "retained_base_tensors": len(base_tensors) - len(target),
        "base_packed_bytes": base_bytes,
        "estimated_output_packed_bytes": base_bytes + replacement_bytes,
        "donor_types": dict(sorted(Counter(donor_tensors[n].tensor_type.name for n in target).items())),
        "transplanted_names": sorted(target),
    }
    return report


def copy_metadata(writer: gguf.GGUFWriter, base: gguf.GGUFReader, variant: str,
                  description: str | None = None):
    replacements = {
        "general.name": f"Qwen3.8 Flash Next Abliterated GSQ-RCO {variant} Hybrid",
        "general.description": description or f"GSQ-RCO {variant} base with abliterated residual-writer tensors transplanted from a Q4_K_M GGUF",
        "general.license": "Qwen Community License 1.0",
    }
    for name, field in base.fields.items():
        if name.startswith("GGUF.") or name.startswith("split.") or name == "general.architecture":
            continue
        if name in {"general.url", "general.source.url"}:
            continue
        value = replacements.pop(name, field.contents())
        subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(name, value, field.types[0], subtype)
    for name, value in replacements.items():
        writer.add_key_value(name, value, gguf.GGUFValueType.STRING)


def write_output(base: list[gguf.GGUFReader], donor: gguf.GGUFReader, output: Path,
                 variant: str = "Q2_0"):
    donor_tensors = index_tensors([donor])
    selected = edited_names()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, "qwen4exp", split_max_tensors=len(base[0].tensors))
    copy_metadata(writer, base[0], variant)
    writer.data_alignment = int(base[0].alignment)
    for reader in base:
        for tensor in reader.tensors:
            chosen = donor_tensors[tensor.name] if tensor.name in selected else tensor
            writer.add_tensor(chosen.name, chosen.data, raw_dtype=chosen.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return writer.format_shard_names(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs=2, type=Path, required=True, metavar=("SHARD1", "SHARD2"))
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("Q2_0", "IQ3_S", "IQ3_XXS"), required=True)
    parser.add_argument("--write", action="store_true", help="Write output only after the input audit passes")
    args = parser.parse_args()
    base = [gguf.GGUFReader(path) for path in args.base]
    donor = gguf.GGUFReader(args.donor)
    report = audit(base, donor)
    print(json.dumps({k: v for k, v in report.items() if k != "transplanted_names"}, indent=2))
    if args.write:
        paths = write_output(base, donor, args.output, args.variant)
        print(json.dumps({"written": [str(path) for path in paths]}, indent=2))
    else:
        print("Audit passed. Re-run with --write to assemble the GGUF.")


if __name__ == "__main__":
    main()
