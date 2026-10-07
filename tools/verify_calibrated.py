"""Check a calibrated local GGUF against its retained GSQ tensors and variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

import gguf

from tools.assemble_calibrated import base_shards, variant_path
from tools.assemble_hybrid import edited_names, index_tensors, metadata
from tools.verify_release import file_sha256


def equal_ranges(first: Path, first_offset: int, second: Path, second_offset: int,
                 length: int, chunk_size: int = 1024 * 1024) -> bool:
    if first.stat().st_size < first_offset + length or second.stat().st_size < second_offset + length:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        left.seek(first_offset)
        right.seek(second_offset)
        for offset in range(0, length, chunk_size):
            take = min(chunk_size, length - offset)
            if left.read(take) != right.read(take):
                return False
    return True


def verify(base: list[gguf.GGUFReader], output: list[gguf.GGUFReader],
           base_paths: list[Path], output_paths: list[Path],
           plan: dict[str, object], variants: Path) -> dict[str, object]:
    if len(base) != 2 or len(output) != 2:
        raise ValueError("Expected two source and two output shards")
    originals = index_tensors(base)
    actual = index_tensors(output)
    base_location = {tensor.name: path for reader, path in zip(base, base_paths)
                     for tensor in reader.tensors}
    output_location = {tensor.name: path for reader, path in zip(output, output_paths)
                       for tensor in reader.tensors}
    if set(originals) != set(actual):
        raise ValueError("Output tensor names differ from the selected GSQ base")
    if set(plan["tensor_types"]) != edited_names():
        raise ValueError("Allocation does not cover exactly the 146 edited tensors")
    if metadata(output[0], "general.architecture") != "qwen4exp":
        raise ValueError("Wrong output architecture")
    if metadata(output[0], "general.license") != "Qwen Community License 1.0":
        raise ValueError("Wrong output license")
    if [t.name for t in output[1].tensors] != [t.name for t in base[1].tensors]:
        raise ValueError("N-gram shard tensor layout changed")
    for index, (name, result) in enumerate(actual.items(), start=1):
        source = originals[name]
        if tuple(result.shape) != tuple(source.shape):
            raise ValueError(f"Tensor shape mismatch: {name}")
        qtype = plan["tensor_types"].get(name)
        if qtype is None:
            if result.tensor_type != source.tensor_type or not equal_ranges(
                base_location[name], source.data_offset,
                output_location[name], result.data_offset, result.n_bytes
            ):
                raise ValueError(f"Retained GSQ tensor differs: {name}")
        else:
            if result.tensor_type != gguf.GGMLQuantizationType[qtype]:
                raise ValueError(f"Edited tensor format mismatch: {name}")
            variant = variant_path(variants, name, qtype)
            if variant.stat().st_size != result.n_bytes or not equal_ranges(
                variant, 0, output_location[name], result.data_offset, result.n_bytes
            ):
                raise ValueError(f"Edited tensor payload mismatch: {name}")
        if index % 100 == 0:
            print(f"Verified {index}/{len(actual)} tensors", file=sys.stderr, flush=True)
    if int(plan["first_shard_budget_bytes"]) < output[0].data_offset + sum(t.n_bytes for t in output[0].tensors):
        raise ValueError("First shard exceeds byte budget")
    return {"verified_tensors": len(actual), "edited": len(edited_names()),
            "retained": len(actual) - len(edited_names()),
            "packed_bytes": sum(t.n_bytes for t in actual.values()),
            "variant_formats": {q: list(plan["tensor_types"].values()).count(q)
                                for q in sorted(set(plan["tensor_types"].values()))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--variants", type=Path, default=Path("data/variants"))
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.allocation.read_text(encoding="utf-8"))
    base_paths = base_shards(plan["base"])
    base = [gguf.GGUFReader(path) for path in base_paths]
    paths = [Path(path) for path in plan["output_shards"]]
    output = [gguf.GGUFReader(path) for path in paths]
    report = verify(base, output, base_paths, paths, plan, args.variants)
    if args.hash:
        report["files"] = {str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                           for path in paths}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
