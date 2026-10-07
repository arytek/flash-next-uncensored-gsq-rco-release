"""Verify that every hybrid tensor matches its declared packed source exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gguf

from tools.assemble_hybrid import audit, edited_names, index_tensors, metadata


def equal_bytes(left, right, chunk_size: int = 8 * 1024 * 1024) -> bool:
    a = memoryview(left.data).cast("B")
    b = memoryview(right.data).cast("B")
    if len(a) != len(b):
        return False
    return all(a[offset:offset + chunk_size] == b[offset:offset + chunk_size]
               for offset in range(0, len(a), chunk_size))


def verify(base, donor, output) -> dict:
    input_report = audit(base, donor)
    if len(output) != 2:
        raise ValueError("Expected exactly two output shards")
    originals = index_tensors(base)
    replacements = index_tensors([donor])
    actual = index_tensors(output)
    if set(actual) != set(originals):
        raise ValueError(f"Tensor-name mismatch: missing={sorted(set(originals)-set(actual))}, "
                         f"extra={sorted(set(actual)-set(originals))}")
    if metadata(output[0], "general.architecture") != "qwen4exp":
        raise ValueError("Wrong output architecture")
    if metadata(output[0], "general.license") != "Qwen Community License 1.0":
        raise ValueError("Wrong output license metadata")
    if [t.name for t in output[1].tensors] != [t.name for t in base[1].tensors]:
        raise ValueError("N-gram shard contents changed")
    changed = edited_names()
    for index, (name, result) in enumerate(actual.items(), start=1):
        source = replacements[name] if name in changed else originals[name]
        if result.tensor_type != source.tensor_type or tuple(result.shape) != tuple(source.shape):
            raise ValueError(f"Type or shape mismatch: {name}")
        if not equal_bytes(source, result):
            raise ValueError(f"Packed data mismatch: {name}")
        if index % 100 == 0:
            print(f"Verified {index}/{len(actual)} tensors", file=sys.stderr, flush=True)
    return {
        "verified_tensors": len(actual),
        "verified_transplants": input_report["transplanted_tensors"],
        "verified_retained": input_report["retained_base_tensors"],
        "output_packed_bytes": sum(t.n_bytes for t in actual.values()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    seen = 0
    next_report = 8 * 1024**3
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            seen += len(block)
            if seen >= next_report:
                print(f"Hashed {seen / 1024**3:.0f} GiB of {path.name}", file=sys.stderr, flush=True)
                next_report += 8 * 1024**3
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs=2, type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", nargs=2, type=Path, required=True)
    parser.add_argument("--hash", action="store_true", help="Also compute output-file SHA-256 checksums")
    args = parser.parse_args()
    base = [gguf.GGUFReader(path) for path in args.base]
    donor = gguf.GGUFReader(args.donor)
    output = [gguf.GGUFReader(path) for path in args.output]
    report = verify(base, donor, output)
    if args.hash:
        report["files"] = {str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                           for path in args.output}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
