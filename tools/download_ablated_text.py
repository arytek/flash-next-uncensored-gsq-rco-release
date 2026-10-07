"""Fetch only the BF16 source shards containing the 146 edited text tensors.

The source is kept local.  This script never uploads weights or metadata.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path


REPO = "windowsxp811203/Qwen3.8-Flash-Next-Abliterated"
REVISION = "deb02632504bb214702bc28b0381a93d3112f500"


def selected_shards(source: Path) -> tuple[list[str], list[str]]:
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    ablation = json.loads((source / "ABLIT_META.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    names = sorted(
        item["tensor"] for item in ablation["stats"]
        if item["tensor"].startswith("model.language_model.")
    )
    if len(names) != 146 or len(set(names)) != 146:
        raise ValueError(f"Expected 146 distinct edited text tensors, found {len(names)}")
    missing = set(names) - weight_map.keys()
    if missing:
        raise ValueError(f"Source index lacks edited tensors: {sorted(missing)}")
    files = sorted({weight_map[name] for name in names})
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in files):
        raise ValueError("Unexpected source shard filename")
    return names, files


def inspect_shard(path: Path, wanted: set[str]) -> dict[str, object]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError(f"Unreasonable safetensors header size: {path}")
        header = json.loads(handle.read(header_size))
    size = path.stat().st_size
    found = {}
    for name in sorted(wanted):
        entry = header.get(name)
        if entry is None or entry["dtype"] != "BF16":
            raise ValueError(f"Missing BF16 source tensor {name} in {path}")
        start, end = entry["data_offsets"]
        elements = 1
        for dimension in entry["shape"]:
            elements *= dimension
        if start < 0 or end - start != elements * 2 or 8 + header_size + end > size:
            raise ValueError(f"Invalid tensor bounds for {name} in {path}")
        found[name] = {"shape": entry["shape"], "bytes": end - start}
    return {"bytes": size, "header_bytes": header_size, "edited_tensors": found}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N shards, for a pilot")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    names, files = selected_shards(args.source)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit:
        files = files[:args.limit]
    print(f"Edited text tensors: {len(names)}; source shards selected: {len(files)}", flush=True)
    if args.download:
        command = [
            "hf", "download", REPO, *files,
            "--revision", REVISION,
            "--local-dir", str(args.source),
            "--max-workers", str(args.max_workers),
        ]
        subprocess.run(command, check=True)

    index = json.loads((args.source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    mapping = index["weight_map"]
    manifest: dict[str, object] = {
        "repository": REPO,
        "revision": REVISION,
        "edited_text_tensors": len(names),
        "required_source_shards": len(files),
        "files": {},
    }
    for filename in files:
        path = args.source / filename
        if not path.is_file():
            print(f"Missing: {path}", flush=True)
            continue
        wanted = {name for name in names if mapping[name] == filename}
        manifest["files"][filename] = inspect_shard(path, wanted)
    complete = len(manifest["files"]) == len(files)
    manifest["complete"] = complete
    output = args.source / "text-source-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Verified {len(manifest['files'])}/{len(files)} shards; manifest: {output}", flush=True)
    if args.download and not complete:
        raise RuntimeError("Some source shards are missing after download")


if __name__ == "__main__":
    main()
