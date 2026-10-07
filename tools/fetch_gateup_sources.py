"""Fetch and verify the three pinned BF16 gate/up pilot shards on M: only.

Default invocation writes a bounded download plan without network activity.
Use --download explicitly. Transfers resume from a private partial file, require
exact HTTP206 ranges, and rename only after full SHA/header/finite checks pass.
No weights are uploaded and no model inference or generated code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/optimization-final-gateup"
SOURCE = FOLDER / "sources"
REVISION = "deb02632504bb214702bc28b0381a93d3112f500"
REPOSITORY = "windowsxp811203/Qwen3.8-Flash-Next-Abliterated"
CEILING = 12 * 1024**3
SHAPE = [512, 1280, 2560]
EXPERT_BYTES = 1280 * 2560 * 2
EXPECTED = {
    8: ("model-00127-of-00131.safetensors", 3355443368,
        "fa0c609906b194b1c5172d7346f971b2ebd41f674af8f1469c2fc9232ccc2506"),
    28: ("model-00077-of-00131.safetensors", 3355443376,
         "a75bf91dc472eab2bd4aa907ec5216452c08b3ab915d57377ffd2acc5b1ce42c"),
    47: ("model-00119-of-00131.safetensors", 3355443376,
         "fae555008bac77f16a20300728b38a0616a8228c9b885be2917ae42c7c25ca31"),
}


class PilotDeadlineReached(RuntimeError):
    pass


class TransferRejected(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_deadline():
    budget = read(FOLDER / "budget.json")
    end = datetime.fromisoformat(budget["pilot_deadline_utc"].replace("Z", "+00:00"))
    if budget["status"] not in ("active-pilot", "pilot-in-progress"):
        raise PilotDeadlineReached("Pilot budget is no longer active")
    if datetime.now(timezone.utc) >= end:
        raise PilotDeadlineReached("The final pilot deadline has been reached; partial files retained")
    if budget["pilot_layers"] != [8, 28, 47] or budget["pilot_source_ceiling_bytes"] > CEILING:
        raise TransferRejected("Pilot layer or source ceiling changed")
    return budget


def require_m_path(path):
    resolved_root = ROOT.resolve()
    resolved = Path(path).resolve()
    if resolved_root.drive.upper() != "M:" or resolved.drive.upper() != "M:":
        raise TransferRejected("Pilot source paths must remain on M:")
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise TransferRejected("Pilot source path escaped the project") from None
    return resolved


def write_json(path, record):
    path = require_m_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = require_m_path(path.with_suffix(path.suffix + ".tmp"))
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            check_deadline()
            digest.update(block)
    return digest.hexdigest()


def disk_usage():
    return sum(path.stat().st_size for path in SOURCE.rglob("*") if path.is_file()) if SOURCE.exists() else 0


def check_storage():
    budget = check_deadline()
    cap = min(CEILING, budget["pilot_source_ceiling_bytes"])
    used = disk_usage()
    remaining = 0
    for filename, expected_size, _ in EXPECTED.values():
        target = require_m_path(SOURCE / filename)
        part = require_m_path(target.with_suffix(target.suffix + ".partial"))
        if target.exists() and part.exists():
            raise TransferRejected("A verified-name source and its partial both exist; preserve and inspect them")
        if target.exists():
            if target.stat().st_size != expected_size:
                raise TransferRejected("Existing source length differs from pinned metadata")
        else:
            count = part.stat().st_size if part.exists() else 0
            if count > expected_size:
                raise TransferRejected("Existing source partial exceeds the pinned length")
            remaining += expected_size - count
    if used + remaining + 2 * 1024**2 > cap:
        raise TransferRejected("Pilot source footprint would exceed its12GiB ceiling")
    available = shutil.disk_usage(require_m_path(ROOT)).free
    if available < remaining + 1024**3:
        raise TransferRejected("M: lacks enough space for the remaining pilot sources and1GiB reserve")
    return {"used_bytes": used, "remaining_payload_bytes": remaining,
            "ceiling_bytes": cap, "free_m_bytes": available}


def record_progress(stage, layer=None, error_type=None):
    counts = {}
    for item_layer, (filename, expected_size, _) in EXPECTED.items():
        target = SOURCE / filename
        part = target.with_suffix(target.suffix + ".partial")
        count = target.stat().st_size if target.exists() else part.stat().st_size if part.exists() else 0
        counts[str(item_layer)] = {"bytes": count, "expected_bytes": expected_size}
    received = sum(item["bytes"] for item in counts.values())
    total = sum(item["expected_bytes"] for item in counts.values())
    record = {"utc": utc(), "status": stage, "active_layer": layer, "received_bytes": received,
              "total_bytes": total, "percent_received": round(received * 100 / total, 2), "layers": counts}
    if error_type:
        record["error_type"] = error_type
    write_json(SOURCE / "download-progress.json", record)
    print(json.dumps(record), flush=True)


def validate_inputs():
    check_deadline()
    require_m_path(SOURCE)
    evidence = read(FOLDER / "source-preflight.json")
    if evidence["source_revision"] != REVISION or evidence["source_repository"] != REPOSITORY:
        raise TransferRejected("Pinned gate/up source evidence changed")
    records = {entry["layer"]: entry for entry in evidence["pilot_fetch"]["full_shard_receipts"]}
    if set(records) != set(EXPECTED):
        raise TransferRejected("The approved pilot shard inventory changed")
    mapping = read(ROOT / "data/source/model.safetensors.index.json")["weight_map"]
    edited = {entry["tensor"] for entry in read(ROOT / "data/source/ABLIT_META.json")["stats"]}
    for layer, (filename, size, digest) in EXPECTED.items():
        item = records[layer]
        tensor = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        if (item["file"], item["bytes"], item["sha256"]) != (filename, size, digest):
            raise TransferRejected("Pilot shard metadata differs from the pinned source tree")
        if mapping[tensor] != filename or tensor in edited:
            raise TransferRejected("Gate/up source mapping or ablation status changed")
    return evidence


def verify_source(path, layer, expected_size, expected_sha):
    check_deadline()
    stat = path.stat()
    if stat.st_size != expected_size or sha(path) != expected_sha:
        raise TransferRejected("Pilot source full checksum or length differs from the pinned source")
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 0 < header_length <= 16 * 1024**2:
            raise TransferRejected("Invalid source safetensors header length")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise TransferRejected("Incomplete source safetensors header")
        header = json.loads(raw_header)
        name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        if set(header) - {"__metadata__"} != {name}:
            raise TransferRejected("Pilot shard has an unexpected tensor inventory")
        item = header[name]
        if item["dtype"] != "BF16" or item["shape"] != SHAPE:
            raise TransferRejected("Pilot source dtype or fused gate/up shape differs")
        begin, end = item["data_offsets"]
        if begin != 0 or end - begin != math.prod(SHAPE) * 2 or 8 + header_length + end != expected_size:
            raise TransferRejected("Pilot source tensor bounds differ")
        tensor_start = 8 + header_length + begin
        stream.seek(tensor_start)
        experts = []
        for expert in range(512):
            check_deadline()
            raw = stream.read(EXPERT_BYTES)
            if len(raw) != EXPERT_BYTES:
                raise TransferRejected("BF16 expert read ended before the expected boundary")
            packed = np.frombuffer(raw, dtype="<u2").reshape(1280, 2560)
            if packed.shape != (1280, 2560):
                raise TransferRejected("BF16 expert index layout differs")
            record = {"expert": expert, "source_byte_offset": tensor_start + expert * EXPERT_BYTES,
                      "bytes": EXPERT_BYTES, "sha256": hashlib.sha256(raw).hexdigest()}
            for half, portion in (("gate", packed[:640]), ("up", packed[640:])):
                matrix = (portion.astype(np.uint32) << 16).view(np.float32)
                if matrix.shape != (640, 2560) or not matrix.flags.c_contiguous or not np.isfinite(matrix).all():
                    raise TransferRejected("BF16 gate/up split contains nonfinite values or an unexpected layout")
                record[half + "_bf16_sha256"] = hashlib.sha256(portion.tobytes()).hexdigest()
                record[half + "_float32_abs_max"] = float(np.abs(matrix).max())
            experts.append(record)
        if stream.read(1):
            raise TransferRejected("Unexpected bytes after the fused gate/up tensor")
    if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise TransferRejected("Source changed while it was being verified")
    return {"layer": layer, "hf_tensor": name, "filename": path.name.removesuffix(".partial"),
        "bytes": expected_size, "sha256": expected_sha, "mtime_ns": stat.st_mtime_ns,
        "header_sha256": hashlib.sha256(raw_length + raw_header).hexdigest(),
        "header_bytes": header_length, "tensor_byte_offset": tensor_start, "tensor_bytes": end - begin,
        "dtype": "BF16", "shape": SHAPE, "finite_experts_verified": 512,
        "split": "axis1at640: first640gate, final640up; contiguous640x2560 per expert; no head permutation",
        "experts": experts, "verified_utc": utc()}


def transfer(layer, filename, expected_size):
    target = require_m_path(SOURCE / filename)
    part = require_m_path(target.with_suffix(target.suffix + ".partial"))
    if target.exists():
        return target
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}"
    for attempt in range(6):
        check_storage()
        count = part.stat().st_size if part.exists() else 0
        if count == expected_size:
            return part
        request = urllib.request.Request(url, headers={"Range": f"bytes={count}-{expected_size - 1}",
                                                        "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {count}-{expected_size - 1}/{expected_size}":
                    raise TransferRejected("The trusted source did not honour the exact resume range")
                if urlsplit(response.geturl()).scheme != "https":
                    raise TransferRejected("The source redirect did not retain HTTPS")
                with part.open("ab") as stream:
                    last_progress = time.monotonic()
                    while count < expected_size:
                        check_deadline()
                        block = response.read(min(8 * 1024**2, expected_size - count))
                        if not block:
                            raise OSError("Incomplete source stream")
                        stream.write(block)
                        count += len(block)
                        if time.monotonic() - last_progress >= 15:
                            stream.flush()
                            record_progress("downloading", layer)
                            last_progress = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            return part
        except (OSError, TimeoutError) as error:
            # Never print the exception text: CDN redirects contain signed URLs.
            record_progress("transfer-interrupted", layer, type(error).__name__)
            if attempt == 5:
                raise TransferRejected("Source transfer retries exhausted; partial bytes retained") from None
            check_deadline()
            time.sleep(2)
    raise AssertionError("unreachable")


def acquire_lock():
    # OS file locking automatically releases if the worker exits or is stopped.
    import msvcrt
    SOURCE.mkdir(parents=True, exist_ok=True)
    path = require_m_path(SOURCE / "download.lock")
    stream = path.open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        stream.close()
        raise TransferRejected("A gate/up source downloader already owns the OS file lock") from None
    return stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    evidence = validate_inputs()
    storage = check_storage()
    plan = {"utc": utc(), "mode": "download" if args.download else "verify-only" if args.verify_only else "plan-only",
        "repository": REPOSITORY, "revision": REVISION,
        "pilot_deadline_utc": read(FOLDER / "budget.json")["pilot_deadline_utc"],
        "source_directory": str(require_m_path(SOURCE)), "total_pinned_bytes": sum(item[1] for item in EXPECTED.values()),
        "storage": storage, "layers": evidence["pilot_fetch"]["full_shard_receipts"],
        "no_c_writes": True, "no_model_loads": True, "privacy": "Local private source download only"}
    write_json(SOURCE / "download-plan.json", plan)
    if not args.download and not args.verify_only:
        print(json.dumps({key: plan[key] for key in ("mode", "source_directory", "total_pinned_bytes", "storage")}), flush=True)
        return
    lock = acquire_lock()
    try:
        manifest_path = SOURCE / "manifest.json"
        manifest = read(manifest_path) if manifest_path.exists() else {
            "repository": REPOSITORY, "revision": REVISION, "pilot_layers": [8, 28, 47], "files": {}, "complete": False,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_index_sha256": evidence["source_index_sha256"], "ablation_meta_sha256": evidence["ablation_meta_sha256"],
            "method": "Pinned complete BF16 source shards: full SHA256, exact header and all512expert split/finite checks before final rename",
            "scope": "Unchanged gate/up from the pinned ablated checkpoint; no complete BF16 whole-model teacher",
            "source_ceiling_bytes": CEILING, "privacy": "Local only"}
        if manifest["repository"] != REPOSITORY or manifest["revision"] != REVISION or manifest["pilot_layers"] != [8,28,47]:
            raise TransferRejected("Existing verified source manifest differs from this pilot")
        if manifest["builder_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise TransferRejected("Downloader changed after sources were recorded")
        for layer, (filename, size, expected_sha) in EXPECTED.items():
            check_deadline()
            target = require_m_path(SOURCE / filename)
            prior = manifest["files"].get(filename)
            if prior and target.exists() and not args.verify_only:
                if target.stat().st_size == prior["bytes"] == size and target.stat().st_mtime_ns == prior["mtime_ns"] and prior["sha256"] == expected_sha and prior["finite_experts_verified"] == 512:
                    continue
                raise TransferRejected("Previously verified BF16 source changed; retain and inspect it")
            if args.verify_only and not target.exists():
                raise TransferRejected("Verify-only requires all three complete source files")
            path = target if args.verify_only else transfer(layer, filename, size)
            record_progress("checking-sha-header-finite", layer)
            verified = verify_source(path, layer, size, expected_sha)
            if path != target:
                path.rename(target)
            verified.update(path=str(target), mtime_ns=target.stat().st_mtime_ns)
            manifest["files"][filename] = verified
            manifest["updated_utc"] = utc()
            manifest["complete"] = len(manifest["files"]) == len(EXPECTED)
            write_json(manifest_path, manifest)
            record_progress("source-verified", layer)
        manifest["complete"] = True
        manifest["updated_utc"] = utc()
        write_json(manifest_path, manifest)
        record_progress("verified-complete")
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Keep log files free of signed CDN URLs, auth headers and urllib traces.
        print(json.dumps({"status": "source-download-failed", "error_type": type(error).__name__,
                          "partial_files_retained": True, "utc": utc()}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
