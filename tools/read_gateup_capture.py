"""Strict reader and provenance checks for the local GUPACT01 token capture."""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GateUpCapture:
    path: Path
    layer: int
    seen: int
    seed: int
    batches: int
    cap: int
    records: np.ndarray
    sha256: str

    @property
    def x(self):
        return self.records["x"]

    @property
    def ids(self):
        return self.records["ids"]

    @property
    def weights(self):
        return self.records["weights"]

    @property
    def y(self):
        return self.records["y"]

    def coverage(self):
        counts = np.bincount(self.ids.ravel(), minlength=512)
        mass = np.bincount(self.ids.ravel(), weights=self.weights.ravel(), minlength=512)
        return counts, mass

    def receipt(self):
        counts, mass = self.coverage()
        return {"path": str(self.path.resolve()), "sha256": self.sha256,
                "bytes": self.path.stat().st_size, "layer": self.layer,
                "stored": len(self.records), "seen": self.seen, "seed": self.seed,
                "batches": self.batches, "cap": self.cap,
                "expert_counts": counts.tolist(), "expert_route_mass": mass.tolist(),
                "weight_sum_min": float(self.weights.sum(1).min()),
                "weight_sum_max": float(self.weights.sum(1).max()),
                "scope": "Actual routed MoE sum before unchanged shared expert and HC scatter"}


def record_dtype(width=2560, topk=10):
    return np.dtype([("token", "<u8"), ("batch", "<u8"),
                     ("row", "<u4"), ("reserved", "<u4"),
                     ("x", "<f4", (width,)), ("ids", "<i4", (topk,)),
                     ("weights", "<f4", (topk,)), ("y", "<f4", (width,))], align=False)


def read_capture(path: Path, expected_layer: int | None = None) -> GateUpCapture:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(64)
    if len(header) != 64 or header[:8] != b"GUPACT01":
        raise ValueError(f"Invalid gate/up capture header: {path}")
    layer, width, experts, topk, cap, stored = struct.unpack("<6I", header[8:32])
    seen, seed, batches, reserved = struct.unpack("<4Q", header[32:64])
    if (width, experts, topk, reserved) != (2560, 512, 10, 0):
        raise ValueError("Unexpected gate/up capture dimensions or reserved header")
    if not (0 <= layer < 48 and 0 < cap <= 8192 and 0 < stored <= cap and stored == min(seen, cap) and batches > 0):
        raise ValueError("Invalid gate/up capture counts")
    if expected_layer is not None and layer != expected_layer:
        raise ValueError("Capture belongs to a different layer")
    dtype = record_dtype(width, topk)
    if path.stat().st_size != 64 + stored * dtype.itemsize:
        raise ValueError("Truncated or overlong gate/up capture")
    records = np.fromfile(path, dtype=dtype, count=stored, offset=64)
    if np.any(records["reserved"]) or np.any(records["token"] < 1) or np.any(records["token"] > seen):
        raise ValueError("Invalid token ordinals/reserved record")
    if len(np.unique(records["token"])) != stored:
        raise ValueError("Duplicate token ordinal in reservoir")
    if np.any(records["batch"] < 1) or np.any(records["batch"] > batches) or np.any(records["row"] >= 2048):
        raise ValueError("Invalid microbatch identity")
    pairs = np.stack([records["batch"], records["row"].astype(np.uint64)], axis=1)
    if len(np.unique(pairs, axis=0)) != stored:
        raise ValueError("Duplicate microbatch/token pairing")
    ids, weights = records["ids"], records["weights"]
    if np.any(ids < 0) or np.any(ids >= experts) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("Invalid or duplicate selected expert IDs")
    if any(not np.isfinite(records[field]).all() for field in ("x", "weights", "y")):
        raise ValueError("Nonfinite captured arrays")
    if np.any(weights < 0) or np.any(weights.sum(1) <= 0) or np.any(weights.sum(1) >= 64):
        raise ValueError("Invalid actual final routing weights")
    records.flags.writeable = False
    return GateUpCapture(path, layer, seen, seed, batches, cap, records, sha256(path))


def verify_calibration_manifest(path: Path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    intervals = {}
    for name in ("train", "validation", "confirmation-text"):
        item = manifest["sets"][name]
        for key, digest_key in (("path", "sha256"), ("excerpt_records", "excerpt_records_sha256")):
            if sha256(Path(item[key])) != item[digest_key]:
                raise ValueError(f"Frozen calibration content changed: {name}/{key}")
        starts = item["starts"]
        if len(starts) != item["excerpts"] or len(set(starts)) != len(starts):
            raise ValueError("Invalid frozen calibration starts")
        intervals[name] = [(start, start + item["chars_per_excerpt"]) for start in starts]
    names = list(intervals)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if any(a < d and c < b for a, b in intervals[name] for c, d in intervals[other]):
                raise ValueError("Calibration/validation/confirmation source intervals overlap")
    if manifest["exclusion"]["overlap_between_new_sets"] != 0 or manifest["exclusion"]["source_interval_overlap_with_prior"] != 0:
        raise ValueError("Frozen manifest reports source overlap")
    root = Path(__file__).resolve().parents[1]
    for relative, digest in manifest["exclusion"]["prior_manifest_sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError("Prior calibration exclusion manifest changed")
    return {"manifest_path": str(path.resolve()), "manifest_sha256": sha256(path),
            "train_sha256": manifest["sets"]["train"]["sha256"],
            "validation_sha256": manifest["sets"]["validation"]["sha256"],
            "train_validation_confirmation_overlap": 0,
            "prior_exclusion_manifests_verified": True,
            "note": "Source-character disjointness; native token chunks may cross excerpt boundaries"}


def verify_capture_linkage(folder: Path, mode: str, calibration_path: Path,
                           control_identity: dict, layers: list[int]):
    """Bind a completed capture to frozen corpus, compiled collector and control.

    The original training launch receipt was recovered during the active run.
    Its saved scope is retained verbatim, rather than treating it as a receipt
    established before process launch. Incomplete/failed runs are never eligible.
    """
    folder, calibration_path = Path(folder), Path(calibration_path)
    if mode not in ("train", "validation"):
        raise ValueError("Only complete training and validation captures may feed the pilot")
    expected_chunks, expected_cap = (32, 2048) if mode == "train" else (16, 1024)
    status_path, launch_path = folder / "status.json", folder / "launch-receipt.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    launch = json.loads(launch_path.read_text(encoding="utf-8-sig"))
    if status.get("status") != "complete" or status.get("exit_code") != 0:
        raise ValueError(f"Capture is not complete with exit0: {mode}")
    for record in (status, launch):
        if (record.get("mode"), record.get("chunks"), record.get("cap")) != (mode, expected_chunks, expected_cap):
            raise ValueError("Capture receipt mode/chunks/cap mismatch")
        if set(record.get("layers", [])) != {8, 28, 47}:
            raise ValueError("Capture receipt pilot-layer set mismatch")
    if status.get("pid") is not None and status["pid"] != launch.get("pid"):
        raise ValueError("Completed capture belongs to a different launch PID")
    calibration_cpu_moe = launch.get("calibration_cpu_moe")
    production_cpu_moe = launch.get("production_cpu_moe")
    if calibration_cpu_moe not in (40, 43) or production_cpu_moe != 43 or not launch.get("placement_scope"):
        raise ValueError("Capture placement is not explicitly scoped to declared calibration/production settings")
    calibration_load_mode = launch.get("calibration_load_mode")
    if calibration_load_mode not in ("auto", "none"):
        raise ValueError("Capture loading mode must be recorded")
    for key in ("calibration_cpu_moe", "production_cpu_moe", "placement_scope", "calibration_load_mode"):
        if key not in status or status[key] != launch[key]:
            raise ValueError("Completed capture and launch placement declarations disagree")
    manifest = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    entry = manifest["sets"][mode]
    corpus = Path(launch["corpus"])
    if corpus.resolve() != Path(entry["path"]).resolve() or launch["corpus_sha256"] != entry["sha256"] or sha256(corpus) != entry["sha256"]:
        raise ValueError("Capture corpus differs from frozen calibration entry")
    executable = Path(launch["engine"])
    if not executable.is_file() or sha256(executable) != launch["engine_sha256"]:
        raise ValueError("Capture executable differs from launch fingerprint")
    root = Path(__file__).resolve().parents[1]
    collector_paths = (root / "tools/gateup_capture.h", root / "third_party/llama.cpp/tools/imatrix/gateup_capture.h")
    if any(sha256(path) != launch["collector_sha256"] for path in collector_paths):
        raise ValueError("Collector source differs from captured launch/build source")
    saved_control = launch["control"]
    if Path(saved_control["path"]).resolve() != Path(control_identity["path"]).resolve():
        raise ValueError("Capture used another control path")
    for key in ("bytes", "mtime_ns", "sha256"):
        if saved_control[key] != control_identity[key]:
            raise ValueError(f"Capture preserved-control fingerprint differs: {key}")
    for key in ("corpus", "corpus_sha256", "engine", "engine_sha256", "collector_sha256", "control"):
        if key in status and status[key] != launch[key]:
            raise ValueError(f"Completed status and launch receipt disagree: {key}")
    captures = {}
    for layer in layers:
        if layer not in (8, 28, 47):
            raise ValueError("Unapproved capture layer")
        capture = read_capture(folder / f"layer{layer}.gup", layer)
        if capture.cap != expected_cap or capture.seen != expected_chunks * 512 or capture.batches != expected_chunks * 4:
            raise ValueError("Parsed capture does not match512-token chunks/128-token microbatches")
        if np.any(capture.records["row"] >= 128):
            raise ValueError("Captured local token index exceeds fixed microbatch")
        ordinals = (capture.records["batch"] - 1) * np.uint64(128) + capture.records["row"].astype(np.uint64) + 1
        if not np.array_equal(ordinals, capture.records["token"]):
            raise ValueError("Captured global token ordinals disagree with microbatch/row grouping")
        captures[layer] = capture
    imatrix = folder / "imatrix.gguf"
    if not imatrix.is_file() or imatrix.stat().st_size < 64:
        raise ValueError("Capture's importance matrix is absent/incomplete")
    receipt = {"mode": mode, "status_sha256": sha256(status_path), "launch_receipt_sha256": sha256(launch_path),
               "chunks": expected_chunks, "cap": expected_cap,
               "corpus": str(corpus.resolve()), "corpus_sha256": entry["sha256"],
               "engine": str(executable.resolve()), "engine_sha256": launch["engine_sha256"],
               "collector_sha256": launch["collector_sha256"], "control": saved_control,
               "imatrix_path": str(imatrix.resolve()), "imatrix_sha256": sha256(imatrix),
               "launch_receipt_scope": launch.get("receipt_scope", "Wrapper saved launch metadata immediately after process creation"),
               "launch_receipt_saved_utc": launch.get("receipt_saved_utc"),
               "calibration_cpu_moe": calibration_cpu_moe, "production_cpu_moe": production_cpu_moe,
               "placement_scope": launch["placement_scope"], "calibration_load_mode": calibration_load_mode,
               "captured_ordinals_verified": True,
               "layers": {layer: capture.receipt() for layer, capture in captures.items()}}
    return captures, receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    print(json.dumps(read_capture(args.capture, args.layer).receipt()))
