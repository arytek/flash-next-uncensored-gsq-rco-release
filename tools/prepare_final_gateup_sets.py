"""Freeze fresh calibration/development/confirmation sets for the final pilot.

Selection is local and deterministic. Earlier sets and unused confirmation
conditions remain untouched. This script never runs a model or generated code.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/optimization-final-gateup"
SEED = 2026100601
CHARS = 2800
PRIOR_BENCHMARK_FOLDERS = (
    "data/eval/selected",
    "data/optimization-48h/development",
    "data/optimization-48h/confirmation",
    "data/optimization-48h/confirmation-q8",
)
PRIOR_CALIBRATION_MANIFESTS = (
    "data/calibration/selected/manifest.json",
    "data/optimization-48h/calibration/manifest.json",
    "data/optimization-48h/calibration/extended-manifest.json",
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows_at(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(row):
    return " ".join(row.get("question", row.get("prompt", "")).split())


def freeze_bytes(path: Path, contents: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f"Previously frozen file would change: {path}")
        return
    path.write_bytes(contents)


def freeze_manifest(path: Path, manifest: dict):
    if path.exists():
        previous = read(path)
        manifest["frozen_utc"] = previous["frozen_utc"]
        if previous != manifest:
            raise ValueError(f"Frozen manifest would change: {path}")
        return
    contents = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    freeze_bytes(path, contents)


def encode_rows(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def prepare_calibration(now, builder_sha):
    source = ROOT / "data/calibration/builds/qwen3.8-flash-next-moe/calib_train.txt"
    original = read(ROOT / PRIOR_CALIBRATION_MANIFESTS[0])
    corpus_sha = sha(source)
    assert corpus_sha == original["source_sha256"]
    text = source.read_text(encoding="utf-8")
    prior_intervals = set()
    prior_records = {}
    for relative in PRIOR_CALIBRATION_MANIFESTS:
        path = ROOT / relative
        item = read(path)
        assert item.get("source_sha256", item.get("corpus_sha256")) == corpus_sha
        for key, values in item.items():
            if key.endswith("_starts"):
                prior_intervals.update((int(start), int(start) + CHARS) for start in values)
        prior_records[relative] = sha(path)
    training_base = original["train_starts"]
    heldout_base = original["heldout_starts"]
    starts = {
        "train": [training_base[index * len(training_base) // 32] + 30000 for index in range(32)],
        "validation": [training_base[2 + index * 8] + 75000 for index in range(16)],
        "confirmation-text": [heldout_base[index * len(heldout_base) // 16] + 7000 for index in range(16)],
    }
    all_new = []
    sets = {}
    for name, positions in starts.items():
        excerpts = []
        provenance = []
        for index, start in enumerate(positions):
            interval = (start, start + CHARS)
            assert 0 <= start and interval[1] <= len(text)
            if name != "confirmation-text":
                assert interval[1] <= len(text) * 9 // 10
            else:
                assert start >= len(text) * 9 // 10
            assert not any(overlaps(interval, old) for old in prior_intervals)
            assert not any(overlaps(interval, other) for other in all_new)
            all_new.append(interval)
            raw = text[start:interval[1]]
            cleaned = "".join(character if character in "\t\n\r" or ord(character) >= 32 else " " for character in raw)
            assert len(cleaned) == CHARS and "\0" not in cleaned
            excerpts.append(cleaned)
            provenance.append({"index": index, "start_character": start, "end_character_exclusive": interval[1],
                "chars": CHARS, "raw_excerpt_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "clean_excerpt_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest()})
        text_path = FOLDER / "calibration" / f"{name}.txt"
        records_path = text_path.with_suffix(".excerpts.jsonl")
        freeze_bytes(text_path, ("\n\n".join(excerpts) + "\n").encode("utf-8"))
        freeze_bytes(records_path, encode_rows(provenance))
        sets[name] = {"path": str(text_path), "sha256": sha(text_path), "excerpts": len(positions),
                      "starts": positions, "chars_per_excerpt": CHARS,
                      "excerpt_records": str(records_path), "excerpt_records_sha256": sha(records_path)}
    manifest = {"frozen_utc": now, "builder_sha256": builder_sha,
        "source": original["source"], "corpus_revision": original["revision"],
        "corpus_sha256": corpus_sha, "corpus_characters": len(text), "sets": sets,
        "selection": "Deterministic widely spaced gaps:32training,16operator-validation,16final confirmation excerpts. Source-character intervals disjoint from all prior selected and extended calibration intervals.",
        "token_scope": "2800 characters per excerpt is approximately512 tokens, not an exact token count. Native capture must verify finite arrays/counts and enforce512-token chunks; concatenated native chunks can straddle excerpt boundaries.",
        "activation_reference": "The preserved abliterated control; fresh unchanged BF16 gate/up plus preserved edited expert-down weights supply block targets. Not a complete BF16 whole-model teacher.",
        "exclusion": {"prior_manifest_sha256": prior_records, "unique_prior_intervals": len(prior_intervals),
                      "source_interval_overlap_with_prior": 0, "overlap_between_new_sets": 0},
        "confirmation_rule": "Confirmation text reserved for only the predeclared compact candidate and fixed references after pilot/development success. Do not inspect resulting answers/logits to tune weights or choose another candidate.",
        "privacy": "Local only", "models_run": 0}
    freeze_manifest(FOLDER / "calibration/manifest.json", manifest)
    return {"counts": {name: item["excerpts"] for name, item in sets.items()},
            "prior_intervals_excluded": len(prior_intervals), "overlap": 0}


def prepare_benchmarks(now, builder_sha):
    original = read(ROOT / "data/eval/selected/manifest.json")
    rng = random.Random(SEED)
    records = {"development": {}, "confirmation": {}}
    exclusions = {}
    sources = {}
    for name in ("mmlu", "gsm8k", "ifeval"):
        excluded = set()
        prior_fingerprints = {}
        for relative in PRIOR_BENCHMARK_FOLDERS:
            path = ROOT / relative / f"{name}.jsonl"
            manifest = read(path.parent / "manifest.json")
            assert sha(path) == manifest["sets"][name]["sha256"]
            prior_fingerprints[relative] = sha(path)
            excluded.update(identity(row) for row in rows_at(path))
        info = original["sources"][name]
        source = ROOT / info["path"]
        assert sha(source) == info["sha256"]
        sources[name] = info
        raw = rows_at(source) if name == "ifeval" else parquet.read_table(source).to_pylist()
        available = {}
        for row in raw:
            key = identity(row)
            assert key
            if key not in excluded:
                available.setdefault(key, row)
        available_before = len(available)
        chosen_sets = {}
        for phase in ("development", "confirmation"):
            pool = list(available.values())
            if name == "mmlu":
                subjects = defaultdict(list)
                for row in pool:
                    subjects[row["subject"]].append(row)
                per_subject = 2 if phase == "development" else 4
                assert len(subjects) == 57 and all(len(group) >= per_subject for group in subjects.values())
                chosen = [row for subject in sorted(subjects) for row in rng.sample(subjects[subject], per_subject)]
                rng.shuffle(chosen)
            else:
                requested = (64 if name == "gsm8k" else 32) if phase == "development" else 128
                if len(pool) < requested and name != "ifeval":
                    raise ValueError("Requested fresh benchmark size infeasible")
                chosen = rng.sample(pool, min(requested, len(pool)))
            keys = {identity(row) for row in chosen}
            assert len(keys) == len(chosen) and not keys & excluded
            for key in keys:
                del available[key]
            chosen_sets[phase] = keys
            path = FOLDER / phase / f"{name}.jsonl"
            freeze_bytes(path, encode_rows(chosen))
            records[phase][name] = {"path": str(path), "rows": len(chosen), "sha256": sha(path)}
        assert not chosen_sets["development"] & chosen_sets["confirmation"]
        exclusions[name] = {"prior_set_sha256": prior_fingerprints, "excluded_question_identities": len(excluded),
            "fresh_pool_before_selection": available_before, "prior_overlap": 0, "development_confirmation_overlap": 0,
            "fresh_pool_after_selection": len(available)}
    for phase in ("development", "confirmation"):
        manifest = {"frozen_utc": now, "builder_sha256": builder_sha, "selection_seed": SEED,
            "phase": phase, "sets": records[phase], "sources": sources, "exclusion": exclusions,
            "counts": [records[phase][name]["rows"] for name in ("mmlu", "gsm8k", "ifeval")],
            "purpose": "New local development screening; never inspect frozen confirmation answers to tune weights" if phase == "development" else "Single compact candidate confirmation with unchanged references; no further artifact search or tuning from resulting answers",
            "candidate_binding": "Confirmation is reserved for one compact candidate predeclared by exact weight SHA/allocation before generation, only after pilot/development gates pass. It is not available to earlier dense-F16/Q8 trials or any later alternative candidate.",
            "runtime": "Fixed production86a24a182/build11199;t12/ncmoe43/ngl999/lazy-on/Q8_0-Q5_1KV/FA; no populated16K speed repeat",
            "mmlu_coverage": "Two questions per each of57subjects" if phase == "development" else "Four questions per each of57subjects",
            "freshness": "Question identities are new to locally selected sets, including both unused old confirmations. Pretraining or benchmark exposure unknown; small samples widen uncertainty.",
            "coding": "All164local HumanEval tasks were previously used. Any repeat is explicitly reused evidence, never fresh confirmation. No project-local MBPP dataset found; generated code requires unchanged verified offline constrained container executor.",
            "models_run": 0, "privacy": "Local only; no uploads, publishing, external messages"}
        freeze_manifest(FOLDER / phase / "manifest.json", manifest)
    return {phase: {name: item["rows"] for name, item in record.items()} for phase, record in records.items()}


def main():
    now = datetime.now(timezone.utc).isoformat()
    builder_sha = sha(Path(__file__))
    calibration = prepare_calibration(now, builder_sha)
    benchmark = prepare_benchmarks(now, builder_sha)
    coding_paths = [str(path) for path in (ROOT / "data/eval").rglob("*mbpp*") if path.is_file()]
    assert not coding_paths, "MBPP inventory changed; inspect provenance before claiming it unavailable"
    print(json.dumps({"calibration": calibration, "benchmarks": benchmark, "mbpp_local_files": coding_paths,
        "models_run": 0, "answers_used_for_tuning": False, "prior_sets_untouched": True}))


if __name__ == "__main__":
    main()
