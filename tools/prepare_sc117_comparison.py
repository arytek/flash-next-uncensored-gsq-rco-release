"""Freeze verified local inputs for an SC117 comparison, without changing weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/comparison-sc117-20261006"
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf
from tools.evaluate_local import score_row

SC = Path(r"models\SC117\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf")
ISTA = Path(r"models\flash-next-uncensored-gsq-rco-models\comparisons\sc117-20261006\baseline\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")
CONTROL = Path(r"models\flash-next-uncensored-gsq-rco-models\Qwen3.8-Flash-Next-Abliterated-GSQ-RCO-IQ3_XXS-Calibrated-00001-of-00002.gguf")
EXPECTED = {
    "sc117": (SC, 47342144896, "03b11926b41a3d7f7a008882b1bc7a6330a3c9c43d3cad4598d31dc73042fdc5"),
    "ista": (ISTA, 47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
    "control": (CONTROL, 54344308416, "84512ed5aaa14930c56345eaaa88adccddce5a80a700b102c0b9fbfdc955f507"),
}
LOOKUP_SHA = "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def check_time():
    end = datetime.fromisoformat(read(FOLDER / "budget.json")["deadline_utc"])
    if datetime.now(timezone.utc) >= end:
        raise RuntimeError("Separate comparison deadline reached")


def verify_file(path, size, expected, cache):
    check_time()
    stat = path.stat()
    assert stat.st_size == size, f"Incomplete model file: {path}"
    key = str(path)
    old = cache.get(key)
    receipt = {"path": key, "bytes": size, "mtime_ns": stat.st_mtime_ns, "sha256": expected}
    if old != receipt:
        for alias, saved in list(cache.items()):
            if (saved["sha256"] == expected and saved["bytes"] == size
                    and saved["mtime_ns"] == stat.st_mtime_ns and Path(alias).exists()
                    and path.samefile(alias)):
                cache[key] = receipt
                old = receipt
                (FOLDER / "file-receipts.json").write_text(json.dumps(cache, indent=2) + "\n")
                break
    if old != receipt:
        print(f"Checking SHA256: {path.name}", flush=True)
        assert digest(path) == expected, f"Model checksum mismatch: {path}"
        assert path.stat().st_mtime_ns == stat.st_mtime_ns, "Model changed while hashing"
        cache[key] = receipt
        (FOLDER / "file-receipts.json").write_text(json.dumps(cache, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-baseline", action="store_true", help="Verify available SC117/control inputs while the deleted baseline is restored")
    args = parser.parse_args()
    cache_file = FOLDER / "file-receipts.json"
    cache = read(cache_file) if cache_file.exists() else {}
    models = {}
    readers = {}
    for label, (path, size, expected) in EXPECTED.items():
        if args.without_baseline and label == "ista":
            continue
        receipt = verify_file(path, size, expected, cache)
        lookup = Path(str(path).replace("00001-of-00002", "00002-of-00002"))
        lookup_receipt = verify_file(lookup, 28800138432, LOOKUP_SHA, cache)
        reader = gguf.GGUFReader(path)
        lookup_reader = gguf.GGUFReader(lookup)
        assert reader.get_field("general.architecture").contents() == "qwen4exp"
        assert len(reader.tensors) == 1223 and len(lookup_reader.tensors) == 1
        assert reader.get_field("split.count").contents() == 2
        assert reader.get_field("split.tensors.count").contents() == 1224
        readers[label] = reader
        down = [t for t in reader.tensors if t.name.endswith("ffn_down_exps.weight")]
        assert len(down) == 48
        models[label] = {"main": receipt, "lookup": lookup_receipt,
                         "total_bytes": size + lookup_receipt["bytes"],
                         "expert_down_formats": dict(Counter(t.tensor_type.name for t in down)),
                         "tensor_formats": dict(Counter(t.tensor_type.name for t in reader.tensors)),
                         "chat_template_sha256": hashlib.sha256(reader.get_field("tokenizer.chat_template").contents().encode()).hexdigest()}
    shapes = [{t.name: t.shape.tolist() for t in reader.tensors} for reader in readers.values()]
    assert all(shape == shapes[0] for shape in shapes), "Tensor shapes or names differ"
    assert len({model["chat_template_sha256"] for model in models.values()}) == 1, "Chat templates differ; review protocol first"
    selected = ROOT / "data/eval/selected"
    selected_manifest = read(selected / "manifest.json")
    inputs = {}
    historical_sources = {}
    for name, entry in selected_manifest["sets"].items():
        path = selected / (name + ".jsonl")
        assert digest(path) == entry["sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == entry["rows"]
        inputs[name] = {"sha256": entry["sha256"], "rows": entry["rows"]}
        for reference, folder_name in (("ista", "ista-xxs-final"), ("control", "calibrated-final")):
            source = ROOT / "data/eval/results" / (f"review-{reference}-code" if name == "humaneval" else folder_name) / (name + ".jsonl")
            answers = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            assert len(answers) == len(rows)
            assert {a["index"] for a in answers} == set(range(len(rows)))
            missing_hashes = 0
            for answer in answers:
                question = rows[answer["index"]]
                if "row_sha256" in answer:
                    assert answer["row_sha256"] == hashlib.sha256(json.dumps(question, sort_keys=True).encode()).hexdigest()
                else:
                    missing_hashes += 1
                    if name in ("mmlu", "gsm8k"):
                        correct, parsed = score_row(name, question, answer["response"])
                        assert answer["correct"] == correct and answer["parsed"] == parsed, "Historical scores disagree with frozen questions"
                        assert answer["protocol"] == ("constrained-choice-v1" if name == "mmlu" else "chat-512-v1")
                    elif name == "ifeval":
                        assert answer["prompt"] == question["prompt"] and answer["key"] == question["key"]
                    else:
                        raise AssertionError("Coding reference lacks a per-row fingerprint")
            historical_sources[f"{reference}-{name}"] = {"path": str(source), "sha256": digest(source),
                "rows_without_per_row_sha256": missing_hashes,
                "identity": "Per-row SHA where available; otherwise frozen dataset/indices, prompt equality or recomputed score consistency. Historical generation lacks full per-row provenance."}
    manifest = {"models": models, "datasets": inputs, "historical_sources": historical_sources,
                "ista_revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
                "sc117_revision": "10fe455ecf05eb0f480bff3ba8e1ace4f7c0e6fd",
                "runtime": {"engine_commit": "86a24a182", "build": 11199, "threads": 12, "cpu_moe": 43,
                            "gpu_layers": 999, "lazy_mode": "on", "cache_k": "q8_0", "cache_v": "q5_1",
                            "flash_attention": "on", "quality_context": 16384, "batch": 512, "microbatch": 128},
                "quality_reference": "Existing frozen ISTA/control answers; verify row hashes; no tuning or new model selection",
                "speed_scope": "Five short synthetic512 generations per model; separate256 prompt. No populated16K repeat",
                "privacy": "Local only; no weights or code published", "old_48h_budget_changed": False}
    out = FOLDER / ("sc117-preflight.json" if args.without_baseline else "manifest.json")
    if out.exists():
        assert read(out) == manifest, "Comparison inputs changed"
    else:
        out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Model fingerprints, shapes, templates and paired question identities verified", flush=True)


if __name__ == "__main__":
    main()
