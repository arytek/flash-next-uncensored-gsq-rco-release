"""Audit completed comparison records and close the separate local comparison.

No inference, generated-code execution, weight changes or new optimisation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools import score_humaneval_isolated as isolated

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/comparison-sc117-20261006"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main():
    manifest = read(FOLDER / "manifest.json")
    for model in manifest["models"].values():
        for part in ("main", "lookup"):
            receipt = model[part]
            stat = Path(receipt["path"]).stat()
            assert stat.st_size == receipt["bytes"] and stat.st_mtime_ns == receipt["mtime_ns"], "Previously verified model changed"
    for source in manifest["historical_sources"].values():
        assert digest(source["path"]) == source["sha256"]
    sample_counts = {}
    all_answers = {}
    for name, entry in manifest["datasets"].items():
        dataset = ROOT / f"data/eval/selected/{name}.jsonl"
        assert digest(dataset) == entry["sha256"]
        questions = rows(dataset)
        answers = rows(ROOT / f"data/eval/results/sc117-20261006/{name}.jsonl")
        assert len(answers) == entry["rows"] and {a["index"] for a in answers} == set(range(entry["rows"]))
        for answer in answers:
            question = questions[answer["index"]]
            assert answer["row_sha256"] == isolated.sha(json.dumps(question, sort_keys=True).encode())
        all_answers[name] = {answer["index"]: answer for answer in answers}
        sample_counts[name] = len(answers)
    coding_manifest = read(FOLDER / "coding/manifest.json")
    assert digest(ROOT / "tools/score_humaneval_isolated.py") == coding_manifest["executor_sha256"]
    assert digest(ROOT / "tools/score_sc117_coding.py") == coding_manifest["script_sha256"]
    assert isolated.sha(isolated.RUNNER.encode()) == coding_manifest["runner_sha256"]
    assert isolated.IMAGE == coding_manifest["image"] and isolated.LIMITS == coding_manifest["limits"]
    assert digest(ROOT / "data/eval/results/sc117-20261006/humaneval.jsonl") == coding_manifest["responses_sha256"]
    prior = ROOT / "data/optimization-48h/coding-review"
    assert digest(prior / "manifest.json") == coding_manifest["reference_manifest_sha256"]
    assert digest(prior / "results.jsonl") == coding_manifest["reference_results_sha256"]
    frozen_hash = isolated.sha(json.dumps(coding_manifest, sort_keys=True).encode())
    coding = rows(FOLDER / "coding/results.jsonl")
    questions = rows(ROOT / "data/eval/selected/humaneval.jsonl")
    assert len(coding) == 164 and {r["index"] for r in coding} == set(range(164))
    for result in coding:
        index = result["index"]
        response = all_answers["humaneval"][index]["response"]
        assert result["task_id"] == questions[index]["task_id"]
        assert result["manifest_sha256"] == frozen_hash
        stopped = isolated.stopped_completion(questions[index]["prompt"], response)
        for mode, text in (("raw", response), ("dedent_stop_diagnostic", stopped)):
            assert result[mode]["program_sha256"] == isolated.sha(isolated.program(questions[index], text).encode())
    coding_summary = read(FOLDER / "coding/summary.json")
    for mode in ("raw", "dedent_stop_diagnostic"):
        assert sum(r[mode]["outcome"] == "passed" for r in coding) == coding_summary[mode]["passed"]
    verification = read(FOLDER / "coding/executor-verification.json")
    assert {k: v["outcome"] for k, v in verification["cases"].items()} == {
        "canonical": "passed", "incorrect": "failed", "timeout": "timed-out", "isolation": "passed"}
    behavior = read(FOLDER / "behavior-review.json")
    assert behavior["samples"] == 20
    assert digest(ROOT / "data/eval/results/sc117-20261006-behavior/ifeval.jsonl") == behavior["answers_sha256"]
    speeds = {}
    for key in ("control-a", "sc117", "ista", "control-b"):
        label = f"sc117compare-20261006-{key}"
        measurements = rows(ROOT / f"logs/bench-{label}.jsonl")
        receipt = read(ROOT / f"logs/bench-{label}.memory.json")
        assert receipt["exit_code"] == 0
        model_key = "control" if key.startswith("control") else key
        for result in measurements:
            assert result["model_filename"] == manifest["models"][model_key]["main"]["path"]
            assert result["build_commit"] == "86a24a182" and result["build_number"] == 11199
            assert (result["n_threads"], result["n_cpu_moe"], result["n_gpu_layers"]) == (12, 43, 999)
            assert (result["type_k"], result["type_v"], result["lazy_mode"], result["flash_attn"]) == ("q8_0", "q5_1", "on", 1)
            assert (result["n_batch"], result["n_ubatch"]) == (2048, 512)
        gen = [r for r in measurements if r["n_gen"] == 512]
        prompt = [r for r in measurements if r["n_prompt"] == 256 and r["n_gen"] == 0]
        assert len(gen) == len(prompt) == 1 and len(gen[0]["samples_ts"]) == 5
        speeds[key] = {"mean": gen[0]["avg_ts"], "minimum": min(gen[0]["samples_ts"]), "prompt_mean": prompt[0]["avg_ts"]}
    telemetry = {}
    for label in ("eval-sc117-20261006", *(f"bench-sc117compare-20261006-{k}" for k in speeds)):
        observations = rows(ROOT / f"logs/{label}.telemetry.jsonl")
        ram = np.array([r["available_ram_mb"] for r in observations])
        gpu = np.array([float(str(r["gpu"]).split(",")[0]) for r in observations])
        pages = np.array([r["pages_input_per_second"] for r in observations])
        telemetry[label] = {"samples": len(observations), "ram_gib_min_median_max": (np.quantile(ram, [0, .5, 1]) / 1024).tolist(),
                            "fraction_ram_at_least_6gib": float((ram >= 6144).mean()), "gpu_used_mib_max": float(gpu.max()),
                            "pages_input_per_second_median_max": np.quantile(pages, [.5, 1]).tolist(),
                            "scope": "All recorded phases include loading; page input includes mapped-file reads and cannot establish pagefile traffic"}
    startup = (ROOT / "logs/server-sc117-20261006.stderr.log").read_text(encoding="utf-8-sig")
    assert "flash_attn            = enabled" in startup and "lazy read enabled" in startup and "n_threads = 12" in startup
    assert "39238.67 MiB" in startup and "9015.51 MiB" in startup
    control_mean = (speeds["control-a"]["mean"] + speeds["control-b"]["mean"]) / 2
    audit = {"utc": datetime.now(timezone.utc).isoformat(), "sample_counts": sample_counts, "coding_cases_verified": 164,
             "model_receipts_unchanged": True, "model_hash_scope": "Full hashes verified during preparation/download; lengths and timestamps unchanged at completion",
             "current_question_and_response_fingerprints_verified": True, "historical_provenance_limit": "Historical quality rows without per-row hashes retain the disclosed dataset/index/score-consistency limitation",
             "coding_executor_and_program_fingerprints_verified": True, "offline_container_probes_passed": True,
             "behavior_review_verified": True, "fixed_benchmark_fields_verified": True, "speed": speeds,
             "control_combined_generation_mean": control_mean, "control_bracket_drift_percent": (speeds["control-b"]["mean"] / speeds["control-a"]["mean"] - 1) * 100,
             "control_faster_than_sc117_percent": (control_mean / speeds["sc117"]["mean"] - 1) * 100,
             "sc117_slower_than_ista_percent": (1 - speeds["sc117"]["mean"] / speeds["ista"]["mean"]) * 100,
             "telemetry": telemetry, "memory_conclusion": "SC117 quality-run RAM minimum0.56GiB, median3.91GiB; sustained6GiB goal not demonstrated. VRAM headroom exceeded1GiB. Paging-source attribution unverified.",
             "scope": "Short synthetic speed, frozen local quality samples; no repeated populated16K test, new ablation/GSQ build, or long-context validation"}
    (FOLDER / "final-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    status = read(FOLDER / "status.json")
    status.update(status="complete", stage="Comparison complete; preserve our model for speed; SC117 is a smaller reference with mixed quality", completed_utc=audit["utc"], candidate_selected=False, no_automatic_work_remaining=True)
    (FOLDER / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    budget = read(FOLDER / "budget.json")
    budget.update(status="completed-comparison", completed_utc=audit["utc"])
    (FOLDER / "budget.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "samples": sample_counts, "coding_verified": 164, "control_speed": control_mean, "memory_review": audit["memory_conclusion"]}), flush=True)


if __name__ == "__main__":
    main()
