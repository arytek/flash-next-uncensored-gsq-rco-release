"""Compare SC117 coding answers through the frozen offline container executor.

Only the deadline provider changes for this separate comparison. Container
limits, image, runner, primary raw protocol and diagnostic are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools import score_humaneval_isolated as isolated
from tools.evaluate_local import wilson

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/comparison-sc117-20261006/coding"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_deadline():
    return datetime.fromisoformat(read(FOLDER.parent / "budget.json")["deadline_utc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    prior_folder = ROOT / "data/optimization-48h/coding-review"
    prior = read(prior_folder / "manifest.json")
    assert digest(ROOT / "tools/score_humaneval_isolated.py") == prior["script_sha256"]
    assert isolated.IMAGE == prior["image"] and isolated.LIMITS == prior["limits"]
    assert isolated.sha(isolated.RUNNER.encode()) == prior["runner_sha256"]
    dataset = ROOT / "data/eval/selected/humaneval.jsonl"
    assert digest(dataset) == prior["dataset_sha256"]
    problems = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    assert len(problems) == 164
    # Keep the old experiment immutable; bind its unchanged guard to this budget.
    isolated.deadline = comparison_deadline
    isolated.check_time()
    assert isolated.cli(["info", "--format", "{{.OSType}}" ]).strip() == "linux"
    isolated.cli(["image", "inspect", isolated.IMAGE])
    verification = isolated.verify_sandbox(problems[0])
    (FOLDER / "executor-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if args.preflight_only:
        print("Offline constrained executor and known-answer probes verified", flush=True)
        return
    source = ROOT / "data/eval/results/sc117-20261006/humaneval.jsonl"
    responses = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    answers = {row["index"]: row for row in responses}
    assert len(responses) == 164 and set(answers) == set(range(164))
    for index, row in answers.items():
        assert row["set"] == "humaneval" and row["correct"] is None
        assert row["row_sha256"] == isolated.sha(json.dumps(problems[index], sort_keys=True).encode())
    for label in ("ista", "control"):
        assert digest(prior["sources"][label]["path"]) == prior["sources"][label]["sha256"]
    manifest = {"executor_sha256": prior["script_sha256"], "script_sha256": digest(__file__),
                "runner_sha256": prior["runner_sha256"], "image": isolated.IMAGE, "limits": isolated.LIMITS,
                "dataset_sha256": digest(dataset), "responses_sha256": digest(source),
                "reference_manifest_sha256": digest(prior_folder / "manifest.json"),
                "reference_results_sha256": digest(prior_folder / "results.jsonl"),
                "primary": prior["primary"], "secondary": prior["secondary"], "generation": prior["generation"],
                "deadline_adapter": "Separate comparison budget only; frozen container execution and old48h files unchanged",
                "scope": "Local raw256 coding comparison; diagnostic separate; not public leaderboard parity or proof of quality equivalence"}
    manifest_path = FOLDER / "manifest.json"
    if manifest_path.exists():
        assert read(manifest_path) == manifest, "Frozen coding inputs changed"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    frozen_hash = isolated.sha(json.dumps(manifest, sort_keys=True).encode())
    output = FOLDER / "results.jsonl"
    done = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["manifest_sha256"] == frozen_hash and row["index"] not in done
            done[row["index"]] = row
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, problem in enumerate(problems):
            isolated.check_time()
            response = answers[index]["response"]
            stopped = isolated.stopped_completion(problem["prompt"], response)
            raw_program = isolated.program(problem, response)
            diagnostic_program = isolated.program(problem, stopped)
            if index in done:
                assert done[index]["raw"]["program_sha256"] == isolated.sha(raw_program.encode())
                assert done[index]["dedent_stop_diagnostic"]["program_sha256"] == isolated.sha(diagnostic_program.encode())
                continue
            raw = isolated.run_case(raw_program)
            diagnostic = isolated.run_case(diagnostic_program) if stopped != response else {**raw, "reused_identical_program": True}
            row = {"model": "sc117", "index": index, "task_id": problem["task_id"], "raw": raw,
                   "dedent_stop_diagnostic": diagnostic, "trimmed_characters": len(response) - len(stopped),
                   "manifest_sha256": frozen_hash, "utc": isolated.utc()}
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
            done[index] = row
            if (index + 1) % 16 == 0 or index == 163:
                print(f"SC117 coding: {index + 1}/164 scored in isolation", flush=True)
    prior_hash = isolated.sha(json.dumps(prior, sort_keys=True).encode())
    references = {}
    for line in (prior_folder / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["manifest_sha256"] == prior_hash
        if row["model"] in ("ista", "control"):
            key = (row["model"], row["index"])
            assert key not in references
            references[key] = row
    assert len(references) == 328
    summary = {"samples": 164, "manifest_sha256": frozen_hash, "scope": manifest["scope"], "comparisons": {}}
    for mode in ("raw", "dedent_stop_diagnostic"):
        candidate = np.array([done[index][mode]["outcome"] == "passed" for index in range(164)], dtype=float)
        count = int(candidate.sum())
        summary[mode] = {"passed": count, "pass_at_1": count / 164, "wilson_95": wilson(count, 164)}
        for reference in ("ista", "control"):
            baseline = np.array([references[reference, index][mode]["outcome"] == "passed" for index in range(164)], dtype=float)
            delta = candidate - baseline
            rng = np.random.default_rng(1729)
            draws = delta[rng.integers(0, 164, size=(5000, 164))].mean(axis=1) * 100
            summary["comparisons"].setdefault(reference, {})[mode] = {
                "difference_pp": float(delta.mean() * 100), "paired_bootstrap_95_pp": np.quantile(draws, [.025, .975]).tolist()}
    (FOLDER / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
