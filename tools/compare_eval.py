"""Compare paired local quality samples, including uncertainty in score changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FILES = {
    "mmlu": ("mmlu.jsonl", "correct", 456),
    "gsm8k": ("gsm8k.jsonl", "correct", 256),
    "ifeval_strict": ("ifeval-score.examples.jsonl", "strict", 128),
}


def values(folder: Path, filename: str, field: str, expected: int) -> np.ndarray:
    rows = [json.loads(line) for line in (folder / filename).read_text(encoding="utf-8").splitlines()]
    by_index = {row["index"]: row[field] for row in rows}
    if len(rows) != expected or set(by_index) != set(range(expected)):
        raise ValueError(f"Incomplete or duplicate sample indices: {folder / filename}")
    if any(value is None for value in by_index.values()):
        raise ValueError(f"Unscored sample: {folder / filename}")
    return np.array([bool(by_index[index]) for index in range(expected)], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--counts',type=int,nargs=3,default=[456,256,128],metavar=('MMLU','GSM8K','IFEVAL'))
    args = parser.parse_args()
    rng = np.random.default_rng(1729)
    differences = {}
    report = {"baseline": str(args.baseline), "candidate": str(args.candidate),
              "task_results": {}, "human_eval": "Generated code requires isolated execution; not scored here"}
    for (name, (filename, field, _)), count in zip(FILES.items(),args.counts):
        candidate = values(args.candidate, filename, field, count)
        baseline = values(args.baseline, filename, field, count)
        diff = candidate - baseline
        samples = diff[rng.integers(0, count, size=(5000, count))].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        differences[name] = diff
        report["task_results"][name] = {
            "samples": count, "candidate_accuracy": float(candidate.mean()),
            "baseline_accuracy": float(baseline.mean()),
            "difference_percentage_points": round(float(diff.mean() * 100), 3),
            "paired_bootstrap_95_percentage_points": [round(float(low * 100), 3),
                                                       round(float(high * 100), 3)],
            "three_point_gate": ("pass" if low >= -0.03 else
                                  "fail" if high < -0.03 else "inconclusive"),
        }
    macro_samples = np.mean([
        diff[rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
        for diff in differences.values()
    ], axis=0)
    low, high = np.quantile(macro_samples, [0.025, 0.975])
    macro = float(np.mean([diff.mean() for diff in differences.values()]))
    report["macro"] = {
        "difference_percentage_points": round(macro * 100, 3),
        "paired_bootstrap_95_percentage_points": [round(float(low * 100), 3),
                                                   round(float(high * 100), 3)],
        "two_point_gate": ("pass" if low >= -0.02 else
                           "fail" if high < -0.02 else "inconclusive"),
        "scope": "MMLU, GSM8K and strict IFEval samples; HumanEval excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
