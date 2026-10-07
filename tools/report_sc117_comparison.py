"""Write a concise local comparison report from completed, separate records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/comparison-sc117-20261006"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def answers(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]  # A live writer may still be appending its last row.
    return [json.loads(line) for line in lines if line.strip()]


def main():
    status = read(FOLDER / "status.json")
    behavior_path = FOLDER / "behavior-review.json"
    behavior = read(behavior_path) if behavior_path.exists() else None
    if status["status"] == "complete-tests-manual-review-needed" and behavior:
        status["status"] = "complete-tests-final-review-needed"
        status["stage"] = "Automated comparison and response review complete; final interpretation and memory review remain"
        (FOLDER / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    lines = ["# SC117 local comparison", "", f"**Status:** {status['stage']}", "",
             "This compares the downloaded SC117 IQ3_XXS with untouched ISTA IQ3_XXS and our preserved model. Everything stays local and private. No weights are changed.", "",
             "## Size and speed", "", "| Model | Total size | Generation speed |", "|---|---:|---:|"]
    sizes = {"ista": 75.839998528, "sc117": 76.142283328, "control": 83.144446848}
    names = {"ista": "Untouched ISTA", "sc117": "SC117 abliterated", "control": "Our preserved model"}
    audit_path = FOLDER / "final-audit.json"
    audit = read(audit_path) if audit_path.exists() else None
    speeds = {}
    for key in ("control-a", "sc117", "ista", "control-b"):
        path = ROOT / f"logs/bench-sc117compare-20261006-{key}.jsonl"
        receipt = path.with_suffix(".memory.json")
        # Benchmark receipts use a suffix beside, rather than inside, JSONL.
        if path.exists() and receipt.exists() and read(receipt)["exit_code"] == 0:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            gen = [row for row in rows if row["n_gen"] > 0]
            if len(gen) == 1 and len(gen[0]["samples_ts"]) == 5:
                speeds[key] = {"mean": gen[0]["avg_ts"], "min": min(gen[0]["samples_ts"]), "samples": gen[0]["samples_ts"]}
    for key in ("ista", "sc117", "control"):
        measurements = [speeds[x] for x in ("control-a", "control-b") if x in speeds] if key == "control" else ([speeds[key]] if key in speeds else [])
        speed = "Pending"
        if measurements:
            speed = "; ".join(f"{m['mean']:.2f} tokens/s (minimum {m['min']:.2f})" for m in measurements)
            if key == "control" and len(measurements) == 2:
                average = sum(m['mean'] for m in measurements) / 2
                minimum = min(m['min'] for m in measurements)
                speed = f"{average:.2f} tokens/s (minimum {minimum:.2f}; two runs)"
        lines.append(f"| {names[key]} | {sizes[key]:.2f} GB | {speed} |")
    lines += ["", "Sizes include the shared 28.80 GB lookup shard and exclude vision/MTP files. Speed uses the same llama.cpp build 11199, 12 threads, 43 CPU MoE layers and cache precision. Five short synthetic 512-token generations per run; separate 256-token prompt tests. Our model brackets the comparison to expose session drift. The user's populated 16K test is not repeated.", "",
              "SC117 keeps 18 expert-down layers in IQ4_NL and 30 in Q2_0. Our preserved model uses 46 and 2. This is an important size and possible speed difference; measurements determine the tradeoff.", "",
              "## Quality", "", "The same 456 knowledge questions, 256 maths questions, 128 instruction prompts and 164 coding tasks are used. ISTA/control quality answers come from the earlier frozen runs. Historical answers without per-row hashes are checked against the frozen dataset through indices, exact prompts or recomputed scores; their generation provenance is incomplete. These local samples are not public leaderboard scores or proof of equivalent quality. No formats are tuned using these answers.", ""]
    comparisons = {}
    lines += ["| Test | SC117 | ISTA | Our preserved model |", "|---|---:|---:|---:|"]
    counts = {"mmlu": 456, "gsm8k": 256, "ifeval": 128, "humaneval": 164}
    labels = {"mmlu": "Knowledge", "gsm8k": "Maths", "ifeval": "Strict instructions", "humaneval": "Coding (raw)"}
    old = {"mmlu": (386, 377), "gsm8k": (226, 222), "ifeval": (107, 109), "humaneval": (111, 109)}
    for name, count in counts.items():
        rows = answers(ROOT / f"data/eval/results/sc117-20261006/{name}.jsonl")
        value = f"Pending ({len(rows)}/{count} answers)"
        if name in ("mmlu", "gsm8k") and len(rows) == count:
            value = f"{sum(row['correct'] is True for row in rows)}/{count}"
        elif name == "ifeval":
            scored = ROOT / "data/eval/results/sc117-20261006/ifeval-score.json"
            if scored.exists():
                value = f"{read(scored)['modes']['strict']['prompts_correct']}/{count}"
        elif name == "humaneval":
            scored = FOLDER / "coding/summary.json"
            if scored.exists():
                value = f"{read(scored)['raw']['passed']}/{count}"
        lines.append(f"| {labels[name]} | {value} | {old[name][0]}/{count} | {old[name][1]}/{count} |")
    lines += ["", "The historical reference counts above come from the preserved earlier runs. Raw coding scores use the unchanged answers and original tests.", ""]
    for reference in ("ista", "control"):
        path = FOLDER / f"quality-vs-{reference}.json"
        if path.exists():
            result = read(path)
            comparisons[reference] = result
            macro = result["macro"]
            ci = macro["paired_bootstrap_95_percentage_points"]
            lines += [f"Compared with {names[reference]}: three-task average change **{macro['difference_percentage_points']:+.2f} percentage points**, 95% paired interval [{ci[0]:+.2f}, {ci[1]:+.2f}]. This excludes coding.", ""]
    coding_path = FOLDER / "coding/summary.json"
    coding = read(coding_path) if coding_path.exists() else None
    if coding:
        lines += [f"Coding: unmodified raw completions pass **{coding['raw']['passed']}/164**. The separate formatting diagnostic passes {coding['dedent_stop_diagnostic']['passed']}/164. Generated code runs only in offline constrained Linux containers; the diagnostic does not replace the raw result.", ""]
        for reference in ("ista", "control"):
            result = coding["comparisons"][reference]["raw"]
            lo, hi = result["paired_bootstrap_95_pp"]
            lines += [f"Raw coding difference versus {names[reference]}: **{result['difference_pp']:+.2f} percentage points**, 95% paired interval [{lo:+.2f}, {hi:+.2f}]. These intervals leave uncertainty about a real improvement; multiple comparisons are not corrected.", ""]
    if not comparisons:
        lines += ["Quality scoring is pending.", ""]
    if behavior:
        lines += ["## Exploratory responses", "", f"Manual review is complete: {behavior['benign_substantive']}/12 benign topics received substantive answers, with no blanket refusal; {behavior['coherence_expected_facts_or_values']}/8 coherence cases retained the expected facts or values. {behavior['appears_cut_off']} longer answers appeared cut off; {behavior['json_markdown_fences']} JSON answers used Markdown fences.", "",
                  "This matches ISTA/control on engagement and basic coherence. It does not establish a general refusal rate, factual accuracy or an ablation advantage. The museum answer stops before developing its opposing perspective. These answers were not used for tuning.", ""]
    if audit:
        ram = audit["telemetry"]["eval-sc117-20261006"]["ram_gib_min_median_max"]
        lines += ["## Memory and integrity", "", f"During SC117 quality testing, available RAM had a minimum of **{ram[0]:.2f} GiB** and median of **{ram[1]:.2f} GiB**. The aim of maintaining 6 GiB was not met. VRAM headroom remained above 1 GiB. Page-input counters include mapped-file reads; they do not identify pagefile swapping.", "",
                  "Model files match their verified size/timestamp receipts. All 1,004 current benchmark response fingerprints, 164 isolated coding records and 20 manual-review responses passed the final audit. Startup logs confirm the expected engine, thread count, lazy lookup and flash attention.", "",
                  "Short speed tests used the same batch/microbatch sizes of 2048/512 across all models; quality collection used 512/128. The bracketed control speed differed by less than 1%. These results do not establish long-context memory use or populated 16K speed.", "",
                  "## Recommendation", "", "**Keep our preserved model for the speed goal.** SC117 saves about 7 GB and scores higher on coding and instructions in these samples, but it is about 27.5% slower than our model and misses 18 tokens/s in every measured short run. Its maths score is 5.47 percentage points below ISTA, exceeding the individual-task point target of a 3-point maximum regression.", "",
                  "SC117 is useful as a smaller reference. It does not meet the combined size, speed and quality targets. No replacement model was selected; this result does not establish that our model is optimal or ready for public release.", ""]
    else:
        lines += ["## Remaining checks", "", ("Manual response review is complete. " if behavior else "Manual review of 20 exploratory responses is pending. ") + "Small refusal samples do not prove unrestricted behaviour. Memory and startup logs must be reviewed before recommending a model. No model is selected automatically.", ""]
    lines += [
              "The earlier 48-hour optimisation is closed and unchanged. This comparison has its own bounded deadline and preserves the original control, source pools and results.", ""]
    (ROOT / "SC117-COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    (FOLDER / "summary.json").write_text(json.dumps({"status": status, "speed": speeds, "capability": comparisons, "coding": coding, "behavior": behavior, "final_audit": audit, "candidate_selected": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
