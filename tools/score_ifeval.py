"""Score saved local IFEval generations with a pinned official evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

OFFICIAL = Path("data/eval/ifeval/official")
sys.path.insert(0, str(OFFICIAL.resolve()))
os.environ["NLTK_DATA"] = str(Path("data/eval/ifeval/nltk_data").resolve())

from instruction_following_eval import evaluation_lib  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path('data/eval/selected/ifeval.jsonl'))
    args = parser.parse_args()
    inputs = [evaluation_lib.InputExample(**json.loads(line)) for line in
              args.inputs.read_text(encoding="utf-8").splitlines()]
    answers = {row["prompt"]: row["response"] for row in
               (json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines())}
    missing = [item.prompt for item in inputs if item.prompt not in answers]
    if missing:
        raise ValueError(f"Missing {len(missing)} IFEval responses")
    report = {"evaluator_revision": "d36068b845da4c2b24927fee2cea1e6ef98dadda",
              "sample_count": len(inputs), "modes": {}}
    example_rows = []
    for mode, scorer in (("strict", evaluation_lib.test_instruction_following_strict),
                         ("loose", evaluation_lib.test_instruction_following_loose)):
        outputs = [scorer(item, answers) for item in inputs]
        for index, item in enumerate(outputs):
            if mode == "strict":
                example_rows.append({"index": index, "key": inputs[index].key,
                                     "strict": bool(item.follow_all_instructions)})
            else:
                example_rows[index]["loose"] = bool(item.follow_all_instructions)
        prompts = sum(item.follow_all_instructions for item in outputs)
        instructions = sum(sum(item.follow_instruction_list) for item in outputs)
        total = sum(len(item.follow_instruction_list) for item in outputs)
        report["modes"][mode] = {
            "prompt_accuracy": prompts / len(outputs), "prompts_correct": prompts,
            "instruction_accuracy": instructions / total,
            "instructions_correct": instructions, "instructions_total": total,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    examples = args.output.with_suffix(".examples.jsonl")
    examples.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in example_rows),
                        encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
