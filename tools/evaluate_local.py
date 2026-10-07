"""Resume fixed local benchmark samples against a running llama-server.

This records HumanEval completions but never executes generated code.  MMLU
and GSM8K are scored automatically.  All requests stay on localhost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def chat(url: str, prompt: str, max_tokens: int, grammar: str | None = None) -> str:
    payload = {
        "model": "local", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "seed": 1729,
        "max_tokens": max_tokens, "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if grammar is not None:
        payload['grammar'] = grammar
    response = post(url + "/v1/chat/completions", payload)
    message = response["choices"][0]["message"]
    return message.get("content") or ""


def completion(url: str, prompt: str) -> str:
    response = post(url + "/completion", {
        "prompt": prompt, "n_predict": 256, "temperature": 0,
        "seed": 1729, "stream": False,
    })
    return response["content"]


def parse_letter(text: str) -> str | None:
    match = re.search(r"(?:^|\s|\()([ABCD])(?:\b|\))", text.strip().upper())
    return match.group(1) if match else None


def parse_number(text: str) -> Decimal | None:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def wilson(correct: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not count:
        return 0.0, 1.0
    p = correct / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return center - radius, center + radius


def score_row(set_name: str, row: dict, text: str) -> tuple[bool | None, str | None]:
    if set_name == "mmlu":
        letter = parse_letter(text)
        return letter == "ABCD"[row["answer"]], letter
    if set_name == "gsm8k":
        answer = parse_number(row["answer"])
        prediction = parse_number(text)
        return prediction == answer, str(prediction) if prediction is not None else None
    return None, None


def prompt_for(set_name: str, row: dict) -> str:
    if set_name == "mmlu":
        choices = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", row["choices"]))
        return f"Answer the following multiple-choice question. Reply with the single correct letter A, B, C, or D.\n\n{row['question']}\n{choices}\nAnswer:"
    if set_name == "gsm8k":
        return f"Solve this math problem. End your answer with a line in the form #### <number>.\n\n{row['question']}"
    return row["prompt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("mmlu", "gsm8k", "humaneval", "ifeval"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--selected", type=Path, default=Path('data/eval/selected'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.selected / f"{args.set}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            expected = 'constrained-choice-v1' if args.set == 'mmlu' else 'chat-512-v1'
            if item.get('protocol') != expected:
                raise ValueError('Saved results use another evaluation protocol; choose a new output path')
            if args.selected != Path('data/eval/selected'):
                row_digest = hashlib.sha256(json.dumps(rows[item['index']],sort_keys=True).encode()).hexdigest()
                if item.get('row_sha256') != row_digest:
                    raise ValueError('Saved development result does not match this dataset row')
            done[item["index"]] = item
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            if index in done:
                continue
            started = time.monotonic()
            text = (completion(args.url, prompt_for(args.set, row)) if args.set == "humaneval"
                    else chat(args.url, prompt_for(args.set, row),
                              1 if args.set == "mmlu" else 512,
                              'root ::= [ABCD]' if args.set == 'mmlu' else None))
            passed, parsed = score_row(args.set, row, text)
            item = {"index": index, "set": args.set, "response": text,
                    "parsed": parsed, "correct": passed,
                    "seconds": round(time.monotonic() - started, 3)}
            item['protocol'] = 'constrained-choice-v1' if args.set == 'mmlu' else 'chat-512-v1'
            item['row_sha256'] = hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            if args.set == "ifeval":
                item["prompt"] = row["prompt"]
                if "key" in row:
                    item["key"] = row["key"]
                elif args.selected != Path('data/eval/selected') and 'review_expectation' in row and 'category' in row:
                    # Authored exploratory review uses the same chat endpoint
                    # but is never scored as IFEval. Preserve its frozen rows.
                    item["key"] = f"local-behavior-{row['id']}"
                else:
                    raise ValueError('Instruction benchmark row is missing its key')
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            done[index] = item
            if (index + 1) % 16 == 0 or index + 1 == len(rows):
                evaluated = [x for x in done.values() if x["correct"] is not None]
                correct = sum(x["correct"] for x in evaluated)
                print(f"{args.set}: {len(done)}/{len(rows)}; {correct}/{len(evaluated)} correct", flush=True)
    evaluated = [x for x in done.values() if x["correct"] is not None]
    correct = sum(x["correct"] for x in evaluated)
    print(json.dumps({"set": args.set, "answered": len(done), "scored": len(evaluated),
                      "correct": correct, "accuracy": correct / len(evaluated) if evaluated else None,
                      "wilson_95": wilson(correct, len(evaluated)) if evaluated else None}, indent=2))


if __name__ == "__main__":
    main()
