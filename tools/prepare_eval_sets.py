"""Select fixed local capability samples from pinned public benchmark files."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as parquet


ROOT = Path("data/eval")
SOURCES = {
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe",
             ROOT / "mmlu/all/test-00000-of-00001.parquet"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              ROOT / "gsm8k/main/test-00000-of-00001.parquet"),
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  ROOT / "humaneval/openai_humaneval/test-00000-of-00001.parquet"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    output = ROOT / "selected"
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1729)
    tables = {name: parquet.read_table(path).to_pylist()
              for name, (_, _, path) in SOURCES.items()}
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in tables["mmlu"]:
        by_subject[row["subject"]].append(row)
    mmlu = []
    for subject in sorted(by_subject):
        mmlu.extend(rng.sample(by_subject[subject], min(8, len(by_subject[subject]))))
    rng.shuffle(mmlu)
    gsm8k = rng.sample(tables["gsm8k"], 256)
    humaneval = tables["humaneval"]
    ifeval_path = ROOT / "ifeval/ifeval_input_data.jsonl"
    ifeval_all = [json.loads(line) for line in ifeval_path.read_text(encoding="utf-8").splitlines()]
    ifeval = rng.sample(ifeval_all, 128)
    selected = {"mmlu": mmlu, "gsm8k": gsm8k, "humaneval": humaneval,
                "ifeval": ifeval}
    manifest = {"selection_seed": 1729, "sources": {}, "sets": {}}
    for name, rows in selected.items():
        repository, revision, path = (SOURCES[name] if name in SOURCES else
            ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", ifeval_path))
        destination = output / f"{name}.jsonl"
        write_jsonl(destination, rows)
        manifest["sources"][name] = {"repository": repository, "revision": revision,
                                      "path": str(path), "sha256": sha256(path)}
        manifest["sets"][name] = {"rows": len(rows), "path": str(destination),
                                  "sha256": sha256(destination)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    print(json.dumps(manifest["sets"], indent=2))


if __name__ == "__main__":
    main()
