"""Make deterministic, disjoint local calibration and validation excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CORPUS_REVISION = "8178d96379feb4c484ad5de8cb3ae61101ae8dfb"


def write_excerpts(text: str, begin: int, end: int, count: int, chars: int, target: Path) -> list[int]:
    span = end - begin
    if span < count * chars:
        raise ValueError("Not enough source text for disjoint excerpts")
    starts = []
    excerpts = []
    for index in range(count):
        position = begin + (index * span // count)
        position = min(position, end - chars)
        starts.append(position)
        excerpt = text[position:position + chars]
        # llama-imatrix reads a C string; embedded NULs in source-code samples
        # would silently truncate the entire calibration run.
        excerpts.append("".join(
            character if character in "\t\n\r" or ord(character) >= 32 else " "
            for character in excerpt
        ))
    target.write_text("\n\n".join(excerpts) + "\n", encoding="utf-8")
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path(
        "data/calibration/builds/qwen3.8-flash-next-moe/calib_train.txt"
    ))
    parser.add_argument("--output", type=Path, default=Path("data/calibration/selected"))
    args = parser.parse_args()
    raw = args.corpus.read_bytes()
    corpus = raw.decode("utf-8")
    split = len(corpus) * 9 // 10
    args.output.mkdir(parents=True, exist_ok=True)
    train = args.output / "train-128x512-approx.txt"
    holdout = args.output / "heldout-64x512-approx.txt"
    train_starts = write_excerpts(corpus, 0, split, 128, 2800, train)
    heldout_starts = write_excerpts(corpus, split, len(corpus), 64, 2800, holdout)
    manifest = {
        "source": "AtomicChat/calib-corpora/builds/qwen3.8-flash-next-moe/calib_train.txt",
        "revision": CORPUS_REVISION,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "selection": "evenly spaced disjoint 2800-character windows; last 10% held out; C0 controls replaced with spaces",
        "train_starts": train_starts,
        "heldout_starts": heldout_starts,
        "train_file": str(train),
        "heldout_file": str(holdout),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {train} and {holdout}")


if __name__ == "__main__":
    main()
