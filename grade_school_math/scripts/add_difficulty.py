#!/usr/bin/env python3
"""
Add a difficulty column to GSM8K-style JSONL files by counting reasoning steps.

Rules for difficulty:
- Each non-empty line in the answer counts as one step.
- Lines beginning with '####' (after optional leading whitespace) are ignored (they are the final answer).

By default, processes data/train.jsonl and data/test.jsonl and writes
data/train_with_difficulty.jsonl and data/test_with_difficulty.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def compute_difficulty(answer: str) -> int:
    steps = 0
    for line in answer.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("####"):
            continue
        steps += 1
    return steps


def process_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as r, dst.open("w", encoding="utf-8") as w:
        for line in r:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["difficulty"] = compute_difficulty(obj.get("answer", ""))
            w.write(json.dumps(obj, ensure_ascii=False))
            w.write("\n")
    print(f"Wrote {dst}")


def default_pairs(root: Path, splits: Iterable[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        src = root / f"{split}.jsonl"
        dst = root / f"{split}_with_difficulty.jsonl"
        pairs.append((src, dst))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add difficulty column to JSONL files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Directory containing input JSONLs (default: data).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Split names to process (default: train test).",
    )
    parser.add_argument(
        "--custom",
        nargs=2,
        metavar=("SRC", "DST"),
        type=Path,
        help="Optional single custom src/dst paths (overrides --splits).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = (
        [(args.custom[0], args.custom[1])]
        if args.custom
        else default_pairs(args.root, args.splits)
    )
    for src, dst in pairs:
        process_file(src, dst)


if __name__ == "__main__":
    main()
