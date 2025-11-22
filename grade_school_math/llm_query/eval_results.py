#!/usr/bin/env python3
"""Compute accuracy for GSM8K response JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_entries(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_correct(gold, pred) -> bool:
    if gold is None or pred is None:
        return False
    try:
        g = float(gold)
        p = float(pred)
        return abs(g - p) < 1e-9
    except (ValueError, TypeError):
        return str(gold).strip() == str(pred).strip()


def evaluate_file(path: Path):
    total = 0
    correct = 0
    ignored = 0
    for entry in load_entries(path):
        response_ans = entry.get("response_ans")
        if response_ans is None:
            ignored += 1
            continue
        total += 1
        if is_correct(entry.get("gold"), response_ans):
            correct += 1
    return total, correct, ignored


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy for response JSONL files under gsm8k/structure_rationale."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/structure_rationale"),
        help="Directory containing per-strategy results.jsonl files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    grand_total = 0
    grand_correct = 0
    grand_ignored = 0
    for jsonl_path in sorted(root.rglob("results.jsonl")):
        total, correct, ignored = evaluate_file(jsonl_path)
        grand_total += total
        grand_correct += correct
        grand_ignored += ignored
        
        if total > 0:
            accuracy = correct / total * 100
            print(f"{jsonl_path}: {correct}/{total} correct ({accuracy:.2f}%)")
        else:
            print(f"{jsonl_path}: No valid entries to evaluate.")
        
        if ignored > 0:
            print(f"  ({ignored} entries were ignored due to missing 'response_ans')")


    if grand_total:
        overall = grand_correct / grand_total * 100
        print(f"\nOverall: {grand_correct}/{grand_total} correct ({overall:.2f}%)")
    else:
        print("No results.jsonl files with valid entries were found.")
        
    if grand_ignored > 0:
        print(f"A total of {grand_ignored} entries were ignored across all files.")


if __name__ == "__main__":
    main()
