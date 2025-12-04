#!/usr/bin/env python
"""Process structured_step/structured_fixed JSONLs for a given size.

Steps:
1) Filter correct entries using eval_results logic.
2) Build overlap subsets (both correct, step-only, fixed-only).
3) Plot step-count (numeric) and bucketed difficulty for the subsets using
   difficulty map from scripts/test_with_step_difficulty.jsonl (problem_index keyed).
Outputs are saved under scripts/comparisons/ts{size}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Add repo root for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_query.eval_results import load_entries, extract_gold, extract_pred, is_correct  # type: ignore

DIFF_PATH = ROOT / "scripts" / "test_with_step_difficulty.jsonl"


def filter_correct(path: Path, training_size: int, strategy: str) -> List[dict]:
    rows: List[dict] = []
    for idx, entry in enumerate(load_entries(path)):
        gold = extract_gold(entry)
        pred = extract_pred(entry)
        if is_correct(gold, pred):
            entry = dict(entry)
            entry.setdefault("problem_index", idx)
            entry["training_size"] = training_size
            entry["strategy"] = strategy
            rows.append(entry)
    return rows


def save_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def build_subsets(step_rows: List[dict], fixed_rows: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    step_map = {r["problem_index"]: r for r in step_rows}
    fixed_map = {r["problem_index"]: r for r in fixed_rows}
    both_idxs = set(step_map) & set(fixed_map)
    step_only_idxs = set(step_map) - set(fixed_map)
    fixed_only_idxs = set(fixed_map) - set(step_map)
    both = [step_map[i] for i in sorted(both_idxs)]
    step_only = [step_map[i] for i in sorted(step_only_idxs)]
    fixed_only = [fixed_map[i] for i in sorted(fixed_only_idxs)]
    return both, step_only, fixed_only


def load_step_difficulty_map(path: Path) -> Dict[int, int]:
    m: Dict[int, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx = rec.get("problem_index")
            d = rec.get("difficulty")
            try:
                m[int(idx)] = int(d)
            except Exception:
                continue
    return m


def bucket(d: int) -> str:
    """Bucket difficulty: Easy 0-3, Medium 4-5, Hard 5+."""
    if d <= 3:
        return "Easy"
    if 4 <= d <= 5:
        return "Medium"
    return "Hard"


def plot_dist(label: str, difficulties: List[int], out_dir: Path) -> None:
    if not difficulties:
        print(f"No mapped difficulties for {label}, skipping plots.")
        return
    num_counts = Counter(difficulties)
    order_num = list(range(min(num_counts), max(num_counts) + 1))
    x_num = [str(i) for i in order_num]
    y_num = [num_counts.get(i, 0) for i in order_num]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x_num, y_num, color="#9ecae1")
    ax.set_xlabel("Difficulty (step count)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: step-count difficulty")
    for rect, val in zip(bars, y_num):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(val), ha="center", va="bottom")
    fig.tight_layout()
    out_path = out_dir / f"{label}_difficulty_numeric.jpg"
    fig.savefig(out_path, format="jpg", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    buckets = [bucket(d) for d in difficulties]
    buck_counts = Counter(buckets)
    order_b = ["Easy", "Medium", "Hard"]
    x_b = order_b
    y_b = [buck_counts.get(k, 0) for k in order_b]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x_b, y_b, color=["#6baed6", "#9ecae1", "#c6dbef"])
    ax.set_xlabel("Difficulty (bucketed)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: bucketed difficulty")
    for rect, val in zip(bars, y_b):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(val), ha="center", va="bottom")
    fig.tight_layout()
    out_path = out_dir / f"{label}_difficulty_bucketed.jpg"
    fig.savefig(out_path, format="jpg", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_venn(label: str, sizes: Tuple[int, int, int], out_dir: Path) -> None:
    """Simple proportional circles for venn-like visualization."""
    only_a, only_b, both = sizes
    total_a = only_a + both
    total_b = only_b + both
    import math

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")
    r_a = math.sqrt(total_a) * 0.05
    r_b = math.sqrt(total_b) * 0.05
    x1, y1 = 0.45, 0.5
    x2, y2 = 0.55, 0.5
    c1 = plt.Circle((x1, y1), r_a, color="skyblue", alpha=0.5, label="Structured Step")
    c2 = plt.Circle((x2, y2), r_b, color="salmon", alpha=0.5, label="Structured Fixed")
    ax.add_artist(c1)
    ax.add_artist(c2)
    plt.text(x1 - r_a / 2, y1, f"Step only\n{only_a}", ha="center", va="center")
    plt.text(x2 + r_b / 2, y2, f"Fixed only\n{only_b}", ha="center", va="center")
    plt.text((x1 + x2) / 2, y1, f"Both\n{both}", ha="center", va="center", fontweight="bold")
    plt.title(f"Correct overlap ({label})")
    out_path = out_dir / f"{label}_venn.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def map_difficulties(rows: List[dict], diff_map: Dict[int, int]) -> List[int]:
    vals: List[int] = []
    for r in rows:
        idx = r.get("problem_index")
        if idx is None:
            continue
        d = diff_map.get(int(idx))
        if d is None:
            continue
        try:
            vals.append(int(d))
        except Exception:
            continue
    return vals


def process(size: int) -> None:
    base = ROOT / "Llama-flatten-label" / "Llama-3.1-8B-Instruct"
    step_input = base / f"structured_step_ts{size}" / f"structured_step_Llama-3.1-8B-Instruct_test_{size}.jsonl"
    fixed_input = base / f"structured_fixed_ts{size}" / f"structured_fixed_Llama-3.1-8B-Instruct_test_{size}.jsonl"
    out_dir = ROOT / "scripts" / "comparisons" / f"ts{size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not step_input.exists() or not fixed_input.exists():
        print(f"Skip ts{size}: missing input file(s)")
        return

    step_correct = filter_correct(step_input, size, "structured_step")
    fixed_correct = filter_correct(fixed_input, size, "structured_fixed")
    save_jsonl(step_correct, out_dir / f"structured_step_ts{size}_correct.jsonl")
    save_jsonl(fixed_correct, out_dir / f"structured_fixed_ts{size}_correct.jsonl")

    both, step_only, fixed_only = build_subsets(step_correct, fixed_correct)
    save_jsonl(both, out_dir / f"ts{size}_both_correct.jsonl")
    save_jsonl(step_only, out_dir / f"ts{size}_step_only_correct.jsonl")
    save_jsonl(fixed_only, out_dir / f"ts{size}_fixed_only_correct.jsonl")

    # Venn
    plot_venn(f"ts{size}", (len(step_only), len(fixed_only), len(both)), out_dir)

    diff_map = load_step_difficulty_map(DIFF_PATH)

    for label, rows in [
        (f"ts{size}_both_correct", both),
        (f"ts{size}_step_only_correct", step_only),
        (f"ts{size}_fixed_only_correct", fixed_only),
    ]:
        diffs = map_difficulties(rows, diff_map)
        plot_dist(label, diffs, out_dir)


def main() -> None:
    for size in (300, 500, 700, 900):
        process(size)


if __name__ == "__main__":
    main()
