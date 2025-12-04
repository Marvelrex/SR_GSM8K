#!/usr/bin/env python3
"""
Plot difficulty distributions (numeric and bucketed) for JSONL files that already
contain a `difficulty` field (e.g., data/train_with_difficulty.jsonl).

Buckets (matching process_ts.py):
- Easy: difficulty <= 3
- Medium: 4 <= difficulty <= 5
- Hard: difficulty > 5

Outputs are written to scripts/comparisons/difficulty_overview/.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

BUCKETS = [
    ("Easy", lambda d: d <= 3),
    ("Medium", lambda d: 4 <= d <= 5),
    ("Hard", lambda d: d > 5),
]


def load_difficulties(path: Path) -> List[int]:
    vals: List[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            d = obj.get("difficulty")
            if d is None:
                continue
            try:
                vals.append(int(d))
            except Exception:
                continue
    return vals


def plot_numeric(label: str, vals: List[int], out_dir: Path) -> None:
    if not vals:
        print(f"No values for {label}, skipping numeric plot")
        return
    counts = Counter(vals)
    order = list(range(min(counts), max(counts) + 1))
    x = [str(i) for i in order]
    y = [counts.get(i, 0) for i in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, y, color="#9ecae1")
    ax.set_xlabel("Difficulty (step count)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: numeric difficulty")
    for rect, val in zip(bars, y):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(val), ha="center", va="bottom")
    fig.tight_layout()
    out = out_dir / f"{label}_difficulty_numeric.jpg"
    fig.savefig(out, dpi=150, format="jpg")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_bucketed(label: str, vals: List[int], out_dir: Path) -> None:
    if not vals:
        print(f"No values for {label}, skipping bucketed plot")
        return
    buckets: List[str] = []
    for v in vals:
        for name, pred in BUCKETS:
            if pred(v):
                buckets.append(name)
                break
    counts = Counter(buckets)
    order = ["Easy", "Medium", "Hard"]
    x = order
    y = [counts.get(k, 0) for k in order]
    colors = ["#6baed6", "#9ecae1", "#c6dbef"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, y, color=colors)
    ax.set_xlabel("Difficulty (bucketed)")
    ax.set_ylabel("Count")
    ax.set_title("Bucketed difficulty (Easy 0-3, Medium 4-5, Hard 5+)")
    for rect, val in zip(bars, y):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(val), ha="center", va="bottom")
    fig.tight_layout()
    out = out_dir / f"{label}_difficulty_bucketed.jpg"
    fig.savefig(out, dpi=150, format="jpg")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_files(paths: Iterable[Path], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        label = path.stem.replace("_with_difficulty", "")
        vals = load_difficulties(path)
        plot_numeric(label, vals, out_root)
        plot_bucketed(label, vals, out_root)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Plot difficulty distributions for train/test with difficulty.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root containing data/ (defaults to repo root inferred from this file).",
    )
    return parser.parse_args()


def resolve_root(cli_root: Optional[Path]) -> Path:
    if cli_root:
        return cli_root.expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    repo_root = resolve_root(args.root)
    data_dir = repo_root / "data"
    sources = [
        data_dir / "train_with_difficulty.jsonl",
        data_dir / "test_with_difficulty.jsonl",
    ]
    out_root = repo_root / "scripts" / "comparisons" / "difficulty_overview"
    plot_files(sources, out_root)


if __name__ == "__main__":
    main()
