#!/usr/bin/env python3
"""Run GSM8K batches for multiple prompt strategies and store outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

print("Executing run_batch_queries.py...")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_SCRIPT_GPT = PROJECT_ROOT / "llm_query" / "query_gpt.py"
QUERY_SCRIPT_GROK = PROJECT_ROOT / "llm_query" / "query_grok.py"
QUERY_SCRIPT_GEMMA = PROJECT_ROOT / "llm_query" / "query_gemma.py"
QUERY_SCRIPT_LLAMA = PROJECT_ROOT / "llm_query" / "query_llama.py"

DEFAULT_STRATEGIES = ["Normal", "noisy", "Step", "Fixed", "freeform"]

STRATEGY_NORMALIZATION = {
    "normal": "Normal",
    "noisy": "Noisy",
    "step": "Step",
    "fixed": "Fixed",
    "freeform": "Free-Form",
}


def normalize_strategy(name: str) -> str:
    key = name.strip().lower()
    if key in STRATEGY_NORMALIZATION:
        return STRATEGY_NORMALIZATION[key]
    raise ValueError(
        f"Unknown strategy '{name}'. Supported: {', '.join(sorted(set(STRATEGY_NORMALIZATION.values())))}"
    )


def build_command(
    strategy: str,
    dataset_path: Path,
    sample_index: int,
    num_samples: int,
    model: str,
    output_dir: Path,
    show_only: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(select_query_script(model)),
        "--dataset-path",
        str(dataset_path),
        "--sample-index",
        str(sample_index),
        "--num-samples",
        str(num_samples),
        "--strategy",
        strategy,
        "--model",
        model,
        "--output-dir",
        str(output_dir),
    ]
    if show_only:
        cmd.append("--show-only")
    return cmd


def select_query_script(model: str) -> Path:
    """Pick the query script based on the model name."""
    model_lower = (model or "").lower()
    if "grok" in model_lower or "xai" in model_lower:
        target = QUERY_SCRIPT_GROK
    elif "gemma" in model_lower:
        target = QUERY_SCRIPT_GEMMA
    elif "llama" in model_lower:
        target = QUERY_SCRIPT_LLAMA
    else:
        target = QUERY_SCRIPT_GPT

    if not target.exists():
        raise SystemExit(f"Could not find query script at {target}")
    return target


def run_batches(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    strategies = [normalize_strategy(raw) for raw in args.strategies]
    max_strategy_retries = 3

    for idx, strategy in enumerate(strategies, start=1):
        strategy_dir_name = strategy.lower().replace(" ", "_").replace("-", "_")
        output_dir = output_root / strategy_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            strategy=strategy,
            dataset_path=dataset_path,
            sample_index=args.sample_index,
            num_samples=args.num_samples,
            model=args.model,
            output_dir=output_dir,
            show_only=args.show_only,
        )

        max_attempts = max_strategy_retries
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            print(
                f"[{idx}/{len(strategies)}] Running strategy '{strategy}' -> {output_dir} "
                f"(attempt {attempt}/{max_attempts})",
                flush=True,
            )
            try:
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
                break
            except subprocess.CalledProcessError as exc:
                if attempt >= max_attempts:
                    print(
                        f"Strategy '{strategy}' failed after {max_attempts} attempts. "
                        "Aborting remaining runs."
                    )
                    raise
                wait_seconds = min(5 * attempt, 30)
                print(
                    f"Strategy '{strategy}' failed with exit code {exc.returncode}; "
                    f"retrying in {wait_seconds} seconds...",
                    flush=True,
                )
                time.sleep(wait_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GSM8K batches across multiple prompt strategies."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "train.jsonl",
        help="Path to the GSM8K jsonl data (default: data/train.jsonl).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based starting index inside the dataset (default: 0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to process per strategy (default: 2000).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to query (e.g., gpt-4o-mini, grok-3-mini, google/gemma-2b-it, meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="List of prompt strategies to run (default: Normal Noise Step Fixed Free-Form).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "structure_rationale",
        help="Directory where per-strategy folders will be created (default: data/structure_rationale).",
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Pass --show-only through to the chosen query script.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batches(args)


if __name__ == "__main__":
    main()
