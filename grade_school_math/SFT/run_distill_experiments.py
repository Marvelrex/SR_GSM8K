#!/usr/bin/env python3
"""
Batch launcher for distill_rationale.py with multiple train sizes.

Defaults target structured step rationales for train sizes
[900, 700, 500, 300], writing checkpoints into distilled-structured-ts<train_size>.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SCRIPT = SCRIPTS_DIR / "distill_rationale.py"

DEFAULT_TRAIN_SIZES = [900, 700, 500, 300]
INTERSECTION_IDS = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_intersection_ids.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple distillation jobs sequentially.")
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help=f"Train sizes to iterate over (default: {DEFAULT_TRAIN_SIZES}).",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "structured"],
        default="structured",
        help="Rationale mode to target (default: structured).",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="JSONL dataset to train on (default: based on mode/strategy).",
    )
    parser.add_argument(
        "--strategy",
        choices=["step", "fixed", "freeform", "noisy"],
        default="step",
        help="Structured strategy to use when --mode=structured (default: step).",
    )
    parser.add_argument(
        "--intersection-file",
        type=Path,
        default=INTERSECTION_IDS,
        help="Optional list of IDs to keep (structured only).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After each run, generate on the test file (first N rows).",
    )
    parser.add_argument(
        "--max-gen-samples",
        type=int,
        default=300,
        help="Number of test samples to generate when --generate is set (default: 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate when --generate is set (default: 256).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Override test JSONL path for generation (default: distill_rationale.py default).",
    )
    parser.add_argument(
        "--gen-output-file",
        type=Path,
        default=None,
        help="Override output predictions path (default: <output_dir>/predictions.jsonl).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps to pass to distill_rationale.py (default: 4).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name/id to pass through (default: meta-llama/Llama-3.2-1B-Instruct).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Checkpoint save frequency (steps) passed to distill_rationale.py (default: 250).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Logging frequency (steps) passed to distill_rationale.py (default: 25).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/jyang001/scratch"),
        help="Base directory to store model checkpoints (default: /home/jyang001/scratch).",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("/home/jyang001/jyang001/projects/SR_GSM8K/grade_school_math"),
        help="Base directory to store prediction JSONL files (default: project root).",
    )
    parser.add_argument(
        "--rationale-weight",
        type=float,
        default=0.5,
        help="Relative loss weight for rationale tokens vs answer tokens (passed to distill_rationale.py).",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=4.0,
        help="Number of epochs to pass through to distill_rationale.py (default: 4).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Limit the number of training examples (passed through).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size to pass through (default: 2).",
    )
    return parser.parse_args()


def default_data_file(mode: str, strategy: str) -> Path:
    if mode == "normal":
        return REPO_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl"
    strat_dir = {
        "step": "step",
        "fixed": "fixed",
        "freeform": "freeform",
        "noisy": "noisy",
    }[strategy]
    return REPO_ROOT / "filtered_gpt5_rationales" / f"filtered_{strat_dir}.jsonl"


def build_base_cmd(
    mode: str,
    data_file: Path,
    strategy: str,
    intersection_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--bf16",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--num-epochs",
        str(args.num_epochs),
        "--mode",
        mode,
        "--data-file",
        str(data_file),
        "--model-name",
        args.model_name,
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
    ]
    if mode == "structured":
        cmd += ["--strategy", strategy]
        if intersection_file:
            cmd += ["--intersection-file", str(intersection_file)]
    if args.generate:
        cmd += ["--generate", "--max-gen-samples", str(args.max_gen_samples), "--max-new-tokens", str(args.max_new_tokens)]
        if args.test_file:
            cmd += ["--test-file", str(args.test_file)]
        if args.gen_output_file:
            cmd += ["--gen-output-file", str(args.gen_output_file)]
    cmd += ["--rationale-weight", str(args.rationale_weight)]
    return cmd


def run_experiment(train_size: int, base_cmd: list[str], mode: str, strategy: str, args: argparse.Namespace) -> None:
    suffix = "structured" if mode == "structured" else "normal"
    strat_suffix = f"_{strategy}" if mode == "structured" else ""
    model_name_arg = Path(base_cmd[base_cmd.index("--model-name") + 1]).name
    output_dir = args.output_root / f"{model_name_arg}" / f"{suffix}{strat_suffix}_ts{train_size}"
    # If a custom gen-output-file is not provided, we set one per run under pred_root
    run_pred_dir = args.pred_root / f"{model_name_arg}" / f"{suffix}{strat_suffix}_ts{train_size}"
    run_pred_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = (args.test_file or Path("test")).stem
    default_pred = run_pred_dir / f"{suffix}{strat_suffix}_{model_name_arg}_{dataset_label}_{train_size}.jsonl"

    cmd = list(base_cmd)
    if args.generate and not args.gen_output_file:
        cmd += ["--gen-output-file", str(default_pred)]
    cmd += [
        "--train-size",
        str(train_size),
        "--output-dir",
        str(output_dir),
    ]
    print(f"\n=== Running {mode} train_size={train_size} -> {output_dir} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    data_path = args.data_file or default_data_file(args.mode, args.strategy)
    base_cmd = build_base_cmd(args.mode, data_path, args.strategy, args.intersection_file, args)
    sizes = [args.train_size] if args.train_size is not None else args.train_sizes
    for size in sizes:
        run_experiment(size, base_cmd, args.mode, args.strategy, args)


if __name__ == "__main__":
    main()
