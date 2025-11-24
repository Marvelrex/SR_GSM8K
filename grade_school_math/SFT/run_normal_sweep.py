#!/usr/bin/env python3
"""Run a sweep of SFT training sizes for normal rationales with checkpoint resume."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint


def has_final_weights(output_dir: Path) -> bool:
    # Detect a completed save (covers sharded safetensors as well)
    if (output_dir / "pytorch_model.bin").exists():
        return True
    if (output_dir / "model.safetensors").exists():
        return True
    if (output_dir / "model.safetensors.index.json").exists():
        return True
    if list(output_dir.glob("model-*-of-*.safetensors")):
        return True
    return False


def run_size(
    size: int,
    base_cmd: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if has_final_weights(output_dir):
        print(f"[size={size}] Final weights found; skipping.")
        return

    resume_arg = None
    last_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if last_ckpt:
        resume_arg = last_ckpt

    cmd = base_cmd + [
        "--max-train-samples",
        str(size),
        "--output-dir",
        str(output_dir),
    ]
    if resume_arg:
        cmd += ["--resume-from-checkpoint", resume_arg]

    print(f"[size={size}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep training sizes for sft_normal.py with checkpoint resume."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[200, 400, 600, 800, 900],
        help="List of training sizes to run.",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("outputs") / "normal_sweep",
        help="Root directory for size-specific outputs.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("filtered_gpt5_rationales") / "filtered_normal.jsonl",
        help="Path to filtered normal JSONL.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Args passed through to sft_normal.py (e.g., --extra-args --num-epochs 1 --batch-size 2).",
    )
    args = parser.parse_args()

    sft_script = Path(__file__).parent / "sft_normal.py"
    if not sft_script.exists():
        raise SystemExit(f"Cannot find sft_normal.py at {sft_script}")

    base_cmd = [
        "python",
        str(sft_script),
        "--train-file",
        str(args.train_file),
    ] + args.extra_args

    for size in args.sizes:
        run_size(size, base_cmd, args.base_output / f"size_{size}")


if __name__ == "__main__":
    main()
