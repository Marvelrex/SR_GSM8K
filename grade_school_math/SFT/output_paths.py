from pathlib import Path
import re


def slugify(text: str) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def build_paths(model_name: str, mode: str, strategy: str, train_size: int, output_root: Path, pred_root: Path, test_file: Path | None) -> tuple[Path, Path]:
    model_slug = slugify(model_name)
    strategy_label = strategy if mode == "structured" else "normal"
    output_dir = output_root / f"{model_slug}" / f"{strategy_label}_ts{train_size}"
    dataset_label = (test_file or Path("test")).stem
    pred_file = pred_root / f"{model_slug}" / f"{strategy_label}_ts{train_size}" / f"{strategy_label}_{model_slug}_{dataset_label}_{train_size}.jsonl"
    return output_dir, pred_file
