#!/usr/bin/env python
"""Interactive heatmap of per-question correctness for Llama-3.1-8B-Instruct runs.

Features:
- Training size dropdown (ts300/ts500/ts700/ts900, or whatever exists).
- Legend toggles to turn prompting strategies on/off.
- Hover shows status per question/strategy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import plotly.io as pio

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        f"plotly is required with interpreter {sys.executable}. "
        "Install it via `python -m pip install plotly`."
    ) from exc

# Avoid pulling in default templates that may inject unexpected trace defaults.
pio.templates.default = "none"
TARGET_SUBPATH = Path("Llama-flatten-label") / "Llama-3.1-8B-Instruct"

STRATEGY_LABELS: Dict[str, str] = {
    "normal": "Normal",
    "structured_fixed": "Structured Fixed",
    "structured_freeform": "Structured Freeform",
    "structured_noisy": "Structured Noisy",
    "structured_step": "Structured Step",
}


def find_default_base_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        probe = candidate / TARGET_SUBPATH
        if probe.exists():
            return probe
    return here / TARGET_SUBPATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heatmap of correctness by question.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=find_default_base_dir(),
        help="Directory containing the strategy folders (default: auto-detected).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: alongside base dir).",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        help="Optional subset of strategies (e.g., normal structured_step).",
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        help="Optional subset of training sizes (e.g., 300 500 700 900).",
    )
    return parser.parse_args()


def normalize_strategy_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def extract_answer(raw: Any) -> Any:
    if isinstance(raw, dict):
        for key in ("ans", "answer", "final_answer", "prediction", "pred"):
            if key in raw:
                return raw[key]
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        try:
            parsed = json.loads(text)
            return extract_answer(parsed)
        except json.JSONDecodeError:
            return text
    return raw


def normalize_numeric(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip().strip('"\'').replace(",", "")
    text = re.sub(r"[.]+$", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def answers_match(gold: Any, pred: Any) -> bool:
    g = normalize_numeric(gold)
    p = normalize_numeric(pred)
    if g is not None and p is not None:
        tol = max(1e-3, 1e-3 * abs(g))
        return abs(g - p) <= tol
    return str(gold).strip().lower() == str(pred).strip().lower()


def extract_index(idx: Any, fallback: int) -> int:
    if isinstance(idx, int):
        return idx
    if isinstance(idx, str):
        m = re.search(r"(\d+)", idx)
        if m:
            return int(m.group(1))
    return fallback


def load_run(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            rec = json.loads(line)
            gold = rec.get("gold_ans")
            pred = extract_answer(rec.get("model_response"))
            correct = 1 if answers_match(gold, pred) else 0
            entries.append(
                {
                    "problem_index": extract_index(rec.get("index"), line_no),
                    "correct": correct,
                    "gold_ans": gold,
                    "pred_ans": pred,
                }
            )
    return entries


def gather_runs(
    base_dir: Path,
    strategies: Optional[Iterable[str]],
    sizes: Optional[Iterable[int]],
) -> List[Dict[str, Any]]:
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    strategy_filter = {normalize_strategy_name(s) for s in strategies} if strategies else None
    size_filter = set(sizes) if sizes else None

    runs: List[Dict[str, Any]] = []
    for folder in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        if "_ts" not in folder.name:
            continue
        slug, size_txt = folder.name.rsplit("_ts", 1)
        size_val = int(size_txt)
        if strategy_filter and slug not in strategy_filter:
            continue
        if size_filter and size_val not in size_filter:
            continue
        jsonl_files = list(folder.glob("*.jsonl"))
        if not jsonl_files:
            continue
        runs.append(
            {
                "strategy": slug,
                "training_size": size_val,
                "file": jsonl_files[0],
                "entries": load_run(jsonl_files[0]),
            }
        )
    if not runs:
        raise SystemExit(f"No runs found under {base_dir} with given filters.")
    return runs


def build_figure(runs: List[Dict[str, Any]]) -> go.Figure:
    all_indices = sorted({e["problem_index"] for r in runs for e in r["entries"]})
    sizes = sorted({r["training_size"] for r in runs})
    strategies = sorted({r["strategy"] for r in runs})
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    # Precompute z/text per (size, strategy).
    combo: Dict[int, Dict[str, Tuple[List[List[float]], List[List[str]]]]] = {}
    for size in sizes:
        combo[size] = {}
        for strategy in strategies:
            match = next(
                (r for r in runs if r["training_size"] == size and r["strategy"] == strategy), None
            )
            if match:
                index_to_correct = {e["problem_index"]: e["correct"] for e in match["entries"]}
                row_vals: List[Optional[float]] = []
                row_texts: List[str] = []
                for idx in all_indices:
                    val = index_to_correct.get(idx)
                    safe_val = 0.5 if val is None else val
                    row_vals.append(safe_val)
                    if val is None:
                        row_texts.append(
                            f"{STRATEGY_LABELS.get(strategy, strategy)} | ts{size}<br>Problem {idx}<br>No data"
                        )
                    else:
                        status = "Correct" if val == 1 else "Incorrect"
                        row_texts.append(
                            f"{STRATEGY_LABELS.get(strategy, strategy)} | ts{size}<br>Problem {idx}<br>Status: {status}"
                        )
                combo[size][strategy] = ([[*row_vals]], [[*row_texts]])
            else:
                empty_row = [0.5] * len(all_indices)
                empty_text = [f"{STRATEGY_LABELS.get(strategy, strategy)} | ts{size}<br>No data"] * len(
                    all_indices
                )
                combo[size][strategy] = ([empty_row], [empty_text])

    default_size = sizes[0]
    colorscale = [[0, "rgb(200,50,50)"], [0.5, "rgb(240,240,240)"], [1, "rgb(50,160,80)"]]

    # Build initial single-trace heatmap (all strategies stacked).
    z_default = [combo[default_size][s][0][0] for s in strategies]
    text_default = [combo[default_size][s][1][0] for s in strategies]

    heatmap = go.Heatmap(
        z=z_default,
        x=all_indices,
        y=labels,
        colorscale=colorscale,
        colorbar=dict(title="Correctness", tickvals=[0, 1], ticktext=["Incorrect", "Correct"]),
        showscale=True,
        hoverinfo="text",
        text=text_default,
        zmin=0,
        zmax=1,
        name="Correctness",
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=f"Correctness heatmap | ts{default_size}",
        xaxis_title="Problem index",
        yaxis_title="Prompting strategy",
        margin=dict(l=120, r=40, t=60, b=60),
        yaxis=dict(autorange="reversed"),
        template="none",
        height=800,
        updatemenus=[],
    )

    # Attach metadata for the custom JS controls.
    fig._heatmap_metadata = {
        "sizes": sizes,
        "strategies": strategies,
        "labels": labels,
        "indices": all_indices,
        "colorscale": colorscale,
        "combo": {
            size: {
                strategy: {"z": combo[size][strategy][0][0], "text": combo[size][strategy][1][0]}
                for strategy in strategies
            }
            for size in sizes
        },
    }
    return fig


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    runs = gather_runs(base_dir, args.strategies, args.sizes)
    runs.sort(key=lambda r: (r["training_size"], STRATEGY_LABELS.get(r["strategy"], r["strategy"])))

    fig = build_figure(runs)
    output_path = args.output or base_dir.parent / "llama_3.1_8b_heatmap.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = getattr(fig, "_heatmap_metadata")
    payload_js = json.dumps(meta)
    layout_js = json.dumps(fig.to_plotly_json()["layout"])
    config_js = json.dumps({"responsive": False})
    # Custom JS adds size dropdown + strategy checkboxes that restyle the single heatmap trace.
    post_script = f"""
(() => {{
  const gd = document.getElementById('heatmap');
  const payload = {payload_js};
  const baseLayout = {layout_js};
  const config = {config_js};
  const container = document.createElement('div');
  container.style.margin = '8px 0';
  const sizeLabel = document.createElement('label');
  sizeLabel.textContent = 'Training size: ';
  const sizeSelect = document.createElement('select');
  payload.sizes.forEach(sz => {{
    const opt = document.createElement('option');
    opt.value = sz;
    opt.textContent = 'ts' + sz;
    sizeSelect.appendChild(opt);
  }});
  const stratLabel = document.createElement('span');
  stratLabel.textContent = '  Strategies: ';
  const checkboxWrap = document.createElement('span');
  payload.strategies.forEach((s, idx) => {{
    const id = 'chk-' + s;
    const label = document.createElement('label');
    label.style.marginRight = '8px';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.value = s;
    cb.checked = true;
    label.appendChild(cb);
    label.appendChild(document.createTextNode(' ' + payload.labels[idx]));
    checkboxWrap.appendChild(label);
  }});
  container.appendChild(sizeLabel);
  container.appendChild(sizeSelect);
  container.appendChild(stratLabel);
  container.appendChild(checkboxWrap);
  gd.parentNode.insertBefore(container, gd);

  function currentSelection() {{
    const size = parseInt(sizeSelect.value, 10);
    const active = Array.from(checkboxWrap.querySelectorAll('input[type=checkbox]'))
      .filter(cb => cb.checked)
      .map(cb => cb.value);
    const labels = payload.labels.filter((_, i) => active.includes(payload.strategies[i]));
    const z = active.map(s => payload.combo[size][s].z);
    const text = active.map(s => payload.combo[size][s].text);
    return {{ size, labels, z, text }};
  }}

  function render() {{
    const sel = currentSelection();
    if (sel.labels.length === 0) return;
    const layout = JSON.parse(JSON.stringify(baseLayout));
    layout.title = 'Correctness heatmap | ts' + sel.size;
    layout.yaxis = layout.yaxis || {{}};
    layout.yaxis.autorange = 'reversed';
    const trace = {{
      type: 'heatmap',
      x: payload.indices,
      y: sel.labels,
      z: sel.z,
      text: sel.text,
      hoverinfo: 'text',
      colorscale: payload.colorscale,
      zmin: 0,
      zmax: 1,
      colorbar: {{
        title: 'Correctness',
        tickvals: [0, 1],
        ticktext: ['Incorrect', 'Correct']
      }}
    }};
    Plotly.react(gd, [trace], layout, config);
  }}

  sizeSelect.addEventListener('change', render);
  checkboxWrap.addEventListener('change', render);
  render();
}})();
"""
    # Give the output a fixed height so Plotly's responsive logic doesn't inherit a
    # zero-height parent and render nothing.
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        default_width="100%",
        default_height="800px",
        config={"responsive": False},
        div_id="heatmap",
        post_script=post_script,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
