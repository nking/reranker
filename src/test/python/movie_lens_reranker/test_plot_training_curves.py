import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _get_project_dir() -> str:
    cwd = os.getcwd()
    head = cwd
    proj_dir = ""
    while head and head != os.sep:
        head, tail = os.path.split(head)
        if tail:  # Add only if not an empty string (e.g., from root or multiple separators)
            if tail == "reranker":
                proj_dir = os.path.join(head, tail)
                break
    return proj_dir
proj_dir = _get_project_dir()
for path in ("src/main/python/movie_lens_reranker", "src/test/python/movie_lens_reranker"):
    target_path = os.path.join(proj_dir, path)
    target_path = str(Path(target_path).resolve())
    if target_path not in sys.path:
        sys.path.append(target_path)

from helper import get_output_working_dir

def exponential_smoothing(
    scalars: Sequence[float], weight: float = 0.6
) -> List[float]:
    """
    Applies exponential smoothing matching `test_plot_training_curves.py`:
    EMA_t = (1 - weight) * EMA_{t-1} + weight * x_t
    """
    smoothed: List[float] = []
    if not scalars:
        return smoothed
    last = float(scalars[0])
    smoothed.append(last)
    for current in scalars[1:]:
        smoothed_val = (1.0 - weight) * last + weight * float(current)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def calc_random_softmax_loss(num_candidates: int) -> float:
    return float(np.log(num_candidates))


def calc_random_focal_loss(
    num_candidates: int, focal_gamma: float = 2.0
) -> float:
    p_rand = 1.0 / float(num_candidates)
    return float(((1.0 - p_rand) ** focal_gamma) * np.log(num_candidates))


def calc_random_ndcg(k: int, pool_size: int) -> float:
    k_eff = min(k, pool_size)
    s = np.sum([1.0 / np.log2(r + 1) for r in range(1, k_eff + 1)])
    return float(s / pool_size)


def calc_random_recall(k: int, pool_size: int) -> float:
    return float(min(k, pool_size) / pool_size)


def calc_random_precision(k: int, pool_size: int) -> float:
    return float(1.0 / pool_size)


def calc_random_mrr(k: int, pool_size: int) -> float:
    k_eff = min(k, pool_size)
    s = np.sum([1.0 / r for r in range(1, k_eff + 1)])
    return float(s / pool_size)


def load_hyperparams_dict(file_path: str) -> Dict[str, Any]:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def list_reranker_metrics(log_history: List[Dict[str, Any]]) -> List[str]:
    """Discovers normalized metric names from Hugging Face Trainer `log_history`."""
    ignored_exact = {
        "epoch",
        "step",
        "learning_rate",
        "grad_norm",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
        "total_flos",
        "lora_update_norm_early",
        "lora_update_norm_mid",
        "lora_effective_rank_ratio",
        "projector_weight_norm",
    }
    seen = set()
    ordered: List[str] = []
    for entry in log_history:
        for key in entry.keys():
            if (
                key in ignored_exact
                or "runtime" in key
                or "per_second" in key
                or key.startswith("final_")
            ):
                continue
            norm = re.sub(r"^(train_|eval_|val_|test_)", "", key)
            if norm not in seen:
                seen.add(norm)
                ordered.append(norm)
    return ordered


def read_reranker_metric_series(
    log_history: List[Dict[str, Any]],
    test_metrics_dict: Optional[Dict[str, Any]],
    scalar_name: str,
) -> Tuple[
    List[int],
    List[float],
    List[int],
    List[float],
    Optional[List[int]],
    Optional[List[float]],
]:
    """
    Extracts aligned step indices and values for train, validation, and test
    from `best_hpo_metrics_history.json` (or `train_metrics_history.json`) and
    `test_metrics_history.json`.
    """
    train_key = (
        scalar_name
        if scalar_name
        in ("loss", "loss_listwise_raw", "loss_sft_raw", "loss_distill_raw")
        else f"train_{scalar_name}"
    )
    eval_key = f"eval_{scalar_name}"

    train_by_step: Dict[int, float] = {}
    eval_by_step: Dict[int, float] = {}

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        step = int(step)
        if train_key in entry and entry[train_key] is not None:
            train_by_step[step] = float(entry[train_key])
        if eval_key in entry and entry[eval_key] is not None:
            # Keep periodic in-training eval at each step (skip duplicate final evaluate())
            if step not in eval_by_step:
                eval_by_step[step] = float(entry[eval_key])

    all_steps = sorted(set(train_by_step.keys()) | set(eval_by_step.keys()))
    step_to_idx = {s: idx for idx, s in enumerate(all_steps)}

    train_steps = [step_to_idx[s] for s in all_steps if s in train_by_step]
    train_values = [train_by_step[s] for s in all_steps if s in train_by_step]

    val_steps = [step_to_idx[s] for s in all_steps if s in eval_by_step]
    val_values = [eval_by_step[s] for s in all_steps if s in eval_by_step]

    test_steps: Optional[List[int]] = None
    test_values: Optional[List[float]] = None
    if test_metrics_dict and scalar_name in test_metrics_dict:
        val = test_metrics_dict[scalar_name]
        if val is not None:
            test_steps = [len(all_steps)]
            test_values = [float(val)]

    return (
        train_steps,
        train_values,
        val_steps,
        val_values,
        test_steps,
        test_values,
    )


def generate_reranker_chart(
    log_history: List[Dict[str, Any]],
    test_metrics_dict: Optional[Dict[str, Any]],
    irred_err_dict: Optional[Dict[str, Any]],
    random_metric: Optional[float],
    scalar_name: str,
    output_path: str,
    smoothing_weight: float = 0.6,
    display_in_notebook: bool = True,
) -> Optional[str]:
    """
    Generates a PNG chart and companion JSON summary matching `generate_tensorboard_chart`
    from `test_plot_training_curves.py`, rendered with Plotly + Kaleido.
    """
    (
        train_steps,
        train_values,
        val_steps,
        val_values,
        test_steps,
        test_values,
    ) = read_reranker_metric_series(
        log_history=log_history,
        test_metrics_dict=test_metrics_dict,
        scalar_name=scalar_name,
    )

    if not train_values and not val_values and not test_values:
        print(f"Could not load data for '{scalar_name}'. Skipping.")
        return None

    is_loss = "loss" in scalar_name

    train_smoothed = exponential_smoothing(train_values, smoothing_weight)
    val_smoothed = exponential_smoothing(val_values, smoothing_weight)
    test_smoothed = (
        exponential_smoothing(test_values, smoothing_weight)
        if test_values
        else []
    )

    # --- Dynamic Bounds Logic ---
    bound_val = None
    bound_moe = None
    bound_label = "irred error" if is_loss else "ceiling"

    if irred_err_dict is not None:
        bound_val = (
            irred_err_dict.get("irred_error")
            if is_loss
            else irred_err_dict.get("ceiling")
        )
        bound_moe = irred_err_dict.get("margin_of_error_on_irred_err")

    # High-contrast color scheme matching ranker plots
    train_color = "#1f77b4"
    train_color_faded = "rgba(174, 199, 232, 0.75)"
    val_color = "#ff7f0e"
    val_color_faded = "rgba(255, 187, 120, 0.75)"
    test_color = "#d62728"
    bound_color = "#2ca02c"
    random_color = "#000000"

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.54, 0.46],
        vertical_spacing=0.09,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    # Raw (faded) and Smoothed (solid) Train Curves
    if train_steps:
        fig.add_trace(
            go.Scatter(
                x=train_steps,
                y=train_values,
                mode="lines",
                line=dict(color=train_color_faded, width=1.6),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=train_steps,
                y=train_smoothed,
                mode="lines",
                line=dict(color=train_color, width=3.2),
                name="train",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Raw (faded) and Smoothed (solid) Validation Curves
    if val_steps:
        fig.add_trace(
            go.Scatter(
                x=val_steps,
                y=val_values,
                mode="lines",
                line=dict(color=val_color_faded, width=1.6),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=val_steps,
                y=val_smoothed,
                mode="lines",
                line=dict(color=val_color, width=3.2),
                name="validation",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Test Marker
    latest_test_step = test_steps[-1] if test_steps else None
    latest_test_value = test_values[-1] if test_values else None
    latest_test_smoothed = test_smoothed[-1] if test_smoothed else None

    if latest_test_step is not None and latest_test_value is not None:
        fig.add_trace(
            go.Scatter(
                x=[latest_test_step],
                y=[latest_test_value],
                mode="markers",
                marker=dict(
                    color=test_color,
                    size=10,
                    line=dict(color="white", width=1.0),
                ),
                name="test",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    all_x = (
        train_steps
        + val_steps
        + ([latest_test_step] if latest_test_step is not None else [])
    )
    max_x = max(all_x) if all_x else 1
    x_min, x_max = -0.6, max_x + 0.7

    # Horizontal Green Dashed Line for Ceiling / Irreducible Error
    if bound_val is not None:
        fig.add_shape(
            type="line",
            x0=x_min,
            x1=x_max,
            y0=bound_val,
            y1=bound_val,
            line=dict(color=bound_color, width=2.0, dash="dash"),
            row=1,
            col=1,
        )
        if bound_moe is not None:
            fig.add_shape(
                type="rect",
                x0=x_min,
                x1=x_max,
                y0=bound_val - bound_moe,
                y1=bound_val + bound_moe,
                fillcolor="rgba(44, 160, 44, 0.15)",
                line_width=0,
                row=1,
                col=1,
            )

    # Horizontal Black Dotted Line for Random Metric
    if random_metric is not None:
        fig.add_shape(
            type="line",
            x0=x_min,
            x1=x_max,
            y0=random_metric,
            y1=random_metric,
            line=dict(color=random_color, width=1.8, dash="dot"),
            row=1,
            col=1,
        )

    # Y-limits: include random_metric when it sits near the curve band so the dotted
    # line has clean headroom and isn't clipped against the top edge.
    all_values = (
        train_values
        + val_values
        + train_smoothed
        + val_smoothed
        + ([latest_test_value] if latest_test_value is not None else [])
    )
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        span = max(y_max - y_min, 0.01)
        if random_metric is not None and (
            y_min - 0.35 * span <= random_metric <= y_max + 0.35 * span
        ):
            y_min = min(y_min, random_metric)
            y_max = max(y_max, random_metric)
            span = max(y_max - y_min, 0.01)
        y_padding = max(span * 0.10, 0.01)
        y_range = [y_min - y_padding, y_max + y_padding]
    else:
        y_range = None

    latest_train_step = train_steps[-1] if train_steps else None
    latest_train_value = train_values[-1] if train_values else None
    latest_train_smoothed = train_smoothed[-1] if train_smoothed else None

    latest_val_step = val_steps[-1] if val_steps else None
    latest_val_value = val_values[-1] if val_values else None
    latest_val_smoothed = val_smoothed[-1] if val_smoothed else None

    if bound_val is not None:
        bound_val_str = f"{bound_val:.4f}"
        if bound_moe is not None:
            bound_val_str += f" ±{bound_moe:.4f}"
    else:
        bound_val_str = "-"

    # Plain-text labels + per-cell color matrix (Plotly go.Table does not parse <span> HTML)
    row_labels = [
        "● train",
        "● validation",
        "● test",
        f"● {bound_label}",
    ]
    row_colors = [
        train_color,
        val_color,
        test_color,
        bound_color,
    ]
    smoothed_col = [
        f"{latest_train_smoothed:.4f}"
        if latest_train_smoothed is not None
        else "-",
        f"{latest_val_smoothed:.4f}" if latest_val_smoothed is not None else "-",
        f"{latest_test_smoothed:.4f}"
        if latest_test_smoothed is not None
        else "-",
        "-",
    ]
    value_col = [
        f"{latest_train_value:.4f}" if latest_train_value is not None else "-",
        f"{latest_val_value:.4f}" if latest_val_value is not None else "-",
        f"{latest_test_value:.4f}" if latest_test_value is not None else "-",
        bound_val_str,
    ]
    step_col = [
        f"{latest_train_step}" if latest_train_step is not None else "-",
        f"{latest_val_step}" if latest_val_step is not None else "-",
        f"{latest_test_step}" if latest_test_step is not None else "-",
        "-",
    ]

    if random_metric is not None:
        row_labels.append("● random")
        row_colors.append(random_color)
        smoothed_col.append("-")
        value_col.append(f"{random_metric:.4f}")
        step_col.append("-")

    n_rows = len(row_labels)
    fig.add_trace(
        go.Table(
            columnwidth=[1.45, 1.15, 1.15, 0.90],
            header=dict(
                values=["", "<b>Smoothed</b>", "<b>Value</b>", "<b>Step</b>"],
                fill_color=["white", "#f9fafb", "#f9fafb", "#f9fafb"],
                line_color=["white", "#f3f4f6", "#f3f4f6", "#f3f4f6"],
                line_width=1.5,
                align=["left", "center", "center", "center"],
                font=dict(
                    color="#4b5563",
                    size=14,
                    family="DejaVu Sans, Arial, sans-serif",
                ),
                height=28,
            ),
            cells=dict(
                values=[row_labels, smoothed_col, value_col, step_col],
                fill_color="#ffffff",
                line_color="#f3f4f6",
                line_width=1.5,
                align=["left", "center", "center", "center"],
                font=dict(
                    color=[
                        row_colors,
                        ["#111111"] * n_rows,
                        ["#111111"] * n_rows,
                        ["#111111"] * n_rows,
                    ],
                    size=14,
                    family="DejaVu Sans, Arial, sans-serif",
                ),
                height=26,
            ),
        ),
        row=2,
        col=1,
    )

    display_title = scalar_name.replace("@", "_")
    fig.update_xaxes(
        domain=[0.22, 0.98],
        range=[x_min, x_max],
        showgrid=True,
        gridcolor="#cbd5e1",
        gridwidth=1.8,
        zeroline=False,
        ticks="outside",
        tickwidth=2,
        ticklen=6,
        tickcolor="black",
        tickfont=dict(
            color="black", size=14, family="DejaVu Sans, Arial, sans-serif"
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=y_range,
        showgrid=True,
        gridcolor="#cbd5e1",
        gridwidth=1.8,
        zeroline=False,
        ticks="outside",
        tickwidth=2,
        ticklen=6,
        tickcolor="black",
        tickfont=dict(
            color="black", size=14, family="DejaVu Sans, Arial, sans-serif"
        ),
        row=1,
        col=1,
    )

    fig_height = 520 if n_rows == 4 else 555
    fig.update_layout(
        title=dict(
            text=f"<b>{display_title}</b>",
            x=0.22,
            y=0.955,
            xanchor="left",
            font=dict(
                size=22, color="black", family="DejaVu Sans, Arial, sans-serif"
            ),
        ),
        width=640,
        height=fig_height,
        margin=dict(l=24, r=24, t=58, b=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, scale=2)

    # Save companion JSON summary matching `test_plot_training_curves.py`
    out_dict = {
        "latest_train_step": latest_train_step,
        "latest_train_value": latest_train_value,
        "latest_train_smoothed": latest_train_smoothed,
        "latest_val_step": latest_val_step,
        "latest_val_value": latest_val_value,
        "latest_val_smoothed": latest_val_smoothed,
        "latest_test_step": latest_test_step,
        "latest_test_value": latest_test_value,
        "latest_test_smoothed": latest_test_smoothed,
        "random_metric": random_metric,
    }
    if is_loss:
        out_dict["irreducible_error"] = bound_val
    else:
        out_dict["ceiling"] = bound_val

    json_summary_path = output_path.replace(".png", ".json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=4)

    if display_in_notebook:
        try:
            from IPython.display import Image, display

            display(Image(filename=output_path))
        except ImportError:
            pass

    return output_path


def plot_all_reranker_curves(
    working_dir: str,
    outdir: Optional[str] = None,
    use_hpo_artifacts: bool = True,
    smoothing_weight: float = 0.6,
    display_in_notebook: bool = True,
) -> Dict[str, str]:
    """
    Reads `best_hyperparameters.json` (or `training_hyperparameters.json`),
    `best_hpo_metrics_history.json` (or `train_metrics_history.json`), and
    `test_metrics_history.json` from `working_dir`, and exports PNG + JSON charts.
    """
    hyperparams_path = os.path.join(
        working_dir,
        "best_hyperparameters.json"
        if use_hpo_artifacts
        else "training_hyperparameters.json",
    )
    history_path = os.path.join(
        working_dir,
        "best_hpo_metrics_history.json"
        if use_hpo_artifacts
        else "train_metrics_history.json",
    )
    test_history_path = os.path.join(working_dir, "test_metrics_history.json")

    if outdir is None:
        outdir = os.path.join(working_dir, "training_metrics_pngs")

    hyperparams_dict = load_hyperparams_dict(hyperparams_path)
    num_candidates = int(hyperparams_dict.get("num_candidates", 20))
    eval_top_k = int(hyperparams_dict.get("eval_top_k", 5))
    focal_gamma = float(hyperparams_dict.get("focal_gamma", 2.0))

    random_ndcg = calc_random_ndcg(k=eval_top_k, pool_size=num_candidates)
    random_recall = calc_random_recall(k=eval_top_k, pool_size=num_candidates)
    random_precision = calc_random_precision(
        k=eval_top_k, pool_size=num_candidates
    )
    random_mrr = calc_random_mrr(k=eval_top_k, pool_size=num_candidates)
    random_softmax_loss = calc_random_softmax_loss(num_candidates)
    random_focal_loss = calc_random_focal_loss(num_candidates, focal_gamma)
    random_top1_prob = 1.0 / float(num_candidates)

    with open(history_path, "r", encoding="utf-8") as f:
        log_history = json.load(f)

    test_metrics_dict: Optional[Dict[str, Any]] = None
    if os.path.exists(test_history_path):
        with open(test_history_path, "r", encoding="utf-8") as f:
            test_payload = json.load(f)
            test_metrics_dict = test_payload.get("metrics", test_payload)

    metrics = list_reranker_metrics(log_history)
    print(f"metrics: {metrics}", flush=True)

    saved_pngs: Dict[str, str] = {}
    for metric in metrics:
        safe_metric_name = metric.replace("@", "_")
        outfile = os.path.join(outdir, f"{safe_metric_name}.png")

        random_metric: Optional[float] = None
        if "recall" in metric or "hit_rate" in metric:
            random_metric = random_recall
        elif "ndcg" in metric:
            random_metric = random_ndcg
        elif "mrr" in metric:
            random_metric = random_mrr
        elif "precision" in metric:
            random_metric = random_precision
        elif metric == "loss_listwise_raw":
            random_metric = random_focal_loss
        elif "entropy" in metric:
            random_metric = random_softmax_loss
        elif "top1_prob" in metric:
            random_metric = random_top1_prob

        written = generate_reranker_chart(
            log_history=log_history,
            test_metrics_dict=test_metrics_dict,
            irred_err_dict=None,
            random_metric=random_metric,
            scalar_name=metric,
            output_path=outfile,
            smoothing_weight=smoothing_weight,
            display_in_notebook=display_in_notebook,
        )
        if written:
            saved_pngs[metric] = written

    print(f"wrote pngs to {outdir}", flush=True)
    return saved_pngs

class PlotRerankerTrainingTest(unittest.TestCase):
    def test_plot(self):
        
        working_dir = os.path.join(get_output_working_dir(), "full")
        outdir = os.path.join(working_dir, "training_metrics_pngs")
        
        saved = plot_all_reranker_curves(
            working_dir=working_dir,
            outdir=outdir,
            use_hpo_artifacts=True,
            display_in_notebook=False,
        )
        self.assertGreater(len(saved), 0)
        for png_path in saved.values():
            self.assertTrue(os.path.exists(png_path))
            self.assertTrue(os.path.exists(png_path.replace(".png", ".json")))


if __name__ == "__main__":
    unittest.main()