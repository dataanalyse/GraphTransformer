import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


MODEL_ORDER = ["baseline", "gcn", "graph_transformer", "graphormer"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "gcn": "GCN",
    "graph_transformer": "Graph Transformer",
    "graphormer": "Graphormer",
}
MODEL_STYLES = {
    "baseline": {"linewidth": 2.8, "linestyle": "--", "marker": "s", "color": "black"},
    "gcn": {"linewidth": 2.0, "linestyle": "-", "marker": "o"},
    "graph_transformer": {"linewidth": 2.0, "linestyle": "-", "marker": "^"},
    "graphormer": {"linewidth": 2.0, "linestyle": "-", "marker": "D"},
}


def _load_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "metrics.csv")


def _load_run_start(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_start.json").read_text(encoding="utf-8"))


def _compute_test_target_mean(run_dir: Path) -> float:
    start = _load_run_start(run_dir)
    cfg = start.get("config", {})
    ds = start.get("dataset_stats", {})
    data_dir = Path(cfg["data_dir"])
    y = torch.load(data_dir / "Y_lcc_v1.pt").float()
    split_t = int(ds["split_t"])
    return float(y[split_t:].mean().item())


def _collect_runs(runs_root: Path, exp_map: dict[str, str]) -> dict[str, list[Path]]:
    collected = {}
    for model_key, exp_name in exp_map.items():
        exp_dir = runs_root / exp_name
        if not exp_dir.exists():
            collected[model_key] = []
            continue
        collected[model_key] = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    return collected


def _build_mean_curves(run_dirs: list[Path]) -> pd.DataFrame | None:
    curves = []
    for run_dir in run_dirs:
        metrics = _load_metrics(run_dir).copy()
        mean_true = _compute_test_target_mean(run_dir)
        metrics["normalized_mae"] = metrics["test_mae"] / mean_true
        metrics["percent_error"] = metrics["normalized_mae"] * 100.0
        curves.append(metrics[["epoch", "test_mae", "normalized_mae", "percent_error"]])
    if not curves:
        return None

    merged = curves[0].rename(
        columns={
            "test_mae": "test_mae_0",
            "normalized_mae": "normalized_mae_0",
            "percent_error": "percent_error_0",
        }
    )
    for idx, curve in enumerate(curves[1:], start=1):
        renamed = curve.rename(
            columns={
                "test_mae": f"test_mae_{idx}",
                "normalized_mae": f"normalized_mae_{idx}",
                "percent_error": f"percent_error_{idx}",
            }
        )
        merged = merged.merge(renamed, on="epoch", how="inner")

    for metric in ["test_mae", "normalized_mae", "percent_error"]:
        cols = [c for c in merged.columns if c.startswith(f"{metric}_")]
        merged[f"mean_{metric}"] = merged[cols].mean(axis=1)
        merged[f"std_{metric}"] = merged[cols].std(axis=1)
    return merged


def _plot_curves(curves: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        ("mean_test_mae", "Raw Test MAE", None),
        ("mean_percent_error", "Normalized Test Error (%)", None),
        ("mean_percent_error", "Normalized Test Error (%)", 60),
    ]

    for model in MODEL_ORDER:
        curve = curves.get(model)
        if curve is None:
            continue
        style = MODEL_STYLES[model]
        epochs = curve["epoch"]
        for ax, (metric, ylabel, xmax) in zip(axes, panels):
            if xmax is not None:
                sub = curve.loc[curve["epoch"] <= xmax]
                epochs = sub["epoch"]
                values = sub[metric]
            else:
                values = curve[metric]
            ax.plot(
                epochs,
                values,
                label=MODEL_LABELS[model],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(epochs) // 10),
                color=style.get("color"),
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)

    axes[0].set_title("Full Training Window")
    axes[1].set_title("Full Training Window")
    axes[2].set_title("Early Epoch Zoom (0-60)")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pilot_lcc_n40_dense_eval_curves.png", dpi=300)
    plt.close(fig)


def _save_summary(curves: dict[str, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for model in MODEL_ORDER:
        curve = curves.get(model)
        if curve is None:
            continue
        best_idx = curve["mean_test_mae"].idxmin()
        row = curve.loc[best_idx]
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "best_epoch_by_mean_mae": int(row["epoch"]),
                "mean_test_mae": float(row["mean_test_mae"]),
                "mean_percent_error": float(row["mean_percent_error"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "pilot_lcc_n40_dense_eval_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="v2/runs")
    parser.add_argument("--out_dir", default="v2/runs/final_figures")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_map = {
        "baseline": "baseline_graph_level_finalpilot_lcc_n40",
        "gcn": "gcn_graph_level_finalpilot_lcc_n40",
        "graph_transformer": "graph_transformer_graph_level_finalpilot_lcc_n40",
        "graphormer": "graphormer_graph_level_finalpilot_lcc_n40",
    }
    run_dirs = _collect_runs(runs_root, exp_map)
    curves = {}
    for model, dirs in run_dirs.items():
        curve = _build_mean_curves(dirs)
        if curve is not None:
            curves[model] = curve
    _plot_curves(curves, out_dir)
    _save_summary(curves, out_dir)


if __name__ == "__main__":
    main()
