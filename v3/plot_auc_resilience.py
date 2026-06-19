from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AUC resilience summaries for scenario runs.")
    parser.add_argument(
        "--input",
        default="v3/data/processed/redundancy_v3/scenario_summary_metrics_v3.csv",
        help="Scenario summary metrics CSV.",
    )
    parser.add_argument(
        "--output",
        default="v3/runs/figures/redundancy_v3/auc_resilience_comparison.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.input)

    condition_order = [
        "baseline_additive",
        "redundant_additive",
        "redundant_buffer",
        "dampened_power_additive",
    ]
    shock_order = ["power", "telecom", "ems", "hospital", "mixed"]
    colors = {
        "power": "#c0392b",
        "telecom": "#2980b9",
        "ems": "#16a085",
        "hospital": "#8e44ad",
        "mixed": "#d35400",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    overall_data = [df.loc[df["condition"] == cond, "auc_resilience"].dropna().values for cond in condition_order]
    box = axes[0].boxplot(overall_data, labels=condition_order, patch_artist=True)
    for patch, cond in zip(box["boxes"], condition_order):
        fill = {
            "baseline_additive": "#e74c3c",
            "redundant_additive": "#9b59b6",
            "redundant_buffer": "#27ae60",
            "dampened_power_additive": "#3498db",
        }[cond]
        patch.set_facecolor(fill)
        patch.set_alpha(0.65)
    axes[0].set_title("AUC Resilience by Condition")
    axes[0].set_ylabel("AUC Resilience")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.3)

    summary = (
        df.groupby(["condition", "shock_type"], as_index=False)["auc_resilience"]
        .mean()
    )
    for shock in shock_order:
        sub = summary[summary["shock_type"] == shock].copy()
        sub["condition"] = pd.Categorical(sub["condition"], categories=condition_order, ordered=True)
        sub = sub.sort_values("condition")
        axes[1].plot(
            sub["condition"].astype(str),
            sub["auc_resilience"],
            marker="o",
            linewidth=2,
            label=shock,
            color=colors[shock],
        )
    axes[1].set_title("Mean AUC Resilience by Condition and Shock Type")
    axes[1].set_ylabel("Mean AUC Resilience")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Wrote AUC resilience figure to {output_path}")


if __name__ == "__main__":
    main()
