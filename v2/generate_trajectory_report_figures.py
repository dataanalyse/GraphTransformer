import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from generate_report_figures import (
    MODEL_LABELS,
    MODEL_OFFSETS,
    MODEL_ORDER,
    MODEL_STYLES,
    dedupe_latest_heldout_runs,
    load_summary,
)


def summarize_trajectory_results(df: pd.DataFrame) -> pd.DataFrame:
    valid_experiments = {
        "baseline_graph_level_lcc_trajectory",
        "gcn_graph_level_lcc_trajectory",
        "graph_transformer_graph_level_lcc_trajectory",
        "graphormer_graph_level_lcc_trajectory",
    }
    traj_df = df.loc[
        (df["graph_target_name"] == "lcc_trajectory")
        & (df["experiment"].isin(valid_experiments))
    ].copy()
    summary = (
        traj_df.groupby(["model_family", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_last_test_mae=("last_test_mae", "mean"),
            std_last_test_mae=("last_test_mae", "std"),
            mean_best_test_mae=("best_test_mae", "mean"),
            std_best_test_mae=("best_test_mae", "std"),
        )
        .sort_values(["num_nodes", "model_family"])
    )
    return summary


def plot_trajectory_vs_size(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = summary_df.loc[summary_df["model_family"] == model]
        if sub.empty:
            continue
        style = MODEL_STYLES[model]
        plt.errorbar(
            sub["num_nodes"] + MODEL_OFFSETS.get(model, 0.0),
            sub["mean_best_test_mae"],
            yerr=sub["std_best_test_mae"].fillna(0.0),
            capsize=4,
            label=MODEL_LABELS[model],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            zorder=style["zorder"],
            color=style.get("color"),
        )
    xticks = sorted(summary_df["num_nodes"].dropna().unique())
    plt.xticks(xticks, [str(int(v)) for v in xticks])
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Best Test MAE")
    plt.title("LCC Trajectory Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_trajectory_training_curves(df: pd.DataFrame, out_dir: Path) -> None:
    valid_experiments = {
        "baseline_graph_level_lcc_trajectory",
        "gcn_graph_level_lcc_trajectory",
        "graph_transformer_graph_level_lcc_trajectory",
        "graphormer_graph_level_lcc_trajectory",
    }
    traj_df = df.loc[
        (df["graph_target_name"] == "lcc_trajectory")
        & (df["experiment"].isin(valid_experiments))
    ].copy()
    metric_cols = ["train_mse", "test_mae"]
    for graph_tag in sorted(traj_df["graph_tag"].dropna().unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        any_data = False
        for model in MODEL_ORDER:
            run_paths = traj_df.loc[
                (traj_df["graph_tag"] == graph_tag) & (traj_df["model_family"] == model),
                "run_path",
            ].dropna().tolist()
            curves = []
            for run_path in run_paths:
                metrics_path = Path(run_path) / "metrics.csv"
                if not metrics_path.exists():
                    continue
                metrics = pd.read_csv(metrics_path)
                if not {"epoch", *metric_cols}.issubset(metrics.columns):
                    continue
                curves.append(metrics[["epoch", *metric_cols]])
            if not curves:
                continue
            any_data = True
            merged = curves[0].rename(columns={col: f"{col}_0" for col in metric_cols})
            for idx, curve in enumerate(curves[1:], start=1):
                renamed = curve.rename(columns={col: f"{col}_{idx}" for col in metric_cols})
                merged = merged.merge(renamed, on="epoch", how="inner")
            for col in metric_cols:
                cols = [c for c in merged.columns if c.startswith(f"{col}_")]
                merged[f"mean_{col}"] = merged[cols].mean(axis=1)
            style = MODEL_STYLES[model]
            axes[0].plot(
                merged["epoch"],
                merged["mean_train_mse"],
                label=MODEL_LABELS[model],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(merged) // 6),
                zorder=style["zorder"],
                color=style.get("color"),
            )
            axes[1].plot(
                merged["epoch"],
                merged["mean_test_mae"],
                label=MODEL_LABELS[model],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(merged) // 6),
                zorder=style["zorder"],
                color=style.get("color"),
            )
        if not any_data:
            plt.close(fig)
            continue
        axes[0].set_title(f"{graph_tag} LCC Trajectory Train")
        axes[1].set_title(f"{graph_tag} LCC Trajectory Test")
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Train MSE")
        axes[1].set_ylabel("Test MAE")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_training_curves_{graph_tag}.png", dpi=300)
        plt.close(fig)


def save_trajectory_markdown(summary_df: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# V2 Directive 3: LCC Trajectory Results",
        "",
        "Target: `Y[t] = [LCC(t+1), ..., LCC(t+5)]`",
        "",
        "| Model | N | Mean Best MAE | Std Best MAE | Seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.sort_values(["num_nodes", "model_family"]).iterrows():
        lines.append(
            f"| {MODEL_LABELS.get(row['model_family'], row['model_family'])} | {int(row['num_nodes'])} | "
            f"{row['mean_best_test_mae']:.4f} | "
            f"{0.0 if pd.isna(row['std_best_test_mae']) else row['std_best_test_mae']:.4f} | "
            f"{int(row['seeds'])} |"
        )
    (out_dir / "trajectory_report_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", default="v2/runs/results_summary.csv")
    parser.add_argument("--out_dir", default="v2/runs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = dedupe_latest_heldout_runs(load_summary(Path(args.summary_path)))
    summary = summarize_trajectory_results(df)
    summary.to_csv(out_dir / "trajectory_results_by_model_size.csv", index=False)
    plot_trajectory_vs_size(summary, out_dir / "trajectory_mae_vs_graph_size.png")
    plot_trajectory_training_curves(df, out_dir)
    save_trajectory_markdown(summary, out_dir)


if __name__ == "__main__":
    main()
