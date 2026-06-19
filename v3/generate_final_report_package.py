import argparse
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
MODEL_OFFSETS = {
    "gcn": -0.45,
    "graph_transformer": -0.15,
    "graphormer": 0.15,
    "baseline": 0.45,
}
GRAPH_TARGET_LABELS = {
    "lcc_fraction": "LCC Fraction",
    "component_fraction": "Component Fraction",
    "diameter_fraction": "Diameter Fraction",
    "edge_survival_ratio": "Edge Survival Ratio",
    "lcc_trajectory": "LCC Trajectory",
}
GRAPH_TARGET_FILES = {
    "lcc_fraction": "Y_lcc_v1.pt",
    "component_fraction": "Y_components_v1.pt",
    "diameter_fraction": "Y_diameter_v1.pt",
    "edge_survival_ratio": "Y_edge_survival_v1.pt",
    "lcc_trajectory": "Y_lcc_traj_v1.pt",
}


def normalize_model_family(experiment: str) -> str:
    prefixes = [
        "baseline_graph_level_",
        "gcn_graph_level_",
        "graph_transformer_graph_level_",
        "graphormer_graph_level_",
    ]
    for prefix in prefixes:
        if experiment.startswith(prefix):
            return prefix.replace("_graph_level_", "")
    if experiment.endswith("_graph_level"):
        return experiment.replace("_graph_level", "")
    return experiment


def infer_task_type(row: pd.Series) -> str:
    graph_target = str(row.get("graph_target_name", "") or "")
    if graph_target:
        return "graph_level"
    return "node_level_health"


def load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    numeric_cols = [
        "num_nodes",
        "seed",
        "last_test_acc",
        "best_test_acc",
        "last_test_mae",
        "best_test_mae",
        "final_train_loss",
        "final_avg_train_loss",
        "final_train_mse",
        "final_logged_epoch",
        "split_t",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["graph_target_name"] = df["graph_target_name"].fillna("")
    df["graph_target_desc"] = df["graph_target_desc"].fillna("")
    df["model_family"] = df["experiment"].apply(normalize_model_family)
    df["task_type"] = df.apply(infer_task_type, axis=1)
    return df


def compute_true_mean(row: pd.Series, data_root: Path) -> float | None:
    graph_target = str(row.get("graph_target_name", "") or "")
    if not graph_target:
        return None
    graph_tag = str(row["graph_tag"])
    split_mode = str(row.get("split_mode", "") or "")
    split_t = int(row["split_t"]) if not pd.isna(row["split_t"]) else None
    if split_mode == "heldout_seed":
        seeds_text = str(row.get("test_seeds", "") or "")
        seeds = [int(part.strip()) for part in seeds_text.split(",") if part.strip()]
        if not seeds:
            return None
        values = []
        for seed in seeds:
            y = torch.load(data_root / graph_tag / f"seed_{seed}" / GRAPH_TARGET_FILES[graph_target]).float()
            values.append(float(y.mean().item()))
        return sum(values) / len(values)
    seed = int(row["seed"]) if not pd.isna(row["seed"]) else None
    if seed is None or split_t is None:
        return None
    y = torch.load(data_root / graph_tag / f"seed_{seed}" / GRAPH_TARGET_FILES[graph_target]).float()
    return float(y[split_t:].mean().item())


def add_normalized_columns(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    df = df.copy()
    true_means = []
    for _, row in df.iterrows():
        try:
            true_means.append(compute_true_mean(row, data_root))
        except Exception:
            true_means.append(None)
    df["mean_true_test"] = true_means
    df["normalized_best_test_mae"] = df["best_test_mae"] / df["mean_true_test"]
    df["percent_best_test_error"] = df["normalized_best_test_mae"] * 100.0
    df["normalized_last_test_mae"] = df["last_test_mae"] / df["mean_true_test"]
    df["percent_last_test_error"] = df["normalized_last_test_mae"] * 100.0
    return df


def summarize_node(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model_family", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_best_test_acc=("best_test_acc", "mean"),
            std_best_test_acc=("best_test_acc", "std"),
        )
        .sort_values(["num_nodes", "model_family"])
    )


def summarize_graph(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["graph_target_name", "graph_target_desc", "model_family", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_best_test_mae=("best_test_mae", "mean"),
            std_best_test_mae=("best_test_mae", "std"),
            mean_percent_best_test_error=("percent_best_test_error", "mean"),
            std_percent_best_test_error=("percent_best_test_error", "std"),
            mean_best_epoch=("final_logged_epoch", "mean"),
        )
        .sort_values(["graph_target_name", "num_nodes", "model_family"])
    )


def plot_metric_vs_size(summary_df: pd.DataFrame, metric_col: str, error_col: str, title: str, ylabel: str, out_path: Path) -> None:
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
            sub[metric_col],
            yerr=sub[error_col].fillna(0.0),
            capsize=4,
            label=MODEL_LABELS[model],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            color=style.get("color"),
        )
    xticks = sorted(summary_df["num_nodes"].dropna().unique())
    plt.xticks(xticks, [str(int(v)) for v in xticks])
    plt.xlabel("Number of Nodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_mean_curve(run_paths: list[str], metric_name: str, data_root: Path, summary_df: pd.DataFrame) -> pd.DataFrame | None:
    curves = []
    for run_path in run_paths:
        metrics_path = Path(run_path) / "metrics.csv"
        if not metrics_path.exists():
            continue
        run_row = summary_df.loc[summary_df["run_path"] == run_path].iloc[0]
        metrics = pd.read_csv(metrics_path)
        if metric_name == "test_acc":
            if "test_acc" not in metrics.columns:
                continue
            curves.append(metrics[["epoch", "test_acc"]])
        else:
            if "test_mae" not in metrics.columns:
                continue
            true_mean = compute_true_mean(run_row, data_root)
            metrics["percent_error"] = (metrics["test_mae"] / true_mean) * 100.0
            curves.append(metrics[["epoch", "test_mae", "percent_error"]])
    if not curves:
        return None

    value_cols = [col for col in curves[0].columns if col != "epoch"]
    merged = curves[0].rename(columns={col: f"{col}_0" for col in value_cols})
    for idx, curve in enumerate(curves[1:], start=1):
        renamed = curve.rename(columns={col: f"{col}_{idx}" for col in value_cols})
        merged = merged.merge(renamed, on="epoch", how="inner")
    for col in value_cols:
        cols = [c for c in merged.columns if c.startswith(f"{col}_")]
        merged[f"mean_{col}"] = merged[cols].mean(axis=1)
    return merged


def plot_graph_curves(df: pd.DataFrame, out_dir: Path, data_root: Path) -> None:
    graph_df = df.loc[(df["task_type"] == "graph_level") & (df["split_mode"].fillna("") != "heldout_seed")].copy()
    for target in sorted(graph_df["graph_target_name"].unique()):
        target_df = graph_df.loc[graph_df["graph_target_name"] == target]
        label = GRAPH_TARGET_LABELS.get(target, target)
        for graph_tag in sorted(target_df["graph_tag"].unique()):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            any_data = False
            for model in MODEL_ORDER:
                run_paths = target_df.loc[
                    (target_df["graph_tag"] == graph_tag) & (target_df["model_family"] == model),
                    "run_path",
                ].dropna().tolist()
                curve = build_mean_curve(run_paths, "test_mae", data_root, graph_df)
                if curve is None:
                    continue
                any_data = True
                style = MODEL_STYLES[model]
                for ax, xlimit in zip(axes, [None, None, 60]):
                    sub = curve if xlimit is None else curve.loc[curve["epoch"] <= xlimit]
                    ycol = "mean_test_mae" if ax is axes[0] else "mean_percent_error"
                    ax.plot(
                        sub["epoch"],
                        sub[ycol],
                        label=MODEL_LABELS[model],
                        linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markevery=max(1, len(sub) // 10),
                        color=style.get("color"),
                    )
            if not any_data:
                plt.close(fig)
                continue
            axes[0].set_title(f"{graph_tag} {label} Raw MAE")
            axes[1].set_title(f"{graph_tag} {label} Error (%)")
            axes[2].set_title(f"{graph_tag} {label} Error (%) Early Zoom")
            axes[0].set_ylabel("Test MAE")
            axes[1].set_ylabel("Percent Error")
            axes[2].set_ylabel("Percent Error")
            for ax in axes:
                ax.set_xlabel("Epoch")
                ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"final_{target}_{graph_tag}_curves.png", dpi=300)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", default="v2/final_runs/results_summary.csv")
    parser.add_argument("--data_root", default="v2/data")
    parser.add_argument("--out_dir", default="v2/runs/final_figures")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)
    df = add_normalized_columns(df, data_root)

    node_df = df.loc[df["task_type"] == "node_level_health"].copy()
    graph_df = df.loc[df["task_type"] == "graph_level"].copy()
    heldout_df = graph_df.loc[graph_df["split_mode"].fillna("") == "heldout_seed"].copy()
    graph_nonheldout_df = graph_df.loc[graph_df["split_mode"].fillna("") != "heldout_seed"].copy()

    node_summary = summarize_node(node_df)
    node_summary.to_csv(out_dir / "final_node_results_raw.csv", index=False)

    graph_summary = summarize_graph(graph_nonheldout_df)
    graph_summary.to_csv(out_dir / "final_graph_results_raw_and_percent.csv", index=False)

    heldout_summary = summarize_graph(heldout_df)
    heldout_summary.to_csv(out_dir / "final_heldout_graph_results_raw_and_percent.csv", index=False)

    traj_summary = graph_summary.loc[graph_summary["graph_target_name"] == "lcc_trajectory"].copy()
    traj_summary.to_csv(out_dir / "final_trajectory_results_raw_and_percent.csv", index=False)

    if not node_summary.empty:
        plot_metric_vs_size(
            node_summary,
            "mean_best_test_acc",
            "std_best_test_acc",
            "Final V2 Node-Level Accuracy",
            "Mean Best Test Accuracy",
            out_dir / "final_node_accuracy_vs_graph_size.png",
        )

    for target in sorted(graph_nonheldout_df["graph_target_name"].unique()):
        target_summary = graph_summary.loc[graph_summary["graph_target_name"] == target]
        label = GRAPH_TARGET_LABELS.get(target, target)
        plot_metric_vs_size(
            target_summary,
            "mean_best_test_mae",
            "std_best_test_mae",
            f"Final {label} Raw MAE",
            "Mean Best Test MAE",
            out_dir / f"final_{target}_mae_vs_graph_size.png",
        )
        plot_metric_vs_size(
            target_summary,
            "mean_percent_best_test_error",
            "std_percent_best_test_error",
            f"Final {label} Error (%)",
            "Mean Best Test Error (%)",
            out_dir / f"final_{target}_percent_vs_graph_size.png",
        )

    for target in sorted(heldout_df["graph_target_name"].unique()):
        target_summary = heldout_summary.loc[heldout_summary["graph_target_name"] == target]
        label = GRAPH_TARGET_LABELS.get(target, target)
        plot_metric_vs_size(
            target_summary,
            "mean_best_test_mae",
            "std_best_test_mae",
            f"Held-Out {label} Raw MAE",
            "Mean Best Test MAE",
            out_dir / f"final_heldout_{target}_mae_vs_graph_size.png",
        )
        plot_metric_vs_size(
            target_summary,
            "mean_percent_best_test_error",
            "std_percent_best_test_error",
            f"Held-Out {label} Error (%)",
            "Mean Best Test Error (%)",
            out_dir / f"final_heldout_{target}_percent_vs_graph_size.png",
        )

    plot_graph_curves(df, out_dir, data_root)


if __name__ == "__main__":
    main()
