import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ["baseline", "gcn", "graph_transformer", "graphormer"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "gcn": "GCN",
    "graph_transformer": "Graph Transformer",
    "graphormer": "Graphormer",
}
MODEL_STYLES = {
    "baseline": {
        "linewidth": 3.0,
        "linestyle": "--",
        "marker": "s",
        "zorder": 6,
        "color": "black",
    },
    "gcn": {"linewidth": 2.0, "linestyle": "-", "marker": "o", "zorder": 3},
    "graph_transformer": {
        "linewidth": 2.0,
        "linestyle": "-",
        "marker": "^",
        "zorder": 4,
    },
    "graphormer": {"linewidth": 2.0, "linestyle": "-", "marker": "D", "zorder": 4},
}
MODEL_OFFSETS = {
    "gcn": -0.45,
    "graph_transformer": -0.15,
    "graphormer": 0.15,
    "baseline": 0.45,
}
GRAPH_TARGET_LABELS = {
    "future_lcc_fraction": "LCC Fraction",
    "lcc_fraction": "LCC Fraction",
    "lcc_trajectory": "LCC Trajectory",
    "component_fraction": "Component Fraction",
    "diameter_fraction": "Diameter Fraction",
    "edge_survival_ratio": "Edge Survival Ratio",
}
GRAPH_TARGET_SLUGS = {
    "future_lcc_fraction": "lcc",
    "lcc_fraction": "lcc",
    "lcc_trajectory": "lcc_trajectory",
    "component_fraction": "components",
    "diameter_fraction": "diameter",
    "edge_survival_ratio": "edge_survival",
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
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["graph_target_name"] = df["graph_target_name"].fillna("")
    df["graph_target_desc"] = df["graph_target_desc"].fillna("")
    df["model_family"] = df["experiment"].apply(normalize_model_family)
    df["task_type"] = df.apply(infer_task_type, axis=1)
    return df


def dedupe_latest_heldout_runs(df: pd.DataFrame) -> pd.DataFrame:
    heldout_df = df.loc[df["split_mode"].fillna("") == "heldout_seed"].copy()
    other_df = df.loc[df["split_mode"].fillna("") != "heldout_seed"].copy()
    if heldout_df.empty:
        return df
    heldout_df = (
        heldout_df.sort_values("run_id")
        .groupby(["experiment", "graph_tag", "graph_target_name"], as_index=False)
        .tail(1)
    )
    return pd.concat([other_df, heldout_df], ignore_index=True)


def filter_comparable_runs(df: pd.DataFrame) -> pd.DataFrame:
    filtered_frames = []

    node_df = df.loc[df["task_type"] == "node_level_health"].copy()
    for num_nodes in sorted(node_df["num_nodes"].dropna().unique()):
        size_df = node_df.loc[node_df["num_nodes"] == num_nodes].copy()
        model_seed_sets = []
        for model in MODEL_ORDER:
            model_seeds = set(size_df.loc[size_df["model_family"] == model, "seed"].dropna().astype(int))
            if model_seeds:
                model_seed_sets.append(model_seeds)
        if len(model_seed_sets) == len(MODEL_ORDER):
            common_seeds = set.intersection(*model_seed_sets)
            if common_seeds:
                filtered_frames.append(size_df.loc[size_df["seed"].isin(sorted(common_seeds))].copy())

    graph_df = df.loc[df["task_type"] == "graph_level"].copy()
    for graph_target_name in sorted(graph_df["graph_target_name"].dropna().unique()):
        target_df = graph_df.loc[graph_df["graph_target_name"] == graph_target_name].copy()
        for num_nodes in sorted(target_df["num_nodes"].dropna().unique()):
            size_df = target_df.loc[target_df["num_nodes"] == num_nodes].copy()
            model_seed_sets = []
            for model in MODEL_ORDER:
                model_seeds = set(size_df.loc[size_df["model_family"] == model, "seed"].dropna().astype(int))
                if model_seeds:
                    model_seed_sets.append(model_seeds)
            if len(model_seed_sets) == len(MODEL_ORDER):
                common_seeds = set.intersection(*model_seed_sets)
                if common_seeds:
                    filtered_frames.append(size_df.loc[size_df["seed"].isin(sorted(common_seeds))].copy())

    if not filtered_frames:
        return df.iloc[0:0].copy()
    return pd.concat(filtered_frames, ignore_index=True)


def summarize_node_results(node_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        node_df.groupby(["model_family", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_last_test_acc=("last_test_acc", "mean"),
            std_last_test_acc=("last_test_acc", "std"),
            mean_best_test_acc=("best_test_acc", "mean"),
            std_best_test_acc=("best_test_acc", "std"),
        )
        .sort_values(["num_nodes", "model_family"])
    )
    return summary


def summarize_graph_results(graph_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        graph_df.groupby(["graph_target_name", "graph_target_desc", "model_family", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_last_test_mae=("last_test_mae", "mean"),
            std_last_test_mae=("last_test_mae", "std"),
            mean_best_test_mae=("best_test_mae", "mean"),
            std_best_test_mae=("best_test_mae", "std"),
        )
        .sort_values(["graph_target_name", "num_nodes", "model_family"])
    )
    return summary


def summarize_graph_results_by_split(graph_df: pd.DataFrame, split_mode: str) -> pd.DataFrame:
    summary = summarize_graph_results(graph_df.loc[graph_df["split_mode"].fillna("") == split_mode].copy())
    return summary


def plot_metric_vs_size(summary_df: pd.DataFrame, out_path: Path, metric_col: str, error_col: str, ylabel: str, title: str) -> None:
    if summary_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for model in MODEL_ORDER:
        sub = summary_df.loc[summary_df["model_family"] == model]
        if sub.empty:
            continue
        style = MODEL_STYLES[model]
        x = sub["num_nodes"] + MODEL_OFFSETS.get(model, 0.0)
        plt.errorbar(
            x,
            sub[metric_col],
            yerr=sub[error_col].fillna(0.0),
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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_bar_comparison(summary_df: pd.DataFrame, out_path: Path, metric_col: str, ylabel: str, title: str) -> None:
    if summary_df.empty:
        return
    pivot = (
        summary_df.pivot(index="num_nodes", columns="model_family", values=metric_col)
        .reindex(columns=[m for m in MODEL_ORDER if m in summary_df["model_family"].unique()])
        .rename(columns=MODEL_LABELS)
    )
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_headline_panel(node_summary: pd.DataFrame, lcc_summary: pd.DataFrame, out_path: Path) -> None:
    if node_summary.empty or lcc_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in MODEL_ORDER:
        node_sub = node_summary.loc[node_summary["model_family"] == model]
        lcc_sub = lcc_summary.loc[lcc_summary["model_family"] == model]
        style = MODEL_STYLES[model]
        if not node_sub.empty:
            axes[0].errorbar(
                node_sub["num_nodes"] + MODEL_OFFSETS.get(model, 0.0),
                node_sub["mean_best_test_acc"],
                yerr=node_sub["std_best_test_acc"].fillna(0.0),
                capsize=4,
                label=MODEL_LABELS[model],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                zorder=style["zorder"],
                color=style.get("color"),
            )
        if not lcc_sub.empty:
            axes[1].errorbar(
                lcc_sub["num_nodes"] + MODEL_OFFSETS.get(model, 0.0),
                lcc_sub["mean_best_test_mae"],
                yerr=lcc_sub["std_best_test_mae"].fillna(0.0),
                capsize=4,
                label=MODEL_LABELS[model],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                zorder=style["zorder"],
                color=style.get("color"),
            )
    for ax, summary_df in zip(axes, [node_summary, lcc_summary]):
        xticks = sorted(summary_df["num_nodes"].dropna().unique())
        ax.set_xticks(xticks, [str(int(v)) for v in xticks])
        ax.set_xlabel("Number of Nodes")
        ax.legend()
    axes[0].set_ylabel("Mean Best Test Accuracy")
    axes[1].set_ylabel("Mean Best Test MAE")
    axes[0].set_title("Node-Level Early Warning")
    axes[1].set_title("Graph-Level LCC Prediction")
    fig.suptitle("V2 Headline Results", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_heldout_headline_panel(graph_summary: pd.DataFrame, out_path: Path) -> None:
    if graph_summary.empty:
        return

    target_order = [
        "lcc_fraction",
        "component_fraction",
        "diameter_fraction",
        "edge_survival_ratio",
    ]
    available_targets = [
        target for target in target_order if target in set(graph_summary["graph_target_name"])
    ]
    if not available_targets:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, graph_target_name in zip(axes, available_targets):
        target_summary = graph_summary.loc[
            graph_summary["graph_target_name"] == graph_target_name
        ]
        for model in MODEL_ORDER:
            sub = target_summary.loc[target_summary["model_family"] == model]
            if sub.empty:
                continue
            style = MODEL_STYLES[model]
            ax.errorbar(
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
        xticks = sorted(target_summary["num_nodes"].dropna().unique())
        ax.set_xticks(xticks, [str(int(v)) for v in xticks])
        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Best Test MAE")
        ax.set_title(GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name))

    for ax in axes[len(available_targets):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODEL_ORDER))
    fig.suptitle("Held-Out Seed Structure Prediction", fontsize=13)
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_mean_curve(run_paths: list[str], value_cols: list[str]) -> pd.DataFrame | None:
    curves = []
    for run_path in run_paths:
        metrics_path = Path(run_path) / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        if not {"epoch", *value_cols}.issubset(metrics.columns):
            continue
        curves.append(metrics[["epoch", *value_cols]])
    if not curves:
        return None
    merged = curves[0].rename(columns={col: f"{col}_0" for col in value_cols})
    for idx, curve in enumerate(curves[1:], start=1):
        renamed = curve.rename(columns={col: f"{col}_{idx}" for col in value_cols})
        merged = merged.merge(renamed, on="epoch", how="inner")
    for col in value_cols:
        cols = [c for c in merged.columns if c.startswith(f"{col}_")]
        merged[f"mean_{col}"] = merged[cols].mean(axis=1)
    return merged


def plot_node_training_curves(node_df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = ["avg_train_loss", "test_acc"]
    for graph_tag in sorted(node_df["graph_tag"].dropna().unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        any_data = False
        for model in MODEL_ORDER:
            run_paths = node_df.loc[
                (node_df["graph_tag"] == graph_tag) & (node_df["model_family"] == model),
                "run_path",
            ].dropna().tolist()
            curve = build_mean_curve(run_paths, metric_cols)
            if curve is None:
                continue
            any_data = True
            style = MODEL_STYLES[model]
            axes[0].plot(curve["epoch"], curve["mean_avg_train_loss"], label=MODEL_LABELS[model], linewidth=style["linewidth"], linestyle=style["linestyle"], marker=style["marker"], markevery=max(1, len(curve)//6), zorder=style["zorder"], color=style.get("color"))
            axes[1].plot(curve["epoch"], curve["mean_test_acc"], label=MODEL_LABELS[model], linewidth=style["linewidth"], linestyle=style["linestyle"], marker=style["marker"], markevery=max(1, len(curve)//6), zorder=style["zorder"], color=style.get("color"))
        if not any_data:
            plt.close(fig)
            continue
        axes[0].set_title(f"{graph_tag} Node-Level Train")
        axes[1].set_title(f"{graph_tag} Node-Level Test")
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Average Train Loss")
        axes[1].set_ylabel("Test Accuracy")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"node_training_curves_{graph_tag}.png", dpi=300)
        plt.close(fig)


def plot_graph_training_curves(graph_df: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = ["train_mse", "test_mae"]
    for graph_target_name in sorted(graph_df["graph_target_name"].dropna().unique()):
        target_df = graph_df.loc[graph_df["graph_target_name"] == graph_target_name]
        slug = GRAPH_TARGET_SLUGS.get(graph_target_name, graph_target_name)
        label = GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name)
        for graph_tag in sorted(target_df["graph_tag"].dropna().unique()):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            any_data = False
            for model in MODEL_ORDER:
                run_paths = target_df.loc[
                    (target_df["graph_tag"] == graph_tag) & (target_df["model_family"] == model),
                    "run_path",
                ].dropna().tolist()
                curve = build_mean_curve(run_paths, metric_cols)
                if curve is None:
                    continue
                any_data = True
                style = MODEL_STYLES[model]
                axes[0].plot(curve["epoch"], curve["mean_train_mse"], label=MODEL_LABELS[model], linewidth=style["linewidth"], linestyle=style["linestyle"], marker=style["marker"], markevery=max(1, len(curve)//6), zorder=style["zorder"], color=style.get("color"))
                axes[1].plot(curve["epoch"], curve["mean_test_mae"], label=MODEL_LABELS[model], linewidth=style["linewidth"], linestyle=style["linestyle"], marker=style["marker"], markevery=max(1, len(curve)//6), zorder=style["zorder"], color=style.get("color"))
            if not any_data:
                plt.close(fig)
                continue
            axes[0].set_title(f"{graph_tag} {label} Train")
            axes[1].set_title(f"{graph_tag} {label} Test")
            axes[0].set_xlabel("Epoch")
            axes[1].set_xlabel("Epoch")
            axes[0].set_ylabel("Train MSE")
            axes[1].set_ylabel("Test MAE")
            axes[0].legend()
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"graph_training_curves_{slug}_{graph_tag}.png", dpi=300)
            plt.close(fig)


def save_report_markdown(node_summary: pd.DataFrame, graph_summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# V2 Report Figure Notes",
        "",
        "## Node-Level Early Warning",
        "",
        "| Model | N | Mean Best Accuracy | Std Best Accuracy | Seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in node_summary.sort_values(["num_nodes", "model_family"]).iterrows():
        lines.append(
            f"| {MODEL_LABELS.get(row['model_family'], row['model_family'])} | {int(row['num_nodes'])} | "
            f"{row['mean_best_test_acc']:.4f} | {0.0 if pd.isna(row['std_best_test_acc']) else row['std_best_test_acc']:.4f} | {int(row['seeds'])} |"
        )

    for graph_target_name in sorted(graph_summary["graph_target_name"].dropna().unique()):
        target_df = graph_summary.loc[graph_summary["graph_target_name"] == graph_target_name]
        label = GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name)
        lines.extend(
            [
                "",
                f"## Graph-Level {label}",
                "",
                "| Model | N | Mean Best MAE | Std Best MAE | Seeds |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in target_df.sort_values(["num_nodes", "model_family"]).iterrows():
            lines.append(
                f"| {MODEL_LABELS.get(row['model_family'], row['model_family'])} | {int(row['num_nodes'])} | "
                f"{row['mean_best_test_mae']:.4f} | {0.0 if pd.isna(row['std_best_test_mae']) else row['std_best_test_mae']:.4f} | {int(row['seeds'])} |"
            )

    (out_dir / "report_summary.md").write_text("\n".join(lines) + "\n")


def save_heldout_report_markdown(graph_summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# V2 Held-Out Seed Report Notes",
        "",
        "Train seeds: `1,2,3`",
        "Test seeds: `4,5`",
    ]
    for graph_target_name in sorted(graph_summary["graph_target_name"].dropna().unique()):
        target_df = graph_summary.loc[graph_summary["graph_target_name"] == graph_target_name]
        label = GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name)
        lines.extend(
            [
                "",
                f"## Held-Out {label}",
                "",
                "| Model | N | Mean Best MAE | Std Best MAE | Seeds |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in target_df.sort_values(["num_nodes", "model_family"]).iterrows():
            lines.append(
                f"| {MODEL_LABELS.get(row['model_family'], row['model_family'])} | {int(row['num_nodes'])} | "
                f"{row['mean_best_test_mae']:.4f} | {0.0 if pd.isna(row['std_best_test_mae']) else row['std_best_test_mae']:.4f} | {int(row['seeds'])} |"
            )
    (out_dir / "heldout_seed_report_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", default="v2/runs/results_summary.csv")
    parser.add_argument("--out_dir", default="v2/runs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = dedupe_latest_heldout_runs(load_summary(Path(args.summary_path)))
    df = filter_comparable_runs(df)
    node_df = df.loc[df["task_type"] == "node_level_health"].copy()
    graph_df = df.loc[df["task_type"] == "graph_level"].copy()

    node_summary = summarize_node_results(node_df)
    graph_summary = summarize_graph_results(graph_df)
    heldout_graph_summary = summarize_graph_results_by_split(graph_df, "heldout_seed")

    node_summary.to_csv(out_dir / "node_results_by_model_size.csv", index=False)
    graph_summary.to_csv(out_dir / "graph_results_by_model_size.csv", index=False)
    heldout_graph_summary.to_csv(out_dir / "heldout_graph_results_by_model_size.csv", index=False)

    plot_metric_vs_size(
        node_summary,
        out_dir / "node_accuracy_vs_graph_size.png",
        "mean_best_test_acc",
        "std_best_test_acc",
        "Mean Best Test Accuracy",
        "Node-Level Early Warning Accuracy vs Graph Size",
    )
    plot_bar_comparison(
        node_summary,
        out_dir / "node_model_comparison_by_size.png",
        "mean_best_test_acc",
        "Mean Best Test Accuracy",
        "Node-Level Early Warning Comparison by Graph Size",
    )

    for graph_target_name in sorted(graph_summary["graph_target_name"].dropna().unique()):
        target_summary = graph_summary.loc[graph_summary["graph_target_name"] == graph_target_name].copy()
        slug = GRAPH_TARGET_SLUGS.get(graph_target_name, graph_target_name)
        label = GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name)
        target_summary.to_csv(out_dir / f"graph_results_by_model_size_{slug}.csv", index=False)
        plot_metric_vs_size(
            target_summary,
            out_dir / f"graph_mae_vs_graph_size_{slug}.png",
            "mean_best_test_mae",
            "std_best_test_mae",
            "Mean Best Test MAE",
            f"Graph-Level {label} MAE vs Graph Size",
        )
        plot_bar_comparison(
            target_summary,
            out_dir / f"graph_model_comparison_by_size_{slug}.png",
            "mean_best_test_mae",
            "Mean Best Test MAE",
            f"Graph-Level {label} Comparison by Graph Size",
        )

    for graph_target_name in sorted(heldout_graph_summary["graph_target_name"].dropna().unique()):
        target_summary = heldout_graph_summary.loc[
            heldout_graph_summary["graph_target_name"] == graph_target_name
        ].copy()
        slug = GRAPH_TARGET_SLUGS.get(graph_target_name, graph_target_name)
        label = GRAPH_TARGET_LABELS.get(graph_target_name, graph_target_name)
        target_summary.to_csv(out_dir / f"heldout_graph_results_by_model_size_{slug}.csv", index=False)
        plot_metric_vs_size(
            target_summary,
            out_dir / f"heldout_graph_mae_vs_graph_size_{slug}.png",
            "mean_best_test_mae",
            "std_best_test_mae",
            "Mean Best Test MAE",
            f"Held-Out Seed {label} MAE vs Graph Size",
        )
        plot_bar_comparison(
            target_summary,
            out_dir / f"heldout_graph_model_comparison_by_size_{slug}.png",
            "mean_best_test_mae",
            "Mean Best Test MAE",
            f"Held-Out Seed {label} Comparison by Graph Size",
        )

    lcc_summary = graph_summary.loc[graph_summary["graph_target_name"] == "future_lcc_fraction"].copy()
    plot_headline_panel(node_summary, lcc_summary, out_dir / "v2_headline_results.png")
    plot_heldout_headline_panel(
        heldout_graph_summary,
        out_dir / "heldout_seed_headline_results.png",
    )
    plot_node_training_curves(node_df, out_dir)
    plot_graph_training_curves(graph_df, out_dir)
    save_report_markdown(node_summary, graph_summary, out_dir)
    save_heldout_report_markdown(heldout_graph_summary, out_dir)
    print(f"Wrote v2 report figures and summary tables to: {out_dir}")


if __name__ == "__main__":
    main()
