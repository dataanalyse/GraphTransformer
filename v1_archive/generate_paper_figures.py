import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    numeric_cols = [
        "num_nodes",
        "seed",
        "last_test_acc",
        "best_test_acc",
        "final_avg_train_loss",
        "final_logged_epoch",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_model_size_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = (
        df.groupby(["experiment", "graph_tag", "num_nodes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_last_test_acc=("last_test_acc", "mean"),
            std_last_test_acc=("last_test_acc", "std"),
            mean_best_test_acc=("best_test_acc", "mean"),
            std_best_test_acc=("best_test_acc", "std"),
        )
        .sort_values(["num_nodes", "experiment"])
    )
    grouped.to_csv(out_dir / "results_by_model_size.csv", index=False)
    return grouped


def plot_accuracy_vs_size(summary_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    offset_map = {
        "gcn": -0.45,
        "graph_transformer": -0.15,
        "graphormer": 0.15,
        "baseline": 0.45,
    }
    style_map = {
        "baseline": {
            "linewidth": 3.2,
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
    ordered = ["gcn", "graph_transformer", "graphormer", "baseline"]
    for experiment in ordered:
        sub = summary_df.loc[summary_df["experiment"] == experiment]
        if sub.empty:
            continue
        style = style_map.get(
            experiment,
            {"linewidth": 2.0, "linestyle": "-", "marker": "o", "zorder": 3},
        )
        x = sub["num_nodes"] + offset_map.get(experiment, 0.0)
        plt.errorbar(
            x,
            sub["mean_last_test_acc"],
            yerr=sub["std_last_test_acc"].fillna(0.0),
            capsize=4,
            label=experiment,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            zorder=style["zorder"],
            color=style.get("color"),
        )
    xticks = sorted(summary_df["num_nodes"].dropna().unique())
    plt.xticks(xticks, [str(int(v)) for v in xticks])
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Test Accuracy")
    plt.title("Accuracy vs Graph Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_graph_size.png", dpi=300)
    plt.close()


def plot_model_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = summary_df.pivot(index="num_nodes", columns="experiment", values="mean_last_test_acc")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title("Per-Model Comparison by Graph Size")
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_by_size.png", dpi=300)
    plt.close()


def plot_training_curves(runs_root: Path, summary_df: pd.DataFrame, out_dir: Path) -> None:
    style_map = {
        "baseline": {"linewidth": 3.0, "linestyle": "--", "marker": "s", "zorder": 5},
        "gcn": {"linewidth": 2.0, "linestyle": "-", "marker": "o", "zorder": 3},
        "graph_transformer": {
            "linewidth": 2.0,
            "linestyle": "-",
            "marker": "^",
            "zorder": 4,
        },
        "graphormer": {"linewidth": 2.0, "linestyle": "-", "marker": "D", "zorder": 4},
    }
    for graph_tag in sorted(summary_df["graph_tag"].dropna().unique()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        any_data = False
        ordered = ["gcn", "graph_transformer", "graphormer", "baseline"]
        for experiment in ordered:
            curves = []
            run_paths = summary_df.loc[
                (summary_df["graph_tag"] == graph_tag) & (summary_df["experiment"] == experiment),
                "run_path",
            ].dropna()
            for run_path in run_paths:
                metrics_path = Path(run_path) / "metrics.csv"
                if not metrics_path.exists():
                    continue
                metrics = pd.read_csv(metrics_path)
                loss_col = None
                if {"epoch", "avg_train_loss", "test_acc"}.issubset(metrics.columns):
                    loss_col = "avg_train_loss"
                elif {"epoch", "train_loss", "test_acc"}.issubset(metrics.columns):
                    loss_col = "train_loss"
                if loss_col is not None:
                    curves.append(
                        metrics[["epoch", loss_col, "test_acc"]].rename(
                            columns={loss_col: "avg_train_loss"}
                        )
                    )
            if not curves:
                continue
            any_data = True
            merged = curves[0].rename(
                columns={
                    "avg_train_loss": "avg_train_loss_0",
                    "test_acc": "test_acc_0",
                }
            )
            for idx, curve in enumerate(curves[1:], start=1):
                merged = merged.merge(
                    curve.rename(
                        columns={
                            "avg_train_loss": f"avg_train_loss_{idx}",
                            "test_acc": f"test_acc_{idx}",
                        }
                    ),
                    on="epoch",
                    how="inner",
                )
            loss_cols = [c for c in merged.columns if c.startswith("avg_train_loss_")]
            acc_cols = [c for c in merged.columns if c.startswith("test_acc_")]
            merged["mean_loss"] = merged[loss_cols].mean(axis=1)
            merged["mean_acc"] = merged[acc_cols].mean(axis=1)
            style = style_map.get(
                experiment,
                {"linewidth": 2.0, "linestyle": "-", "marker": "o", "zorder": 3},
            )
            axes[0].plot(
                merged["epoch"],
                merged["mean_loss"],
                label=experiment,
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(merged) // 6),
                zorder=style["zorder"],
                color=style.get("color"),
            )
            axes[1].plot(
                merged["epoch"],
                merged["mean_acc"],
                label=experiment,
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
        loss_values = []
        acc_values = []
        for line in axes[0].lines:
            loss_values.extend(line.get_ydata())
        for line in axes[1].lines:
            acc_values.extend(line.get_ydata())
        if loss_values:
            ymin = min(loss_values)
            ymax = max(loss_values)
            pad = max(0.005, (ymax - ymin) * 0.15)
            axes[0].set_ylim(ymin - pad, ymax + pad)
        if acc_values:
            ymin = min(acc_values)
            ymax = max(acc_values)
            pad = max(0.005, (ymax - ymin) * 0.15)
            axes[1].set_ylim(ymin - pad, ymax + pad)
        axes[0].set_title(f"{graph_tag} Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Average Train Loss")
        axes[1].set_title(f"{graph_tag} Test Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Test Accuracy")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"training_curves_{graph_tag}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", default="runs/results_summary.csv")
    parser.add_argument("--out_dir", default="runs/figures")
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)
    summary_df = save_model_size_summary(df, out_dir)
    plot_accuracy_vs_size(summary_df, out_dir)
    plot_model_comparison(summary_df, out_dir)
    plot_training_curves(Path("runs"), df, out_dir)
    print(f"Wrote figures and summary tables to: {out_dir}")


if __name__ == "__main__":
    main()
