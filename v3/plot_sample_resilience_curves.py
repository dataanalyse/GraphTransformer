from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd


def sample_scenarios(summary_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    unique_runs = summary_df[["scenario_id", "condition"]].drop_duplicates().copy()
    pairs = list(unique_runs.itertuples(index=False, name=None))
    if len(pairs) <= n:
        chosen = pairs
    else:
        chosen = rng.sample(pairs, n)
    chosen_df = pd.DataFrame(chosen, columns=["scenario_id", "condition"])
    sampled = summary_df.merge(chosen_df, on=["scenario_id", "condition"], how="inner")
    return sampled.sort_values(["condition", "shock_type", "scenario_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random sample resilience curves from scenario runs.")
    parser.add_argument(
        "--time-series",
        default="v3/data/processed/redundancy_v3/scenario_time_series_v3.csv",
        help="Scenario time-series CSV.",
    )
    parser.add_argument(
        "--summary",
        default="v3/data/processed/redundancy_v3/scenario_summary_metrics_v3.csv",
        help="Scenario summary CSV.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=12,
        help="Number of random scenarios to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        default="v3/runs/figures/redundancy_v3/random_sample_lcc_curves.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--sample-out",
        default="v3/data/processed/redundancy_v3/random_sample_scenarios.csv",
        help="Output CSV listing sampled scenarios.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_df = pd.read_csv(args.time_series)
    summary_df = pd.read_csv(args.summary)
    sampled = sample_scenarios(summary_df, args.n, args.seed)
    sampled_keys = sampled[["scenario_id", "condition"]].drop_duplicates().copy()
    sampled_keys["sample_key"] = sampled_keys["condition"] + " | " + sampled_keys["scenario_id"]

    sampled_time = time_df.merge(sampled_keys, on=["scenario_id", "condition"], how="inner")
    sampled_time = sampled_time.merge(
        sampled[["scenario_id", "condition", "shock_type", "min_lcc", "auc_resilience"]].drop_duplicates(),
        on=["scenario_id", "condition", "shock_type"],
        how="left",
        suffixes=("", "_summary"),
    )

    colors = {
        "baseline_additive": "#e74c3c",
        "redundant_additive": "#9b59b6",
        "redundant_buffer": "#27ae60",
        "dampened_power_additive": "#3498db",
    }

    sample_keys = sampled_keys["sample_key"].tolist()
    n = len(sample_keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, sample_key in zip(axes, sample_keys):
        sub = sampled_time[sampled_time["sample_key"] == sample_key].sort_values("timestep")
        if sub.empty:
            ax.axis("off")
            continue
        scenario_id = sub["scenario_id"].iloc[0]
        condition = sub["condition"].iloc[0]
        shock_type = sub["shock_type"].iloc[0]
        min_lcc = sub["min_lcc"].iloc[0]
        auc = sub["auc_resilience"].iloc[0]
        ax.plot(
            sub["timestep"],
            sub["lcc_fraction"],
            color=colors.get(condition, "#34495e"),
            linewidth=2,
        )
        ax.set_title(
            f"{scenario_id}\n{condition} | {shock_type}\nminLCC={min_lcc:.3f}, AUC={auc:.3f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.02)

    for ax in axes[len(sample_keys):]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel("Timestep")
    for idx in range(0, len(axes), ncols):
        axes[idx].set_ylabel("LCC Fraction")

    fig.suptitle("Random Sample of Resilience Curves (LCC Over Time)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    sample_out_path = Path(args.sample_out)
    sample_out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled[
        [
            "scenario_id",
            "condition",
            "shock_type",
            "seed_k",
            "mean_seed_degree",
            "dependency_concentration",
            "min_lcc",
            "auc_resilience",
            "peak_damage_nodes",
            "recovery_time",
        ]
    ].drop_duplicates().to_csv(sample_out_path, index=False)

    print(f"Wrote random sample plot to {output_path}")
    print(f"Wrote sampled scenario list to {sample_out_path}")


if __name__ == "__main__":
    main()
