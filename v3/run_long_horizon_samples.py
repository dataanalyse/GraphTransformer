from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import pandas as pd

try:
    import v3.run_scenarios_v1 as scenario_utils
    import v3.simulator_v3 as sim3
except ModuleNotFoundError:
    import run_scenarios_v1 as scenario_utils
    import simulator_v3 as sim3


CONDITION_TO_EDGE_FILE = {
    "baseline_additive": "baseline_edges.csv",
    "redundant_additive": "redundant_edges.csv",
    "redundant_buffer": "redundant_edges.csv",
    "dampened_power_additive": "dampened_power_edges.csv",
}


def run_one(
    scenario_meta: dict,
    condition: str,
    nodes_path: Path,
    edges_path: Path,
    sim_edges_path: Path,
    timesteps: int,
) -> tuple[pd.DataFrame, dict]:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    sim_edges = sim3.ensure_simulation_edges(str(nodes_path), str(edges_path), str(sim_edges_path))
    structural_graph = sim3.build_structural_graph(nodes, edges)

    original_params = copy.deepcopy(sim3.SECTOR_PARAMS)
    try:
        sim3.SECTOR_PARAMS = scenario_utils.apply_parameter_overrides(
            scenario_meta["shock_severity_scale"],
            scenario_meta["recovery_scale"],
            scenario_meta["propagation_scale"],
        )
        seeds = scenario_meta["seed_nodes"].split(";") if scenario_meta["seed_nodes"] else []
        metrics_df, info = sim3.run_cascade_v3(
            nodes,
            sim_edges,
            structural_graph,
            scenario_meta["scenario_id"],
            seeds,
            timesteps,
            aggregation_mode="redundancy_buffer" if condition == "redundant_buffer" else "additive_exposure",
            redundancy_threshold=0.5,
        )
    finally:
        sim3.SECTOR_PARAMS = original_params

    metrics_df.insert(0, "condition", condition)
    metrics_df.insert(1, "scenario_id", scenario_meta["scenario_id"])
    metrics_df.insert(2, "shock_type", scenario_meta["shock_type"])

    info_row = {
        "scenario_id": scenario_meta["scenario_id"],
        "condition": condition,
        "shock_type": scenario_meta["shock_type"],
        "timesteps": timesteps,
        "final_lcc": float(metrics_df["lcc_fraction"].iloc[-1]),
        "min_lcc": float(metrics_df["lcc_fraction"].min()),
        "min_lcc_timestep": int(metrics_df.loc[metrics_df["lcc_fraction"].idxmin(), "timestep"]),
        "final_failed_nodes": int(metrics_df["failed_nodes"].iloc[-1]),
        "final_degraded_nodes": int(metrics_df["degraded_nodes"].iloc[-1]),
        "peak_failed_nodes": int(metrics_df["failed_nodes"].max()),
        "peak_degraded_nodes": int(metrics_df["degraded_nodes"].max()),
        "auc_resilience": float(metrics_df["lcc_fraction"].mean()),
    }
    return metrics_df, info_row


def plot_results(time_df: pd.DataFrame, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline_additive": "#e74c3c",
        "redundant_additive": "#9b59b6",
        "redundant_buffer": "#27ae60",
        "dampened_power_additive": "#3498db",
    }

    scenario_ids = list(dict.fromkeys(time_df["scenario_id"].tolist()))
    fig, axes = plt.subplots(len(scenario_ids), 2, figsize=(14, 5 * len(scenario_ids)), sharex=True)
    if len(scenario_ids) == 1:
        axes = [axes]

    for row_axes, scenario_id in zip(axes, scenario_ids):
        sub = time_df[time_df["scenario_id"] == scenario_id].copy()
        for condition, cond_df in sub.groupby("condition"):
            cond_df = cond_df.sort_values("timestep")
            row_axes[0].plot(
                cond_df["timestep"],
                cond_df["lcc_fraction"],
                label=condition,
                color=colors.get(condition, "#555555"),
                linewidth=2,
            )
            row_axes[1].plot(
                cond_df["timestep"],
                cond_df["failed_nodes"],
                label=condition,
                color=colors.get(condition, "#555555"),
                linewidth=2,
            )
        shock_type = sub["shock_type"].iloc[0]
        row_axes[0].set_title(f"{scenario_id} | {shock_type} | LCC to T={int(sub['timestep'].max())}")
        row_axes[1].set_title(f"{scenario_id} | {shock_type} | Failed Nodes to T={int(sub['timestep'].max())}")
        row_axes[0].set_ylabel("LCC Fraction")
        row_axes[1].set_ylabel("Failed Nodes")
        row_axes[0].grid(alpha=0.3)
        row_axes[1].grid(alpha=0.3)
        row_axes[0].legend()

    axes[-1][0].set_xlabel("Timestep")
    axes[-1][1].set_xlabel("Timestep")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected long-horizon sample scenarios.")
    parser.add_argument("--scenario-ids", default="v3_041,v3_126")
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--metadata", default="v3/data/processed/redundancy_v3/scenario_metadata_v3.csv")
    parser.add_argument("--graph-dir", default="v3/data/processed/graph_variants")
    parser.add_argument("--output-dir", default="v3/data/processed/long_horizon_samples")
    parser.add_argument("--figures-dir", default="v3/runs/figures/long_horizon_samples")
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata)
    scenario_ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
    selected = metadata_df[metadata_df["scenario_id"].isin(scenario_ids)].copy()
    if selected.empty:
        raise ValueError("No matching scenario IDs found.")

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_time = []
    all_summary = []
    nodes_path = graph_dir / "dependency_graph_nodes.csv"
    for meta in selected.to_dict("records"):
        for condition, edge_file in CONDITION_TO_EDGE_FILE.items():
            metrics_df, summary_row = run_one(
                meta,
                condition,
                nodes_path,
                graph_dir / edge_file,
                graph_dir / f"{condition}_long_horizon_sim_edges.csv",
                args.timesteps,
            )
            all_time.append(metrics_df)
            all_summary.append(summary_row)

    time_df = pd.concat(all_time, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    time_path = output_dir / "long_horizon_time_series.csv"
    summary_path = output_dir / "long_horizon_summary.csv"
    plot_path = figures_dir / "long_horizon_power_samples.png"
    time_df.to_csv(time_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_results(time_df, plot_path)

    print(f"Wrote time series to {time_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote figure to {plot_path}")


if __name__ == "__main__":
    main()
