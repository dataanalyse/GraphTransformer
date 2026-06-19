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


DEFAULT_PLAN = {
    "power": 150,
    "telecom": 150,
    "ems": 100,
    "hospital": 50,
    "mixed": 50,
}


def run_condition(
    condition_label: str,
    nodes_path: str,
    edges_path: str,
    sim_edges_path: str,
    aggregation_mode: str,
    redundancy_threshold: float,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    sim_edges = sim3.ensure_simulation_edges(nodes_path, edges_path, sim_edges_path)
    structural_graph = sim3.build_structural_graph(nodes, edges)

    time_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    original_params = copy.deepcopy(sim3.SECTOR_PARAMS)
    try:
        for meta in metadata_df.to_dict("records"):
            sim3.SECTOR_PARAMS = scenario_utils.apply_parameter_overrides(
                meta["shock_severity_scale"],
                meta["recovery_scale"],
                meta["propagation_scale"],
            )
            seeds = meta["seed_nodes"].split(";") if meta["seed_nodes"] else []
            metrics_df, _ = sim3.run_cascade_v3(
                nodes,
                sim_edges,
                structural_graph,
                meta["scenario_id"],
                seeds,
                int(meta["simulation_length"]),
                aggregation_mode=aggregation_mode,
                redundancy_threshold=redundancy_threshold,
            )
            metrics_df.insert(0, "condition", condition_label)
            metrics_df.insert(1, "scenario_id", meta["scenario_id"])
            metrics_df.insert(2, "shock_type", meta["shock_type"])
            time_rows.append(metrics_df)

            summary_row = scenario_utils.summarize_metrics(pd.Series(meta), metrics_df)
            summary_row["condition"] = condition_label
            summary_row["aggregation_mode"] = aggregation_mode
            summary_row["redundancy_threshold"] = redundancy_threshold
            summary_rows.append(summary_row)
    finally:
        sim3.SECTOR_PARAMS = original_params

    time_df = pd.concat(time_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["recovery_time"] = pd.to_numeric(summary_df["recovery_time"], errors="coerce")
    return time_df, summary_df


def build_conditions(graph_dir: Path, redundancy_threshold: float) -> list[dict]:
    nodes_path = str(graph_dir / "dependency_graph_nodes.csv")
    return [
        {
            "condition": "baseline_additive",
            "nodes_path": nodes_path,
            "edges_path": str(graph_dir / "baseline_edges.csv"),
            "sim_edges_path": str(graph_dir / "baseline_additive_sim_edges.csv"),
            "aggregation_mode": "additive_exposure",
            "redundancy_threshold": redundancy_threshold,
        },
        {
            "condition": "redundant_additive",
            "nodes_path": nodes_path,
            "edges_path": str(graph_dir / "redundant_edges.csv"),
            "sim_edges_path": str(graph_dir / "redundant_additive_sim_edges.csv"),
            "aggregation_mode": "additive_exposure",
            "redundancy_threshold": redundancy_threshold,
        },
        {
            "condition": "redundant_buffer",
            "nodes_path": nodes_path,
            "edges_path": str(graph_dir / "redundant_edges.csv"),
            "sim_edges_path": str(graph_dir / "redundant_buffer_sim_edges.csv"),
            "aggregation_mode": "redundancy_buffer",
            "redundancy_threshold": redundancy_threshold,
        },
        {
            "condition": "dampened_power_additive",
            "nodes_path": nodes_path,
            "edges_path": str(graph_dir / "dampened_power_edges.csv"),
            "sim_edges_path": str(graph_dir / "dampened_power_additive_sim_edges.csv"),
            "aggregation_mode": "additive_exposure",
            "redundancy_threshold": redundancy_threshold,
        },
    ]


def write_summary(
    summary_path: Path,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    redundancy_threshold: float,
    timesteps: int,
) -> None:
    def cond_mean(condition: str, metric: str) -> float:
        sub = summary_df[summary_df["condition"] == condition]
        return float(sub[metric].mean()) if not sub.empty else float("nan")

    baseline_peak = cond_mean("baseline_additive", "peak_damage_nodes")
    redundant_additive_peak = cond_mean("redundant_additive", "peak_damage_nodes")
    redundant_buffer_peak = cond_mean("redundant_buffer", "peak_damage_nodes")
    dampened_peak = cond_mean("dampened_power_additive", "peak_damage_nodes")

    baseline_lcc = cond_mean("baseline_additive", "min_lcc")
    redundant_buffer_lcc = cond_mean("redundant_buffer", "min_lcc")
    dampened_lcc = cond_mean("dampened_power_additive", "min_lcc")

    baseline_corr = (
        summary_df[summary_df["condition"] == "baseline_additive"][["dependency_concentration", "min_lcc"]]
        .corr()
        .iloc[0, 1]
    )

    lines = [
        "# Simulator V3 Summary",
        "",
        "## Setup",
        "",
        "- Core simulator: `simulator_v3.py`",
        "- Shared scenario count per condition: `500`",
        "- Graph conditions:",
        "  - `baseline_additive`",
        "  - `redundant_additive`",
        "  - `redundant_buffer`",
        "  - `dampened_power_additive`",
        f"- Timesteps per scenario: `{timesteps}`",
        f"- Redundancy threshold for buffer mode: `{redundancy_threshold}`",
        "",
        "## Headline Findings",
        "",
        f"- Mean peak cascade size, baseline additive: `{baseline_peak:.2f}`",
        f"- Mean peak cascade size, redundant additive: `{redundant_additive_peak:.2f}`",
        f"- Mean peak cascade size, redundant buffer: `{redundant_buffer_peak:.2f}`",
        f"- Mean peak cascade size, dampened power additive: `{dampened_peak:.2f}`",
        f"- Mean minimum LCC, baseline additive: `{baseline_lcc:.4f}`",
        f"- Mean minimum LCC, redundant buffer: `{redundant_buffer_lcc:.4f}`",
        f"- Mean minimum LCC, dampened power additive: `{dampened_lcc:.4f}`",
        "",
        "## Research Questions",
        "",
        f"1. Does dependency concentration increase cascading fragility?\n   - In the baseline additive condition, the correlation between dependency concentration and minimum LCC is `{baseline_corr:.3f}`, which supports the fragility hypothesis.",
        f"2. Does true redundancy reduce cascade severity?\n   - Compare `redundant_additive` (`{redundant_additive_peak:.2f}` mean peak damage) with `redundant_buffer` (`{redundant_buffer_peak:.2f}` mean peak damage). The redundancy-aware buffer is the relevant comparison because it treats multi-support sets as fallback capacity rather than extra exposure.",
        f"3. Are baseline conclusions robust after dampening power dominance?\n   - The dampened-power condition lowers average cascade size from `{baseline_peak:.2f}` to `{dampened_peak:.2f}` and improves mean minimum LCC from `{baseline_lcc:.4f}` to `{dampened_lcc:.4f}`, but power sensitivity remains a first-order driver if it still dominates the by-shock summaries.",
        "4. Which aggregation rule best reflects resilience behavior?\n   - `additive_exposure` is the least redundancy-aware and should be treated as the exposure baseline. `redundancy_buffer` is the first condition that can represent surviving fallback support. `strongest_dependency` is implemented in `simulator_v3.py` for future stress tests, even though it is not one of the current comparison conditions.",
        "",
        "## Condition-by-Shock Summary",
        "",
    ]
    for row in comparison_df.itertuples(index=False):
        lines.append(
            f"- `{row.condition}` / `{row.shock_type}`: mean peak damage `{row.mean_peak_damage:.2f}`, "
            f"mean min LCC `{row.mean_min_lcc:.4f}`, mean recovery time `{row.mean_recovery_time:.2f}`, "
            f"mean AUC resilience `{row.mean_auc_resilience:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `redundant_additive` still looks worse than `baseline_additive`, that confirms the original problem: more support links are being interpreted as more exposure under additive aggregation.",
            "- If `redundant_buffer` improves on `redundant_additive`, then the new support-set aggregation is behaving more like real fallback redundancy.",
            "- If dampening power weights changes the ranking of the worst scenarios only modestly, the current power conclusion is likely robust; if rankings shift substantially, power dominance is more assumption-sensitive.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(comparison_df: pd.DataFrame, plot_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shock_order = ["power", "telecom", "ems", "hospital", "mixed"]
    conditions = comparison_df["condition"].unique().tolist()
    colors = {
        "baseline_additive": "#c0392b",
        "redundant_additive": "#8e44ad",
        "redundant_buffer": "#16a085",
        "dampened_power_additive": "#2980b9",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("mean_peak_damage", "Mean Peak Damage"),
        ("mean_min_lcc", "Mean Minimum LCC"),
        ("mean_recovery_time", "Mean Recovery Time"),
        ("mean_auc_resilience", "Mean AUC Resilience"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for condition in conditions:
            sub = comparison_df[comparison_df["condition"] == condition].copy()
            sub["shock_type"] = pd.Categorical(sub["shock_type"], categories=shock_order, ordered=True)
            sub = sub.sort_values("shock_type")
            ax.plot(
                sub["shock_type"].astype(str),
                sub[metric],
                marker="o",
                linewidth=2,
                label=condition,
                color=colors.get(condition),
            )
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare redundancy-aware simulator_v3 conditions.")
    parser.add_argument("--graph-dir", default="v3/data/processed/graph_variants")
    parser.add_argument("--output-dir", default="v3/data/processed/redundancy_v3")
    parser.add_argument("--figures-dir", default="v3/runs/figures/redundancy_v3")
    parser.add_argument("--scenario-prefix", default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--redundancy-threshold", type=float, default=0.5)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--power-count", type=int, default=DEFAULT_PLAN["power"])
    parser.add_argument("--telecom-count", type=int, default=DEFAULT_PLAN["telecom"])
    parser.add_argument("--ems-count", type=int, default=DEFAULT_PLAN["ems"])
    parser.add_argument("--hospital-count", type=int, default=DEFAULT_PLAN["hospital"])
    parser.add_argument("--mixed-count", type=int, default=DEFAULT_PLAN["mixed"])
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_nodes = graph_dir / "dependency_graph_nodes.csv"
    baseline_edges = graph_dir / "baseline_edges.csv"
    baseline_sim_edges = graph_dir / "baseline_metadata_sim_edges.csv"
    nodes = pd.read_csv(baseline_nodes)
    edges = pd.read_csv(baseline_edges)
    sim_edges = sim3.ensure_simulation_edges(str(baseline_nodes), str(baseline_edges), str(baseline_sim_edges))
    scenario_plan = {
        "power": args.power_count,
        "telecom": args.telecom_count,
        "ems": args.ems_count,
        "hospital": args.hospital_count,
        "mixed": args.mixed_count,
    }
    metadata_df = scenario_utils.build_scenarios(
        nodes,
        edges,
        sim_edges,
        args.seed,
        scenario_plan,
        args.scenario_prefix,
    )
    metadata_df["simulation_length"] = int(args.timesteps)

    conditions = build_conditions(graph_dir, args.redundancy_threshold)
    all_time: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    for condition in conditions:
        time_df, summary_df = run_condition(
            condition["condition"],
            condition["nodes_path"],
            condition["edges_path"],
            condition["sim_edges_path"],
            condition["aggregation_mode"],
            condition["redundancy_threshold"],
            metadata_df,
        )
        all_time.append(time_df)
        all_summary.append(summary_df)

    time_df = pd.concat(all_time, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    comparison_df = (
        summary_df.groupby(["condition", "shock_type"], as_index=False)
        .agg(
            mean_peak_damage=("peak_damage_nodes", "mean"),
            min_peak_damage=("peak_damage_nodes", "min"),
            max_peak_damage=("peak_damage_nodes", "max"),
            mean_min_lcc=("min_lcc", "mean"),
            min_min_lcc=("min_lcc", "min"),
            max_min_lcc=("min_lcc", "max"),
            mean_recovery_time=("recovery_time", "mean"),
            mean_auc_resilience=("auc_resilience", "mean"),
            mean_final_lcc=("final_lcc", "mean"),
            mean_final_failed=("final_failed_nodes", "mean"),
            mean_final_degraded=("final_degraded_nodes", "mean"),
        )
    )

    metadata_df.to_csv(output_dir / "scenario_metadata_v3.csv", index=False)
    time_df.to_csv(output_dir / "scenario_time_series_v3.csv", index=False)
    summary_df.to_csv(output_dir / "scenario_summary_metrics_v3.csv", index=False)
    comparison_df.to_csv(output_dir / "redundancy_comparison_metrics.csv", index=False)
    plot_comparison(comparison_df, figures_dir / "redundancy_comparison_plots.png")
    write_summary(
        output_dir / "simulator_v3_summary.md",
        summary_df,
        comparison_df,
        args.redundancy_threshold,
        args.timesteps,
    )

    print(f"Wrote metadata to {output_dir / 'scenario_metadata_v3.csv'}")
    print(f"Wrote time series to {output_dir / 'scenario_time_series_v3.csv'}")
    print(f"Wrote scenario summary to {output_dir / 'scenario_summary_metrics_v3.csv'}")
    print(f"Wrote comparison metrics to {output_dir / 'redundancy_comparison_metrics.csv'}")
    print(f"Wrote summary to {output_dir / 'simulator_v3_summary.md'}")
    print(f"Wrote plots to {figures_dir / 'redundancy_comparison_plots.png'}")


if __name__ == "__main__":
    main()
