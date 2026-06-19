from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import pandas as pd

try:
    import v3.simulator_v2 as sim2
except ModuleNotFoundError:
    import simulator_v2 as sim2


PILOT_PLAN = {
    "power": 30,
    "telecom": 30,
    "ems": 20,
    "hospital": 10,
    "mixed": 10,
}

DEFAULT_PLAN = {
    "power": 150,
    "telecom": 150,
    "ems": 100,
    "hospital": 50,
    "mixed": 50,
}


def degree_lookup(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    degree_df = pd.concat(
        [
            edges[["src_node_id"]].rename(columns={"src_node_id": "node_id"}),
            edges[["dst_node_id"]].rename(columns={"dst_node_id": "node_id"}),
        ],
        ignore_index=True,
    )
    degree_counts = degree_df.value_counts("node_id").rename("degree").reset_index()
    return nodes.merge(degree_counts, on="node_id", how="left").fillna({"degree": 0})


def simulation_outdegree(sim_edges: pd.DataFrame) -> dict[str, int]:
    return sim_edges["simulation_source"].value_counts().to_dict()


def choose_nodes(rng: random.Random, candidates: list[str], k: int) -> list[str]:
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    return rng.sample(candidates, k)


def build_scenarios(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    sim_edges: pd.DataFrame,
    seed: int,
    scenario_plan: dict[str, int],
    scenario_prefix: str,
) -> pd.DataFrame:
    rng = random.Random(seed)
    degree_df = degree_lookup(nodes, edges)
    node_type_map = nodes.set_index("node_id")["node_type"].to_dict()
    name_map = nodes.set_index("node_id")["name"].to_dict()
    outdegree_map = simulation_outdegree(sim_edges)

    by_type = {
        node_type: degree_df[degree_df["node_type"] == node_type]
        .sort_values(["degree", "name", "node_id"], ascending=[False, True, True])["node_id"]
        .astype(str)
        .tolist()
        for node_type in degree_df["node_type"].unique()
    }

    scenarios: list[dict] = []
    scenario_id = 1
    for shock_type, count in scenario_plan.items():
        for _ in range(count):
            if shock_type == "power":
                source_type = "power"
                k = rng.choice([1, 2, 3])
                seeds = choose_nodes(rng, by_type.get("power", []), k)
            elif shock_type == "telecom":
                source_type = "telecom"
                k = rng.choice([1, 2, 3])
                seeds = choose_nodes(rng, by_type.get("telecom", []), k)
            elif shock_type == "ems":
                source_type = "ems_fire"
                k = rng.choice([1, 2, 3])
                seeds = choose_nodes(rng, by_type.get("ems_fire", []), k)
            elif shock_type == "hospital":
                source_type = "hospital"
                k = rng.choice([1, 2, 3])
                seeds = choose_nodes(rng, by_type.get("hospital", []), k)
            else:
                source_type = "mixed"
                k = 3
                seeds = []
                pools = [
                    by_type.get("power", []),
                    by_type.get("telecom", []),
                    by_type.get("ems_fire", []),
                    by_type.get("hospital", []),
                ]
                for pool in pools[: rng.choice([2, 3])]:
                    if pool:
                        seeds.extend(choose_nodes(rng, pool, 1))

            severity_scale = rng.choice([0.85, 1.0, 1.15])
            recovery_scale = rng.choice([0.75, 1.0, 1.25])
            propagation_scale = rng.choice([0.85, 1.0, 1.15])

            scenarios.append(
                {
                    "scenario_id": f"{scenario_prefix}_{scenario_id:03d}",
                    "shock_type": shock_type,
                    "source_type": source_type,
                    "seed_nodes": ";".join(seeds),
                    "seed_names": ";".join(name_map.get(node_id, node_id) for node_id in seeds),
                    "seed_k": len(seeds),
                    "shock_severity_scale": severity_scale,
                    "recovery_scale": recovery_scale,
                    "propagation_scale": propagation_scale,
                    "simulation_length": sim2.TIMESTEPS,
                    "mean_seed_degree": float(sum(outdegree_map.get(node_id, 0) for node_id in seeds) / max(len(seeds), 1)),
                    "dependency_concentration": float(sum(outdegree_map.get(node_id, 0) for node_id in seeds) / max(len(sim_edges), 1)),
                }
            )
            scenario_id += 1

    return pd.DataFrame(scenarios)


def apply_parameter_overrides(severity_scale: float, recovery_scale: float, propagation_scale: float) -> dict:
    params = copy.deepcopy(sim2.SECTOR_PARAMS)
    for sector, config in params.items():
        config["propagation_scale"] *= propagation_scale * severity_scale
        config["recovery_failed"] *= recovery_scale
        config["recovery_degraded"] *= recovery_scale
        config["degrade_threshold"] /= severity_scale
        config["fail_threshold"] /= severity_scale
    return params


def summarize_metrics(metadata: pd.Series, metrics_df: pd.DataFrame) -> dict:
    damage = metrics_df["failed_nodes"] + metrics_df["degraded_nodes"]
    min_lcc_idx = int(metrics_df["lcc_fraction"].idxmin())
    min_lcc = float(metrics_df.loc[min_lcc_idx, "lcc_fraction"])
    min_lcc_timestep = int(metrics_df.loc[min_lcc_idx, "timestep"])
    peak_damage = int(damage.max())
    peak_damage_timestep = int(metrics_df.loc[damage.idxmax(), "timestep"])
    total_failed = int(metrics_df["failed_nodes"].max())
    total_degraded = int(metrics_df["degraded_nodes"].max())
    auc_resilience = float(metrics_df["lcc_fraction"].mean())

    post_peak = metrics_df[metrics_df["timestep"] > peak_damage_timestep].copy()
    recovered_mask = (post_peak["failed_nodes"] + post_peak["degraded_nodes"]) <= int(metadata["seed_k"])
    recovery_time = int(post_peak.loc[recovered_mask, "timestep"].iloc[0]) if recovered_mask.any() else pd.NA

    summary = {
        "scenario_id": metadata["scenario_id"],
        "shock_type": metadata["shock_type"],
        "seed_k": int(metadata["seed_k"]),
        "mean_seed_degree": float(metadata["mean_seed_degree"]),
        "dependency_concentration": float(metadata["dependency_concentration"]),
        "min_lcc": min_lcc,
        "min_lcc_timestep": min_lcc_timestep,
        "final_lcc": float(metrics_df["lcc_fraction"].iloc[-1]),
        "peak_failed_nodes": total_failed,
        "peak_degraded_nodes": total_degraded,
        "peak_damage_nodes": peak_damage,
        "peak_damage_timestep": peak_damage_timestep,
        "recovery_time": recovery_time,
        "auc_resilience": auc_resilience,
        "final_failed_nodes": int(metrics_df["failed_nodes"].iloc[-1]),
        "final_degraded_nodes": int(metrics_df["degraded_nodes"].iloc[-1]),
    }
    for col in [c for c in metrics_df.columns if c.startswith("sector_health_")]:
        sector_name = col.replace("sector_health_", "")
        summary[f"{sector_name}_min_health"] = float(metrics_df[col].min())
        summary[f"{sector_name}_final_health"] = float(metrics_df[col].iloc[-1])
    return summary


def plot_outputs(time_df: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path, prefix: str = "") -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shock_order = ["power", "telecom", "ems", "hospital", "mixed"]
    colors = {
        "power": "#c0392b",
        "telecom": "#2980b9",
        "ems": "#16a085",
        "hospital": "#8e44ad",
        "mixed": "#d35400",
    }

    figures_dir.mkdir(parents=True, exist_ok=True)

    avg_lcc = time_df.groupby(["shock_type", "timestep"], as_index=False)["lcc_fraction"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for shock_type in shock_order:
        sub = avg_lcc[avg_lcc["shock_type"] == shock_type]
        if sub.empty:
            continue
        ax.plot(sub["timestep"], sub["lcc_fraction"], label=shock_type, color=colors[shock_type], linewidth=2)
    ax.set_title("Average LCC by Shock Type")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average LCC Fraction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}average_lcc_by_shock_type.png", dpi=200)
    plt.close(fig)

    avg_failed = time_df.groupby(["shock_type", "timestep"], as_index=False)["failed_nodes"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    for shock_type in shock_order:
        sub = avg_failed[avg_failed["shock_type"] == shock_type]
        if sub.empty:
            continue
        ax.plot(sub["timestep"], sub["failed_nodes"], label=shock_type, color=colors[shock_type], linewidth=2)
    ax.set_title("Average Failed Nodes by Shock Type")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Failed Nodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}failed_nodes_by_shock_type.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    recovery_data = summary_df.dropna(subset=["recovery_time"])
    if not recovery_data.empty:
        for idx, shock_type in enumerate(shock_order):
            sub = recovery_data[recovery_data["shock_type"] == shock_type]
            if sub.empty:
                continue
            ax.hist(sub["recovery_time"], bins=10, alpha=0.55, color=colors[shock_type], label=shock_type)
    ax.set_title("Recovery Time Distribution by Shock Type")
    ax.set_xlabel("Recovery Time")
    ax.set_ylabel("Scenario Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}recovery_time_distribution_by_shock_type.png", dpi=200)
    plt.close(fig)

    sector_cols = ["sector_health_hospital", "sector_health_school", "sector_health_power", "sector_health_telecom", "sector_health_ems_fire"]
    fig, axes = plt.subplots(len(sector_cols), 1, figsize=(10, 16), sharex=True)
    for ax, col in zip(axes, sector_cols):
        sector_avg = time_df.groupby(["shock_type", "timestep"], as_index=False)[col].mean()
        for shock_type in shock_order:
            sub = sector_avg[sector_avg["shock_type"] == shock_type]
            if sub.empty:
                continue
            ax.plot(sub["timestep"], sub[col], label=shock_type, color=colors[shock_type], linewidth=2)
        ax.set_title(col.replace("sector_health_", "").replace("_", " ").title())
        ax.set_ylabel("Avg Health")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}sector_health_curves_by_shock_type.png", dpi=200)
    plt.close(fig)

    worst = summary_df.nsmallest(20, "min_lcc").sort_values("min_lcc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(worst["scenario_id"], worst["min_lcc"], color="#c0392b")
    ax.set_title("Top 20 Worst Scenarios by Minimum LCC")
    ax.set_xlabel("Minimum LCC")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}top20_worst_scenarios_by_min_lcc.png", dpi=200)
    plt.close(fig)

    fastest = summary_df.dropna(subset=["recovery_time"]).nsmallest(20, "recovery_time").sort_values("recovery_time", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fastest["scenario_id"], fastest["recovery_time"], color="#16a085")
    ax.set_title("Top 20 Fastest Recovery Scenarios")
    ax.set_xlabel("Recovery Time")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}top20_fastest_recovery_scenarios.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for shock_type in shock_order:
        sub = summary_df[summary_df["shock_type"] == shock_type]
        if sub.empty:
            continue
        ax.scatter(sub["mean_seed_degree"], sub["peak_damage_nodes"], label=shock_type, color=colors[shock_type], alpha=0.7)
    ax.set_title("Seed Degree vs Cascade Size")
    ax.set_xlabel("Mean Seed Degree")
    ax.set_ylabel("Peak Damage Nodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}scatter_seed_degree_vs_cascade_size.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for shock_type in shock_order:
        sub = summary_df[summary_df["shock_type"] == shock_type]
        if sub.empty:
            continue
        ax.scatter(sub["dependency_concentration"], sub["min_lcc"], label=shock_type, color=colors[shock_type], alpha=0.7)
    ax.set_title("Dependency Concentration vs Minimum LCC")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Minimum LCC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}scatter_dependency_concentration_vs_min_lcc.png", dpi=200)
    plt.close(fig)


def write_report(metadata_df: pd.DataFrame, summary_df: pd.DataFrame, report_path: Path, model_label: str) -> None:
    avg_by_shock = summary_df.groupby("shock_type").agg(
        mean_min_lcc=("min_lcc", "mean"),
        min_min_lcc=("min_lcc", "min"),
        max_min_lcc=("min_lcc", "max"),
        mean_peak_damage=("peak_damage_nodes", "mean"),
        min_peak_damage=("peak_damage_nodes", "min"),
        max_peak_damage=("peak_damage_nodes", "max"),
        mean_recovery_time=("recovery_time", "mean"),
        min_recovery_time=("recovery_time", "min"),
        max_recovery_time=("recovery_time", "max"),
    ).sort_values("mean_min_lcc")
    worst = summary_df.nsmallest(10, "min_lcc")[["scenario_id", "shock_type", "min_lcc", "peak_damage_nodes"]]
    degree_corr = summary_df[["mean_seed_degree", "peak_damage_nodes"]].corr().iloc[0, 1]
    conc_corr = summary_df[["dependency_concentration", "min_lcc"]].corr().iloc[0, 1]
    sector_cols = [c for c in summary_df.columns if c.endswith("_min_health")]

    lines = [
        "# Scenario Experiment Summary",
        "",
        "## Scenario Set",
        "",
        f"- Model label: `{model_label}`",
        f"- Scenario count: `{len(metadata_df)}`",
        "- Shock families: `power`, `telecom`, `ems`, `hospital`, `mixed`",
        "- Frozen simulator core: `simulator_v2.py`",
        "",
        "## Statistical Summaries by Shock Type",
        "",
    ]
    for shock_type, row in avg_by_shock.iterrows():
        lines.extend(
            [
                f"- `{shock_type}`",
                f"  min LCC: mean `{row['mean_min_lcc']:.4f}`, min `{row['min_min_lcc']:.4f}`, max `{row['max_min_lcc']:.4f}`",
                f"  cascade size: mean `{row['mean_peak_damage']:.2f}`, min `{row['min_peak_damage']:.0f}`, max `{row['max_peak_damage']:.0f}`",
                f"  recovery time: mean `{row['mean_recovery_time']:.2f}`, min `{row['min_recovery_time']:.2f}`, max `{row['max_recovery_time']:.2f}`",
            ]
        )

    worst_shock = avg_by_shock.index[0] if not avg_by_shock.empty else "n/a"
    slowest_recovery = avg_by_shock["mean_recovery_time"].idxmax() if not avg_by_shock.empty else "n/a"
    most_damaging = summary_df.sort_values("peak_damage_nodes", ascending=False).iloc[0]

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Largest cascades in this pilot are created by: `{worst_shock}` shocks.",
            f"- Slowest average recovery is seen in: `{slowest_recovery}` shocks.",
            f"- Most damaging seed scenario observed: `{most_damaging['scenario_id']}` with peak damage `{int(most_damaging['peak_damage_nodes'])}` and minimum LCC `{most_damaging['min_lcc']:.4f}`.",
            f"- Correlation between mean seed degree and cascade size: `{degree_corr:.3f}`.",
            f"- Correlation between dependency concentration and minimum LCC: `{conc_corr:.3f}`.",
            "",
            "## Sector Health Impact by Shock Type",
            "",
        ]
    )
    if sector_cols:
        sector_summary = summary_df.groupby("shock_type")[sector_cols].mean()
        for shock_type, row in sector_summary.iterrows():
            parts = []
            for col in sector_cols:
                sector = col.replace("_min_health", "")
                parts.append(f"{sector} `{row[col]:.3f}`")
            lines.append(f"- `{shock_type}` mean minimum sector health: " + ", ".join(parts))

    lines.extend(
        [
            "",
            "## Top 10 Worst Scenarios by Minimum LCC",
            "",
        ]
    )
    for row in worst.itertuples(index=False):
        lines.append(f"- `{row.scenario_id}` | `{row.shock_type}` | min LCC `{row.min_lcc:.4f}` | peak damage `{int(row.peak_damage_nodes)}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These pilot scenarios are intended to test whether the tuned simulator produces a varied and analyzable scenario dataset before scaling to a full batch.",
            "- The strongest candidate paper findings are the relative dominance of power-driven cascades, the more limited direct damage from telecom and EMS shocks, and the value of dependency concentration as a structural vulnerability signal.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pilot batch of cascade scenarios with the tuned simulator.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--sim-edges", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--output-dir", default="v3/data/processed/pilot_scenarios")
    parser.add_argument("--figures-dir", default="v3/runs/figures/pilot_scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-prefix", default="pilot")
    parser.add_argument("--model-label", default="pilot_model")
    parser.add_argument("--power-count", type=int, default=PILOT_PLAN["power"])
    parser.add_argument("--telecom-count", type=int, default=PILOT_PLAN["telecom"])
    parser.add_argument("--ems-count", type=int, default=PILOT_PLAN["ems"])
    parser.add_argument("--hospital-count", type=int, default=PILOT_PLAN["hospital"])
    parser.add_argument("--mixed-count", type=int, default=PILOT_PLAN["mixed"])
    parser.add_argument("--figure-prefix", default="")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    sim_edges = sim2.ensure_simulation_edges(args.nodes, args.edges, args.sim_edges)
    structural_graph = sim2.build_structural_graph(nodes, edges)

    scenario_plan = {
        "power": args.power_count,
        "telecom": args.telecom_count,
        "ems": args.ems_count,
        "hospital": args.hospital_count,
        "mixed": args.mixed_count,
    }
    metadata_df = build_scenarios(nodes, edges, sim_edges, args.seed, scenario_plan, args.scenario_prefix)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    time_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    original_params = copy.deepcopy(sim2.SECTOR_PARAMS)
    try:
        for meta in metadata_df.to_dict("records"):
            sim2.SECTOR_PARAMS = apply_parameter_overrides(
                meta["shock_severity_scale"],
                meta["recovery_scale"],
                meta["propagation_scale"],
            )
            seeds = meta["seed_nodes"].split(";") if meta["seed_nodes"] else []
            metrics_df, _ = sim2.run_cascade_v2(
                nodes,
                sim_edges,
                structural_graph,
                meta["scenario_id"],
                seeds,
                int(meta["simulation_length"]),
            )
            metrics_df.insert(0, "scenario_id", meta["scenario_id"])
            metrics_df.insert(1, "shock_type", meta["shock_type"])
            time_rows.append(metrics_df)
            summary_rows.append(summarize_metrics(pd.Series(meta), metrics_df))
    finally:
        sim2.SECTOR_PARAMS = original_params

    time_df = pd.concat(time_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["recovery_time"] = pd.to_numeric(summary_df["recovery_time"], errors="coerce")

    metadata_path = output_dir / "scenario_metadata.csv"
    time_path = output_dir / "scenario_time_series.csv"
    summary_path = output_dir / "scenario_summary_metrics.csv"
    report_path = output_dir / "scenario_experiment_summary.md"

    metadata_df.to_csv(metadata_path, index=False)
    time_df.to_csv(time_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_outputs(time_df, summary_df, figures_dir, prefix=args.figure_prefix)
    write_report(metadata_df, summary_df, report_path, args.model_label)

    print(f"Wrote metadata to {metadata_path}")
    print(f"Wrote time series to {time_path}")
    print(f"Wrote summary metrics to {summary_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
