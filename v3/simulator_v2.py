from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from v3.simulator_v1 import ensure_simulation_edges
except ModuleNotFoundError:
    from simulator_v1 import ensure_simulation_edges


TIMESTEPS = 50
SHOCK_K = 3
STATE_LABELS = {1.0: "normal", 0.5: "degraded", 0.0: "failed"}

SECTOR_PARAMS = {
    "power": {
        "degrade_threshold": 0.22,
        "fail_threshold": 0.60,
        "propagation_scale": 1.00,
        "recovery_failed": 0.01,
        "recovery_degraded": 0.04,
        "min_failed_duration": 8,
        "min_degraded_duration": 4,
    },
    "telecom": {
        "degrade_threshold": 0.28,
        "fail_threshold": 0.72,
        "propagation_scale": 0.70,
        "recovery_failed": 0.02,
        "recovery_degraded": 0.06,
        "min_failed_duration": 6,
        "min_degraded_duration": 3,
    },
    "ems_fire": {
        "degrade_threshold": 0.26,
        "fail_threshold": 0.68,
        "propagation_scale": 0.80,
        "recovery_failed": 0.02,
        "recovery_degraded": 0.07,
        "min_failed_duration": 6,
        "min_degraded_duration": 3,
    },
    "hospital": {
        "degrade_threshold": 0.18,
        "fail_threshold": 0.52,
        "propagation_scale": 0.85,
        "recovery_failed": 0.01,
        "recovery_degraded": 0.03,
        "min_failed_duration": 8,
        "min_degraded_duration": 4,
    },
    "school": {
        "degrade_threshold": 0.25,
        "fail_threshold": 0.65,
        "propagation_scale": 0.75,
        "recovery_failed": 0.015,
        "recovery_degraded": 0.05,
        "min_failed_duration": 5,
        "min_degraded_duration": 3,
    },
    "emergency_management": {
        "degrade_threshold": 0.30,
        "fail_threshold": 0.75,
        "propagation_scale": 0.60,
        "recovery_failed": 0.03,
        "recovery_degraded": 0.08,
        "min_failed_duration": 4,
        "min_degraded_duration": 2,
    },
}


def stable_unit_float(*parts: object) -> float:
    text = "|".join(str(part) for part in parts)
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 1_000_003
    return value / 1_000_003.0


def deterministic_trial(probability: float, *parts: object) -> bool:
    probability = max(0.0, min(1.0, probability))
    return stable_unit_float(*parts) < probability


def build_structural_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.node_id, node_type=row.node_type, name=row.name)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src_node_id, row.dst_node_id)
    return graph


def degree_table(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    degree_df = pd.concat(
        [
            edges[["src_node_id"]].rename(columns={"src_node_id": "node_id"}),
            edges[["dst_node_id"]].rename(columns={"dst_node_id": "node_id"}),
        ],
        ignore_index=True,
    )
    degree_counts = degree_df.value_counts("node_id").rename("degree").reset_index()
    return nodes.merge(degree_counts, on="node_id", how="left").fillna({"degree": 0})


def pick_multi_seeds(nodes: pd.DataFrame, edges: pd.DataFrame, shock_k: int) -> dict[str, list[str]]:
    merged = degree_table(nodes, edges)
    seeds: dict[str, list[str]] = {}
    for node_type, label in [("power", "power"), ("telecom", "telecom"), ("ems_fire", "ems")]:
        subset = merged[merged["node_type"] == node_type].sort_values(
            ["degree", "name", "node_id"], ascending=[False, True, True]
        )
        if subset.empty:
            raise ValueError(f"No nodes available for node_type={node_type}")
        seeds[label] = subset.head(shock_k)["node_id"].astype(str).tolist()
    return seeds


def build_outgoing(sim_edges: pd.DataFrame) -> dict[str, list[dict]]:
    outgoing: dict[str, list[dict]] = {}
    for row in sim_edges.to_dict("records"):
        outgoing.setdefault(row["simulation_source"], []).append(row)
    return outgoing


def transition_state(current_state: float, node_type: str, cumulative_impact: float, t: int, node_id: str) -> float:
    params = SECTOR_PARAMS[node_type]
    if current_state == 0.0:
        return 0.0

    degrade_prob = min(0.95, cumulative_impact / max(params["degrade_threshold"], 1e-6))
    fail_prob = min(0.95, cumulative_impact / max(params["fail_threshold"], 1e-6))

    if current_state == 1.0:
        if cumulative_impact >= params["fail_threshold"] and deterministic_trial(fail_prob, "fail", t, node_id):
            return 0.0
        if cumulative_impact >= params["degrade_threshold"] and deterministic_trial(
            degrade_prob, "degrade", t, node_id
        ):
            return 0.5
        return 1.0

    if cumulative_impact >= params["fail_threshold"] and deterministic_trial(fail_prob, "fail", t, node_id):
        return 0.0
    return 0.5


def recovery_step(
    states: dict[str, float],
    node_types: dict[str, str],
    damage_load: dict[str, float],
    last_downgrade_timestep: dict[str, int | None],
    t: int,
) -> None:
    for node_id, state in list(states.items()):
        params = SECTOR_PARAMS[node_types[node_id]]
        downtime = math.inf if last_downgrade_timestep[node_id] is None else t - int(last_downgrade_timestep[node_id])
        if state == 0.0:
            if downtime < params["min_failed_duration"]:
                continue
            if damage_load[node_id] > 0.20:
                continue
            if deterministic_trial(params["recovery_failed"], "recover_failed", t, node_id):
                states[node_id] = 0.5
        elif state == 0.5:
            if downtime < params["min_degraded_duration"]:
                continue
            if damage_load[node_id] > 0.12:
                continue
            if deterministic_trial(params["recovery_degraded"], "recover_degraded", t, node_id):
                states[node_id] = 1.0


def run_cascade_v2(
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
    structural_graph: nx.Graph,
    scenario_name: str,
    seed_nodes: list[str],
    timesteps: int,
) -> tuple[pd.DataFrame, dict]:
    node_ids = nodes["node_id"].tolist()
    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    node_names = nodes.set_index("node_id")["name"].to_dict()
    states = {node_id: 1.0 for node_id in node_ids}
    first_change_timestep: dict[str, int] = {}
    last_downgrade_timestep: dict[str, int | None] = {node_id: None for node_id in node_ids}
    damage_load: dict[str, float] = {node_id: 0.0 for node_id in node_ids}
    for seed in seed_nodes:
        states[seed] = 0.0
        first_change_timestep[seed] = 0
        last_downgrade_timestep[seed] = 0
        damage_load[seed] = 1.0

    outgoing = build_outgoing(sim_edges)
    pending_impacts: dict[int, list[tuple[str, float, str]]] = {}
    records: list[dict] = []

    def schedule_from(node_id: str, now_t: int) -> None:
        source_state = states[node_id]
        if source_state >= 1.0:
            return
        source_severity = 1.0 if source_state == 0.0 else 0.5
        source_type = node_types[node_id]
        source_scale = SECTOR_PARAMS[source_type]["propagation_scale"]
        for edge in outgoing.get(node_id, []):
            delay = int(edge["delay"])
            arrival_t = now_t + delay
            if arrival_t > timesteps:
                continue
            target_id = edge["simulation_target"]
            target_type = node_types[target_id]
            semantic_type = edge.get("semantic_type", "dependency")
            base_prob = 0.60 if semantic_type == "dependency" else 0.45
            weight = float(edge["weight"])
            transmission_prob = min(0.95, base_prob * weight * source_scale * (1.0 if source_state == 0.0 else 0.8))
            if not deterministic_trial(transmission_prob, "edge", scenario_name, now_t, node_id, target_id, arrival_t):
                continue
            impact = weight * source_severity * source_scale
            if target_type == "hospital":
                impact *= 1.10
            elif target_type == "school":
                impact *= 0.95
            pending_impacts.setdefault(arrival_t, []).append((target_id, impact, node_id))

    for seed in seed_nodes:
        schedule_from(seed, 0)

    for t in range(timesteps + 1):
        incoming = pending_impacts.pop(t, [])
        cumulative: dict[str, float] = {}
        for target_id, impact, _source_id in incoming:
            cumulative[target_id] = cumulative.get(target_id, 0.0) + impact
        for node_id in node_ids:
            damage_load[node_id] *= 0.90

        changed_nodes: list[str] = []
        for node_id, impact in cumulative.items():
            damage_load[node_id] = min(1.5, damage_load[node_id] + impact)
            effective_impact = impact + 0.6 * damage_load[node_id]
            new_state = transition_state(states[node_id], node_types[node_id], effective_impact, t, node_id)
            if new_state < states[node_id]:
                states[node_id] = new_state
                first_change_timestep.setdefault(node_id, t)
                last_downgrade_timestep[node_id] = t
                changed_nodes.append(node_id)

        for node_id in changed_nodes:
            schedule_from(node_id, t)

        if t > 0:
            pre_recovery = states.copy()
            recovery_step(states, node_types, damage_load, last_downgrade_timestep, t)
            for node_id in node_ids:
                if states[node_id] != pre_recovery[node_id] and node_id not in first_change_timestep:
                    first_change_timestep[node_id] = t

        active_nodes = [node_id for node_id, state in states.items() if state > 0.0]
        surviving = structural_graph.subgraph(active_nodes)
        if active_nodes:
            components = list(nx.connected_components(surviving))
            largest_component = max((len(c) for c in components), default=0)
            lcc_fraction = largest_component / float(len(node_ids))
            component_count = len(components)
        else:
            lcc_fraction = 0.0
            component_count = 0

        row = {
            "scenario": scenario_name,
            "timestep": t,
            "failed_nodes": sum(1 for state in states.values() if state == 0.0),
            "degraded_nodes": sum(1 for state in states.values() if state == 0.5),
            "lcc_fraction": lcc_fraction,
            "component_count": component_count,
        }
        for sector in sorted(nodes["node_type"].unique()):
            sector_nodes = nodes.loc[nodes["node_type"] == sector, "node_id"].tolist()
            row[f"sector_health_{sector}"] = sum(states[n] for n in sector_nodes) / float(len(sector_nodes))
        records.append(row)

    final_failed = sum(1 for state in states.values() if state == 0.0)
    final_degraded = sum(1 for state in states.values() if state == 0.5)
    scenario_info = {
        "scenario": scenario_name,
        "seed_nodes": seed_nodes,
        "seed_names": [node_names[node_id] for node_id in seed_nodes],
        "final_failed_nodes": final_failed,
        "final_degraded_nodes": final_degraded,
        "final_lcc_fraction": float(records[-1]["lcc_fraction"]),
        "final_component_count": int(records[-1]["component_count"]),
    }
    return pd.DataFrame(records), scenario_info


def plot_comparison(metrics: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"power": "#c0392b", "telecom": "#2980b9", "ems": "#16a085"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario, df in metrics.items():
        ax.plot(df["timestep"], df["lcc_fraction"], label=scenario, color=colors.get(scenario))
    ax.set_title("LCC Fraction Under Multi-Node Infrastructure Shocks")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("LCC Fraction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_lcc_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario, df in metrics.items():
        ax.plot(df["timestep"], df["failed_nodes"], label=scenario, color=colors.get(scenario))
    ax.set_title("Failed Nodes Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Failed Nodes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_failed_nodes_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, sector in zip(axes, ["power", "telecom", "ems_fire"]):
        column = f"sector_health_{sector}"
        for scenario, df in metrics.items():
            ax.plot(df["timestep"], df[column], label=scenario, color=colors.get(scenario))
        ax.set_title(f"{sector} Average Health")
        ax.set_ylabel("Average Health")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_sector_health_comparison.png", dpi=200)
    plt.close(fig)


def write_summary(summary_path: Path, nodes: pd.DataFrame, sim_edges: pd.DataFrame, scenario_infos: list[dict], shock_k: int) -> None:
    lines = [
        "# Simulator V2 Summary",
        "",
        "## Setup",
        "",
        f"- Nodes: `{len(nodes)}`",
        f"- Simulation edges: `{len(sim_edges)}`",
        "- State space: `1.0 = normal`, `0.5 = degraded`, `0.0 = failed`",
        f"- Timesteps: `{TIMESTEPS}`",
        f"- Initial shock size per scenario: `{shock_k}`",
        "- Propagation is stochastic and sector-specific.",
        "- Recovery is enabled for failed and degraded nodes.",
        "",
        "## Scenario Outcomes",
        "",
    ]
    for info in scenario_infos:
        lines.extend(
            [
                f"### {info['scenario']}",
                "",
                f"- Seed nodes: `{', '.join(info['seed_nodes'])}`",
                f"- Seed assets: `{'; '.join(info['seed_names'])}`",
                f"- Final failed nodes: `{info['final_failed_nodes']}`",
                f"- Final degraded nodes: `{info['final_degraded_nodes']}`",
                f"- Final LCC fraction: `{info['final_lcc_fraction']:.4f}`",
                f"- Final component count: `{info['final_component_count']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `simulator_v2` extends `v1` with multi-node shocks, stochastic transmission, recovery, and sector-specific vulnerability thresholds.",
            "- Compared with `v1`, this version is intended to produce more realistic partial cascades and non-monotonic health trajectories.",
            "- These trajectories are suitable for the next step of simulator stress-testing before ML dataset generation.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulator v2 with stochastic propagation and recovery.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--sim-edges-out", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--shock-k", type=int, default=SHOCK_K)
    parser.add_argument("--metrics-dir", default="v3/data/processed")
    parser.add_argument("--figures-dir", default="v3/runs/figures")
    parser.add_argument("--summary-out", default="v3/data/processed/simulator_v2_summary.md")
    parser.add_argument("--prefix", default="v2")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    sim_edges = ensure_simulation_edges(args.nodes, args.edges, args.sim_edges_out)
    structural_graph = build_structural_graph(nodes, edges)
    scenario_seeds = pick_multi_seeds(nodes, edges, args.shock_k)

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    summary_path = Path(args.summary_out)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_metrics: dict[str, pd.DataFrame] = {}
    scenario_infos: list[dict] = []
    for scenario_name, seeds in scenario_seeds.items():
        metrics_df, info = run_cascade_v2(nodes, sim_edges, structural_graph, scenario_name, seeds, args.timesteps)
        scenario_metrics[scenario_name] = metrics_df
        scenario_infos.append(info)
        metrics_df.to_csv(metrics_dir / f"cascade_metrics_{scenario_name}_{args.prefix}.csv", index=False)

    plot_comparison(scenario_metrics, figures_dir, args.prefix)
    write_summary(summary_path, nodes, sim_edges, scenario_infos, args.shock_k)

    print(f"Wrote metrics to {metrics_dir}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
