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
SEVERITY_BY_STATE = {1.0: 0.0, 0.5: 0.5, 0.0: 1.0}
AGGREGATION_MODES = {"additive_exposure", "strongest_dependency", "redundancy_buffer"}

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
) -> list[str]:
    recovered_nodes: list[str] = []
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
                recovered_nodes.append(node_id)
        elif state == 0.5:
            if downtime < params["min_degraded_duration"]:
                continue
            if damage_load[node_id] > 0.12:
                continue
            if deterministic_trial(params["recovery_degraded"], "recover_degraded", t, node_id):
                states[node_id] = 1.0
                recovered_nodes.append(node_id)
    return recovered_nodes


def build_support_maps(
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, list[dict]], dict[str, dict[str, list[dict]]]]:
    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    outgoing: dict[str, list[dict]] = {}
    incoming: dict[str, dict[str, list[dict]]] = {}

    for edge_idx, row in enumerate(sim_edges.to_dict("records")):
        source = row["simulation_source"]
        target = row["simulation_target"]
        support_class = node_types[source]
        edge = {
            "edge_id": str(edge_idx),
            "source": source,
            "target": target,
            "support_class": support_class,
            "weight": float(row["weight"]),
            "delay": int(row["delay"]),
            "dependency_type": row.get("dependency_type", f"{support_class} -> {node_types[target]}"),
            "semantic_type": row.get("semantic_type", "dependency"),
        }
        outgoing.setdefault(source, []).append(edge)
        incoming.setdefault(target, {}).setdefault(support_class, []).append(edge)

    return node_types, outgoing, incoming


def class_impact(
    support_edges: list[dict],
    perceived_severity: dict[str, float],
    aggregation_mode: str,
    redundancy_threshold: float,
) -> float:
    weighted_impacts = [edge["weight"] * perceived_severity.get(edge["edge_id"], 0.0) for edge in support_edges]
    total_weight = sum(edge["weight"] for edge in support_edges)
    max_weight = max((edge["weight"] for edge in support_edges), default=0.0)

    if aggregation_mode == "additive_exposure":
        return sum(weighted_impacts)

    if aggregation_mode == "strongest_dependency":
        return max(weighted_impacts, default=0.0)

    weighted_share = sum(weighted_impacts) / max(total_weight, 1e-6)
    if weighted_share <= 0:
        return 0.0
    if weighted_share < redundancy_threshold:
        return max_weight * weighted_share * 0.15

    overflow = (weighted_share - redundancy_threshold) / max(1.0 - redundancy_threshold, 1e-6)
    return max_weight * (0.35 + 0.65 * overflow)


def aggregate_target_impact(
    target_id: str,
    incoming_supports: dict[str, dict[str, list[dict]]],
    perceived_severity: dict[str, float],
    aggregation_mode: str,
    redundancy_threshold: float,
) -> float:
    class_map = incoming_supports.get(target_id, {})
    return sum(
        class_impact(edges, perceived_severity, aggregation_mode, redundancy_threshold)
        for edges in class_map.values()
    )


def run_cascade_v3(
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
    structural_graph: nx.Graph,
    scenario_name: str,
    seed_nodes: list[str],
    timesteps: int,
    aggregation_mode: str = "additive_exposure",
    redundancy_threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    if aggregation_mode not in AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation mode: {aggregation_mode}")

    node_ids = nodes["node_id"].tolist()
    node_types, outgoing, incoming_supports = build_support_maps(nodes, sim_edges)
    node_names = nodes.set_index("node_id")["name"].to_dict()
    states = {node_id: 1.0 for node_id in node_ids}
    first_change_timestep: dict[str, int] = {}
    last_downgrade_timestep: dict[str, int | None] = {node_id: None for node_id in node_ids}
    damage_load: dict[str, float] = {node_id: 0.0 for node_id in node_ids}
    perceived_severity: dict[str, float] = {}
    pending_updates: dict[int, list[tuple[str, float]]] = {}

    def schedule_node_update(node_id: str, now_t: int) -> None:
        source_severity = SEVERITY_BY_STATE[states[node_id]]
        for edge in outgoing.get(node_id, []):
            arrival_t = now_t + int(edge["delay"])
            if arrival_t > timesteps:
                continue
            pending_updates.setdefault(arrival_t, []).append((edge["edge_id"], source_severity))

    for seed in seed_nodes:
        states[seed] = 0.0
        first_change_timestep[seed] = 0
        last_downgrade_timestep[seed] = 0
        damage_load[seed] = 1.0
        schedule_node_update(seed, 0)

    records: list[dict] = []

    for t in range(timesteps + 1):
        for edge_id, severity in pending_updates.pop(t, []):
            perceived_severity[edge_id] = severity

        for node_id in node_ids:
            damage_load[node_id] *= 0.90

        changed_nodes: list[str] = []
        for node_id in node_ids:
            impact = aggregate_target_impact(
                node_id,
                incoming_supports,
                perceived_severity,
                aggregation_mode,
                redundancy_threshold,
            )
            if impact > 0:
                damage_load[node_id] = min(1.5, damage_load[node_id] + impact)

            effective_impact = impact + 0.6 * damage_load[node_id]
            new_state = transition_state(states[node_id], node_types[node_id], effective_impact, t, node_id)
            if new_state < states[node_id]:
                states[node_id] = new_state
                first_change_timestep.setdefault(node_id, t)
                last_downgrade_timestep[node_id] = t
                changed_nodes.append(node_id)

        for node_id in changed_nodes:
            schedule_node_update(node_id, t)

        if t > 0:
            recovered_nodes = recovery_step(states, node_types, damage_load, last_downgrade_timestep, t)
            for node_id in recovered_nodes:
                schedule_node_update(node_id, t)
                first_change_timestep.setdefault(node_id, t)

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
        "aggregation_mode": aggregation_mode,
        "redundancy_threshold": redundancy_threshold,
    }
    return pd.DataFrame(records), scenario_info


def write_summary(
    summary_path: Path,
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
    scenario_infos: list[dict],
    shock_k: int,
    aggregation_mode: str,
    redundancy_threshold: float,
) -> None:
    lines = [
        "# Simulator V3 Summary",
        "",
        "## Setup",
        "",
        f"- Nodes: `{len(nodes)}`",
        f"- Simulation edges: `{len(sim_edges)}`",
        f"- Timesteps: `{TIMESTEPS}`",
        f"- Initial shock size per scenario: `{shock_k}`",
        f"- Aggregation mode: `{aggregation_mode}`",
        f"- Redundancy threshold: `{redundancy_threshold}`",
        "- `simulator_v3` updates delayed perceived support state on each dependency edge and aggregates support loss by support class at the dependent node.",
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
            "- `additive_exposure` preserves the existing logic where each upstream failure adds to the target's damage burden.",
            "- `strongest_dependency` turns multi-support nodes into max-exposure systems.",
            "- `redundancy_buffer` treats multiple supports within the same support class as fallback capacity, so a node is partially shielded while enough healthy support remains.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulator v3 with support-set redundancy-aware aggregation.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--sim-edges-out", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--shock-k", type=int, default=SHOCK_K)
    parser.add_argument("--metrics-dir", default="v3/data/processed")
    parser.add_argument("--summary-out", default="v3/data/processed/simulator_v3_summary.md")
    parser.add_argument("--aggregation-mode", default="redundancy_buffer", choices=sorted(AGGREGATION_MODES))
    parser.add_argument("--redundancy-threshold", type=float, default=0.5)
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    sim_edges = ensure_simulation_edges(args.nodes, args.edges, args.sim_edges_out)
    structural_graph = build_structural_graph(nodes, edges)

    degree_df = pd.concat(
        [
            edges[["src_node_id"]].rename(columns={"src_node_id": "node_id"}),
            edges[["dst_node_id"]].rename(columns={"dst_node_id": "node_id"}),
        ],
        ignore_index=True,
    )
    degree_counts = degree_df.value_counts("node_id").rename("degree").reset_index()
    merged = nodes.merge(degree_counts, on="node_id", how="left").fillna({"degree": 0})
    scenario_seeds: dict[str, list[str]] = {}
    for node_type, label in [("power", "power"), ("telecom", "telecom"), ("ems_fire", "ems")]:
        subset = merged[merged["node_type"] == node_type].sort_values(
            ["degree", "name", "node_id"], ascending=[False, True, True]
        )
        scenario_seeds[label] = subset.head(args.shock_k)["node_id"].astype(str).tolist()

    metrics_dir = Path(args.metrics_dir)
    summary_path = Path(args.summary_out)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scenario_infos: list[dict] = []
    for scenario_name, seeds in scenario_seeds.items():
        metrics_df, info = run_cascade_v3(
            nodes,
            sim_edges,
            structural_graph,
            scenario_name,
            seeds,
            args.timesteps,
            aggregation_mode=args.aggregation_mode,
            redundancy_threshold=args.redundancy_threshold,
        )
        scenario_infos.append(info)
        metrics_df.to_csv(metrics_dir / f"cascade_metrics_{scenario_name}_v3.csv", index=False)

    write_summary(
        summary_path,
        nodes,
        sim_edges,
        scenario_infos,
        args.shock_k,
        args.aggregation_mode,
        args.redundancy_threshold,
    )
    print(f"Wrote v3 metrics to {metrics_dir}")
    print(f"Wrote v3 summary to {summary_path}")


if __name__ == "__main__":
    main()
