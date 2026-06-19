from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from standardize_simulation_edges import standardize_edges
except ModuleNotFoundError:
    from v3.standardize_simulation_edges import standardize_edges


TIMESTEPS = 50
SEVERITY_BY_STATE = {1.0: 0.0, 0.5: 0.5, 0.0: 1.0}
STATE_LABELS = {1.0: "normal", 0.5: "degraded", 0.0: "failed"}


def ensure_simulation_edges(nodes_path: str, edges_path: str, out_path: str) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    sim_edges, _ = standardize_edges(edges)
    node_ids = set(nodes["node_id"])
    sim_edges = sim_edges[
        sim_edges["simulation_source"].isin(node_ids) & sim_edges["simulation_target"].isin(node_ids)
    ].copy()
    sim_edges.to_csv(out_path, index=False)
    return sim_edges


def build_graphs(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[nx.DiGraph, nx.Graph]:
    directed = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        directed.add_node(row.node_id, node_type=row.node_type, name=row.name)
    for row in edges.itertuples(index=False):
        directed.add_edge(
            row.src_node_id,
            row.dst_node_id,
            weight=float(row.weight),
            delay=int(row.delay),
            dependency_type=f"{row.src_type} -> {row.dst_type}",
        )
    undirected = directed.to_undirected()
    return directed, undirected


def pick_seed_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, str]:
    degree_df = pd.concat(
        [
            edges[["src_node_id"]].rename(columns={"src_node_id": "node_id"}),
            edges[["dst_node_id"]].rename(columns={"dst_node_id": "node_id"}),
        ],
        ignore_index=True,
    )
    degree_counts = degree_df.value_counts("node_id").rename("degree").reset_index()
    merged = nodes.merge(degree_counts, on="node_id", how="left").fillna({"degree": 0})

    seeds = {}
    for node_type, label in [("power", "power"), ("telecom", "telecom"), ("ems_fire", "ems")]:
        subset = merged[merged["node_type"] == node_type].sort_values(
            ["degree", "name", "node_id"], ascending=[False, True, True]
        )
        if subset.empty:
            raise ValueError(f"No nodes available for node_type={node_type}")
        seeds[label] = str(subset.iloc[0]["node_id"])
    return seeds


def next_state(current_state: float, impact: float) -> float:
    if current_state == 0.0:
        return 0.0
    if current_state == 1.0:
        if impact >= 0.75:
            return 0.0
        if impact >= 0.30:
            return 0.5
        return 1.0
    if impact >= 0.30:
        return 0.0
    return 0.5


def run_cascade(
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
    structural_graph: nx.Graph,
    scenario_name: str,
    seed_node: str,
    timesteps: int = TIMESTEPS,
) -> tuple[pd.DataFrame, dict]:
    node_ids = nodes["node_id"].tolist()
    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    states = {node_id: 1.0 for node_id in node_ids}
    states[seed_node] = 0.0
    state_change_time = {seed_node: 0}

    outgoing: dict[str, list[dict]] = {}
    for row in sim_edges.to_dict("records"):
        outgoing.setdefault(row["simulation_source"], []).append(row)

    event_queue: list[tuple[int, str, float]] = []

    def schedule_from(node_id: str, time_now: int) -> None:
        severity = SEVERITY_BY_STATE[states[node_id]]
        if severity <= 0:
            return
        for edge in outgoing.get(node_id, []):
            arrival = time_now + int(edge["delay"])
            if arrival > timesteps:
                continue
            impact = float(edge["weight"]) * severity
            if impact <= 0:
                continue
            heapq.heappush(event_queue, (arrival, edge["simulation_target"], impact))

    schedule_from(seed_node, 0)
    records: list[dict] = []

    for t in range(timesteps + 1):
        impacts: dict[str, float] = {}
        while event_queue and event_queue[0][0] == t:
            _, target, impact = heapq.heappop(event_queue)
            impacts[target] = impacts.get(target, 0.0) + impact

        changed_nodes: list[str] = []
        for node_id, impact in impacts.items():
            new_state = next_state(states[node_id], impact)
            if new_state < states[node_id]:
                states[node_id] = new_state
                state_change_time[node_id] = t
                changed_nodes.append(node_id)

        for node_id in changed_nodes:
            schedule_from(node_id, t)

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
            sector_nodes = [node_id for node_id in node_ids if node_types[node_id] == sector]
            row[f"sector_health_{sector}"] = sum(states[n] for n in sector_nodes) / float(len(sector_nodes))
        records.append(row)

    final_states = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_type": [node_types[node_id] for node_id in node_ids],
            "final_state": [states[node_id] for node_id in node_ids],
            "final_state_label": [STATE_LABELS[states[node_id]] for node_id in node_ids],
            "first_change_timestep": [state_change_time.get(node_id) for node_id in node_ids],
        }
    )
    scenario_info = {
        "scenario": scenario_name,
        "seed_node": seed_node,
        "seed_name": nodes.set_index("node_id").loc[seed_node, "name"],
        "seed_type": nodes.set_index("node_id").loc[seed_node, "node_type"],
        "final_failed_nodes": int((final_states["final_state"] == 0.0).sum()),
        "final_degraded_nodes": int((final_states["final_state"] == 0.5).sum()),
        "final_lcc_fraction": float(records[-1]["lcc_fraction"]),
        "final_component_count": int(records[-1]["component_count"]),
    }
    return pd.DataFrame(records), scenario_info


def plot_comparison(metrics: dict[str, pd.DataFrame], output_dir: Path) -> None:
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
    ax.set_title("LCC Fraction Under Seeded Infrastructure Failures")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("LCC Fraction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "lcc_comparison.png", dpi=200)
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
    fig.savefig(output_dir / "failed_nodes_comparison.png", dpi=200)
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
    fig.savefig(output_dir / "sector_health_comparison.png", dpi=200)
    plt.close(fig)


def write_summary(summary_path: Path, scenario_infos: list[dict], nodes: pd.DataFrame, sim_edges: pd.DataFrame) -> None:
    lines = [
        "# Simulator V1 Summary",
        "",
        "## Setup",
        "",
        f"- Nodes: `{len(nodes)}`",
        f"- Simulation edges: `{len(sim_edges)}`",
        "- State space: `1.0 = normal`, `0.5 = degraded`, `0.0 = failed`",
        "- Timesteps: `50`",
        "- Dependency edges were reversed into failure-flow direction before simulation.",
        "",
        "## Seed Scenarios",
        "",
    ]
    for info in scenario_infos:
        lines.extend(
            [
                f"### {info['scenario']}",
                "",
                f"- Seed node: `{info['seed_node']}`",
                f"- Seed asset: `{info['seed_name']}`",
                f"- Seed type: `{info['seed_type']}`",
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
            "- `simulator_v1` is a deterministic first-pass cascade engine intended to verify that the cleaned dependency graph produces non-trivial degradation trajectories.",
            "- Failed or degraded sources transmit impact to dependents after the configured edge delay, scaled by edge weight.",
            "- Sector-health trajectories and LCC decline can now be inspected before introducing any learning models.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first cascade simulator on the cleaned Montgomery graph.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--sim-edges-out", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--metrics-dir", default="v3/data/processed")
    parser.add_argument("--figures-dir", default="v3/runs/figures")
    parser.add_argument("--summary-out", default="v3/data/processed/simulator_v1_summary.md")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    sim_edges = ensure_simulation_edges(args.nodes, args.edges, args.sim_edges_out)
    directed_graph, undirected_graph = build_graphs(nodes, edges)
    seeds = pick_seed_nodes(nodes, edges)

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    summary_path = Path(args.summary_out)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_metrics: dict[str, pd.DataFrame] = {}
    scenario_infos: list[dict] = []
    for scenario_name, seed_node in seeds.items():
        metrics_df, info = run_cascade(nodes, sim_edges, undirected_graph, scenario_name, seed_node, args.timesteps)
        scenario_metrics[scenario_name] = metrics_df
        scenario_infos.append(info)
        metrics_df.to_csv(metrics_dir / f"cascade_metrics_{scenario_name}.csv", index=False)

    plot_comparison(scenario_metrics, figures_dir)
    write_summary(summary_path, scenario_infos, nodes, sim_edges)
    print(f"Wrote simulation edges to {args.sim_edges_out}")
    print(f"Wrote metrics to {metrics_dir}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
