import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

from feature_utils import (
    build_saved_feature_tensors,
    feature_order_documentation,
    normalize_feature_flags,
    parse_bool_flag,
)
from graph_factory import build_graph, to_directed_edge_index, to_message_passing_edge_index
from sc_sim_v0 import SimParams, simulate


def compute_graph_target_series(G: nx.DiGraph, node_df: pd.DataFrame) -> dict[str, list[float]]:
    num_nodes = G.number_of_nodes()
    num_edges = max(1, G.number_of_edges())
    series = {
        "lcc_fraction": [],
        "component_fraction": [],
        "diameter_fraction": [],
        "edge_survival_ratio": [],
    }

    for t in sorted(node_df["t"].unique()):
        df_t = node_df[node_df["t"] == t]
        healthy_nodes = df_t.loc[df_t["health"] == 1, "node"].tolist()
        if not healthy_nodes:
            series["lcc_fraction"].append(0.0)
            series["component_fraction"].append(0.0)
            series["diameter_fraction"].append(0.0)
            series["edge_survival_ratio"].append(0.0)
            continue

        healthy_subgraph = G.subgraph(healthy_nodes).copy()
        weak_components = list(nx.weakly_connected_components(healthy_subgraph))
        largest_component_nodes = max(weak_components, key=len)
        largest_component = healthy_subgraph.subgraph(largest_component_nodes).copy()

        series["lcc_fraction"].append(len(largest_component_nodes) / float(num_nodes))
        series["component_fraction"].append(len(weak_components) / float(num_nodes))
        series["edge_survival_ratio"].append(
            healthy_subgraph.number_of_edges() / float(num_edges)
        )

        if largest_component.number_of_nodes() <= 1:
            series["diameter_fraction"].append(0.0)
        else:
            largest_component_undirected = largest_component.to_undirected()
            diameter = nx.diameter(largest_component_undirected)
            series["diameter_fraction"].append(diameter / float(max(1, num_nodes - 1)))

    return series


def build_shifted_graph_targets(
    node_df: pd.DataFrame,
    G: nx.DiGraph,
    prediction_horizon: int,
) -> dict[str, torch.Tensor]:
    target_series = compute_graph_target_series(G, node_df)
    num_steps = len(next(iter(target_series.values())))
    usable_steps = num_steps - prediction_horizon
    if usable_steps <= 0:
        raise ValueError("prediction_horizon is too large to build graph-level targets.")

    shifted = {}
    for target_name, values in target_series.items():
        shifted[target_name] = torch.tensor(values[prediction_horizon:], dtype=torch.float32)
    return shifted


def build_lcc_trajectory_targets(
    node_df: pd.DataFrame,
    G: nx.DiGraph,
    prediction_horizon: int,
) -> torch.Tensor:
    target_series = compute_graph_target_series(G, node_df)["lcc_fraction"]
    num_steps = len(target_series)
    usable_steps = num_steps - prediction_horizon
    if usable_steps <= 0:
        raise ValueError("prediction_horizon is too large to build graph-level trajectory targets.")

    trajectories = []
    for t in range(usable_steps):
        trajectories.append(target_series[t + 1 : t + prediction_horizon + 1])
    return torch.tensor(trajectories, dtype=torch.float32)


def build_tensors(
    node_csv: Path,
    out_dir: Path,
    feature_flags: dict[str, bool],
    prediction_horizon: int,
):
    df = pd.read_csv(node_csv).sort_values(["t", "node"])
    X, Y, feature_names = build_saved_feature_tensors(
        df, feature_flags, prediction_horizon=prediction_horizon
    )
    torch.save(X, out_dir / "X_v1.pt")
    torch.save(Y, out_dir / "Y_v1.pt")
    return X, Y, feature_names


def save_graph_png(G, out_path: Path, graph_type: str, seed: int) -> None:
    if graph_type == "chain":
        pos = {n: (n, 0) for n in sorted(G.nodes())}
    else:
        pos = nx.spring_layout(G, seed=seed)

    labels = {n: f"{n}\n{G.nodes[n]['role']}" for n in G.nodes()}
    plt.figure(figsize=(max(6, len(G.nodes()) * 1.2), 2.6))
    nx.draw_networkx_nodes(G, pos, node_size=1800)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=28,
        width=2.2,
        min_source_margin=20,
        min_target_margin=24,
        connectionstyle="arc3,rad=0.02",
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--graph_type", type=str, default="chain")
    parser.add_argument("--graph_tag", type=str, default="")
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--p_shock", type=float, default=0.05)
    parser.add_argument("--p_propagate", type=float, default=0.35)
    parser.add_argument("--p_recover", type=float, default=0.25)
    parser.add_argument("--prediction_horizon", type=int, default=1)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use_health", type=parse_bool_flag, default=True)
    parser.add_argument("--use_exposure", type=parse_bool_flag, default=True)
    parser.add_argument("--use_time_to_recovery", type=parse_bool_flag, default=True)
    parser.add_argument("--use_betweenness", type=parse_bool_flag, default=True)
    args = parser.parse_args()

    graph_tag = args.graph_tag or f"N{args.num_nodes}_{args.graph_type}"
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_flags = normalize_feature_flags(
        {
            "use_health": args.use_health,
            "use_exposure": args.use_exposure,
            "use_time_to_recovery": args.use_time_to_recovery,
            "use_betweenness": args.use_betweenness,
        }
    )

    G = build_graph(args.num_nodes, args.graph_type, seed=args.seed)
    params = SimParams(
        T=args.T,
        seed=args.seed,
        p_shock=args.p_shock,
        p_propagate=args.p_propagate,
        p_recover=args.p_recover,
    )

    node_df, sys_df = simulate(G, params)
    node_csv = out_dir / "node_observables.csv"
    sys_csv = out_dir / "system_observables.csv"
    node_df.to_csv(node_csv, index=False)
    sys_df.to_csv(sys_csv, index=False)

    X, Y, saved_feature_names = build_tensors(
        node_csv,
        out_dir,
        feature_flags,
        prediction_horizon=args.prediction_horizon,
    )
    graph_targets = build_shifted_graph_targets(
        node_df, G, prediction_horizon=args.prediction_horizon
    )
    graph_target_files = {
        "lcc_fraction": "Y_lcc_v1.pt",
        "component_fraction": "Y_components_v1.pt",
        "diameter_fraction": "Y_diameter_v1.pt",
        "edge_survival_ratio": "Y_edge_survival_v1.pt",
    }
    for target_name, file_name in graph_target_files.items():
        torch.save(graph_targets[target_name], out_dir / file_name)
    Y_lcc_traj = build_lcc_trajectory_targets(
        node_df, G, prediction_horizon=args.prediction_horizon
    )
    torch.save(Y_lcc_traj, out_dir / "Y_lcc_traj_v1.pt")
    directed_edge_index = to_directed_edge_index(G)
    message_passing_edge_index = to_message_passing_edge_index(G)
    torch.save(directed_edge_index, out_dir / "edge_index.pt")
    torch.save(directed_edge_index, out_dir / "edge_index_directed.pt")
    torch.save(message_passing_edge_index, out_dir / "edge_index_message_passing.pt")
    graph_png = out_dir / "supply_chain_graph.png"
    save_graph_png(G, graph_png, args.graph_type, args.seed)

    meta = {
        "graph_tag": graph_tag,
        "graph_type": args.graph_type,
        "num_nodes": args.num_nodes,
        "num_edges_physical": int(G.number_of_edges()),
        "num_edges_directed": int(directed_edge_index.shape[1]),
        "num_edges_message_passing": int(message_passing_edge_index.shape[1]),
        "edge_index_default": "edge_index.pt",
        "edge_index_directed": "edge_index_directed.pt",
        "edge_index_message_passing": "edge_index_message_passing.pt",
        "roles": [G.nodes[i]["role"] for i in sorted(G.nodes())],
        "sim_params": {
            "T": args.T,
            "seed": args.seed,
            "p_shock": args.p_shock,
            "p_propagate": args.p_propagate,
            "p_recover": args.p_recover,
        },
        "target_definition": {
            "prediction_horizon": args.prediction_horizon,
            "node_target_name": "future_health",
            "node_label_semantics": f"Y[t, node] = health at t+{args.prediction_horizon}",
            "graph_target_name": "future_lcc_fraction",
            "graph_label_semantics": (
                f"Y_lcc[t] = LCC fraction of the healthy-node subgraph at t+{args.prediction_horizon}"
            ),
            "available_graph_targets": {
                "lcc_fraction": (
                    f"LCC fraction of the healthy-node subgraph at t+{args.prediction_horizon}"
                ),
                "component_fraction": (
                    f"Number of weakly connected healthy components divided by N at t+{args.prediction_horizon}"
                ),
                "diameter_fraction": (
                    f"Diameter of the largest healthy component divided by (N-1) at t+{args.prediction_horizon}"
                ),
                "edge_survival_ratio": (
                    f"Healthy-subgraph edge count divided by original edge count at t+{args.prediction_horizon}"
                ),
            },
            "graph_trajectory_target_name": "future_lcc_trajectory",
            "graph_trajectory_label_semantics": (
                f"Y_lcc_traj[t, h] = LCC fraction at t+(h+1) for h=0..{args.prediction_horizon-1}"
            ),
        },
        "feature_flags": feature_flags,
        "feature_order": feature_order_documentation(),
        "saved_feature_list": saved_feature_names,
        "betweenness_handling": (
            "betweenness_centrality is appended during trainer input preparation when "
            "enabled, rather than being saved directly into X_v1.pt."
        ),
        "tensor_shapes": {
            "X": list(X.shape),
            "Y": list(Y.shape),
            "Y_lcc": list(graph_targets["lcc_fraction"].shape),
            "Y_components": list(graph_targets["component_fraction"].shape),
            "Y_diameter": list(graph_targets["diameter_fraction"].shape),
            "Y_edge_survival": list(graph_targets["edge_survival_ratio"].shape),
            "Y_lcc_traj": list(Y_lcc_traj.shape),
        },
        "graph_target_files": graph_target_files,
        "graph_png": str(graph_png).replace("\\", "/"),
    }
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()
