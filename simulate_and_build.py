import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch

from graph_factory import build_graph, to_message_passing_edge_index
from sc_sim_v0 import SimParams, simulate


def build_tensors(node_csv: Path, out_dir: Path):
    df = pd.read_csv(node_csv).sort_values(["t", "node"])
    feature_cols = ["health", "exposure", "time_to_recovery"]

    T = df["t"].nunique()
    N = df["node"].nunique()
    F = len(feature_cols)

    X = torch.zeros(T - 1, N, F, dtype=torch.float32)
    Y = torch.zeros(T - 1, N, dtype=torch.float32)

    for t in range(T - 1):
        df_t = df[df["t"] == t]
        df_t1 = df[df["t"] == t + 1]
        X[t] = torch.tensor(df_t[feature_cols].values, dtype=torch.float32)
        Y[t] = torch.tensor(df_t1["health"].values, dtype=torch.float32)

    torch.save(X, out_dir / "X_v1.pt")
    torch.save(Y, out_dir / "Y_v1.pt")
    return X, Y


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
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    graph_tag = args.graph_tag or f"N{args.num_nodes}_{args.graph_type}"
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    X, Y = build_tensors(node_csv, out_dir)
    edge_index = to_message_passing_edge_index(G)
    torch.save(edge_index, out_dir / "edge_index.pt")
    graph_png = out_dir / "supply_chain_graph.png"
    save_graph_png(G, graph_png, args.graph_type, args.seed)

    meta = {
        "graph_tag": graph_tag,
        "graph_type": args.graph_type,
        "num_nodes": args.num_nodes,
        "num_edges_physical": int(G.number_of_edges()),
        "num_edges_message_passing": int(edge_index.shape[1]),
        "roles": [G.nodes[i]["role"] for i in sorted(G.nodes())],
        "sim_params": {
            "T": args.T,
            "seed": args.seed,
            "p_shock": args.p_shock,
            "p_propagate": args.p_propagate,
            "p_recover": args.p_recover,
        },
        "tensor_shapes": {
            "X": list(X.shape),
            "Y": list(Y.shape),
        },
        "graph_png": str(graph_png).replace("\\", "/"),
    }
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote dataset to: {out_dir}")


if __name__ == "__main__":
    main()
