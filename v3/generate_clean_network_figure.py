import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graph_factory import build_graph


def compute_tier_levels(G: nx.DiGraph) -> dict[int, int]:
    return dict(nx.single_source_shortest_path_length(G, 0))


def compute_layered_positions(G: nx.DiGraph, levels: dict[int, int]) -> dict[int, tuple[float, float]]:
    layers: dict[int, list[int]] = {}
    for node, level in levels.items():
        layers.setdefault(level, []).append(node)

    pos: dict[int, tuple[float, float]] = {}
    max_level = max(layers)
    for level, nodes in sorted(layers.items()):
        nodes = sorted(nodes, key=lambda n: (G.out_degree(n), -G.in_degree(n), n), reverse=True)
        count = len(nodes)
        for i, node in enumerate(nodes):
            y = 0.0 if count == 1 else 1.0 - (2.0 * i / (count - 1))
            pos[node] = (level / max(1, max_level), y)
    return pos


def draw_clean_network(G: nx.DiGraph, out_path: Path, title: str) -> None:
    levels = compute_tier_levels(G)
    pos = compute_layered_positions(G, levels)

    palette = {
        0: "#1f4e79",
        1: "#4f81bd",
        2: "#9bbb59",
        3: "#c0504d",
        4: "#8064a2",
    }
    node_colors = [palette.get(levels[n], "#7f7f7f") for n in G.nodes()]
    node_sizes = [320 + 80 * G.out_degree(n) + 60 * G.in_degree(n) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#9aa4b2",
        width=1.5,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        min_source_margin=10,
        min_target_margin=12,
        connectionstyle="arc3,rad=0.05",
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=1.2,
    )

    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_weight="bold",
        font_color="white",
    )

    unique_levels = sorted(set(levels.values()))
    for level in unique_levels:
        x = level / max(1, max(unique_levels))
        ax.text(
            x,
            1.12,
            f"Tier {level + 1}",
            transform=ax.transData,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#334155",
        )

    ax.set_title(title, fontsize=13, pad=18)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--graph_type", type=str, default="tiered_scale_free")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    G = build_graph(args.num_nodes, args.graph_type, seed=args.seed)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Synthetic Directed {args.graph_type.replace('_', ' ').title()} Network (N={args.num_nodes})"
    draw_clean_network(G, out_path, title)


if __name__ == "__main__":
    main()
