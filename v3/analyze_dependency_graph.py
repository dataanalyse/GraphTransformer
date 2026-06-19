from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


NODE_COLORS = {
    "hospital": "#2a9d8f",
    "school": "#7aa6c2",
    "telecom": "#8b6fb3",
    "power": "#d17c2f",
    "ems_fire": "#c84d4d",
    "emergency_management": "#222222",
}

DISPLAY_LABELS = {
    "hospital": "Hospitals",
    "school": "Schools",
    "telecom": "Telecom",
    "power": "Power",
    "ems_fire": "EMS / Fire",
    "emergency_management": "Emergency Mgmt",
}

FOCUS_TYPES = [
    "power",
    "telecom",
    "hospital",
    "school",
    "ems_fire",
    "emergency_management",
]


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in nodes.iterrows():
        g.add_node(row["node_id"], **row.to_dict())
    for _, row in edges.iterrows():
        g.add_edge(row["src_node_id"], row["dst_node_id"], **row.to_dict())
    return g


def compute_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[nx.DiGraph, dict, pd.DataFrame]:
    g = build_graph(nodes, edges)
    undirected = g.to_undirected()

    degree = dict(g.degree())
    in_degree = dict(g.in_degree())
    out_degree = dict(g.out_degree())
    betweenness = nx.betweenness_centrality(g, normalized=True)

    weak_components = list(nx.weakly_connected_components(g))
    strong_components = list(nx.strongly_connected_components(g))
    largest_weak_component = max((len(c) for c in weak_components), default=0)

    node_metrics = nodes.copy()
    node_metrics["degree"] = node_metrics["node_id"].map(degree)
    node_metrics["in_degree"] = node_metrics["node_id"].map(in_degree)
    node_metrics["out_degree"] = node_metrics["node_id"].map(out_degree)
    node_metrics["betweenness"] = node_metrics["node_id"].map(betweenness)

    articulation_points = set(nx.articulation_points(undirected))
    bridges = {frozenset(edge) for edge in nx.bridges(undirected)}
    node_metrics["is_articulation_point"] = node_metrics["node_id"].isin(articulation_points)

    summary = {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "num_weak_components": len(weak_components),
        "num_strong_components": len(strong_components),
        "largest_weak_component_size": largest_weak_component,
        "average_degree": sum(degree.values()) / len(degree) if degree else 0.0,
        "average_in_degree": sum(in_degree.values()) / len(in_degree) if in_degree else 0.0,
        "average_out_degree": sum(out_degree.values()) / len(out_degree) if out_degree else 0.0,
        "isolated_nodes": int(sum(1 for _, d in g.degree() if d == 0)),
        "num_articulation_points": len(articulation_points),
        "num_bridges": len(bridges),
    }
    return g, summary, node_metrics


def graph_metrics_table(g: nx.DiGraph, node_metrics: pd.DataFrame, edges: pd.DataFrame, summary: dict) -> pd.DataFrame:
    rows: list[dict] = []

    for key, value in summary.items():
        rows.append({"section": "graph_summary", "metric": key, "group": "", "value": value})

    node_counts = node_metrics["node_type"].value_counts().sort_index()
    for node_type, count in node_counts.items():
        rows.append({"section": "node_counts", "metric": "node_count_by_type", "group": node_type, "value": int(count)})

    degree_by_type = node_metrics.groupby("node_type")["degree"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    for _, row in degree_by_type.iterrows():
        for stat in ["count", "mean", "median", "min", "max"]:
            rows.append(
                {
                    "section": "degree_distribution",
                    "metric": f"degree_{stat}",
                    "group": row["node_type"],
                    "value": row[stat],
                }
            )

    dependency_type_counts = (edges["src_type"] + " -> " + edges["dst_type"]).value_counts().sort_index()
    for dep_type, count in dependency_type_counts.items():
        rows.append({"section": "edge_counts", "metric": "edge_count_by_dependency_type", "group": dep_type, "value": int(count)})

    for metric in ["weight", "delay"]:
        rows.extend(
            [
                {"section": "edge_summary", "metric": f"{metric}_min", "group": "", "value": edges[metric].min()},
                {"section": "edge_summary", "metric": f"{metric}_max", "group": "", "value": edges[metric].max()},
                {"section": "edge_summary", "metric": f"{metric}_mean", "group": "", "value": edges[metric].mean()},
                {"section": "edge_summary", "metric": f"{metric}_median", "group": "", "value": edges[metric].median()},
            ]
        )

    top_degree = node_metrics.sort_values(["degree", "betweenness"], ascending=[False, False]).head(10)
    for rank, (_, row) in enumerate(top_degree.iterrows(), start=1):
        rows.append({"section": "top_nodes", "metric": "highest_degree_node", "group": rank, "value": row["node_id"]})

    return pd.DataFrame(rows)


def label_name(name: str, fallback: str) -> str:
    if pd.isna(name) or not str(name).strip():
        return fallback
    return str(name)[:40]


def plot_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    focus_only: bool = False,
) -> None:
    plot_nodes = nodes.copy()
    plot_edges = edges.copy()
    if focus_only:
        plot_nodes = plot_nodes[plot_nodes["node_type"].isin(FOCUS_TYPES)].copy()
        node_ids = set(plot_nodes["node_id"])
        plot_edges = plot_edges[plot_edges["src_node_id"].isin(node_ids) & plot_edges["dst_node_id"].isin(node_ids)].copy()

    merged_nodes = plot_nodes.merge(
        node_metrics[["node_id", "degree", "betweenness"]],
        on="node_id",
        how="left",
    )
    pos = {row["node_id"]: (row["longitude"], row["latitude"]) for _, row in merged_nodes.iterrows()}

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_facecolor("white")

    for _, edge in plot_edges.iterrows():
        src = pos.get(edge["src_node_id"])
        dst = pos.get(edge["dst_node_id"])
        if src is None or dst is None:
            continue
        ax.plot(
            [src[0], dst[0]],
            [src[1], dst[1]],
            color="#6b7280",
            alpha=0.12 if not focus_only else 0.2,
            linewidth=0.35 + 1.3 * float(edge["weight"]),
            zorder=1,
        )

    for node_type in [t for t in FOCUS_TYPES if t in set(merged_nodes["node_type"])]:
        layer = merged_nodes[merged_nodes["node_type"] == node_type]
        if layer.empty:
            continue
        sizes = 18 + layer["degree"].fillna(0) * (10 if focus_only else 6)
        ax.scatter(
            layer["longitude"],
            layer["latitude"],
            s=sizes,
            c=NODE_COLORS.get(node_type, "#999999"),
            edgecolors="white",
            linewidths=0.4,
            alpha=0.92,
            label=DISPLAY_LABELS.get(node_type, node_type),
            zorder=2,
        )

    label_count = 15 if not focus_only else 20
    important = merged_nodes.sort_values(["degree", "betweenness"], ascending=[False, False]).head(label_count)
    for _, row in important.iterrows():
        ax.text(
            row["longitude"] + 0.004,
            row["latitude"] + 0.002,
            label_name(row["name"], row["node_id"]),
            fontsize=7.5 if not focus_only else 8.0,
            color="#222222",
            zorder=3,
        )

    xmin = merged_nodes["longitude"].min()
    xmax = merged_nodes["longitude"].max()
    ymin = merged_nodes["latitude"].min()
    ymax = merged_nodes["latitude"].max()
    xpad = max((xmax - xmin) * 0.08, 0.01)
    ypad = max((ymax - ymin) * 0.08, 0.01)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=15, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.28)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def research_summary(
    g: nx.DiGraph,
    node_metrics: pd.DataFrame,
    edges: pd.DataFrame,
    summary: dict,
) -> str:
    undirected = g.to_undirected()
    articulation_points = set(nx.articulation_points(undirected))
    bridges = list(nx.bridges(undirected))

    top_degree = node_metrics.sort_values(["degree", "betweenness"], ascending=[False, False]).head(10)
    top_betweenness = node_metrics.sort_values(["betweenness", "degree"], ascending=[False, False]).head(10)
    isolated = node_metrics[node_metrics["degree"] == 0].copy()

    edge_types = (edges["src_type"] + " -> " + edges["dst_type"]).value_counts()
    dominant_types = edge_types.head(5)
    central_sector_counts = node_metrics.groupby("node_type")["degree"].mean().sort_values(ascending=False)

    articulation_df = node_metrics[node_metrics["node_id"].isin(articulation_points)].sort_values(
        ["betweenness", "degree"], ascending=[False, False]
    )

    lines = [
        "# Graph Research Summary",
        "",
        "## Graph Summary",
        "",
        f"- Total nodes: `{summary['num_nodes']}`",
        f"- Total edges: `{summary['num_edges']}`",
        f"- Weakly connected components: `{summary['num_weak_components']}`",
        f"- Strongly connected components: `{summary['num_strong_components']}`",
        f"- Largest weakly connected component size: `{summary['largest_weak_component_size']}`",
        f"- Average degree: `{summary['average_degree']:.4f}`",
        f"- Isolated nodes: `{summary['isolated_nodes']}`",
        "",
        "## Node Analysis",
        "",
        "### Node Counts by Infrastructure Type",
        "",
    ]
    for node_type, count in node_metrics["node_type"].value_counts().sort_index().items():
        lines.append(f"- `{node_type}`: `{count}`")

    lines.extend(["", "### Top 10 Highest-Degree Nodes", ""])
    for _, row in top_degree.iterrows():
        lines.append(
            f"- `{row['node_id']}` | `{label_name(row['name'], row['node_id'])}` | `{row['node_type']}` | degree `{int(row['degree'])}` | betweenness `{row['betweenness']:.4f}`"
        )

    lines.extend(["", "### Isolated Nodes", ""])
    if isolated.empty:
        lines.append("- No isolated nodes.")
    else:
        for _, row in isolated.iterrows():
            lines.append(f"- `{row['node_id']}` | `{label_name(row['name'], row['node_id'])}` | `{row['node_type']}`")

    lines.extend(["", "## Edge Analysis", "", "### Edge Counts by Dependency Type", ""])
    for dep_type, count in edge_types.sort_index().items():
        lines.append(f"- `{dep_type}`: `{count}`")

    lines.extend(
        [
            "",
            "### Weight Summary",
            "",
            f"- min: `{edges['weight'].min():.4f}`",
            f"- max: `{edges['weight'].max():.4f}`",
            f"- mean: `{edges['weight'].mean():.4f}`",
            f"- median: `{edges['weight'].median():.4f}`",
            "",
            "### Delay Summary",
            "",
            f"- min: `{edges['delay'].min():.4f}`",
            f"- max: `{edges['delay'].max():.4f}`",
            f"- mean: `{edges['delay'].mean():.4f}`",
            f"- median: `{edges['delay'].median():.4f}`",
            "",
            "## Fragility Assessment",
            "",
            f"- Critical hubs (high degree) are concentrated around `{top_degree.iloc[0]['node_type']}` and power-support relationships.",
            f"- Number of articulation points: `{len(articulation_points)}`",
            f"- Number of bridge edges: `{len(bridges)}`",
            "",
            "### Likely Critical Hubs",
            "",
        ]
    )
    for _, row in top_degree.head(5).iterrows():
        lines.append(f"- `{label_name(row['name'], row['node_id'])}` | `{row['node_type']}` | degree `{int(row['degree'])}`")

    lines.extend(["", "### Likely Bottlenecks / Bridge Nodes", ""])
    for _, row in top_betweenness.head(5).iterrows():
        lines.append(
            f"- `{label_name(row['name'], row['node_id'])}` | `{row['node_type']}` | betweenness `{row['betweenness']:.4f}`"
        )

    lines.extend(["", "### Articulation Points / Single Points of Failure", ""])
    if articulation_df.empty:
        lines.append("- No articulation points detected in the undirected backbone.")
    else:
        for _, row in articulation_df.head(10).iterrows():
            lines.append(
                f"- `{label_name(row['name'], row['node_id'])}` | `{row['node_type']}` | degree `{int(row['degree'])}` | betweenness `{row['betweenness']:.4f}`"
            )

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            f"- **Does the graph appear realistic?** Broadly yes as a first-pass modeled dependency graph: hospitals, schools, EMS/fire, telecom, power, and emergency management are all represented with geographically plausible dependency links. It is still a simplified approximation rather than a validated operational network.",
            f"- **What infrastructure sectors are most central?** The highest average degree and most dominant dependency counts are associated with `power`, `telecom`, and the large `school` layer, with power acting as the main shared dependency backbone.",
            f"- **What nodes are likely to trigger the largest cascades?** Nodes with the highest degree and betweenness, especially key power facilities and the duplicated telecom hubs, are the strongest cascade candidates because many institutions route through them.",
            f"- **What infrastructure dependencies dominate the network?** The graph is dominated by `school -> power`, `school -> telecom`, `school -> ems_fire`, and medical dependencies on power and EMS/fire.",
            f"- **Is the graph suitable for cascading failure simulation?** Yes, it is suitable for a first simulator prototype because it is connected enough, multi-sector, and weighted with delays. It already expresses plausible directional vulnerability structure.",
            f"- **What should be improved before simulator development?** The main cleanup items are resolving duplicated telecom node IDs, deduplicating repeated telecom-to-power edges, and eventually enriching dependencies beyond proximity alone with stronger domain-informed or road/access-based rules.",
            "",
            "## Degree Distribution by Node Type",
            "",
        ]
    )
    degree_stats = node_metrics.groupby("node_type")["degree"].agg(["count", "mean", "median", "min", "max"]).sort_values("mean", ascending=False)
    for node_type, row in degree_stats.iterrows():
        lines.append(
            f"- `{node_type}`: count `{int(row['count'])}`, mean `{row['mean']:.2f}`, median `{row['median']:.2f}`, min `{row['min']:.0f}`, max `{row['max']:.0f}`"
        )

    lines.extend(["", "## Dominant Dependency Types", ""])
    for dep_type, count in dominant_types.items():
        lines.append(f"- `{dep_type}`: `{count}`")

    lines.extend(["", "## Sector Centrality Ranking (Mean Degree)", ""])
    for node_type, value in central_sector_counts.items():
        lines.append(f"- `{node_type}`: `{value:.2f}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize the v3 dependency graph.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--overview-out", default="v3/runs/figures/graph_overview.png")
    parser.add_argument("--dependency-view-out", default="v3/runs/figures/graph_dependency_view.png")
    parser.add_argument("--summary-out", default="v3/data/processed/graph_research_summary.md")
    parser.add_argument("--metrics-out", default="v3/data/processed/graph_metrics.csv")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    nodes = nodes.drop_duplicates(subset=["node_id"]).copy()
    g, summary, node_metrics = compute_metrics(nodes, edges)
    metrics_df = graph_metrics_table(g, node_metrics, edges, summary)
    summary_text = research_summary(g, node_metrics, edges, summary)

    plot_graph(nodes, edges, node_metrics, Path(args.overview_out), "Dependency Graph Overview", focus_only=False)
    plot_graph(nodes, edges, node_metrics, Path(args.dependency_view_out), "Infrastructure Dependency View", focus_only=True)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_out, index=False)
    Path(args.summary_out).write_text(summary_text, encoding="utf-8")

    print(f"Wrote overview figure to {args.overview_out}")
    print(f"Wrote dependency view figure to {args.dependency_view_out}")
    print(f"Wrote research summary to {args.summary_out}")
    print(f"Wrote metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
