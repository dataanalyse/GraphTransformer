from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml


def load_allowed_dependency_pairs(config_path: Path) -> set[tuple[str, str]]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {
        (rule["src_type"], rule["dst_type"])
        for rule in config.get("dependency_rules", [])
    }


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in nodes.iterrows():
        g.add_node(row["node_id"], **row.to_dict())
    for _, row in edges.iterrows():
        g.add_edge(row["src_node_id"], row["dst_node_id"], **row.to_dict())
    return g


def compute_validation(nodes: pd.DataFrame, edges: pd.DataFrame, allowed_pairs: set[tuple[str, str]]) -> tuple[pd.DataFrame, str]:
    duplicate_node_ids_df = nodes[nodes.duplicated(subset=["node_id"], keep=False)].copy()
    duplicate_node_ids = sorted(duplicate_node_ids_df["node_id"].unique().tolist())
    node_ids = set(nodes["node_id"])
    src_missing = sorted(set(edges["src_node_id"]) - node_ids)
    dst_missing = sorted(set(edges["dst_node_id"]) - node_ids)

    weights_valid = edges["weight"].between(0, 1, inclusive="both").all()
    # Zero-delay dependencies are allowed for immediate local support links.
    delays_nonnegative_int = ((edges["delay"] >= 0) & (edges["delay"].astype(int) == edges["delay"])).all()
    duplicate_edge_mask = edges.duplicated(subset=["src_node_id", "dst_node_id"], keep=False)
    duplicate_edges = edges.loc[duplicate_edge_mask, ["src_node_id", "dst_node_id"]].drop_duplicates()
    self_loops = edges[edges["src_node_id"] == edges["dst_node_id"]]

    edges["dependency_type"] = edges["src_type"] + " -> " + edges["dst_type"]
    impossible_pairs = sorted(
        set(zip(edges["src_type"], edges["dst_type"])) - allowed_pairs
    )

    g = build_graph(nodes, edges)
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    avg_in_degree = sum(dict(g.in_degree()).values()) / num_nodes if num_nodes else 0.0
    avg_out_degree = sum(dict(g.out_degree()).values()) / num_nodes if num_nodes else 0.0
    isolated_nodes = sorted(nx.isolates(g))
    weak_components = list(nx.weakly_connected_components(g))
    weak_component_sizes = sorted((len(c) for c in weak_components), reverse=True)
    lcc_size = weak_component_sizes[0] if weak_component_sizes else 0

    node_counts = nodes.drop_duplicates(subset=["node_id"])["node_type"].value_counts().sort_index()
    edge_counts = edges["dependency_type"].value_counts().sort_index()

    in_degree_series = pd.Series(dict(g.in_degree()), name="in_degree").sort_values(ascending=False)
    out_degree_series = pd.Series(dict(g.out_degree()), name="out_degree").sort_values(ascending=False)

    node_lookup = nodes.drop_duplicates(subset=["node_id"]).set_index("node_id")

    top_in = (
        in_degree_series.head(10)
        .rename_axis("node_id")
        .reset_index()
        .merge(node_lookup[["name", "node_type", "layer_key"]], left_on="node_id", right_index=True, how="left")
    )
    top_out = (
        out_degree_series.head(10)
        .rename_axis("node_id")
        .reset_index()
        .merge(node_lookup[["name", "node_type", "layer_key"]], left_on="node_id", right_index=True, how="left")
    )
    top_weight = (
        edges.sort_values(["weight", "distance_km"], ascending=[False, True])
        .head(10)[["src_node_id", "dst_node_id", "src_type", "dst_type", "weight", "delay", "distance_km"]]
    )

    stats_rows: list[dict] = [
        {"section": "basic", "metric": "num_nodes", "group": "", "value": num_nodes},
        {"section": "basic", "metric": "num_edges", "group": "", "value": num_edges},
        {"section": "basic", "metric": "average_in_degree", "group": "", "value": round(avg_in_degree, 4)},
        {"section": "basic", "metric": "average_out_degree", "group": "", "value": round(avg_out_degree, 4)},
        {"section": "basic", "metric": "isolated_nodes", "group": "", "value": len(isolated_nodes)},
        {"section": "basic", "metric": "weakly_connected_components", "group": "", "value": len(weak_components)},
        {"section": "basic", "metric": "largest_connected_component_size", "group": "", "value": lcc_size},
        {"section": "sanity", "metric": "missing_source_ids", "group": "", "value": len(src_missing)},
        {"section": "sanity", "metric": "missing_target_ids", "group": "", "value": len(dst_missing)},
        {"section": "sanity", "metric": "duplicate_node_ids", "group": "", "value": len(duplicate_node_ids)},
        {"section": "sanity", "metric": "weights_between_0_and_1", "group": "", "value": int(weights_valid)},
        {"section": "sanity", "metric": "delays_nonnegative_integers", "group": "", "value": int(delays_nonnegative_int)},
        {"section": "sanity", "metric": "duplicate_edges", "group": "", "value": len(duplicate_edges)},
        {"section": "sanity", "metric": "self_loops", "group": "", "value": len(self_loops)},
        {"section": "sanity", "metric": "impossible_dependency_type_pairs", "group": "", "value": len(impossible_pairs)},
    ]

    for node_type, count in node_counts.items():
        stats_rows.append({"section": "node_counts", "metric": "nodes_by_asset_type", "group": node_type, "value": int(count)})
    for dep_type, count in edge_counts.items():
        stats_rows.append({"section": "edge_counts", "metric": "edges_by_dependency_type", "group": dep_type, "value": int(count)})

    stats_df = pd.DataFrame(stats_rows)

    plausibility_lines = []
    if src_missing or dst_missing:
        plausibility_lines.append("- The graph has broken references between edge endpoints and the node table.")
    else:
        plausibility_lines.append("- All edges reference valid node IDs in the node table.")
    if duplicate_node_ids:
        plausibility_lines.append(
            f"- The node table contains `{len(duplicate_node_ids)}` duplicated node IDs, currently concentrated in the cellular tower layer, so some assets are being collapsed into the same graph node."
        )
    if not weights_valid:
        plausibility_lines.append("- Some weights fall outside the expected [0, 1] range.")
    if not delays_nonnegative_int:
        plausibility_lines.append("- Some delays are not non-negative integers.")
    if len(duplicate_edges):
        plausibility_lines.append(f"- There are `{len(duplicate_edges)}` duplicate edge pairs that should likely be removed.")
    if len(self_loops):
        plausibility_lines.append(f"- There are `{len(self_loops)}` self-loops, which are unusual for this dependency setting.")
    if len(impossible_pairs):
        plausibility_lines.append("- Some dependency directions are not present in the configured rule set.")
    if not isolated_nodes:
        plausibility_lines.append("- There are no isolated nodes among the included point-like assets.")
    else:
        plausibility_lines.append(f"- There are `{len(isolated_nodes)}` isolated nodes, suggesting some assets are not yet integrated into the dependency rules.")
    plausibility_lines.append(
        "- The current graph is a plausible first-pass proximity-based infrastructure dependency graph, but it remains a modeled approximation rather than a validated operational dependency network."
    )

    md_lines = [
        "# Graph Validation Summary",
        "",
        "## 1. Basic Graph Statistics",
        "",
        f"- Number of nodes: `{num_nodes}`",
        f"- Number of edges: `{num_edges}`",
        f"- Average in-degree: `{avg_in_degree:.4f}`",
        f"- Average out-degree: `{avg_out_degree:.4f}`",
        f"- Isolated nodes: `{len(isolated_nodes)}`",
        f"- Weakly connected components: `{len(weak_components)}`",
        f"- Largest connected component size: `{lcc_size}`",
        "",
        "### Node Counts by Asset Type",
        "",
    ]
    for node_type, count in node_counts.items():
        md_lines.append(f"- `{node_type}`: `{count}`")

    md_lines.extend(["", "### Edge Counts by Dependency Type", ""])
    for dep_type, count in edge_counts.items():
        md_lines.append(f"- `{dep_type}`: `{count}`")

    md_lines.extend(
        [
            "",
            "## 2. Sanity Checks",
            "",
            f"- All edge source IDs present in node table: `{'yes' if not src_missing else 'no'}`",
            f"- All edge target IDs present in node table: `{'yes' if not dst_missing else 'no'}`",
            f"- Duplicate node IDs in node table: `{len(duplicate_node_ids)}`",
            f"- Weights between 0 and 1: `{'yes' if weights_valid else 'no'}`",
            f"- Delays are non-negative integers: `{'yes' if delays_nonnegative_int else 'no'}`",
            f"- Duplicate edges: `{len(duplicate_edges)}`",
            f"- Self-loops: `{len(self_loops)}`",
            f"- Impossible dependency directions: `{len(impossible_pairs)}`",
            "",
            "## 3. Interpretability",
            "",
            "### Top 10 Highest In-Degree Nodes",
            "",
        ]
    )
    for _, row in top_in.iterrows():
        md_lines.append(
            f"- `{row['node_id']}` | `{row['name']}` | `{row['node_type']}` | in-degree `{int(row['in_degree'])}`"
        )

    md_lines.extend(["", "### Top 10 Highest Out-Degree Nodes", ""])
    for _, row in top_out.iterrows():
        md_lines.append(
            f"- `{row['node_id']}` | `{row['name']}` | `{row['node_type']}` | out-degree `{int(row['out_degree'])}`"
        )

    md_lines.extend(["", "### Top 10 Strongest-Weight Edges", ""])
    for _, row in top_weight.iterrows():
        md_lines.append(
            f"- `{row['src_node_id']} -> {row['dst_node_id']}` | `{row['src_type']} -> {row['dst_type']}` | weight `{row['weight']}` | delay `{int(row['delay'])}` | distance `{row['distance_km']}` km"
        )

    md_lines.extend(["", "### Plausibility Assessment", ""])
    md_lines.extend(plausibility_lines)

    return stats_df, "\n".join(md_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the v3 dependency graph.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--config", default="v3/configs/infrastructure_layers.yaml")
    parser.add_argument("--summary-out", default="v3/data/processed/graph_validation_summary.md")
    parser.add_argument("--stats-out", default="v3/data/processed/graph_validation_stats.csv")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    allowed_pairs = load_allowed_dependency_pairs(Path(args.config))

    stats_df, summary_md = compute_validation(nodes, edges, allowed_pairs)

    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.stats_out, index=False)
    Path(args.summary_out).write_text(summary_md, encoding="utf-8")

    print(f"Wrote summary to {args.summary_out}")
    print(f"Wrote stats to {args.stats_out}")


if __name__ == "__main__":
    main()
