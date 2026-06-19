from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import yaml


POINT_LAYERS = [
    "hospitals",
    "ems_fire",
    "power_plants",
    "county_telecom",
    "local_eoc",
    "public_schools",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def layer_key_to_node_type(layer_key: str) -> str:
    mapping = {
        "hospitals": "hospital",
        "ems_fire": "ems_fire",
        "power_plants": "power",
        "county_telecom": "telecom",
        "local_eoc": "emergency_management",
        "public_schools": "school",
    }
    return mapping.get(layer_key, layer_key)


def build_edges(nodes: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    typed_nodes = nodes.copy()
    typed_nodes["node_type"] = typed_nodes["layer_key"].map(layer_key_to_node_type)

    edges: list[dict] = []
    for rule in rules:
        src_nodes = typed_nodes[typed_nodes["node_type"] == rule["src_type"]]
        dst_nodes = typed_nodes[typed_nodes["node_type"] == rule["dst_type"]]
        if src_nodes.empty or dst_nodes.empty:
            continue

        max_targets = int(rule.get("max_targets", 1))
        max_distance_km = float(rule.get("max_distance_km", 999999))

        for _, src in src_nodes.iterrows():
            candidates: list[tuple[float, pd.Series]] = []
            for _, dst in dst_nodes.iterrows():
                if src["node_id"] == dst["node_id"]:
                    continue
                dist_km = haversine_km(src["latitude"], src["longitude"], dst["latitude"], dst["longitude"])
                if dist_km <= max_distance_km:
                    candidates.append((dist_km, dst))

            candidates.sort(key=lambda x: x[0])
            for dist_km, dst in candidates[:max_targets]:
                edges.append(
                    {
                        "src_node_id": src["node_id"],
                        "dst_node_id": dst["node_id"],
                        "src_layer_key": src["layer_key"],
                        "dst_layer_key": dst["layer_key"],
                        "src_type": rule["src_type"],
                        "dst_type": rule["dst_type"],
                        "weight": rule["weight"],
                        "delay": rule["delay"],
                        "distance_km": round(dist_km, 4),
                    }
                )

    return pd.DataFrame(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-pass interdependent infrastructure graph for v3.")
    parser.add_argument(
        "--config",
        default="v3/configs/infrastructure_layers.yaml",
        help="Path to infrastructure layer config.",
    )
    parser.add_argument(
        "--inventory",
        default="v3/data/processed/asset_inventory.parquet",
        help="Path to processed asset inventory parquet.",
    )
    parser.add_argument(
        "--nodes-out",
        default="v3/data/processed/dependency_graph_nodes.csv",
        help="Output node CSV path.",
    )
    parser.add_argument(
        "--edges-out",
        default="v3/data/processed/dependency_graph_edges.csv",
        help="Output edge CSV path.",
    )
    parser.add_argument(
        "--summary-out",
        default="v3/data/processed/dependency_graph_summary.md",
        help="Output summary markdown path.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    nodes = pd.read_parquet(args.inventory)
    nodes = nodes[nodes["layer_key"].isin(POINT_LAYERS)].copy()
    nodes = nodes.dropna(subset=["latitude", "longitude"])

    edges = build_edges(nodes, config.get("dependency_rules", []))

    nodes_out = Path(args.nodes_out)
    edges_out = Path(args.edges_out)
    summary_out = Path(args.summary_out)
    for out in [nodes_out, edges_out, summary_out]:
        out.parent.mkdir(parents=True, exist_ok=True)

    nodes.to_csv(nodes_out, index=False)
    edges.to_csv(edges_out, index=False)

    lines = [
        "# V3 Dependency Graph Summary",
        "",
        f"- Nodes: `{len(nodes)}`",
        f"- Edges: `{len(edges)}`",
        "",
        "## Node Counts by Layer",
        "",
    ]
    for layer_key, count in nodes["layer_key"].value_counts().items():
        lines.append(f"- `{layer_key}`: `{count}`")

    lines.extend(["", "## Edge Counts by Dependency Type", ""])
    if edges.empty:
        lines.append("- No edges were constructed.")
    else:
        edge_counts = edges.groupby(["src_type", "dst_type"]).size().sort_values(ascending=False)
        for (src_type, dst_type), count in edge_counts.items():
            lines.append(f"- `{src_type} -> {dst_type}`: `{count}`")

    summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote nodes to {nodes_out}")
    print(f"Wrote edges to {edges_out}")
    print(f"Wrote summary to {summary_out}")
    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}")


if __name__ == "__main__":
    main()
