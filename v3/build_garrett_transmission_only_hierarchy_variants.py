from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import v3.run_redundancy_comparison as redundancy_utils
    import v3.simulator_v3 as sim3
    from v3.build_montgomery_telecom_hierarchy_variants import (
        TELECOM_EDGE_DEFAULTS,
        SERVICE_WEIGHTS,
        cluster_tower_sites,
        nearest_assignment,
        normalize_label,
        point_distance_km,
    )
    from v3.run_final_power_timing_experiment import (
        PLANT_TO_TRANSMISSION_DELAY,
        PLANT_TO_TRANSMISSION_WEIGHT,
        SUBSTATION_TYPE_PARAMS,
        TRANSMISSION_DELAY,
        TRANSMISSION_WEIGHT,
    )
except ModuleNotFoundError:
    import run_redundancy_comparison as redundancy_utils
    import simulator_v3 as sim3
    from build_montgomery_telecom_hierarchy_variants import (
        TELECOM_EDGE_DEFAULTS,
        SERVICE_WEIGHTS,
        cluster_tower_sites,
        nearest_assignment,
        normalize_label,
        point_distance_km,
    )
    from run_final_power_timing_experiment import (
        PLANT_TO_TRANSMISSION_DELAY,
        PLANT_TO_TRANSMISSION_WEIGHT,
        SUBSTATION_TYPE_PARAMS,
        TRANSMISSION_DELAY,
        TRANSMISSION_WEIGHT,
    )


POWER_DIRECT_CONSUMERS = {"school", "hospital", "ems_fire"}
TELECOM_CONSUMER_TYPES = {"school", "hospital"}
REPLACED_FLAT_TELECOM_TYPE = "telecom"


def load_matched_transmission_nodes(osm_substations: pd.DataFrame) -> pd.DataFrame:
    matched = osm_substations[osm_substations["matched_hifld_substation"].fillna("") != ""].copy()
    matched = matched.sort_values(["matched_hifld_substation", "match_distance_m"]).drop_duplicates(
        "matched_hifld_substation"
    )
    matched["layer_key"] = "transmission_substations"
    matched["node_type"] = "transmission_substation"
    matched["source_id"] = matched["matched_hifld_substation"].astype(str)
    matched["name"] = matched["matched_hifld_substation"].astype(str)
    matched["node_id"] = matched["source_id"].map(
        lambda x: f"transmission_substations::{normalize_label(x)}"
    )
    return matched[
        ["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id"]
    ].copy()


def load_exchange_nodes(exchange_df: pd.DataFrame) -> pd.DataFrame:
    exchange_nodes = exchange_df.copy()
    exchange_nodes["layer_key"] = "telecom_exchanges"
    exchange_nodes["node_type"] = "telecom_exchange"
    exchange_nodes["source_id"] = exchange_nodes["osm_id"].astype(str)
    exchange_nodes["node_id"] = exchange_nodes["source_id"].map(
        lambda x: f"telecom_exchanges::{normalize_label(x)}"
    )
    exchange_nodes["name"] = exchange_nodes["name"].fillna("").astype(str)
    exchange_nodes["name"] = exchange_nodes.apply(
        lambda row: row["name"] if str(row["name"]).strip() else str(row["osm_id"]),
        axis=1,
    )
    return exchange_nodes[
        ["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id", "telecom", "office", "operator"]
    ].copy()


def build_transmission_network_edges(
    substation_edges: pd.DataFrame,
    transmission_name_to_node: dict[str, str],
    transmission_nodes: pd.DataFrame,
) -> pd.DataFrame:
    meta = transmission_nodes.set_index("name")[["latitude", "longitude"]].to_dict("index")
    rows: list[dict] = []
    for row in substation_edges.itertuples(index=False):
        src_name = str(row.source_substation)
        dst_name = str(row.target_substation)
        if src_name not in transmission_name_to_node or dst_name not in transmission_name_to_node:
            continue
        src_id = transmission_name_to_node[src_name]
        dst_id = transmission_name_to_node[dst_name]
        src_meta = meta[src_name]
        dst_meta = meta[dst_name]
        dist_km = point_distance_km(
            float(src_meta["longitude"]),
            float(src_meta["latitude"]),
            float(dst_meta["longitude"]),
            float(dst_meta["latitude"]),
        )
        base = {
            "src_layer_key": "transmission_substations",
            "dst_layer_key": "transmission_substations",
            "src_type": "transmission_substation",
            "dst_type": "transmission_substation",
            "weight": float(TRANSMISSION_WEIGHT),
            "delay": int(TRANSMISSION_DELAY),
            "distance_km": float(dist_km),
        }
        rows.append({"src_node_id": src_id, "dst_node_id": dst_id, **base})
        rows.append({"src_node_id": dst_id, "dst_node_id": src_id, **base})
    return pd.DataFrame(rows).drop_duplicates(["src_node_id", "dst_node_id"]).reset_index(drop=True)


def build_plant_to_transmission_edges(
    plant_connections: pd.DataFrame,
    transmission_name_to_node: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    matched = plant_connections[plant_connections["substation"].fillna("") != ""].copy()
    for row in matched.itertuples(index=False):
        sub_name = str(row.substation)
        if sub_name not in transmission_name_to_node:
            continue
        rows.append(
            {
                "src_node_id": str(row.power_node_id),
                "dst_node_id": transmission_name_to_node[sub_name],
                "src_layer_key": "power",
                "dst_layer_key": "transmission_substations",
                "src_type": "power",
                "dst_type": "transmission_substation",
                "weight": float(PLANT_TO_TRANSMISSION_WEIGHT),
                "delay": int(PLANT_TO_TRANSMISSION_DELAY),
                "distance_km": float(row.distance_km) if str(row.distance_km).strip() else 0.0,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["src_node_id", "dst_node_id"]).reset_index(drop=True)


def build_power_feed_edges(
    source_df: pd.DataFrame,
    transmission_nodes: pd.DataFrame,
    source_prefix: str,
    source_type: str,
    source_layer_key: str,
    fan_in: int,
    edge_weight: float,
    edge_delay: int,
) -> pd.DataFrame:
    assignments = nearest_assignment(
        source_df,
        transmission_nodes[["source_id", "name", "latitude", "longitude"]].rename(
            columns={"source_id": "transmission_source_id", "name": "transmission_name"}
        ),
        source_prefix + "_source_id",
        source_prefix + "_name",
        "transmission_source_id",
        "transmission_name",
    )
    if fan_in == 1:
        ranked = assignments.assign(upstream_rank=1)
    else:
        rows: list[dict] = []
        targets = transmission_nodes[["source_id", "name", "latitude", "longitude", "node_id"]].rename(
            columns={"source_id": "transmission_source_id", "name": "transmission_name"}
        )
        target_records = targets.to_dict("records")
        for source in source_df.to_dict("records"):
            candidates: list[tuple[float, str, dict]] = []
            for target in target_records:
                dist_km = point_distance_km(
                    float(source["longitude"]),
                    float(source["latitude"]),
                    float(target["longitude"]),
                    float(target["latitude"]),
                )
                candidates.append((dist_km, str(target["transmission_source_id"]), target))
            candidates.sort(key=lambda item: (item[0], item[1]))
            for rank, (dist_km, _name, target) in enumerate(candidates[:fan_in], start=1):
                rows.append(
                    {
                        source_prefix + "_source_id": str(source[source_prefix + "_source_id"]),
                        source_prefix + "_name": str(source[source_prefix + "_name"]),
                        "distance_km": float(dist_km),
                        "transmission_source_id": str(target["transmission_source_id"]),
                        "transmission_name": str(target["transmission_name"]),
                        "upstream_rank": int(rank),
                    }
                )
        ranked = pd.DataFrame(rows)

    transmission_source_to_node = transmission_nodes.set_index("source_id")["node_id"].astype(str).to_dict()
    source_source_to_node = source_df.set_index(source_prefix + "_source_id")["node_id"].astype(str).to_dict()

    edge_rows: list[dict] = []
    for row in ranked.itertuples(index=False):
        edge_rows.append(
            {
                "src_node_id": source_source_to_node[str(getattr(row, source_prefix + "_source_id"))],
                "dst_node_id": transmission_source_to_node[str(row.transmission_source_id)],
                "src_layer_key": source_layer_key,
                "dst_layer_key": "transmission_substations",
                "src_type": source_type,
                "dst_type": "transmission_substation",
                "weight": float(edge_weight),
                "delay": int(edge_delay),
                "distance_km": float(row.distance_km),
                "upstream_rank": int(row.upstream_rank),
            }
        )
    return pd.DataFrame(edge_rows)


def summarize_dependencies(edges: pd.DataFrame) -> pd.Series:
    return edges.groupby(["src_type", "dst_type"]).size().sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Garrett transmission-only power hierarchy variants with telecom tower/exchange structure."
    )
    parser.add_argument("--baseline-nodes", default="v3/data/processed/garrett_dependency_graph_nodes.csv")
    parser.add_argument("--baseline-edges", default="v3/data/processed/garrett_dependency_graph_edges.csv")
    parser.add_argument(
        "--plant-connections",
        default="v3/data/processed/transmission_inspection_garrett/power_plant_substation_connections.csv",
    )
    parser.add_argument(
        "--substation-edges",
        default="v3/data/processed/transmission_inspection_garrett/montgomery_substation_edges.csv",
    )
    parser.add_argument(
        "--osm-substations",
        default="v3/data/processed/osm_garrett_substations/osm_montgomery_substations.csv",
    )
    parser.add_argument(
        "--telecom-exchanges",
        default="v3/data/processed/osm_garrett_telecom/osm_montgomery_telecom_exchanges.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="v3/data/processed/garrett_transmission_only_hierarchy_graph_variants",
    )
    parser.add_argument("--redundancy-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_nodes = pd.read_csv(args.baseline_nodes)
    baseline_edges = pd.read_csv(args.baseline_edges)
    plant_connections = pd.read_csv(args.plant_connections)
    substation_edges = pd.read_csv(args.substation_edges)
    osm_substations = pd.read_csv(args.osm_substations)
    exchange_df = pd.read_csv(args.telecom_exchanges)

    transmission_nodes = load_matched_transmission_nodes(osm_substations)
    exchange_nodes = load_exchange_nodes(exchange_df)

    flat_telecom = baseline_nodes[baseline_nodes["node_type"] == REPLACED_FLAT_TELECOM_TYPE][
        ["node_id", "name", "latitude", "longitude"]
    ].copy()
    tower_clusters, tower_members = cluster_tower_sites(flat_telecom, threshold_km=0.1)
    tower_members.to_csv(output_dir / "telecom_tower_cluster_members.csv", index=False)
    tower_clusters.to_csv(output_dir / "telecom_tower_clusters.csv", index=False)
    transmission_nodes.to_csv(output_dir / "matched_transmission_nodes.csv", index=False)
    exchange_nodes.to_csv(output_dir / "telecom_exchange_nodes.csv", index=False)

    transmission_name_to_node = transmission_nodes.set_index("name")["node_id"].astype(str).to_dict()
    tower_source_to_node = tower_clusters.set_index("source_id")["node_id"].astype(str).to_dict()
    exchange_source_to_node = exchange_nodes.set_index("source_id")["node_id"].astype(str).to_dict()

    plant_to_transmission_edges = build_plant_to_transmission_edges(plant_connections, transmission_name_to_node)
    transmission_network_edges = build_transmission_network_edges(
        substation_edges,
        transmission_name_to_node,
        transmission_nodes,
    )

    telecom_removed_nodes = baseline_nodes[baseline_nodes["node_type"] != REPLACED_FLAT_TELECOM_TYPE].copy()
    power_removed_edges = baseline_edges[baseline_edges["dst_type"] != "power"].copy()
    telecom_removed_edges = power_removed_edges[
        ~power_removed_edges["src_type"].eq(REPLACED_FLAT_TELECOM_TYPE)
        & ~power_removed_edges["dst_type"].eq(REPLACED_FLAT_TELECOM_TYPE)
    ].copy()

    base_nodes = pd.concat(
        [
            telecom_removed_nodes,
            transmission_nodes[["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id"]],
            tower_clusters.assign(
                layer_key="telecom_towers",
                node_type="telecom_tower",
            )[["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id"]],
            exchange_nodes[["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id"]],
        ],
        ignore_index=True,
    )

    tower_to_exchange = nearest_assignment(
        tower_clusters[["source_id", "name", "latitude", "longitude"]].rename(
            columns={"source_id": "tower_source_id", "name": "tower_name"}
        ),
        exchange_nodes[["source_id", "name", "latitude", "longitude"]].rename(
            columns={"source_id": "exchange_source_id", "name": "exchange_name"}
        ),
        "tower_source_id",
        "tower_name",
        "exchange_source_id",
        "exchange_name",
    )
    tower_to_exchange.to_csv(output_dir / "tower_exchange_assignments.csv", index=False)

    consumer_nodes = baseline_nodes[baseline_nodes["node_type"].isin(sorted(TELECOM_CONSUMER_TYPES))][
        ["node_id", "name", "node_type", "latitude", "longitude"]
    ].rename(columns={"node_id": "consumer_node_id", "name": "consumer_name", "node_type": "consumer_type"})
    consumer_to_tower = nearest_assignment(
        consumer_nodes,
        tower_clusters[["source_id", "name", "latitude", "longitude"]].rename(
            columns={"source_id": "tower_source_id", "name": "tower_name"}
        ),
        "consumer_node_id",
        "consumer_name",
        "tower_source_id",
        "tower_name",
    )
    consumer_to_tower = consumer_to_tower.merge(
        consumer_nodes[["consumer_node_id", "consumer_type"]],
        on="consumer_node_id",
        how="left",
    )
    consumer_to_tower.to_csv(output_dir / "consumer_tower_assignments.csv", index=False)

    new_edge_rows: list[dict] = []
    for row in consumer_to_tower.itertuples(index=False):
        cfg = TELECOM_EDGE_DEFAULTS[str(row.consumer_type)]
        new_edge_rows.append(
            {
                "src_node_id": str(row.consumer_node_id),
                "dst_node_id": tower_source_to_node[str(row.tower_source_id)],
                "src_layer_key": str(row.consumer_type),
                "dst_layer_key": "telecom_towers",
                "src_type": str(row.consumer_type),
                "dst_type": "telecom_tower",
                "weight": float(cfg["weight"]),
                "delay": int(cfg["delay"]),
                "distance_km": float(row.distance_km),
            }
        )
    for row in tower_to_exchange.itertuples(index=False):
        cfg = TELECOM_EDGE_DEFAULTS["tower_exchange"]
        new_edge_rows.append(
            {
                "src_node_id": tower_source_to_node[str(row.tower_source_id)],
                "dst_node_id": exchange_source_to_node[str(row.exchange_source_id)],
                "src_layer_key": "telecom_towers",
                "dst_layer_key": "telecom_exchanges",
                "src_type": "telecom_tower",
                "dst_type": "telecom_exchange",
                "weight": float(cfg["weight"]),
                "delay": int(cfg["delay"]),
                "distance_km": float(row.distance_km),
            }
        )

    hierarchy_base_edges = pd.concat(
        [
            telecom_removed_edges,
            transmission_network_edges,
            plant_to_transmission_edges,
            pd.DataFrame(new_edge_rows),
        ],
        ignore_index=True,
    )

    power_sources = baseline_nodes[baseline_nodes["node_type"].isin(sorted(POWER_DIRECT_CONSUMERS))][
        ["node_id", "name", "node_type", "latitude", "longitude"]
    ].copy()

    baseline_power_edges_frames = []
    redundant_power_edges_frames = []
    for source_type, weight, delay in [
        ("school", 0.7, 2),
        ("hospital", 0.9, 2),
        ("ems_fire", 0.5, 0),
    ]:
        source_df = power_sources[power_sources["node_type"] == source_type].copy()
        if source_df.empty:
            continue
        source_df = source_df.rename(columns={"name": f"{source_type}_name"})
        source_df[f"{source_type}_source_id"] = source_df["node_id"].astype(str)
        baseline_power_edges_frames.append(
            build_power_feed_edges(
                source_df,
                transmission_nodes,
                source_type,
                source_type,
                source_type,
                fan_in=1,
                edge_weight=weight,
                edge_delay=delay,
            )
        )
        redundant_power_edges_frames.append(
            build_power_feed_edges(
                source_df,
                transmission_nodes,
                source_type,
                source_type,
                source_type,
                fan_in=2,
                edge_weight=weight,
                edge_delay=delay,
            )
        )

    tower_power_sources = tower_clusters[["source_id", "name", "latitude", "longitude", "node_id"]].rename(
        columns={"source_id": "tower_source_id", "name": "tower_name"}
    )
    exchange_power_sources = exchange_nodes[["source_id", "name", "latitude", "longitude", "node_id"]].rename(
        columns={"source_id": "exchange_source_id", "name": "exchange_name"}
    )

    baseline_power_edges_frames.append(
        build_power_feed_edges(
            tower_power_sources,
            transmission_nodes,
            "tower",
            "telecom_tower",
            "telecom_towers",
            fan_in=1,
            edge_weight=TELECOM_EDGE_DEFAULTS["tower_power"]["weight"],
            edge_delay=TELECOM_EDGE_DEFAULTS["tower_power"]["delay"],
        )
    )
    redundant_power_edges_frames.append(
        build_power_feed_edges(
            tower_power_sources,
            transmission_nodes,
            "tower",
            "telecom_tower",
            "telecom_towers",
            fan_in=2,
            edge_weight=TELECOM_EDGE_DEFAULTS["tower_power"]["weight"],
            edge_delay=TELECOM_EDGE_DEFAULTS["tower_power"]["delay"],
        )
    )
    baseline_power_edges_frames.append(
        build_power_feed_edges(
            exchange_power_sources,
            transmission_nodes,
            "exchange",
            "telecom_exchange",
            "telecom_exchanges",
            fan_in=1,
            edge_weight=TELECOM_EDGE_DEFAULTS["exchange_power"]["weight"],
            edge_delay=TELECOM_EDGE_DEFAULTS["exchange_power"]["delay"],
        )
    )
    redundant_power_edges_frames.append(
        build_power_feed_edges(
            exchange_power_sources,
            transmission_nodes,
            "exchange",
            "telecom_exchange",
            "telecom_exchanges",
            fan_in=2,
            edge_weight=TELECOM_EDGE_DEFAULTS["exchange_power"]["weight"],
            edge_delay=TELECOM_EDGE_DEFAULTS["exchange_power"]["delay"],
        )
    )

    baseline_edges = pd.concat(
        [hierarchy_base_edges] + baseline_power_edges_frames,
        ignore_index=True,
    )
    redundant_edges = pd.concat(
        [hierarchy_base_edges] + redundant_power_edges_frames,
        ignore_index=True,
    )

    nodes_path = output_dir / "dependency_graph_nodes.csv"
    base_edges_path = output_dir / "hierarchy_base_edges.csv"
    base_sim_path = output_dir / "hierarchy_base_sim_edges.csv"
    baseline_edges_path = output_dir / "baseline_edges.csv"
    redundant_edges_path = output_dir / "redundant_edges.csv"
    baseline_sim_path = output_dir / "baseline_additive_sim_edges.csv"
    redundant_additive_sim_path = output_dir / "redundant_additive_sim_edges.csv"
    redundant_buffer_sim_path = output_dir / "redundant_buffer_sim_edges.csv"

    base_nodes.to_csv(nodes_path, index=False)
    hierarchy_base_edges.to_csv(base_edges_path, index=False)
    baseline_edges.to_csv(baseline_edges_path, index=False)
    redundant_edges.to_csv(redundant_edges_path, index=False)

    base_sim_edges = sim3.ensure_simulation_edges(str(nodes_path), str(base_edges_path), str(base_sim_path))
    baseline_sim_edges = sim3.ensure_simulation_edges(str(nodes_path), str(baseline_edges_path), str(baseline_sim_path))
    redundant_additive_sim_edges = sim3.ensure_simulation_edges(
        str(nodes_path), str(redundant_edges_path), str(redundant_additive_sim_path)
    )
    redundant_buffer_sim_edges = sim3.ensure_simulation_edges(
        str(nodes_path), str(redundant_edges_path), str(redundant_buffer_sim_path)
    )

    conditions = [
        condition
        for condition in redundancy_utils.build_conditions(output_dir, args.redundancy_threshold)
        if condition["condition"] in {"baseline_additive", "redundant_additive", "redundant_buffer"}
    ]

    summary = {
        "hierarchy_type": "garrett_transmission_only_power_plus_telecom_two_tier",
        "node_counts": base_nodes["node_type"].value_counts().to_dict(),
        "edge_type_counts": {
            f"{src}->{dst}": int(count)
            for (src, dst), count in hierarchy_base_edges.groupby(["src_type", "dst_type"]).size().items()
        },
        "matched_transmission_nodes": int(len(transmission_nodes)),
        "flat_telecom_nodes_replaced": int(len(flat_telecom)),
        "tower_clusters": int(len(tower_clusters)),
        "telecom_exchanges": int(len(exchange_nodes)),
        "matched_power_plants": int(plant_to_transmission_edges["src_node_id"].nunique()) if not plant_to_transmission_edges.empty else 0,
        "baseline_sim_edges": int(len(baseline_sim_edges)),
        "redundant_additive_sim_edges": int(len(redundant_additive_sim_edges)),
        "redundant_buffer_sim_edges": int(len(redundant_buffer_sim_edges)),
        "service_weights": SERVICE_WEIGHTS,
    }
    (output_dir / "garrett_transmission_only_hierarchy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Garrett Transmission-Only Hierarchy Variants",
        "",
        "This Garrett build uses a simpler tier structure than Montgomery:",
        "",
        "- Power: `power plants -> transmission substations -> consumers`",
        "- Telecom: `schools/hospitals -> telecom_tower -> telecom_exchange`, with tower/exchange power feeds drawn directly from transmission substations",
        "- No distribution-substation tier is used in Garrett",
        "",
        "## Node Counts",
        "",
    ]
    for node_type, count in base_nodes["node_type"].value_counts().items():
        lines.append(f"- `{node_type}`: `{int(count)}`")
    lines.extend(["", "## Base Edge Counts", ""])
    dep_summary = summarize_dependencies(hierarchy_base_edges)
    for (src_type, dst_type), count in dep_summary.items():
        lines.append(f"- `{src_type} -> {dst_type}`: `{int(count)}`")
    lines.extend(
        [
            "",
            f"- Matched transmission substations used: `{len(transmission_nodes)}`",
            f"- Power plants matched into transmission network: `{summary['matched_power_plants']}`",
            f"- Flat telecom nodes replaced: `{len(flat_telecom)}`",
            f"- Telecom tower clusters: `{len(tower_clusters)}`",
            f"- Telecom exchanges: `{len(exchange_nodes)}`",
        ]
    )
    (output_dir / "garrett_transmission_only_hierarchy_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote Garrett transmission-only hierarchy variants to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
