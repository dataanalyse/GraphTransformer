from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUPPORT_EDGE_TYPES = {"emergency_management -> ems_fire"}


def semantic_type_for(dependency_type: str) -> str:
    if dependency_type in SUPPORT_EDGE_TYPES:
        return "support"
    return "dependency"


def failure_flow_direction_for(semantic_type: str) -> str:
    mapping = {
        "dependency": "supporting_asset_failure_to_dependent_asset",
        "support": "support_provider_failure_to_supported_asset",
        "interdependency": "interdependent_failure_flow",
    }
    return mapping[semantic_type]


def join_unique(values: pd.Series) -> str:
    cleaned = [str(v) for v in values.dropna().astype(str).tolist()]
    unique = []
    seen = set()
    for value in cleaned:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return ";".join(unique)


def standardize_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = edges.copy()
    work["dependency_type"] = work["src_type"] + " -> " + work["dst_type"]
    work["semantic_type"] = work["dependency_type"].map(semantic_type_for)
    work["failure_flow_direction"] = work["semantic_type"].map(failure_flow_direction_for)

    work["original_source"] = work["src_node_id"]
    work["original_target"] = work["dst_node_id"]

    is_dependency = work["semantic_type"] == "dependency"
    work["simulation_source"] = work["original_source"]
    work["simulation_target"] = work["original_target"]
    work.loc[is_dependency, "simulation_source"] = work.loc[is_dependency, "original_target"]
    work.loc[is_dependency, "simulation_target"] = work.loc[is_dependency, "original_source"]

    zero_delay_count = int((work["delay"] <= 0).sum())
    work["delay"] = work["delay"].clip(lower=1).astype(int)

    duplicate_pair_count = int(work.duplicated(subset=["simulation_source", "simulation_target"]).sum())

    grouped = (
        work.groupby(
            [
                "simulation_source",
                "simulation_target",
                "semantic_type",
                "failure_flow_direction",
                "dependency_type",
            ],
            as_index=False,
        )
        .agg(
            original_source=("original_source", join_unique),
            original_target=("original_target", join_unique),
            weight=("weight", "max"),
            delay=("delay", "min"),
        )
    )

    grouped = grouped[
        [
            "original_source",
            "original_target",
            "simulation_source",
            "simulation_target",
            "semantic_type",
            "failure_flow_direction",
            "weight",
            "delay",
            "dependency_type",
        ]
    ].sort_values(["semantic_type", "dependency_type", "simulation_source", "simulation_target"]).reset_index(drop=True)

    metadata = {
        "input_edge_count": int(len(work)),
        "output_edge_count": int(len(grouped)),
        "zero_delay_normalized_count": zero_delay_count,
        "duplicate_pair_count_before_aggregation": duplicate_pair_count,
    }
    return grouped, metadata


def validate_simulation_edges(sim_edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    node_ids = set(nodes["node_id"])
    invalid_sources = sorted(set(sim_edges["simulation_source"]) - node_ids)
    invalid_targets = sorted(set(sim_edges["simulation_target"]) - node_ids)
    duplicate_pairs = sim_edges.duplicated(subset=["simulation_source", "simulation_target"]).sum()
    weights_valid = sim_edges["weight"].between(0, 1, inclusive="both").all()
    delays_positive = ((sim_edges["delay"] > 0) & (sim_edges["delay"].astype(int) == sim_edges["delay"])).all()

    return {
        "invalid_source_count": int(len(invalid_sources)),
        "invalid_target_count": int(len(invalid_targets)),
        "duplicate_pair_count": int(duplicate_pairs),
        "weights_valid": bool(weights_valid),
        "delays_positive": bool(delays_positive),
        "invalid_sources": invalid_sources,
        "invalid_targets": invalid_targets,
    }


def build_summary(sim_edges: pd.DataFrame, metadata: dict, validation: dict) -> str:
    semantic_counts = sim_edges["semantic_type"].value_counts().sort_index()
    dependency_counts = sim_edges["dependency_type"].value_counts().sort_index()

    lines = [
        "# Simulation Edge Semantics Summary",
        "",
        "## Standardization Rules Applied",
        "",
        "- Dependency edges were reversed so failure flows from supporting assets to dependent assets.",
        "- Support edges were kept in their original direction.",
        "- Zero-delay edges were normalized to delay `1` so the simulator receives strictly positive time lags.",
        "- Duplicate simulation source-target pairs were aggregated into single simulation edges while preserving original provenance in `original_source` and `original_target`.",
        "",
        "## Edge Counts",
        "",
        f"- Input dependency edges: `{metadata['input_edge_count']}`",
        f"- Output simulation edges: `{metadata['output_edge_count']}`",
        f"- Zero-delay edges normalized: `{metadata['zero_delay_normalized_count']}`",
        f"- Duplicate simulation pairs before aggregation: `{metadata['duplicate_pair_count_before_aggregation']}`",
        "",
        "### Counts by Semantic Type",
        "",
    ]

    for semantic_type, count in semantic_counts.items():
        lines.append(f"- `{semantic_type}`: `{count}`")

    lines.extend(["", "### Counts by Dependency Type", ""])
    for dependency_type, count in dependency_counts.items():
        lines.append(f"- `{dependency_type}`: `{count}`")

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Invalid simulation source IDs: `{validation['invalid_source_count']}`",
            f"- Invalid simulation target IDs: `{validation['invalid_target_count']}`",
            f"- Duplicate simulation source-target pairs after aggregation: `{validation['duplicate_pair_count']}`",
            f"- All weights in [0,1]: `{'yes' if validation['weights_valid'] else 'no'}`",
            f"- All delays positive integers: `{'yes' if validation['delays_positive'] else 'no'}`",
        ]
    )

    if validation["invalid_sources"]:
        lines.extend(["", "### Invalid Simulation Sources", ""])
        for node_id in validation["invalid_sources"]:
            lines.append(f"- `{node_id}`")

    if validation["invalid_targets"]:
        lines.extend(["", "### Invalid Simulation Targets", ""])
        for node_id in validation["invalid_targets"]:
            lines.append(f"- `{node_id}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `simulation_source -> simulation_target` is the direction in which disruption/failure effects should propagate in the simulator.",
            "- Most edges are now supporting-asset to dependent-asset links.",
            "- The remaining support edge family is emergency management to EMS/fire coordination.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize dependency graph edges into simulator failure-flow edges.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--out", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--summary-out", default="v3/data/processed/simulation_edge_semantics_summary.md")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)

    sim_edges, metadata = standardize_edges(edges)
    validation = validate_simulation_edges(sim_edges, nodes)
    summary = build_summary(sim_edges, metadata, validation)

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sim_edges.to_csv(out_path, index=False)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote simulation edges to {out_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
