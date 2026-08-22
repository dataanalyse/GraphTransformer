from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from v3.compare_substation_assignment_scopes import distance_to_boundary_km, point_distance_km
    from v3.inspect_transmission_lines import load_boundary_rings
except ModuleNotFoundError:
    from compare_substation_assignment_scopes import distance_to_boundary_km, point_distance_km
    from inspect_transmission_lines import load_boundary_rings


INFRA_TYPES = ("school", "hospital", "telecom", "ems_fire")
TRANSMISSION_ONLY_EXTRA_INFRA_TYPES = ("telecom_tower", "telecom_exchange")
COMPARISON_SCOPE = "B_inside_plus_boundary10km"


def distribution_stats(series: pd.Series) -> dict[str, float]:
    s = series.astype(float)
    return {
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p90": float(s.quantile(0.9)),
        "max": float(s.max()),
    }


def telecom_sector_count(series: pd.Series) -> int:
    return int(series.isin(["telecom", "telecom_tower", "telecom_exchange"]).sum())


def assign_nearest(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_prefix: str,
    target_prefix: str,
    connection_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    targets = target_df.to_dict("records")
    for source in source_df.to_dict("records"):
        best = None
        for target in targets:
            distance = point_distance_km(
                float(source["longitude"]),
                float(source["latitude"]),
                float(target["longitude"]),
                float(target["latitude"]),
            )
            candidate = (
                distance,
                str(target[target_prefix + "_id"]),
                str(target[target_prefix + "_name"]),
                target,
            )
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        target = best[3]
        row = {
            source_prefix + "_id": str(source[source_prefix + "_id"]),
            source_prefix + "_name": str(source[source_prefix + "_name"]),
            source_prefix + "_type": str(source[source_prefix + "_type"]),
            source_prefix + "_latitude": float(source["latitude"]),
            source_prefix + "_longitude": float(source["longitude"]),
            target_prefix + "_id": str(target[target_prefix + "_id"]),
            target_prefix + "_name": str(target[target_prefix + "_name"]),
            target_prefix + "_type": str(target[target_prefix + "_type"]),
            target_prefix + "_latitude": float(target["latitude"]),
            target_prefix + "_longitude": float(target["longitude"]),
            target_prefix + "_inside_montgomery": bool(target["inside_montgomery"]),
            target_prefix + "_distance_to_boundary_km": float(target["boundary_distance_km"]),
            "distance_km": float(best[0]),
            "connection_label": connection_label,
        }
        for optional_col in ("voltage", "matched_hifld_substation"):
            target_col = target_prefix + "_" + optional_col
            if optional_col in target:
                row[target_col] = target[optional_col]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only validation of a candidate plant->transmission->distribution->infrastructure hierarchy for Montgomery.")
    parser.add_argument("--boundary-geojson", default="v3/data/raw/boundaries/montgomery_county_boundary.geojson")
    parser.add_argument("--power-plant-connections", default="v3/data/processed/transmission_inspection_montgomery/power_plant_substation_connections.csv")
    parser.add_argument("--derived-substations", default="v3/data/processed/transmission_inspection_montgomery/derived_substation_locations.csv")
    parser.add_argument("--osm-substations", default="v3/data/processed/osm_montgomery_substations/osm_montgomery_substations.csv")
    parser.add_argument("--dependency-nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--dependency-edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--scope-summary", default="v3/data/processed/transmission_inspection_montgomery/substation_assignment_scope_comparison/scope_summary.csv")
    parser.add_argument("--scope-fanout", default="v3/data/processed/transmission_inspection_montgomery/substation_assignment_scope_comparison/substation_fanout_by_scope.csv")
    parser.add_argument("--output-dir", default="v3/data/processed/final_power_hierarchy_validation")
    parser.add_argument("--boundary-km-threshold", type=float, default=10.0)
    parser.add_argument("--transmission-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rings = load_boundary_rings(Path(args.boundary_geojson))

    power_connections = pd.read_csv(args.power_plant_connections).copy()
    derived_substations = pd.read_csv(args.derived_substations).copy()
    osm_substations = pd.read_csv(args.osm_substations).copy()
    dependency_nodes = pd.read_csv(args.dependency_nodes).copy()
    dependency_edges = pd.read_csv(args.dependency_edges).copy()
    scope_summary = pd.read_csv(args.scope_summary).copy()
    scope_fanout = pd.read_csv(args.scope_fanout).copy()

    derived_substations["boundary_distance_km"] = derived_substations.apply(
        lambda row: 0.0
        if bool(row["inside_montgomery"])
        else distance_to_boundary_km(float(row["longitude"]), float(row["latitude"]), rings),
        axis=1,
    )
    eligible_transmission = derived_substations[
        (derived_substations["inside_montgomery"] == True)
        | (derived_substations["boundary_distance_km"] <= float(args.boundary_km_threshold))
    ].copy()
    eligible_transmission["transmission_id"] = eligible_transmission["substation"].astype(str)
    eligible_transmission["transmission_name"] = eligible_transmission["substation"].astype(str)
    eligible_transmission["transmission_type"] = "transmission_network"

    osm_substations["name"] = osm_substations["name"].fillna("")
    osm_substations["matched_hifld_substation"] = osm_substations["matched_hifld_substation"].fillna("")
    if args.transmission_only:
        candidate_distribution = osm_substations[
            osm_substations["matched_hifld_substation"].fillna("").astype(str).str.strip() != ""
        ].copy()
        candidate_distribution = candidate_distribution.sort_values(
            ["matched_hifld_substation", "match_distance_m"]
        ).drop_duplicates("matched_hifld_substation")
        candidate_distribution["distribution_id"] = candidate_distribution["matched_hifld_substation"].astype(str)
        candidate_distribution["distribution_name"] = candidate_distribution["matched_hifld_substation"].astype(str)
        candidate_distribution["distribution_type"] = "transmission_terminal"
        candidate_distribution["distribution_voltage"] = candidate_distribution["voltage"].fillna("").astype(str)
        candidate_distribution["voltage"] = candidate_distribution["distribution_voltage"]
        candidate_distribution["boundary_distance_km"] = candidate_distribution["distance_to_boundary_km"].astype(float)
        infra_types = tuple(sorted(set(INFRA_TYPES) | set(TRANSMISSION_ONLY_EXTRA_INFRA_TYPES)))
    else:
        candidate_distribution = osm_substations[
            (osm_substations["candidate_local_distribution"] == True)
            & (
                (osm_substations["inside_montgomery"] == True)
                | (osm_substations["distance_to_boundary_km"] <= float(args.boundary_km_threshold))
            )
        ].copy()
        candidate_distribution["distribution_id"] = candidate_distribution["osm_id"].astype(str)
        candidate_distribution["distribution_name"] = candidate_distribution["name"].where(
            candidate_distribution["name"].astype(str).str.strip() != "",
            candidate_distribution["osm_id"].astype(str),
        )
        candidate_distribution["distribution_type"] = candidate_distribution["substation_type"].astype(str)
        candidate_distribution["distribution_voltage"] = candidate_distribution["voltage"].fillna("").astype(str)
        candidate_distribution["voltage"] = candidate_distribution["distribution_voltage"]
        candidate_distribution["boundary_distance_km"] = candidate_distribution["distance_to_boundary_km"].astype(float)
        infra_types = INFRA_TYPES

    infrastructure = dependency_nodes[dependency_nodes["node_type"].isin(infra_types)].copy()
    infrastructure["infra_id"] = infrastructure["node_id"].astype(str)
    infrastructure["infra_name"] = infrastructure["name"].astype(str)
    infrastructure["infra_type"] = infrastructure["node_type"].astype(str)

    infra_assignments = assign_nearest(
        source_df=infrastructure[["infra_id", "infra_name", "infra_type", "latitude", "longitude"]].copy(),
        target_df=candidate_distribution[
            [
                "distribution_id",
                "distribution_name",
                "distribution_type",
                "distribution_voltage",
                "latitude",
                "longitude",
                "inside_montgomery",
                "boundary_distance_km",
            ]
        ].copy(),
        source_prefix="infra",
        target_prefix="distribution",
        connection_label="spatially_inferred_local_service",
    )
    infra_assignments = infra_assignments.merge(
        candidate_distribution[
            ["distribution_id", "distribution_voltage", "inside_montgomery", "boundary_distance_km"]
        ].rename(
            columns={
                "inside_montgomery": "distribution_inside_montgomery",
                "boundary_distance_km": "distribution_distance_to_boundary_km",
            }
        ),
        on="distribution_id",
        how="left",
        suffixes=("", "_dup"),
    )
    infra_assignments = infra_assignments.loc[:, ~infra_assignments.columns.str.endswith("_dup")]
    infra_assignments.to_csv(output_dir / "infrastructure_distribution_assignments.csv", index=False)

    if args.transmission_only:
        dist_assignments = pd.DataFrame(
            columns=[
                "distribution_id",
                "distribution_name",
                "distribution_type",
                "transmission_id",
                "transmission_name",
                "transmission_type",
                "distance_km",
            ]
        )
    else:
        dist_assignments = assign_nearest(
            source_df=candidate_distribution[
                [
                    "distribution_id",
                    "distribution_name",
                    "distribution_type",
                    "distribution_voltage",
                    "latitude",
                    "longitude",
                ]
            ].copy(),
            target_df=eligible_transmission[
                [
                    "transmission_id",
                    "transmission_name",
                    "transmission_type",
                    "latitude",
                    "longitude",
                    "inside_montgomery",
                    "boundary_distance_km",
                ]
            ].copy(),
            source_prefix="distribution",
            target_prefix="transmission",
            connection_label="spatially_inferred_transmission_distribution",
        )
        dist_assignments = dist_assignments.merge(
            candidate_distribution[
                ["distribution_id", "distribution_voltage", "inside_montgomery", "boundary_distance_km"]
            ].rename(
                columns={
                    "inside_montgomery": "distribution_inside_montgomery",
                    "boundary_distance_km": "distribution_distance_to_boundary_km",
                }
            ),
            on="distribution_id",
            how="left",
            suffixes=("", "_dup"),
        )
        dist_assignments = dist_assignments.loc[:, ~dist_assignments.columns.str.endswith("_dup")]
        dist_assignments["matched_hifld_substation"] = dist_assignments["transmission_name"]
        dist_assignments.to_csv(output_dir / "distribution_transmission_assignments.csv", index=False)

    infra_sector_totals = infrastructure["infra_type"].value_counts().to_dict()
    infra_sector_totals["telecom"] = int(
        infrastructure["infra_type"].isin(["telecom", "telecom_tower", "telecom_exchange"]).sum()
    )
    total_infra = int(len(infrastructure))
    total_distribution = int(len(candidate_distribution))

    distribution_fanout = (
        infra_assignments.groupby(
            [
                "distribution_id",
                "distribution_name",
                "distribution_type",
                "distribution_voltage",
                "distribution_inside_montgomery",
                "distribution_distance_to_boundary_km",
            ],
            as_index=False,
        )
        .agg(
            assigned_infrastructure_count=("infra_id", "count"),
            schools=("infra_type", lambda s: int((s == "school").sum())),
            hospitals=("infra_type", lambda s: int((s == "hospital").sum())),
            telecom=("infra_type", telecom_sector_count),
            ems_fire=("infra_type", lambda s: int((s == "ems_fire").sum())),
            mean_assignment_distance_km=("distance_km", "mean"),
            max_assignment_distance_km=("distance_km", "max"),
        )
    )
    distribution_fanout["pct_of_all_infrastructure"] = (
        distribution_fanout["assigned_infrastructure_count"] / float(total_infra) * 100.0
    )
    distribution_fanout.sort_values(
        ["assigned_infrastructure_count", "distribution_name"],
        ascending=[False, True],
        inplace=True,
    )
    distribution_fanout.to_csv(output_dir / "distribution_fanout_report.csv", index=False)

    if args.transmission_only:
        transmission_fanout = pd.DataFrame(
            columns=[
                "transmission_id",
                "transmission_name",
                "transmission_type",
                "assigned_distribution_count",
                "mean_assignment_distance_km",
                "max_assignment_distance_km",
                "pct_of_all_distribution_substations",
            ]
        )
    else:
        transmission_fanout = (
            dist_assignments.groupby(
                [
                    "transmission_id",
                    "transmission_name",
                    "transmission_type",
                    "transmission_inside_montgomery",
                    "transmission_distance_to_boundary_km",
                ],
                as_index=False,
            )
            .agg(
                assigned_distribution_count=("distribution_id", "count"),
                mean_assignment_distance_km=("distance_km", "mean"),
                max_assignment_distance_km=("distance_km", "max"),
            )
        )
        transmission_fanout["pct_of_all_distribution_substations"] = (
            transmission_fanout["assigned_distribution_count"] / float(total_distribution) * 100.0
        )
        transmission_fanout.sort_values(
            ["assigned_distribution_count", "transmission_name"],
            ascending=[False, True],
            inplace=True,
        )
        transmission_fanout.to_csv(output_dir / "transmission_distribution_fanout_report.csv", index=False)

    previous_scope = scope_summary[scope_summary["scope"] == COMPARISON_SCOPE].iloc[0]
    anchor_substation = "BELLS MILL"
    anchor_rows = scope_fanout[
        (scope_fanout["scope"] == COMPARISON_SCOPE) & (scope_fanout["substation"] == anchor_substation)
    ].copy()
    if anchor_rows.empty:
        top_previous = scope_fanout[scope_fanout["scope"] == COMPARISON_SCOPE].sort_values(
            ["assigned_infrastructure_count", "substation"],
            ascending=[False, True],
        )
        if top_previous.empty:
            raise ValueError(f"No scope_fanout rows found for comparison scope {COMPARISON_SCOPE!r}.")
        anchor_rows = top_previous.head(1).copy()
        anchor_substation = str(anchor_rows.iloc[0]["substation"])
    previous_anchor = anchor_rows.iloc[0]

    new_anchor_direct_infra = int(
        (infra_assignments["distribution_name"].astype(str).str.upper() == anchor_substation.upper()).sum()
    )
    new_anchor_upstream_distribution = (
        int((dist_assignments["transmission_name"].astype(str).str.upper() == anchor_substation.upper()).sum())
        if not dist_assignments.empty
        else 0
    )

    infra_distance_stats = distribution_stats(infra_assignments["distance_km"])
    trans_distance_stats = (
        distribution_stats(dist_assignments["distance_km"])
        if not dist_assignments.empty
        else {"median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    )

    distribution_receiving_count = int((distribution_fanout["assigned_infrastructure_count"] > 0).sum())
    max_distribution_fanout = int(distribution_fanout["assigned_infrastructure_count"].max())
    median_distribution_fanout = float(distribution_fanout["assigned_infrastructure_count"].median())
    mean_distribution_fanout = float(distribution_fanout["assigned_infrastructure_count"].mean())
    largest_distribution_share = float(distribution_fanout["pct_of_all_infrastructure"].max())

    transmission_receiving_count = (
        int((transmission_fanout["assigned_distribution_count"] > 0).sum()) if not transmission_fanout.empty else 0
    )
    max_transmission_fanin = int(transmission_fanout["assigned_distribution_count"].max()) if not transmission_fanout.empty else 0
    median_transmission_fanin = float(transmission_fanout["assigned_distribution_count"].median()) if not transmission_fanout.empty else 0.0
    largest_transmission_share = float(transmission_fanout["pct_of_all_distribution_substations"].max()) if not transmission_fanout.empty else 0.0

    current_direct_power_edges = dependency_edges[
        dependency_edges["src_type"].isin(INFRA_TYPES) & (dependency_edges["dst_type"] == "power")
    ].copy()
    current_direct_power_node_count = int(current_direct_power_edges["src_node_id"].nunique())

    flag_large_total = distribution_fanout[distribution_fanout["pct_of_all_infrastructure"] > 15.0].copy()
    flag_large_sector_rows: list[dict] = []
    for row in distribution_fanout.itertuples(index=False):
        for sector_col, sector_name in (
            ("schools", "school"),
            ("hospitals", "hospital"),
            ("telecom", "telecom"),
            ("ems_fire", "ems_fire"),
        ):
            total_sector = float(infra_sector_totals.get(sector_name, 0))
            pct = 0.0 if total_sector == 0 else float(getattr(row, sector_col)) / total_sector * 100.0
            if pct > 20.0:
                flag_large_sector_rows.append(
                    {
                        "distribution_name": row.distribution_name,
                        "sector": sector_name,
                        "sector_count": int(getattr(row, sector_col)),
                        "sector_pct": pct,
                    }
                )
    flag_large_sector = pd.DataFrame(flag_large_sector_rows)
    flag_infra_gt10 = infra_assignments[infra_assignments["distance_km"] > 10.0].copy()
    flag_infra_gt20 = infra_assignments[infra_assignments["distance_km"] > 20.0].copy()
    flag_dist_gt20 = dist_assignments[dist_assignments["distance_km"] > 20.0].copy() if not dist_assignments.empty else pd.DataFrame()
    flag_no_distribution = infra_assignments[infra_assignments["distribution_id"].isna()].copy()

    recommendation = "PASS"
    fail_reasons: list[str] = []
    if not flag_large_total.empty:
        recommendation = "FAIL"
        if args.transmission_only:
            fail_reasons.append(
                f"{len(flag_large_total)} transmission substations serve more than 15% of all infrastructure"
            )
        else:
            fail_reasons.append(
                f"{len(flag_large_total)} OSM distribution substations serve more than 15% of all infrastructure"
            )
    if not flag_large_sector.empty:
        recommendation = "FAIL"
        fail_reasons.append(
            f"{len(flag_large_sector)} substation-sector combinations exceed 20% of a sector"
        )
    if not flag_infra_gt10.empty:
        recommendation = "FAIL"
        fail_reasons.append(
            f"{len(flag_infra_gt10)} infrastructure-to-{'transmission' if args.transmission_only else 'distribution'} assignments exceed 10 km"
        )
    if not flag_dist_gt20.empty:
        recommendation = "FAIL"
        if not args.transmission_only:
            fail_reasons.append(
                f"{len(flag_dist_gt20)} distribution-to-transmission assignments exceed 20 km"
            )
    if not flag_no_distribution.empty:
        recommendation = "FAIL"
        fail_reasons.append(
            f"{len(flag_no_distribution)} infrastructure nodes lack a distribution assignment"
        )

    report_lines = [
        "# Final Power Hierarchy Validation",
        "",
        "This is a read-only candidate hierarchy audit. No production graph files or simulator logic were modified.",
        "",
        "## Candidate Hierarchy",
        "",
        "- Power Plant -> HIFLD transmission network uses the existing validated file:",
        f"  `{args.power_plant_connections}`",
        f"- Validated plant->transmission connections reused: `{len(power_connections)}` rows across `{power_connections['power_node_id'].nunique()}` power plants",
        f"- Eligible HIFLD transmission substations: `{len(eligible_transmission)}`",
        f"- Eligible {'transmission service substations' if args.transmission_only else 'OSM local/distribution substations'}: `{len(candidate_distribution)}`",
        f"- Infrastructure nodes assigned through the hierarchy: `{total_infra}`",
        "",
        f"## Infrastructure -> {'Transmission' if args.transmission_only else 'Distribution'} Layer",
        "",
        f"- {'Transmission' if args.transmission_only else 'OSM distribution'} substations receiving infrastructure: `{distribution_receiving_count}`",
        f"- Maximum fan-out: `{max_distribution_fanout}`",
        f"- Median fan-out: `{median_distribution_fanout:.1f}`",
        f"- Mean fan-out: `{mean_distribution_fanout:.2f}`",
        f"- Largest substation share of all infrastructure: `{largest_distribution_share:.1f}%`",
        f"- Distance median/mean/p90/max km: `{infra_distance_stats['median']:.3f}` / `{infra_distance_stats['mean']:.3f}` / `{infra_distance_stats['p90']:.3f}` / `{infra_distance_stats['max']:.3f}`",
        f"- Count >5 km: `{int((infra_assignments['distance_km'] > 5.0).sum())}`",
        f"- Count >10 km: `{int((infra_assignments['distance_km'] > 10.0).sum())}`",
        f"- Count >20 km: `{int((infra_assignments['distance_km'] > 20.0).sum())}`",
        "",
        f"Top 10 highest-fanout {'transmission' if args.transmission_only else 'distribution'} substations:",
        "",
    ]
    for row in distribution_fanout.head(10).itertuples(index=False):
        report_lines.append(
            f"- `{row.distribution_name}`: total `{int(row.assigned_infrastructure_count)}`, schools `{int(row.schools)}`, hospitals `{int(row.hospitals)}`, telecom `{int(row.telecom)}`, ems `{int(row.ems_fire)}`, mean km `{float(row.mean_assignment_distance_km):.3f}`, max km `{float(row.max_assignment_distance_km):.3f}`"
        )

    report_lines.extend(
        [
            "",
            f"## {'Critical Comparison vs Previous Transmission-Only Assignment' if not args.transmission_only else 'Critical Comparison vs Previous Scope'}",
            "",
            f"- Previous `{anchor_substation}` infrastructure fan-out (`{COMPARISON_SCOPE}`): `{int(previous_anchor['assigned_infrastructure_count'])}`",
            f"- New `{anchor_substation}` direct infrastructure fan-out: `{new_anchor_direct_infra}`",
            f"- {'OSM distribution substations assigned upstream to' if not args.transmission_only else 'Direct upstream-transmission count anchored at'} `{anchor_substation}`: `{new_anchor_upstream_distribution}`",
            f"- Previous maximum infrastructure fan-out: `{int(previous_scope['max_fanout'])}` at `{previous_scope['max_fanout_substation']}`",
            f"- New maximum {'transmission' if args.transmission_only else 'distribution-substation'} infrastructure fan-out: `{max_distribution_fanout}` at `{distribution_fanout.iloc[0]['distribution_name']}`",
            f"- Previous median infrastructure assignment distance km: `{float(previous_scope['median_distance_km']):.3f}`",
            f"- New median infrastructure -> {'transmission' if args.transmission_only else 'distribution'} distance km: `{infra_distance_stats['median']:.3f}`",
            "",
            "## Sanity Checks",
            "",
            f"- Any {'transmission' if args.transmission_only else 'OSM distribution'} substation serving >15% of all infrastructure: `{len(flag_large_total)}`",
            f"- Any {'transmission' if args.transmission_only else 'OSM distribution'} substation serving >20% of one infrastructure sector: `{len(flag_large_sector)}`",
            f"- Any infrastructure -> {'transmission' if args.transmission_only else 'distribution'} assignment >10 km: `{len(flag_infra_gt10)}`",
            f"- Any {'distribution -> transmission' if not args.transmission_only else 'second-tier upstream'} assignment >20 km: `{len(flag_dist_gt20)}`",
            f"- Any infrastructure node that remains directly connected to a power plant in the candidate hierarchy: `0`",
            f"- Any infrastructure node with no {'transmission' if args.transmission_only else 'distribution-substation'} assignment: `{len(flag_no_distribution)}`",
            "",
            "Current production graph context:",
            "",
            f"- Infrastructure nodes with direct power-plant dependencies in the current production graph: `{current_direct_power_node_count}`",
            f"- Direct infrastructure -> power dependency edges currently present: `{len(current_direct_power_edges)}`",
            "",
            "## Recommendation",
            "",
            f"- `{recommendation}`",
        ]
    )
    if fail_reasons:
        report_lines.extend(["", "Reasons:"])
        for reason in fail_reasons:
            report_lines.append(f"- {reason}")
    else:
        report_lines.append("- Topology is reasonable enough to proceed to the timing experiment.")

    report_path = output_dir / "final_power_hierarchy_validation_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "recommendation": recommendation,
        "fail_reasons": fail_reasons,
        "infrastructure_node_count": total_infra,
        "eligible_distribution_count": int(len(candidate_distribution)),
        "eligible_transmission_count": int(len(eligible_transmission)),
        "previous_scope": COMPARISON_SCOPE,
        "comparison_anchor_substation": anchor_substation,
        "previous_anchor_fanout": int(previous_anchor["assigned_infrastructure_count"]),
        "new_anchor_direct_fanout": int(new_anchor_direct_infra),
        "new_anchor_upstream_distribution_count": int(new_anchor_upstream_distribution),
        "infra_distance_stats": infra_distance_stats,
        "trans_distance_stats": trans_distance_stats,
        "flag_counts": {
            "gt15pct_total": int(len(flag_large_total)),
            "gt20pct_sector": int(len(flag_large_sector)),
            "infra_gt10km": int(len(flag_infra_gt10)),
            "infra_gt20km": int(len(flag_infra_gt20)),
            "dist_gt20km": int(len(flag_dist_gt20)),
            "no_distribution": int(len(flag_no_distribution)),
        },
    }
    (output_dir / "final_power_hierarchy_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
