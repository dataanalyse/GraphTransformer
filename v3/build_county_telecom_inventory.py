from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACTIVE_ACTIONS = {
    "recommended",
    "pending - not complete",
    "under review",
    "info only",
}

INACTIVE_ACTIONS = {
    "withdrawn",
    "not recommended",
}

GRAPH_READY_STRUCTURES = {
    "tower",
    "monopole",
    "water tank",
}


def normalize_action(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def latest_site_records(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["action_date_parsed"] = pd.to_datetime(work["Action Date"], errors="coerce")
    work["rcvd_parsed"] = pd.to_datetime(work["Rcvd"], errors="coerce")
    work["effective_date"] = work["action_date_parsed"].fillna(work["rcvd_parsed"])
    work["action_normalized"] = work["Action"].map(normalize_action)
    work = work.sort_values(["SiteID", "effective_date", "Permit_Number"], ascending=[True, True, True])
    latest = work.groupby("SiteID", as_index=False).tail(1).copy()
    latest["site_status"] = latest["action_normalized"].map(
        lambda x: "active" if x in ACTIVE_ACTIONS else ("inactive" if x in INACTIVE_ACTIONS else "review")
    )
    return latest


def summarize_sites(latest: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "total_latest_site_records", "value": int(len(latest))},
        {"metric": "active_sites", "value": int((latest["site_status"] == "active").sum())},
        {"metric": "inactive_sites", "value": int((latest["site_status"] == "inactive").sum())},
        {"metric": "review_sites", "value": int((latest["site_status"] == "review").sum())},
        {"metric": "unique_coordinates", "value": int(latest[["LAT", "LONG_"]].dropna().drop_duplicates().shape[0])},
        {"metric": "siteid_unique", "value": int(latest["SiteID"].nunique(dropna=True))},
    ]
    action_counts = latest["Action"].fillna("NA").value_counts().sort_index()
    for action, count in action_counts.items():
        rows.append({"metric": "latest_action_count", "group": action, "value": int(count)})
    return pd.DataFrame(rows)


def build_active_inventory(latest: pd.DataFrame) -> pd.DataFrame:
    active = latest[latest["site_status"] == "active"].copy()
    active["node_id"] = active["SiteID"].astype(str).map(lambda x: f"county_telecom::{x}")
    active["display_name"] = active["SiteName"].fillna(active["SiteID"].astype(str))
    active["carrier_count_sitewide"] = (
        latest.groupby("SiteID")["CarrierName"].transform("nunique").reindex(active.index).fillna(1).astype(int)
    )

    keep = [
        "node_id",
        "SiteID",
        "display_name",
        "SiteName",
        "CarrierName",
        "carrier_count_sitewide",
        "Address",
        "Street",
        "City",
        "Zone",
        "Owner",
        "Type",
        "Type2",
        "Structure",
        "Height",
        "LAT",
        "LONG_",
        "Action",
        "site_status",
        "effective_date",
        "Permit_Number",
        "webUrl",
    ]
    existing = [c for c in keep if c in active.columns]
    return active[existing].sort_values(["City", "SiteID"]).reset_index(drop=True)


def build_graph_ready_inventory(active_df: pd.DataFrame) -> pd.DataFrame:
    graph_ready = active_df.copy()
    graph_ready["structure_normalized"] = graph_ready["Structure"].fillna("").astype(str).str.strip().str.lower()
    graph_ready = graph_ready[graph_ready["structure_normalized"].isin(GRAPH_READY_STRUCTURES)].copy()
    graph_ready["layer_key"] = "county_telecom"
    graph_ready["node_type"] = "telecom"
    graph_ready["source_id"] = graph_ready["SiteID"].astype(str)
    graph_ready["name"] = graph_ready["display_name"]
    graph_ready["latitude"] = pd.to_numeric(graph_ready["LAT"], errors="coerce")
    graph_ready["longitude"] = pd.to_numeric(graph_ready["LONG_"], errors="coerce")
    graph_ready["node_id"] = graph_ready["layer_key"] + "::" + graph_ready["source_id"]
    keep = [
        "layer_key",
        "node_type",
        "source_id",
        "name",
        "latitude",
        "longitude",
        "node_id",
        "SiteID",
        "display_name",
        "CarrierName",
        "carrier_count_sitewide",
        "City",
        "Structure",
        "Height",
        "Action",
        "site_status",
        "effective_date",
        "Permit_Number",
        "webUrl",
    ]
    existing = [c for c in keep if c in graph_ready.columns]
    return graph_ready[existing].sort_values(["City", "SiteID"]).reset_index(drop=True)


def build_markdown_summary(summary_df: pd.DataFrame, active_df: pd.DataFrame) -> str:
    metric_lookup = summary_df.set_index("metric")["value"].to_dict()
    action_counts = summary_df[summary_df["metric"] == "latest_action_count"][["group", "value"]]

    lines = [
        "# County Telecom Inventory Summary",
        "",
        "## Latest-Record Site Inventory",
        "",
        f"- Latest-site records: `{int(metric_lookup['total_latest_site_records'])}`",
        f"- Active sites kept: `{int(metric_lookup['active_sites'])}`",
        f"- Inactive sites excluded: `{int(metric_lookup['inactive_sites'])}`",
        f"- Review-needed sites: `{int(metric_lookup['review_sites'])}`",
        f"- Unique coordinate pairs across latest site records: `{int(metric_lookup['unique_coordinates'])}`",
        "",
        "## Latest Action Counts",
        "",
    ]
    for _, row in action_counts.iterrows():
        lines.append(f"- `{row['group']}`: `{int(row['value'])}`")

    lines.extend(["", "## Active Site Sample", ""])
    for _, row in active_df.head(15).iterrows():
        lines.append(
            f"- `SiteID {row['SiteID']}` | `{row['display_name']}` | `{row['City']}` | `{row['Structure']}` | latest action `{row['Action']}`"
        )
    return "\n".join(lines) + "\n"


def build_graph_ready_markdown(graph_ready_df: pd.DataFrame) -> str:
    lines = [
        "# Graph-Ready County Telecom Summary",
        "",
        "## Filtering Rule",
        "",
        "- Source: latest active county telecom site inventory.",
        "- Kept structures: `Tower`, `Monopole`, `Water Tank`.",
        "- Excluded for now: `Building`, `Utility Pole`, `Light Pole`, `COW`, and other small or temporary structures.",
        "",
        f"- Graph-ready telecom nodes: `{len(graph_ready_df)}`",
        "",
        "## Structure Counts",
        "",
    ]
    for structure, count in graph_ready_df["Structure"].fillna("NA").value_counts().items():
        lines.append(f"- `{structure}`: `{count}`")
    lines.extend(["", "## Sample", ""])
    for _, row in graph_ready_df.head(15).iterrows():
        lines.append(
            f"- `SiteID {row['SiteID']}` | `{row['display_name']}` | `{row['City']}` | `{row['Structure']}` | latest action `{row['Action']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaned Montgomery County telecom site inventory from the county workbook.")
    parser.add_argument(
        "--input",
        default="/Users/ramnathsankaran/Downloads/MC Database 5_29_26.xlsx",
        help="Path to the county telecom Excel export.",
    )
    parser.add_argument(
        "--sheet",
        default="DB_Export",
        help="Worksheet name.",
    )
    parser.add_argument(
        "--active-out",
        default="v3/data/processed/county_telecom_active_sites.csv",
        help="Output CSV for one-row-per-active-site telecom inventory.",
    )
    parser.add_argument(
        "--latest-out",
        default="v3/data/processed/county_telecom_latest_site_records.csv",
        help="Output CSV for one-row-per-SiteID latest record inventory.",
    )
    parser.add_argument(
        "--summary-out",
        default="v3/data/processed/county_telecom_inventory_summary.csv",
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--summary-md-out",
        default="v3/data/processed/county_telecom_inventory_summary.md",
        help="Output markdown summary.",
    )
    parser.add_argument(
        "--graph-ready-out",
        default="v3/data/processed/county_telecom_graph_ready.csv",
        help="Output CSV for the graph-ready filtered county telecom subset.",
    )
    parser.add_argument(
        "--graph-ready-md-out",
        default="v3/data/processed/county_telecom_graph_ready_summary.md",
        help="Output markdown summary for the graph-ready filtered county telecom subset.",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    latest = latest_site_records(df)
    summary_df = summarize_sites(latest)
    active_df = build_active_inventory(latest)
    summary_md = build_markdown_summary(summary_df, active_df)
    graph_ready_df = build_graph_ready_inventory(active_df)
    graph_ready_md = build_graph_ready_markdown(graph_ready_df)

    for path_str in [args.active_out, args.latest_out, args.summary_out, args.summary_md_out, args.graph_ready_out, args.graph_ready_md_out]:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    active_df.to_csv(args.active_out, index=False)
    latest.to_csv(args.latest_out, index=False)
    summary_df.to_csv(args.summary_out, index=False)
    Path(args.summary_md_out).write_text(summary_md, encoding="utf-8")
    graph_ready_df.to_csv(args.graph_ready_out, index=False)
    Path(args.graph_ready_md_out).write_text(graph_ready_md, encoding="utf-8")

    print(f"Wrote active telecom inventory to {args.active_out}")
    print(f"Wrote latest-site inventory to {args.latest_out}")
    print(f"Wrote summary CSV to {args.summary_out}")
    print(f"Wrote summary markdown to {args.summary_md_out}")
    print(f"Wrote graph-ready telecom inventory to {args.graph_ready_out}")
    print(f"Wrote graph-ready telecom summary to {args.graph_ready_md_out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
