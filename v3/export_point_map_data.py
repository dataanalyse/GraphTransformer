from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


POINT_LAYERS = [
    "hospitals",
    "ems_fire",
    "power_plants",
    "county_telecom",
    "local_eoc",
    "public_schools",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export point-like v3 infrastructure assets for mapping.")
    parser.add_argument(
        "--inventory",
        default="v3/data/processed/asset_inventory.parquet",
        help="Path to the normalized asset inventory parquet.",
    )
    parser.add_argument(
        "--output",
        default="v3/data/processed/point_asset_inventory.csv",
        help="Path to the map-ready CSV export.",
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inventory_path)
    point_df = df[df["layer_key"].isin(POINT_LAYERS)].copy()
    point_df = point_df.dropna(subset=["latitude", "longitude"])

    point_df["display_name"] = point_df["name"].fillna(point_df["node_id"])
    point_df["layer_label"] = point_df["layer_key"].map(
        {
            "hospitals": "Hospitals",
            "ems_fire": "EMS / Fire",
            "power_plants": "Power Plants",
            "county_telecom": "Telecom Sites",
            "local_eoc": "Local EOC",
            "public_schools": "Public Schools",
        }
    )

    keep_cols = [
        "node_id",
        "layer_key",
        "layer_label",
        "node_type",
        "display_name",
        "name",
        "source_id",
        "latitude",
        "longitude",
    ]
    point_df[keep_cols].to_csv(output_path, index=False)
    print(f"Wrote map-ready point asset CSV to {output_path}")
    print(point_df["layer_key"].value_counts().to_string())


if __name__ == "__main__":
    main()
