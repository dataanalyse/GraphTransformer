from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LAYER_ORDER = [
    "public_schools",
    "cellular_towers",
    "power_plants",
    "ems_fire",
    "hospitals",
    "local_eoc",
]

LAYER_STYLE = {
    "public_schools": {"color": "#7aa6c2", "size": 10, "label": "Public Schools"},
    "cellular_towers": {"color": "#8b6fb3", "size": 28, "label": "Cellular Towers"},
    "power_plants": {"color": "#d17c2f", "size": 42, "label": "Power Plants"},
    "ems_fire": {"color": "#c84d4d", "size": 30, "label": "EMS / Fire"},
    "hospitals": {"color": "#2a9d8f", "size": 48, "label": "Hospitals"},
    "local_eoc": {"color": "#222222", "size": 80, "label": "Local EOC"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Montgomery County infrastructure point assets.")
    parser.add_argument(
        "--input",
        default="v3/data/processed/point_asset_inventory.csv",
        help="Path to map-ready point asset CSV.",
    )
    parser.add_argument(
        "--output",
        default="v3/runs/figures/montgomery_point_assets_plain.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--basemap",
        action="store_true",
        help="Add a light web basemap with labels using contextily.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Point asset CSV is empty.")

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.set_facecolor("white")

    for layer_key in LAYER_ORDER:
        layer_df = df[df["layer_key"] == layer_key]
        if layer_df.empty:
            continue
        style = LAYER_STYLE[layer_key]
        ax.scatter(
            layer_df["longitude"],
            layer_df["latitude"],
            s=style["size"],
            c=style["color"],
            alpha=0.85,
            label=style["label"],
            edgecolors="white",
            linewidths=0.4,
        )

    xmin = df["longitude"].min()
    xmax = df["longitude"].max()
    ymin = df["latitude"].min()
    ymax = df["latitude"].max()
    xpad = max((xmax - xmin) * 0.08, 0.01)
    ypad = max((ymax - ymin) * 0.08, 0.01)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Montgomery County Infrastructure Assets", fontsize=15, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")

    if args.basemap:
        import contextily as ctx

        ax.grid(False)
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.CartoDB.Positron,
            attribution_size=6,
            zoom=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    map_type = "basemap" if args.basemap else "plain"
    print(f"Wrote {map_type} point asset map to {output_path}")


if __name__ == "__main__":
    main()
