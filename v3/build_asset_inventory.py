from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def _pick_existing_column(df: pd.DataFrame, configured: str | None, fallbacks: list[str]) -> str | None:
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(fallbacks)
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _bbox_centroid_series(series: pd.Series, axis: str) -> pd.Series:
    def extract(value):
        if isinstance(value, dict):
            if axis == "lat":
                ymin = value.get("ymin")
                ymax = value.get("ymax")
                if ymin is not None and ymax is not None:
                    return (float(ymin) + float(ymax)) / 2.0
            if axis == "lon":
                xmin = value.get("xmin")
                xmax = value.get("xmax")
                if xmin is not None and xmax is not None:
                    return (float(xmin) + float(xmax)) / 2.0
        return None

    return series.map(extract)


def _build_source_id(df: pd.DataFrame, layer_cfg: dict) -> pd.Series:
    composite_cols = layer_cfg.get("composite_id_columns") or []
    if composite_cols:
        resolved_cols: list[str] = []
        for col in composite_cols:
            resolved = _pick_existing_column(df, col, [])
            if resolved:
                resolved_cols.append(resolved)
        if resolved_cols:
            parts = [df[col].astype(str).str.strip() for col in resolved_cols]
            source_id = parts[0]
            for part in parts[1:]:
                source_id = source_id + "__" + part
            return source_id

    id_col = _pick_existing_column(df, layer_cfg.get("id_column"), ["id", "objectid", "globalid", "fid"])
    if id_col:
        return df[id_col].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)


def _normalize_layer(df: pd.DataFrame, layer_cfg: dict, region_cfg: dict) -> pd.DataFrame:
    county_col = _pick_existing_column(df, layer_cfg.get("county_column"), ["county", "county_name", "cnty_name"])
    state_col = _pick_existing_column(df, layer_cfg.get("state_column"), ["state", "state_name", "state_abbr", "st"])
    city_col = _pick_existing_column(df, layer_cfg.get("city_column"), ["city", "city_name"])
    zip_col = _pick_existing_column(df, layer_cfg.get("zip_column"), ["zip", "zipcode", "zip_code", "postal_code"])

    if county_col:
        df = df[df[county_col].astype(str).str.lower() == str(region_cfg["county"]).lower()]
    if state_col:
        state_value = str(region_cfg["state"])
        state_aliases = {
            "maryland": {"maryland", "md"},
            "virginia": {"virginia", "va"},
            "district of columbia": {"district of columbia", "dc"},
        }
        allowed = state_aliases.get(state_value.lower(), {state_value.lower(), state_value[:2].lower()})
        df = df[df[state_col].astype(str).str.lower().isin(allowed)]

    # Some HIFLD layers do not carry a county field; use provisional city/ZIP filters until
    # we replace this with a stricter geographic boundary clip.
    allowed_cities = layer_cfg.get("allowed_cities") or []
    allowed_zips = layer_cfg.get("allowed_zips") or []
    if not county_col and (allowed_cities or allowed_zips):
        city_mask = pd.Series(False, index=df.index)
        zip_mask = pd.Series(False, index=df.index)
        if city_col and allowed_cities:
            allowed_city_set = {str(x).lower() for x in allowed_cities}
            city_mask = df[city_col].astype(str).str.lower().isin(allowed_city_set)
        if zip_col and allowed_zips:
            allowed_zip_set = {str(x) for x in allowed_zips}
            zip_mask = df[zip_col].astype(str).isin(allowed_zip_set)
        df = df[city_mask | zip_mask]

    name_col = _pick_existing_column(df, layer_cfg.get("name_column"), ["name", "facility_name", "site_name"])
    lat_col = _pick_existing_column(df, layer_cfg.get("lat_column"), ["latitude", "lat", "y"])
    lon_col = _pick_existing_column(df, layer_cfg.get("lon_column"), ["longitude", "lon", "lng", "x"])
    bbox_col = _pick_existing_column(df, None, ["bbox"])

    latitude = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None
    longitude = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None

    if bbox_col:
        bbox_lat = _bbox_centroid_series(df[bbox_col], "lat")
        bbox_lon = _bbox_centroid_series(df[bbox_col], "lon")
        if latitude is None:
            latitude = pd.to_numeric(bbox_lat, errors="coerce")
        else:
            latitude = latitude.fillna(pd.to_numeric(bbox_lat, errors="coerce"))
        if longitude is None:
            longitude = pd.to_numeric(bbox_lon, errors="coerce")
        else:
            longitude = longitude.fillna(pd.to_numeric(bbox_lon, errors="coerce"))

    source_id = _build_source_id(df, layer_cfg)

    out = pd.DataFrame(
        {
            "layer_key": layer_cfg["layer_key"],
            "node_type": layer_cfg["node_type"],
            "source_id": source_id,
            "name": df[name_col].astype(str) if name_col else None,
            "latitude": latitude,
            "longitude": longitude,
        }
    )
    out["node_id"] = out["layer_key"] + "::" + out["source_id"]
    return out


def build_inventory(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_root = Path(config["paths"]["raw_root"])
    region_cfg = config["region"]
    inventories: list[pd.DataFrame] = []
    layer_summaries: list[dict] = []

    for layer in config.get("layers", []):
        if not layer.get("enabled", True):
            continue

        pattern = layer.get("parquet_glob") or layer.get("csv_glob")
        root_key = layer.get("root_key", "raw_root")
        root = Path(config["paths"][root_key])
        matches = sorted(root.glob(pattern)) if pattern else []
        if not matches:
            layer_summaries.append(
                {"layer_key": layer["layer_key"], "node_type": layer["node_type"], "file_count": 0, "asset_count": 0}
            )
            continue

        frames = []
        for path in matches:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            frames.append(_normalize_layer(df, layer, region_cfg))

        layer_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        inventories.append(layer_df)
        layer_summaries.append(
            {
                "layer_key": layer["layer_key"],
                "node_type": layer["node_type"],
                "file_count": len(matches),
                "asset_count": int(len(layer_df)),
                "with_coordinates": int(layer_df[["latitude", "longitude"]].dropna().shape[0]) if not layer_df.empty else 0,
            }
        )

    inventory_df = pd.concat(inventories, ignore_index=True) if inventories else pd.DataFrame(
        columns=["layer_key", "node_type", "source_id", "name", "latitude", "longitude", "node_id"]
    )
    summary_df = pd.DataFrame(layer_summaries)
    return inventory_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a first-pass infrastructure asset inventory from parquet layers.")
    parser.add_argument(
        "--config",
        default="v3/configs/infrastructure_layers.yaml",
        help="Path to infrastructure layer config.",
    )
    parser.add_argument(
        "--inventory-out",
        default="v3/data/processed/asset_inventory.parquet",
        help="Path for the normalized asset inventory parquet.",
    )
    parser.add_argument(
        "--summary-out",
        default="v3/data/processed/asset_inventory_summary.csv",
        help="Path for the layer summary CSV.",
    )
    args = parser.parse_args()

    inventory_out = Path(args.inventory_out)
    summary_out = Path(args.summary_out)
    inventory_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    inventory_df, summary_df = build_inventory(Path(args.config))
    inventory_df.to_parquet(inventory_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Wrote inventory to {inventory_out}")
    print(f"Wrote summary to {summary_out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
