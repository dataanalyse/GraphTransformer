from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def _read_sample(path: Path, n: int) -> pd.DataFrame:
    return pd.read_parquet(path).head(n)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda x: x.hex() if isinstance(x, (bytes, bytearray)) else x
        )
    return safe_df.to_dict(orient="records")


def inspect_layers(config_path: Path, max_rows: int) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_root = Path(config["paths"]["raw_root"])
    results: list[dict] = []

    for layer in config.get("layers", []):
        if not layer.get("enabled", True):
            continue

        matches = sorted(raw_root.glob(layer["parquet_glob"]))
        layer_result = {
            "layer_key": layer["layer_key"],
            "node_type": layer["node_type"],
            "pattern": layer["parquet_glob"],
            "match_count": len(matches),
            "files": [],
        }

        for path in matches:
            try:
                sample = _read_sample(path, max_rows)
                file_info = {
                    "path": str(path),
                    "columns": sample.columns.tolist(),
                    "sample_rows": _json_safe_records(sample),
                }
            except Exception as exc:
                file_info = {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            layer_result["files"].append(file_info)

        results.append(layer_result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured parquet layers for v3.")
    parser.add_argument(
        "--config",
        default="v3/configs/infrastructure_layers.yaml",
        help="Path to infrastructure layer config.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=3,
        help="Number of sample rows to show per parquet file.",
    )
    parser.add_argument(
        "--output",
        default="v3/data/processed/parquet_layer_inspection.json",
        help="Where to write the inspection summary.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = inspect_layers(config_path, args.max_rows)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote parquet inspection summary to {output_path}")
    for item in results:
        print(f"{item['layer_key']}: {item['match_count']} file(s)")


if __name__ == "__main__":
    main()
