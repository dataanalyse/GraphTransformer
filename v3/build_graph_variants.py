from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

try:
    from v3.build_dependency_graph import POINT_LAYERS, build_edges
except ModuleNotFoundError:
    from build_dependency_graph import POINT_LAYERS, build_edges


def load_nodes(inventory_path: str) -> pd.DataFrame:
    nodes = pd.read_parquet(inventory_path)
    nodes = nodes[nodes["layer_key"].isin(POINT_LAYERS)].copy()
    nodes = nodes.dropna(subset=["latitude", "longitude"])
    return nodes


def redundant_rules(rules: list[dict]) -> list[dict]:
    out = []
    for rule in rules:
        rule = dict(rule)
        if rule["src_type"] in {"hospital", "school", "telecom", "ems_fire"}:
            rule["max_targets"] = max(int(rule.get("max_targets", 1)), 3)
        out.append(rule)
    return out


def dampened_power_edges(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    mask = out["dst_type"] == "power"
    out.loc[mask, "weight"] = out.loc[mask, "weight"] * 0.8
    return out


def write_summary(path: Path, label: str, edges: pd.DataFrame) -> None:
    counts = edges.groupby(["src_type", "dst_type"]).size().sort_values(ascending=False)
    lines = [
        f"# {label} Graph Variant Summary",
        "",
        f"- Edge count: `{len(edges)}`",
        "",
        "## Edge Counts by Dependency Type",
        "",
    ]
    for (src, dst), count in counts.items():
        lines.append(f"- `{src} -> {dst}`: `{count}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build alternate dependency-graph edge variants.")
    parser.add_argument("--inventory", default="v3/data/processed/asset_inventory.parquet")
    parser.add_argument("--config", default="v3/configs/infrastructure_layers.yaml")
    parser.add_argument("--output-dir", default="v3/data/processed/graph_variants")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    nodes = load_nodes(args.inventory)
    rules = config.get("dependency_rules", [])
    baseline_edges = build_edges(nodes, rules)
    redundant_edges = build_edges(nodes, redundant_rules(rules))
    dampened_edges = dampened_power_edges(baseline_edges)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes.to_csv(output_dir / "dependency_graph_nodes.csv", index=False)
    baseline_edges.to_csv(output_dir / "baseline_edges.csv", index=False)
    redundant_edges.to_csv(output_dir / "redundant_edges.csv", index=False)
    dampened_edges.to_csv(output_dir / "dampened_power_edges.csv", index=False)

    write_summary(output_dir / "baseline_summary.md", "Baseline", baseline_edges)
    write_summary(output_dir / "redundant_summary.md", "Redundant", redundant_edges)
    write_summary(output_dir / "dampened_power_summary.md", "Dampened Power", dampened_edges)

    print(f"Wrote graph variants to {output_dir}")


if __name__ == "__main__":
    main()
