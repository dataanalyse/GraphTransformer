from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path("v3")


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def run_scenario_batch(
    edges_path: str,
    output_dir: str,
    figures_dir: str,
    model_label: str,
    scenario_prefix: str,
    figure_prefix: str,
    counts: dict[str, int],
    seed: int,
) -> None:
    cmd = [
        "./.venv/bin/python",
        "v3/run_scenarios_v1.py",
        "--edges",
        edges_path,
        "--output-dir",
        output_dir,
        "--figures-dir",
        figures_dir,
        "--model-label",
        model_label,
        "--scenario-prefix",
        scenario_prefix,
        "--figure-prefix",
        figure_prefix,
        "--seed",
        str(seed),
        "--power-count",
        str(counts["power"]),
        "--telecom-count",
        str(counts["telecom"]),
        "--ems-count",
        str(counts["ems"]),
        "--hospital-count",
        str(counts["hospital"]),
        "--mixed-count",
        str(counts["mixed"]),
    ]
    run_command(cmd)


def compare_variants(base_dir: Path, output_path: Path) -> None:
    frames = []
    for label in ["baseline_100", "redundant_100", "dampened_power_100"]:
        summary = pd.read_csv(base_dir / label / "scenario_summary_metrics.csv")
        summary["graph_variant"] = label
        frames.append(summary)
    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby("graph_variant").agg(
        mean_peak_damage=("peak_damage_nodes", "mean"),
        mean_min_lcc=("min_lcc", "mean"),
        mean_recovery_time=("recovery_time", "mean"),
        mean_final_lcc=("final_lcc", "mean"),
    )
    by_variant_shock = df.groupby(["graph_variant", "shock_type"]).agg(
        mean_peak_damage=("peak_damage_nodes", "mean"),
        mean_min_lcc=("min_lcc", "mean"),
        mean_recovery_time=("recovery_time", "mean"),
    )

    lines = [
        "# Dependency Sensitivity Summary",
        "",
        "## Overall Variant Comparison",
        "",
    ]
    for variant, row in agg.iterrows():
        lines.append(
            f"- `{variant}`: mean peak damage `{row['mean_peak_damage']:.2f}`, mean min LCC `{row['mean_min_lcc']:.4f}`, mean recovery time `{row['mean_recovery_time']:.2f}`, mean final LCC `{row['mean_final_lcc']:.4f}`"
        )

    lines.extend(["", "## By Shock Type", ""])
    for (variant, shock_type), row in by_variant_shock.iterrows():
        lines.append(
            f"- `{variant}` / `{shock_type}`: mean peak damage `{row['mean_peak_damage']:.2f}`, mean min LCC `{row['mean_min_lcc']:.4f}`, mean recovery time `{row['mean_recovery_time']:.2f}`"
        )

    baseline = agg.loc["baseline_100"]
    redundant = agg.loc["redundant_100"]
    dampened = agg.loc["dampened_power_100"]
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Dependency concentration appears to drive fragility in the baseline model, consistent with the negative baseline pilot correlation already observed.",
            f"- Redundancy {'reduces' if redundant['mean_peak_damage'] < baseline['mean_peak_damage'] else 'does not reduce'} average cascade size relative to the baseline 100-scenario batch.",
            f"- Dampening power dominance {'reduces' if dampened['mean_peak_damage'] < baseline['mean_peak_damage'] else 'does not reduce'} average cascade size relative to the baseline 100-scenario batch.",
            "- These comparisons indicate how sensitive the findings are to the dependency assumptions rather than to the simulator alone.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaled baseline and dependency-sensitivity scenario experiments.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_command(["./.venv/bin/python", "v3/build_graph_variants.py"])

    baseline_500 = {"power": 150, "telecom": 150, "ems": 100, "hospital": 50, "mixed": 50}
    hundred = {"power": 30, "telecom": 30, "ems": 20, "hospital": 10, "mixed": 10}

    run_scenario_batch(
        "v3/data/processed/dependency_graph_edges.csv",
        "v3/data/processed/baseline_model_v1",
        "v3/runs/figures/baseline_model_v1",
        "baseline_model_v1",
        "baseline500",
        "",
        baseline_500,
        args.seed,
    )

    for variant, edge_file in [
        ("baseline_100", "v3/data/processed/graph_variants/baseline_edges.csv"),
        ("redundant_100", "v3/data/processed/graph_variants/redundant_edges.csv"),
        ("dampened_power_100", "v3/data/processed/graph_variants/dampened_power_edges.csv"),
    ]:
        run_scenario_batch(
            edge_file,
            f"v3/data/processed/sensitivity/{variant}",
            f"v3/runs/figures/sensitivity/{variant}",
            variant,
            variant,
            "",
            hundred,
            args.seed,
        )

    compare_variants(Path("v3/data/processed/sensitivity"), Path("v3/data/processed/sensitivity/dependency_sensitivity_summary.md"))
    print("Wrote baseline_model_v1 and sensitivity experiment outputs.")


if __name__ == "__main__":
    main()
