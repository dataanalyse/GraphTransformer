from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    from v3.simulator_v1 import ensure_simulation_edges
except ModuleNotFoundError:
    from simulator_v1 import ensure_simulation_edges


CONDITION_TO_EDGE_FILE = {
    "baseline_additive": "baseline_edges.csv",
    "redundant_additive": "redundant_edges.csv",
    "redundant_buffer": "redundant_edges.csv",
    "dampened_power_additive": "dampened_power_edges.csv",
}


def build_condition_features(graph_dir: Path, metadata_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metadata_map = metadata_df.set_index("scenario_id").to_dict("index")

    for condition, edge_file in CONDITION_TO_EDGE_FILE.items():
        edges_path = graph_dir / edge_file
        sim_edges_path = graph_dir / f"{condition}_for_model_sim_edges.csv"
        sim_edges = ensure_simulation_edges(
            str(graph_dir / "dependency_graph_nodes.csv"),
            str(edges_path),
            str(sim_edges_path),
        )
        outdeg = sim_edges["simulation_source"].value_counts().to_dict()
        mean_delay = sim_edges.groupby("simulation_source")["delay"].mean().to_dict()
        total_edges = float(len(sim_edges))

        subset = summary_df[summary_df["condition"] == condition].copy()
        for row in subset.itertuples(index=False):
            meta = metadata_map[row.scenario_id]
            seed_nodes = meta["seed_nodes"].split(";") if isinstance(meta["seed_nodes"], str) and meta["seed_nodes"] else []
            seed_outdegs = [outdeg.get(seed, 0.0) for seed in seed_nodes]
            seed_delays = [mean_delay.get(seed, 0.0) for seed in seed_nodes]
            mean_seed_degree = float(np.mean(seed_outdegs)) if seed_outdegs else 0.0
            dependency_concentration = float(sum(seed_outdegs) / total_edges) if total_edges > 0 else 0.0
            mean_propagation_delay = float(np.mean(seed_delays)) if seed_delays else 0.0
            rows.append(
                {
                    "scenario_id": row.scenario_id,
                    "condition": condition,
                    "shock_type": row.shock_type,
                    "min_lcc": row.min_lcc,
                    "seed_k": row.seed_k,
                    "shock_severity_scale": meta["shock_severity_scale"],
                    "recovery_scale": meta["recovery_scale"],
                    "propagation_scale": meta["propagation_scale"],
                    "mean_seed_degree_cond": mean_seed_degree,
                    "dependency_concentration_cond": dependency_concentration,
                    "mean_propagation_delay_cond": mean_propagation_delay,
                    "aggregation_mode": row.aggregation_mode,
                    "redundancy_threshold": row.redundancy_threshold,
                }
            )
    return pd.DataFrame(rows)


def partial_r2(full_model, reduced_model) -> float:
    return max(0.0, float(full_model.rsquared - reduced_model.rsquared))


def standardized_numeric_betas(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    numeric_cols = ["dependency_concentration_cond", "mean_seed_degree_cond", "mean_propagation_delay_cond"]
    for col in numeric_cols + ["min_lcc"]:
        work[col] = (work[col] - work[col].mean()) / work[col].std(ddof=0)

    model = smf.ols(
        "min_lcc ~ dependency_concentration_cond + mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type) + C(condition)",
        data=work,
    ).fit()
    rows = []
    for col in numeric_cols:
        rows.append(
            {
                "variable": col,
                "standardized_beta": model.params[col],
                "p_value": model.pvalues[col],
            }
        )
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, importance_df: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "power": "#c0392b",
        "telecom": "#2980b9",
        "ems": "#16a085",
        "hospital": "#8e44ad",
        "mixed": "#d35400",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for shock_type, sub in df.groupby("shock_type"):
        ax.scatter(
            sub["dependency_concentration_cond"],
            sub["min_lcc"],
            alpha=0.45,
            label=shock_type,
            color=colors.get(shock_type, "#666666"),
        )
    ax.set_title("Minimum LCC vs Dependency Concentration")
    ax.set_xlabel("Condition-Specific Dependency Concentration")
    ax.set_ylabel("Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dependency_concentration_multivariate_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp = importance_df.sort_values("partial_r2_drop", ascending=True)
    ax.barh(imp["variable_group"], imp["partial_r2_drop"], color="#34495e")
    ax.set_title("Variable Importance by Partial R-squared Drop")
    ax.set_xlabel("Drop in R-squared when variable/group is removed")
    fig.tight_layout()
    fig.savefig(output_dir / "resilience_factor_importance.png", dpi=200)
    plt.close(fig)


def write_report(report_path: Path, model, importance_df: pd.DataFrame, std_beta_df: pd.DataFrame) -> None:
    ci = model.conf_int()
    lines = [
        "# Multivariate Resilience Factors",
        "",
        "Model:",
        "`minimum_LCC ~ dependency_concentration + seed_node_degree + shock_type + propagation_delay + redundancy_condition`",
        "",
        "## Model Fit",
        "",
        f"- N: `{int(model.nobs)}`",
        f"- R-squared: `{model.rsquared:.3f}`",
        f"- Adjusted R-squared: `{model.rsquared_adj:.3f}`",
        "",
        "## Key Coefficients",
        "",
    ]

    for var in ["dependency_concentration_cond", "mean_seed_degree_cond", "mean_propagation_delay_cond"]:
        lines.append(
            f"- `{var}`: beta `{model.params[var]:.3f}`, 95% CI `[{ci.loc[var,0]:.3f}, {ci.loc[var,1]:.3f}]`, p=`{model.pvalues[var]:.3g}`"
        )

    lines.extend(["", "## Variable Importance Ranking", ""])
    for row in importance_df.sort_values("partial_r2_drop", ascending=False).itertuples(index=False):
        lines.append(f"- `{row.variable_group}`: partial R-squared drop `{row.partial_r2_drop:.3f}`")

    lines.extend(["", "## Standardized Numeric Effects", ""])
    for row in std_beta_df.sort_values("standardized_beta", key=lambda s: s.abs(), ascending=False).itertuples(index=False):
        lines.append(f"- `{row.variable}`: standardized beta `{row.standardized_beta:.3f}`, p=`{row.p_value:.3g}`")

    strongest = importance_df.sort_values("partial_r2_drop", ascending=False).iloc[0]["variable_group"]
    lines.extend(
        [
            "",
            "## Answer",
            "",
            f"- The strongest explanatory factor in this multivariate model is: `{strongest}`.",
            "- In this setup, resilience outcomes are explained jointly by structural concentration, shock family, and graph-condition choice rather than by any single scalar alone.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a multivariate model for minimum LCC.")
    parser.add_argument("--metadata", default="v3/data/processed/redundancy_v3/scenario_metadata_v3.csv")
    parser.add_argument("--summary", default="v3/data/processed/redundancy_v3/scenario_summary_metrics_v3.csv")
    parser.add_argument("--graph-dir", default="v3/data/processed/graph_variants")
    parser.add_argument("--output-dir", default="v3/data/processed/resilience_factor_model")
    parser.add_argument("--figures-dir", default="v3/runs/figures/resilience_factor_model")
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata)
    summary_df = pd.read_csv(args.summary)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_df = build_condition_features(graph_dir, metadata_df, summary_df)
    model_df.to_csv(output_dir / "resilience_model_dataset.csv", index=False)

    formula = (
        "min_lcc ~ dependency_concentration_cond + mean_seed_degree_cond + "
        "mean_propagation_delay_cond + C(shock_type) + C(condition)"
    )
    model = smf.ols(formula, data=model_df).fit()

    importance_rows = []
    full_r2 = model.rsquared
    reduced_specs = {
        "dependency_concentration": "min_lcc ~ mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type) + C(condition)",
        "seed_node_degree": "min_lcc ~ dependency_concentration_cond + mean_propagation_delay_cond + C(shock_type) + C(condition)",
        "propagation_delay": "min_lcc ~ dependency_concentration_cond + mean_seed_degree_cond + C(shock_type) + C(condition)",
        "shock_type": "min_lcc ~ dependency_concentration_cond + mean_seed_degree_cond + mean_propagation_delay_cond + C(condition)",
        "redundancy_condition": "min_lcc ~ dependency_concentration_cond + mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type)",
    }
    for label, reduced_formula in reduced_specs.items():
        reduced = smf.ols(reduced_formula, data=model_df).fit()
        importance_rows.append({"variable_group": label, "partial_r2_drop": partial_r2(model, reduced)})

    importance_df = pd.DataFrame(importance_rows).sort_values("partial_r2_drop", ascending=False)
    std_beta_df = standardized_numeric_betas(model_df)

    coef_rows = []
    ci = model.conf_int()
    for name, beta in model.params.items():
        coef_rows.append(
            {
                "term": name,
                "coefficient": beta,
                "ci_low": ci.loc[name, 0],
                "ci_high": ci.loc[name, 1],
                "p_value": model.pvalues[name],
            }
        )
    coef_df = pd.DataFrame(coef_rows)

    importance_df.to_csv(output_dir / "resilience_factor_importance.csv", index=False)
    std_beta_df.to_csv(output_dir / "resilience_factor_standardized_betas.csv", index=False)
    coef_df.to_csv(output_dir / "resilience_factor_model_coefficients.csv", index=False)
    write_report(output_dir / "resilience_factor_model_summary.md", model, importance_df, std_beta_df)
    make_plots(model_df, importance_df, figures_dir)

    print(f"Wrote model dataset to {output_dir / 'resilience_model_dataset.csv'}")
    print(f"Wrote importance to {output_dir / 'resilience_factor_importance.csv'}")
    print(f"Wrote coefficients to {output_dir / 'resilience_factor_model_coefficients.csv'}")
    print(f"Wrote summary to {output_dir / 'resilience_factor_model_summary.md'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
