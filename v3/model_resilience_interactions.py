from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_models(df: pd.DataFrame):
    base_controls = "mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type) + C(condition)"
    redundancy_model = smf.ols(
        f"min_lcc ~ dependency_concentration_cond * C(condition) + mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type)",
        data=df,
    ).fit()
    shock_model = smf.ols(
        f"min_lcc ~ dependency_concentration_cond * C(shock_type) + mean_seed_degree_cond + mean_propagation_delay_cond + C(condition)",
        data=df,
    ).fit()
    degree_model = smf.ols(
        f"min_lcc ~ dependency_concentration_cond * mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type) + C(condition)",
        data=df,
    ).fit()
    threshold_model = smf.ols(
        f"min_lcc ~ dependency_concentration_cond + I(dependency_concentration_cond ** 2) + {base_controls}",
        data=df,
    ).fit()
    return redundancy_model, shock_model, degree_model, threshold_model


def interaction_slopes(model, base_label: str, interaction_prefix: str, levels: list[str]) -> pd.DataFrame:
    rows = []
    base_beta = model.params[base_label]
    base_var = model.cov_params().loc[base_label, base_label]
    rows.append({"level": levels[0], "slope": base_beta, "se": np.sqrt(base_var), "p_value": model.pvalues[base_label]})
    for level in levels[1:]:
        term = f"{interaction_prefix}[T.{level}]"
        if term not in model.params.index:
            continue
        slope = base_beta + model.params[term]
        var = (
            model.cov_params().loc[base_label, base_label]
            + model.cov_params().loc[term, term]
            + 2 * model.cov_params().loc[base_label, term]
        )
        rows.append({"level": level, "slope": slope, "se": np.sqrt(max(var, 0.0)), "p_value": model.pvalues[term]})
    return pd.DataFrame(rows)


def make_prediction_grid(df: pd.DataFrame, dep_grid: np.ndarray) -> dict[str, pd.DataFrame]:
    common = {
        "mean_seed_degree_cond": df["mean_seed_degree_cond"].median(),
        "mean_propagation_delay_cond": df["mean_propagation_delay_cond"].median(),
        "shock_type": "power",
        "condition": "baseline_additive",
    }
    q25 = float(df["mean_seed_degree_cond"].quantile(0.25))
    q75 = float(df["mean_seed_degree_cond"].quantile(0.75))

    condition_grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "dependency_concentration_cond": dep_grid,
                    "condition": cond,
                    "shock_type": common["shock_type"],
                    "mean_seed_degree_cond": common["mean_seed_degree_cond"],
                    "mean_propagation_delay_cond": common["mean_propagation_delay_cond"],
                }
            )
            for cond in ["baseline_additive", "redundant_additive", "redundant_buffer", "dampened_power_additive"]
        ],
        ignore_index=True,
    )

    shock_grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "dependency_concentration_cond": dep_grid,
                    "shock_type": shock,
                    "condition": "baseline_additive",
                    "mean_seed_degree_cond": common["mean_seed_degree_cond"],
                    "mean_propagation_delay_cond": common["mean_propagation_delay_cond"],
                }
            )
            for shock in ["power", "telecom", "ems", "mixed"]
        ],
        ignore_index=True,
    )

    degree_grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "dependency_concentration_cond": dep_grid,
                    "seed_degree_band": band,
                    "condition": "baseline_additive",
                    "shock_type": "power",
                    "mean_seed_degree_cond": seed_degree,
                    "mean_propagation_delay_cond": common["mean_propagation_delay_cond"],
                }
            )
            for band, seed_degree in [("low", q25), ("high", q75)]
        ],
        ignore_index=True,
    )
    return {"condition": condition_grid, "shock": shock_grid, "degree": degree_grid}


def make_plots(df: pd.DataFrame, redundancy_model, shock_model, degree_model, threshold_model, figures_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dep_grid = np.linspace(df["dependency_concentration_cond"].min(), df["dependency_concentration_cond"].max(), 120)
    grids = make_prediction_grid(df, dep_grid)

    cond_pred = grids["condition"].copy()
    cond_pred["predicted_min_lcc"] = redundancy_model.predict(cond_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {
        "baseline_additive": "#c0392b",
        "redundant_additive": "#8e44ad",
        "redundant_buffer": "#16a085",
        "dampened_power_additive": "#2980b9",
    }
    for cond, sub in cond_pred.groupby("condition"):
        ax.plot(sub["dependency_concentration_cond"], sub["predicted_min_lcc"], label=cond, color=colors[cond], linewidth=2)
    ax.set_title("Partial Dependence: Concentration x Redundancy Condition")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Predicted Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "partial_dependence_condition.png", dpi=200)
    plt.close(fig)

    shock_pred = grids["shock"].copy()
    shock_pred["predicted_min_lcc"] = shock_model.predict(shock_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    shock_colors = {"power": "#c0392b", "telecom": "#2980b9", "ems": "#16a085", "mixed": "#d35400"}
    for shock, sub in shock_pred.groupby("shock_type"):
        ax.plot(sub["dependency_concentration_cond"], sub["predicted_min_lcc"], label=shock, color=shock_colors[shock], linewidth=2)
    ax.set_title("Interaction Plot: Concentration x Shock Type")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Predicted Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "interaction_concentration_shock_type.png", dpi=200)
    plt.close(fig)

    degree_pred = grids["degree"].copy()
    degree_pred["predicted_min_lcc"] = degree_model.predict(degree_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    band_colors = {"low": "#2980b9", "high": "#c0392b"}
    for band, sub in degree_pred.groupby("seed_degree_band"):
        ax.plot(sub["dependency_concentration_cond"], sub["predicted_min_lcc"], label=f"{band} seed degree", color=band_colors[band], linewidth=2)
    ax.set_title("Interaction Plot: Concentration x Seed Degree")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Predicted Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "interaction_concentration_seed_degree.png", dpi=200)
    plt.close(fig)

    quad_pred = pd.DataFrame(
        {
            "dependency_concentration_cond": dep_grid,
            "mean_seed_degree_cond": df["mean_seed_degree_cond"].median(),
            "mean_propagation_delay_cond": df["mean_propagation_delay_cond"].median(),
            "shock_type": "power",
            "condition": "baseline_additive",
        }
    )
    quad_pred["predicted_min_lcc"] = threshold_model.predict(quad_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        df["dependency_concentration_cond"],
        df["min_lcc"],
        s=12,
        alpha=0.2,
        color="#7f8c8d",
        label="scenarios",
    )
    ax.plot(dep_grid, quad_pred["predicted_min_lcc"], color="black", linewidth=2.5, label="quadratic fit")
    ax.set_title("Threshold Check: Concentration vs Predicted Minimum LCC")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Predicted Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_effect_concentration.png", dpi=200)
    plt.close(fig)


def write_summary(
    output_path: Path,
    redundancy_model,
    shock_model,
    degree_model,
    threshold_model,
    redundancy_slopes: pd.DataFrame,
    shock_slopes: pd.DataFrame,
    degree_interaction_p: float,
) -> None:
    quad_term = "I(dependency_concentration_cond ** 2)"
    lines = [
        "# Resilience Interaction Models",
        "",
        "## Models Tested",
        "",
        "- `dependency_concentration × redundancy_condition`",
        "- `dependency_concentration × shock_type`",
        "- `dependency_concentration × seed_degree`",
        "- quadratic concentration term for threshold effects",
        "",
        "## Question 1: Does redundancy weaken the effect of dependency concentration?",
        "",
    ]
    for row in redundancy_slopes.itertuples(index=False):
        lines.append(f"- `{row.level}` slope: `{row.slope:.3f}`")

    buffer_slope = float(redundancy_slopes[redundancy_slopes["level"] == "redundant_buffer"]["slope"].iloc[0])
    baseline_slope = float(redundancy_slopes[redundancy_slopes["level"] == "baseline_additive"]["slope"].iloc[0])
    if abs(buffer_slope) < abs(baseline_slope):
        lines.append("- Yes. The concentration slope is weaker under `redundant_buffer` than under `baseline_additive`, which is consistent with redundancy dampening fragility.")
    else:
        lines.append("- No clear weakening is visible; redundancy does not appear to reduce the concentration slope in this model.")

    lines.extend(["", "## Question 2: Is concentration more dangerous under power shocks than EMS shocks?", ""])
    for row in shock_slopes.itertuples(index=False):
        lines.append(f"- `{row.level}` slope: `{row.slope:.3f}`")
    power_slope = float(shock_slopes[shock_slopes["level"] == "power"]["slope"].iloc[0])
    ems_slope = float(shock_slopes[shock_slopes["level"] == "ems"]["slope"].iloc[0])
    power_term = "dependency_concentration_cond:C(shock_type)[T.power]"
    power_interaction_p = float(shock_model.pvalues.get(power_term, np.nan))
    if power_interaction_p < 0.05:
        if abs(power_slope) > abs(ems_slope):
            lines.append("- Yes. The fitted concentration slope is steeper for power shocks than for EMS shocks, and the interaction is statistically supported.")
        else:
            lines.append("- No. EMS shocks appear at least as concentration-sensitive as power shocks, and the difference is statistically supported.")
    else:
        lines.append("- Numerically, the power slope is somewhat steeper than the EMS slope, but the shock-type interaction terms are not statistically strong in this model. So this part should be treated as suggestive rather than conclusive.")

    lines.extend(["", "## Question 3: Are there threshold effects?", ""])
    lines.append(
        f"- Quadratic concentration term coefficient: `{threshold_model.params.get(quad_term, float('nan')):.3f}` with p=`{threshold_model.pvalues.get(quad_term, float('nan')):.3g}`"
    )
    if threshold_model.pvalues.get(quad_term, 1.0) < 0.05:
        lines.append("- Yes. The quadratic term is statistically significant, which supports a nonlinear or threshold-like concentration effect.")
    else:
        lines.append("- No strong quadratic threshold signal appears in this global model; any threshold behavior is likely conditional on shock type or redundancy state.")

    lines.extend(["", "## Question 4: Does seed degree modify the concentration effect?", ""])
    interaction_term = "dependency_concentration_cond:mean_seed_degree_cond"
    lines.append(
        f"- Interaction coefficient: `{degree_model.params.get(interaction_term, float('nan')):.6f}` with p=`{degree_interaction_p:.3g}`"
    )
    if degree_interaction_p < 0.05:
        lines.append("- Yes. The marginal effect of concentration changes with seed degree.")
    else:
        lines.append("- No strong interaction appears between concentration and seed degree.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare the partial-dependence and interaction plots for the practical picture.",
            "- If `redundant_buffer` visibly flattens the concentration curve, that is the cleanest evidence that true redundancy weakens structural fragility.",
            "- If the power and EMS curves separate meaningfully, then concentration danger is shock-family specific rather than universal.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit interaction models for resilience outcomes.")
    parser.add_argument("--input", default="v3/data/processed/resilience_factor_model/resilience_model_dataset.csv")
    parser.add_argument("--output-dir", default="v3/data/processed/resilience_interaction_model")
    parser.add_argument("--figures-dir", default="v3/runs/figures/resilience_interaction_model")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    redundancy_model, shock_model, degree_model, threshold_model = fit_models(df)

    redundancy_levels = ["baseline_additive", "redundant_additive", "redundant_buffer", "dampened_power_additive"]
    redundancy_slopes = interaction_slopes(
        redundancy_model,
        "dependency_concentration_cond",
        "dependency_concentration_cond:C(condition)",
        redundancy_levels,
    )

    shock_levels = ["ems", "hospital", "mixed", "power", "telecom"]
    shock_slopes = interaction_slopes(
        shock_model,
        "dependency_concentration_cond",
        "dependency_concentration_cond:C(shock_type)",
        shock_levels,
    )

    interaction_term = "dependency_concentration_cond:mean_seed_degree_cond"
    degree_interaction_p = float(degree_model.pvalues[interaction_term])

    coef_rows = []
    for model_name, model in [
        ("redundancy", redundancy_model),
        ("shock", shock_model),
        ("seed_degree", degree_model),
        ("threshold", threshold_model),
    ]:
        ci = model.conf_int()
        for term, coef in model.params.items():
            coef_rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "coefficient": coef,
                    "ci_low": ci.loc[term, 0],
                    "ci_high": ci.loc[term, 1],
                    "p_value": model.pvalues[term],
                }
            )
    coef_df = pd.DataFrame(coef_rows)

    redundancy_slopes.to_csv(output_dir / "redundancy_condition_slopes.csv", index=False)
    shock_slopes.to_csv(output_dir / "shock_type_slopes.csv", index=False)
    coef_df.to_csv(output_dir / "interaction_model_coefficients.csv", index=False)
    write_summary(
        output_dir / "resilience_interaction_summary.md",
        redundancy_model,
        shock_model,
        degree_model,
        threshold_model,
        redundancy_slopes,
        shock_slopes,
        degree_interaction_p,
    )
    make_plots(df, redundancy_model, shock_model, degree_model, threshold_model, figures_dir)

    print(f"Wrote slopes to {output_dir}")
    print(f"Wrote coefficients to {output_dir / 'interaction_model_coefficients.csv'}")
    print(f"Wrote summary to {output_dir / 'resilience_interaction_summary.md'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
