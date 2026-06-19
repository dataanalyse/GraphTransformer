from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_simple_regression(df: pd.DataFrame):
    return smf.ols("min_lcc ~ dependency_concentration", data=df).fit()


def pearson_stats(df: pd.DataFrame) -> tuple[float, float]:
    model = smf.ols("min_lcc ~ dependency_concentration", data=df).fit()
    r = float(np.corrcoef(df["dependency_concentration"], df["min_lcc"])[0, 1])
    return r, float(model.pvalues["dependency_concentration"])


def spearman_stats(df: pd.DataFrame) -> tuple[float, float]:
    ranked = df[["dependency_concentration", "min_lcc"]].rank(method="average")
    model = smf.ols("min_lcc ~ dependency_concentration", data=ranked).fit()
    r = float(np.corrcoef(ranked["dependency_concentration"], ranked["min_lcc"])[0, 1])
    return r, float(model.pvalues["dependency_concentration"])


def correlation_row(df: pd.DataFrame, label: str) -> dict:
    pearson_r, pearson_p = pearson_stats(df)
    spearman_r, spearman_p = spearman_stats(df)
    model = fit_simple_regression(df)
    ci_low, ci_high = model.conf_int().loc["dependency_concentration"].tolist()
    return {
        "subset": label,
        "n": len(df),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "r_squared": model.rsquared,
        "beta1": model.params["dependency_concentration"],
        "beta1_p": model.pvalues["dependency_concentration"],
        "beta1_ci_low": ci_low,
        "beta1_ci_high": ci_high,
    }


def top_outliers(df: pd.DataFrame, model, top_k: int = 10) -> pd.DataFrame:
    influence = model.get_influence()
    out = df.copy()
    out["fitted"] = model.fittedvalues
    out["resid"] = model.resid
    out["studentized_resid"] = influence.resid_studentized_external
    out["leverage"] = influence.hat_matrix_diag
    out["cooks_d"] = influence.cooks_distance[0]
    out["abs_studentized_resid"] = out["studentized_resid"].abs()
    out = out.sort_values(["cooks_d", "abs_studentized_resid", "leverage"], ascending=False).head(top_k)
    return out[
        [
            "scenario_id",
            "shock_type",
            "dependency_concentration",
            "min_lcc",
            "mean_seed_degree",
            "seed_k",
            "fitted",
            "resid",
            "studentized_resid",
            "leverage",
            "cooks_d",
        ]
    ].copy()


def make_plots(df: pd.DataFrame, outliers: pd.DataFrame, output_dir: Path) -> None:
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

    model = fit_simple_regression(df)
    x_sorted = df["dependency_concentration"].sort_values()
    pred = model.predict(pd.DataFrame({"dependency_concentration": x_sorted}))

    fig, ax = plt.subplots(figsize=(9, 7))
    for shock_type, sub in df.groupby("shock_type"):
        ax.scatter(
            sub["dependency_concentration"],
            sub["min_lcc"],
            label=shock_type,
            color=colors.get(shock_type, "#555555"),
            alpha=0.65,
        )
    ax.plot(x_sorted, pred, color="black", linewidth=2, label="OLS fit")
    if not outliers.empty:
        ax.scatter(
            outliers["dependency_concentration"],
            outliers["min_lcc"],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="Top leverage/outliers",
        )
    ax.set_title("Dependency Concentration vs Minimum LCC")
    ax.set_xlabel("Dependency Concentration")
    ax.set_ylabel("Minimum LCC")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dependency_concentration_vs_min_lcc.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, shock_type in zip(axes.flatten(), ["power", "telecom", "ems", "mixed"]):
        sub = df[df["shock_type"] == shock_type].copy()
        if sub.empty:
            continue
        ax.scatter(sub["dependency_concentration"], sub["min_lcc"], color=colors[shock_type], alpha=0.7)
        if sub["dependency_concentration"].nunique() > 1:
            sub_model = fit_simple_regression(sub)
            x_sub = sub["dependency_concentration"].sort_values()
            pred_sub = sub_model.predict(pd.DataFrame({"dependency_concentration": x_sub}))
            ax.plot(x_sub, pred_sub, color="black", linewidth=1.8)
        ax.set_title(f"{shock_type.title()} Shocks")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Dependency Concentration")
    axes[1, 1].set_xlabel("Dependency Concentration")
    axes[0, 0].set_ylabel("Minimum LCC")
    axes[1, 0].set_ylabel("Minimum LCC")
    fig.tight_layout()
    fig.savefig(output_dir / "dependency_concentration_vs_min_lcc_by_shock.png", dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    overall: dict,
    cleaned: dict,
    shock_rows: list[dict],
    controlled_model,
    outliers: pd.DataFrame,
) -> None:
    ci_low, ci_high = controlled_model.conf_int().loc["dependency_concentration"].tolist()
    lines = [
        "# Dependency Concentration Validation",
        "",
        "## Overall Relationship",
        "",
        f"- N: `{overall['n']}`",
        f"- Pearson r: `{overall['pearson_r']:.3f}` (p=`{overall['pearson_p']:.3g}`)",
        f"- Spearman r: `{overall['spearman_r']:.3f}` (p=`{overall['spearman_p']:.3g}`)",
        f"- Simple-regression R-squared: `{overall['r_squared']:.3f}`",
        f"- Simple-regression coefficient: `{overall['beta1']:.3f}`",
        f"- Simple-regression 95% CI: `[{overall['beta1_ci_low']:.3f}, {overall['beta1_ci_high']:.3f}]`",
        f"- Simple-regression significance: p=`{overall['beta1_p']:.3g}`",
        "",
        "## After Removing Top 10 Leverage/Outlier Points",
        "",
        f"- N: `{cleaned['n']}`",
        f"- Pearson r: `{cleaned['pearson_r']:.3f}` (p=`{cleaned['pearson_p']:.3g}`)",
        f"- Spearman r: `{cleaned['spearman_r']:.3f}` (p=`{cleaned['spearman_p']:.3g}`)",
        f"- R-squared: `{cleaned['r_squared']:.3f}`",
        f"- Coefficient: `{cleaned['beta1']:.3f}`",
        f"- 95% CI: `[{cleaned['beta1_ci_low']:.3f}, {cleaned['beta1_ci_high']:.3f}]`",
        f"- Significance: p=`{cleaned['beta1_p']:.3g}`",
        "",
        "## Top 10 Leverage/Outlier Scenarios",
        "",
    ]
    for row in outliers.itertuples(index=False):
        lines.append(
            f"- `{row.scenario_id}` | `{row.shock_type}` | dep conc `{row.dependency_concentration:.3f}` | "
            f"min LCC `{row.min_lcc:.3f}` | Cook's D `{row.cooks_d:.4f}` | leverage `{row.leverage:.4f}`"
        )

    lines.extend(["", "## By Shock Type", ""])
    for row in shock_rows:
        lines.append(
            f"- `{row['subset']}`: Pearson `{row['pearson_r']:.3f}` (p=`{row['pearson_p']:.3g}`), "
            f"Spearman `{row['spearman_r']:.3f}` (p=`{row['spearman_p']:.3g}`), "
            f"R-squared `{row['r_squared']:.3f}`, beta `{row['beta1']:.3f}`"
        )

    lines.extend(
        [
            "",
            "## Regression With Shock-Type Controls",
            "",
            "Model:",
            "`minimum_LCC = beta0 + beta1 * dependency_concentration + shock_type indicators`",
            "",
            f"- Controlled coefficient on dependency concentration: `{controlled_model.params['dependency_concentration']:.3f}`",
            f"- 95% CI: `[{ci_low:.3f}, {ci_high:.3f}]`",
            f"- p-value: `{controlled_model.pvalues['dependency_concentration']:.3g}`",
            f"- Model R-squared: `{controlled_model.rsquared:.3f}`",
            "",
            "## Answer",
            "",
        ]
    )

    if controlled_model.pvalues["dependency_concentration"] < 0.05:
        lines.append(
            "- Dependency concentration remains a statistically significant predictor of minimum LCC even after controlling for shock type."
        )
    else:
        lines.append(
            "- After controlling for shock type, dependency concentration no longer looks like a statistically reliable predictor of minimum LCC."
        )

    lines.append(
        "- Shock type still matters strongly, so the concentration result should be presented as structural but shock-conditional rather than as a universal standalone law."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dependency concentration as a predictor of minimum LCC.")
    parser.add_argument("--input", default="v3/data/processed/baseline_model_v1/scenario_summary_metrics.csv")
    parser.add_argument("--output-dir", default="v3/data/processed/dependency_concentration_validation")
    parser.add_argument("--figures-dir", default="v3/runs/figures/dependency_concentration_validation")
    args = parser.parse_args()

    df = pd.read_csv(args.input).copy()
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall = correlation_row(df, "overall")
    simple_model = fit_simple_regression(df)
    outliers = top_outliers(df, simple_model, top_k=10)
    cleaned_df = df[~df["scenario_id"].isin(outliers["scenario_id"])].copy()
    cleaned = correlation_row(cleaned_df, "overall_without_top10")

    shock_rows = []
    for shock_type in ["power", "telecom", "ems", "mixed"]:
        sub = df[df["shock_type"] == shock_type].copy()
        if len(sub) >= 3 and sub["dependency_concentration"].nunique() > 1:
            shock_rows.append(correlation_row(sub, shock_type))

    controlled_model = smf.ols(
        "min_lcc ~ dependency_concentration + C(shock_type)",
        data=df[df["shock_type"].isin(["power", "telecom", "ems", "mixed", "hospital"])],
    ).fit()

    stats_df = pd.DataFrame([overall, cleaned] + shock_rows)
    stats_df.to_csv(output_dir / "dependency_concentration_stats.csv", index=False)
    outliers.to_csv(output_dir / "dependency_concentration_outliers.csv", index=False)
    write_report(
        output_dir / "dependency_concentration_validation_summary.md",
        overall,
        cleaned,
        shock_rows,
        controlled_model,
        outliers,
    )
    make_plots(df, outliers, figures_dir)

    print(f"Wrote stats to {output_dir / 'dependency_concentration_stats.csv'}")
    print(f"Wrote outliers to {output_dir / 'dependency_concentration_outliers.csv'}")
    print(f"Wrote report to {output_dir / 'dependency_concentration_validation_summary.md'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
