from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


SCENARIOS = ["power", "telecom", "ems"]
COLORS = {"power": "#c0392b", "telecom": "#2980b9", "ems": "#16a085"}
LABELS = {"power": "Power Shock", "telecom": "Telecom Shock", "ems": "EMS Shock"}
CONSUMER_SECTORS = ["hospital", "school"]


def load_metrics(metrics_dir: Path, suffix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        filename = f"cascade_metrics_{scenario}{suffix}.csv" if suffix else f"cascade_metrics_{scenario}.csv"
        out[scenario] = pd.read_csv(metrics_dir / filename)
    return out


def plot_consumer_health(metrics: dict[str, pd.DataFrame], output: Path, title_prefix: str = "") -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, sector in zip(axes, CONSUMER_SECTORS):
        column = f"sector_health_{sector}"
        for scenario, df in metrics.items():
            ax.plot(
                df["timestep"],
                df[column],
                label=LABELS[scenario],
                color=COLORS[scenario],
                linewidth=2,
            )
        ax.set_title(f"{title_prefix}{sector.capitalize()} Average Health")
        ax.set_ylabel("Average Health")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    axes[0].legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot consumer-sector health under infrastructure shock scenarios.")
    parser.add_argument("--metrics-dir", default="v3/data/processed")
    parser.add_argument("--suffix", default="_v2", help="Metrics suffix such as '_v2' or empty for v1.")
    parser.add_argument("--output", default="v3/runs/figures/v2_consumer_sector_health_comparison.png")
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_dir), args.suffix)
    plot_consumer_health(metrics, Path(args.output), args.title_prefix)
    print(f"Wrote consumer sector health figure to {args.output}")


if __name__ == "__main__":
    main()
