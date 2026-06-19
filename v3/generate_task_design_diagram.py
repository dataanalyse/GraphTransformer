from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, xy, width, height, text, facecolor, edgecolor="#334155", fontsize=12):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0f172a",
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#64748b",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main() -> None:
    out_path = Path("v2/runs/final_figures/task_design_diagram.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        (0.08, 0.37),
        0.24,
        0.26,
        "Input Snapshot\n\n$X[t]$\nnode features for all nodes\nat time $t$",
        facecolor="#dbeafe",
    )

    add_box(
        ax,
        (0.40, 0.42),
        0.18,
        0.16,
        "Prediction\nModel",
        facecolor="#e2e8f0",
        fontsize=13,
    )

    add_box(
        ax,
        (0.68, 0.70),
        0.24,
        0.18,
        "Target Type 1\n\nNode health at $t + K$",
        facecolor="#dcfce7",
    )
    add_box(
        ax,
        (0.68, 0.41),
        0.24,
        0.18,
        "Target Type 2\n\nGraph metric at $t + K$\n(LCC, diameter,\nfragmentation, edge survival)",
        facecolor="#fef3c7",
    )
    add_box(
        ax,
        (0.68, 0.12),
        0.24,
        0.18,
        "Target Type 3\n\nTrajectory over\n$t+1, \\ldots, t+K$",
        facecolor="#fce7f3",
    )

    add_arrow(ax, (0.32, 0.50), (0.40, 0.50))
    add_arrow(ax, (0.58, 0.50), (0.68, 0.79))
    add_arrow(ax, (0.58, 0.50), (0.68, 0.50))
    add_arrow(ax, (0.58, 0.50), (0.68, 0.21))

    ax.text(
        0.5,
        0.95,
        "Task Design Across Prediction Objectives",
        ha="center",
        va="center",
        fontsize=16,
        color="#0f172a",
        fontweight="bold",
    )

    ax.text(
        0.5,
        0.05,
        "Same input snapshot, different future targets",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
