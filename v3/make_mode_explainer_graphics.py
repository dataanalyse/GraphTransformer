from __future__ import annotations

import os
from pathlib import Path


def ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


OUTPUT_DIR = Path("v3/runs/figures/explainers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PLANT_POS = {
    "Plant A": (0.15, 0.80),
    "Plant B": (0.15, 0.50),
    "Plant C": (0.15, 0.20),
}
HOSPITAL_POS = (0.78, 0.50)


def draw_node(ax, xy, label, facecolor, edgecolor="#2c3e50", size=2000):
    ax.scatter([xy[0]], [xy[1]], s=size, c=facecolor, edgecolors=edgecolor, linewidths=2, zorder=3)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=11, weight="bold", zorder=4)


def draw_edge(ax, start, end, label, color, linestyle="-", linewidth=2.5, alpha=1.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=linewidth, linestyle=linestyle, alpha=alpha),
        zorder=2,
    )
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    ax.text(mid_x, mid_y + 0.05, label, color=color, fontsize=10, ha="center", va="center")


def make_network_figure():
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    modes = [
        {
            "title": "1. Baseline Additive",
            "subtitle": "Two power supports.\nImpacts from failed supports add together.",
            "edges": [("Plant A", "w=1.0"), ("Plant B", "w=1.0")],
            "edge_color": "#c0392b",
            "style": "-",
        },
        {
            "title": "2. Dampened Power Additive",
            "subtitle": "Same graph as baseline.\nPower impacts are weaker, but still additive.",
            "edges": [("Plant A", "w=0.8"), ("Plant B", "w=0.8")],
            "edge_color": "#2980b9",
            "style": "-",
        },
        {
            "title": "3. Redundant Additive",
            "subtitle": "Three supports added.\nBut failures still add as extra exposure.",
            "edges": [("Plant A", "w=1.0"), ("Plant B", "w=1.0"), ("Plant C", "w=1.0")],
            "edge_color": "#8e44ad",
            "style": "-",
        },
        {
            "title": "4. Redundancy Buffer",
            "subtitle": "Three supports treated as a support set.\nOne surviving support can buffer damage.",
            "edges": [("Plant A", "power set"), ("Plant B", "power set"), ("Plant C", "power set")],
            "edge_color": "#16a085",
            "style": "-",
        },
    ]

    for ax, mode in zip(axes, modes):
        for plant, pos in PLANT_POS.items():
            draw_node(ax, pos, plant.replace(" ", "\n"), "#f7dc6f")
        draw_node(ax, HOSPITAL_POS, "Hospital", "#f5b7b1", size=2600)

        for plant, label in mode["edges"]:
            draw_edge(ax, PLANT_POS[plant], HOSPITAL_POS, label, mode["edge_color"], linestyle=mode["style"])

        if mode["title"] == "4. Redundancy Buffer":
            ax.text(
                0.42,
                0.90,
                "Group support by class:\nPower A + B + C\nEvaluate fraction lost",
                ha="center",
                va="top",
                fontsize=10,
                color=mode["edge_color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor=mode["edge_color"]),
            )

        ax.set_title(f"{mode['title']}\n{mode['subtitle']}", fontsize=13, weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.suptitle("Toy Infrastructure Example: Same Hospital, Different Dependency Logic", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "toy_mode_networks.png", dpi=220)
    plt.close(fig)


def make_response_figure():
    plt = ensure_matplotlib()

    timesteps = [0, 1, 2, 3]
    shock_story = {
        "Plant A": [1, 0, 0, 0],
        "Plant B": [1, 1, 0, 0],
        "Plant C": [1, 1, 1, 0],
    }
    hospital_health = {
        "baseline_additive": [1.0, 0.5, 0.0, 0.5],
        "dampened_power_additive": [1.0, 0.75, 0.25, 0.6],
        "redundant_additive": [1.0, 0.5, 0.0, 0.0],
        "redundancy_buffer": [1.0, 1.0, 0.5, 0.0],
    }
    mode_labels = {
        "baseline_additive": "Baseline Additive",
        "dampened_power_additive": "Dampened Power Additive",
        "redundant_additive": "Redundant Additive",
        "redundancy_buffer": "Redundancy Buffer",
    }
    colors = {
        "baseline_additive": "#c0392b",
        "dampened_power_additive": "#2980b9",
        "redundant_additive": "#8e44ad",
        "redundancy_buffer": "#16a085",
    }

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for plant, vals in shock_story.items():
        ax0.step(timesteps, vals, where="post", linewidth=2.5, label=plant)
    ax0.set_ylim(-0.05, 1.1)
    ax0.set_ylabel("Plant Health")
    ax0.set_title(
        "Shared Toy Shock Sequence\n"
        "t=1: Plant A fails, t=2: Plant B fails, t=3: Plant C fails",
        fontsize=14,
        weight="bold",
    )
    ax0.legend(ncol=3)
    ax0.grid(alpha=0.3)

    for key, vals in hospital_health.items():
        ax1.step(timesteps, vals, where="post", linewidth=3, label=mode_labels[key], color=colors[key])
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Hospital Health")
    ax1.set_title("Hospital Response Under the Four Modes", fontsize=14, weight="bold")
    ax1.legend(ncol=2)
    ax1.grid(alpha=0.3)

    ax1.text(0.05, 0.08, "1.0 = normal, 0.5 = degraded, 0.0 = failed", transform=ax1.transAxes, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "toy_mode_response_curves.png", dpi=220)
    plt.close(fig)


def make_individual_response_figures():
    plt = ensure_matplotlib()

    timesteps = [0, 1, 2, 3]
    hospital_health = {
        "baseline_additive": [1.0, 0.5, 0.0, 0.5],
        "dampened_power_additive": [1.0, 0.75, 0.25, 0.6],
        "redundant_additive": [1.0, 0.5, 0.0, 0.0],
        "redundancy_buffer": [1.0, 1.0, 0.5, 0.0],
    }
    mode_labels = {
        "baseline_additive": "Baseline Additive",
        "dampened_power_additive": "Dampened Power Additive",
        "redundant_additive": "Redundant Additive",
        "redundancy_buffer": "Redundancy Buffer",
    }
    mode_notes = {
        "baseline_additive": "Two failed supports stack damage immediately.",
        "dampened_power_additive": "Same additive logic, but each power hit is weaker.",
        "redundant_additive": "Extra support link still acts like extra exposure.",
        "redundancy_buffer": "Remaining support buffers damage until enough capacity is lost.",
    }
    colors = {
        "baseline_additive": "#c0392b",
        "dampened_power_additive": "#2980b9",
        "redundant_additive": "#8e44ad",
        "redundancy_buffer": "#16a085",
    }

    for key, vals in hospital_health.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.step(timesteps, vals, where="post", linewidth=4, color=colors[key])
        ax.scatter(timesteps, vals, s=80, color=colors[key], zorder=3)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(-0.1, 3.1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Hospital Health")
        ax.set_title(mode_labels[key], fontsize=16, weight="bold")
        ax.grid(alpha=0.3)
        ax.text(
            0.03,
            0.92,
            "Shock story: A fails at t=1, B at t=2, C at t=3",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
        )
        ax.text(
            0.03,
            0.84,
            mode_notes[key],
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            color=colors[key],
        )
        ax.text(
            0.03,
            0.08,
            "1.0 = normal, 0.5 = degraded, 0.0 = failed",
            transform=ax.transAxes,
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{key}_hospital_response.png", dpi=220)
        plt.close(fig)


def make_notes():
    notes = """# Toy Mode Explainer

## Slide-Friendly Interpretation

- `Baseline Additive`: a hospital with two power supports gets hit by the sum of failed-support impacts.
- `Dampened Power Additive`: same graph as baseline, but each power hit is weaker.
- `Redundant Additive`: three power supports are added, but the simulator still treats them as extra exposure paths.
- `Redundancy Buffer`: the same three power supports are treated as a support set, so one remaining support can buffer the hospital.

## Toy Shock Story

- `t=1`: Plant A fails
- `t=2`: Plant B fails
- `t=3`: Plant C fails

## Key Message

- The graph can look similar while the cascade logic is very different.
- `Redundant Additive` models "more links, more ways to get hurt."
- `Redundancy Buffer` models "more links, more fallback capacity."
"""
    (OUTPUT_DIR / "toy_mode_explainer.md").write_text(notes, encoding="utf-8")


def main():
    make_network_figure()
    make_response_figure()
    make_individual_response_figures()
    make_notes()
    print(f"Wrote explainer assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
