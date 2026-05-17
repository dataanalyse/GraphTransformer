import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    run_root = str(script_dir / "runs")
    graph_tag = "N40_tiered_scale_free"
    graph_type = "tiered_scale_free"

    jobs = [
        (
            "gcn",
            "gcn_graph_level_finalpilot_lcc_n40",
            "train_gcn_graph_level.py",
            [
                "--lr", "0.01",
                "--epochs", "300",
                "--eval_every", "5",
                "--hidden_dim", "16",
            ],
        ),
        (
            "graph_transformer",
            "graph_transformer_graph_level_finalpilot_lcc_n40",
            "train_graph_transformer_graph_level.py",
            [
                "--lr", "0.005",
                "--epochs", "300",
                "--eval_every", "5",
                "--d_model", "32",
                "--num_heads", "4",
                "--num_layers", "2",
                "--ff_dim", "64",
                "--dropout", "0.1",
            ],
        ),
        (
            "graphormer",
            "graphormer_graph_level_finalpilot_lcc_n40",
            "train_graphormer_graph_level.py",
            [
                "--lr", "0.005",
                "--epochs", "300",
                "--eval_every", "5",
                "--d_model", "32",
                "--num_heads", "4",
                "--num_layers", "2",
                "--ff_dim", "64",
                "--dropout", "0.1",
                "--max_dist", "4",
                "--max_degree", "16",
            ],
        ),
    ]

    for _, experiment_name, script_name, extra_args in jobs:
        for seed in [1, 2, 3]:
            data_dir = script_dir / "data" / graph_tag / f"seed_{seed}"
            cmd = [
                python,
                str(script_dir / script_name),
                "--seed", str(seed),
                "--graph_type", graph_type,
                "--graph_tag", graph_tag,
                "--data_dir", str(data_dir),
                "--graph_target_key", "lcc_fraction",
                "--run_root", run_root,
                "--experiment_name", experiment_name,
                *extra_args,
            ]
            run_cmd(cmd)

    run_cmd([python, str(script_dir / "generate_final_figures_pilot.py")])


if __name__ == "__main__":
    main()
