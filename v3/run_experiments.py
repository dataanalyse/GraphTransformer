import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = script_dir / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    python = sys.executable
    graph_type = cfg["graph"]["type"]
    sizes = cfg["graph"]["sizes"]
    seeds = cfg["seeds"]
    sim = cfg["simulator"]
    prediction_horizon = cfg.get("prediction_horizon", 1)
    features = cfg.get("features", {})
    train = cfg["training"]
    data_root = script_dir / cfg.get("data_root", "data")
    run_root = str(script_dir / cfg.get("run_root", "runs"))

    for num_nodes in sizes:
        graph_tag = f"N{num_nodes}_{graph_type}"
        for seed in seeds:
            data_dir = data_root / graph_tag / f"seed_{seed}"

            run_cmd(
                    [
                        python,
                        str(script_dir / "simulate_and_build.py"),
                        "--num_nodes",
                        str(num_nodes),
                    "--graph_type",
                    graph_type,
                    "--graph_tag",
                    graph_tag,
                    "--T",
                    str(sim["T"]),
                    "--seed",
                    str(seed),
                    "--p_shock",
                    str(sim["p_shock"]),
                    "--p_propagate",
                    str(sim["p_propagate"]),
                    "--p_recover",
                    str(sim["p_recover"]),
                    "--prediction_horizon",
                    str(prediction_horizon),
                    "--use_health",
                    str(features.get("use_health", True)).lower(),
                    "--use_exposure",
                    str(features.get("use_exposure", True)).lower(),
                    "--use_time_to_recovery",
                    str(features.get("use_time_to_recovery", True)).lower(),
                    "--use_betweenness",
                    str(features.get("use_betweenness", True)).lower(),
                    "--data_dir",
                    str(data_dir),
                ]
            )

            if train["baseline"]["enabled"]:
                b = train["baseline"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_baseline_v1.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(b["lr"]),
                        "--epochs",
                        str(b["epochs"]),
                        "--eval_every",
                        str(b["eval_every"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "baseline",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train["gcn"]["enabled"]:
                g = train["gcn"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_gcn_v2.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(g["lr"]),
                        "--epochs",
                        str(g["epochs"]),
                        "--eval_every",
                        str(g["eval_every"]),
                        "--hidden_dim",
                        str(g["hidden_dim"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "gcn",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("graph_transformer", {}).get("enabled", False):
                gt = train["graph_transformer"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_graph_transformer_v1.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(gt["lr"]),
                        "--epochs",
                        str(gt["epochs"]),
                        "--eval_every",
                        str(gt["eval_every"]),
                        "--d_model",
                        str(gt["d_model"]),
                        "--num_heads",
                        str(gt["num_heads"]),
                        "--num_layers",
                        str(gt["num_layers"]),
                        "--ff_dim",
                        str(gt["ff_dim"]),
                        "--dropout",
                        str(gt["dropout"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "graph_transformer",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("graph_transformer_pyg", {}).get("enabled", False):
                gt = train["graph_transformer_pyg"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_graph_transformer_pyg.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(gt["lr"]),
                        "--epochs",
                        str(gt["epochs"]),
                        "--eval_every",
                        str(gt["eval_every"]),
                        "--hidden_dim",
                        str(gt["hidden_dim"]),
                        "--num_heads",
                        str(gt["num_heads"]),
                        "--num_layers",
                        str(gt["num_layers"]),
                        "--dropout",
                        str(gt["dropout"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "graph_transformer_pyg",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("graphormer", {}).get("enabled", False):
                gt = train["graphormer"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_graphormer_v1.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(gt["lr"]),
                        "--epochs",
                        str(gt["epochs"]),
                        "--eval_every",
                        str(gt["eval_every"]),
                        "--d_model",
                        str(gt["d_model"]),
                        "--num_heads",
                        str(gt["num_heads"]),
                        "--num_layers",
                        str(gt["num_layers"]),
                        "--ff_dim",
                        str(gt["ff_dim"]),
                        "--dropout",
                        str(gt["dropout"]),
                        "--max_dist",
                        str(gt["max_dist"]),
                        "--max_degree",
                        str(gt["max_degree"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "graphormer",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("baseline_graph_level", {}).get("enabled", False):
                b = train["baseline_graph_level"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_baseline_graph_level.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(b["lr"]),
                        "--epochs",
                        str(b["epochs"]),
                        "--eval_every",
                        str(b["eval_every"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "baseline_graph_level",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("graph_transformer_graph_level", {}).get("enabled", False):
                gt = train["graph_transformer_graph_level"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_graph_transformer_graph_level.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(gt["lr"]),
                        "--epochs",
                        str(gt["epochs"]),
                        "--eval_every",
                        str(gt["eval_every"]),
                        "--d_model",
                        str(gt["d_model"]),
                        "--num_heads",
                        str(gt["num_heads"]),
                        "--num_layers",
                        str(gt["num_layers"]),
                        "--ff_dim",
                        str(gt["ff_dim"]),
                        "--dropout",
                        str(gt["dropout"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "graph_transformer_graph_level",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("gcn_graph_level", {}).get("enabled", False):
                g = train["gcn_graph_level"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_gcn_graph_level.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(g["lr"]),
                        "--epochs",
                        str(g["epochs"]),
                        "--eval_every",
                        str(g["eval_every"]),
                        "--hidden_dim",
                        str(g["hidden_dim"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "gcn_graph_level",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

            if train.get("graphormer_graph_level", {}).get("enabled", False):
                gt = train["graphormer_graph_level"]
                run_cmd(
                    [
                        python,
                        str(script_dir / "train_graphormer_graph_level.py"),
                        "--seed",
                        str(seed),
                        "--graph_type",
                        graph_type,
                        "--graph_tag",
                        graph_tag,
                        "--data_dir",
                        str(data_dir),
                        "--lr",
                        str(gt["lr"]),
                        "--epochs",
                        str(gt["epochs"]),
                        "--eval_every",
                        str(gt["eval_every"]),
                        "--d_model",
                        str(gt["d_model"]),
                        "--num_heads",
                        str(gt["num_heads"]),
                        "--num_layers",
                        str(gt["num_layers"]),
                        "--ff_dim",
                        str(gt["ff_dim"]),
                        "--dropout",
                        str(gt["dropout"]),
                        "--max_dist",
                        str(gt["max_dist"]),
                        "--max_degree",
                        str(gt["max_degree"]),
                        "--run_root",
                        run_root,
                        "--experiment_name",
                        "graphormer_graph_level",
                        "--prediction_horizon",
                        str(prediction_horizon),
                    ]
                )

    try:
        run_cmd(
            [
                python,
                str(script_dir / "generate_results_summary.py"),
                "--runs_root",
                run_root,
                "--out_path",
                str(Path(run_root) / "results_summary.csv"),
            ]
        )
    except subprocess.CalledProcessError:
        print("Summary generation failed; training runs still completed.")
    print("All experiments completed.")


if __name__ == "__main__":
    main()
