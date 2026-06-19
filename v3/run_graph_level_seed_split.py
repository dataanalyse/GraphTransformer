import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


TARGET_FILES = {
    "lcc_fraction": "Y_lcc_v1.pt",
    "component_fraction": "Y_components_v1.pt",
    "diameter_fraction": "Y_diameter_v1.pt",
    "edge_survival_ratio": "Y_edge_survival_v1.pt",
}


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments_multiseed.yaml")
    parser.add_argument("--train_seeds", type=str, default="1,2,3")
    parser.add_argument("--test_seeds", type=str, default="4,5")
    parser.add_argument("--graph_target_key", type=str, default="lcc_fraction")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = script_dir / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    python = sys.executable
    graph_type = cfg["graph"]["type"]
    sizes = cfg["graph"]["sizes"]
    sim = cfg["simulator"]
    prediction_horizon = cfg.get("prediction_horizon", 1)
    features = cfg.get("features", {})
    train_cfg = cfg["training"]
    data_root = script_dir / cfg.get("data_root", "data")
    run_root = str(script_dir / cfg.get("run_root", "runs"))

    all_seed_values = sorted(
        {int(seed) for seed in args.train_seeds.split(",") if seed}
        | {int(seed) for seed in args.test_seeds.split(",") if seed}
    )

    for num_nodes in sizes:
        graph_tag = f"N{num_nodes}_{graph_type}"
        for seed in all_seed_values:
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

        common_args = [
            "--graph_type",
            graph_type,
            "--graph_tag",
            graph_tag,
            "--data_root",
            str(data_root / graph_tag),
            "--train_seeds",
            args.train_seeds,
            "--test_seeds",
            args.test_seeds,
            "--graph_target_key",
            args.graph_target_key,
            "--prediction_horizon",
            str(prediction_horizon),
            "--run_root",
            run_root,
        ]

        if train_cfg.get("baseline_graph_level", {}).get("enabled", False):
            b = train_cfg["baseline_graph_level"]
            run_cmd(
                [
                    python,
                    str(script_dir / "train_graph_level_seed_split.py"),
                    "--model_type",
                    "baseline",
                    "--experiment_name",
                    "baseline_graph_level_seed_split",
                    "--lr",
                    str(b["lr"]),
                    "--epochs",
                    str(b["epochs"]),
                    "--eval_every",
                    str(b["eval_every"]),
                    *common_args,
                ]
            )

        if train_cfg.get("gcn_graph_level", {}).get("enabled", False):
            g = train_cfg["gcn_graph_level"]
            run_cmd(
                [
                    python,
                    str(script_dir / "train_graph_level_seed_split.py"),
                    "--model_type",
                    "gcn",
                    "--experiment_name",
                    "gcn_graph_level_seed_split",
                    "--lr",
                    str(g["lr"]),
                    "--epochs",
                    str(g["epochs"]),
                    "--eval_every",
                    str(g["eval_every"]),
                    "--hidden_dim",
                    str(g["hidden_dim"]),
                    *common_args,
                ]
            )

        if train_cfg.get("graph_transformer_graph_level", {}).get("enabled", False):
            gt = train_cfg["graph_transformer_graph_level"]
            run_cmd(
                [
                    python,
                    str(script_dir / "train_graph_level_seed_split.py"),
                    "--model_type",
                    "graph_transformer",
                    "--experiment_name",
                    "graph_transformer_graph_level_seed_split",
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
                    *common_args,
                ]
            )

        if train_cfg.get("graphormer_graph_level", {}).get("enabled", False):
            go = train_cfg["graphormer_graph_level"]
            run_cmd(
                [
                    python,
                    str(script_dir / "train_graph_level_seed_split.py"),
                    "--model_type",
                    "graphormer",
                    "--experiment_name",
                    "graphormer_graph_level_seed_split",
                    "--lr",
                    str(go["lr"]),
                    "--epochs",
                    str(go["epochs"]),
                    "--eval_every",
                    str(go["eval_every"]),
                    "--d_model",
                    str(go["d_model"]),
                    "--num_heads",
                    str(go["num_heads"]),
                    "--num_layers",
                    str(go["num_layers"]),
                    "--ff_dim",
                    str(go["ff_dim"]),
                    "--dropout",
                    str(go["dropout"]),
                    "--max_dist",
                    str(go["max_dist"]),
                    "--max_degree",
                    str(go["max_degree"]),
                    *common_args,
                ]
            )

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


if __name__ == "__main__":
    main()
