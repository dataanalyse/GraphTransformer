import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


GRAPH_TARGET_FILES = {
    "component_fraction": "Y_components_v1.pt",
    "diameter_fraction": "Y_diameter_v1.pt",
    "edge_survival_ratio": "Y_edge_survival_v1.pt",
}
EXPERIMENT_SUFFIX = {
    "component_fraction": "components",
    "diameter_fraction": "diameter",
    "edge_survival_ratio": "edge_survival",
}


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, check=True, env=env)


def run_graph_level_model(
    python: str,
    script_dir: Path,
    script_name: str,
    experiment_name: str,
    common_args: list[str],
    extra_args: list[str],
    run_root: str,
) -> None:
    run_cmd(
        [
            python,
            str(script_dir / script_name),
            *common_args,
            *extra_args,
            "--run_root",
            run_root,
            "--experiment_name",
            experiment_name,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments_multiseed.yaml")
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

    targets = ["component_fraction", "diameter_fraction", "edge_survival_ratio"]

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

            common = [
                "--seed",
                str(seed),
                "--graph_type",
                graph_type,
                "--graph_tag",
                graph_tag,
                "--data_dir",
                str(data_dir),
                "--prediction_horizon",
                str(prediction_horizon),
            ]

            for target_key in targets:
                suffix = EXPERIMENT_SUFFIX[target_key]
                y_name = GRAPH_TARGET_FILES[target_key]
                target_args = [
                    "--y_name",
                    y_name,
                    "--graph_target_key",
                    target_key,
                ]

                if train.get("baseline_graph_level", {}).get("enabled", False):
                    b = train["baseline_graph_level"]
                    run_graph_level_model(
                        python,
                        script_dir,
                        "train_baseline_graph_level.py",
                        f"baseline_graph_level_{suffix}",
                        common,
                        target_args
                        + [
                            "--lr",
                            str(b["lr"]),
                            "--epochs",
                            str(b["epochs"]),
                            "--eval_every",
                            str(b["eval_every"]),
                        ],
                        run_root,
                    )

                if train.get("gcn_graph_level", {}).get("enabled", False):
                    g = train["gcn_graph_level"]
                    run_graph_level_model(
                        python,
                        script_dir,
                        "train_gcn_graph_level.py",
                        f"gcn_graph_level_{suffix}",
                        common,
                        target_args
                        + [
                            "--lr",
                            str(g["lr"]),
                            "--epochs",
                            str(g["epochs"]),
                            "--eval_every",
                            str(g["eval_every"]),
                            "--hidden_dim",
                            str(g["hidden_dim"]),
                        ],
                        run_root,
                    )

                if train.get("graph_transformer_graph_level", {}).get("enabled", False):
                    gt = train["graph_transformer_graph_level"]
                    run_graph_level_model(
                        python,
                        script_dir,
                        "train_graph_transformer_graph_level.py",
                        f"graph_transformer_graph_level_{suffix}",
                        common,
                        target_args
                        + [
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
                        ],
                        run_root,
                    )

                if train.get("graphormer_graph_level", {}).get("enabled", False):
                    go = train["graphormer_graph_level"]
                    run_graph_level_model(
                        python,
                        script_dir,
                        "train_graphormer_graph_level.py",
                        f"graphormer_graph_level_{suffix}",
                        common,
                        target_args
                        + [
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
                        ],
                        run_root,
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
