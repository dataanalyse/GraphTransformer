import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from feature_utils import final_feature_names


FEATURE_VARIANTS = {
    "A": {
        "use_health": True,
        "use_exposure": False,
        "use_time_to_recovery": False,
        "use_betweenness": False,
    },
    "B": {
        "use_health": True,
        "use_exposure": False,
        "use_time_to_recovery": True,
        "use_betweenness": False,
    },
    "C": {
        "use_health": True,
        "use_exposure": True,
        "use_time_to_recovery": True,
        "use_betweenness": False,
    },
    "D": {
        "use_health": True,
        "use_exposure": True,
        "use_time_to_recovery": True,
        "use_betweenness": True,
    },
}


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
    train = cfg["training"]
    data_root = script_dir / cfg.get("data_root", "data")
    run_root = str(script_dir / cfg.get("run_root", "runs"))

    for variant_name, feature_flags in FEATURE_VARIANTS.items():
        active_features = final_feature_names(
            feature_flags,
            include_runtime_betweenness=feature_flags["use_betweenness"],
        )
        active_features_arg = ",".join(active_features)

        for num_nodes in sizes:
            graph_tag = f"N{num_nodes}_{graph_type}"
            for seed in seeds:
                data_dir = data_root / f"variant_{variant_name}" / graph_tag / f"seed_{seed}"

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
                        str(feature_flags["use_health"]).lower(),
                        "--use_exposure",
                        str(feature_flags["use_exposure"]).lower(),
                        "--use_time_to_recovery",
                        str(feature_flags["use_time_to_recovery"]).lower(),
                        "--use_betweenness",
                        str(feature_flags["use_betweenness"]).lower(),
                        "--data_dir",
                        str(data_dir),
                    ]
                )

                common_train_args = [
                    "--seed",
                    str(seed),
                    "--graph_type",
                    graph_type,
                    "--graph_tag",
                    graph_tag,
                    "--data_dir",
                    str(data_dir),
                    "--run_root",
                    run_root,
                    "--feature_variant",
                    variant_name,
                    "--active_features",
                    active_features_arg,
                    "--prediction_horizon",
                    str(prediction_horizon),
                ]

                if train["baseline"]["enabled"]:
                    b = train["baseline"]
                    run_cmd(
                        [
                            python,
                            str(script_dir / "train_baseline_v1.py"),
                            *common_train_args,
                            "--lr",
                            str(b["lr"]),
                            "--epochs",
                            str(b["epochs"]),
                            "--eval_every",
                            str(b["eval_every"]),
                            "--experiment_name",
                            "baseline",
                        ]
                    )

                if train["gcn"]["enabled"]:
                    g = train["gcn"]
                    run_cmd(
                        [
                            python,
                            str(script_dir / "train_gcn_v2.py"),
                            *common_train_args,
                            "--lr",
                            str(g["lr"]),
                            "--epochs",
                            str(g["epochs"]),
                            "--eval_every",
                            str(g["eval_every"]),
                            "--hidden_dim",
                            str(g["hidden_dim"]),
                            "--experiment_name",
                            "gcn",
                        ]
                    )

                if train.get("graph_transformer", {}).get("enabled", False):
                    gt = train["graph_transformer"]
                    run_cmd(
                        [
                            python,
                            str(script_dir / "train_graph_transformer_v1.py"),
                            *common_train_args,
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
                            "--experiment_name",
                            "graph_transformer",
                        ]
                    )

                if train.get("graph_transformer_pyg", {}).get("enabled", False):
                    gt = train["graph_transformer_pyg"]
                    run_cmd(
                        [
                            python,
                            str(script_dir / "train_graph_transformer_pyg.py"),
                            *common_train_args,
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
                            "--experiment_name",
                            "graph_transformer_pyg",
                        ]
                    )

                if train.get("graphormer", {}).get("enabled", False):
                    gt = train["graphormer"]
                    run_cmd(
                        [
                            python,
                            str(script_dir / "train_graphormer_v1.py"),
                            *common_train_args,
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
                            "--experiment_name",
                            "graphormer",
                        ]
                    )

    try:
        run_cmd([python, str(script_dir / "generate_results_summary.py")])
    except subprocess.CalledProcessError:
        print("Summary generation failed; training runs still completed.")
    print("All feature-variant experiments completed.")


if __name__ == "__main__":
    main()
