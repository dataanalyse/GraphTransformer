import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    python = sys.executable
    graph_type = cfg["graph"]["type"]
    sizes = cfg["graph"]["sizes"]
    seeds = cfg["seeds"]
    sim = cfg["simulator"]
    train = cfg["training"]
    data_root = Path(cfg.get("data_root", "data"))
    run_root = cfg.get("run_root", "runs")

    for num_nodes in sizes:
        graph_tag = f"N{num_nodes}_{graph_type}"
        for seed in seeds:
            data_dir = data_root / graph_tag / f"seed_{seed}"

            run_cmd(
                [
                    python,
                    "simulate_and_build.py",
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
                    "--data_dir",
                    str(data_dir),
                ]
            )

            if train["baseline"]["enabled"]:
                b = train["baseline"]
                run_cmd(
                    [
                        python,
                        "train_baseline_v1.py",
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
                    ]
                )

            if train["gcn"]["enabled"]:
                g = train["gcn"]
                run_cmd(
                    [
                        python,
                        "train_gcn_v2.py",
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
                    ]
                )

            if train.get("graph_transformer", {}).get("enabled", False):
                gt = train["graph_transformer"]
                run_cmd(
                    [
                        python,
                        "train_graph_transformer_v1.py",
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
                    ]
                )

    try:
        run_cmd([python, "generate_results_summary.py"])
    except subprocess.CalledProcessError:
        print("Summary generation failed; training runs still completed.")
    print("All experiments completed.")


if __name__ == "__main__":
    main()
