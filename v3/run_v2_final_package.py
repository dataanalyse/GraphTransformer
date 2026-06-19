import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    config = str(script_dir / "configs" / "experiments_multiseed_final.yaml")

    run_cmd([python, str(script_dir / "run_experiments.py"), "--config", config])
    run_cmd([python, str(script_dir / "run_structure_target_experiments.py"), "--config", config])
    run_cmd([python, str(script_dir / "run_lcc_trajectory_experiments.py"), "--config", config])

    for target in [
        "lcc_fraction",
        "component_fraction",
        "diameter_fraction",
        "edge_survival_ratio",
    ]:
        run_cmd(
            [
                python,
                str(script_dir / "run_graph_level_seed_split.py"),
                "--config",
                config,
                "--train_seeds",
                "1,2,3",
                "--test_seeds",
                "4,5",
                "--graph_target_key",
                target,
            ]
        )

    run_cmd(
        [
            python,
            str(script_dir / "generate_final_report_package.py"),
            "--summary_path",
            str(script_dir / "final_runs" / "results_summary.csv"),
            "--data_root",
            str(script_dir / "data"),
            "--out_dir",
            str(script_dir / "runs" / "final_figures"),
        ]
    )


if __name__ == "__main__":
    main()
