import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "v2" / "compute_benchmark_runs"
OUT_CSV = REPO_ROOT / "v2" / "runs" / "final_figures" / "compute_benchmark_summary.csv"


def count_parameters(model_state_path: Path) -> int:
    state = torch.load(model_state_path, map_location="cpu")
    return int(sum(t.numel() for t in state.values() if torch.is_tensor(t)))


def read_best_epoch_and_mae(metrics_path: Path) -> tuple[int, float]:
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    best_row = min(rows, key=lambda r: float(r["test_mae"]))
    return int(best_row["epoch"]), float(best_row["test_mae"])


def latest_run_dir(exp_dir: Path) -> Path:
    run_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {exp_dir}")
    return run_dirs[-1]


def run_one(label: str, script: str, experiment_name: str, extra_args: list[str]) -> dict:
    exp_dir = BENCH_ROOT / experiment_name
    before = set(p.name for p in exp_dir.iterdir()) if exp_dir.exists() else set()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "v2" / script),
        "--seed",
        "1",
        "--graph_type",
        "tiered_scale_free",
        "--graph_tag",
        "N40_tiered_scale_free",
        "--data_dir",
        str(REPO_ROOT / "v2" / "data" / "N40_tiered_scale_free" / "seed_1"),
        "--y_name",
        "Y_lcc_v1.pt",
        "--graph_target_key",
        "lcc_fraction",
        "--epochs",
        "300",
        "--eval_every",
        "5",
        "--run_root",
        str(BENCH_ROOT),
        "--experiment_name",
        experiment_name,
    ] + extra_args

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")

    started = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    total_runtime_sec = time.perf_counter() - started

    after = set(p.name for p in exp_dir.iterdir())
    new_dirs = sorted(after - before)
    run_dir = exp_dir / new_dirs[-1] if new_dirs else latest_run_dir(exp_dir)

    metrics_path = run_dir / "metrics.csv"
    model_state_path = run_dir / "model_state.pt"
    run_start_path = run_dir / "run_start.json"

    best_epoch, best_test_mae = read_best_epoch_and_mae(metrics_path)
    param_count = count_parameters(model_state_path)
    with run_start_path.open(encoding="utf-8") as f:
        start = json.load(f)
    epochs = int(start["config"]["epochs"])

    return {
        "model_family": label,
        "task": "graph_level_lcc_fraction",
        "graph_tag": "N40_tiered_scale_free",
        "seed": 1,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_test_mae": f"{best_test_mae:.4f}",
        "total_runtime_sec": f"{total_runtime_sec:.2f}",
        "sec_per_epoch": f"{(total_runtime_sec / epochs):.4f}",
        "param_count": param_count,
        "run_path": str(run_dir),
    }


def main() -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("baseline", "train_baseline_graph_level.py", "baseline_graph_level_benchmark", []),
        ("gcn", "train_gcn_graph_level.py", "gcn_graph_level_benchmark", []),
        ("graph_transformer", "train_graph_transformer_graph_level.py", "graph_transformer_graph_level_benchmark", []),
        ("graphormer", "train_graphormer_graph_level.py", "graphormer_graph_level_benchmark", []),
    ]

    rows = [run_one(*cfg) for cfg in configs]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_family",
                "task",
                "graph_tag",
                "seed",
                "epochs",
                "best_epoch",
                "best_test_mae",
                "total_runtime_sec",
                "sec_per_epoch",
                "param_count",
                "run_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(OUT_CSV)


if __name__ == "__main__":
    main()
