import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch


def _to_builtin(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def create_run_dir(experiment: str, root: str = "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(root) / experiment / stamp
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = Path(f"{base}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_start(
    run_dir: Path,
    model: torch.nn.Module,
    config: Dict,
    dataset_stats: Optional[Dict] = None,
) -> None:
    (run_dir / "architecture.txt").write_text(str(model), encoding="utf-8")

    payload = {"config": {k: _to_builtin(v) for k, v in config.items()}}
    if dataset_stats is not None:
        payload["dataset_stats"] = {k: _to_builtin(v) for k, v in dataset_stats.items()}

    (run_dir / "run_start.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def append_metric(run_dir: Path, row: Dict) -> None:
    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists()
    clean_row = {k: _to_builtin(v) for k, v in row.items()}
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def save_run_end(
    run_dir: Path,
    model: torch.nn.Module,
    summary: Optional[Dict] = None,
) -> None:
    torch.save(model.state_dict(), run_dir / "model_state.pt")

    if summary is not None:
        payload = {k: _to_builtin(v) for k, v in summary.items()}
        (run_dir / "run_end.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
