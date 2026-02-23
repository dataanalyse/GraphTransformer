import csv
import json
from pathlib import Path
from typing import Any, List


def _safe_shape(shape: Any) -> List[int]:
    if isinstance(shape, list) and all(isinstance(v, int) for v in shape):
        return shape
    return []


def main() -> None:
    runs_root = Path("runs")
    out_path = runs_root / "results_summary.csv"
    rows = []

    if runs_root.exists():
        for exp_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
            for run_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
                start_path = run_dir / "run_start.json"
                end_path = run_dir / "run_end.json"
                metrics_path = run_dir / "metrics.csv"

                if not start_path.exists():
                    continue

                start = json.loads(start_path.read_text(encoding="utf-8"))
                end = (
                    json.loads(end_path.read_text(encoding="utf-8"))
                    if end_path.exists()
                    else {}
                )
                cfg = start.get("config", {})
                ds = start.get("dataset_stats", {})
                x_shape = _safe_shape(ds.get("X_shape", []))
                num_nodes = ds.get("num_nodes", x_shape[1] if len(x_shape) >= 2 else "")
                num_timesteps = ds.get(
                    "num_timesteps", x_shape[0] if len(x_shape) >= 1 else ""
                )
                num_features = ds.get(
                    "num_features", x_shape[2] if len(x_shape) >= 3 else ""
                )

                edge_index = ds.get("edge_index", [])
                num_edges_message_passing = ds.get("num_edges_message_passing", "")
                num_edges_physical = ds.get("num_edges_physical", "")
                if (
                    num_edges_message_passing == ""
                    and isinstance(edge_index, list)
                    and len(edge_index) == 2
                    and all(isinstance(row, list) for row in edge_index)
                ):
                    num_edges_message_passing = min(len(edge_index[0]), len(edge_index[1]))
                    if num_edges_physical == "":
                        undirected = {
                            tuple(sorted((u, v)))
                            for u, v in zip(edge_index[0], edge_index[1])
                            if u != v
                        }
                        num_edges_physical = len(undirected)

                graph_type = cfg.get("graph_type", "")
                graph_tag = cfg.get("graph_tag", "")
                if graph_tag == "" and num_nodes != "":
                    suffix = graph_type if graph_type else "graph"
                    graph_tag = f"N{num_nodes}_{suffix}"

                best_test_acc = ""
                final_logged_epoch = ""
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8", newline="") as f:
                        mrows = list(csv.DictReader(f))
                    if mrows:
                        final_logged_epoch = mrows[-1].get("epoch", "")
                        acc_vals = []
                        for row in mrows:
                            value = row.get("test_acc", "")
                            try:
                                acc_vals.append(float(value))
                            except Exception:
                                pass
                        if acc_vals:
                            best_test_acc = max(acc_vals)

                rows.append(
                    {
                        "experiment": exp_dir.name,
                        "run_id": run_dir.name,
                        "script": cfg.get("script", ""),
                        "graph_tag": graph_tag,
                        "graph_type": graph_type,
                        "num_nodes": num_nodes,
                        "num_edges_physical": num_edges_physical,
                        "num_edges_message_passing": num_edges_message_passing,
                        "seed": cfg.get("seed", ""),
                        "epochs": cfg.get("epochs", ""),
                        "lr": cfg.get("lr", ""),
                        "hidden_dim": cfg.get("hidden_dim", ""),
                        "d_model": cfg.get("d_model", ""),
                        "num_heads": cfg.get("num_heads", ""),
                        "num_layers": cfg.get("num_layers", ""),
                        "ff_dim": cfg.get("ff_dim", ""),
                        "dropout": cfg.get("dropout", ""),
                        "split_t": ds.get("split_t", ""),
                        "num_timesteps": num_timesteps,
                        "num_features": num_features,
                        "X_shape": ds.get("X_shape", ""),
                        "Y_shape": ds.get("Y_shape", ""),
                        "pos_weight": ds.get("pos_weight", ""),
                        "last_test_acc": end.get("last_test_acc", ""),
                        "best_test_acc": best_test_acc,
                        "final_train_loss": end.get("final_train_loss", ""),
                        "final_avg_train_loss": end.get("final_avg_train_loss", ""),
                        "final_logged_epoch": final_logged_epoch,
                        "run_path": str(run_dir).replace("\\", "/"),
                    }
                )

    fieldnames = [
        "experiment",
        "run_id",
        "script",
        "graph_tag",
        "graph_type",
        "num_nodes",
        "num_edges_physical",
        "num_edges_message_passing",
        "seed",
        "epochs",
        "lr",
        "hidden_dim",
        "d_model",
        "num_heads",
        "num_layers",
        "ff_dim",
        "dropout",
        "split_t",
        "num_timesteps",
        "num_features",
        "X_shape",
        "Y_shape",
        "pos_weight",
        "last_test_acc",
        "best_test_acc",
        "final_train_loss",
        "final_avg_train_loss",
        "final_logged_epoch",
        "run_path",
    ]

    runs_root.mkdir(parents=True, exist_ok=True)
    target_path = out_path
    try:
        with target_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        target_path = runs_root / "results_summary_latest.csv"
        with target_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Could not write {out_path} (file locked). Wrote {target_path} instead.")
    else:
        print(f"Wrote {target_path}")

    print(target_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
