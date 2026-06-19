import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from feature_utils import append_runtime_betweenness, format_feature_list, print_feature_summary
from run_logger import append_metric, create_run_dir, save_run_end, save_run_start
from train_gcn_graph_level import GCNGraphLevel, norm_adj
from train_graph_transformer_graph_level import (
    GraphTransformerGraphLevel,
    build_graph_attention_mask,
)
from train_graphormer_graph_level import GraphormerGraphLevel, build_structural_inputs


TARGET_FILES = {
    "lcc_fraction": "Y_lcc_v1.pt",
    "component_fraction": "Y_components_v1.pt",
    "diameter_fraction": "Y_diameter_v1.pt",
    "edge_survival_ratio": "Y_edge_survival_v1.pt",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def evaluate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def load_seed_dataset(
    data_dir: Path,
    graph_target_key: str,
) -> dict:
    meta_path = data_dir / "graph_meta.json"
    graph_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    target_meta = graph_meta.get("target_definition", {})
    graph_target_desc = target_meta.get("available_graph_targets", {}).get(
        graph_target_key,
        graph_target_key,
    )

    X = torch.load(data_dir / "X_v1.pt")
    Y = torch.load(data_dir / TARGET_FILES[graph_target_key]).float()
    edge_index = torch.load(data_dir / "edge_index.pt").long()
    feature_flags = graph_meta.get("feature_flags", {})
    X, feature_names, betweenness = append_runtime_betweenness(X, feature_flags, edge_index)

    dataset = {
        "X": X,
        "Y": Y,
        "edge_index": edge_index,
        "graph_meta": graph_meta,
        "feature_names": feature_names,
        "betweenness": betweenness,
        "graph_target_desc": graph_target_desc,
    }
    return dataset


def build_model(args, num_features: int):
    if args.model_type == "baseline":
        return nn.Sequential(nn.Linear(args.num_nodes * num_features, 1))
    if args.model_type == "gcn":
        return GCNGraphLevel(num_features, hid=args.hidden_dim)
    if args.model_type == "graph_transformer":
        return GraphTransformerGraphLevel(
            in_dim=num_features,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        )
    if args.model_type == "graphormer":
        return GraphormerGraphLevel(
            in_dim=num_features,
            num_heads=args.num_heads,
            d_model=args.d_model,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_dist=args.max_dist,
            max_degree=args.max_degree,
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def stack_baseline_examples(datasets: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [dataset["X"].reshape(dataset["X"].shape[0], -1) for dataset in datasets]
    ys = [dataset["Y"].reshape(dataset["Y"].shape[0], 1) for dataset in datasets]
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def predict_single(model, dataset: dict, t: int, args) -> torch.Tensor:
    x_t = dataset["X"][t]
    if args.model_type == "baseline":
        return model(x_t.reshape(1, -1)).squeeze()
    if args.model_type == "gcn":
        return model(x_t, dataset["A_hat"])
    if args.model_type == "graph_transformer":
        return model(x_t, dataset["attn_mask"])
    if args.model_type == "graphormer":
        return model(
            x_t,
            dataset["in_degree"],
            dataset["out_degree"],
            dataset["dist_bucket"],
        )
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def prepare_dataset_state(dataset: dict, args) -> None:
    edge_index = dataset["edge_index"]
    num_nodes = dataset["X"].shape[1]
    if args.model_type == "gcn":
        dataset["A_hat"] = norm_adj(num_nodes, edge_index)
    elif args.model_type == "graph_transformer":
        dataset["attn_mask"] = build_graph_attention_mask(num_nodes, edge_index)
    elif args.model_type == "graphormer":
        in_degree, out_degree, dist_bucket = build_structural_inputs(
            num_nodes, edge_index, max_dist=args.max_dist
        )
        dataset["in_degree"] = in_degree
        dataset["out_degree"] = out_degree
        dataset["dist_bucket"] = dist_bucket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph_type", type=str, default="chain")
    parser.add_argument("--graph_tag", type=str, default="N3_chain")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_seeds", type=str, default="1,2,3")
    parser.add_argument("--test_seeds", type=str, default="4,5")
    parser.add_argument("--graph_target_key", type=str, default="lcc_fraction")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_dist", type=int, default=4)
    parser.add_argument("--max_degree", type=int, default=16)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--prediction_horizon", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    train_seed_list = parse_seed_list(args.train_seeds)
    test_seed_list = parse_seed_list(args.test_seeds)
    if not train_seed_list or not test_seed_list:
        raise ValueError("Both train_seeds and test_seeds must be non-empty.")

    data_root = Path(args.data_root)
    train_datasets = [
        load_seed_dataset(data_root / f"seed_{seed}", args.graph_target_key)
        for seed in train_seed_list
    ]
    test_datasets = [
        load_seed_dataset(data_root / f"seed_{seed}", args.graph_target_key)
        for seed in test_seed_list
    ]

    reference = train_datasets[0]
    num_timesteps, num_nodes, num_features = reference["X"].shape
    args.num_nodes = num_nodes

    for dataset in train_datasets + test_datasets:
        prepare_dataset_state(dataset, args)

    print("Loaded held-out-seed datasets.")
    print_feature_summary(reference["feature_names"], reference["X"])
    print(f"Graph target: {args.graph_target_key} -> {reference['graph_target_desc']}")
    print(f"Train seeds: {train_seed_list}")
    print(f"Test seeds: {test_seed_list}")

    model = build_model(args, num_features)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    run_dir = create_run_dir(args.experiment_name, root=args.run_root)

    save_run_start(
        run_dir=run_dir,
        model=model,
        config={
            "script": "train_graph_level_seed_split.py",
            "seed": args.seed,
            "graph_type": args.graph_type,
            "graph_tag": args.graph_tag,
            "lr": args.lr,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim if args.model_type == "gcn" else "",
            "d_model": args.d_model if args.model_type in {"graph_transformer", "graphormer"} else "",
            "num_heads": args.num_heads if args.model_type in {"graph_transformer", "graphormer"} else "",
            "num_layers": args.num_layers if args.model_type in {"graph_transformer", "graphormer"} else "",
            "ff_dim": args.ff_dim if args.model_type in {"graph_transformer", "graphormer"} else "",
            "dropout": args.dropout if args.model_type in {"graph_transformer", "graphormer"} else "",
            "max_dist": args.max_dist if args.model_type == "graphormer" else "",
            "max_degree": args.max_degree if args.model_type == "graphormer" else "",
            "prediction_horizon": args.prediction_horizon,
            "graph_target_name": args.graph_target_key,
            "graph_target_desc": reference["graph_target_desc"],
            "split_mode": "heldout_seed",
            "train_seeds": args.train_seeds,
            "test_seeds": args.test_seeds,
            "model_type": args.model_type,
            "data_root": str(data_root),
        },
        dataset_stats={
            "X_shape": list(reference["X"].shape),
            "Y_shape": list(reference["Y"].shape),
            "num_timesteps": num_timesteps,
            "num_nodes": num_nodes,
            "num_features": num_features,
            "feature_list": reference["feature_names"],
            "active_features": format_feature_list(reference["feature_names"]),
            "prediction_horizon": args.prediction_horizon,
            "split_t": "",
            "num_edges_physical": reference["graph_meta"].get("num_edges_physical", ""),
            "num_edges_message_passing": reference["graph_meta"].get("num_edges_message_passing", ""),
            "num_train_examples": int(sum(ds["X"].shape[0] for ds in train_datasets)),
            "num_test_examples": int(sum(ds["X"].shape[0] for ds in test_datasets)),
        },
    )

    last_test_mae = None
    final_train_mse = 0.0
    baseline_train_X = baseline_train_Y = baseline_test_X = baseline_test_Y = None
    if args.model_type == "baseline":
        baseline_train_X, baseline_train_Y = stack_baseline_examples(train_datasets)
        baseline_test_X, baseline_test_Y = stack_baseline_examples(test_datasets)

    for epoch in range(args.epochs):
        model.train()
        if args.model_type == "baseline":
            pred = model(baseline_train_X)
            loss = loss_fn(pred, baseline_train_Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_train_mse = loss.item()
        else:
            loss_total = 0.0
            train_count = 0
            for dataset in train_datasets:
                for t in range(dataset["X"].shape[0]):
                    pred = predict_single(model, dataset, t, args)
                    loss = loss_fn(pred.view(1), dataset["Y"][t].view(1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    loss_total += loss.item()
                    train_count += 1
            final_train_mse = loss_total / max(1, train_count)

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                if args.model_type == "baseline":
                    pred_tensor = model(baseline_test_X).view(-1)
                    target_tensor = baseline_test_Y.view(-1)
                else:
                    preds = []
                    targets = []
                    for dataset in test_datasets:
                        for t in range(dataset["X"].shape[0]):
                            preds.append(predict_single(model, dataset, t, args).view(1))
                            targets.append(dataset["Y"][t].view(1))
                    pred_tensor = torch.cat(preds)
                    target_tensor = torch.cat(targets)
                test_mae = evaluate_mae(pred_tensor, target_tensor)
                last_test_mae = test_mae
            append_metric(
                run_dir,
                {
                    "epoch": epoch + 1,
                    "train_mse": final_train_mse,
                    "test_mae": test_mae,
                },
            )
            print(f"epoch {epoch+1:03d}  train_mse {final_train_mse:.4f}  heldout_test_mae {test_mae:.4f}")

    model.eval()
    with torch.no_grad():
        heldout_preds = []
        for dataset in test_datasets:
            pred_series = torch.stack(
                [predict_single(model, dataset, t, args) for t in range(dataset["X"].shape[0])]
            )
            heldout_preds.append(pred_series)

    torch.save(heldout_preds, run_dir / "heldout_preds.pt")
    save_run_end(
        run_dir=run_dir,
        model=model,
        summary={
            "final_train_mse": final_train_mse,
            "last_test_mae": last_test_mae,
        },
    )
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
