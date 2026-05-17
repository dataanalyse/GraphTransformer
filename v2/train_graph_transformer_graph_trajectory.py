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
from train_graph_transformer_graph_level import (
    GraphTransformerBlock,
    build_graph_attention_mask,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_data_paths(data_dir: str, x_name: str, y_name: str, edge_name: str):
    if data_dir:
        base = Path(data_dir)
        return (
            base / x_name,
            base / y_name,
            base / edge_name,
            base / "graph_meta.json",
        )
    return Path(x_name), Path(y_name), Path(edge_name), Path("graph_meta.json")


class GraphTransformerTrajectory(nn.Module):
    def __init__(
        self,
        in_dim: int,
        horizon: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [
                GraphTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x_t: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)
        h = self.norm(h)
        graph_repr = h.mean(dim=1)
        return self.head(graph_repr).squeeze(0)


def evaluate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph_type", type=str, default="chain")
    parser.add_argument("--graph_tag", type=str, default="N3_chain")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--x_name", type=str, default="X_v1.pt")
    parser.add_argument("--y_name", type=str, default="Y_lcc_traj_v1.pt")
    parser.add_argument("--edge_name", type=str, default="edge_index.pt")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, default="graph_transformer_graph_trajectory")
    parser.add_argument("--prediction_horizon", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    x_path, y_path, edge_path, meta_path = resolve_data_paths(
        args.data_dir, args.x_name, args.y_name, args.edge_name
    )
    X = torch.load(x_path)
    Y = torch.load(y_path).float()

    graph_meta = {}
    if meta_path.exists():
        graph_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_meta = graph_meta.get("target_definition", {})
    prediction_horizon = int(target_meta.get("prediction_horizon", args.prediction_horizon))

    if edge_path.exists():
        edge_index = torch.load(edge_path).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    feature_flags = graph_meta.get("feature_flags", {})
    X, feature_names, _ = append_runtime_betweenness(X, feature_flags, edge_index)
    T, N, F = X.shape
    horizon = Y.shape[1]
    print("Loaded trajectory tensors.")
    print_feature_summary(feature_names, X)
    print("Y_lcc_traj shape:", tuple(Y.shape))
    print(f"Prediction horizon: {prediction_horizon}")

    attn_mask = build_graph_attention_mask(N, edge_index)
    loss_fn = nn.MSELoss()
    model = GraphTransformerTrajectory(
        in_dim=F,
        horizon=horizon,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    run_dir = create_run_dir(args.experiment_name, root=args.run_root)
    split_t = int(T * 0.7)

    save_run_start(
        run_dir=run_dir,
        model=model,
        config={
            "script": "train_graph_transformer_graph_trajectory.py",
            "seed": args.seed,
            "graph_type": args.graph_type,
            "graph_tag": args.graph_tag,
            "lr": args.lr,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "prediction_horizon": prediction_horizon,
            "graph_target_name": "lcc_trajectory",
            "graph_target_desc": target_meta.get("graph_trajectory_label_semantics", ""),
        },
        dataset_stats={
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
            "num_timesteps": T,
            "num_nodes": N,
            "num_features": F,
            "feature_list": feature_names,
            "active_features": format_feature_list(feature_names),
            "prediction_horizon": prediction_horizon,
            "split_t": split_t,
            "num_edges_physical": graph_meta.get("num_edges_physical", ""),
            "num_edges_message_passing": graph_meta.get("num_edges_message_passing", ""),
            "edge_index": edge_index,
        },
    )

    opt = optim.Adam(model.parameters(), lr=args.lr)
    last_test_mae = None
    final_train_mse = 0.0

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for t in range(split_t):
            pred = model(X[t], attn_mask)
            loss = loss_fn(pred, Y[t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += loss.item()

        final_train_mse = loss_total / split_t
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                preds = []
                targets = []
                for t in range(split_t, T):
                    preds.append(model(X[t], attn_mask))
                    targets.append(Y[t])
                pred_tensor = torch.stack(preds)
                target_tensor = torch.stack(targets)
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
            print(
                f"epoch {epoch+1:03d}  train_mse {final_train_mse:.4f}  test_mae {test_mae:.4f}"
            )

    model.eval()
    with torch.no_grad():
        preds_all = torch.stack([model(X[t], attn_mask) for t in range(T)])

    torch.save(preds_all, run_dir / "preds_all.pt")
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
