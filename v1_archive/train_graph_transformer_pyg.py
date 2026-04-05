import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from run_logger import append_metric, create_run_dir, save_run_end, save_run_start

try:
    from torch_geometric.nn import TransformerConv
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: torch_geometric. Install PyTorch Geometric before "
        "running train_graph_transformer_pyg.py."
    ) from exc


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


class GraphTransformerPyG(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    beta=True,
                )
            )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)
        for conv, norm in zip(self.layers, self.norms):
            h_next = conv(h, edge_index)
            h = norm(h + self.dropout(h_next))
        return self.head(h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph_type", type=str, default="chain")
    parser.add_argument("--graph_tag", type=str, default="N3_chain")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--x_name", type=str, default="X_v1.pt")
    parser.add_argument("--y_name", type=str, default="Y_v1.pt")
    parser.add_argument("--edge_name", type=str, default="edge_index.pt")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, default="graph_transformer_pyg")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")

    x_path, y_path, edge_path, meta_path = resolve_data_paths(
        args.data_dir, args.x_name, args.y_name, args.edge_name
    )
    X = torch.load(x_path)
    Y = torch.load(y_path)
    T, N, F = X.shape

    graph_meta = {}
    if meta_path.exists():
        graph_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if edge_path.exists():
        edge_index = torch.load(edge_path).long()
    else:
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    num_edges_message_passing = int(edge_index.shape[1])
    src, dst = edge_index
    unique_undirected_edges = {
        tuple(sorted((int(u), int(v))))
        for u, v in zip(src.tolist(), dst.tolist())
        if int(u) != int(v)
    }
    num_edges_physical = len(unique_undirected_edges)

    Yf = Y.reshape(T * N, 1)
    pos = Yf.sum()
    neg = (1 - Yf).sum()
    pos_weight = neg / (pos + 1e-8)
    print("pos_weight:", float(pos_weight))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GraphTransformerPyG(
        in_dim=F,
        hidden_dim=args.hidden_dim,
        heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    run_dir = create_run_dir(args.experiment_name, root=args.run_root)
    split_t = int(T * 0.7)

    save_run_start(
        run_dir=run_dir,
        model=model,
        config={
            "script": "train_graph_transformer_pyg.py",
            "seed": args.seed,
            "graph_type": args.graph_type,
            "graph_tag": args.graph_tag,
            "lr": args.lr,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "deterministic_algorithms": True,
            "data_dir": args.data_dir,
        },
        dataset_stats={
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
            "num_timesteps": T,
            "num_nodes": N,
            "num_features": F,
            "split_t": split_t,
            "pos_weight": pos_weight,
            "num_edges_physical": graph_meta.get("num_edges_physical", num_edges_physical),
            "num_edges_message_passing": graph_meta.get(
                "num_edges_message_passing", num_edges_message_passing
            ),
            "edge_index": edge_index,
        },
    )

    opt = optim.Adam(model.parameters(), lr=args.lr)
    last_test_acc = None
    final_avg_train_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for t in range(split_t):
            logits = model(X[t], edge_index)
            y_next = Y[t].reshape(N, 1)
            loss = loss_fn(logits, y_next)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += loss.item()

        final_avg_train_loss = loss_total / split_t
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for t in range(split_t, T):
                    probs = torch.sigmoid(model(X[t], edge_index))
                    preds = (probs >= 0.5).float()
                    y_true = Y[t].reshape(N, 1)
                    correct += (preds.eq(y_true)).sum().item()
                    total += y_true.numel()
                acc = correct / total
                last_test_acc = acc

            append_metric(
                run_dir,
                {
                    "epoch": epoch + 1,
                    "avg_train_loss": final_avg_train_loss,
                    "test_acc": acc,
                },
            )
            print(
                f"epoch {epoch+1:03d}  avg_train_loss {final_avg_train_loss:.4f}  test_acc {acc:.3f}"
            )

    model.eval()
    with torch.no_grad():
        probs_last = []
        for t in range(T - 5, T):
            probs_last.append(torch.sigmoid(model(X[t], edge_index)).squeeze(-1))
        probs_last = torch.stack(probs_last, dim=0)

    torch.save(probs_last, run_dir / "probs_last5.pt")
    save_run_end(
        run_dir=run_dir,
        model=model,
        summary={
            "final_avg_train_loss": final_avg_train_loss,
            "last_test_acc": last_test_acc,
        },
    )
    print(f"Run artifacts saved to: {run_dir}")
    print("\nPredicted P(healthy at t+1) for last 5 timesteps:")
    print(probs_last)


if __name__ == "__main__":
    main()
