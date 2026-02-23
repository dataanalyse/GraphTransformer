import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from run_logger import append_metric, create_run_dir, save_run_end, save_run_start


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_data_paths(data_dir: str, x_name: str, y_name: str):
    if data_dir:
        base = Path(data_dir)
        return base / x_name, base / y_name, base / "graph_meta.json"
    return Path(x_name), Path(y_name), Path("graph_meta.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph_type", type=str, default="chain")
    parser.add_argument("--graph_tag", type=str, default="N3_chain")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--x_name", type=str, default="X_v1.pt")
    parser.add_argument("--y_name", type=str, default="Y_v1.pt")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, default="baseline")
    args = parser.parse_args()
    set_seed(args.seed)

    x_path, y_path, meta_path = resolve_data_paths(args.data_dir, args.x_name, args.y_name)
    X = torch.load(x_path)
    Y = torch.load(y_path)
    print("Loaded tensors.")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Y unique:", torch.unique(Y))

    graph_meta = {}
    if meta_path.exists():
        graph_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    T1, N, F = X.shape
    Xf = X.reshape(T1 * N, F)
    Yf = Y.reshape(T1 * N, 1)

    pos = Yf.sum()
    neg = (1 - Yf).sum()
    pos_weight = neg / (pos + 1e-8)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("pos_weight:", float(pos_weight))

    model = nn.Sequential(nn.Linear(F, 1))
    run_dir = create_run_dir(args.experiment_name, root=args.run_root)

    split_t = int(T1 * 0.7)
    train_idx = torch.arange(0, split_t * N)
    test_idx = torch.arange(split_t * N, T1 * N)

    save_run_start(
        run_dir=run_dir,
        model=model,
        config={
            "script": "train_baseline_v1.py",
            "seed": args.seed,
            "graph_type": args.graph_type,
            "graph_tag": args.graph_tag,
            "lr": args.lr,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "deterministic_algorithms": True,
            "data_dir": args.data_dir,
        },
        dataset_stats={
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
            "num_timesteps": T1,
            "num_nodes": N,
            "num_features": F,
            "split_t": split_t,
            "pos_weight": pos_weight,
            "num_edges_physical": graph_meta.get("num_edges_physical", ""),
            "num_edges_message_passing": graph_meta.get("num_edges_message_passing", ""),
        },
    )

    opt = optim.Adam(model.parameters(), lr=args.lr)
    last_test_acc = None
    loss = torch.tensor(0.0)

    for epoch in range(args.epochs):
        model.train()
        logits = model(Xf[train_idx])
        loss = loss_fn(logits, Yf[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(Xf[test_idx]))
                preds = (probs >= 0.5).float()
                acc = (preds.eq(Yf[test_idx])).float().mean().item()
                last_test_acc = acc
            append_metric(
                run_dir,
                {
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "test_acc": acc,
                },
            )
            print(f"epoch {epoch+1:03d}  loss {loss.item():.4f}  test_acc {acc:.3f}")

    model.eval()
    with torch.no_grad():
        probs_all = torch.sigmoid(model(Xf)).reshape(T1, N)

    torch.save(probs_all, run_dir / "probs_all.pt")
    save_run_end(
        run_dir=run_dir,
        model=model,
        summary={
            "final_train_loss": loss.item(),
            "last_test_acc": last_test_acc,
        },
    )
    print(f"Run artifacts saved to: {run_dir}")
    print("\nPredicted P(healthy at t+1) for last 5 timesteps:")
    print(probs_all[-5:])


if __name__ == "__main__":
    main()
