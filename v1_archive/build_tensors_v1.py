import pandas as pd
import torch

NODE_CSV = "demo2_node_observables.csv"

df = pd.read_csv(NODE_CSV).sort_values(["t", "node"])

FEATURE_COLS = ["health", "exposure", "time_to_recovery"]

T = df["t"].nunique()
N = df["node"].nunique()
F = len(FEATURE_COLS)

# X[t,node,f] predicts Y[t,node] = health at t+1
X = torch.zeros(T - 1, N, F, dtype=torch.float32)
Y = torch.zeros(T - 1, N, dtype=torch.float32)

for t in range(T - 1):
    df_t = df[df["t"] == t]
    df_t1 = df[df["t"] == t + 1]

    X[t] = torch.tensor(df_t[FEATURE_COLS].values, dtype=torch.float32)
    Y[t] = torch.tensor(df_t1["health"].values, dtype=torch.float32)

torch.save(X, "X_v1.pt")
torch.save(Y, "Y_v1.pt")

print("Saved X_v1.pt, Y_v1.pt")
print("X shape:", tuple(X.shape))  # (T-1, N, F)
print("Y shape:", tuple(Y.shape))  # (T-1, N)
