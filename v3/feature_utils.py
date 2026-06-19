from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd
import torch


FEATURE_SPECS = [
    {
        "name": "health",
        "flag": "use_health",
        "column": "health",
        "saved_in_x": True,
    },
    {
        "name": "exposure",
        "flag": "use_exposure",
        "column": "exposure",
        "saved_in_x": True,
    },
    {
        "name": "time_to_recovery",
        "flag": "use_time_to_recovery",
        "column": "time_to_recovery",
        "saved_in_x": True,
    },
    {
        "name": "betweenness_centrality",
        "flag": "use_betweenness",
        "column": None,
        "saved_in_x": False,
    },
]

DEFAULT_FEATURE_FLAGS = {spec["flag"]: True for spec in FEATURE_SPECS}


def parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean flag from: {value}")


def normalize_feature_flags(feature_flags: dict[str, Any] | None) -> dict[str, bool]:
    flags = dict(DEFAULT_FEATURE_FLAGS)
    if feature_flags:
        for key, value in feature_flags.items():
            if key in flags:
                flags[key] = parse_bool_flag(value)
    return flags


def feature_order_documentation() -> list[str]:
    return [spec["name"] for spec in FEATURE_SPECS]


def saved_feature_names(feature_flags: dict[str, Any] | None) -> list[str]:
    flags = normalize_feature_flags(feature_flags)
    return [
        spec["name"]
        for spec in FEATURE_SPECS
        if spec["saved_in_x"] and flags[spec["flag"]]
    ]


def final_feature_names(
    feature_flags: dict[str, Any] | None,
    include_runtime_betweenness: bool = False,
) -> list[str]:
    flags = normalize_feature_flags(feature_flags)
    names = saved_feature_names(flags)
    if include_runtime_betweenness and flags["use_betweenness"]:
        names.append("betweenness_centrality")
    return names


def build_saved_feature_tensors(
    df: pd.DataFrame,
    feature_flags: dict[str, Any] | None,
    prediction_horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    flags = normalize_feature_flags(feature_flags)
    feature_cols = [
        spec["column"]
        for spec in FEATURE_SPECS
        if spec["saved_in_x"] and flags[spec["flag"]]
    ]
    feature_names = saved_feature_names(flags)

    T = df["t"].nunique()
    N = df["node"].nunique()
    F = len(feature_cols)
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1")
    if prediction_horizon >= T:
        raise ValueError(
            f"prediction_horizon={prediction_horizon} is too large for T={T}. "
            "It must be smaller than the number of timesteps."
        )

    usable_steps = T - prediction_horizon
    X = torch.zeros(usable_steps, N, F, dtype=torch.float32)
    Y = torch.zeros(usable_steps, N, dtype=torch.float32)

    for t in range(usable_steps):
        df_t = df[df["t"] == t]
        df_t_target = df[df["t"] == t + prediction_horizon]
        if feature_cols:
            X[t] = torch.tensor(df_t[feature_cols].values, dtype=torch.float32)
        Y[t] = torch.tensor(df_t_target["health"].values, dtype=torch.float32)

    return X, Y, feature_names


def build_betweenness_feature(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    src, dst = edge_index.tolist()
    graph.add_edges_from(zip(src, dst))

    centrality = nx.betweenness_centrality(graph, normalized=True)
    values = [float(centrality[node]) for node in range(num_nodes)]
    return torch.tensor(values, dtype=torch.float32)


def append_runtime_betweenness(
    X: torch.Tensor,
    feature_flags: dict[str, Any] | None,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
    flags = normalize_feature_flags(feature_flags)
    names = final_feature_names(flags, include_runtime_betweenness=flags["use_betweenness"])
    if not flags["use_betweenness"]:
        return X, names, None

    T, N, _ = X.shape
    betweenness = build_betweenness_feature(N, edge_index)
    betweenness_feature = betweenness.view(1, N, 1).expand(T, N, 1)
    return torch.cat([X, betweenness_feature], dim=2), names, betweenness


def print_feature_summary(feature_names: list[str], X: torch.Tensor) -> None:
    print("Final feature list:", feature_names)
    print("X shape:", tuple(X.shape))


def format_feature_list(feature_names: list[str]) -> str:
    return ",".join(feature_names)
