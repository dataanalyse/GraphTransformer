import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class SimParams:
    T: int = 20
    seed: int = 7

    # Exogenous disruption: chance a healthy node fails at each step
    p_shock: float = 0.05

    # Propagation: chance a node fails if it has disrupted upstream neighbors
    p_propagate: float = 0.35

    # Recovery: chance a disrupted node recovers at each step
    p_recover: float = 0.25


def build_tiny_supply_chain_graph() -> nx.DiGraph:
    """
    3-node directed supply chain:
      0 (Supplier) -> 1 (Manufacturer) -> 2 (Retailer)
    """
    G = nx.DiGraph()
    G.add_node(0, role="supplier")
    G.add_node(1, role="manufacturer")
    G.add_node(2, role="retailer")

    # Material flow direction
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    return G


def compute_exposure(G: nx.DiGraph, health: Dict[int, int]) -> Dict[int, float]:
    """
    Exposure of node i = fraction of upstream neighbors that are disrupted.
    upstream = predecessors (suppliers feeding into i).
    If a node has no upstream, exposure = 0.
    """
    exposure = {}
    for i in G.nodes():
        preds = list(G.predecessors(i))
        if not preds:
            exposure[i] = 0.0
            continue
        disrupted_upstream = sum(1 for j in preds if health[j] == 0)
        exposure[i] = disrupted_upstream / float(len(preds))
    return exposure


def simulate(G: nx.DiGraph, params: SimParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate node disruptions over time and log:
      - health[i,t] in {0,1}
      - exposure[i,t] in [0,1]
      - time_to_recovery[i,t]: if node is disrupted, counts steps since it went down; else 0
    Returns:
      node_df: rows per (t, node)
      sys_df: rows per t (system-level metrics)
    """
    random.seed(params.seed)
    np.random.seed(params.seed)

    nodes = sorted(G.nodes())
    N = len(nodes)

    # 1 = healthy, 0 = disrupted
    health = {i: 1 for i in nodes}

    # Track how long a node has been down (for time-to-recovery proxy)
    down_age = {i: 0 for i in nodes}

    node_rows: List[dict] = []
    sys_rows: List[dict] = []

    for t in range(params.T):
        # Compute exposure BEFORE state transitions (what the system "sees" at time t)
        exposure = compute_exposure(G, health)

        # Log current state (observables at time t)
        healthy_pct = sum(health.values()) / N
        sys_rows.append(
            {
                "t": t,
                "healthy_pct": healthy_pct,
                "num_disrupted": N - sum(health.values()),
            }
        )

        for i in nodes:
            node_rows.append(
                {
                    "t": t,
                    "node": i,
                    "role": G.nodes[i]["role"],
                    "health": health[i],
                    "exposure": exposure[i],
                    "time_to_recovery": down_age[i] if health[i] == 0 else 0,
                }
            )

        # --- State transition to t+1 (shock, propagate, recover) ---
        next_health = dict(health)

        # A) Exogenous shocks (hit healthy nodes randomly)
        for i in nodes:
            if health[i] == 1 and random.random() < params.p_shock:
                next_health[i] = 0

        # B) Propagation from upstream disruptions
        # If node is healthy, and exposure>0, it may fail with prob p_propagate * exposure
        for i in nodes:
            if next_health[i] == 1:
                p_fail = params.p_propagate * exposure[i]
                if random.random() < p_fail:
                    next_health[i] = 0

        # C) Recovery
        for i in nodes:
            if next_health[i] == 0:
                # Node is disrupted -> chance to recover
                if random.random() < params.p_recover:
                    next_health[i] = 1

        # Update down ages based on the transition
        for i in nodes:
            if health[i] == 1 and next_health[i] == 0:
                # just went down
                down_age[i] = 1
            elif health[i] == 0 and next_health[i] == 0:
                # stayed down
                down_age[i] += 1
            else:
                # healthy (either stayed healthy or recovered)
                down_age[i] = 0

        health = next_health

    node_df = pd.DataFrame(node_rows)
    sys_df = pd.DataFrame(sys_rows)
    return node_df, sys_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--p_shock", type=float, default=0.05)
    parser.add_argument("--p_propagate", type=float, default=0.35)
    parser.add_argument("--p_recover", type=float, default=0.25)
    parser.add_argument("--out_prefix", type=str, default="sc_v0")
    args = parser.parse_args()

    params = SimParams(
        T=args.T,
        seed=args.seed,
        p_shock=args.p_shock,
        p_propagate=args.p_propagate,
        p_recover=args.p_recover,
    )

    G = build_tiny_supply_chain_graph()
    node_df, sys_df = simulate(G, params)

    # Save outputs
    node_path = f"{args.out_prefix}_node_observables.csv"
    sys_path = f"{args.out_prefix}_system_observables.csv"
    node_df.to_csv(node_path, index=False)
    sys_df.to_csv(sys_path, index=False)

    # Print a quick peek
    print("\n=== System-level observables (first 10 rows) ===")
    print(sys_df.head(10).to_string(index=False))

    print("\n=== Node-level observables (first 15 rows) ===")
    print(node_df.head(15).to_string(index=False))

    print(f"\nWrote:\n  {node_path}\n  {sys_path}")


if __name__ == "__main__":
    main()
