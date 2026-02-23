import random
from typing import List, Optional, Tuple

import networkx as nx
import torch


def make_roles(num_nodes: int) -> List[str]:
    roles = []
    for i in range(num_nodes):
        if i == 0:
            roles.append("supplier")
        elif i == num_nodes - 1:
            roles.append("retailer")
        else:
            roles.append(f"manufacturer_{i}")
    return roles


def build_graph(
    num_nodes: int, graph_type: str = "chain", seed: Optional[int] = None
) -> nx.DiGraph:
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")

    G = nx.DiGraph()
    roles = make_roles(num_nodes)
    for i, role in enumerate(roles):
        G.add_node(i, role=role)

    if graph_type == "chain":
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)
        return G

    if graph_type == "star":
        for i in range(1, num_nodes):
            G.add_edge(0, i)
        return G

    if graph_type == "skip_chain":
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)
        for i in range(num_nodes - 2):
            G.add_edge(i, i + 2)
        return G

    if graph_type == "scale_free":
        # Barabasi-Albert graph gives heavy-tailed degree distribution.
        # We orient edges from older nodes to newer nodes for a causal flow.
        m = 1 if num_nodes < 4 else 2
        U = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        for u, v in U.edges():
            if u < v:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        return G

    if graph_type == "tiered_scale_free":
        rng = random.Random(seed)

        def _split_middle(nodes: List[int], chunks: int) -> List[List[int]]:
            if not nodes:
                return []
            chunks = max(1, min(chunks, len(nodes)))
            out = []
            size_base = len(nodes) // chunks
            extra = len(nodes) % chunks
            start = 0
            for i in range(chunks):
                step = size_base + (1 if i < extra else 0)
                out.append(nodes[start : start + step])
                start += step
            return [chunk for chunk in out if chunk]

        def _sample_without_replacement(
            candidates: List[int], weights: List[float], k: int
        ) -> List[int]:
            chosen = []
            cand = candidates[:]
            w = weights[:]
            k = min(k, len(cand))
            for _ in range(k):
                total = sum(w)
                r = rng.random() * total
                csum = 0.0
                pick = 0
                for i, wi in enumerate(w):
                    csum += wi
                    if r <= csum:
                        pick = i
                        break
                chosen.append(cand.pop(pick))
                w.pop(pick)
            return chosen

        if num_nodes == 2:
            G.add_edge(0, 1)
            return G

        supplier_tier = [0]
        retailer_tier = [num_nodes - 1]
        middle = list(range(1, num_nodes - 1))

        if len(middle) <= 2:
            middle_tiers = _split_middle(middle, chunks=len(middle))
        elif len(middle) <= 4:
            middle_tiers = _split_middle(middle, chunks=2)
        else:
            middle_tiers = _split_middle(middle, chunks=3)

        tiers = [supplier_tier] + middle_tiers + [retailer_tier]

        for i in range(len(tiers) - 1):
            src_tier = tiers[i]
            dst_tier = tiers[i + 1]

            for u in src_tier:
                weights = [float(G.in_degree(v) + 1) for v in dst_tier]
                max_k = min(len(dst_tier), 2)
                k = 1 if max_k == 1 else rng.randint(1, max_k)
                targets = _sample_without_replacement(dst_tier, weights, k)
                for v in targets:
                    G.add_edge(u, v)

            # Ensure every node in the next tier has support from previous tier.
            for v in dst_tier:
                incoming_from_prev = any(G.has_edge(u, v) for u in src_tier)
                if not incoming_from_prev:
                    u = max(src_tier, key=lambda n: G.out_degree(n))
                    G.add_edge(u, v)

        return G

    raise ValueError(f"Unsupported graph_type: {graph_type}")


def to_message_passing_edge_index(G: nx.DiGraph) -> torch.Tensor:
    undirected_edges: List[Tuple[int, int]] = []
    for u, v in G.edges():
        undirected_edges.append((u, v))
        undirected_edges.append((v, u))
    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    return edge_index
