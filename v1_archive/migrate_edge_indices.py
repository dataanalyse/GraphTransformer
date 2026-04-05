import json
from pathlib import Path

import torch

from graph_factory import build_graph, to_directed_edge_index, to_message_passing_edge_index


def migrate_dataset_dir(dataset_dir: Path) -> bool:
    meta_path = dataset_dir / "graph_meta.json"
    legacy_edge_path = dataset_dir / "edge_index.pt"
    if not meta_path.exists():
        return False

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    num_nodes = meta.get("num_nodes")
    graph_type = meta.get("graph_type")
    seed = meta.get("sim_params", {}).get("seed", 7)
    if num_nodes is None or not graph_type:
        return False

    graph = build_graph(int(num_nodes), str(graph_type), seed=int(seed))
    directed_edge_index = to_directed_edge_index(graph)
    message_passing_edge_index = to_message_passing_edge_index(graph)

    archive_path = dataset_dir / "edge_index_message_passing_legacy.pt"
    if legacy_edge_path.exists() and not archive_path.exists():
        legacy_edge_index = torch.load(legacy_edge_path)
        torch.save(legacy_edge_index, archive_path)

    torch.save(directed_edge_index, dataset_dir / "edge_index.pt")
    torch.save(directed_edge_index, dataset_dir / "edge_index_directed.pt")
    torch.save(message_passing_edge_index, dataset_dir / "edge_index_message_passing.pt")

    meta["num_edges_directed"] = int(directed_edge_index.shape[1])
    meta["num_edges_message_passing"] = int(message_passing_edge_index.shape[1])
    meta["edge_index_default"] = "edge_index.pt"
    meta["edge_index_directed"] = "edge_index_directed.pt"
    meta["edge_index_message_passing"] = "edge_index_message_passing.pt"
    if archive_path.exists():
        meta["edge_index_message_passing_legacy"] = archive_path.name
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True


def main() -> None:
    migrated = 0
    for meta_path in Path("data").glob("*/seed_*/graph_meta.json"):
        if migrate_dataset_dir(meta_path.parent):
            migrated += 1
    print(f"Migrated {migrated} dataset directories.")


if __name__ == "__main__":
    main()
