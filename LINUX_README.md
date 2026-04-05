# Linux / Ubuntu Run Guide

This project can be run on Ubuntu with a standard Python virtual environment.

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

If PyTorch needs a platform-specific install, use the appropriate Linux wheel first, then run the rest of the requirements.

## 3. Configure experiments

Edit [`configs/experiments.yaml`](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/configs/experiments.yaml):

- set `graph.type`
- set `graph.sizes`
- enable or disable models under `training`

Example:

```yaml
graph:
  type: tiered_scale_free
  sizes: [20, 30]
```

## 4. Run the full pipeline

```bash
python run_experiments.py --config configs/experiments.yaml
```

This will:
- generate datasets
- train enabled models
- write run folders under `runs/`
- update `runs/results_summary.csv`

## Edge format

All generated datasets use **directed** supply-chain edges by default.

For each generated dataset folder, the files are:

- `edge_index.pt` = directed default
- `edge_index_directed.pt` = directed copy
- `edge_index_message_passing.pt` = symmetrized compatibility version

So if you run the standard trainers without overriding `--edge_name`, they use directed edges.

## Running one model directly

Example: Graphormer on an existing dataset

```bash
python train_graphormer_v1.py \
  --seed 7 \
  --graph_type tiered_scale_free \
  --graph_tag N20_tiered_scale_free \
  --data_dir data/N20_tiered_scale_free/seed_7
```

Then refresh the summary:

```bash
python generate_results_summary.py
```

## Notes

- `run_experiments.py` already sets a safe Matplotlib backend/config for headless runs.
- `train_graph_transformer_pyg.py` requires `torch_geometric`, which is not part of `requirements.txt` yet.
- Main comparison file: `runs/results_summary.csv`
