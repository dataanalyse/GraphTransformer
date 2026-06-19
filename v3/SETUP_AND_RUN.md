# V3 Setup & Run Guide

This guide is the clean starting point for running `v3` independently.

## Scope

`v3` begins as a copy of the stable `v2` code scaffold, but it is intended for the next research line rather than for reproducing the finished `v2` paper package.

At the moment:

- the core experiment scripts are present and runnable
- the configs are inherited starter configs
- the exact `v3` research direction will be refined from here

## Recommended Environment

If the existing virtual environment is already available, the simplest check is:

```bash
./.venv/bin/python --version
```

If a fresh environment is needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib networkx
```

Notes:

- the inherited main pipeline is plain PyTorch based
- `train_graph_transformer_pyg.py` is present, but PyG is not required unless `v3` is extended in that direction

## Repo Structure

Important paths:

- `v3/configs/`
- `v3/configs/infrastructure_layers.yaml`
- `v3/inspect_parquet_layers.py`
- `v3/build_asset_inventory.py`
- `v3/simulate_and_build.py`
- `v3/run_experiments.py`
- `v3/run_structure_target_experiments.py`
- `v3/run_lcc_trajectory_experiments.py`
- `v3/run_graph_level_seed_split.py`

Working output locations:

- `v3/data/`
- `v3/runs/`

## Data Setup First

The recommended first move in `v3` is to establish the real infrastructure asset inventory before modifying the simulator.

### 1. Place parquet files under

```bash
v3/data/raw/
```

### 2. Update the layer config

Edit:

```bash
v3/configs/infrastructure_layers.yaml
```

and fill in:

- parquet filename patterns
- county/state column names if available
- id/name/lat/lon column names once known

### 3. Inspect parquet schemas and sample rows

```bash
./.venv/bin/python v3/inspect_parquet_layers.py
```

This writes:

- `v3/data/processed/parquet_layer_inspection.json`

### 4. Build the first-pass asset inventory

```bash
./.venv/bin/python v3/build_asset_inventory.py
```

This writes:

- `v3/data/processed/asset_inventory.parquet`
- `v3/data/processed/asset_inventory_summary.csv`

This is the first network-setup stage: a unified node inventory across infrastructure layers.

## Starter Commands

These are inherited from `v2` and provide a working baseline while `v3` is still being defined.

### Generate one dataset manually

```bash
./.venv/bin/python v3/simulate_and_build.py \
  --num_nodes 40 \
  --graph_type tiered_scale_free \
  --graph_tag N40_tiered_scale_free \
  --T 200 \
  --seed 1 \
  --prediction_horizon 5 \
  --data_dir v3/data/N40_tiered_scale_free/seed_1
```

### Main multiseed runs

```bash
./.venv/bin/python v3/run_experiments.py --config v3/configs/experiments_multiseed.yaml
```

### Additional structure-level targets

```bash
./.venv/bin/python v3/run_structure_target_experiments.py --config v3/configs/experiments_multiseed.yaml
```

### Trajectory experiments

```bash
./.venv/bin/python v3/run_lcc_trajectory_experiments.py --config v3/configs/experiments_multiseed.yaml
```

### Held-out seed generalization

```bash
./.venv/bin/python v3/run_graph_level_seed_split.py --graph_target_key lcc_fraction
```

## Important Caveat

The scripts are ready to run, but `v3` should not yet be treated as a finalized experiment package. Before sharing `v3` results externally, update:

- the configs
- any hard-coded labels or file names that still say `v2`
- the report-generation path once the new research design is set

## Working Style

- use `v3/` for new experiments only
- leave `v2/` unchanged unless you intentionally need a reference check
- record key `v3` design decisions in the root `SESSION_NOTES.md`
