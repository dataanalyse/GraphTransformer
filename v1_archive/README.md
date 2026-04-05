# GraphTransformer Research Sandbox

End-to-end research pipeline for **supply-chain disruption prediction** on synthetic directed graphs.

The project currently supports:
- Data simulation on configurable network topologies (`chain`, `star`, `skip_chain`)
- Tensor generation for next-step node-health prediction
- Baseline model training (logistic regression)
- GCN model training
- Run logging and cross-run aggregation into a single summary CSV

`Graph Transformer` model is planned but not yet implemented in training scripts.

## Pipeline Overview

For each experiment setting (graph size + seed + hyperparameters):

1. Build graph topology
2. Simulate time-series node states (`health`, `exposure`, `time_to_recovery`)
3. Build tensors:
   - `X[t, node, feature]`
   - `Y[t, node] = health at t+1`
4. Train baseline and/or GCN
5. Log model architecture, config/state, metrics, and checkpoint
6. Aggregate all runs into `runs/results_summary.csv` (or fallback file if locked)

## Repository Structure

- `graph_factory.py`: Graph constructors and message-passing edge conversion.
- `simulate_and_build.py`: Simulation + tensor generation + graph image export.
- `train_baseline_v1.py`: Baseline logistic regression trainer.
- `train_gcn_v2.py`: GCN trainer.
- `run_logger.py`: Per-run artifact logging utilities.
- `run_experiments.py`: Orchestrator for sweeps from YAML config.
- `generate_results_summary.py`: Aggregates all run folders into one CSV.
- `configs/experiments.yaml`: Main experiment config.
- `data/`: Generated datasets by graph/seed.
- `runs/`: Training runs and aggregated summaries.

## Environment

Python 3.10+ recommended.

Install core dependencies:

```bash
pip install torch pandas numpy networkx matplotlib pyyaml
```

## Quick Start

Run the configured experiment sweep:

```bash
python run_experiments.py --config configs/experiments.yaml
```

Default config currently runs:
- graph sizes: `3, 5, 7`
- graph type: `chain`
- seed: `7`
- models: baseline + GCN

## Generate Data Only

Example: generate a 7-node chain dataset (no training):

```bash
python simulate_and_build.py \
  --num_nodes 7 \
  --graph_type chain \
  --graph_tag N7_chain \
  --seed 7 \
  --data_dir data/N7_chain/seed_7
```

Outputs in that folder:
- `node_observables.csv`
- `system_observables.csv`
- `X_v1.pt`, `Y_v1.pt`
- `edge_index.pt`
- `graph_meta.json`
- `supply_chain_graph.png`

## Train Individual Models

Baseline:

```bash
python train_baseline_v1.py \
  --seed 7 \
  --graph_type chain \
  --graph_tag N7_chain \
  --data_dir data/N7_chain/seed_7
```

GCN:

```bash
python train_gcn_v2.py \
  --seed 7 \
  --graph_type chain \
  --graph_tag N7_chain \
  --data_dir data/N7_chain/seed_7
```

Both trainers use deterministic seeding and log full run metadata.

## Run Artifacts

Each run writes to:

- `runs/baseline/<timestamp>/...`
- `runs/gcn/<timestamp>/...`

Files include:
- `architecture.txt`
- `run_start.json` (config + dataset stats)
- `metrics.csv` (epoch-level metrics)
- `model_state.pt`
- `run_end.json` (final metrics)
- model-specific probability tensors

## Results Aggregation

Generate unified summary:

```bash
python generate_results_summary.py
```

Primary output:
- `runs/results_summary.csv`

If that file is open/locked (for example in VS Code), fallback output:
- `runs/results_summary_latest.csv`

Important columns for research comparisons:
- `graph_tag`, `graph_type`
- `num_nodes`, `num_edges_physical`, `num_edges_message_passing`
- `seed`
- `last_test_acc`, `best_test_acc`

## Reproducibility Notes

- Use the same `seed` across baseline and GCN for fair initialization randomness control.
- Compare models across **multiple seeds**, not a single seed.
- Keep train/test split consistent (`split_t` in run metadata).
- Record graph metadata (`graph_tag`, node/edge counts) for each run.

## Current Limitations

- Current simulator uses simple stochastic disruption/recovery dynamics.
- Default topology sweep uses chain graphs; scale-free and other realistic structures should be added for richer studies.
- Graph Transformer training script is pending.

## Next Recommended Extensions

1. Add scale-free directed topology support in `graph_factory.py`.
2. Add Graph Transformer model with the same logging interface.
3. Extend config to run multi-seed sweeps (for example 5-10 seeds).
4. Add a reporting script for mean/std metrics by model and graph size.
