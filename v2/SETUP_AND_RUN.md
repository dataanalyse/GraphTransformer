# V2 Setup & Run Guide

This guide is intended to let a collaborator run `v2` independently.

## Scope

`v2` contains the second research line built on synthetic directed supply-chain graphs.

Main experiment families:

- node-level early-warning prediction
- structure-level prediction
- propagation-of-change trajectory prediction
- held-out seed generalization
- controlled compute benchmark

## Recommended Environment

The simplest option is to use the existing local virtual environment if it is already available:

```bash
./.venv/bin/python --version
```

If a fresh environment is needed, create one and install the packages used by `v2`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib networkx
```

Notes:

- The current `v2` pipeline is built around plain PyTorch, not PyG, for the main runs.
- `train_graph_transformer_pyg.py` exists, but PyG is not required for reproducing the main reported `v2` results.

## Repo Structure

Important paths:

- `v2/configs/`
- `v2/run_experiments.py`
- `v2/run_structure_target_experiments.py`
- `v2/run_lcc_trajectory_experiments.py`
- `v2/run_graph_level_seed_split.py`
- `v2/run_compute_benchmark.py`
- `v2/runs/final_figures/`

Paper-facing summaries:

- `v2/V2_PROGRESS.md`
- `v2/HELDOUT_RESULTS_SUMMARY.md`

## Core Data Generation

The experiment runners automatically generate the needed data under `v2/data/` if it does not already exist.

If you want to generate one dataset manually:

```bash
./.venv/bin/python v2/simulate_and_build.py \
  --num_nodes 40 \
  --graph_type tiered_scale_free \
  --graph_tag N40_tiered_scale_free \
  --T 200 \
  --seed 1 \
  --prediction_horizon 5 \
  --data_dir v2/data/N40_tiered_scale_free/seed_1
```

## Main Reproduction Commands

### 1. Node-Level + Graph-Level Multiseed Runs

This is the main multiseed experiment entry point:

```bash
./.venv/bin/python v2/run_experiments.py --config v2/configs/experiments_multiseed.yaml
```

This covers:

- node-level early-warning prediction
- graph-level `LCC_fraction`

Outputs go to:

- `v2/runs/`

### 2. Additional Structure-Level Targets

This runs the other graph-level targets:

```bash
./.venv/bin/python v2/run_structure_target_experiments.py --config v2/configs/experiments_multiseed.yaml
```

Targets covered:

- `component_fraction`
- `diameter_fraction`
- `edge_survival_ratio`

### 3. Propagation-of-Change / Trajectory Runs

This runs the `LCC` trajectory experiments:

```bash
./.venv/bin/python v2/run_lcc_trajectory_experiments.py --config v2/configs/experiments_multiseed.yaml
```

Target:

- `Y[t] = [LCC(t+1), ..., LCC(t+5)]`

### 4. Held-Out Seed Generalization

This evaluates training on seeds `1,2,3` and testing on unseen seeds `4,5`.

Example for `LCC_fraction`:

```bash
./.venv/bin/python v2/run_graph_level_seed_split.py --graph_target_key lcc_fraction
```

Other supported values:

- `component_fraction`
- `diameter_fraction`
- `edge_survival_ratio`

## Final Paper Package

The denser final package uses:

- `300` epochs for baseline too
- `eval_every = 5`

To regenerate the full final package:

```bash
./.venv/bin/python v2/run_v2_final_package.py
```

This writes:

- final summaries under `v2/final_runs/`
- paper-facing outputs under `v2/runs/final_figures/`

## Compute Benchmark

To reproduce the controlled compute comparison:

```bash
./.venv/bin/python v2/run_compute_benchmark.py
```

This benchmarks the four model families on the same representative task:

- graph-level `LCC_fraction`
- `N40_tiered_scale_free`
- `seed = 1`
- `300` epochs

Output:

- `v2/runs/final_figures/compute_benchmark_summary.csv`

## Key Paper-Facing Outputs

Representative files:

- `v2/runs/final_figures/final_node_accuracy_vs_graph_size.png`
- `v2/runs/final_figures/final_lcc_fraction_mae_vs_graph_size.png`
- `v2/runs/final_figures/final_lcc_trajectory_mae_vs_graph_size.png`
- `v2/runs/final_figures/final_graph_results_raw_and_percent.csv`
- `v2/runs/final_figures/paper_graph_results_summary_rounded.csv`
- `v2/runs/final_figures/paper_heldout_graph_results_summary_rounded.csv`
- `v2/runs/final_figures/paper_node_results_summary_rounded.csv`
- `v2/runs/final_figures/compute_benchmark_summary_rounded.csv`

## Notes on Reproducibility

- Raw `v2/data/`, `v2/final_runs/`, and `v2/compute_benchmark_runs/` are intentionally not required in git for normal use.
- The repo keeps the code, configs, and paper-facing outputs needed to rerun experiments and verify reported results.
- If exact historical raw runs are needed, they should be regenerated from the scripts/configs above or shared separately as archived artifacts.
