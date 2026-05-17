# V2 Progress

## Yesterday

- Built `v2/` as the clean working line while keeping `v1_archive/` untouched.
- Added configurable feature ablation for:
  - `health`
  - `exposure`
  - `time_to_recovery`
  - `betweenness_centrality`
- Added config-driven feature variants and feature logging in run metadata.
- Implemented early-warning prediction:
  - changed node-level target from `health at t+1` to `health at t+K`
  - current default tested horizon: `K = 5`

## Node-Level Early Warning Results

- Ran node-level `t+5` prediction on `N20_tiered_scale_free`, seed `7`.
- Full-feature run:
  - baseline final test accuracy: `0.689`
  - graph transformer best observed test accuracy: `0.703`
  - graph transformer final test accuracy: `0.613`
- No-exposure run (`health + time_to_recovery`, with runtime betweenness enabled):
  - baseline final test accuracy: `0.741`
  - graph transformer best observed test accuracy: `0.723`
  - graph transformer final test accuracy: `0.652`
- Takeaway:
  - early-warning alone did not produce a clean stable graph-transformer advantage

## Today

- Implemented Direction 1: structure-level prediction.
- Kept the same node-feature snapshot input:
  - `X[t, node, feature]`
- Added a new graph-level target:
  - `Y_lcc_v1.pt`
  - `Y_lcc[t] = LCC fraction of the healthy-node induced subgraph at t+K`
- Added graph-level trainers:
  - `train_baseline_graph_level.py`
  - `train_graph_transformer_graph_level.py`
- Integrated graph-level runs into:
  - `v2/configs/experiments.yaml`
  - `v2/configs/experiments_multiseed.yaml`
  - `v2/run_experiments.py`
  - `v2/generate_results_summary.py`

## Graph-Level Results

- Ran graph-level `LCC_fraction at t+5` prediction on `N20_tiered_scale_free`, seed `7`.
- Results:
  - baseline graph-level best test MAE: `0.1133`
  - graph transformer graph-level best test MAE: `0.0865`
- Takeaway:
  - this is the first clear `v2` result where the graph transformer outperformed the baseline on a task more directly tied to graph structure

## Current Interpretation

- Node-level prediction remains too easy to approximate from local / engineered signals.
- Structure-level prediction is more aligned with what a graph transformer should help with.
- Current best direction in `v2` is to continue exploring graph-level targets and related structure-aware tasks.
