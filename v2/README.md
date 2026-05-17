# V2 Workspace

Use this folder for the next research direction.

For a clean reproduction guide, see:

- `v2/SETUP_AND_RUN.md`

Suggested practice:

- keep new code, configs, data, and runs inside `v2/`
- leave `v1_archive/` unchanged except when you intentionally revisit the older line
- record major design shifts in the root `SESSION_NOTES.md`

Bootstrap status:

- `v2/configs/experiments.yaml` and `v2/configs/experiments_multiseed.yaml` are copied starters from `v1_archive/`
- these are copies only; `v1_archive/` remains the runnable reference line
- `v2` configs already include a `features:` section for feature ablation with defaults that preserve current behavior
- `v2/run_feature_variants.py` runs the fixed A/B/C/D feature-ablation sweep and writes variant-aware summaries into `v2/runs/results_summary.csv`
- `v2` now also supports graph-level structure prediction with `Y_lcc_v1.pt`, plus config-driven `baseline_graph_level` and `graph_transformer_graph_level` runs
- graph-level `LCC_fraction` experiments are now supported across all four model families: baseline, GCN, Graph Transformer, and Graphormer

Current intent for v2:

- move toward a more graph-oriented setup where structural information is not already heavily pre-encoded into node features
