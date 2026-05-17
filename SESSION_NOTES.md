# Session Notes

Date: 2026-03-13
Project: GraphTransformer

## Active Topic
Interpreting WRDS/Compustat supplier-customer related tables for possible supply-chain graph construction.

## Research Core
- The core research objective is supply-chain disruption prediction on directed graphs.
- The main prediction task is next-step node health: given network state at time `t`, predict node health at time `t+1`.
- The disruption process combines:
  - exogenous shocks
  - endogenous propagation through upstream dependency links
  - probabilistic recovery
- The methodological goal is to compare graph-aware models against simpler baselines:
  - logistic regression baseline
  - GCN
  - Graph Transformer planned
- The current empirical bridge to real data is WRDS/Compustat table discovery for constructing more realistic supplier-customer graph edges.
- Current WRDS takeaway: `wrds_seg_customer` is customer exposure/concentration data, not a clean named firm-to-firm edge table.

## Current Findings
- `comp.wrds_seg_customer` is not a clean firm-to-firm supplier-customer edge table.
- It represents customer exposure / segment customer disclosures from Compustat segment reporting.
- Sample customer names are often buckets such as `U.S. Government`, `Commercial`, `Other Government and Defense`, `North America`, and `Europe/Africa`.
- Main columns interpreted so far:
  - `gvkey`: reporting firm
  - `cid`: customer record identifier
  - `cnms`: customer or customer bucket name
  - `ctype`: customer type code
  - `gareac`: area code
  - `gareat`: area type
  - `salecs`: sales tied to that customer or bucket
  - `sid`: segment identifier
  - `stype`: segment type
  - `srcdate`: source/report date
- `comp.wrds_seg_customer` and `comp.seg_customer` appear materially identical in current WRDS results:
  - same row count: `158246`
  - same visible sample rows
  - only notable schema difference observed: `srcdate` vs `datadate`
- `comp.it_r_rltn` is not a supplier-customer relationship table.
- `comp.it_r_rltn` is a small reference table with 32 rows and 2 columns:
  - `itrltncd`
  - `itrltndesc`
- It stores insider-relationship codes such as `CE` = Chief Executive Officer, `CF` = Chief Financial Officer, `D` = Director, `AF` = Affiliated Person.

## Decisions
- Do not treat `wrds_seg_customer` as a direct named supply-chain edge source without additional entity-resolution work.

## Open Questions
- Which WRDS/Compustat table is the best source for actual named customer firms rather than buckets?
- What are the full code meanings for `ctype` and `stype` across the dataset?
- Does the WRDS account include a separate Supply Chain with IDs product or another table with firm-to-firm identifiers?

## Update Log
- 2026-03-13 00: Initial note created. Captured findings from direct WRDS inspection of `comp.wrds_seg_customer` and `comp.seg_customer`.
- 2026-03-13 01: Added the high-level research summary so future sessions can recover the project objective and current WRDS subproblem without restating context.
- 2026-04-02: User preference noted: keep responses shorter, sharper, and less wordy by default.
- 2026-04-04: Project pivoted fully to directed edges by default. `edge_index.pt` is now directed for newly generated datasets.
- 2026-04-04: Current default graph family in `configs/experiments.yaml` is `tiered_scale_free`.
- 2026-04-04: Graphormer currently includes directed betweenness centrality as an additional in-memory node feature during training; `X_v1.pt` on disk is still the original 3 features.
- 2026-04-04: Fresh full rerun was completed for directed `N3/N5/N7_chain`, but current forward path is tiered-scale-free only.
- 2026-04-04: `N20_tiered_scale_free` was generated and run successfully. Latest results:
  - baseline: `0.8642`
  - gcn: `0.8142`
  - graph_transformer: `0.8650`
  - graphormer: `0.8642` with best `0.8708`
- 2026-04-05: Repo was reorganized for the next phase. The completed first research line now lives under `v1_archive/`, while `v2/` is the clean workspace for the next graph-centric direction. Root now keeps only repo-level continuity files plus the two top-level work areas.
- 2026-04-05: For bootstrapping `v2`, the recommended carry-over from `v1_archive` is only reusable config scaffolding: experiment YAML schema shape, seed/graph/simulator/training/run-root sections, and reproducibility conventions. Do not copy over `v1` defaults blindly if they encode the old task assumptions.
- 2026-04-05: `v2/` now has a copied runnable pipeline scaffold from `v1_archive/` plus configurable feature ablation. `simulate_and_build.py` now builds saved `X_v1.pt` dynamically from `use_health`, `use_exposure`, and `use_time_to_recovery`; `train_graphormer_v1.py` still appends `betweenness_centrality` at training time when `use_betweenness` is true so the current default behavior is preserved without changing model internals.
- 2026-04-05: `v2` feature ordering is now explicitly documented and logged as `health`, `exposure`, `time_to_recovery`, `betweenness_centrality`. Training scripts print the final feature list and `X.shape` before training and record `feature_list` in run metadata.
- 2026-04-05: Added `v2/run_feature_variants.py` to run the fixed ablation sweep automatically:
  - A = health only
  - B = health + time_to_recovery
  - C = health + time_to_recovery + exposure
  - D = health + time_to_recovery + exposure + betweenness
- 2026-04-05: All `v2` trainers now append runtime betweenness when enabled, so Variant D is a real input variant for every model without changing model architectures. `v2/generate_results_summary.py` now writes `feature_variant` and `active_features` columns.
- 2026-04-18: Advisor-driven redesign directions to prioritize next: (1) early-warning prediction at horizon `T+K`, (2) propagation-of-change / multi-step response prediction, (3) structure-level graph outcome prediction such as connectivity/LCC/diameter. Best minimal-change path from the current codebase is early warning first, because it reuses the existing simulator outputs, graph objects, and per-timestep training scaffold while only changing label construction and evaluation horizon.
- 2026-04-18: Implemented the first early-warning version in `v2`: `simulate_and_build.py` now supports configurable `prediction_horizon`, with starter configs set to `5`. Labels are now `Y[t, node] = health at t+K`, and trainers / runners / summaries now record `prediction_horizon` explicitly.
- 2026-04-18: Added the first Direction 1 structure-level path in `v2`. Datasets now also save `Y_lcc_v1.pt`, where `Y_lcc[t]` is the largest weakly connected component fraction of the healthy-node induced subgraph at `t+K`. New graph-level trainers were added in `train_baseline_graph_level.py` and `train_graph_transformer_graph_level.py`.
- 2026-04-19: Extended the Direction 1 graph-level `LCC_fraction` path to all four model families in `v2`: baseline, GCN, Graph Transformer, and Graphormer. `run_experiments.py` and the config files now support multiseed graph-level runs over `N=20,40` and seeds `1,2,3`.
- 2026-04-19: Added `v2/generate_report_figures.py` to create report-ready `v2` figures and summary tables from `v2/runs/results_summary.csv`. Outputs now live under `v2/runs/figures/` and are split cleanly into node-level early-warning accuracy plots and graph-level `LCC_fraction` MAE plots, using only the comparable multiseed runs shared across all four model families.
- 2026-04-19: Added a combined report figure `v2/runs/figures/v2_headline_results.png` with a two-panel comparison: node-level early-warning accuracy and graph-level `LCC_fraction` MAE. This is the best single summary figure for the current `v2` research report draft.
- 2026-04-19: Expanded the Direction 1 dataset builder to save three additional graph-level structure targets in `v2`: `component_fraction`, `diameter_fraction`, and `edge_survival_ratio`, all shifted to `t+K` in the same way as `LCC_fraction`. Graph-level trainers are now target-selectable via `--graph_target_key`, and a dedicated sweep script `v2/run_structure_target_experiments.py` was added to run these three targets across all four graph-level model families.
- 2026-04-19: Extended `v2/generate_report_figures.py` so the report plots now cover all four graph-level structure targets: `LCC_fraction`, `component_fraction`, `diameter_fraction`, and `edge_survival_ratio`. New report artifacts are written under `v2/runs/figures/` with per-target CSVs, comparison plots, and training-curve plots.
- 2026-04-19: Added a true held-out-seed graph-level evaluation path in `v2` via `train_graph_level_seed_split.py` and `run_graph_level_seed_split.py`. This trains on full datasets from one seed set and evaluates on unseen seed-generated datasets instead of using the within-seed late-timestep split. First run used `LCC_fraction` with train seeds `1,2,3` and test seeds `4,5`. Latest held-out `best_test_mae` results:
  - `N20`: baseline `0.1299`, GCN `0.1244`, Graph Transformer `0.1177`, Graphormer `0.1146`
  - `N40`: baseline `0.1126`, GCN `0.1239`, Graph Transformer `0.1236`, Graphormer `0.1261`
- 2026-04-19: Held-out-seed evaluation was extended to the other three structure targets and added to the report figures. Current held-out `best_test_mae` results with train seeds `1,2,3` and test seeds `4,5`:
  - `component_fraction`: `N20` baseline `0.0320`, GCN `0.0200`, Graph Transformer `0.0151`, Graphormer `0.0182`; `N40` baseline `0.0395`, GCN `0.0271`, Graph Transformer `0.0265`, Graphormer `0.0261`
  - `diameter_fraction`: `N20` baseline `0.0483`, GCN `0.0455`, Graph Transformer `0.0445`, Graphormer `0.0448`; `N40` baseline `0.0392`, GCN `0.0252`, Graph Transformer `0.0260`, Graphormer `0.0253`
  - `edge_survival_ratio`: `N20` baseline `0.1539`, GCN `0.1465`, Graph Transformer `0.1461`, Graphormer `0.1450`; `N40` baseline `0.1421`, GCN `0.1575`, Graph Transformer `0.1431`, Graphormer `0.1570`
- 2026-04-19: Added combined held-out-seed headline figure `v2/runs/figures/heldout_seed_headline_results.png`, showing all four graph-level structure targets in one four-panel MAE comparison.
- 2026-04-19: Added `v2/HELDOUT_RESULTS_SUMMARY.md` as a concise report-facing summary of the held-out-seed results, including setup, per-target MAE tables, key takeaways, and figure/CSV artifact references.
- 2026-04-26: Added a minimal Directive #3 path in `v2` for propagation-of-change via future `LCC` trajectory prediction. `simulate_and_build.py` now saves `Y_lcc_traj_v1.pt`, where `Y_lcc_traj[t] = [LCC(t+1), ..., LCC(t+K)]`, and metadata records the trajectory semantics. Added `train_baseline_graph_trajectory.py` and `train_graph_transformer_graph_trajectory.py`. First smoke run on `N20_tiered_scale_free`, `seed=1`, `K=5` completed successfully:
  - baseline trajectory: final `test_mae = 0.0867`
  - graph transformer trajectory: best observed `test_mae = 0.0772` at epoch 150, final `test_mae = 0.0861`
  This is a lightweight but usable implementation of Directive #3 without changing the simulator mechanics.
- 2026-04-26: Extended the minimal Directive #3 trajectory path to all 4 models and ran a full multiseed sweep (`N=20,40`, seeds `1,2,3`) using the target `Y[t] = [LCC(t+1), ..., LCC(t+5)]`. Added:
  - `v2/train_gcn_graph_trajectory.py`
  - `v2/train_graphormer_graph_trajectory.py`
  - `v2/run_lcc_trajectory_experiments.py`
  - `v2/generate_trajectory_report_figures.py`
  Results summary (`mean best_test_mae` across seeds):
  - `N20`: baseline `0.1392`, GCN `0.1155`, Graph Transformer `0.1161`, Graphormer `0.1159`
  - `N40`: baseline `0.1428`, GCN `0.1067`, Graph Transformer `0.0910`, Graphormer `0.0964`
  Artifacts:
  - `v2/runs/figures/trajectory_mae_vs_graph_size.png`
  - `v2/runs/figures/trajectory_training_curves_N20_tiered_scale_free.png`
  - `v2/runs/figures/trajectory_training_curves_N40_tiered_scale_free.png`
  - `v2/runs/figures/trajectory_results_by_model_size.csv`
  - `v2/runs/figures/trajectory_report_summary.md`
- 2026-04-27: Built a pilot `final_figures` path to test denser x-axis sampling and normalized y-axis presentation without touching existing report artifacts. Added:
  - `v2/generate_final_figures_pilot.py`
  - `v2/run_finalpilot_lcc_n40.py`
  Pilot setting:
  - graph-level `LCC_fraction`
  - `N=40`
  - all 4 models
  - seeds `1,2,3`
  - `epochs = 300` for all models
  - `eval_every = 5`
  New artifacts:
  - `v2/runs/final_figures/pilot_lcc_n40_dense_eval_curves.png`
  - `v2/runs/final_figures/pilot_lcc_n40_dense_eval_summary.csv`
  Pilot takeaway:
  - denser x-axis clearly reveals that most models improve sharply before roughly `40-100` epochs, then flatten or oscillate
  - normalized percent error is interpretable here because `LCC_fraction` is positive and bounded
  - best epochs by mean MAE in the pilot were: baseline `165`, GCN `40`, Graph Transformer `240`, Graphormer `105`
- 2026-05-05: Launched a dedicated overnight full final-package run in the background using `v2/run_v2_final_package.py`, with output redirected to `v2/final_runs/final_package_nohup.log`. Background PID at launch: `48070`.
- 2026-05-08: Added `v2/generate_clean_network_figure.py` and exported cleaner layered synthetic network figures for paper use:
  - `v2/runs/final_figures/clean_network_N20_tiered_scale_free.png`
  - `v2/runs/final_figures/clean_network_N40_tiered_scale_free.png`
  These replace the old squished spring-layout network PNGs as the recommended paper figures.
- 2026-05-08: Added `v2/generate_task_design_diagram.py` and exported `v2/runs/final_figures/task_design_diagram.png` for the Methods section. The figure shows one shared input snapshot `X[t]` feeding three target formulations: node health at `t+K`, graph metric at `t+K`, and multi-step trajectory over `t+1...t+K`.
- 2026-05-09: Added reduced journal-facing structure-results tables:
  - `v2/runs/final_figures/paper_graph_results_summary.csv`
  - `v2/runs/final_figures/paper_graph_results_summary_rounded.csv`
  These trim the master graph-results CSV down to target, graph size, model family, MAE, MAE std, and normalized percent error.
- 2026-05-09: Added reduced journal-facing held-out structure-results tables:
  - `v2/runs/final_figures/paper_heldout_graph_results_summary.csv`
  - `v2/runs/final_figures/paper_heldout_graph_results_summary_rounded.csv`
  Because the held-out evaluation is a single train/test seed split, these tables omit standard deviation columns and keep only target, graph size, model family, MAE, and normalized percent error.
- 2026-05-10: Added reduced journal-facing node-level early-warning tables:
  - `v2/runs/final_figures/paper_node_results_summary.csv`
  - `v2/runs/final_figures/paper_node_results_summary_rounded.csv`
  These keep graph size, model family, mean best test accuracy, and standard deviation.
- 2026-05-11: Added a separate controlled compute benchmark that does not alter existing result packages. New script:
  - `v2/run_compute_benchmark.py`
  It benchmarks the 4 model families on the same representative task (`graph_level_lcc_fraction`, `N40_tiered_scale_free`, `seed=1`, `300` epochs) and writes:
  - `v2/runs/final_figures/compute_benchmark_summary.csv`
  - `v2/runs/final_figures/compute_benchmark_summary_rounded.csv`
  Reported fields: model family, best epoch, best test MAE, total runtime in seconds, seconds per epoch, parameter count, and run path.

## Next Step Candidates
- Compare `wrds_seg_customer` with other candidate tables in the catalog.
- Profile distinct values of `ctype`, `stype`, and `cnms`.
- Search for tables with explicit customer firm identifiers instead of broad customer categories.
- Search WRDS libraries for separate supply-chain / customer-ID products beyond `comp`.

## Current Working Strategy
- Do not start from the full WRDS universe.
- Start from a seed-centric ego network:
  - choose one focal public firm
  - collect its direct suppliers/customers
  - expand outward by breadth-first search to depth 3 or 5
- Treat the first research product as a filtered subgraph construction pipeline, not a full-market graph.
- For synthetic experiments going forward, default to `tiered_scale_free`.
- Keep responses short and direct unless more detail is requested.

## Model Implementation Note
- The current Graph Transformer in the repo is implemented in plain PyTorch, not PyTorch Geometric.
- It uses `torch.nn.MultiheadAttention` with a graph-derived attention mask so each node can attend only to graph neighbors plus itself.
- The current GCN is also hand-implemented in plain PyTorch using normalized adjacency multiplication, not PyG layers.
- The next model-design step is to replace the current masked-attention implementation with a graph-native transformer architecture while preserving the same next-step node-health prediction task.
- A new PyG-based trainer scaffold has been added using `torch_geometric.nn.TransformerConv` in `train_graph_transformer_pyg.py`.
- The current `.venv` does not yet have `torch_geometric` installed, so the new trainer is ready but not runnable until that dependency is added.
- A separate Graphormer-style trainer has been added in `train_graphormer_v1.py`.
- The current Graphormer-style positional/structural encoding uses:
  - learned in-degree embeddings
  - learned out-degree embeddings
  - learned shortest-path-distance attention bias
- The current Graphormer trainer now also appends directed betweenness centrality as an additional node feature across all timesteps.
- This version uses full node-to-node attention with structural bias, rather than hard neighbor masking.
