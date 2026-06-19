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
- 2026-05-17: Added a standalone reproduction guide for collaborators:
  - `v2/SETUP_AND_RUN.md`
  and linked it from `v2/README.md`. This file covers environment setup, main experiment commands, final package regeneration, and compute benchmark reproduction.

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
- 2026-05-17: Created a clean `v3/` workspace from the stable `v2` scaffold to support the next research line without disturbing the finished `v2` package. Copied core code/config files only, created fresh `v3/data/` and `v3/runs/` working directories with `.gitkeep`, added `v3/README.md`, `v3/SETUP_AND_RUN.md`, and `v3/V3_PLAN.md`, and extended `.gitignore` so `v3` raw data/runs stay out of git by default.
- 2026-05-17: Added `v3/RESEARCH_CONTEXT.md` as the primary free-form note file for capturing the new `v3` research direction, findings, and open questions before implementation changes begin.
- 2026-05-17: Added the first real-data `v3` setup path: `v3/configs/infrastructure_layers.yaml`, `v3/inspect_parquet_layers.py`, and `v3/build_asset_inventory.py`. These support schema inspection of HIFLD-style parquet files and creation of a first-pass normalized asset inventory for Montgomery County, MD.
- 2026-05-17: Added `v3/validate_dependency_graph.py` and generated `v3/data/processed/graph_validation_summary.md` plus `graph_validation_stats.csv`. Key finding: the first dependency graph is structurally plausible, but the cellular tower layer contains 2 duplicated node IDs that collapse multiple telecom assets into shared graph nodes, and the edge list contains 5 duplicate source-target pairs that should be cleaned before simulator work proceeds.
- 2026-05-17: Fixed the EMS coordinate gap by deriving `latitude`/`longitude` from the EMS `bbox` centroid in `v3/build_asset_inventory.py`, rebuilt the inventory/graph, and confirmed that `ems_fire` is now present in `v3/data/processed/dependency_graph_nodes.csv` with 42 nodes and active edges in `dependency_graph_edges.csv`. Refreshed `v3/data/processed/graph_validation_summary.md` and adjusted `v3/validate_dependency_graph.py` so delay validation matches the intended rule semantics (`delay >= 0`, since some support links are immediate). Remaining graph issue is the cellular layer: 10 telecom rows collapse to 2 effective graph nodes because `UniqSysID` is not unique within the Montgomery subset.
- 2026-05-18: Added `v3/standardize_simulation_edges.py` and generated `v3/data/processed/simulation_edges.csv` plus `simulation_edge_semantics_summary.md`. Standardization rules: dependency edges are reversed into failure-flow direction, support edges keep direction, zero-delay links are normalized to delay `1`, and duplicate simulation source-target pairs are aggregated while preserving original provenance in `original_source` and `original_target`.
- 2026-05-28: Refreshed `v3/data/processed/point_asset_inventory.csv` from the EMS-inclusive asset inventory and regenerated both map figures: `v3/runs/figures/montgomery_point_assets_plain.png` and `v3/runs/figures/montgomery_point_assets_basemap.png`. The current map input now includes EMS/fire (42 points).
- 2026-05-30: Added `v3/analyze_dependency_graph.py` to generate pre-simulator graph diagnostics and visuals from `dependency_graph_nodes.csv` and `dependency_graph_edges.csv`. Outputs now include `graph_overview.png`, `graph_dependency_view.png`, `graph_research_summary.md`, and `graph_metrics.csv`. The analysis uses deduplicated `node_id` rows so report rankings reflect the effective graph rather than repeated raw telecom records.
- 2026-05-30: Fixed telecom node identity by configuring the cellular layer to use a composite key (`UniqSysID + latdec + londec`) in `v3/build_asset_inventory.py` and `v3/configs/infrastructure_layers.yaml`. After rebuilding, the dependency graph now has `291` effective nodes with `10` telecom nodes instead of 2 collapsed super-nodes; refreshed `graph_research_summary.md`, `graph_metrics.csv`, and validation outputs accordingly.
- 2026-05-30: Rebuilt `v3/data/processed/simulation_edges.csv` and `simulation_edge_semantics_summary.md` from the corrected 291-node dependency graph. Current simulator edge layer has 723 unique failure-flow edges, 0 invalid node references, 0 duplicate simulation pairs, and telecom-related flow now reflects 10 spatial telecom nodes rather than the earlier collapsed 2-node structure.
- 2026-05-30: Saved `v3/data/processed/hifld_county_telecom_matches.csv` comparing each HIFLD Montgomery telecom point to its nearest county telecom site. All 10 HIFLD points matched within 0.25 km, reinforcing that HIFLD is geographically credible but much sparser than the county permitting inventory.
- 2026-05-30: Switched the forward-looking telecom pipeline away from HIFLD without rebuilding the graph yet. `v3/configs/infrastructure_layers.yaml` now disables `cellular_towers` and points future telecom ingestion to `county_telecom_graph_ready.csv` under `processed_root`. Added `build_county_telecom_inventory.py` outputs for a graph-ready county telecom subset filtered to active latest records and structures `{Tower, Monopole, Water Tank}`. A clean inventory check sees `300` county telecom nodes ready for the next graph rebuild.
- 2026-05-31: Rebuilt the `v3` asset inventory, map export, dependency graph, validation, graph analysis, simulator-edge layer, and point maps using the new `county_telecom` layer instead of HIFLD telecom. Also fixed `export_point_map_data.py` and `build_dependency_graph.py`, which were still hard-coded to `cellular_towers` and silently excluding county telecom. Current graph state: `581` nodes, `1355` edges, one weakly connected component, zero isolated nodes. Node counts: `telecom=300`, `school=211`, `ems_fire=42`, `power=15`, `hospital=11`, `emergency_management=2`. Current point export and map figures now also include the county telecom layer.
- 2026-05-31: Added `v3/simulator_v1.py` as the first deterministic cascade engine over the cleaned Montgomery dependency graph. It regenerates `simulation_edges.csv` in failure-flow direction, seeds a single failed node at `t=0`, propagates delayed weighted impacts over 50 timesteps, and writes scenario metrics plus comparison plots. Initial seeded scenarios use the highest-degree power, telecom, and EMS nodes: `power_plants::56668` (NIH COGENERATION FACILITY), `county_telecom::3.0` (Oakmont Monopole), and `ems_fire::398e79a6-eb99-4af4-b652-7ebd48658a56` (Rockville Volunteer Fire Department - Station 3). Headline outcomes: power failure causes the strongest cascade (`100` failed, `35` degraded, final `LCC=0.8124`), telecom failure remains highly localized (`1` failed, `6` degraded, final `LCC=0.9983`), and EMS failure creates a modest cross-sector cascade (`4` failed, `10` degraded, final `LCC=0.9931`).
- 2026-05-31: Added `v3/simulator_v2.py` as a richer cascade engine with multi-node shocks (`k=3`), stochastic transmission, sector-specific thresholds, and recovery. Outputs are written alongside `v1` with `_v2`-suffixed metrics and `v2_*.png` figures. Current `v2` behavior is much softer than `v1`: early cascades appear, especially for power, but most sectors recover by the end of 50 timesteps. Final outcomes currently look almost fully recovered (`power`: `1` failed / `0` degraded / `LCC=0.9983`; `telecom`: `0` failed / `0` degraded / `LCC=1.0`; `ems`: `0` failed / `0` degraded / `LCC=1.0`). This is useful, but likely indicates recovery is now too strong or transmission too weak for long-horizon persistent degradation experiments.
- 2026-06-05: Tuned `v3/simulator_v2.py` toward more persistent cascades by lowering recovery probabilities, adding minimum failed/degraded dwell times, and introducing lingering damage memory (`damage_load`) that decays slowly over time. Updated `simulator_v2_summary.md`, `_v2` metrics, and `v2_*.png` figures. New outcomes are more balanced: the multi-node power shock now leaves lasting damage (`47` failed, `21` degraded, final `LCC=0.9191`), while telecom and EMS shocks still remain mostly localized (`telecom`: `0` failed / `2` degraded / `LCC=1.0`; `ems`: `1` failed / `1` degraded / `LCC=0.9983`). This tuned `v2` is a better middle ground between overly harsh `v1` and overly forgiving initial `v2`.
- 2026-06-07: Added `v3/plot_consumer_sector_health.py` and generated `v3/runs/figures/v2_consumer_sector_health_comparison.png` to visualize downstream consumer-sector effects (`hospital`, `school`) under the tuned `v2` shock scenarios. Current read: power shocks clearly damage both hospitals and schools, while telecom and EMS shocks have only minor school effects and essentially no visible hospital degradation in the present dependency assumptions.
- 2026-06-07: Added `v3/build_interactive_cascade_html.py` and generated a self-contained interactive cascade viewer plus backing node-state history:
  - `v3/runs/figures/interactive_cascade_v2.html`
  - `v3/data/processed/cascade_node_state_history_v2.csv`
  The viewer supports scenario selection (`power`, `telecom`, `ems`) and timestep scrubbing so the spread of failures/degradation can be inspected spatially over the Montgomery node layout.
- 2026-06-07: Added `v3/run_scenarios_v1.py` and generated the first pilot scenario dataset under `v3/data/processed/pilot_scenarios/` plus analysis plots under `v3/runs/figures/pilot_scenarios/`. The pilot uses 100 scenarios across five shock families (`power=30`, `telecom=30`, `ems=20`, `hospital=10`, `mixed=10`) while keeping `simulator_v2.py` as the frozen core and varying seed nodes, seed count, severity scale, recovery scale, and propagation scale. Current pilot findings: power shocks dominate (`mean min LCC=0.9325`, `mean peak damage=42.9`, recovery times up to `41` timesteps), mixed shocks are the next most damaging, and telecom/EMS/hospital shocks are mostly localized. Structural signals look promising: correlation between mean seed degree and cascade size is `0.833`, and correlation between dependency concentration and minimum LCC is `-0.853`.
- 2026-06-07: Generalized `v3/run_scenarios_v1.py` so it can produce arbitrary scenario batch sizes and output folders, then added:
  - `v3/build_graph_variants.py`
  - `v3/run_baseline_and_sensitivity.py`
  The baseline 500-scenario batch is now written under `v3/data/processed/baseline_model_v1/` with matching figures under `v3/runs/figures/baseline_model_v1/`. Headline baseline findings: power shocks are still the dominant cascade family (`mean min LCC=0.9695`, worst observed min LCC `0.8158`, mean cascade size `17.6`), while telecom/EMS/hospital shocks remain mostly local (`mean cascade size about 2`). The 500-scenario baseline correlations are weaker but still informative: seed degree vs cascade size `0.543`, dependency concentration vs minimum LCC `-0.620`.
- 2026-06-07: Added dependency-sensitivity runs under `v3/data/processed/sensitivity/` for three graph assumptions:
  - `baseline_100`
  - `redundant_100`
  - `dampened_power_100`
  plus summary file `v3/data/processed/sensitivity/dependency_sensitivity_summary.md`. Important methodological finding: the current "redundant" graph makes cascades worse, not better (`mean peak damage=31.27` vs `16.08` baseline; `mean min LCC=0.9577` vs `0.9746` baseline), because the simulator still interprets extra support links as extra exposure rather than true fallback redundancy. By contrast, reducing power-edge weights by 20% dampens average damage (`mean peak damage=13.22`, `mean min LCC=0.9809`). This means the next model-design question is not just scenario scale-up but whether redundancy should be encoded structurally, dynamically, or both.
- 2026-06-07: Added `v3/simulator_v3.py` with three support-set aggregation modes:
- 2026-06-12: Extended `v3/run_redundancy_comparison.py` with a `--timesteps` override, then reran the full `simulator_v3` comparison batch at `t=150` (`2000` scenarios total across `baseline_additive`, `redundant_additive`, `redundant_buffer`, and `dampened_power_additive`). Refreshed downstream outputs in `v3/data/processed/redundancy_v3/`, `v3/data/processed/resilience_factor_model/`, `v3/data/processed/resilience_interaction_model/`, and the updated AUC/sample-curve figures under `v3/runs/figures/redundancy_v3/`. Key takeaway: extending the horizon materially changes recovery- and AUC-style metrics, but it does not change the main minimum-LCC / peak-damage conclusions; those are driven by early collapse dynamics and remain essentially identical. Additive power and mixed scenarios still often fail to recover fully even by `t=150`, while `redundant_buffer` continues to show substantially better recovery behavior.
  - `additive_exposure`
  - `strongest_dependency`
  - `redundancy_buffer`
  `simulator_v3` keeps delayed support propagation and recovery from `v2`, but now tracks delayed perceived support state on each edge and aggregates by support class at the dependent node. The key conceptual change is that `redundancy_buffer` treats multiple same-class supports as fallback capacity rather than additive damage channels.
- 2026-06-07: Added `v3/run_redundancy_comparison.py` and generated a full 2,000-scenario redundancy comparison under `v3/data/processed/redundancy_v3/` with figure `v3/runs/figures/redundancy_v3/redundancy_comparison_plots.png`. Conditions:
  - `baseline_additive`
  - `redundant_additive`
  - `redundant_buffer`
  - `dampened_power_additive`
  Main findings:
  - `redundant_additive` is much worse than baseline (`mean peak damage=92.25`, `mean min LCC=0.8403`), confirming that extra edges under additive logic still mean extra exposure rather than resilience.
  - `redundant_buffer` materially improves resilience relative to both baseline additive and redundant additive (`mean peak damage=31.83`, `mean min LCC=0.9671`).
  - Baseline additive still shows a very strong dependency-concentration signal (`corr(dependency_concentration, min_lcc) = -0.983`).
  - The current 20% dampening of power-edge weights barely changes headline outcomes under additive aggregation, suggesting either threshold saturation or that the power result is robust to this modest weight change.
- 2026-06-11: Added `v3/validate_dependency_concentration.py` and generated a dedicated validation package under `v3/data/processed/dependency_concentration_validation/` plus figures under `v3/runs/figures/dependency_concentration_validation/`. Using the 500-scenario `baseline_model_v1` batch:
  - Overall Pearson(`dependency_concentration`, `min_lcc`) = `-0.620` with p `2.22e-54`
  - Spearman = `-0.504` with p `1.24e-33`
  - After removing the top 10 Cook's-distance leverage/outlier points, Pearson remains `-0.611`
  - In a regression with shock-type controls, the dependency-concentration coefficient remains significant (`beta=-0.354`, 95% CI `[-0.416, -0.293]`, p `9.83e-27`, model R^2 `0.404`)
  - By shock type, the effect is strongest for `ems`, still clear for `power` and `telecom`, and weak/nonlinear for `mixed`
  This supports presenting dependency concentration as a real structural predictor of resilience, but one that is still shock-conditional rather than universal.
- 2026-06-11: Added `v3/model_resilience_factors.py` and generated a multivariate resilience model under `v3/data/processed/resilience_factor_model/` plus figures under `v3/runs/figures/resilience_factor_model/`. The model uses condition-specific:
  - `dependency_concentration`
  - `mean_seed_degree`
  - `mean_propagation_delay`
  - `shock_type`
  - `redundancy_condition`
  to predict `minimum_LCC` across all 2,000 `simulator_v3` comparison runs. Headline results:
  - Model R^2 = `0.827`
  - strongest factor by partial R^2 drop: `dependency_concentration` (`0.168`)
  - second strongest: `redundancy_condition` (`0.087`)
  - much smaller contributions: `seed_node_degree` (`0.002`), `shock_type` (`0.0004`), `propagation_delay` (`0.00004`)
  - standardized numeric effects: dependency concentration `-0.788`, seed degree `-0.098`, propagation delay near zero / not significant
  Practical interpretation: once structural concentration and graph condition are included, raw shock type contributes surprisingly little extra explanatory power for minimum LCC in the current scenario design.
- 2026-06-12: Added `v3/plot_auc_resilience.py` and generated `v3/runs/figures/redundancy_v3/auc_resilience_comparison.png`. This is the dedicated AUC-resilience visual for the 2,000-scenario redundancy comparison and includes:
  - a condition-level boxplot of `auc_resilience`
  - a mean AUC comparison by `condition × shock_type`
- 2026-06-12: Added `v3/plot_sample_resilience_curves.py` and generated:
  - `v3/runs/figures/redundancy_v3/random_sample_lcc_curves.png`
  - `v3/data/processed/redundancy_v3/random_sample_scenarios.csv`
  This gives a reproducible random sample of 12 actual `LCC vs timestep` trajectories from the 2,000-scenario `simulator_v3` comparison dataset, with scenario IDs and summary stats saved for lookup.
  - Same-day fix: the first version accidentally grouped only by `scenario_id`, which repeats across the four graph conditions. That produced ECG-like zig-zag plots by connecting four different condition trajectories inside one panel. The script was corrected to sample and plot unique `condition + scenario_id` runs.
- 2026-06-12: Added `v3/run_long_horizon_samples.py` and generated long-horizon outputs for two severe/moderate power scenarios (`v3_041`, `v3_126`) out to `t=150`:
  - `v3/data/processed/long_horizon_samples/long_horizon_time_series.csv`
  - `v3/data/processed/long_horizon_samples/long_horizon_summary.csv`
  - `v3/runs/figures/long_horizon_samples/long_horizon_power_samples.png`
  Key finding: `t=50` was indeed truncating recovery for additive power cases. Example `v3_126 baseline_additive`: LCC rises from `0.556` at `t=50` to `0.738` at `t=150`; failed nodes drop from `257` to `152`. Recovery remains incomplete, but the longer horizon reveals substantial late rebound that the original 50-step window hid.
- 2026-06-19: First `v3` code push completed on branch `new_v2`. Commit `1e8528e` (`Add v3 infrastructure cascade modeling pipeline`) was created and pushed to `origin/new_v2`. Generated `v3/data/processed/*` and `v3/runs/figures/*` artifacts remain ignored; the remote now contains the full `v3` code/config/report scaffolding, updated `.gitignore`, and continuity notes without the large derived outputs.
- 2026-06-19: A dedicated branch `v3` was created from the synced `new_v2` state and pushed to `origin/v3` so future work can use a branch name that matches the current project phase. Important note: `v3` generated figures and processed data are still not archived in git because `v3/data/*` and `v3/runs/*` remain ignored except for `.gitkeep`.
