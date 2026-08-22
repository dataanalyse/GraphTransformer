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
- 2026-06-28: Verified the Seref follow-up artifacts are present under `v3`: clipped regression figures in `v3/runs/figures/regression_clipped/`, OIPT slope/bias interpretation in `v3/data/processed/oipt_slope_bias_interpretation.md`, and the IP capability decision tree in `v3/runs/figures/decision_tree/`. Current status for the three advisor items: regression replot done, slope-vs-bias interpretation done, decision tree done.
- 2026-07-04: User preference: when asked to check a paper, start with Zotero as the primary lookup location because that is where saved papers/PDFs are expected to live.
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
 - 2026-08-22: County source-provenance clarification for the real-network layers.
   Montgomery current base graph:
   - HIFLD: `power_plants`, `hospitals`, `ems_fire`, `local_eoc`
   - county telecom inventory / permitting-derived telecom sites: `county_telecom` (active Montgomery telecom layer)
   - NCES-style public school dataset: `public_schools`
   - public road GIS layer: `primary_roads`
   - HIFLD `cellular_towers` remains in config but is disabled for Montgomery
   Montgomery hierarchy additions:
   - HIFLD transmission-lines parquet from the Source Cooperative mirror
   - transmission substations inferred from line endpoints and cross-matched with OSM
   - OSM local/distribution substations
   - OSM telecom exchange / office / data-center facilities for the telecom hierarchy
   - Census / TIGER county boundary for clipping and inclusion tests
   Garrett current flat graph:
   - HIFLD: `power_plants`, `hospitals`, `ems_fire`, `local_eoc`, `cellular_towers`
   - NCES-style public school dataset: `public_schools`
   - public road GIS layer: `primary_roads`
   - county telecom layer is disabled for Garrett
   Garrett hierarchy additions:
   - HIFLD transmission-lines parquet from the same mirrored source family
   - transmission substations inferred from transmission topology and matched to OSM where possible
   - OSM substations for Garrett transmission/substation inspection
   - OSM telecom facilities for Garrett exchange/backbone telecom nodes
   - Census / TIGER Garrett boundary for clipping and validation
- 2026-08-16: Added service-weighted resilience as a new read-only outcome layer in `v3/service_weighted_resilience.py`, with a one-scenario demo runner in `v3/run_weighted_resilience_demo.py`. The metric uses per-type weights `{hospital 10, emergency_management 10, ems_fire 8, telecom 5, school 2, power/transmission/distribution 0}` and computes a normalized 0-100% weighted service percentage over time from existing `sector_health_*` simulator outputs. Demo output for Montgomery hierarchy `redundant_additive`, no-intervention, DICKERSON seed is under `v3/data/processed/montgomery_weighted_resilience_demo/`. Headline demo result: weighted resilience `4.6317%`, with per-timestep total service weight dominated by telecom (`1500`) vs school (`422`), EMS (`336`), hospital (`110`), emergency management (`20`).
- 2026-08-17: Completed a read-only telecom-source audit across HIFLD, county, and OSM. HIFLD Montgomery telecom is only `10` sparse wireless-structure records (`MTOWER/UPOLE/POLE/TANK/TOWER`), not central-office backbone facilities. County graph-ready telecom remains `300` wireless support sites (`144` monopoles, `143` towers, `13` water tanks), which collapse to `264` site clusters under a 100 m co-location rule. OSM in-county communications features returned `177` total objects, with `8` exchange-like candidates (`5` telecom=exchange, `1` telecom=data_center, `2` office=telecommunication). Proposed read-only two-tier telecom hierarchy diagnostics show those `8` exchange-like OSM nodes can anchor the `264` tower clusters; nearest-exchange fan-out ranges from `10` to `79` raw tower nodes per exchange candidate.
- 2026-08-18: Built the first Montgomery telecom-hierarchy graph variants in `v3/data/processed/montgomery_telecom_hierarchy_graph_variants/`. New saved OSM telecom exchange extraction lives in `v3/data/processed/osm_montgomery_telecom/` (`8` in-county exchange-like facilities). Rebuilt hierarchy replaces the old flat `telecom` layer with `256` deduplicated tower clusters plus `8` exchange nodes; note this is slightly lower than the earlier hand-counted `264`, likely because the implemented 100 m clustering merges a few close triplets (e.g. Blair HS / Silver Spring VFD / Fire Station 16) into single sites. Final baseline-additive hierarchy counts: `681` nodes, `1547` structural edges, `1478` sim edges. Node counts: `school=211`, `hospital=11`, `ems_fire=42`, `emergency_management=2`, `power=15`, `transmission_substation=61`, `distribution_substation=75`, `telecom_tower=256`, `telecom_exchange=8`.
- 2026-08-18: Dependency-direction audit for the telecom hierarchy is saved in `downstream_dependency_by_type.csv` and `downstream_dependency_edge_matrix.csv` under `v3/data/processed/montgomery_telecom_hierarchy_graph_variants/`. Seedable node types on the rebuilt Montgomery graph are: `distribution_substation`, `emergency_management`, `ems_fire`, `power`, `telecom_exchange`, `telecom_tower`, `transmission_substation`. As expected, `school` and `hospital` are terminal. `emergency_management` has downstream dependents (`ems_fire`, `hospital`), and `ems_fire` has downstream dependents (`hospital`, `school`).
- 2026-08-18: Updated service-weighted resilience defaults in `v3/service_weighted_resilience.py` to the life-safety scheme `{hospital 100, emergency_management 100, ems_fire 30, telecom_exchange 40, telecom_tower 2, school 3}` while retaining legacy `telecom=5` for old flat graphs. On the new Montgomery telecom hierarchy, total per-timestep service weights are: `ems_fire=1260`, `hospital=1100`, `school=633`, `telecom_tower=512`, `telecom_exchange=320`, `emergency_management=200`. Weight table saved to `v3/data/processed/montgomery_telecom_hierarchy_graph_variants/service_weight_by_type.csv`.
- 2026-08-18: Stage-4 sanity check completed in `v3/data/processed/montgomery_telecom_hierarchy_sanity/` using one broad-seed scenario on `Germantown Substation` (`distribution_substations::way_168420825`, `62` outgoing simulation edges). No intervention, same seed, compared `baseline_additive` vs `redundant_buffer` on the weighted metric. Results: `baseline_additive weighted_resilience = 91.2255%`, `min_lcc = 0.8561`, `t_min = 3`; `redundant_buffer weighted_resilience = 91.3416%`, `min_lcc = 0.8678`, `t_min = 3`. The requested sanity condition passed: buffer > baseline on weighted resilience. Curve plot saved as `weighted_resilience_curve_comparison.png`.
- 2026-08-18: Re-ran the Montgomery telecom-hierarchy sanity check with an UPSTREAM seed above the redundancy layer at `transmission_substations::BELLS_MILL` (`BELLS MILL`), output folder `v3/data/processed/montgomery_telecom_hierarchy_sanity_transmission_bells_mill/`. This is the decisive test for the `transmission -> distribution` redundancy. Weighted threatened downstream service from this seed is `3645 / 4025 = 90.56%` of all weighted service (`522` weighted downstream nodes), versus about `9%` for the earlier `Germantown Substation` distribution-level seed. Results diverged sharply in the expected direction: `baseline_additive weighted_resilience = 12.2617%`, `min_lcc = 0.0162`, `t_min = 14`; `redundant_buffer weighted_resilience = 50.0962%`, `min_lcc = 0.5404`, `t_min = 51`. This confirms the upstream redundancy is functioning and meaningfully delaying / softening collapse when the failure is seeded above the buffered links.
- 2026-08-19: Recovery audit for the Montgomery telecom hierarchy completed before the overnight broad-seed runs. In `v3/simulator_v3.py`, recovery is a real node-restoration mechanism in `recovery_step(...)`: failed nodes can recover to degraded with per-type probability `recovery_failed` after `min_failed_duration` and only if `damage_load <= 0.20`; degraded nodes can recover to healthy with probability `recovery_degraded` after `min_degraded_duration` and only if `damage_load <= 0.12`. `recovery_scale` multiplies both per-type recovery probabilities in `v3/run_scenarios_v1.py::apply_parameter_overrides`. Replay of the saved Bells Mill transmission-seed sanity case found zero realized recoveries in both `baseline_additive` and `redundant_buffer`. Diagnosis: recovery is not missing, but in this extreme scenario it was effectively gated off by the low-damage eligibility thresholds. In baseline, no node ever became eligible; in buffer, at most one failed node became eligible and deterministic recovery still did not fire. Interpretation for the overnight run: recovery exists in code but is largely inactive in the most severe cascade phase, so results should be read as mostly cascade-phase resilience rather than substantial restoration dynamics.
- 2026-08-19: Added dedicated Montgomery telecom-hierarchy broad-seed weighted runners:
  - `v3/montgomery_telecom_weighted_utils.py`
  - `v3/run_montgomery_telecom_broad_semantic_weighted.py`
  - `v3/run_montgomery_telecom_broad_policy_weighted.py`
  - `v3/run_montgomery_telecom_weighted_overnight.sh`
  Smoke tests completed successfully:
  - semantic smoke output: `v3/data/processed/_smoke_montgomery_telecom_broad_semantic_weighted/`
  - policy smoke output: `v3/data/processed/_smoke_montgomery_telecom_broad_policy_weighted/`
  Smoke semantic ordering on weighted outcome was sensible (`redundant_buffer > baseline_additive ≈ redundant_additive`), and smoke policy completed the full path including priority-union exhaustive search and regenerated shatter-impact.
- 2026-08-19: Full overnight Montgomery telecom-hierarchy run launched in persistent exec session `24733` at about `2026-08-18 20:23 EDT` using `v3/run_montgomery_telecom_weighted_overnight.sh`. This wrapper runs sequentially:
  1. `v3/run_montgomery_telecom_broad_semantic_weighted.py` with `500` broad-seed scenarios to `v3/data/processed/montgomery_telecom_hierarchy_broad_semantic_weighted_N500/`, log `v3/logs/montgomery_telecom_broad_semantic_weighted_N500.log`
  2. `v3/run_montgomery_telecom_broad_policy_weighted.py` with `30` broad-seed paired scenarios, `B=3`, `top_k=15`, and `impact_runs=1000` to `v3/data/processed/montgomery_telecom_hierarchy_redundant_additive_policy_weighted_N30_B3/`, log `v3/logs/montgomery_telecom_redundant_additive_policy_weighted_N30_B3.log`
  Master log: `v3/logs/montgomery_telecom_weighted_overnight_master.log`
- 2026-08-12: Added read-only OSM substation audit for Montgomery in `v3/fetch_osm_montgomery_substations.py`. Outputs in `v3/data/processed/osm_montgomery_substations/` currently show 260 OSM `power=substation` features within Montgomery + 10 km, with 86 matching existing HIFLD-derived transmission substations within 1 km and 174 unmatched. After normalizing OSM voltage tags from volts to kV, counts are: `distribution=40`, `minor_distribution=56`, `transmission=81`, `industrial=14`, `generation=1`, `unknown=68`; 47 features have voltage <100 kV and 43 <69 kV. The audit suggests OSM likely fills part of the missing local/distribution substation layer that HIFLD transmission-line endpoints do not capture.
- 2026-08-12: Added a second read-only OSM topology audit in `v3/trace_osm_distribution_upstream.py` with outputs under `v3/data/processed/osm_distribution_upstream_audit/`. It attempts to connect the 75 OSM candidate local/distribution substations to the HIFLD-backed transmission backbone using observed OSM `power=line|cable|minor_line` geometry and substation way/relation geometry, without forcing nearest-substation joins. Result: `0 / 75` candidates traced to the HIFLD-matched backbone and all 75 remained unresolved under strict observed topology. Important nuance: many candidates are still geographically close to OSM electrical features (`6` within `10 m`, `22` within `25 m`, `30` within `50 m` of the nearest line/cable), so the limitation appears to be missing explicit topological joins in OSM rather than absence of nearby electrical infrastructure.
- 2026-08-12: Built a read-only candidate hierarchy validator in `v3/build_final_power_hierarchy_validation.py` with outputs under `v3/data/processed/final_power_hierarchy_validation/`. Using the established spatial abstraction `power plant -> HIFLD transmission -> OSM local/distribution -> infrastructure`, the hierarchy materially reduced the old transmission-only concentration problem: BELLS MILL direct infrastructure fan-out drops from `116` to `0`, the new maximum infrastructure fan-out is `66` at `Germantown Substation`, and the median infrastructure assignment distance drops from `4.540 km` to `2.190 km`. But the validator still returned `FAIL` because `44` infrastructure-to-distribution assignments exceed `10 km` (mostly telecom/remote fringe assets) and `Hunting Hill Substation` captures `3/11` hospitals (`27.3%`, above the 20% sector threshold). Distribution->transmission distances looked acceptable (`max 11.024 km`, `0` over `20 km`).
- 2026-08-12: Added a final minimal cleanup pass in `v3/finalize_power_hierarchy_cleanup.py`, still read-only, writing `final_infrastructure_distribution_assignments.csv` and `final_cleanup_report.md` into `v3/data/processed/final_power_hierarchy_validation/`. Result: no assignments changed. All 44 infrastructure nodes above `10 km` were already attached to their nearest eligible OSM distribution substation, and none had a materially better alternative under the user’s “do not rebalance, do not move just to reduce fan-out” rule. The hierarchy is now marked `READY` under the requested criteria: largest substation share stays `11.7%`, no assignments exceed `20 km`, and no infrastructure nodes are unassigned.
- 2026-08-12: Ran the final controlled A/B timing experiment in `v3/run_final_power_timing_experiment.py`, outputs under `v3/data/processed/final_power_timing_experiment/`. Structural checks passed: the final hierarchy has `0` direct `school/hospital/telecom/ems_fire -> power` edges, with `61` transmission substations, `75` total OSM distribution substations (`42` serving infrastructure), `75` transmission->distribution edges, and `564` distribution->infrastructure edges. Across the 10 corrected Montgomery power scenarios, ORIGINAL had `median first_school_failure_t = 2`, `median t_min_lcc = 4`, `mean min_lcc = 0.7556`, `mean peak_damage_nodes = 142.0`. FINAL_HIERARCHY shifted timing later (`median first_school_failure_t = 5`, `median t_min_lcc = 11`, `mean t_min_lcc = 8`) but also increased damage sharply on connected seeds (`mean min_lcc = 0.3517`, `mean peak_damage_nodes = 440.5`). Important nuance: outcomes are bimodal because 3 of the 9 unique power seed plants in the scenario set are unmatched to the HIFLD transmission backbone (`power_plants::65945`, `::65610`, `::61608`), so their FINAL_HIERARCHY scenarios are nearly inert (`t_min_lcc = 0`, `min_lcc = 0.9735`, `peak_damage_nodes = 1`), while matched scenarios collapse severely and late (e.g. `power_plants::62910` gives `t_min_lcc = 11`, `min_lcc = 0.0976`, `peak_damage_nodes = 618`). Relative to the earlier transmission-only substation experiment (`median t_min_lcc = 8`, `mean min_lcc = 0.3005`, `mean peak_damage_nodes = 441.3`), FINAL_HIERARCHY preserves the later timing and slightly improves mean damage metrics overall, but mostly because of the unmatched-seed no-op cases.
- 2026-08-11: Garrett delay diagnostic summary from saved policy-refresh outputs: the recalibrated delay runs (`garrett_delay12_refresh_N30_B3`, `garrett_delay4_refresh_N30_B3`) changed timing more than outcomes. Mean ladder values stayed close to the original run (`none 0.4731 -> 0.4731 -> 0.4769`, `priority 0.5090` in all three, `best 0.4878 -> 0.4878 -> 0.4917`). The 4h delays pushed `t_min` later overall (`mean 4.93`, max `24`) and reduced the intervention-impossible share for power shocks (`30%` at 12h to `10%` at 4h), but Garrett’s graph still has all `11` schools directly dependent on power via one-hop `school -> power` edges. In the saved power scenarios, seeded power sets expose `0` to `6` schools immediately, so the main LCC drop is still dominated by direct power-to-school failure rather than the slower hospital/EMS tiers.
- 2026-08-11: Overnight `redundant_additive` policy sweep launched after automated preflight passed. Checks:
  - Garrett redundant_additive counts: `52` nodes, `170` edges/sim-edges; HIGH/MED/LOW capacities resolved to `3/1/0`.
  - Montgomery redundant_additive counts: `581` nodes, `2988` edges/sim-edges; HIGH/MED/LOW capacities resolved to `54/16/5`.
  - Delay4 clip floor did not bind in either county (`effective_delay_clip_count = 0`, `clip_changed_any_delay = false`).
  - Delay4 latest new-failure timestep stayed well below horizon `300`: Garrett max `24`, Montgomery max `36`.
  - Garrett delay4 redundant precheck `t_min`: min `0`, median `2`, mean `7.67`, max `24`; intervention-possible `18/30`.
  - Montgomery delay4 redundant precheck `t_min`: min `0`, median `2`, mean `12.33`, max `36`; intervention-possible `25/30`.
  - Launch sequence uses `./.venv/bin/python` and logs to `v3/logs/`.
  - Long-running launch session id: `59756`.
  - Sequence: Garrett original redundant_additive refresh -> Garrett shatter-impact -> Garrett redundant_additive delay4 -> Montgomery original redundant_additive refresh -> Montgomery shatter-impact -> Montgomery redundant_additive delay4.
- 2026-08-11 morning status check: completed artifacts exist for:
  - `garrett_redundant_additive_positive_indegree_refresh_N30_B3`
  - `garrett_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3`
  - `garrett_redundant_additive_delay4_refresh_N30_B3`
  - `montgomery_redundant_additive_positive_indegree_refresh_N30_B3`
  - `montgomery_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3`
  Garrett logs show:
  - original + shatter-impact + delay4 all completed
  - Garrett redundant-additive original ladder: no-intervention `0.7288`, priority `0.7872`, best-combinatorial `0.7378`, shatter_impact_topk `0.7365`
  Montgomery logs show:
  - original refresh completed with old-style shatter placeholder; mean min_lcc new: none `0.8245`, priority `0.8394`, best `0.8320`, shatter_topk `0.8245`
  - impact-based shatter rerun completed; ladder: no-intervention `0.8245`, priority `0.8394`, best-combinatorial `0.8320`, shatter_impact_topk `0.8246`
  - final arm `montgomery_redundant_additive_delay4_refresh_N30_B3` appears incomplete/stalled: only preview, graph build, baseline priority reference, and candidate selection files exist; no exhaustive results, no shatter-impact subfolder, no `delay4_summary.json`, and log file remained zero bytes at last check.
- 2026-08-09: Added delay-calibrated config copies for the current county graphs:
  - `v3/configs/infrastructure_layers_delay12.yaml`
  - `v3/configs/infrastructure_layers_garrett_delay12.yaml`
  and a dedicated runner:
  - `v3/run_delay12_policy_refresh.py`
  This workflow rebuilds the current graph structure with delay-only changes, reuses the corrected positive-indegree 30-scenario metadata, reruns the four-policy ladder (`none`, `priority`, `best-combinatorial`, `shatter_impact_topk`), and logs paired differences plus `t_min` / intervention-possible diagnostics.
- 2026-08-09: Garrett delay-12 run completed at `v3/data/processed/garrett_delay12_refresh_N30_B3/`.
  - Graph structure stayed identical to current Garrett: `52` nodes, `85` edges, `85` simulation edges.
  - Delay caveat: `34` YAML zero-delay edges are still clipped to effective delay `1` by `v3/standardize_simulation_edges.py`, so the school-delay `0` request is only partially realized under current simulator semantics.
  - Latest timestep with any new failure across the 30-scenario replay: `8`.
  - Replay `t_min` distribution under priority-B: min `0`, median `1.0`, mean `1.9`, max `8`.
- 2026-08-13: Diagnosed the positive AUC-vs-concentration pilot slopes on the Montgomery hierarchy in a read-only pass. The old `dependency_concentration` axis in `v3/run_scenarios_v1.py` still uses seed simulation outdegree divided by total sim edges, which no longer reflects consumer exposure once power flows through substations. Saved diagnostic outputs to `v3/data/processed/montgomery_hierarchy_concentration_axis_diagnostic/`:
  - `scenario_reach_metrics.csv`
  - `auc_slope_axis_comparison.csv`
  - `old_vs_new_axis_correlation.csv`
  Findings:
  - old-axis pilot slopes stay positive: baseline `+24.3209`, redundant_additive `+27.4055`, redundant_buffer `+10.2693`
  - replacing the x-axis with true downstream consumer reach fraction flips the slopes negative:
    - baseline_additive `-0.9612`, `r=-0.99995`
    - redundant_additive `-0.9499`, `r=-0.99995`
    - redundant_buffer `-0.3560`, `r=-0.5949`
  - old vs new axis correlation is weak and slightly negative in all three conditions (`about -0.22`), confirming the old concentration variable is no longer the right fragility axis on the hierarchy.
- 2026-08-14: Reworked the hierarchy fragility axis to the distribution-substation service layer, as requested. New read-only outputs are in `v3/data/processed/montgomery_hierarchy_distribution_concentration_diagnostic/`:
  - `distribution_substation_consumer_fanout.csv`
  - `distribution_fanout_summary.csv`
  - `scenario_distribution_concentration.csv`
  - `auc_slope_vs_distribution_concentration.csv`
  Definition used: for each scenario and condition, find the first downstream distribution substations reached from each seed in the simulation graph, sum their direct consumer fan-outs, and divide by total consumers. Results:
  - distribution-substation fan-out now clearly varies: `75` substations total, `42` with positive consumer load, median fan-out `2`, mean `7.52`, max `66`, `23` unique fan-out values
  - AUC slopes vs this distribution-layer concentration are all negative:
    - baseline_additive `-4.8511`, `r=-0.6884`
    - redundant_additive `-2.7543`, `r=-0.7877`
    - redundant_buffer `-2.0395`, `r=-0.9260`
  - concentration also varies across scenarios: `6` unique scenario-level values per condition; power scenarios now span substantial spread instead of collapsing to one nearly flat value.
- 2026-08-14: Ran the full Montgomery hierarchy semantic experiment on POWER scenarios only using the distribution-substation concentration axis. New output folder: `v3/data/processed/montgomery_hierarchy_power_semantic_N500/`. Setup:
  - `500` power scenarios
  - `4` bulk seedable plants: `DICKERSON`, `MONTGOMERY COUNTY OAKS LFGE PLANT`, `MONTGOMERY COUNTY RESOURCE RECOVERY`, `NIH COGENERATION FACILITY`
  - `14` possible 1/2/3-seed combinations
  - `7` distinct distribution-concentration values available from the seed space before scale variation
  - same 7 distinct concentration values were realized in the full run for all three conditions
  Full-run AUC slopes vs distribution concentration:
  - baseline_additive: slope `-0.045510`, intercept `0.130967`, `r=-0.823439`
  - redundant_additive: slope `-0.029526`, intercept `0.045487`, `r=-0.807787`
  - redundant_buffer: slope `-1.965910`, intercept `0.987930`, `r=-0.775438`
  Realized concentration spread:
  - baseline_additive: min `0.010638`, median `0.108156`, max `0.202128`
  - redundant_additive / redundant_buffer: min `0.108156`, median `0.235816`, max `0.445035`
  Main artifacts:
  - `seed_combo_distribution_concentration_preview.csv`
  - `distribution_substation_consumer_fanout.csv`
  - `scenario_metadata_power_only.csv`
  - `scenario_summary_metrics_power_only.csv`
  - `scenario_distribution_concentration.csv`
  - `auc_fragility_slopes_distribution_axis.csv`
  - `power_semantic_report.md`
- 2026-08-15: Reconciled the pilot vs full-run slope inversion for the Montgomery hierarchy power-only semantic experiment. New diagnostic folder: `v3/data/processed/montgomery_hierarchy_power_semantic_diagnostics_2026-08-15/`. Key saved artifacts:
  - `slope_reconciliation_table.csv`
  - `baseline_auc_by_concentration_pilot_vs_full.csv`
  - `full_power_auc_scatter_points_raw.csv`
  - `full_power_auc_scatter_points_aggregated.csv`
  - `power_semantic_auc_vs_distribution_concentration_scatter.png`
  Findings:
  - the common concentration overlap across all three full-run conditions is only `0.108156` to `0.202128`
  - over that common range, baseline remains mildly negative: slope `-0.044125`, `r=-0.965215`
  - redundant_additive and redundant_buffer collapse to only `2` x-values each in the common range, so overlap-only slope comparisons are not stable or very meaningful
  - baseline pilot vs full are reconciled by scenario mix, not a changed axis formula:
    - pilot baseline had only `4` concentration levels in 10 scenarios
    - full baseline had `7` concentration levels in 500 scenarios
    - full baseline populated new mid-range values `0.108156`, `0.118794`, `0.191489`
  - baseline AUC is not constant; in the full run it spans `0.121413` to `0.134972`, median `0.125209`, with a clear downward staircase as concentration increases, but over a narrow x-range the OLS slope magnitude stays numerically small because AUC itself lives on a tight band.
  - Scenarios with `t_min <= detection_latency`: `21 / 30 = 70%`, which is worse than the old Garrett reference of about `37%`.
  - New ladder matched the old one numerically:
    - none `0.4731`
    - priority `0.5090`
    - best-combinatorial `0.4878`
    - shatter-impact `0.4731`
  - Paired differences vs none:
    - priority mean `+0.0359`, strictly better in `21/30`
    - best-combinatorial mean `+0.0147`, strictly better in `13/30`
    - shatter-impact mean about `0.0000`, strictly better in `10/30`
- 2026-08-09: Montgomery delay-12 run started at `v3/data/processed/montgomery_delay12_refresh_N30_B3/`.
  - Dry run confirmed current Montgomery graph structure is preserved exactly: `581` nodes, `1355` edges, `1355` simulation edges.
  - Delay caveat is much larger here: `633` YAML zero-delay edges are still clipped to effective delay `1`.
  - Fair-pool exhaustive preview under capped top-15:
    - full priority union size `65`
    - full combinations `43,680`
    - capped search combinations `455`
    - estimated total runs `13,710`
    - measured per-run time about `0.3495 s`
    - estimated wall-clock about `1h 19m 51s`
- 2026-08-09: Completed the finer 4-hour-timestep Garrett rerun under `v3/data/processed/garrett_delay4_refresh_N30_B3/` using new delay-only configs:
  - `v3/configs/infrastructure_layers_delay4.yaml`
  - `v3/configs/infrastructure_layers_garrett_delay4.yaml`
  and a dedicated runner:
  - `v3/run_delay4_policy_refresh.py`
  Key mechanics/results:
  - Graph structure stayed identical to current Garrett: `52` nodes, `85` edges, `85` simulation edges.
  - Clip-floor check succeeded cleanly: all requested delays were already `>= 1`, and effective clipped-delay count was `0`; the standardization floor did not alter any 4-hour delay values.
  - Horizon was raised to `300` timesteps for the 4-hour run. Latest timestep with any new failure across the 30-scenario replay was `24`, so the run was not near truncation.
  - Replay `t_min` distribution under priority-B: min `0`, median `1.0`, mean `4.93`, max `24`.
  - Intervention-impossible share improved versus the 12-hour run but remained high: `18 / 30 = 60%` versus `21 / 30 = 70%` at 12h.
  - Shock-type impossible breakdown:
    - `ems`: `6/7`
    - `mixed`: `1/3`
    - `power`: `1/10`
    - `telecom`: `10/10`
  - 4-hour Garrett ladder:
    - none `0.4769`
    - priority `0.5090`
    - best-combinatorial `0.4917`
    - shatter-impact `0.4769`
  - Trend versus earlier runs:
    - none: original `0.4731` -> 12h `0.4731` -> 4h `0.4769`
    - priority: original `0.5090` -> 12h `0.5090` -> 4h `0.5090`
    - best-combinatorial: original `0.4878` -> 12h `0.4878` -> 4h `0.4917`
    - shatter-impact: original `0.4756` -> 12h `0.4731` -> 4h `0.4769`
  - Paired priority-minus-none at 4h: mean `+0.0321`, strictly better in `20/30` scenarios.
- 2026-08-09: Started the finer 4-hour-timestep Montgomery rerun under `v3/data/processed/montgomery_delay4_refresh_N30_B3/`.
  - Dry run confirmed current Montgomery graph structure is preserved exactly: `581` nodes, `1355` edges, `1355` simulation edges.
  - Clip-floor check also succeeded cleanly here: effective clipped-delay count `0`; the floor did not alter any of the 4-hour delay values.
  - Horizon for the 4-hour run is `300` timesteps.
  - Fair-pool exhaustive preview under capped top-15:
    - full priority union size `65`
    - full combinations `43,680`
    - capped search combinations `455`
    - estimated total runs `13,710`
    - measured per-run time about `0.7052 s`
    - estimated wall-clock about `2h 41m 8s`
  - At the last check in this session, the Montgomery priority baseline had finished and the process had advanced into the exhaustive combo loop, but the final `per_run_results.csv` / ladder outputs were not yet written.
  plus summary file `v3/data/processed/sensitivity/dependency_sensitivity_summary.md`. Important methodological finding: the current "redundant" graph makes cascades worse, not better (`mean peak damage=31.27` vs `16.08` baseline; `mean min LCC=0.9577` vs `0.9746` baseline), because the simulator still interprets extra support links as extra exposure rather than true fallback redundancy. By contrast, reducing power-edge weights by 20% dampens average damage (`mean peak damage=13.22`, `mean min LCC=0.9809`). This means the next model-design question is not just scenario scale-up but whether redundancy should be encoded structurally, dynamically, or both.
- 2026-07-04: Searched the local Zotero library by querying a safe copy of `~/Zotero/zotero.sqlite` because the live DB was locked. Strongest infrastructure-cascade title matches found were Li et al. (2024) on cascading failures across interdependent infrastructure systems, Schneider et al. (2024) on cascading effects in Cologne flood infrastructure, and König et al. (2018) on the CERBERUS approach; also noted adjacent supply-chain propagation papers already in the library.
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
- 2026-06-19: Began the organizational IP-capacity extension for `v3`. Added:
  - `v3/PLAN_ip_layer.md`
  - `v3/configs/ip_capacity_profiles.json`
  - IP-profile hooks in `v3/simulator_v3.py`
  - `v3/run_ip_moderation.py`
  - `v3/analyze_ip_moderation.py`
  - `v3/tests/test_ip_capacity.py`
  Current intervention implementation is default-off (`NONE`) and regression-tested against baseline behavior. A small pilot (`160` runs = `10` scenarios × `4` conditions × `4` IP profiles) was completed under `v3/data/processed/ip_moderation_pilot/` and `v3/runs/figures/ip_moderation_pilot/`. The pipeline works end to end, but the first pilot indicates only a very weak moderation effect with the current intervention design, so a full sweep should probably wait until the intervention semantics are strengthened or better targeted.
- 2026-06-19: Strengthened the `v3` IP-capacity semantics by moving interventions to the start of each timestep in `simulator_v3`, so a profile with latency `tau=1` can block the first downstream propagation wave instead of reacting one wave late. Added a regression test in `v3/tests/test_ip_capacity.py` to lock in that behavior and updated `v3/analyze_ip_moderation.py` to use an in-repo NumPy OLS path instead of the unavailable `statsmodels` dependency. Reran the 160-run pilot under `v3/data/processed/ip_moderation_pilot/` and refreshed `FINDINGS.md`, coefficients, slopes, and the figure. Updated pilot headline: `HIGH` now shows a materially stronger effect than before (`mean min_lcc = 0.9327` vs `0.8822` for `NONE`; `mean peak_damage_nodes = 39.2` vs `70.7`; strongest gains are in `power` and `mixed`), while `LOW` and `MED` remain close to baseline on minimum-LCC even though they still improve AUC/recovery behavior.
- 2026-06-19: Promoted matched-budget random targeting to a permanent comparison arm in the main IP sweep. Added `HIGH_RANDOM_MATCHED` to `v3/configs/ip_capacity_profiles.json`, extended `run_cascade_v3(...)` with an optional total `intervention_budget`, updated `v3/run_ip_moderation.py` so matched profiles can borrow realized per-run budgets from a reference profile (`HIGH`), and extended tests/analysis to cover the new arm. Full 100-scenario sweep completed under `v3/data/processed/ip_moderation_fullscale/` with artifacts in `v3/runs/figures/ip_moderation_fullscale/`. Full-scale headline results:
  - Overall means: `NONE min_lcc=0.8948`, `LOW=0.8972`, `MED=0.8978`, `HIGH=0.9480`, `HIGH_RANDOM_MATCHED=0.9344`
  - Overall means: `peak_damage_nodes` drops from `64.64` (`NONE`) to `63.30` (`LOW`), `60.22` (`MED`), `31.07` (`HIGH`), `38.35` (`HIGH_RANDOM_MATCHED`)
  - Overall means: `auc_resilience` rises from `0.9082` (`NONE`) to `0.9317` (`LOW`), `0.9410` (`MED`), `0.9738` (`HIGH`), `0.9531` (`HIGH_RANDOM_MATCHED`)
  - Capability result survives at scale and at matched count: `HIGH - HIGH_RANDOM_MATCHED` gives `+0.0135` `min_lcc`, `-7.28` peak damaged nodes, `+0.0207` AUC, with almost identical intervention count (`84.44` vs `84.34`)
  - The `HIGH` advantage remains concentrated in cascading shocks: `power` (`+0.0358` `min_lcc`, `-19.4` peak damage vs matched random) and `mixed` (`+0.0281`, `-14.6`), with near-zero differences for already localized `telecom`, `ems`, and `hospital` shocks
  - The `LOW/MED` pattern also survives at scale: modest capacity materially improves recovery/AUC, especially for `power` and `mixed`, but barely moves the collapse point (`min_lcc` gains only about `0.004` to `0.009`)
- 2026-06-20: Began Garrett County, MD port of the `v3` county pipeline by inspecting each HIFLD layer one by one before rebuilding anything. Current Garrett first-pass filtering decisions:
  - Hospitals: clean `COUNTY=GARRETT`, `STATE=MD` filter, yielding `1` hospital (`Garrett County Memorial Hospital`)
  - Power plants: clean `COUNTY=GARRETT`, `STATE=MD` filter, yielding `12` power assets
  - Public schools: clean `COUNTY=GARRETT`, `STATE=MD` filter, yielding `12` schools
  - Local EOC: clean `COUNTY=GARRETT`, `STATE=MD` filter, yielding `1` EOC
  - EMS/fire: no usable county field; adopted a Montgomery-style fallback using Garrett-only `allowed_cities` plus ZIP fallback. Conservative city pass gave `10` obvious Garrett EMS/fire rows; the built inventory currently yields `11` because ZIP fallback also picks up `Deer Park Community Volunteer Fire Department Station 20`
  - Telecom: HIFLD `LocCounty=GARRETT`, `LocState=MD` yields `15` raw rows but only `3` unique `UniqSysID` values. As in Montgomery, the correct repair is a composite telecom key `UniqSysID + latdec + londec`, which gives `15` effective distinct sites
- 2026-06-20: Added Garrett-specific config scaffolding:
  - `v3/configs/infrastructure_layers_garrett.yaml` uses Garrett county/state filters for the clean layers, Garrett EMS city/ZIP fallback, enables HIFLD telecom with composite IDs, and disables county telecom
  - `v3/configs/ip_capacity_profiles_garrett.json` switches IP capacity to proportional scaling so Garrett is not confounded by Montgomery-sized absolute capacities. Fractions were set to match Montgomery’s node-count proportions: `LOW=0.0207`, `MED=0.0622`, `HIGH=0.2075`, which correspond to about `2`, `4`, and `11` interventions respectively on a ~52-node Garrett graph
  - Built first-pass Garrett inventory with `v3/build_asset_inventory.py --config v3/configs/infrastructure_layers_garrett.yaml`, producing `v3/data/processed/asset_inventory_garrett.parquet` and `asset_inventory_garrett_summary.csv`. Current non-road counts are hospital `1`, EMS/fire `11`, power `12`, telecom `15`, EOC `1`, schools `12`
- 2026-06-20: Completed the main Garrett county pipeline in separate outputs without overwriting Montgomery artifacts. Added Garrett HIFLD telecom support to `v3/build_dependency_graph.py` so `cellular_towers` is treated as a telecom point layer, then built:
  - `v3/data/processed/garrett_dependency_graph_nodes.csv`
  - `v3/data/processed/garrett_dependency_graph_edges.csv`
  - `v3/data/processed/garrett_graph_validation_summary.md`
  - `v3/data/processed/garrett_graph_research_summary.md`
  - `v3/data/processed/garrett_simulation_edges.csv`
  - `v3/data/processed/garrett_graph_variants/`
  Garrett graph headline: `52` nodes, `85` edges, `3` weakly connected components, `0` isolated nodes. Sector composition is hospital `1`, emergency management `1`, EMS/fire `11`, power `12`, school `12`, telecom `15`. The singleton hospital/EOC do appear structurally important, but the graph is still simulation-ready.
- 2026-06-20: Ran Garrett county scenario experiments:
  - `v3/data/processed/garrett_pilot_scenarios/`
  - `v3/data/processed/garrett_redundancy_v3/`
  - `v3/data/processed/garrett_ip_moderation_fullscale/`
  plus matching figures under `v3/runs/figures/garrett_*`.
  Garrett `t=150` redundancy results differ sharply from Montgomery’s additive baseline scale: baseline Garrett is much more fragile (`mean min_lcc = 0.5378`, `mean peak_damage = 6.78` on a 52-node graph), but redundancy still helps strongly rather than disappearing. Condition means:
  - `baseline_additive`: `min_lcc = 0.5378`, `peak_damage = 6.78`, `auc = 0.5692`
  - `redundant_additive`: `min_lcc = 0.7700`, `peak_damage = 11.12`, `auc = 0.8263`
  - `redundant_buffer`: `min_lcc = 0.9049`, `peak_damage = 5.67`, `auc = 0.9394`
  - `dampened_power_additive`: effectively identical to baseline in Garrett as well
  Power and mixed shocks remain the hardest cases, with baseline Garrett power at `mean min_lcc = 0.4524`.
- 2026-06-20: Ran Garrett full-scale IP moderation with proportional capacities (`LOW≈2`, `MED≈4`, `HIGH≈11`) and permanent matched-budget random comparison. Garrett results preserve the qualitative Montgomery IP story:
  - overall means: `NONE min_lcc=0.6806`, `LOW=0.6819`, `MED=0.6884`, `HIGH=0.7653`, `HIGH_RANDOM_MATCHED=0.7547`
  - overall AUC: `NONE=0.7199`, `LOW=0.7614`, `MED=0.7666`, `HIGH=0.8013`, `HIGH_RANDOM_MATCHED=0.7913`
  - matched-budget capability result survives: `HIGH - HIGH_RANDOM_MATCHED` gives `+0.0107` `min_lcc`, `-0.45` peak damaged nodes, `+0.0100` AUC with essentially identical intervention count (`7.305` vs `7.283`)
  - the `HIGH` advantage again lives mainly in `power` (`+0.0332` `min_lcc`, `-1.425` peak damage vs matched random) and is near-zero in already localized shocks
  - `LOW` and `MED` again mostly improve recovery/AUC with little movement in the collapse point
  Garrett moderation slopes from `FINDINGS.md`: `NONE -2.357`, `LOW -2.345`, `MED -2.316`, `HIGH -1.156`, `HIGH_RANDOM_MATCHED -1.463`, supporting stronger concentration moderation only at high capacity.
- 2026-06-20: Added Garrett map artifacts in the Montgomery slide style. Exported `v3/data/processed/point_asset_inventory_garrett.csv` and rendered the tiled basemap figure `v3/runs/figures/garrett_point_assets_basemap.png`. Updated `v3/export_point_map_data.py` to include `cellular_towers` and `v3/plot_point_asset_map.py` to accept a custom `--title`.
- 2026-06-23: Advisor follow-up triage focused on the Garrett regression-line anomaly. Key code finding: current line plots in `v3/analyze_ip_moderation.py` and `v3/model_resilience_interactions.py` are partial-dependence style predictions, not per-line overlays against the exact subset of observed points. In `analyze_ip_moderation.py`, lines are generated by fixing `shock_type="power"`, `condition="baseline_additive"`, and `mean_seed_degree=median`, then sweeping concentration with no point overlay. This can visually suggest a mismatch when compared against broader Garrett point clouds, especially because Garrett `min_lcc` spans very different ranges across conditions/profiles. For the Garrett IP moderation data, the current model implies very steep endpoint predictions for some profiles (for example `NONE` in the fixed baseline/power slice can drop near `0.03` at the max observed concentration), while actual observed `baseline_additive + power` profile means are closer to `0.41-0.50` and minima `0.17-0.35`. Immediate next step should be a diagnostic replot with each fitted line shown only alongside the points from the exact subset used for that line, plus explicit intercept/bias reporting.
- 2026-06-24/25: Implemented multiplier-aware graph rebuilding for the advisor-requested radius sweep:
  - `v3/build_dependency_graph.py` now supports `--distance-multiplier`
  - `v3/build_graph_variants.py` now supports `--distance-multiplier`
  - Added `v3/run_radius_sweep.py` to orchestrate county-by-multiplier graph rebuilds, baseline-connectivity diagnostics, aggregation logic, and clipped regression plotting
  - Added `docs/cutoff_justification.md`, `PLAN.md`, `analysis/radius_sweep_report.md`, `results/radius_sweep_baseline_connectivity.csv`, and clipped regression figures under `runs/figures/regression_clipped/`
- Fast baseline-connectivity result from the radius sweep:
  - Montgomery baseline LCC fractions by multiplier: `0.7x=0.9776`, `1.0x=1.0000`, `1.3x=1.0000`, `1.6x=1.0000`
  - Garrett baseline LCC fractions by multiplier: `0.7x=0.6538`, `1.0x=0.6538`, `1.3x=0.6538`, `1.6x=0.6538`
  This means Garrett's low starting `min_lcc` is not a regression-line artifact alone; under the current rule template, the graph remains structurally split even after uniformly relaxing all radii up to `1.6x`.
- The full multi-multiplier IP moderation rerun is much heavier than the baseline step. A longer background job was launched to continue the full sweep:
  - PID at launch: `65685`
  - command: `./.venv/bin/python -u v3/run_radius_sweep.py > results/radius_sweep_full.log 2>&1 &`
  - current log path: `results/radius_sweep_full.log`
  If the background job is still alive in the next session, check that log and the eventual `results/radius_sweep.parquet` output before restarting anything.
- 2026-06-25: Diagnosed the `18` Garrett nodes outside the largest weakly connected component (`52 - 34 = 18`). Breakdown:
  - `telecom=7`
  - `power=5`
  - `ems_fire=4`
  - `school=2`
- 2026-06-30: Regenerated `v3/runs/figures/regression_clipped/{montgomery,garrett}_clipped_regression.png` without `dampened_power_additive`. `v3/run_radius_sweep.py` now uses a dedicated clipped-plot condition list with only `baseline_additive`, `redundant_additive`, and `redundant_buffer`.
- 2026-06-30: Added `v3/generate_substitutes_gap_figure.py` and exported the paper-facing Section 5.3 figure under `v3/runs/figures/section_53/`:
  - `high_vs_random_by_semantic_condition.png`
  - `high_vs_random_by_semantic_condition.pdf`
  - `high_vs_random_by_semantic_condition.csv`
  The figure shows the paired mean `HIGH - HIGH_RANDOM_MATCHED` gap in `min_lcc` by semantic condition for Montgomery and Garrett. In both counties the gap is positive under additive semantics and shrinks to approximately zero under `redundant_buffer`, supporting the substitutes interpretation.
- 2026-07-01: Added `v3/generate_intro_regression_plots.py` and exported pooled single-line county regression figures under `v3/runs/figures/regression_intro/` for report sequencing before the semantic decomposition:
  - `montgomery_single_line_regression.png`
  - `garrett_single_line_regression.png`
  - `combined_single_line_regression.png`
  - `single_line_regression_stats.csv`
  These pool `baseline_additive`, `redundant_additive`, and `redundant_buffer` together for `ip_profile=NONE`, producing one fitted line per county instead of the three-line semantic breakdown.
- 2026-07-03: Made the intervention exogeneity assumption explicit in `v3/simulator_v3.py` via `ORG_DECISION_DISRUPTABLE = False`, with a comment that EMS / emergency-management facilities can fail as network nodes while organizational decision capability remains exogenous in the current study. The flag is now also surfaced in `scenario_info` and in `v3/run_ip_moderation.py` run-level outputs as `org_decision_disruptable`.
- 2026-07-04: Installed `zotero-mcp-server` v`0.6.1` into an isolated environment at `~/.codex/venvs/zotero-mcp` and registered it in `~/.codex/config.toml` as `[mcp_servers.zotero]` with `ZOTERO_LOCAL = "true"`. Verification status:
  - server install OK
  - Codex config entry added
  - Zotero desktop app is running
  - local read test reached Zotero API but returned `403 Local API is not enabled`
  This means the intended use case is feasible after the user enables Zotero's local API preference and restarts/reloads Codex so the new MCP server is picked up.
- 2026-07-04: Re-checked the Zotero MCP setup from the GraphTransformer workspace. Current state is still "installed but not ready in this live Codex session": `~/.codex/config.toml` contains `[mcp_servers.zotero]`, the venv contains `zotero-mcp-server 0.6.1`, and Zotero is running, but Zotero MCP tools are not exposed in the current session and the earlier verification still shows `403 Local API is not enabled`. Practical next steps remain: enable Zotero's local API, then restart/reload Codex so the new MCP server is actually available to tools.
- 2026-07-04: Checked Zotero again from the GraphTransformer workspace. Current state:
  - `~/.codex/config.toml` still contains the `zotero` MCP server entry
  - Zotero desktop is running
  - direct localhost API probe to `http://127.0.0.1:23119/...` returned connection failure (`curl` code `7` / HTTP `000`), so the live local API is still not usable from this session
  - local fallback via a safe copy of `~/Zotero/zotero.sqlite` still works
  Using the SQLite fallback, a creator search for `Zobel` found one library item: Li and Zobel (2020), "Exploring supply chain network resilience in the presence of the ripple effect," *International Journal of Production Economics*, DOI `10.1016/j.ijpe.2020.107693`.
- 2026-07-04: Zotero lookup for Premkumar, Ramamurthy, and Saunders (2005) succeeded via safe SQLite copy. Exact item found:
  - item key `H3E8DT3H`
  - attachment key `TJZ5J6XE`
  - DOI `10.1080/07421222.2003.11045841`
  - title `Information Processing View of Organizations: An Exploratory Examination of Fit in the Context of Interorganizational Relationships`
  - journal `Journal of Management Information Systems`, `22(1)`, pp. `257-294`
  The attached Zotero PDF exists under `~/Zotero/storage/TJZ5J6XE/`, but local rendering suggests the file may be a scanned or mismatched attachment: first-page preview shows Tushman and Nadler's older article `Information Processing as an Integrating Concept in Organizational Design` rather than the Premkumar et al. title page. Future paper checks should verify whether this is a packet/scan issue or an attachment mismatch before relying on PDF text extraction.
- 2026-07-04: Installed open-source OCR tooling with Homebrew:
  - `tesseract 5.5.2`
  - `ocrmypdf 17.8.0`
  - `poppler 26.07.0` (`pdftotext`, `pdftoppm`)
  End-to-end OCR smoke test succeeded on a temporary copy of the Zotero attachment using `ocrmypdf --sidecar`, producing readable OCR text. That OCR output confirmed the attachment mismatch more strongly: the PDF attached to the Premkumar et al. Zotero item actually OCRs as Tushman and Nadler's `Information Processing as an Integrating Concept in Organizational Design`, not the 2005 JMIS paper metadata attached to the parent item.
- 2026-07-04: Re-checked the Premkumar, Ramamurthy, and Saunders (2005) paper with a broader Zotero storage scan and found that there are actually two local PDFs with the same filename:
  - `~/Zotero/storage/TJZ5J6XE/...pdf` is the mismatched older 12-page scan (metadata points to a 2001 Photoshop-produced file and does not yield the target article text).
  - `~/Zotero/storage/E5RIIU5K/...pdf` is the correct 39-page JSTOR PDF for `Information Processing View of Organizations: An Exploratory Examination of Fit in the Context of Interorganizational Relationships`.
  The correct local copy can now be used for future OIPT/paper interpretation without needing another web lookup.
- 2026-07-04: Checked Srinivasan and Swink (2018), `An Investigation of Visibility and Flexibility as Complements to Supply Chain Analytics: An Organizational Information Processing Theory Perspective`, from local Zotero PDF `~/Zotero/storage/BTGESH7H/`. Key takeaway for current research framing: this is a strong OIPT paper for arguing that information-processing capability alone is not enough; its value depends on complementary organizational flexibility and on environmental uncertainty/volatility. Good fit for the county-resilience framing as a conceptual bridge from analytics/visibility capacity to the ability to act on disruption information, though still not a direct county/public-sector empirical precedent.
  No hospitals or EOC are outside the LCC, but the disconnected set is not just remote support-only assets; it includes consumer and emergency-service nodes. This argues against a simple denominator-only explanation and supports the interpretation that the current rural Garrett rule template leaves genuinely isolated local subnetworks under the modeled dependencies.
- 2026-06-28: Added a visual map of the three Garrett weakly connected subnetworks:
  - figure: `v3/runs/figures/garrett_component_map.png`
  - node/component table: `v3/runs/figures/garrett_component_map.csv`
  The rendered map shows:
  - Component 1 (`n=34`) as the main west/central blue network
  - Component 2 (`n=14`) as a distinct eastern red network containing telecom, EMS/fire, power, and schools
  - Component 3 (`n=4`) as a small southwestern green power/telecom pocket
- 2026-06-28: Moved the clipped regression artifacts into the `v3` output tree for consistency. Current paths:
  - `v3/runs/figures/regression_clipped/garrett_clipped_regression.png`
  - `v3/runs/figures/regression_clipped/montgomery_clipped_regression.png`
- 2026-07-07: Refactored `v3/simulator_v3.py` edge identity so intervention logs now carry a stable edge key derived from `(simulation_source, simulation_target, dependency_type)` while preserving the old row-index `edge_id` internally for behavior stability. `build_support_maps(...)` now emits bidirectional ID/key mappings plus an `edge_identity_audit`, and duplicate tuple keys are deterministically disambiguated by appending the row-index string as a fourth tuple element if needed. Added focused coverage in `v3/tests/test_ip_capacity.py`. Regression check on Montgomery `ipfull_001` with `HIGH` confirmed identical timestep trajectories and summary metrics pre/post refactor (`min_lcc`, `auc_resilience`, `component_count`, `lcc_fraction`, failed/degraded trajectories all unchanged). Current Montgomery audit found `0` duplicate stable-key groups across `1355` simulation edges.
- 2026-07-07: Added profile-level whole-run intervention budgets in `v3/simulator_v3.py` via optional `total_intervention_budget`. This is off by default and preserves old behavior when absent. If present, the simulator now stops all further interventions once the cumulative protected-edge count reaches the budget, while keeping the existing per-timestep edge ranking/selection logic. Explicit `intervention_budget` arguments still take precedence so matched-budget workflows remain intact. Added a unit test for the profile-level cap. Garrett sanity check on scenario `garrett_ip_001` with a custom `B=3` profile confirmed exactly `3` protected edges, with stable keys:
  - `(power_plants::59147, ems_fire::37c249cc-e959-43cb-a8db-b580154677e7, ems_fire -> power)`
  - `(power_plants::59147, cellular_towers::12469__39.4088888889__-79.4127777778, telecom -> power)`
  - `(power_plants::59147, cellular_towers::12791__39.3951666667__-79.4648055556, telecom -> power)`
- 2026-07-08: Added `targeting_mode="forced"` to `v3/simulator_v3.py`. Profiles can now pass an explicit `forced_edge_keys` list of stable tuple keys `(simulation_source, simulation_target, dependency_type[, disambiguator])`. In forced mode, only those edges are eligible for protection, they are logged by tuple key, and they are protected with the existing binary `halt_edge` mechanism once their source becomes intervention-eligible after detection latency. Existing `priority` and `random` modes are unchanged; forced mode is off by default. Added a Garrett-backed unit test in `v3/tests/test_ip_capacity.py` using scenario `garrett_ip_001` and a fixed 3-edge set. Verification passed: exactly those 3 keys, and no others, appeared in the intervention log in the requested order.
- 2026-07-08: Added `v3/run_forced_exhaustive_search.py` for exhaustive search over forced-edge sets under a whole-run budget. Current Garrett run completed at `C=15`, `B=3`, `N=30`, objective `mean min_lcc` over a fixed 30-scenario set generated with scenario seed `42` and plan `{power=9, telecom=9, ems=6, hospital=3, mixed=3}`. Candidate filter uses the existing simulator outdegree-load logic: rank each edge by `max(outdegree_load(source), outdegree_load(target))`, with the normalized dependency-concentration analog logged as `edge_score / total_sim_edges`. Garrett candidate pool is highly concentrated: the top 15 edges span only `2` distinct source nodes, mainly `power_plants::59147` plus `ems_fire::37c249cc-e959-43cb-a8db-b580154677e7`. Search size was `15 choose 3 = 455` edge sets, `13,710` total simulation runs including baselines, measured per-run time about `0.085s`, estimated wall clock about `19m 29s`. Results:
  - no intervention: mean `min_lcc = 0.5115`, sd `0.1464`
  - priority targeting capped at `B=3`: mean `min_lcc = 0.5372`, sd `0.1334`, mean intervention count `1.8` (confirmed capped at `B`)
  - best exhaustive forced set: mean `min_lcc = 0.5192`, sd `0.1375`
  Best Garrett forced set clustered on a single source node (`1` distinct source node): `(power_plants::59147, cellular_towers::12469__39.4088888889__-79.4127777778, telecom -> power)`, `(power_plants::59147, cellular_towers::12791__39.3951666667__-79.4648055556, telecom -> power)`, and `(power_plants::59147, ems_fire::37c249cc-e959-43cb-a8db-b580154677e7, ems_fire -> power)`. Full artifacts live under `v3/data/processed/garrett_forced_exhaustive_search_C15_B3_N30/`.
- 2026-07-08: Extended `v3/run_forced_exhaustive_search.py` with `candidate_source="priority_union"`, which builds a fair forced-search pool from the distinct stable tuple keys actually protected by the capped-`B` priority baseline in a prior run. It can also reuse the exact saved `scenario_metadata.csv` from that prior run so the scenario set stays identical. Dry run on Garrett using the priority-union pool from `v3/data/processed/garrett_forced_exhaustive_search_C15_B3_N30/` produced a `29`-edge pool spanning `15` distinct source nodes, `29 choose 3 = 3,654` combinations, `109,680` total simulation runs including baselines, and an estimated wall clock of about `2h 37m 27s` at measured per-run time `0.086s`. The fair-pool output path defaults to `v3/data/processed/garrett_forced_exhaustive_search_priority_union_B3_N30/`.
- 2026-07-09: The fair-pool Garrett exhaustive search completed successfully under `v3/data/processed/garrett_forced_exhaustive_search_priority_union_B3_N30/`. Setup reused the exact 30-scenario Garrett set from the first exhaustive run, but replaced the static top-15 candidate pool with the `29` distinct stable tuple keys actually protected by capped-`B=3` priority in those scenarios. Search size was `29 choose 3 = 3,654` edge sets, `109,680` total simulation runs including baselines, with measured per-run time about `0.0867s` and realized wall clock a little under five hours in this session. Results:
  - no intervention: mean `min_lcc = 0.5115`, sd `0.1464`
  - dynamic priority capped at `B=3`: mean `min_lcc = 0.5372`, sd `0.1334`, mean intervention count `1.8`
  - best exhaustive forced set on the fair pool: mean `min_lcc = 0.5205`, sd `0.1341`
  Best minus dynamic priority was `-0.0167`, which is only about `-0.125` times the priority run-to-run sd, so the fair-pool exhaustive search still did not beat dynamic priority. Best fair-pool set used `2` distinct source nodes: `(power_plants::59147, cellular_towers::12469__39.4088888889__-79.4127777778, telecom -> power)`, `(power_plants::59147, ems_fire::37c249cc-e959-43cb-a8db-b580154677e7, ems_fire -> power)`, and `(power_plants::57300, cellular_towers::12469__39.4088888889__-79.4127777778, telecom -> power)`. The top of the ranking remained essentially flat: top-10 `mean_min_lcc` range only `0.0013`, well below the run-to-run sd scale.
- 2026-07-09: Prepared the Montgomery fair-pool search without changing code. Reused the saved 30-scenario set in `v3/data/processed/montgomery_forced_exhaustive_search_C15_B3_N30/scenario_metadata.csv` and ran a small reference job under `v3/data/processed/montgomery_priority_union_reference_B3_N30/` to capture capped-`B=3` dynamic-priority intervention logs on that exact scenario set. The resulting priority union contains `44` distinct stable tuple edges spanning `20` distinct source nodes across the 30 scenarios. A dry run of the corresponding fair-pool exhaustive search (`candidate_source="priority_union"`) produced `44 choose 3 = 13,244` edge combinations, `397,380` total simulation runs including baselines, measured per-run time about `0.419s`, and estimated wall-clock about `46h 13m`. Per the user’s explicit stop rule for very large pools, the full Montgomery fair-pool exhaustive search was not auto-started.
- 2026-07-09: Added two narrow capabilities to `v3/run_forced_exhaustive_search.py` for feasible Montgomery fair-pool search: (1) `--priority-union-top-k` to cap a priority-union pool to the top `K` stable tuple edges by dynamic-priority selection frequency, and (2) automatic reuse of `baseline_none` / `baseline_priority_b` rows and intervention logs from the supplied priority-union reference directory. Then ran the capped Montgomery fair-pool exhaustive search under `v3/data/processed/montgomery_forced_exhaustive_search_priority_union_top15_B3_N30/` with pool source `priority union, top-15 by frequency`, `B=3`, `N=30`, and objective `mean min_lcc`. Dry run gave `15 choose 3 = 455` combinations, `13,710` total simulation runs on the accounting preview, measured per-run time about `0.421s`, estimated wall-clock about `1h 36m`, and realized wall-clock about `1h 54m`. The selected top-15 pool spans `9` distinct source nodes. Results:
  - no intervention: mean `min_lcc = 0.9005`, sd `0.1428`
  - dynamic priority capped at `B=3` (reused reference baseline): mean `min_lcc = 0.9051`, sd `0.1419`, mean intervention count `2.0`
  - best exhaustive forced set from the capped fair pool: mean `min_lcc = 0.9030`, sd `0.1404`
  Best minus dynamic priority was `-0.0021`, only about `-0.015` times the priority run-to-run sd, so fixed forced sets still did not beat dynamic priority in Montgomery. The best set clustered on a single source node (`power_plants::62910`) with three telecom edges: `(power_plants::62910, county_telecom::295.0, telecom -> power)`, `(power_plants::62910, county_telecom::393.0, telecom -> power)`, `(power_plants::62910, county_telecom::413.0, telecom -> power)`. The ranking was extremely flat: top-10 `mean_min_lcc` range only `0.0005`.
- 2026-07-13: Created two advisor-facing intervention artifacts focused only on the recent post-last-Monday simulation-intervention work:
  - updated `DECISION_LOG.md` so it now explicitly scopes to the recent intervention mechanics / forced targeting / Garrett and Montgomery exhaustive-search decisions
  - added `docs/INTERVENTION_REPORT_GARRETT_MONTGOMERY.md`, a compact cross-county report comparing Garrett fair-pool results and Montgomery capped fair-pool results, with the shared intervention design, county ladders, best fixed sets, flatness checks, and main takeaways
- 2026-07-21: Added `v3/run_shatter_map_analysis.py` as a new read-only analysis layer on top of the existing cascade simulator. It does not change simulator behavior; instead it reuses the same cascade logic to record final connected components across many scenario draws, computes node-pair same-component fractions, node largest-component fractions, and a simulation-edge fault-line ranking, then builds a `shatter_topk` forced policy from the top-`B` fault-line edges and compares it against no intervention, capped priority, and the saved best-combinatorial policy on the same 30-scenario county set. First Garrett run completed under `v3/data/processed/garrett_shatter_map_N1000_B3/` with `N=1000`, `B=3`, objective `min_lcc`, and comparison set reused from `v3/data/processed/garrett_forced_exhaustive_search_priority_union_B3_N30/`. Dry-run estimate was about `1m 24s`; realized wall-clock was about `90s`. Garrett shatter-map findings:
  - top fault-line edges all centered on `hospitals::0006321550`, with the top 7 edges tied at fault-line frequency `0.203`
  - fault-line counts: `0` always-cut (`>0.9`), `60` never-cut (`<0.1`), `25` in between
  - `shatter_topk` chose:
    - `(cellular_towers::12469__39.4088888889__-79.4127777778, hospitals::0006321550, hospital -> telecom)`
    - `(cellular_towers::12791__39.3951666667__-79.4648055556, hospitals::0006321550, hospital -> telecom)`
    - `(ems_fire::37c249cc-e959-43cb-a8db-b580154677e7, hospitals::0006321550, hospital -> ems_fire)`
  - Garrett ladder on the identical 30-scenario comparison set:
    - no intervention: mean `min_lcc = 0.5115`, sd `0.1464`
    - priority `B=3`: mean `min_lcc = 0.5372`, sd `0.1334`
    - best combinatorial: mean `min_lcc = 0.5205`, sd `0.1341`
    - shatter_topk: mean `min_lcc = 0.5128`, sd `0.1467`
  Shatter-topk underperformed both capped priority and the saved best-combinatorial Garrett set (`shatter - priority = -0.0244`, about `-0.183 x` priority sd).
- 2026-07-21: Ran the same shatter-map analysis for Montgomery under `v3/data/processed/montgomery_shatter_map_N1000_B3/` with `N=1000`, `B=3`, objective `min_lcc`, and comparison set reused from `v3/data/processed/montgomery_forced_exhaustive_search_priority_union_top15_B3_N30/`. Dry-run estimate was about `7m 6s`; realized wall-clock was about `470s`. Montgomery shatter-map findings:
  - top fault-line edges centered on school-facing support bundles, especially `public_schools::240048000820` and `public_schools::240048000868`, with the top 6 edges tied at fault-line frequency `0.174`
  - fault-line counts: `0` always-cut (`>0.9`), `922` never-cut (`<0.1`), `433` in between
  - `shatter_topk` chose:
    - `(county_telecom::357.0, public_schools::240048000820, school -> telecom)`
    - `(county_telecom::357.0, public_schools::240048000868, school -> telecom)`
    - `(ems_fire::0f7967fe-2db9-4d9f-9c18-a08aa1df0091, public_schools::240048000820, school -> ems_fire)`
  - Montgomery ladder on the identical 30-scenario comparison set:
    - no intervention: mean `min_lcc = 0.9005`, sd `0.1428`
    - priority `B=3`: mean `min_lcc = 0.9051`, sd `0.1419`
    - best combinatorial: mean `min_lcc = 0.9030`, sd `0.1404`
    - shatter_topk: mean `min_lcc = 0.9008`, sd `0.1420`
  Shatter-topk again underperformed both capped priority and the saved best-combinatorial Montgomery set (`shatter - priority = -0.0043`, about `-0.030 x` priority sd).
- 2026-07-27: Added a corrected scenario-generation path that samples disruption seeds only from nodes with dependency-graph in-degree `> 0`, while leaving the dependency graph and simulator behavior unchanged. Implemented this by adding positive-in-degree seed filtering to `v3/run_scenarios_v1.py`, plumbing the flag through `v3/run_forced_exhaustive_search.py` and `v3/run_shatter_map_analysis.py`, and creating `v3/run_positive_indegree_policy_refresh.py` to regenerate 30-scenario county sets in fresh output folders, rerun capped-priority baselines, rebuild capped fair-pool exhaustive search from the new priority union, and regenerate shatter rankings under the same corrected rule.
- 2026-07-27: Completed the Garrett positive-in-degree refresh under `v3/data/processed/garrett_positive_indegree_refresh_N30_B3/`. Eligible seed counts became: emergency_management `1/1`, ems_fire `8/11`, hospital `0/1`, power `12/12`, school `0/12`, telecom `6/15`. Because hospitals and schools had no eligible seeds, the regenerated 30-scenario plan was redistributed to `power=10`, `telecom=10`, `ems=7`, `mixed=3`. Dependency-concentration shifted upward from old `(min 0.0000, median 0.0235, mean 0.0525, max 0.2353)` to new `(min 0.0000, median 0.0824, mean 0.0792, max 0.2353)`. The new priority union had `35` edges, capped to top-`15` by frequency for a `455`-combo exhaustive search (about `19m 26s` estimated). New Garrett ladder:
  - no intervention: mean `min_lcc = 0.4731`, sd `0.1342`
  - priority `B=3`: mean `min_lcc = 0.5090`, sd `0.1278`
  - best combinatorial: mean `min_lcc = 0.4878`, sd `0.1376`
  - shatter_topk: mean `min_lcc = 0.4776`, sd `0.1334`
  Relative to the old 30-scenario set, all four policies declined because the corrected scenarios removed many structurally null seeds. New fault-line counts were `0` always-cut, `53` never-cut, `32` in-between. New best fixed set remained school-telecom focused, while new shatter-topk again centered on the hospital support bundle.
- 2026-07-27: Completed the Montgomery positive-in-degree refresh under `v3/data/processed/montgomery_positive_indegree_refresh_N30_B3/`. Eligible seed counts became: emergency_management `2/2`, ems_fire `40/42`, hospital `0/11`, power `15/15`, school `0/211`, telecom `123/300`. The regenerated 30-scenario plan again became `power=10`, `telecom=10`, `ems=7`, `mixed=3`. Dependency-concentration changed only modestly from old `(min 0.0000, median 0.0085, mean 0.0331, max 0.1830)` to new `(min 0.0007, median 0.0092, mean 0.0344, max 0.1830)`. The new priority union had `65` edges, capped to top-`15` by frequency for a `455`-combo exhaustive search (about `1h 37m 36s` estimated; realized overnight-scale run completed successfully). New Montgomery ladder:
  - no intervention: mean `min_lcc = 0.8940`, sd `0.1431`
  - priority `B=3`: mean `min_lcc = 0.9012`, sd `0.1431`
  - best combinatorial: mean `min_lcc = 0.8971`, sd `0.1410`
  - shatter_topk: mean `min_lcc = 0.8944`, sd `0.1422`
  Relative to the old 30-scenario set, all four Montgomery policies also declined slightly, but much less than Garrett because the concentration distribution changed only a little. New fault-line counts were `0` always-cut, `794` never-cut, `561` in-between. The new best fixed set again clustered on `power_plants::62910` with three telecom edges, while the new shatter-topk set remained school-facing.
- 2026-07-28: Added a new read-only impact-based shatter analysis runner, `v3/run_shatter_impact_analysis.py`, without changing simulator behavior or overwriting the earlier frequency-based shatter outputs. The new analysis measures edge impact at `t_min`, defined as the first timestep where LCC reaches its minimum, on `N=1000` positive-indegree-seeded scenarios using the same corrected scenario allocation as the positive-indegree refresh (`power=334`, `telecom=333`, `ems=222`, `mixed=111`). For each edge `(i -> j)` in failure-propagation direction, impact in a run is the count of alive nodes reachable downstream from `j` at `t_min` when `i` is failed and `j` is alive; edge `mean_impact` is the average over all `N` runs, and `n_runs_eligible` records how often the edge actually scored. The runner also emits `edge_impact_visualization.csv` with source/target lat-long and a normalized `1-5` impact scale for geographic redraws.
- 2026-07-28: Garrett impact-based shatter analysis completed under `v3/data/processed/garrett_shatter_impact_map_positive_indegree_N1000_B3/`. Wall-clock estimate was about `1m 28s`; realized runtime was about `90s`. `t_min` distribution was concentrated very early: min `0`, median `2`, mean `1.71`, max `4`, with counts `{0:200, 1:149, 2:501, 3:37, 4:113}`. The new ranking did discriminate: `mean_impact` min `0.0000`, median `0.0110`, max `0.0740`, top1-top2 gap `0.0070`, top10 range `0.0260`, and `9` unique top-10 values. Top impact edges shifted away from the old hospital bundle and toward school-facing telecom / EMS edges, led by `(cellular_towers::12469__39.6858333333__-79.0858333333, public_schools::240036000668, school -> telecom)` at `mean_impact = 0.074`. The new policy ladder on the corrected 30-scenario comparison set was:
  - no intervention: mean `min_lcc = 0.4731`, sd `0.1342`
  - priority `B=3`: mean `min_lcc = 0.5090`, sd `0.1278`
  - best combinatorial: mean `min_lcc = 0.4878`, sd `0.1376`
  - shatter_impact_topk: mean `min_lcc = 0.4756`, sd `0.1364`
  So the impact-based top-3 still did not beat priority or the best fixed combinatorial set in Garrett.
- 2026-07-28: Montgomery impact-based shatter analysis completed under `v3/data/processed/montgomery_shatter_impact_map_positive_indegree_N1000_B3/`. Wall-clock estimate was about `7m 10s`; realized runtime was about `448s`. `t_min` was also early but slightly later than Garrett: min `1`, median `2`, mean `2.76`, max `4`, with counts `{1:15, 2:531, 3:132, 4:322}`. The new ranking was effectively flat: `mean_impact` min `0.0000`, median `0.0000`, max only `0.0010`, top1-top2 gap `0.0000`, top10 range `0.0010`, and only `2` unique top-10 values. Only two edges had positive `mean_impact` at all, both school-facing telecom edges with `n_runs_eligible = 1`; the third protected edge in `shatter_impact_topk` therefore had `mean_impact = 0.0` because the top-3 rule was still applied mechanically. The new policy ladder on the corrected 30-scenario comparison set was:
  - no intervention: mean `min_lcc = 0.8940`, sd `0.1431`
  - priority `B=3`: mean `min_lcc = 0.9012`, sd `0.1431`
  - best combinatorial: mean `min_lcc = 0.8971`, sd `0.1410`
  - shatter_impact_topk: mean `min_lcc = 0.8941`, sd `0.1430`
  So Montgomery’s impact-based ranking was essentially non-discriminative and again did not beat priority or the best fixed combinatorial set.
- 2026-08-01: Completed a read-only provenance audit across the current Montgomery and Garrett dependency-graph builds and the saved June-July result folders. Current graphs on disk are:
  - Montgomery: `581` nodes, `1355` edges, one weakly connected component; config `v3/configs/infrastructure_layers.yaml` with `county_telecom` enabled and `cellular_towers` disabled.
  - Garrett: `52` nodes, `85` edges, three weakly connected components of sizes `34, 14, 4`; config `v3/configs/infrastructure_layers_garrett.yaml` with `cellular_towers` enabled and `county_telecom` disabled.
  Provenance verdicts:
  - The June 2026 semantic / redundancy result folders (`v3/data/processed/redundancy_v3/`, `resilience_interaction_model/`, `garrett_redundancy_v3/`, `garrett_resilience_interactions/`) were built on the same current graphs by node/edge counts and scenario metadata ranges, but they used the older unrestricted seed sampler rather than the later positive-in-degree-only rule. Exact deck slope values around Montgomery `-2.841 / -4.511 / -1.070` and Garrett `-1.837 / -3.396 / -1.232` were not found verbatim in the repo; the closest saved slope artifacts on disk are Montgomery `-2.699 / -4.971 / -1.204` and Garrett `-1.698 / -3.177 / -1.296`.
  - The June 2026 IP moderation folders (`v3/data/processed/ip_moderation_fullscale/`, `garrett_ip_moderation_fullscale/`) were also built on the same current graphs, used the older unrestricted seed sampler, and used `100` base scenarios per county. The headline HIGH-vs-NONE mean `min_lcc` gaps in the saved county comparison tables match the deck values up to rounding: Montgomery `0.947960 - 0.894789 = 0.053171` and Garrett `0.765337 - 0.680577 = 0.084760`.
  - The July 2026 positive-indegree refresh folders and their downstream four-policy ladders (`v3/data/processed/{garrett,montgomery}_positive_indegree_refresh_N30_B3/` plus shatter-impact subfolders) were built on the same current graphs and use the newer positive-in-degree-only seeding rule, so these are the cleanest “current graph + current seed rule” policy results.
- 2026-08-01: Read-only extraction of saved slope/IP moderation comparison values. Pearson `r` for the semantic county-condition fits was computed directly from the archived `v3/data/processed/{redundancy_v3,garrett_redundancy_v3}/scenario_summary_metrics_v3.csv` files: Montgomery `baseline_additive -0.983406`, `redundant_additive -0.973867`, `redundant_buffer -0.810128`; Garrett `baseline_additive -0.758756`, `redundant_additive -0.870714`, `redundant_buffer -0.791744`. OIPT intercept/slope comparisons were read from `v3/data/processed/oipt_slope_bias_interpretation.md`; overall profile mean-`min_lcc` gaps were read from `v3/data/processed/county_comparison_min_lcc_table.csv`.
- 2026-08-03: Read-only provenance check on cascade parameter origins. Current repo evidence says dependency `weight`/`delay` values were manually hardcoded into `v3/configs/infrastructure_layers*.yaml`, sector cascade parameters (`degrade_threshold`, `fail_threshold`, `propagation_scale`, `recovery_*`, dwell times) were manually hardcoded into `v3/simulator_v2.py` / `v3/simulator_v3.py`, and scenario-level multiplier sets were manually chosen in `v3/run_scenarios_v1.py`. I did not find a specific user-supplied literature citation attached to those numeric values; repo wording only says the framework is “literature-informed,” and one explicit comment ties the organizational-exogeneity assumption to advisor guidance.
- 2026-08-05: Generated a Montgomery `baseline_additive` concentration scatter directly from the archived `v3/data/processed/redundancy_v3/scenario_summary_metrics_v3.csv` rows and saved it to `v3/runs/figures/resilience_interaction_model/montgomery_baseline_additive_concentration_scatter.png`, with summary stats in `v3/data/processed/resilience_interaction_model/montgomery_baseline_additive_scatter_stats.csv`. Important distinction: the baseline-only simple OLS scatter slope is `-2.794055` with Pearson `r = -0.983406` and `R^2 = 0.967088`; the archived `-2.699177` value in `redundancy_condition_slopes.csv` is the baseline slope from the pooled interaction model `min_lcc ~ dependency_concentration_cond * C(condition) + mean_seed_degree_cond + mean_propagation_delay_cond + C(shock_type)` over all `2000` scenarios, not the standalone 500-point baseline-only line.
- 2026-08-05: Generated the matching Garrett `baseline_additive` concentration scatter from archived `v3/data/processed/garrett_redundancy_v3/scenario_summary_metrics_v3.csv` rows and saved it to `v3/runs/figures/garrett_resilience_interactions/garrett_baseline_additive_concentration_scatter.png`, with summary stats in `v3/data/processed/garrett_resilience_interactions/garrett_baseline_additive_scatter_stats.csv`. For Garrett, the baseline-only simple OLS slope is `-1.698382`, exactly matching the archived `baseline_additive` slope in `v3/data/processed/garrett_resilience_interactions/condition_slopes.csv`; Pearson `r = -0.758756`, `R^2 = 0.575711`, and the shock-type-controlled coefficient is `-1.664524` with 95% CI `[-1.848075, -1.480973]`.
- 2026-08-05: Regenerated both Montgomery and Garrett baseline-additive concentration scatter PNGs with cleaner annotation layout after the user noted overlap. The shock-type legend now sits outside the plotting area at the upper right margin, and the beta/Pearson/R-squared stats box sits in the lower-right corner of the axes. File paths were unchanged; only figure layout was updated.
- 2026-08-05: Exported baseline-additive scatter point data for both counties to `v3/data/processed/baseline_additive_scatter_exports/baseline_additive_scatter_points.csv` with columns `county, shock_type, dependency_concentration, min_lcc`, plus per-county simple-OLS line endpoints in `v3/data/processed/baseline_additive_scatter_exports/baseline_additive_ols_line_endpoints.csv`. Endpoint values: Montgomery intercept `0.996124`, slope `-2.794055`, line from `(0.000000, 0.996124)` to `(0.201476, 0.433189)`; Garrett intercept `0.622808`, slope `-1.698382`, line from `(0.000000, 0.622808)` to `(0.235294, 0.223189)`.
- 2026-08-06: Exported all three semantic-condition scatter point tables for both counties to the user's Downloads folder as `semantic_scatter_points_all_conditions.csv`, with columns `county, condition, shock_type, dependency_concentration, min_lcc` rounded to 4 decimals. Also exported per-county, per-condition simple-OLS line endpoints to `semantic_scatter_ols_endpoints_all_conditions.csv` with rounded `ols_intercept`, `ols_slope`, `x_min`, `y_hat_at_x_min`, `x_max`, and `y_hat_at_x_max`.
- 2026-08-06: Computed the archived `HIGH - HIGH_RANDOM_MATCHED` `min_lcc` gap by semantic condition from the saved IP moderation parquet files using paired scenario-level differences and 95% t-based confidence intervals. Montgomery means/CIs: `baseline_additive 0.013511 [0.008720, 0.018303]`, `dampened_power_additive 0.013287 [0.008576, 0.017999]`, `redundant_additive 0.029811 [0.020599, 0.039022]`, `redundant_buffer -0.002530 [-0.004136, -0.000924]`. Garrett means/CIs: `baseline_additive 0.001731 [-0.000986, 0.004448]`, `dampened_power_additive 0.002692 [-0.000608, 0.005992]`, `redundant_additive 0.038654 [0.023194, 0.054113]`, `redundant_buffer -0.000385 [-0.002412, 0.001643]`.
- 2026-08-06: Exported the paired `HIGH - HIGH_RANDOM_MATCHED` gap-by-condition table to the user's Downloads folder as `high_minus_matched_random_gap_by_condition.csv`.
- 2026-08-07: Final session handoff saved. Confirmed Result 3 design from archived IP moderation outputs: every county × condition × profile cell has `n=100`; `HIGH` and `HIGH_RANDOM_MATCHED` were run on the same `scenario_metadata_ip.csv` scenarios (paired by `scenario_id` and `condition`); the reported gaps are means of paired per-scenario `HIGH - HIGH_RANDOM_MATCHED` differences; and the CI bounds in `~/Downloads/high_minus_matched_random_gap_by_condition.csv` were verified exactly against recomputation from the saved parquet files. Also confirmed the concrete IP profile numbers from `v3/configs/ip_capacity_profiles.json`: `LOW = tau 8, k 5`, `MED = tau 3, k 15`, `HIGH = tau 1, k 50`, with `HIGH_RANDOM_MATCHED` sharing `HIGH`'s latency/capacity settings but using random targeting plus matched realized intervention budget. Recent export artifacts available in `~/Downloads`: `semantic_scatter_points_all_conditions.csv`, `semantic_scatter_ols_endpoints_all_conditions.csv`, and `high_minus_matched_random_gap_by_condition.csv`.
- 2026-08-10: Refactored the policy-loading path to support named graph conditions and percentage-based IP capacity without launching a full experiment rerun. `v3/run_forced_exhaustive_search.py` now exposes real-file condition mappings for `plain`, `baseline_additive`, `redundant_additive`, and `redundant_buffer`, with default aggregation mode tied to condition (`redundant_buffer -> redundancy_buffer`, others additive). The policy entry points `v3/run_positive_indegree_policy_refresh.py`, `v3/run_shatter_impact_analysis.py`, `v3/run_delay12_policy_refresh.py`, and `v3/run_delay4_policy_refresh.py` now accept `--condition`, default to `plain`, and resolve aggregation mode from the chosen condition unless explicitly overridden. The delay refresh runners were also adjusted to preserve the selected graph condition's node/edge structure and only rewrite delays from the YAML rules, rather than silently rebuilding the plain graph from inventory. `v3/simulator_v3.py` and `v3/run_ip_moderation.py` now support `intervention_capacity_pct`, interpreted as `round(pct * total_edges_in_graph)` with percentage taking precedence over legacy integer capacity when present; `v3/configs/ip_capacity_profiles.json` now includes pct values `HIGH=0.04`, `MED=0.012`, `LOW=0.004` alongside the existing integer fields. Sanity check passed: all named condition mappings resolved to real files, `redundant_additive` loaded successfully for both counties (`Garrett 52 nodes / 170 sim edges`, `Montgomery 581 nodes / 2988 sim edges`), and percentage-based capacities on the current plain graphs resolve to `Garrett HIGH/MED/LOW = 3/1/0` and `Montgomery HIGH/MED/LOW = 54/16/5`.
- 2026-08-10: Adjusted the percentage-capacity denominator so it is pinned to each county's baseline edge count rather than the currently loaded condition's edge count. In [v3/simulator_v3.py], `intervention_capacity_for_graph(...)` now resolves `intervention_capacity_pct` against a county baseline denominator inferred from the baseline graph size fingerprint (`52 nodes -> 85 edges` for Garrett, `581 nodes -> 1355 edges` for Montgomery), falling back to the loaded edge count only for unknown graph sizes. Verified outcome: `HIGH/MED/LOW` now stay fixed across `plain` and `redundant_additive` for both counties. Garrett remains `3/1/0` even when loading `170` redundant edges, and Montgomery remains `54/16/5` even when loading `2988` redundant edges.
- 2026-08-10: Read-only comparison of the Montgomery policy graph (`plain`) versus the semantics/IP `redundant_additive` graph for the same power-seeded scenario parameters (`forcedsearch_002` / `ipfull_002`, both seeded by `power_plants::62910` with identical scales `severity=0.85`, `recovery=1.25`, `propagation=1.15`). In both graphs, the nearest school is one directed failure-propagation hop away from the seeded power node (`shortest path length = 1`) and the first school failure occurs at timestep `2`, consistent with the direct `school -> power` dependency edges having standardized delay `2`. The graphs are not otherwise identical: in the plain policy graph `power_plants::62910` has only `1` direct school successor, whereas in `redundant_additive` it has `27`; nonetheless the first school failures in the redundant graph still include many direct `school -> power` dependents at timestep `2`.
- 2026-08-10: Built an advisor-facing Montgomery explainer PNG at `v3/runs/figures/montgomery_t2_lcc_explainer.png` from archived semantics run `v3/data/processed/redundancy_v3/`, using real power-seeded `redundant_additive` scenario `v3_102` (seed `power_plants::60820`, no intervention). This scenario was chosen because the saved run truly has `t_min = 2` with a flat plateau afterward, unlike the newer positive-indegree policy refresh power scenarios which bottom out later. Panel 1 uses a representative two-hop failure-propagation neighborhood with `94` nodes (`44 telecom`, `42 school`, `6 EMS/fire`, `1 hospital`, `1 power`); all `42` schools in that sampled neighborhood fail at `t=2`. Panel 2 uses the saved LCC curve from `v3/data/processed/redundancy_v3/scenario_time_series_v3.csv` and was also exported to `v3/data/processed/montgomery_t2_lcc_explainer_lcc_curve.csv`; the key values are `t0=0.986231`, `t1=0.919105`, `t2=0.838210`, and then flat at `0.838210` thereafter.
- 2026-08-10: Read-only comparison of Garrett original vs delay-12 vs delay-4 policy refresh outputs to explain why delay recalibration did not materially improve outcomes. The graph structure again stayed fixed (`52` nodes, `85` edges), while timing stretched: Garrett `t_min` replay summaries moved from the earlier reference around `2` to delay-12 `median 1, mean 1.9, max 8` and delay-4 `median 1, mean 4.93, max 24`. Ladder changes remained small: original `none/priority/best = 0.4731 / 0.5090 / 0.4878`, delay-12 identical, and delay-4 only slightly higher at `0.4769 / 0.5090 / 0.4917`. Unlike Montgomery, Garrett did gain a little under 4h delays because some scenarios' minima shifted later, but the impact stayed modest because topology still dominated and many scenarios remained structurally impossible for intervention (`telecom 10/10 impossible` even under delay-4; power improved from `3/10` impossible in delay-12 to `1/10` in delay-4). Paired priority-minus-none differences remained largely the same scenario by scenario across delay-12 and delay-4, so spreading the timeline out did not create much extra leverage for a whole-run budget of `B=3`.
- 2026-08-11: Saved current `redundant_additive` policy-experiment headline findings before moving on to substation planning. Completed result sets:
  - `v3/data/processed/garrett_redundant_additive_positive_indegree_refresh_N30_B3/`
  - `v3/data/processed/garrett_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3/`
  - `v3/data/processed/garrett_redundant_additive_delay4_refresh_N30_B3/`
  - `v3/data/processed/montgomery_redundant_additive_positive_indegree_refresh_N30_B3/`
  - `v3/data/processed/montgomery_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3/`
  Headline ladders (`min_lcc`, mean ± sd, whole-run `B=3`):
  - Garrett original redundant_additive:
    - none `0.7288 ± 0.2071`
    - priority `0.7872 ± 0.2063`
    - best-combinatorial `0.7378 ± 0.2091`
    - shatter_impact_topk `0.7365 ± 0.2058`
  - Garrett redundant_additive @ 4h:
    - none `0.7192 ± 0.1940`
    - priority `0.7814 ± 0.1771`
    - best-combinatorial `0.7340 ± 0.1897`
    - shatter_impact_topk `0.7205 ± 0.1939`
  - Montgomery original redundant_additive:
    - none `0.8245 ± 0.2227`
    - priority `0.8394 ± 0.2109`
    - best-combinatorial `0.8320 ± 0.2137`
    - shatter_impact_topk `0.8246 ± 0.2225`
  Additional finished diagnostics:
  - Garrett 4h paired differences:
    - priority minus none: mean `+0.0622`, sd `0.0628`, better in `27/30`
    - best minus none: mean `+0.0147`, sd `0.0187`, better in `14/30`
    - shatter_impact minus none: mean `+0.0013`, sd `0.0049`, better in `4/30`
  - Garrett 4h `t_min`: min `0`, median `2`, mean `7.67`, max `24`
  - Montgomery original priority baseline used almost the full whole-run budget on average: mean intervention count `2.9`
  Current interpretation from completed arms:
  - dynamic priority is the strongest policy in every finished redundant_additive arm so far
  - best fixed combinatorial edge sets help somewhat but do not beat dynamic priority
  - shatter-impact remains essentially flat with no-intervention in the completed redundant_additive runs
  - one result remains open: `v3/data/processed/montgomery_redundant_additive_delay4_refresh_N30_B3/`
  - this open Montgomery 4h arm was later confirmed to still be actively running as child PID `97212` with command `Python v3/run_delay4_policy_refresh.py --county montgomery --condition redundant_additive --confirm-run`
- 2026-08-11: Final wrap note for the `redundant_additive` policy batch. All four target arms are now complete and stored here:
  - Garrett original: `v3/data/processed/garrett_redundant_additive_positive_indegree_refresh_N30_B3/`
  - Garrett impact-based shatter comparison: `v3/data/processed/garrett_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3/`
  - Garrett 4h delay refresh: `v3/data/processed/garrett_redundant_additive_delay4_refresh_N30_B3/`
  - Montgomery original: `v3/data/processed/montgomery_redundant_additive_positive_indegree_refresh_N30_B3/`
  - Montgomery impact-based shatter comparison: `v3/data/processed/montgomery_redundant_additive_shatter_impact_map_positive_indegree_N1000_B3/`
  - Montgomery 4h delay refresh: `v3/data/processed/montgomery_redundant_additive_delay4_refresh_N30_B3/`
  Most useful output files to revisit:
  - policy ladders:
    - `.../new_policy_ladder.csv` in each refresh folder
    - `.../policy_ladder.csv` in each shatter-impact folder
  - paired differences:
    - `.../paired_policy_difference_summary.csv`
  - timing summaries:
    - `.../delay4_summary.json` for delay4 runs
  Final headline numbers (`mean min_lcc ± sd`, whole-run `B=3`):
  - Garrett original:
    - none `0.7288 ± 0.2071`
    - priority `0.7872 ± 0.2063`
    - best-combinatorial `0.7378 ± 0.2091`
    - shatter_impact_topk `0.7365 ± 0.2058`
  - Garrett 4h:
    - none `0.7192 ± 0.1940`
    - priority `0.7814 ± 0.1771`
    - best-combinatorial `0.7340 ± 0.1897`
    - shatter_impact_topk `0.7205 ± 0.1939`
  - Montgomery original:
    - none `0.8245 ± 0.2227`
    - priority `0.8394 ± 0.2109`
    - best-combinatorial `0.8320 ± 0.2137`
    - shatter_impact_topk `0.8246 ± 0.2225`
  - Montgomery 4h:
    - none `0.8184 ± 0.2190`
    - priority `0.8283 ± 0.2163`
    - best-combinatorial `0.8215 ± 0.2169`
    - shatter_impact_topk `0.8185 ± 0.2187`
  Timing/takeaway:
  - `t_min` remains very early in both counties even after 4h delays:
    - Garrett 4h: min `0`, median `2`, mean `7.67`, max `24`
    - Montgomery 4h: min `0`, median `2`, mean `12.33`, max `36`
  - Dynamic priority is the strongest policy in every completed arm.
  - Best fixed combinatorial sets help somewhat but do not beat dynamic priority.
  - Shatter-impact remains effectively flat with no-intervention in these runs.
  - The next research move is to test a richer network structure (for example adding substation/transmission layers) to see whether `t_min` can be pushed materially later than about `2`.
- 2026-08-11: Began the network-upgrade track by adding authoritative Montgomery County boundary and a read-only HIFLD transmission-line inspection layer. Downloaded Montgomery County directly from Maryland iMAP county boundaries service and saved:
  - `v3/data/raw/boundaries/montgomery_county_boundary.geojson`
  Sanity check on the saved boundary:
  - geometry type: `Polygon`
  - CRS requested/exported in EPSG `4326`
  - bounds: `xmin -77.52767798612213`, `ymin 38.93423326473835`, `xmax -76.88765787381035`, `ymax 39.35431959242619`
  - approximate area from equirectangular projection: `1312.07 km^2`
  Added read-only inspection script:
  - `v3/inspect_transmission_lines.py`
  This script clips the HIFLD transmission-lines parquet to a real county polygon, prefers `STATUS == "IN SERVICE"`, and writes line/substation inspection outputs without touching graph/simulator code.
  Ran it against:
  - boundary: `v3/data/raw/boundaries/montgomery_county_boundary.geojson`
  - transmission parquet: `v3/data/raw/transmission_lines/transmission-lines.parquet`
  Outputs written under:
  - `v3/data/processed/transmission_inspection_montgomery/`
  Files:
  - `montgomery_transmission_lines.parquet`
  - `montgomery_substations.csv`
  - `montgomery_substation_edges.csv`
  - `montgomery_substation_degree_top30.csv`
  - `inspection_summary.json`
  Headline transmission inspection counts:
  - bbox candidates: `163`
  - true intersecting transmission lines: `163`
  - `IN SERVICE` lines: `126`
  - final exported lines (after STATUS preference): `126`
  - unique named substations: `79`
  - null/blank `SUB_1`: `0`
  - null/blank `SUB_2`: `0`
  Top connected substations so far:
  - `BURTONSVILLE 15`
  - `QUINCE ORCHARD 14`
  - `DICKERSON 13`
  - `DOUBS 13`
  - `MOUNT ZION 12`
  - `BRIGHTON 11`
  Immediate modeling takeaway:
  - the transmission-lines layer has no clean county/state columns, so county filtering should be done by geometric intersection, not attribute filtering
  - `SUB_1` / `SUB_2` appear useful enough to start building a substation-centered augmentation of the power network
- 2026-08-12: Added a second read-only transmission utility:
  - `v3/derive_substation_locations.py`
  It derives one latitude/longitude per named substation by clustering the endpoints of all incident transmission lines, without assuming geometry start maps to `SUB_1`.
  Outputs written under:
  - `v3/data/processed/transmission_inspection_montgomery/derived_substation_locations.csv`
  - `v3/data/processed/transmission_inspection_montgomery/derived_substation_location_report.md`
  - `v3/data/processed/transmission_inspection_montgomery/derived_substation_location_summary.json`
  Headline geolocation result:
  - successfully geolocated substations: `79 / 79`
  - inside Montgomery County polygon: `15`
  - outside Montgomery County polygon: `64`
  - unresolved substations: `0`
  - inconsistent substations above the `2.0 km` endpoint-spread threshold: `0`
  Inside-Montgomery names:
  - `QUINCE ORCHARD`
  - `DICKERSON`
  - `MOUNT ZION`
  - `BRIGHTON`
  - `BETHESDA`
  - `BELLS MILL`
  - `NORBECK`
  - `CLARKSBURG`
  - `RISER176098`
  - `RISER176604`
  - `UNKNOWN122896`
  - `UNKNOWN122905`
  - `MONTGOMERY COUNTY RESOURCE RECOVERY`
  - `RISER176861`
  - `RISER176901`
  Important note:
  - the first inside/outside pass wrongly marked all `79` substations as inside because the old manual ring test was overpermissive for points. The saved final outputs are from the corrected polygon-path containment check and are the trustworthy version.
- 2026-08-12: Added a read-only power-to-substation matching audit on top of the current Montgomery graph:
  - script: `v3/inspect_power_plant_substation_connections.py`
  - outputs:
    - `v3/data/processed/transmission_inspection_montgomery/power_plant_substation_connections.csv`
    - `v3/data/processed/transmission_inspection_montgomery/power_plant_receiving_substations.csv`
    - `v3/data/processed/transmission_inspection_montgomery/power_plant_substation_connection_report.md`
    - `v3/data/processed/transmission_inspection_montgomery/power_plant_substation_connection_summary.json`
  - inputs stay read-only:
    - existing power nodes from `v3/data/processed/dependency_graph_nodes.csv`
    - derived substation coordinates and Montgomery transmission topology from `v3/data/processed/transmission_inspection_montgomery/`
  - matching rule:
    - prefer actual nearby transmission-line geometry plus named line endpoints
    - do not force a nearest-substation match when the line evidence is weak
  - headline result:
    - total existing power plants: `15`
    - successfully connected: `10`
    - unmatched: `5`
    - ambiguous: `0`
  - substations receiving plant connections:
    - `MOUNT ZION` `4`
    - `QUINCE ORCHARD` `2`
    - `DICKERSON` `1`
    - `CLARKSBURG` `1`
    - `MONTGOMERY COUNTY RESOURCE RECOVERY` `1`
    - `BETHESDA` `1`
  - unmatched plants:
    - `BROOKVILLE SMART BUS DEPOT MICROGRID`
    - `CENTRAL UTILITY PLANT AT WHITE OAK`
    - `MO32 (CSG)`
    - `MONTGOMERY COUNTY SOLAR`
    - `NIST SOLAR`
  - sanity check:
    - the `DICKERSON` power plant matched naturally to the `DICKERSON` substation on transmission line `127374` at `0.762 km`, confidence `high`.
- 2026-08-12: Built and ran the first Montgomery A/B substation timing experiment in a separate branch:
  - script: `v3/run_substation_timing_experiment.py`
  - output folder: `v3/data/processed/substation_timing_experiment/`
  - key artifacts:
    - `graph_validation_report.md`
    - `path_comparison.csv`
    - `paired_scenario_results.csv`
    - `sector_failures_by_timestep.csv`
    - `timestep_diagnostics.csv`
    - `summary_report.md`
    - experiment graph files: `baseline_*` and `substation_*`
  Experimental graph design:
  - removed all direct Montgomery `school/hospital/telecom/ems_fire -> power` edges
  - added `79` substation nodes from the HIFLD-derived topology
  - added `252` directed substation-substation transmission dependency edges (bidirectional representation of `126` line records)
  - used `10` matched power-plant-to-substation connections from the read-only audit
  - assigned each infra node to its nearest inside-Montgomery substation(s), preserving the original number of direct power supports per node
  Validation:
  - baseline direct power edges:
    - `school -> power = 211`
    - `hospital -> power = 22`
    - `telecom -> power = 593`
    - `ems_fire -> power = 42`
  - substation graph direct power edges after replacement:
    - all four counts reduced to `0`
  - concentration warning:
    - largest substation assignment share = `30.1%`
    - top fan-out substation = `BELLS MILL` with `170` unique assigned infrastructure nodes
    - this was flagged as implausibly concentrated in `graph_validation_report.md`
  Path-test result (failure-propagation graph):
  - unique power plants used by the saved power scenarios: `9`
  - plants with no reachable schools in substation graph: `3` (`BROOKVILLE SMART BUS DEPOT MICROGRID`, `MO32 (CSG)`, `MONTGOMERY COUNTY SOLAR`) because they remained unmatched in the conservative plant-to-substation audit
  - for matched plants that still reach schools, shortest power-to-school path changed from about `1-2` hops in baseline to `2-7` hops in the substation graph, with path delays increasing from about `2-3` to about `3-8`
  Controlled replay on the saved `10` Montgomery power scenarios (`forcedsearch_001` to `forcedsearch_010`):
  - baseline:
    - median `first_school_failure_t = 2`
    - median `t_min_lcc = 4`
    - mean `min_lcc = 0.7556`
    - mean `peak_damage_nodes = 142.0`
  - substation:
    - median `first_school_failure_t = 3`
    - median `t_min_lcc = 8`
    - mean `min_lcc = 0.3005`
    - mean `peak_damage_nodes = 441.3`
  - paired deltas (`substation - baseline`):
    - mean `first_school_failure_t` on matched scenarios where schools still fail in both conditions: `+1.0`
    - mean `t_min_lcc = +1.8`
    - mean `min_lcc = -0.4551`
    - mean `peak_damage_nodes = +299.3`
  Interpretation of this first A/B:
  - the trough did move later on matched plants, so the added intermediate topology increased propagation distance
  - but the graph also became much harsher overall, not milder, because the nearest-substation assignment concentrated too many infrastructure nodes onto a small set of substations
  - this means the current result does **not** isolate a clean “more realistic topology only” effect yet; it mixes distance with strong new concentration/fan-out artifacts
  Representative timestep diagnostics:
  - `diag_dickerson`, `diag_resource_recovery`, `diag_nih_cogen`, and `diag_worst_forcedsearch_003` are saved in `timestep_diagnostics.csv`
  - these show the same pattern: baseline school failures begin around `t=2-3`, while the substation graph often delays the first school failures by about one timestep but then collapses many more schools once the concentrated local substations fail
- 2026-08-12: Added a read-only nearest-substation scope comparison to test whether restricting assignments to inside-county substations caused the concentration problem:
  - script: `v3/compare_substation_assignment_scopes.py`
  - output folder: `v3/data/processed/transmission_inspection_montgomery/substation_assignment_scope_comparison/`
  - key files:
    - `assignment_scope_comparison_report.md`
    - `nearest_substation_assignments_all_scopes.csv`
    - `scope_summary.csv`
    - `substation_fanout_by_scope.csv`
    - `outside_substations_receiving_assignments.csv`
    - `bells_mill_scope_comparison.csv`
  Comparison setup:
  - infrastructure nodes: `564` (`school`, `hospital`, `telecom`, `ems_fire`)
  - one nearest-substation assignment per infrastructure node
  - three substation pools:
    - A = inside Montgomery only (`15`)
    - B = inside + outside substations within `10 km` of county boundary (`61`)
    - C = all `79` transmission-connected substations
  Headline result:
  - B and C are identical in this dataset, meaning every outside-county substation that actually becomes the nearest assignment target is already within `10 km` of the county boundary. The extra farther-away substations in C do not attract any Montgomery infrastructure nodes.
  Scope summaries:
  - A inside-only:
    - inside assignments `564`, outside assignments `0`
    - distance km: median `5.525`, mean `5.597`, p90 `9.506`, max `15.601`
    - max fan-out `128` at `BELLS MILL`
  - B inside + boundary-10km:
    - inside assignments `446`, outside assignments `118`
    - distance km: median `4.540`, mean `4.660`, p90 `7.773`, max `10.515`
    - max fan-out `116` at `BELLS MILL`
  - C all 79:
    - identical to B on all assignment and distance metrics
  BELLS MILL comparison:
  - current multiplicity-preserving experiment assignment: `170` unique infrastructure nodes
  - nearest-only inside-A assignment: `128`
  - nearest-only B/C assignment: `116`
  Interpretation:
  - allowing near-boundary outside substations reduces distance and slightly relieves the worst local concentration, but it does not remove `BELLS MILL` as the largest fan-out node
  - the big concentration problem in the first substation A/B test was partly caused by inside-only restriction, but not fully; even after allowing nearby outside substations, `BELLS MILL` still carries `116 / 564 = 20.6%` of all infrastructure in the nearest-only assignment
  Outside-county substations that pick up Montgomery infrastructure under B/C:
  - strongest receivers: `METZEROTT ROAD (41)`, `BURTONSVILLE (21)`, `TOKOMA (17)`, then `SWINKS MILL (7)`, `CIA (6)`
  - all of these are close to the county boundary (`0.15 km` to `4.83 km`)
- 2026-08-12: Added a read-only voltage/role audit for the substations that actually receive Montgomery infrastructure under scope B (`inside + within 10 km of county boundary`):
  - output folder reused: `v3/data/processed/transmission_inspection_montgomery/substation_assignment_scope_comparison/`
  - new files:
    - `substation_role_voltage_audit_scopeB.csv`
    - `rank_by_fanout_scopeB.csv`
    - `rank_by_degree_scopeB.csv`
    - `rank_by_max_voltage_scopeB.csv`
    - `rank_by_mean_distance_scopeB.csv`
  Clarification:
  - the scope-B nearest assignment uses `25` receiving substations, not `15`
  BELLS MILL audit:
  - inside Montgomery = `True`
  - incident transmission lines = `6`
  - voltage range = `138 kV` to `230 kV`
  - voltage classes = `100-161; 220-287`
  - nearest-assignment fan-out = `116`
  - mean assigned-infrastructure distance = `5.02 km`
  Interpretation:
  - BELLS MILL behaves like a transmission / subtransmission node, not a clearly local low-voltage service substation
  - the other high-fanout nodes are similar: `BETHESDA` is `138 kV`, `MOUNT ZION` is `230 kV`, `METZEROTT ROAD` is `230 kV`, `BRIGHTON` reaches `500 kV`, and `QUINCE ORCHARD` is `230 kV`
  - this strongly suggests the HIFLD transmission-line-derived topology is missing the lower-voltage distribution substations that would normally sit between these bulk nodes and end users like schools
- 2026-08-13: Read-only readiness audit of the finalized Montgomery power hierarchy base using `build_final_hierarchy_graph(...)` from [v3/run_final_power_timing_experiment.py](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/v3/run_final_power_timing_experiment.py:259).
  - Final base graph counts:
    - nodes `717`
    - edges `1338`
    - simulation edges after normalization/aggregation `1269`
  - Node counts by tier:
    - `power 15`, `transmission_substation 61`, `distribution_substation 75`, `school 211`, `hospital 11`, `telecom 300`, `ems_fire 42`, `emergency_management 2`
  - Edge counts by tier:
    - `telecom -> distribution_substation 300`
    - `school -> distribution_substation 211`
    - `school -> ems_fire 211`
    - `school -> telecom 211`
    - `transmission_substation -> transmission_substation 202`
    - `distribution_substation -> transmission_substation 75`
    - `ems_fire -> distribution_substation 42`
    - `hospital -> ems_fire 22`
    - `hospital -> telecom 22`
    - `hospital -> distribution_substation 11`
    - `hospital -> emergency_management 11`
    - `emergency_management -> ems_fire 10`
    - `transmission_substation -> power 10`
  - Connectivity verdict:
    - all consumer nodes (`school`, `hospital`, `telecom`, `ems_fire`) have a directed path to at least one power node
    - no direct consumer-to-power edges remain (`0`)
    - the base is not fully complete on the generation side: `5` of `15` power plants have no downstream transmission-substation connection
    - unmatched plants:
      - `BROOKVILLE SMART BUS DEPOT MICROGRID`
      - `MONTGOMERY COUNTY SOLAR`
      - `NIST SOLAR`
      - `CENTRAL UTILITY PLANT AT WHITE OAK`
      - `MO32 (CSG)`
  - Variant-builder verdict:
    - current builder [v3/build_graph_variants.py](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/v3/build_graph_variants.py:15) still rebuilds variants from `asset_inventory.parquet` point layers plus YAML dependency rules, via `POINT_LAYERS` / `build_edges(...)` in [v3/build_dependency_graph.py](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/v3/build_dependency_graph.py:11)
    - it does not yet accept the finalized multi-tier hierarchy base (`transmission_substation` + `distribution_substation`) as a direct input the way the old network was built
- 2026-08-13: Built a parallel hierarchy semantic-variant builder for Montgomery in [v3/build_hierarchy_graph_variants.py](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/v3/build_hierarchy_graph_variants.py:1). This does not touch the legacy `graph_variants` folders.
  - New output folder: `v3/data/processed/montgomery_hierarchy_graph_variants/`
  - Frozen finalized hierarchy base:
    - nodes `717`
    - edges `1338`
    - sim edges `1269`
  - Built hierarchy semantic variants with redundancy only on `distribution_substation -> transmission_substation`:
    - `baseline_additive`: one upstream transmission feed per distribution substation
    - `redundant_additive`: two upstream transmission feeds per distribution substation
    - `redundant_buffer`: same two-feed topology with `redundancy_buffer` aggregation
  - Variant counts:
    - `baseline_additive`: nodes `717`, edges `1338`, sim edges `1269`
    - `redundant_additive`: nodes `717`, edges `1413`, sim edges `1344`
    - `redundant_buffer`: nodes `717`, edges `1413`, sim edges `1344`
  - Loader confirmation:
    - existing semantics loader in `run_redundancy_comparison.py` can point at this folder via `--graph-dir`; no overwrite or loader rewrite required for the sanity step
  - One-scenario sanity check used existing power-seeded scenario `forcedsearch_002` (`MNCPPC GERMANTOWN SOLAR`)
    - `baseline_additive`: `min_lcc = 0.0976`, `t_min = 11`
    - `redundant_buffer`: `min_lcc = 0.0446`, `t_min = 113`
    - result did **not** satisfy the expected direction on `min_lcc`; buffer delayed the trough substantially and improved `auc_resilience`, but the eventual trough was deeper
  - Current interpretation:
    - the hierarchy variant machinery works and the redundancy was not ignored
    - however, under the current metric (`min_lcc`), the one-scenario sanity check failed the expected “buffer should be better” criterion, so no full semantics rerun was launched
- 2026-08-13: Tightened Montgomery power-shock seeding for the hierarchy semantic variants to exclude distributed generation by raw HIFLD plant type, not just transmission proximity.
  - Raw source used: `v3/data/raw/power_plants/power_plants.parquet`
  - Exclusion rule:
    - exclude any plant classified as solar / photovoltaic, CSG, microgrid, or campus utility plant
    - retain only transmission-connected plants that are not distributed by type
  - Corrected seedable bulk list saved to:
    - `v3/data/processed/montgomery_hierarchy_graph_variants/power_seed_groups.csv`
    - `v3/data/processed/montgomery_hierarchy_graph_variants/bulk_transmission_connected_power_plants.csv`
    - `v3/data/processed/montgomery_hierarchy_graph_variants/excluded_distributed_backup_power_plants.csv`
  - Corrected bulk-seedable plants count: `4`
    - `DICKERSON`
    - `MONTGOMERY COUNTY RESOURCE RECOVERY`
    - `NIH COGENERATION FACILITY`
    - `MONTGOMERY COUNTY OAKS LFGE PLANT`
  - Re-ran one-scenario hierarchy sanity check on genuine bulk plant `DICKERSON` across all three semantic conditions:
    - `baseline_additive`: `min_lcc = 0.0976`, `t_min = 10`, `auc = 0.1279`
    - `redundant_additive`: `min_lcc = 0.0084`, `t_min = 10`, `auc = 0.0411`
    - `redundant_buffer`: `min_lcc = 0.0181`, `t_min = 147`, `auc = 0.5007`
  - Time-aware pass criterion result:
    - `redundant_buffer` improves `t_min` and `auc` versus `baseline_additive`
    - `redundant_buffer` also improves `auc` versus `redundant_additive`
    - overall sanity verdict: `PASS` on time-aware resilience (`t_min`, `auc_resilience`), even though `min_lcc` remains lower than baseline
- 2026-08-13: Ran a small Montgomery hierarchy semantic pilot with the corrected bulk-power seed filter.
  - New runner: [v3/run_hierarchy_semantic_pilot.py](/Users/ramnathsankaran/Library/CloudStorage/GoogleDrive-ramnath217@gmail.com/My%20Drive/Spring%202026/GraphTransformer/v3/run_hierarchy_semantic_pilot.py:1)
  - Output folder: `v3/data/processed/montgomery_hierarchy_semantic_pilot_N30/`
  - Shared pilot scenario plan: `10 power`, `10 telecom`, `5 ems`, `5 mixed` = `30` total scenarios per condition
  - Bulk power seed pool used: `4` plants
    - `DICKERSON`
    - `MONTGOMERY COUNTY OAKS LFGE PLANT`
    - `MONTGOMERY COUNTY RESOURCE RECOVERY`
    - `NIH COGENERATION FACILITY`
  - Distinct power-seed diversity in the pilot:
    - distinct individual bulk plants used: `4 / 4`
    - distinct multi-seed power sets used: `6`
    - usage counts:
      - `NIH COGENERATION FACILITY`: `6`
      - `DICKERSON`: `5`
      - `MONTGOMERY COUNTY RESOURCE RECOVERY`: `5`
      - `MONTGOMERY COUNTY OAKS LFGE PLANT`: `2`
  - AUC fragility slopes (`auc_resilience ~ dependency_concentration`) across the 30-scenario pilot:
    - `baseline_additive`: slope `24.3209`, mean AUC `0.5471`
    - `redundant_additive`: slope `27.4055`, mean AUC `0.5088`
    - `redundant_buffer`: slope `10.2693`, mean AUC `0.8019`
  - Pilot interpretation:
    - the semantic ordering holds at pilot scale on AUC: `redundant_buffer` is flattest and most resilient, `redundant_additive` is steepest/worst, `baseline_additive` sits in between
    - power-seed diversity is limited but not degenerate: all four allowed bulk plants appeared, though usage is uneven
- 2026-08-15: Built Montgomery hierarchy AUC policy runner `v3/run_hierarchy_policy_experiment.py` for `redundant_additive` using the 4 bulk power seeds, HIGH resolved off hierarchy baseline sim-edge count (`1269` -> `51`), whole-run budget `B=3`, and 30 paired scenarios. Outputs written under `v3/data/processed/_hierarchy_policy_smoke`.
- 2026-08-15: Built Montgomery hierarchy AUC policy runner `v3/run_hierarchy_policy_experiment.py` for `redundant_additive` using the 4 bulk power seeds, HIGH resolved off hierarchy baseline sim-edge count (`1269` -> `51`), whole-run budget `B=3`, and 30 paired scenarios. Outputs written under `v3/data/processed/montgomery_hierarchy_redundant_additive_policy_auc_N30_B3`.
- 2026-08-19: Overnight Montgomery telecom-hierarchy weighted run completed successfully. Master log: `v3/logs/montgomery_telecom_weighted_overnight_master.log`.
  - Semantic weighted experiment output: `v3/data/processed/montgomery_telecom_hierarchy_broad_semantic_weighted_N500/`
    - Mean weighted resilience: `redundant_buffer 79.3133`, `baseline_additive 67.8911`, `redundant_additive 66.7116`
    - Mean min LCC: `0.7488`, `0.6469`, `0.6534` respectively
    - Mean AUC resilience: `0.8052`, `0.6667`, `0.6722`
    - Paired differences vs baseline: `buffer +11.4222`, `redundant_additive -1.1794`
  - Policy weighted experiment output: `v3/data/processed/montgomery_telecom_hierarchy_redundant_additive_policy_weighted_N30_B3/`
    - Mean weighted resilience ladder: `priority 84.4105`, `best_combinatorial 72.3017`, `none 65.8246`, `shatter_impact_topk 65.8246`
    - Mean AUC resilience ladder: `priority 0.8555`, `best_combinatorial 0.7400`, `none 0.6754`, `shatter 0.6755`
    - Paired vs none: `priority +18.5858 (30/30 better)`, `best +6.4770 (9/30 better)`, `shatter +0.0000 (0/30 better)`
  - Logs:
    - `v3/logs/montgomery_telecom_broad_semantic_weighted_N500.log`
    - `v3/logs/montgomery_telecom_redundant_additive_policy_weighted_N30_B3.log`
- 2026-08-21: Diagnosed why `shatter_impact_topk` was effectively equal to `none` in `v3/data/processed/montgomery_telecom_hierarchy_redundant_additive_policy_weighted_N30_B3/`.
  - The shatter map was regenerated in-run and produced `1553` ranked edges; the actual forced list was exactly `3` edges because the policy uses `edge_impact_df.head(B)` with `B=3`, not HIGH's percentage capacity.
  - HIGH resolved to `59` from `0.04 * 1478` baseline sim edges, but that only sets per-step candidate capacity; whole-run protection was still capped at `B=3`.
  - Top-3 forced edges:
    - `SWINKS_MILL -> CIA`
    - `SWINKS_MILL -> distribution_substations::node_6065215158`
    - `CIA -> TAP176570`
  - In the N30 policy set, only `8/30` scenarios ever routed through the `SWINKS_MILL/CIA/TAP176570` corridor at all; those were `broadpolicy_001` through `broadpolicy_008`.
  - Concrete original-1000-run fan-out check on `broadimpact_001`:
    - top edge `SWINKS_MILL -> CIA` downstream closure = `649` nodes (`211` schools, `11` hospitals, `42` EMS, `8` telecom exchanges, `256` towers, `72` distribution substations, `49` transmission substations), weighted service total `3825`
    - low-ranked edge `UNKNOWN116991 -> way_436236811` downstream closure = `3` nodes (`1` distribution substation, `2` towers), weighted service total `4`
  - Reconciliation:
    - the top edge is genuinely high-impact conditional on its source failing, but in many N30 scenarios that corridor was either not reached or was reached too late; e.g. in `broadpolicy_001` shatter first acted at `t=11` after the weighted/LCC collapse had already bottomed out.
- 2026-08-21: Traced `broadpolicy_005` at node-path level relative to shatter's protected edges `SWINKS_MILL -> CIA` and `SWINKS_MILL -> distribution_substations::node_6065215158`.
  - The protected-target descendant basin has `649` nodes; `647` of them fail under `NONE` in this scenario.
  - Path classification of those `647` failed descendants:
    - `622` failed via alternate unprotected paths only
    - `19` had both an alternate unprotected path and a path using the protected edges
    - `6` were protected-edge-only
  - Crucially, the `6` protected-edge-only nodes were all upstream zero-weight nodes: `CIA`, `TAP176570`, `K06TP1 Fisher Ave TPSS`, `Falls Church Substation`, `Livingston Heights Substation`, and `way_1184743628`.
  - Under `SHATTER`, all 6 protected-edge-only nodes survive, but no hospitals, schools, EMS, telecom towers, or telecom exchanges were uniquely dependent on those edges in this scenario; all positive-weight consumer-sector failures had alternate unprotected routes from the seeds.
- 2026-08-21: Fixed the Montgomery telecom-hierarchy shatter-arm budget bug in `v3/run_montgomery_telecom_broad_policy_weighted.py`.
  - Old behavior: shatter selected `head(B)` edges and used whole-run budget `B=3`.
  - New behavior: shatter selects `head(high_capacity)` edges and uses whole-run budget `high_capacity`, where HIGH resolves off baseline sim-edge count.
  - Montgomery resolved HIGH = `59` edges from `0.04 * 1478`.
  - Reran only the shatter arm on the saved `redundant_additive`, `N=30` scenario set and reused the original `none`, `priority`, and `best_combinatorial` outputs.
  - New output dir: `v3/data/processed/montgomery_telecom_hierarchy_redundant_additive_policy_weighted_N30_shatter_highcapacity_fix/`
  - Updated ladder:
    - `priority`: weighted `84.4105`, AUC `0.8555`
    - `best_combinatorial`: weighted `72.3017`, AUC `0.7400`
    - `shatter_impact_topk` corrected: weighted `66.3206`, AUC `0.6786`, mean interventions `17.9667`
    - `none`: weighted `65.8246`, AUC `0.6754`
  - Corrected paired differences:
    - `priority_minus_none`: `+18.5858`, `30/30` better
    - `shatter_minus_none`: `+0.4960`, `18/30` better
- 2026-08-21: Ready to move this same shatter-budget fix workflow to Garrett next.
  - Garrett action plan:
    - apply the same corrected shatter budget logic (`head(high_capacity)`, total shatter budget = resolved HIGH)
    - resolve Garrett HIGH off baseline sim-edge count
    - rerun only the Garrett shatter arm on the saved `redundant_additive`, `N=30` scenario set
    - reuse existing Garrett `none`, `priority`, and `best_combinatorial` outputs to rebuild the ladder
- 2026-08-21: Audited Garrett readiness for porting the Montgomery hierarchy pipeline.
  - Garrett still only has the flat-network assets in place:
    - config: `v3/configs/infrastructure_layers_garrett.yaml`
    - base graph: `v3/data/processed/garrett_dependency_graph_nodes.csv`, `v3/data/processed/garrett_dependency_graph_edges.csv`, `v3/data/processed/garrett_simulation_edges.csv`
    - old flat graph variants: `v3/data/processed/garrett_graph_variants/`
  - Montgomery hierarchy builders are partly reusable but the raw county inputs are still Montgomery-only:
    - present raw inputs: `v3/data/raw/boundaries/montgomery_county_boundary.geojson`, `v3/data/raw/transmission_lines/transmission-lines.parquet`, `v3/data/raw/osm_substations/montgomery_osm_substations_overpass.json`, `v3/data/raw/osm_substations/montgomery_osm_electrical_network_overpass.json`, `v3/data/raw/osm_telecom/montgomery_osm_telecom_overpass.json`
    - no Garrett-specific boundary, transmission inspection, OSM substation, or OSM telecom files are present yet
  - Reusable or nearly reusable hierarchy scripts:
    - `v3/run_final_power_timing_experiment.py`: contains the generic `build_final_hierarchy_graph(...)` constructor used by later hierarchy workflows
    - `v3/build_hierarchy_graph_variants.py`: builds baseline/redundant variants from a finalized hierarchy base, but its default file paths are Montgomery-specific
    - `v3/run_hierarchy_policy_experiment.py`: hierarchy policy runner, currently hardwired to Montgomery hierarchy directories and Montgomery scenario folders
  - Clearly Montgomery-specific scripts/inputs that cannot run for Garrett as-is:
    - `v3/inspect_transmission_lines.py` plus Montgomery boundary/input defaults
    - `v3/derive_substation_locations.py`
    - `v3/build_final_power_hierarchy_validation.py`
    - `v3/finalize_power_hierarchy_cleanup.py`
    - `v3/fetch_osm_montgomery_substations.py`
    - `v3/fetch_osm_montgomery_telecom.py`
    - `v3/build_montgomery_telecom_hierarchy_variants.py`
    - `v3/run_montgomery_telecom_*` weighted experiment runners
  - First artifact Garrett needs before any hierarchy build can start:
    - a Garrett County boundary GeoJSON, then Garrett-specific transmission-line intersection outputs and OSM substation/telecom extracts
  - Practical Garrett port sequence:
    1. create Garrett boundary file
    2. inspect/filter transmission lines for Garrett
    3. derive Garrett transmission substations and substation edges
    4. fetch/extract Garrett OSM substations and telecom facilities
    5. validate power hierarchy and telecom hierarchy for Garrett
    6. build Garrett hierarchy graph variants
    7. run Garrett semantic/policy experiments on the hierarchy network
- 2026-08-21: Started the Garrett hierarchy port and completed the first county-specific geometry steps.
  - Created authoritative Garrett County boundary from U.S. Census TIGERweb and saved:
    - `v3/data/raw/boundaries/garrett_county_boundary.geojson`
    - feature IDs confirmed: `GEOID=24023`, `STATE=24`, `COUNTY=023`, `NAME=Garrett County`
    - sanity bounds: `(-79.4876510, 39.2021070, -78.9284160, 39.7228830)`
    - approximate area from the saved EPSG:4326 polygon: `1705.64 km^2`
  - Ran the existing transmission inspection pipeline on Garrett using the new boundary:
    - output dir: `v3/data/processed/transmission_inspection_garrett/`
    - summary:
      - candidate bbox matches: `61`
      - intersecting transmission lines: `61`
      - `IN SERVICE` lines retained: `40`
      - unique named substations from those lines: `44`
      - top degree node: `MOUNT STORM` with degree `6`
    - note: the script still writes Montgomery-named filenames inside the Garrett output dir:
      - `montgomery_transmission_lines.parquet`
      - `montgomery_substations.csv`
      - `montgomery_substation_edges.csv`
      - `montgomery_substation_degree_top30.csv`
      This is only a naming quirk so far, not a data-content problem.
  - Ran first-pass substation geolocation on the Garrett transmission subset:
    - outputs:
      - `v3/data/processed/transmission_inspection_garrett/derived_substation_locations.csv`
      - `v3/data/processed/transmission_inspection_garrett/derived_substation_location_report.md`
    - summary:
      - successfully geolocated substations: `42`
      - inside-county substations: `13`
      - unresolved substations: `2`
      - inconsistent substations above the endpoint-spread threshold: `0`
    - caveat: the inherited report field still says `inside_montgomery`; this needs a county-neutral rename before the Garrett hierarchy validation/reporting stage.
- 2026-08-22: Completed the next Garrett county-specific OSM and hierarchy-input steps, and ran the first Garrett power-hierarchy validation pass.
  - OSM substations fetched for Garrett:
    - output dir: `v3/data/processed/osm_garrett_substations/`
    - raw/cache dir: `v3/data/raw/osm_substations_garrett/`
    - summary:
      - total OSM substations: `43`
      - matched to HIFLD-derived transmission substations: `22`
      - unmatched/new: `21`
      - typed as `transmission=23`, `minor_distribution=2`, `generation=1`, `unknown=17`
      - candidate local/distribution substations under the current rule: only `2`
    - note: inherited filename is still `osm_montgomery_substations.csv` inside the Garrett output dir; content is Garrett, naming is legacy only.
  - OSM telecom exchanges fetched for Garrett:
    - output dir: `v3/data/processed/osm_garrett_telecom/`
    - summary:
      - exchange-like facilities found: `1`
      - current file: `osm_montgomery_telecom_exchanges.csv`
      - identified facility: `Neubeam`
    - implication: Garrett telecom is much sparser in OSM than Montgomery, so the two-tier telecom hierarchy will likely need a county-specific simplification or alternate source.
  - Garrett power-plant to transmission-substation matching completed:
    - output dir: `v3/data/processed/transmission_inspection_garrett/`
    - summary:
      - existing Garrett power plants in current graph: `12`
      - successfully connected to transmission substations: `6`
      - unmatched: `6`
      - ambiguous: `0`
  - Ran Garrett assignment-scope comparison:
    - output dir: `v3/data/processed/transmission_inspection_garrett/substation_assignment_scope_comparison/`
  - Patched `v3/build_final_power_hierarchy_validation.py` to remove a Montgomery-only `BELLS MILL` assumption so it can validate non-Montgomery counties.
  - First Garrett power-hierarchy validation output:
    - dir: `v3/data/processed/final_power_hierarchy_validation_garrett/`
    - result: `FAIL`
    - key failure reasons:
      - only `2` eligible OSM distribution substations
      - `2` distribution substations serve more than `15%` of all infrastructure
      - `7` substation-sector combinations exceed `20%` of a sector
      - `31` infrastructure-to-distribution assignments exceed `10 km`
    - headline distance stats:
      - infrastructure -> distribution median `17.32 km`, mean `17.03 km`, max `34.35 km`
      - distribution -> transmission median `8.70 km`, max `13.36 km`
    - takeaway: the Montgomery-style spatial hierarchy does not transfer cleanly to Garrett with the currently available OSM local-substation layer.
- 2026-08-22: Rebuilt Garrett as a transmission-only power hierarchy with telecom kept as a two-tier tower/exchange structure, then reran validation.
  - New builder:
    - `v3/build_garrett_transmission_only_hierarchy_variants.py`
    - output dir: `v3/data/processed/garrett_transmission_only_hierarchy_graph_variants/`
  - Shared validator was extended with a `--transmission-only` mode so Garrett can be checked with the same county-neutral validation path:
    - `v3/build_final_power_hierarchy_validation.py`
    - validation output dir: `v3/data/processed/final_power_hierarchy_validation_garrett_transmission_only/`
  - Garrett transmission-only hierarchy structure explicitly differs from Montgomery:
    - power: `power plants -> transmission substations -> consumers`
    - telecom: `schools/hospitals -> telecom_tower -> telecom_exchange`, with tower/exchange power fed directly from transmission substations
    - no distribution-substation tier in Garrett
  - Built graph summary:
    - nodes: `22 transmission_substation`, `12 power`, `12 school`, `11 ems_fire`, `11 telecom_tower`, `1 hospital`, `1 emergency_management`, `1 telecom_exchange`
    - matched transmission substations used: `22`
    - flat telecom nodes replaced: `15` -> `11` tower clusters + `1` exchange
    - matched power plants feeding transmission network: `5`
  - Transmission-only validation result: `FAIL`
    - fail reasons:
      - `1` transmission substation serves more than `15%` of all infrastructure
      - `3` substation-sector combinations exceed `20%` of a sector
    - important non-failure diagnostics:
      - infrastructure -> transmission median distance `4.74 km`
      - mean `5.11 km`
      - p90 `9.00 km`
      - max `9.51 km`
      - count `>10 km`: `0`
    - exact concentration problem:
      - `UNKNOWN122607` serves `11 / 36 = 30.56%` of all infrastructure
      - sector concentrations at `UNKNOWN122607`:
        - hospital: `1 / 1 = 100%`
        - school: `5 / 12 = 41.67%`
        - telecom: `3 / 12 = 25.00%`
    - takeaway:
      - the direct transmission-only Garrett hierarchy fixes the long-distance problem, but still fails the concentration test because one transmission substation becomes an oversized service bottleneck
      - per user instruction, stop here and do not attempt more Garrett redesigns this weekend; treat this as a Garrett-specific data-resolution limitation for follow-up with Zobel in the fall
