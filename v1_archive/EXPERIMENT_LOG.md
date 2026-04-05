# Experiment Log

This file is the research log for major modeling changes, experiment runs, and interpretations.

## Entry Template

### Date
- YYYY-MM-DD

### Experiment
- Short name:
- Graph type / size:
- Seed(s):
- Models run:

### What Changed
- Describe the code, feature, or modeling change.

### Why
- State the reason for the change or hypothesis being tested.

### Data / Features
- Input features used:
- Edge type used:
- Any new structural encoding:

### Results
- Baseline:
- GCN:
- Graph Transformer:
- Graphormer:

### Interpretation
- Short conclusion from this run.

### Files / Code References
- Main files touched:
- Relevant output folders:

---

## 2026-04-04

### Experiment
- Short name: Directed tiered-scale-free transition
- Graph type / size: tiered_scale_free; N20 and N40
- Seed(s): 7
- Models run: baseline, gcn, graph_transformer, graphormer

### What Changed
- Directed edges became the default in `edge_index.pt`.
- Default graph family was switched to `tiered_scale_free`.
- Graphormer was updated to append betweenness centrality in memory during training.

### Why
- To make graph structure more faithful to supply-chain directionality.
- To test whether Graphormer benefits more from richer topology and a global structural feature.

### Data / Features
- Input features used: `health`, `exposure`, `time_to_recovery`
- Graphormer-only extra feature: `betweenness centrality`
- Edge type used: directed
- Structural encoding in Graphormer: in-degree, out-degree, shortest-path attention bias

### Results
- N20 tiered_scale_free
- Baseline: `0.8642`
- GCN: `0.8142`
- Graph Transformer: `0.8650`
- Graphormer: `0.8642` with best `0.8708`

- N40 tiered_scale_free
- Baseline: `0.8342`
- GCN: `0.7750`
- Graph Transformer: `0.8292`
- Graphormer: `0.8213`

### Interpretation
- Transformer-based models are clearly stronger than GCN in these directed tiered-scale-free runs.
- They are not yet clearly outperforming the baseline.
- Increasing graph size to 40 did not yet produce a Graphormer advantage over baseline.

### Files / Code References
- Main files touched:
  - `train_graphormer_v1.py`
  - `graph_factory.py`
  - `simulate_and_build.py`
  - `configs/experiments.yaml`
- Relevant output folders:
  - `data/N20_tiered_scale_free/seed_7`
  - `data/N40_tiered_scale_free/seed_7`
  - `runs/`

---

## 2026-04-05

### Experiment
- Short name: Multi-seed tiered-scale-free comparison
- Graph type / size: tiered_scale_free; N20 and N40
- Seed(s): 1, 2, 3
- Models run: baseline, gcn, graph_transformer, graphormer

### What Changed
- Added a dedicated multi-seed config for controlled comparison.
- Added automated summary-table and figure generation from the final run outputs.

### Why
- To move from single-run results to mean/std evidence across seeds.
- To test whether graph-based models separate more clearly from the baseline under repeated runs.

### Data / Features
- Input features used: `health`, `exposure`, `time_to_recovery`
- Graphormer-only extra feature: `betweenness centrality`
- Edge type used: directed
- Structural encoding in Graphormer: in-degree, out-degree, shortest-path attention bias

### Results
- N20 tiered_scale_free mean last-test accuracy
- Baseline: `0.8639 +/- 0.0356`
- GCN: `0.7936 +/- 0.0471`
- Graph Transformer: `0.8636 +/- 0.0356`
- Graphormer: `0.8422 +/- 0.0468`

- N40 tiered_scale_free mean last-test accuracy
- Baseline: `0.8456 +/- 0.0196`
- GCN: `0.7810 +/- 0.0182`
- Graph Transformer: `0.8457 +/- 0.0235`
- Graphormer: `0.8392 +/- 0.0161`

### Interpretation
- Multi-seed evidence reinforces the current conclusion.
- The masked Graph Transformer is competitive with the baseline.
- Graphormer remains competitive but does not show a clear advantage over the baseline.
- GCN is consistently weaker than both transformer variants and the baseline.

### Files / Code References
- Main files touched:
  - `configs/experiments_multiseed.yaml`
  - `generate_paper_figures.py`
- Relevant output folders:
  - `runs/`
  - `runs/figures/`
