# V2 Held-Out Seed Results Summary

## Setup

- Train seeds: `1,2,3`
- Held-out test seeds: `4,5`
- Prediction horizon: `t+5`
- Input: `X[t, node, feature]`
- Target: one future graph-level structure metric at `t+5`
- Metric: best held-out test `MAE`; lower is better

This is stronger than the earlier within-seed split because the model is tested on simulations it never saw during training.

## Results

### LCC Fraction

| Model | N=20 MAE | N=40 MAE |
| --- | ---: | ---: |
| Baseline | 0.1299 | **0.1126** |
| GCN | 0.1244 | 0.1239 |
| Graph Transformer | 0.1177 | 0.1236 |
| Graphormer | **0.1146** | 0.1261 |

Takeaway: graph-aware models win for `N=20`, but baseline is best for `N=40`.

### Component Fraction

| Model | N=20 MAE | N=40 MAE |
| --- | ---: | ---: |
| Baseline | 0.0320 | 0.0395 |
| GCN | 0.0200 | 0.0271 |
| Graph Transformer | **0.0151** | 0.0265 |
| Graphormer | 0.0182 | **0.0261** |

Takeaway: this is the strongest held-out result for graph-aware models. They beat baseline clearly for both graph sizes.

### Diameter Fraction

| Model | N=20 MAE | N=40 MAE |
| --- | ---: | ---: |
| Baseline | 0.0483 | 0.0392 |
| GCN | 0.0455 | **0.0252** |
| Graph Transformer | **0.0445** | 0.0260 |
| Graphormer | 0.0448 | 0.0253 |

Takeaway: graph-aware models improve over baseline, especially for `N=40`.

### Edge Survival Ratio

| Model | N=20 MAE | N=40 MAE |
| --- | ---: | ---: |
| Baseline | 0.1539 | **0.1421** |
| GCN | 0.1465 | 0.1575 |
| Graph Transformer | 0.1461 | 0.1431 |
| Graphormer | **0.1450** | 0.1570 |

Takeaway: gains are smaller here. Graph-aware models help slightly at `N=20`, but baseline is competitive or best at `N=40`.

## Overall Interpretation

- Structure-level prediction is a better fit for graph-aware models than node-level health prediction.
- Held-out results are strongest for `component_fraction` and `diameter_fraction`.
- `LCC_fraction` and `edge_survival_ratio` are more mixed, especially at `N=40`.
- The current held-out evaluation uses one train/test seed split, so the result is promising but should not be treated as final statistical evidence yet.

## Report Artifacts

- Combined figure: `v2/runs/figures/heldout_seed_headline_results.png`
- Detailed held-out summary: `v2/runs/figures/heldout_seed_report_summary.md`
- Per-target CSVs:
  - `v2/runs/figures/heldout_graph_results_by_model_size_lcc.csv`
  - `v2/runs/figures/heldout_graph_results_by_model_size_components.csv`
  - `v2/runs/figures/heldout_graph_results_by_model_size_diameter.csv`
  - `v2/runs/figures/heldout_graph_results_by_model_size_edge_survival.csv`
