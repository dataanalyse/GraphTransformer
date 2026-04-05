# Paper Preparation Notes

This file is the working bridge between code experiments and a future academic paper.
It is meant to be appended over time.
Each future update should preserve the same core section structure so the paper narrative stays consistent.

## 1. Research Memory

The project studies supply-chain disruption prediction on directed graphs.
Each node represents a firm/entity, and each edge represents directional dependency flow.

The prediction task is:
- given node and graph state at time `t`
- predict node health at time `t+1`

The simulator currently includes:
- exogenous shocks
- endogenous propagation through upstream dependency
- probabilistic recovery

## 2. Current Experimental Setup

### Graph Family
- Current default graph family: `tiered_scale_free`
- Current larger runs completed: `N20_tiered_scale_free`, `N40_tiered_scale_free`

### Edge Semantics
- Directed edges are now the default
- `edge_index.pt` is directional by default for newly generated datasets

### Node Features
- `health`
- `exposure`
- `time_to_recovery`

### Graphormer Extra Feature
- Graphormer currently adds `betweenness centrality` in memory during training
- This is not yet baked into `X_v1.pt`

## 3. Model Families

### Baseline
- Uses node-local features only
- No graph aggregation

### GCN
- Uses graph structure through adjacency-based message passing
- Local neighborhood aggregation with fixed graph normalization

### Masked Graph Transformer
- Uses self-attention over nodes
- Attention restricted by graph adjacency mask

### Graphormer
- Uses full node-to-node attention
- Adds graph structure via:
  - in-degree encoding
  - out-degree encoding
  - shortest-path attention bias
  - betweenness centrality as an added node feature

## 4. Current Generated Artifacts

### Supply-Chain Graph Artifacts
- Each dataset run generates a supply-chain graph image: `supply_chain_graph.png`
- This is the input graph visualization for the simulated network

### AI Model Artifacts
- Each training run currently saves:
  - `architecture.txt`
  - `metrics.csv`
  - `run_end.json`
  - `probs_last5.pt`
- So the project already stores model definitions, training metrics, final scores, and recent predictions

### Current Gap
- The project does not yet automatically generate polished model/result figures for the paper, such as:
  - training-curve plots
  - comparison charts across models
  - paper-style architecture figures per run

## 5. Current Evidence

### N20 tiered_scale_free
- Baseline: `0.8642`
- GCN: `0.8142`
- Graph Transformer: `0.8650`
- Graphormer: `0.8642`
- Graphormer best checkpoint: `0.8708`

### N40 tiered_scale_free
- Baseline: `0.8342`
- GCN: `0.7750`
- Graph Transformer: `0.8292`
- Graphormer: `0.8213`

### Multi-seed tiered_scale_free summary

#### N20 tiered_scale_free, seeds = 1, 2, 3
- Baseline: `0.8639 +/- 0.0356`
- GCN: `0.7936 +/- 0.0471`
- Graph Transformer: `0.8636 +/- 0.0356`
- Graphormer: `0.8422 +/- 0.0468`

#### N40 tiered_scale_free, seeds = 1, 2, 3
- Baseline: `0.8456 +/- 0.0196`
- GCN: `0.7810 +/- 0.0182`
- Graph Transformer: `0.8457 +/- 0.0235`
- Graphormer: `0.8392 +/- 0.0161`

## 6. Current Interpretation

- Graph-based neural models are clearly outperforming GCN in the current larger tiered-scale-free directed experiments.
- However, they are not yet consistently outperforming the baseline.
- At the current stage, the strongest claim is:
  - transformer-based graph models remain competitive as graph complexity increases
  - but a decisive advantage over the baseline has not yet been established

## 7. What Is Still Needed For A Paper

### Experiment Log
- Keep appending `EXPERIMENT_LOG.md`
- This preserves the research trail and model evolution

### Stronger Evidence
- Current status: multi-seed evidence now exists for N20 and N40 tiered-scale-free runs
- Next step: extend multi-seed evidence to additional graph sizes or regimes if needed
- Keep model definitions fixed during final comparison runs
- Compare models under the same graph family and parameter regime

### Stable Methodology
- Freeze a final methodology section describing:
  - graph generation
  - node features
  - edge semantics
  - model architectures
  - train/test split
  - metrics

### Result Narrative
- Explain not only what won
- But also:
  - what changed
  - why performance changed
  - where graph models help
  - where they do not yet help

### Figures And Tables
- Current status: generated from the multi-seed sweep in `runs/figures/`
- Available now:
  - mean/std summary table by model and graph size
  - accuracy vs graph size
  - per-model comparison by graph size
  - training curves for N20 and N40 tiered-scale-free
- Still useful later:
  - selected supply-chain graph examples

## 8. Working Conclusion

The project is in a strong research prototype state.
It is not yet in final paper-submission state, but it is close enough that systematic logging, multi-seed evaluation, and a stabilized methodology can convert it into a paper-ready workflow.
