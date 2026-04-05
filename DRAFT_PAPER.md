# Feature Engineering Versus Graph Neural Architectures for Synthetic Supply-Chain Disruption Prediction

## Abstract
We study next-step disruption prediction in synthetic supply-chain networks using directed graphs with tiered scale-free structure. We compare a feature-only baseline, a graph convolutional network (GCN), a masked graph transformer, and a Graphormer-style model. Node features include health state, disruption exposure, and time to recovery; the Graphormer model additionally uses betweenness centrality. Across the current experimental settings, graph-based transformer models outperform the GCN but do not consistently outperform the feature-only baseline. This suggests that, in the present simulator, important relational information is already encoded into node-level features, limiting the incremental value of more complex graph neural architectures.

## Introduction
Supply-chain disruptions propagate through dependency networks rather than isolated firms. This motivates graph-based predictive models that can represent how failures spread across connected entities. In principle, graph neural networks and graph transformers should outperform feature-only baselines because they can aggregate information across network structure.

At the same time, graph learning models do not operate in isolation from feature design. If features already summarize much of the relevant network effect, then a non-graph baseline may remain highly competitive. This issue is especially important in simulation studies, where node features may already encode relational structure by construction.

This study examines that question in a controlled synthetic setting. We simulate directed supply-chain graphs and compare a feature-only baseline against graph-based neural architectures. Our main aim is not only to measure predictive accuracy, but also to understand when graph architectures add value beyond engineered node-level features.

## Methodology
### Problem Setup
The task is next-step node health prediction. Given node features at time `t`, models predict whether each node is healthy or disrupted at time `t+1`.

### Graph Generation
We use directed supply-chain graphs with a tiered scale-free structure. Nodes are assigned to ordered tiers, and edges are allowed only between adjacent tiers. This preserves hierarchical supply-chain semantics while retaining hub-like connectivity.

### Simulation
Node states evolve through three mechanisms:
- exogenous shock
- endogenous propagation from disrupted upstream neighbors
- probabilistic recovery

### Features
The dataset input tensor `X` contains:
- `health`
- `exposure`
- `time_to_recovery`

These features summarize both node-local status and part of the relational effect of the graph. In the current Graphormer implementation, betweenness centrality is also appended in memory as an additional node feature.

### Models
- Baseline: feature-only node classifier with no graph aggregation
- GCN: graph convolution with directed edge structure
- Masked Graph Transformer: self-attention restricted by graph adjacency
- Graphormer: full attention with graph-aware structural bias, including degree encodings and shortest-path bias

### Evaluation
We use chronological train/test splits on simulated sequences and compare final test accuracy across models. Recent experiments focused on directed `tiered_scale_free` graphs with larger node counts.

## Results
In the current directed tiered-scale-free experiments:

### N20 tiered_scale_free
- Baseline: `0.8642`
- GCN: `0.8142`
- Graph Transformer: `0.8650`
- Graphormer: `0.8642`

### N40 tiered_scale_free
- Baseline: `0.8342`
- GCN: `0.7750`
- Graph Transformer: `0.8292`
- Graphormer: `0.8213`

## Findings
Three findings stand out.

First, graph-based transformer models are stronger than the GCN in the current setup. This indicates that more flexible attention-based architectures handle the simulated dependency structure better than simple adjacency-based aggregation.

Second, neither the masked graph transformer nor the Graphormer consistently outperforms the feature-only baseline. The baseline remains highly competitive, even as graph size increases.

Third, this pattern is consistent with the idea that crucial relational information is already encoded into node-level features. In particular, `exposure` directly summarizes upstream disruption pressure, and betweenness centrality adds a structural importance signal. When such information is already available at the node level, the marginal gain from explicit graph reasoning can narrow substantially.

## Conclusion
Our current results support the following interpretation: in this simulation framework, key relational information is substantially pre-encoded into node features, especially through exposure and structural descriptors. As a result, a feature-only baseline performs competitively with graph-based models, while more complex graph architectures provide only limited incremental benefit.

This conclusion should be stated narrowly. It applies to the current simulator, feature set, and evaluation design. It does not imply that graph neural architectures are broadly unnecessary for supply-chain prediction. Rather, it suggests that their advantage depends strongly on how much graph information is already captured through feature engineering. A natural next step is to test whether graph-based models gain clearer advantage under settings where relational information is less explicitly encoded at the node level.
