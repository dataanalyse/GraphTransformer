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
