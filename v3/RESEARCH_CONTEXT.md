# Summer 2026 Research Context

## Working Research Direction

The Summer 2026 paper extends prior work on graph-aware forecasting for supply-chain disruption prediction into the domain of cascading infrastructure propagation and resilience forecasting.

The central research idea is to investigate whether graph-aware learning models can forecast multi-stage cascading structural degradation in geographically grounded interdependent infrastructure systems.

The project does NOT attempt to build a perfect digital twin or exact real-world infrastructure dependency model. Instead, it develops a literature-informed stochastic propagation simulation framework grounded in realistic infrastructure topology and dependency abstractions.

---

# Proposed Paper Direction

## Candidate Title
Forecasting Cascading Structural Disruption in Dynamic Supply Networks

Alternative framing:
Forecasting cascading infrastructure degradation using graph-aware learning.

---

# Core Research Questions

1. Can graph-aware learning models forecast cascading structural degradation in interdependent infrastructure networks?

2. Under what propagation conditions do graph-aware models outperform simpler baselines?

3. How do topology, dependency structure, and delayed cascading influence long-horizon forecasting performance?

---

# Core Research Contribution

The contribution is NOT a new graph transformer architecture.

The contribution is:
- geographically grounded infrastructure propagation simulation
- graph-aware forecasting of cascading disruption
- structural trajectory prediction under delayed propagation
- resilience forecasting in interdependent infrastructure systems

---

# Literature Grounding

## Paper 1
### Schauer et al.
### "Analyzing Cascading Effects among Critical Infrastructures"

Main ideas extracted:
- disruptions spread through dependency relationships
- dependencies have varying strengths
- cascading failures occur over time
- propagation is probabilistic
- systems may partially degrade rather than completely fail
- supplier-of-supplier propagation matters

Simulator implications:
- weighted dependency edges
- delayed propagation
- stochastic transitions
- multi-state node health
- directed dependency graphs
- multi-hop propagation

Important insight:
The paper models cascading disruption as a delayed probabilistic dependency-driven process.

---

## Paper 2
### Schneider et al.
### "Cascading Effects of Critical Infrastructures in a Flood Scenario"

Main ideas extracted:
- localized disruption can trigger system-wide cascading failures
- flood propagation causes secondary infrastructure failures
- delayed indirect effects are important
- infrastructure systems are interdependent
- real-world grounding can still use simulated abstraction

Simulator implications:
- spatially grounded propagation
- directional spread
- delayed cascading
- multi-state degradation
- interacting disruptions
- topology-sensitive outcomes

Important insight:
The paper uses real-world infrastructure geography but simulated cascading behavior.

---

## Paper 3
### Li et al. (Argonne National Laboratory, 2024)
### "Modeling and Solving Cascading Failures across Interdependent Infrastructure Systems"

Main ideas extracted:
- interdependent infrastructure dependency graphs
- weighted dependency modeling
- multi-stage cascading failures
- service-level degradation
- geographically grounded infrastructure abstraction

Important insight:
The paper uses real-world infrastructure topology and dependency patterns but simulated cascading dynamics.

Important modeling insights:
- dependency weights are modeled/estimated rather than exactly known
- proximity-based dependency weighting is acceptable
- infrastructure dependencies can be stochastic and uncertain
- cascading failures evolve over multiple stages

Simulator implications:
- weighted edges
- service-level degradation
- staged cascading propagation
- uncertainty-aware propagation
- geographically grounded infrastructure graphs

---

# Real-World Infrastructure Data Source

## Primary Data Source
HIFLD Archive:
https://source.coop/seerai/hifld

## Planned Initial Infrastructure Layers
- hospitals
- EMS/fire stations
- power plants
- cellular towers
- primary roads

Optional later additions:
- water facilities
- schools
- railroads
- substations if available

---

# Geographic Scope

Initial study region:
Montgomery County, Maryland

Reasoning:
- manageable scale
- infrastructure-dense
- geographically coherent
- realistic propagation environment
- feasible within Summer timeline

---

# Planned Infrastructure Graph

## Nodes
Infrastructure assets such as:
- hospitals
- EMS stations
- power nodes
- telecom nodes
- transportation nodes

## Edges
Dependency relationships between infrastructure assets.

Examples:
- hospitals depend on power
- hospitals depend on roads
- hospitals depend on telecom
- EMS depends on roads
- telecom depends on power

---

# Planned Edge Properties

## Weighted Dependencies
Edge weights represent dependency strength.

Examples:
- strong dependency = 0.9
- weak dependency = 0.2

Weights may be assigned using:
- infrastructure type
- geographic proximity
- literature-informed assumptions
- inverse-distance weighting

---

## Delayed Propagation

Disruptions should NOT spread instantly.

Each edge may contain:
- propagation delay
- delayed cascading effects

Examples:
- power failure affects hospitals after delay
- transportation disruption affects EMS immediately

---

# Planned Node States

Initial multi-state degradation framework:

0 = normal
1 = stressed
2 = degraded
3 = failed

This replaces binary healthy/failed modeling.

---

# Planned Propagation Logic

Disruption propagation should be:
- stochastic
- weighted
- delayed
- topology-sensitive
- multi-hop

The network should behave as a propagating dynamic system rather than independent node failures.

---

# Planned Structural Metrics

Metrics to forecast:
- LCC fraction
- component fraction
- diameter fraction
- edge survival ratio
- average service level
- recovery trajectory
- fragmentation trajectory

---

# Planned Prediction Tasks

## Structural Forecasting
Predict:
- future LCC
- future fragmentation
- future service degradation

## Trajectory Forecasting
Predict:
- network behavior at T+1
- T+5
- T+10
- T+20

## Cascading Propagation Forecasting
Forecast:
- how disruptions spread
- how fast degradation propagates
- recovery evolution over time

---

# Planned ML Models

## Baselines
- MLP
- non-graph temporal baseline

## Graph Models
- GCN
- graph transformer
- Graphormer-style architectures (optional)

---

# Central Hypothesis Direction

Graph-aware models become more useful when:
- disruptions propagate across multiple hops
- cascading effects evolve over time
- topology significantly influences system degradation
- long-range relational reasoning becomes necessary

---

# Important Research Framing

This project is NOT:
- a perfect digital twin
- exact infrastructure reconstruction
- enterprise operational forecasting
- utility-grade infrastructure simulation

This project IS:
- a literature-informed cascading propagation framework
- a geographically grounded resilience forecasting study
- an investigation of graph-aware forecasting under cascading disruption dynamics

---

# Expected Research Narrative

Spring paper:
Graph structure is not always useful.

Summer paper:
Graph-aware models become more valuable when forecasting multi-stage cascading propagation dynamics in interdependent infrastructure systems.