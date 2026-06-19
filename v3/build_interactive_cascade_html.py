from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from v3.simulator_v2 import (
        TIMESTEPS,
        build_outgoing,
        build_structural_graph,
        deterministic_trial,
        ensure_simulation_edges,
        pick_multi_seeds,
        recovery_step,
        transition_state,
    )
except ModuleNotFoundError:
    from simulator_v2 import (
        TIMESTEPS,
        build_outgoing,
        build_structural_graph,
        deterministic_trial,
        ensure_simulation_edges,
        pick_multi_seeds,
        recovery_step,
        transition_state,
    )


def run_cascade_with_history(
    nodes: pd.DataFrame,
    sim_edges: pd.DataFrame,
    scenario_name: str,
    seed_nodes: list[str],
    timesteps: int,
) -> pd.DataFrame:
    node_ids = nodes["node_id"].tolist()
    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    states = {node_id: 1.0 for node_id in node_ids}
    last_downgrade_timestep: dict[str, int | None] = {node_id: None for node_id in node_ids}
    damage_load: dict[str, float] = {node_id: 0.0 for node_id in node_ids}
    for seed in seed_nodes:
        states[seed] = 0.0
        last_downgrade_timestep[seed] = 0
        damage_load[seed] = 1.0

    outgoing = build_outgoing(sim_edges)
    pending_impacts: dict[int, list[tuple[str, float, str]]] = {}
    history_rows: list[dict] = []

    def schedule_from(node_id: str, now_t: int) -> None:
        source_state = states[node_id]
        if source_state >= 1.0:
            return
        try:
            from v3.simulator_v2 import SECTOR_PARAMS
        except ModuleNotFoundError:
            from simulator_v2 import SECTOR_PARAMS

        source_severity = 1.0 if source_state == 0.0 else 0.5
        source_scale = SECTOR_PARAMS[node_types[node_id]]["propagation_scale"]
        for edge in outgoing.get(node_id, []):
            delay = int(edge["delay"])
            arrival_t = now_t + delay
            if arrival_t > timesteps:
                continue
            target_id = edge["simulation_target"]
            target_type = node_types[target_id]
            semantic_type = edge.get("semantic_type", "dependency")
            base_prob = 0.60 if semantic_type == "dependency" else 0.45
            weight = float(edge["weight"])
            transmission_prob = min(0.95, base_prob * weight * source_scale * (1.0 if source_state == 0.0 else 0.8))
            if not deterministic_trial(transmission_prob, "edge", scenario_name, now_t, node_id, target_id, arrival_t):
                continue
            impact = weight * source_severity * source_scale
            if target_type == "hospital":
                impact *= 1.10
            elif target_type == "school":
                impact *= 0.95
            pending_impacts.setdefault(arrival_t, []).append((target_id, impact, node_id))

    for seed in seed_nodes:
        schedule_from(seed, 0)

    for t in range(timesteps + 1):
        incoming = pending_impacts.pop(t, [])
        cumulative: dict[str, float] = {}
        for target_id, impact, _source_id in incoming:
            cumulative[target_id] = cumulative.get(target_id, 0.0) + impact
        for node_id in node_ids:
            damage_load[node_id] *= 0.90

        changed_nodes: list[str] = []
        for node_id, impact in cumulative.items():
            damage_load[node_id] = min(1.5, damage_load[node_id] + impact)
            effective_impact = impact + 0.6 * damage_load[node_id]
            new_state = transition_state(states[node_id], node_types[node_id], effective_impact, t, node_id)
            if new_state < states[node_id]:
                states[node_id] = new_state
                last_downgrade_timestep[node_id] = t
                changed_nodes.append(node_id)

        for node_id in changed_nodes:
            schedule_from(node_id, t)

        if t > 0:
            recovery_step(states, node_types, damage_load, last_downgrade_timestep, t)

        for row in nodes.itertuples(index=False):
            history_rows.append(
                {
                    "scenario": scenario_name,
                    "timestep": t,
                    "node_id": row.node_id,
                    "node_type": row.node_type,
                    "name": row.name,
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                    "state": states[row.node_id],
                    "is_seed": row.node_id in seed_nodes,
                }
            )

    return pd.DataFrame(history_rows)


def build_html(history: pd.DataFrame, output_path: Path) -> None:
    scenario_labels = {"power": "Power Shock", "telecom": "Telecom Shock", "ems": "EMS Shock"}
    scenarios = list(history["scenario"].drop_duplicates())
    node_types = sorted(history["node_type"].drop_duplicates())

    lon_min = float(history["longitude"].min())
    lon_max = float(history["longitude"].max())
    lat_min = float(history["latitude"].min())
    lat_max = float(history["latitude"].max())

    payload = {
        "meta": {
            "timesteps": int(history["timestep"].max()),
            "scenario_labels": scenario_labels,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "node_types": node_types,
        },
        "rows": history.to_dict(orient="records"),
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Interactive Cascade Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .controls {{ display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }}
    .panel {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; }}
    .meta {{ display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; font-size: 14px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; font-size: 14px; }}
    .swatch {{ display:inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; }}
    svg {{ width: 100%; height: 760px; background: #fbfbfb; border: 1px solid #e5e5e5; border-radius: 8px; }}
    .caption {{ color: #555; margin-top: 10px; font-size: 14px; }}
    .small {{ font-size: 13px; color: #666; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Montgomery County Cascade Viewer</h2>
    <div class="panel">
      <div class="controls">
        <label>Scenario:
          <select id="scenarioSelect"></select>
        </label>
        <label>Timestep:
          <input id="timeSlider" type="range" min="0" max="{int(history['timestep'].max())}" step="1" value="0" />
          <span id="timeValue">0</span>
        </label>
        <button id="playPause">Play</button>
      </div>
      <svg id="viz" viewBox="0 0 980 760"></svg>
      <div class="meta">
        <div><strong>Failed nodes:</strong> <span id="failedCount">0</span></div>
        <div><strong>Degraded nodes:</strong> <span id="degradedCount">0</span></div>
        <div><strong>Healthy nodes:</strong> <span id="healthyCount">0</span></div>
      </div>
      <div class="legend">
        <div><span class="swatch" style="background:#c0392b"></span>Failed</div>
        <div><span class="swatch" style="background:#f39c12"></span>Degraded</div>
        <div><span class="swatch" style="background:#8f8f8f"></span>Normal</div>
        <div><span class="swatch" style="background:#111"></span>Seed node outline</div>
      </div>
      <div class="caption">This view shows node-level state over time for the tuned `simulator_v2`. Red nodes are failed, orange nodes are degraded, and gray nodes remain normal.</div>
      <div class="small">Coordinates are shown in county space without basemap tiles to keep the artifact self-contained.</div>
    </div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const svg = document.getElementById('viz');
    const scenarioSelect = document.getElementById('scenarioSelect');
    const timeSlider = document.getElementById('timeSlider');
    const timeValue = document.getElementById('timeValue');
    const failedCount = document.getElementById('failedCount');
    const degradedCount = document.getElementById('degradedCount');
    const healthyCount = document.getElementById('healthyCount');
    const playPause = document.getElementById('playPause');

    const width = 980;
    const height = 760;
    const pad = 28;
    const rows = payload.rows;
    const meta = payload.meta;
    const scenarios = [...new Set(rows.map(r => r.scenario))];
    const scenarioLabels = meta.scenario_labels;
    const byScenarioTime = {{}};

    function xScale(lon) {{
      return pad + ((lon - meta.lon_min) / (meta.lon_max - meta.lon_min || 1)) * (width - 2 * pad);
    }}
    function yScale(lat) {{
      return height - pad - ((lat - meta.lat_min) / (meta.lat_max - meta.lat_min || 1)) * (height - 2 * pad);
    }}
    function stateColor(state) {{
      if (state === 0) return '#c0392b';
      if (state === 0.5) return '#f39c12';
      return '#8f8f8f';
    }}
    function radiusByType(nodeType) {{
      if (nodeType === 'hospital') return 7;
      if (nodeType === 'power') return 6.5;
      if (nodeType === 'ems_fire') return 5.5;
      if (nodeType === 'emergency_management') return 8;
      if (nodeType === 'school') return 3.8;
      return 4.6;
    }}

    rows.forEach(row => {{
      if (!byScenarioTime[row.scenario]) byScenarioTime[row.scenario] = {{}};
      if (!byScenarioTime[row.scenario][row.timestep]) byScenarioTime[row.scenario][row.timestep] = [];
      byScenarioTime[row.scenario][row.timestep].push(row);
    }});

    scenarios.forEach(s => {{
      const option = document.createElement('option');
      option.value = s;
      option.textContent = scenarioLabels[s] || s;
      scenarioSelect.appendChild(option);
    }});

    function render() {{
      const scenario = scenarioSelect.value;
      const timestep = Number(timeSlider.value);
      timeValue.textContent = timestep;
      const points = byScenarioTime[scenario][timestep] || [];
      let failed = 0, degraded = 0, healthy = 0;

      svg.innerHTML = '';
      points.forEach(point => {{
        if (point.state === 0) failed++;
        else if (point.state === 0.5) degraded++;
        else healthy++;

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', xScale(point.longitude));
        circle.setAttribute('cy', yScale(point.latitude));
        circle.setAttribute('r', radiusByType(point.node_type));
        circle.setAttribute('fill', stateColor(point.state));
        circle.setAttribute('fill-opacity', point.state === 1 ? '0.65' : '0.92');
        circle.setAttribute('stroke', point.is_seed ? '#111' : '#fff');
        circle.setAttribute('stroke-width', point.is_seed ? '2.0' : '0.6');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${{point.name}} (${{point.node_type}}) | state=${{point.state}}`;
        circle.appendChild(title);
        svg.appendChild(circle);
      }});

      failedCount.textContent = failed;
      degradedCount.textContent = degraded;
      healthyCount.textContent = healthy;
    }}

    let playing = false;
    let timer = null;
    playPause.addEventListener('click', () => {{
      playing = !playing;
      playPause.textContent = playing ? 'Pause' : 'Play';
      if (playing) {{
        timer = setInterval(() => {{
          const next = Number(timeSlider.value) + 1;
          if (next > Number(timeSlider.max)) {{
            playing = false;
            playPause.textContent = 'Play';
            clearInterval(timer);
            return;
          }}
          timeSlider.value = String(next);
          render();
        }}, 500);
      }} else if (timer) {{
        clearInterval(timer);
      }}
    }});
    scenarioSelect.addEventListener('change', render);
    timeSlider.addEventListener('input', render);
    render();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained interactive HTML cascade viewer.")
    parser.add_argument("--nodes", default="v3/data/processed/dependency_graph_nodes.csv")
    parser.add_argument("--edges", default="v3/data/processed/dependency_graph_edges.csv")
    parser.add_argument("--sim-edges", default="v3/data/processed/simulation_edges.csv")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--history-out", default="v3/data/processed/cascade_node_state_history_v2.csv")
    parser.add_argument("--html-out", default="v3/runs/figures/interactive_cascade_v2.html")
    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    sim_edges = ensure_simulation_edges(args.nodes, args.edges, args.sim_edges)
    _structural_graph = build_structural_graph(nodes, edges)
    seeds = pick_multi_seeds(nodes, edges, 3)

    history_frames = []
    for scenario_name, seed_nodes in seeds.items():
        history_frames.append(run_cascade_with_history(nodes, sim_edges, scenario_name, seed_nodes, args.timesteps))
    history = pd.concat(history_frames, ignore_index=True)

    history_path = Path(args.history_out)
    html_path = Path(args.html_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    build_html(history, html_path)
    print(f"Wrote node-state history to {history_path}")
    print(f"Wrote interactive cascade HTML to {html_path}")


if __name__ == "__main__":
    main()
