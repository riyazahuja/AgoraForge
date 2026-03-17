import json
import os


def summarize_query_model_diagnostics(trajectories):
    if not trajectories:
        return []

    agent_totals = {}
    for trajectory in trajectories:
        for step in trajectory.get("steps", []):
            for diag in step.get("state_before", {}).get("query_model_quality", []):
                agent_id = int(diag["agent_id"])
                totals = agent_totals.setdefault(
                    agent_id,
                    {
                        "count": 0,
                        "mae_all": 0.0,
                        "rmse_all": 0.0,
                        "mae_feasible": 0.0,
                        "rmse_feasible": 0.0,
                    },
                )
                totals["count"] += 1
                totals["mae_all"] += float(diag.get("mae_all", 0.0))
                totals["rmse_all"] += float(diag.get("rmse_all", 0.0))
                totals["mae_feasible"] += float(diag.get("mae_feasible", 0.0))
                totals["rmse_feasible"] += float(diag.get("rmse_feasible", 0.0))

    summary = []
    for agent_id in sorted(agent_totals):
        totals = agent_totals[agent_id]
        denom = max(totals["count"], 1)
        summary.append(
            {
                "agent_id": agent_id,
                "mae_all": totals["mae_all"] / denom,
                "rmse_all": totals["rmse_all"] / denom,
                "mae_feasible": totals["mae_feasible"] / denom,
                "rmse_feasible": totals["rmse_feasible"] / denom,
                "num_steps": totals["count"],
            }
        )
    return summary


def _trajectory_html_document(title, payload_json):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffaf0;
      --ink: #1f1a14;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --line: #d6c7ad;
      --muted: #6b6458;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f7f3ea 0%, #efe5d2 100%);
      color: var(--ink);
    }}
    header, section {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px 24px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 30px rgba(31, 26, 20, 0.06);
    }}
    .panel h2, .panel h3 {{
      margin-top: 0;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    input[type="range"] {{
      flex: 1;
      min-width: 240px;
      accent-color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.12);
      color: var(--accent);
      font-size: 12px;
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .pill.warn {{
      background: rgba(194, 65, 12, 0.12);
      color: var(--accent-2);
    }}
    svg {{
      width: 100%;
      height: 220px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      max-height: 320px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="muted" id="summary"></div>
  </header>

  <section>
    <div class="panel">
      <div class="controls">
        <label for="stepRange"><strong>Step</strong></label>
        <input id="stepRange" type="range" min="0" value="0">
        <strong id="stepLabel"></strong>
      </div>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Cumulative Returns</h2>
      <svg id="returnsChart" viewBox="0 0 640 220"></svg>
    </div>
    <div class="panel">
      <h2>Cash Over Time</h2>
      <svg id="cashChart" viewBox="0 0 640 220"></svg>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Environment State</h2>
      <div id="envState"></div>
    </div>
    <div class="panel">
      <h2>Agent Actions</h2>
      <div id="actions"></div>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Offers</h2>
      <div id="offers"></div>
    </div>
    <div class="panel">
      <h2>Raw Step Payload</h2>
      <pre id="rawPayload"></pre>
    </div>
  </section>

  <script>
    const trajectory = {payload_json};
    const steps = trajectory.steps || [];
    const agents = trajectory.initial_state.agents.length;
    const range = document.getElementById("stepRange");
    const stepLabel = document.getElementById("stepLabel");
    const summary = document.getElementById("summary");

    summary.textContent = `Thread ${{trajectory.thread_index}} | steps=${{steps.length}} | agents=${{agents}}`;
    range.max = Math.max(steps.length - 1, 0);

    function polyline(valuesByAgent, colors) {{
      const width = 640;
      const height = 220;
      const pad = 24;
      const all = valuesByAgent.flat();
      const min = Math.min(...all, 0);
      const max = Math.max(...all, 1);
      const span = Math.max(max - min, 1e-6);

      function point(ix, value, total) {{
        const x = pad + ((width - 2 * pad) * ix / Math.max(total - 1, 1));
        const y = height - pad - ((height - 2 * pad) * (value - min) / span);
        return `${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}

      const axes = `
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="#b8ab95" stroke-width="1"/>
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="#b8ab95" stroke-width="1"/>
        <text x="${{pad}}" y="16" fill="#6b6458" font-size="12">max=${{max.toFixed(2)}}</text>
        <text x="${{pad}}" y="${{height - 6}}" fill="#6b6458" font-size="12">min=${{min.toFixed(2)}}</text>`;

      const lines = valuesByAgent.map((series, idx) => {{
        const pts = series.map((value, ix) => point(ix, value, series.length)).join(" ");
        return `<polyline fill="none" stroke="${{colors[idx % colors.length]}}" stroke-width="3" points="${{pts}}"/>`;
      }}).join("");
      return axes + lines;
    }}

    function renderCharts() {{
      const returnSeries = Array.from({{ length: agents }}, (_, agentIdx) =>
        [0].concat(steps.map(step => step.cumulative_returns[agentIdx]))
      );
      const cashSeries = Array.from({{ length: agents }}, (_, agentIdx) =>
        [trajectory.initial_state.agents[agentIdx].cash].concat(
          steps.map(step => step.state_after.agents[agentIdx].cash)
        )
      );
      const colors = ["#0f766e", "#c2410c", "#2563eb", "#9333ea", "#65a30d", "#db2777"];
      document.getElementById("returnsChart").innerHTML = polyline(returnSeries, colors);
      document.getElementById("cashChart").innerHTML = polyline(cashSeries, colors);
    }}

    function htmlList(items, warn = false) {{
      if (!items || items.length === 0) {{
        return '<span class="muted">none</span>';
      }}
      return items.map(item => `<span class="pill${{warn ? ' warn' : ''}}">${{item}}</span>`).join("");
    }}

    function agentTable(step) {{
      const rows = step.state_after.agents.map((agent, idx) => {{
        const action = step.actions[idx];
        const job = agent.positions.filter(pos => !pos.settled).map(pos =>
          `${{pos.side}} phi=${{pos.target}} d=${{pos.deadline}} l=${{pos.loss.toFixed(2)}}`
        );
        const resolved = agent.library.resolved.map(item => item.formula);
        return `
          <tr>
            <td>${{idx}}</td>
            <td>${{action.type}}</td>
            <td>${{step.rewards[idx].toFixed(3)}}</td>
            <td>${{step.cumulative_returns[idx].toFixed(3)}}</td>
            <td>${{agent.cash.toFixed(3)}}</td>
            <td>${{agent.worst_case_balance.toFixed(3)}}</td>
            <td>${{htmlList(job)}}</td>
            <td>${{htmlList(resolved)}}</td>
          </tr>`;
      }}).join("");
      return `
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Action</th>
              <th>Reward</th>
              <th>Cum Return</th>
              <th>Cash</th>
              <th>Worst Case</th>
              <th>Open Positions</th>
              <th>Resolved</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function offerTable(step) {{
      const offers = step.state_after.offers || [];
      if (offers.length === 0) {{
        return '<span class="muted">No active offers</span>';
      }}
      const rows = offers.map(offer => `
        <tr>
          <td>${{offer.offer_id}}</td>
          <td>${{offer.poster}}</td>
          <td>${{offer.side}}</td>
          <td>${{offer.target}}</td>
          <td>${{offer.deadline}}</td>
          <td>${{offer.loss.toFixed(2)}}</td>
          <td>${{offer.price.toFixed(2)}}</td>
        </tr>`).join("");
      return `
        <table>
          <thead>
            <tr>
              <th>Offer</th>
              <th>Poster</th>
              <th>Side</th>
              <th>Target</th>
              <th>Deadline</th>
              <th>Loss</th>
              <th>Price</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function renderStep(idx) {{
      const step = steps[Math.max(0, Math.min(idx, steps.length - 1))];
      if (!step) {{
        stepLabel.textContent = "No captured steps";
        return;
      }}
      stepLabel.textContent = `${{idx + 1}} / ${{steps.length}}`;
      document.getElementById("envState").innerHTML = `
        <div class="pill">t=${{step.state_after.timestep}}</div>
        <div class="pill">done=${{step.done}}</div>
        <h3>Public Resolved</h3>
        <div>${{htmlList(step.state_after.public_library.resolved.map(item => item.formula))}}</div>
        <h3>Query Model Quality</h3>
        <div>${{htmlList((step.state_before.query_model_quality || []).map(diag =>
          `agent ${{diag.agent_id}}: mae=${{diag.mae_all.toFixed(3)}} rmse=${{diag.rmse_all.toFixed(3)}} ` +
          `(feasible mae=${{diag.mae_feasible.toFixed(3)}})`
        ))}}</div>
        <h3>Jobs</h3>
        <div>${{htmlList(step.state_after.jobs.map((job, i) => job ? `agent ${{i}}: ${{job.type}} phi=${{job.target}} tau=${{job.tau_rem}}` : null).filter(Boolean), true)}}</div>
      `;
      document.getElementById("actions").innerHTML = agentTable(step);
      document.getElementById("offers").innerHTML = offerTable(step);
      document.getElementById("rawPayload").textContent = JSON.stringify(step, null, 2);
    }}

    renderCharts();
    renderStep(0);
    range.addEventListener("input", event => renderStep(Number(event.target.value)));
  </script>
</body>
</html>"""


def write_trajectory_artifacts(trajectories, output_dir, split, epoch):
    os.makedirs(output_dir, exist_ok=True)
    written_paths = []

    for trajectory in trajectories:
        base_name = f"{split}_epoch_{epoch + 1:04d}_thread_{trajectory['thread_index']:02d}"
        json_path = os.path.join(output_dir, f"{base_name}.json")
        html_path = os.path.join(output_dir, f"{base_name}.html")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(trajectory, handle, indent=2)

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(
                _trajectory_html_document(
                    title=f"Trajectory Viewer: {base_name}",
                    payload_json=json.dumps(trajectory),
                )
            )

        written_paths.append({
            'json': json_path,
            'html': html_path,
            'thread_index': trajectory['thread_index'],
        })

    return written_paths
