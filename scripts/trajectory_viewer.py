from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _infer_f_size(trajectory: dict) -> int:
    max_phi = -1

    def visit_formula(phi: Optional[int]) -> None:
        nonlocal max_phi
        if phi is not None:
            max_phi = max(max_phi, int(phi))

    def visit_library(lib: dict) -> None:
        for phi in lib.get("concrete", []):
            visit_formula(phi)
        for item in lib.get("resolved", []):
            visit_formula(item.get("formula"))
            for dep in item.get("deps", []):
                visit_formula(dep)

    snapshots = [trajectory["initial_state"]]
    snapshots.extend(step["state_after"] for step in trajectory.get("steps", []))
    for snapshot in snapshots:
        visit_library(snapshot.get("public_library", {}))
        for agent in snapshot.get("agents", []):
            visit_library(agent.get("library", {}))
            for pos in agent.get("positions", []):
                visit_formula(pos.get("target"))
        for offer in snapshot.get("offers", []):
            visit_formula(offer.get("target"))
        for job in snapshot.get("jobs", []):
            if job is not None:
                visit_formula(job.get("target"))
        for response in snapshot.get("query_responses", []):
            if response is not None:
                visit_formula(response.get("formula"))

    for step in trajectory.get("steps", []):
        for action in step.get("actions", []):
            visit_formula(action.get("formula"))

    return max(max_phi + 1, 0)


def _formula_label(phi: int, num_theorems: int) -> str:
    theorem_id = int(phi) % num_theorems
    sign = 0 if int(phi) < num_theorems else 1
    return f"¬φ{theorem_id}" if sign else f"φ{theorem_id}"


def _agent_label(agent_id: int) -> str:
    return f"a_{int(agent_id)}"


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _format_optional_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "?"
    return _format_float(float(value), digits)


def _serialize_formula_list(formulas: Iterable[int], num_theorems: int) -> List[str]:
    return [_formula_label(int(phi), num_theorems) for phi in formulas]


def _offer_lookup(snapshot: dict) -> List[dict]:
    return sorted(snapshot.get("offers", []), key=lambda offer: int(offer["offer_id"]))


def _offer_for_accept(snapshot: dict, slot: Optional[int]) -> Optional[dict]:
    if slot is None:
        return None
    offers = _offer_lookup(snapshot)
    if 0 <= int(slot) < len(offers):
        return offers[int(slot)]
    return None


def _offer_for_cancel(snapshot: dict, agent_id: int, slot: Optional[int]) -> Optional[dict]:
    if slot is None:
        return None
    own_offers = [offer for offer in _offer_lookup(snapshot) if int(offer["poster"]) == int(agent_id)]
    if 0 <= int(slot) < len(own_offers):
        return own_offers[int(slot)]
    return None


def _action_targets(action: dict, snapshot: dict, agent_id: int) -> List[int]:
    action_type = action.get("type")
    formula = action.get("formula")
    if formula is not None:
        return [int(formula)]
    if action_type == "accept":
        offer = _offer_for_accept(snapshot, action.get("offer_slot"))
        return [] if offer is None else [int(offer["target"])]
    if action_type == "cancel":
        offer = _offer_for_cancel(snapshot, agent_id, action.get("offer_slot"))
        return [] if offer is None else [int(offer["target"])]
    return []


def _format_offer_summary(offer: dict, num_theorems: int) -> str:
    return (
        f"{offer['side']} {_formula_label(int(offer['target']), num_theorems)} "
        f"x{int(offer['quantity'])} d={int(offer['deadline'])} "
        f"loss={_format_float(float(offer['loss']), 2)} @ {_format_float(float(offer['price']), 2)}"
    )


def _format_position_summary(pos: dict, num_theorems: int) -> str:
    pnl = f" pnl={_format_float(float(pos.get('pnl', 0.0)), 2)}"
    return (
        f"{pos['side']} {_formula_label(int(pos['target']), num_theorems)} "
        f"x{int(pos['quantity'])} d={int(pos['deadline'])} "
        f"loss={_format_float(float(pos['loss']), 2)}{pnl}"
    )


def _format_action(action: Optional[dict], snapshot: dict, agent_id: int, cfg: Any, num_theorems: int) -> str:
    if not action:
        return "—"

    action_type = action.get("type", "noop")
    formula = action.get("formula")

    if action_type in {"noop", "market_noop"}:
        return "noop"

    if action_type in {"prove", "conj"}:
        tau = action.get("budget")
        if cfg is not None and tau is not None:
            tau = cfg.budget_levels[int(tau)]
        label = _formula_label(int(formula), num_theorems) if formula is not None else "?"
        return f"noop ({action_type} {label}, τ={tau})"

    if action_type == "pub":
        return f"pub({_formula_label(int(formula), num_theorems)})"

    if action_type == "qry":
        return f"qry({_formula_label(int(formula), num_theorems)})"

    if action_type == "create_post":
        deadline = action.get("deadline")
        loss = action.get("loss")
        price = action.get("price")
        if cfg is not None:
            if deadline is not None:
                deadline = cfg.deadline_levels[int(deadline)]
            if loss is not None:
                loss = cfg.loss_levels[int(loss)]
            if price is not None:
                price = cfg.price_levels[int(price)]
        return (
            "create("
            f"{_formula_label(int(formula), num_theorems)}, "
            f"d={deadline}, loss={_format_optional_float(loss, 2)}, "
            f"{action.get('side')}, p={_format_optional_float(price, 2)})"
        )

    if action_type == "accept":
        qty_labels = {0: "one", 1: "half", 2: "all"}
        qty_tag = qty_labels.get(action.get("accept_quantity"), "one")
        offer = _offer_for_accept(snapshot, action.get("offer_slot"))
        if offer is None:
            return f"accept(slot {action.get('offer_slot')}, qty={qty_tag})"
        return f"accept({_format_offer_summary(offer, num_theorems)}, qty={qty_tag})"

    if action_type == "cancel":
        offer = _offer_for_cancel(snapshot, agent_id, action.get("offer_slot"))
        if offer is None:
            return f"cancel(slot {action.get('offer_slot')})"
        return f"cancel({_format_offer_summary(offer, num_theorems)})"

    return action_type


def _resolved_formulas(snapshot: dict, agent_id: Optional[int] = None) -> set[int]:
    library = snapshot["public_library"] if agent_id is None else snapshot["agents"][agent_id]["library"]
    return {int(item["formula"]) for item in library.get("resolved", [])}


def _concrete_formulas(snapshot: dict, agent_id: Optional[int] = None) -> set[int]:
    library = snapshot["public_library"] if agent_id is None else snapshot["agents"][agent_id]["library"]
    return {int(phi) for phi in library.get("concrete", [])}


def _derive_step_results(step: dict, num_theorems: int) -> List[str]:
    results: List[str] = []
    before_snapshot = step["state_before"]
    after_snapshot = step["state_after"]
    num_agents = len(after_snapshot.get("agents", []))

    public_before = _resolved_formulas(before_snapshot)
    public_after = _resolved_formulas(after_snapshot)
    before_offer_ids = {int(offer["offer_id"]) for offer in before_snapshot.get("offers", [])}
    after_offers = {int(offer["offer_id"]): offer for offer in after_snapshot.get("offers", [])}

    for agent_id in range(num_agents):
        parts: List[str] = []
        action = step["actions"][agent_id]
        before_agent = before_snapshot["agents"][agent_id]
        after_agent = after_snapshot["agents"][agent_id]

        before_resolved = _resolved_formulas(before_snapshot, agent_id)
        after_resolved = _resolved_formulas(after_snapshot, agent_id)
        before_concrete = _concrete_formulas(before_snapshot, agent_id)
        after_concrete = _concrete_formulas(after_snapshot, agent_id)

        before_job = before_snapshot["jobs"][agent_id]
        after_job = after_snapshot["jobs"][agent_id]

        if action.get("type") == "qry":
            response = after_snapshot["query_responses"][agent_id]
            if response is not None:
                parts.append(
                    "query "
                    f"{_formula_label(int(response['formula']), num_theorems)} -> "
                    f"p̂={_format_float(float(response['p_hat']))}, "
                    f"τ̂={_format_float(float(response['tau_hat']))}"
                )

        if before_job is not None and after_job is None:
            target = int(before_job["target"])
            if before_job["type"] == "prove":
                success = target in (after_resolved - before_resolved)
                verdict = "succeeded" if success else "failed"
                parts.append(f"prove {_formula_label(target, num_theorems)} {verdict}")
            elif before_job["type"] == "conj":
                new_concrete = sorted(after_concrete - before_concrete)
                if new_concrete:
                    labels = ", ".join(_serialize_formula_list(new_concrete, num_theorems))
                    parts.append(f"conj -> {labels}")
                else:
                    parts.append("conj failed")

        if action.get("type") == "pub":
            newly_public = sorted(public_after - public_before)
            if newly_public:
                parts.append(f"published {', '.join(_serialize_formula_list(newly_public, num_theorems))}")

        if action.get("type") == "create_post":
            new_offers = [
                offer for offer in after_snapshot.get("offers", [])
                if int(offer["offer_id"]) not in before_offer_ids and int(offer["poster"]) == agent_id
            ]
            for offer in new_offers:
                parts.append(f"posted {_format_offer_summary(offer, num_theorems)}")

        if action.get("type") == "cancel":
            before_own = {
                int(offer["offer_id"]): offer
                for offer in before_snapshot.get("offers", [])
                if int(offer["poster"]) == agent_id
            }
            canceled = [offer for offer_id, offer in before_own.items() if offer_id not in after_offers]
            for offer in canceled:
                parts.append(f"canceled {_format_offer_summary(offer, num_theorems)}")

        if action.get("type") == "accept":
            before_positions = {
                (
                    pos["target"],
                    pos["deadline"],
                    pos["loss"],
                    pos["side"],
                    pos["quantity"],
                )
                for pos in before_agent.get("positions", [])
                if not pos.get("settled", False)
            }
            new_positions = [
                pos for pos in after_agent.get("positions", [])
                if not pos.get("settled", False)
                and (
                    pos["target"],
                    pos["deadline"],
                    pos["loss"],
                    pos["side"],
                    pos["quantity"],
                ) not in before_positions
            ]
            for pos in new_positions:
                parts.append(f"accepted {_format_position_summary(pos, num_theorems)}")

        results.append("; ".join(parts))

    return results


def _build_formula_nodes(cfg: Any, num_theorems: int) -> List[dict]:
    nodes: List[dict] = []
    cols = max(1, math.ceil(math.sqrt(num_theorems)))
    x_spacing = 260
    y_spacing = 190
    pair_gap = 72
    for theorem_id in range(num_theorems):
        col = theorem_id % cols
        row = theorem_id // cols
        center_x = 120 + col * x_spacing
        center_y = 110 + row * y_spacing
        for sign in (0, 1):
            phi = theorem_id + sign * num_theorems
            is_true = bool(cfg.truth_map[theorem_id] == sign) if cfg is not None else sign == 0
            nodes.append(
                {
                    "phi": int(phi),
                    "theorem_id": int(theorem_id),
                    "sign": int(sign),
                    "label": _formula_label(phi, num_theorems),
                    "truth": bool(is_true),
                    "difficulty": float(cfg.difficulty_map[theorem_id]) if cfg is not None else 0.5,
                    "is_false_member": not is_true,
                    "x": float(center_x + (-pair_gap / 2 if sign == 0 else pair_gap / 2)),
                    "y": float(center_y),
                }
            )
    return nodes


def _build_utility_edges(cfg: Any) -> List[dict]:
    if cfg is None:
        return []
    edges: List[dict] = []
    num_theorems = int(cfg.num_theorems)
    for (src_theorem, dst_theorem), weight in sorted(cfg.utility_weights.items()):
        src_phi = int(src_theorem) + int(cfg.truth_map[int(src_theorem)]) * num_theorems
        dst_phi = int(dst_theorem) + int(cfg.truth_map[int(dst_theorem)]) * num_theorems
        edges.append(
            {
                "source": src_phi,
                "target": dst_phi,
                "weight": float(weight),
                "is_dependency": float(weight) == 1.0,
            }
        )
    return edges


def _build_agent_colors(num_agents: int) -> List[str]:
    palette = [
        "#0f766e",
        "#c2410c",
        "#2563eb",
        "#8b5cf6",
        "#65a30d",
        "#db2777",
        "#0891b2",
        "#b45309",
    ]
    return [palette[idx % len(palette)] for idx in range(num_agents)]


def build_viewer_payload(
    trajectory: dict,
    *,
    cfg: Any = None,
    oracle_series: Optional[List[float]] = None,
    seed_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    steps = trajectory.get("steps", [])
    initial_state = trajectory["initial_state"]
    num_agents = len(initial_state.get("agents", []))
    f_size = int(cfg.F_size) if cfg is not None else _infer_f_size(trajectory)
    num_theorems = max(f_size // 2, 1)

    economic_series = [[0.0] for _ in range(num_agents)]
    rmse_series = [
        [float(initial_state.get("query_model_quality", [{}] * num_agents)[agent_id].get("rmse_all", 0.0))]
        for agent_id in range(num_agents)
    ]
    public_resolved_series = [float(len(initial_state.get("public_library", {}).get("resolved", [])))]
    private_resolved_series = [
        [float(len(initial_state["agents"][agent_id]["library"].get("resolved", [])))]
        for agent_id in range(num_agents)
    ]

    for step in steps:
        for agent_id in range(num_agents):
            economic_series[agent_id].append(float(step.get("cumulative_economic_returns", [0.0] * num_agents)[agent_id]))
            rmse_series[agent_id].append(float(step["state_after"].get("query_model_quality", [{}] * num_agents)[agent_id].get("rmse_all", 0.0)))
            private_resolved_series[agent_id].append(float(len(step["state_after"]["agents"][agent_id]["library"].get("resolved", []))))
        public_resolved_series.append(float(len(step["state_after"]["public_library"].get("resolved", []))))

    economic_mean = [
        float(sum(series[t] for series in economic_series) / max(num_agents, 1))
        for t in range(len(economic_series[0]) if economic_series else 0)
    ]

    previous_results = [_derive_step_results(step, num_theorems) for step in steps]

    frames: List[dict] = []
    current_snapshot = initial_state
    for frame_index in range(len(steps) + 1):
        previous_step = steps[frame_index - 1] if frame_index > 0 else None
        current_step = steps[frame_index] if frame_index < len(steps) else None
        if frame_index > 0:
            current_snapshot = previous_step["state_after"]

        current_actions = []
        previous_actions = []
        previous_result_texts = previous_results[frame_index - 1] if frame_index > 0 else [""] * num_agents
        touched: Dict[str, List[int]] = {}
        holdings = []
        offers = []
        cash = []
        worst_case = []
        private_concrete_agents = []
        private_resolved_agents = []

        for agent_id in range(num_agents):
            previous_actions.append(
                _format_action(
                    None if previous_step is None else previous_step["actions"][agent_id],
                    initial_state if previous_step is None else previous_step["state_before"],
                    agent_id,
                    cfg,
                    num_theorems,
                )
            )
            current_actions.append(
                _format_action(
                    None if current_step is None else current_step["actions"][agent_id],
                    current_snapshot,
                    agent_id,
                    cfg,
                    num_theorems,
                )
            )
            if current_step is not None:
                for phi in _action_targets(current_step["actions"][agent_id], current_snapshot, agent_id):
                    touched.setdefault(str(int(phi)), []).append(agent_id)

            agent_state = current_snapshot["agents"][agent_id]
            cash.append(float(agent_state["cash"]))
            worst_case.append(float(agent_state["worst_case_balance"]))
            holdings.append([
                _format_position_summary(pos, num_theorems)
                for pos in agent_state.get("positions", [])
                if not pos.get("settled", False)
            ])
            offers.append([
                _format_offer_summary(offer, num_theorems)
                for offer in _offer_lookup(current_snapshot)
                if int(offer["poster"]) == agent_id
            ])
            private_concrete_agents.append(sorted(int(phi) for phi in agent_state["library"].get("concrete", [])))
            private_resolved_agents.append(sorted(int(item["formula"]) for item in agent_state["library"].get("resolved", [])))

        frames.append(
            {
                "frame_index": int(frame_index),
                "timestep": int(current_snapshot["timestep"]),
                "public_concrete": sorted(int(phi) for phi in current_snapshot["public_library"].get("concrete", [])),
                "public_resolved": sorted(int(item["formula"]) for item in current_snapshot["public_library"].get("resolved", [])),
                "private_concrete_agents": private_concrete_agents,
                "private_resolved_agents": private_resolved_agents,
                "cash": cash,
                "worst_case_balance": worst_case,
                "holdings": holdings,
                "offers": offers,
                "all_offers": [
                    {
                        "offer_id": int(offer["offer_id"]),
                        "target": _formula_label(int(offer["target"]), num_theorems),
                        "target_raw": int(offer["target"]),
                        "deadline": int(offer["deadline"]),
                        "loss": float(offer["loss"]),
                        "side": offer["side"],
                        "price": float(offer["price"]),
                        "quantity": int(offer["quantity"]),
                        "poster": int(offer["poster"]),
                    }
                    for offer in _offer_lookup(current_snapshot)
                ],
                "current_actions": current_actions,
                "previous_actions": previous_actions,
                "previous_results": previous_result_texts,
                "touched": touched,
            }
        )

    return {
        "title": label or f"Trajectory Viewer: thread {trajectory.get('thread_index', 0)}",
        "seed_dir": seed_dir or "",
        "thread_index": int(trajectory.get("thread_index", 0)),
        "num_agents": int(num_agents),
        "num_theorems": int(num_theorems),
        "f_size": int(f_size),
        "formula_nodes": _build_formula_nodes(cfg, num_theorems),
        "utility_edges": _build_utility_edges(cfg),
        "agent_colors": _build_agent_colors(num_agents),
        "frames": frames,
        "series": {
            "economic": economic_series,
            "economic_mean": economic_mean,
            "rmse": rmse_series,
            "public_resolved": public_resolved_series,
            "private_resolved": private_resolved_series,
            "oracle_public_resolved": [] if oracle_series is None else [float(v) for v in oracle_series],
        },
        "config_available": bool(cfg is not None),
        "oracle_available": bool(oracle_series),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__PAGE_TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/webcola@3.4.0/WebCola/cola.min.js"></script>
  <script src="https://unpkg.com/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1f1a14;
      --muted: #655d52;
      --line: #d9ccbb;
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --warning: #b45309;
      --shadow: 0 14px 30px rgba(31, 26, 20, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.75), transparent 36%),
        linear-gradient(180deg, #f8f4ec 0%, #ece1cf 100%);
    }
    .page {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px 24px 32px;
    }
    .header {
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.1;
    }
    .muted {
      color: var(--muted);
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      font-size: 13px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: 0 0 auto;
    }
    .controls {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px 18px;
      margin-bottom: 18px;
    }
    .controls-row {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }
    .controls-row + .controls-row {
      margin-top: 10px;
    }
    button {
      border: 1px solid rgba(15, 118, 110, 0.35);
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      cursor: pointer;
    }
    button:hover {
      background: rgba(15, 118, 110, 0.2);
    }
    label {
      font-weight: 600;
    }
    input[type="range"] {
      flex: 1 1 320px;
      accent-color: var(--accent);
    }
    .step-readout {
      min-width: 180px;
      text-align: right;
      font-weight: 700;
    }
    .toggle-group {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .toggle-group label {
      font-weight: 500;
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      grid-template-areas:
        "table table"
        "graph economic"
        "resolved rmse"
        "offers .";
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 16px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .panel.graph { grid-area: graph; }
    .panel.economic { grid-area: economic; }
    .panel.resolved { grid-area: resolved; }
    .panel.rmse { grid-area: rmse; }
    .panel.offers { grid-area: offers; }
    .panel.table { grid-area: table; }
    #offersTable { font-size: 13px; }
    #offersTable td, #offersTable th { padding: 4px 8px; }
    .chart-shell {
      width: 100%;
      overflow: hidden;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
    }
    .chart-svg {
      width: 100%;
      height: 300px;
      display: block;
    }
    .graph-shell {
      position: relative;
      width: 100%;
      overflow: hidden;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      min-height: 430px;
    }
    #cy {
      width: 100%;
      height: 430px;
      display: block;
    }
    .tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      max-width: 260px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(31, 26, 20, 0.94);
      color: #fffaf2;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 8px 22px rgba(31, 26, 20, 0.3);
      z-index: 2;
    }
    .tooltip .green {
      color: #8ae38a;
    }
    .panel-note {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      background: #fff8ef;
      z-index: 1;
    }
    td.current-action, th.current-action {
      font-weight: 700;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
    }
    .contract-list {
      margin: 0;
      padding-left: 18px;
    }
    .contract-list.empty {
      padding-left: 0;
      list-style: none;
      color: var(--muted);
    }
    .hidden-prev-action .prev-action-col,
    .hidden-prev-result .prev-result-col {
      display: none;
    }
    .pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      font-size: 12px;
    }
    @media (max-width: 980px) {
      .layout {
        grid-template-columns: minmax(0, 1fr);
        grid-template-areas:
          "table"
          "graph"
          "economic"
          "resolved"
          "rmse"
          "offers";
      }
      .step-readout {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>__PAGE_HEADING__</h1>
      <div class="muted" id="summary"></div>
      <div class="legend" id="agentLegend"></div>
    </div>

    <div class="controls">
      <div class="controls-row">
        <label for="stepRange">Timestep</label>
        <input id="stepRange" type="range" min="0" value="0">
        <button id="playPause" type="button">Autoplay</button>
        <div class="step-readout" id="stepReadout"></div>
      </div>
      <div class="controls-row">
        <div class="toggle-group">
          <label><input id="togglePrevAction" type="checkbox" checked> Show previous action</label>
          <label><input id="togglePrevResult" type="checkbox" checked> Show previous result</label>
        </div>
      </div>
    </div>

    <div class="layout" id="layoutRoot">
      <section class="panel table">
        <h2>Agent State Table</h2>
        <div class="table-wrap">
          <table id="agentTable">
            <thead>
              <tr>
                <th>Agent</th>
                <th class="prev-action-col">Previous Action</th>
                <th class="prev-result-col">Previous Result</th>
                <th class="current-action current-action-col">Current Action</th>
                <th>Cash</th>
                <th>Worst Case</th>
                <th>Private Holdings</th>
                <th>Available Offers</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>

      <section class="panel graph">
        <h2>Library State Graph</h2>
        <div class="graph-shell" id="graphShell">
          <div id="cy"></div>
          <canvas id="cyOverlay" style="position:absolute;top:0;left:0;pointer-events:none;"></canvas>
          <div class="tooltip" id="graphTooltip"></div>
        </div>
        <div class="panel-note" id="graphNote"></div>
      </section>

      <section class="panel economic">
        <h2>Economic Return Over Time</h2>
        <div class="chart-shell">
          <svg class="chart-svg" id="economicChart" viewBox="0 0 840 300"></svg>
        </div>
      </section>

      <section class="panel resolved">
        <h2>Resolved Formulas Over Time</h2>
        <div class="chart-shell">
          <svg class="chart-svg" id="resolvedChart" viewBox="0 0 840 300"></svg>
        </div>
      </section>

      <section class="panel rmse">
        <h2>Query Model RMSE Over Time</h2>
        <div class="chart-shell">
          <svg class="chart-svg" id="rmseChart" viewBox="0 0 840 300"></svg>
        </div>
      </section>

      <section class="panel offers">
        <h2>Market Offers</h2>
        <div class="table-wrap">
          <table id="offersTable">
            <thead>
              <tr>
                <th>ID</th>
                <th>Formula</th>
                <th>Side</th>
                <th>Price</th>
                <th>Qty</th>
                <th>Deadline</th>
                <th>Loss</th>
                <th>Poster</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script id="viewerData" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById("viewerData").textContent);
    const frames = payload.frames;
    const series = payload.series;
    const agentColors = payload.agent_colors;
    const formulaNodes = payload.formula_nodes;
    const utilityEdges = payload.utility_edges;
    const stepRange = document.getElementById("stepRange");
    const stepReadout = document.getElementById("stepReadout");
    const summary = document.getElementById("summary");
    const playPause = document.getElementById("playPause");
    const layoutRoot = document.getElementById("layoutRoot");
    const tooltip = document.getElementById("graphTooltip");
    const graphNote = document.getElementById("graphNote");
    const agentLegend = document.getElementById("agentLegend");
    let currentFrame = 0;
    let autoplayTimer = null;

    summary.textContent = `Seed ${payload.seed_dir || "?"} | thread=${payload.thread_index} | timesteps=${Math.max(frames.length - 1, 0)} | agents=${payload.num_agents}`;
    stepRange.max = Math.max(frames.length - 1, 0);

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderLegend() {
      agentLegend.innerHTML = agentColors.map((color, idx) => (
        `<span class="legend-item"><span class="swatch" style="background:${color}"></span>${escapeHtml(`a_${idx}`)}</span>`
      )).join("");
    }

    function difficultyColor(value) {
      const clamp = Math.max(0, Math.min(1, Number(value) || 0));
      const low = [59, 130, 246];
      const mid = [250, 204, 21];
      const high = [239, 68, 68];
      const useHigh = clamp >= 0.5;
      const alpha = useHigh ? (clamp - 0.5) / 0.5 : clamp / 0.5;
      const left = useHigh ? mid : low;
      const right = useHigh ? high : mid;
      const rgb = left.map((start, idx) => Math.round(start + (right[idx] - start) * alpha));
      return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    }

    function formulaSet(items) {
      return new Set(items.map(value => Number(value)));
    }

    function buildFrameState(frame) {
      const publicConcrete = formulaSet(frame.public_concrete);
      const publicResolved = formulaSet(frame.public_resolved);
      const privateConcrete = frame.private_concrete_agents.map(formulaSet);
      const privateResolved = frame.private_resolved_agents.map(formulaSet);
      return { publicConcrete, publicResolved, privateConcrete, privateResolved };
    }

    function tooltipHtml(node, state) {
      const phi = Number(node.phi);
      const publicConcrete = state.publicConcrete.has(phi);
      const publicResolved = state.publicResolved.has(phi);
      const privateConcreteAgents = [];
      const privateResolvedAgents = [];
      for (let agentId = 0; agentId < payload.num_agents; agentId += 1) {
        if (state.privateResolved[agentId].has(phi)) {
          privateResolvedAgents.push(agentId);
        } else if (state.privateConcrete[agentId].has(phi)) {
          privateConcreteAgents.push(agentId);
        }
      }
      const lines = [
        `<strong>${escapeHtml(node.label)} (id=${phi})</strong>`,
        `Public state: ${publicConcrete ? "concrete" : "ghost"}`,
        `Publicly resolved: ${publicResolved ? "yes" : "no"}`,
        `Truth: ${node.truth ? "true" : "false"}`,
        `Difficulty: ${Number(node.difficulty).toFixed(3)}`
      ];
      if (!publicConcrete && privateConcreteAgents.length) {
        lines.push(`Private concrete: ${privateConcreteAgents.map(id => escapeHtml(`a_${id}`)).join(", ")}`);
      }
      if (!publicResolved && privateResolvedAgents.length) {
        lines.push(`<span class="green">Private resolved: ${privateResolvedAgents.map(id => escapeHtml(`a_${id}`)).join(", ")}</span>`);
      }
      return lines.join("<br>");
    }

    function describeGraphAvailability() {
      graphNote.textContent = payload.config_available
        ? "Drag nodes to rearrange (physics responds). Scroll to zoom, drag background to pan."
        : "Run metadata was missing, so the graph falls back to limited defaults.";
    }

    // ---- Striped SVG for private-concrete fill ----
    function stripesSvg(color) {
      return "data:image/svg+xml," + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="46" height="46">' +
        '<rect width="46" height="46" fill="' + color + '"/>' +
        '<g stroke="rgba(255,255,255,0.55)" stroke-width="3">' +
        '<line x1="-2" y1="6" x2="6" y2="-2"/><line x1="-2" y1="14" x2="14" y2="-2"/>' +
        '<line x1="-2" y1="22" x2="22" y2="-2"/><line x1="-2" y1="30" x2="30" y2="-2"/>' +
        '<line x1="-2" y1="38" x2="38" y2="-2"/><line x1="-2" y1="46" x2="46" y2="-2"/>' +
        '<line x1="6" y1="48" x2="48" y2="6"/><line x1="14" y1="48" x2="48" y2="14"/>' +
        '<line x1="22" y1="48" x2="48" y2="22"/><line x1="30" y1="48" x2="48" y2="30"/>' +
        '<line x1="38" y1="48" x2="48" y2="38"/>' +
        '</g></svg>'
      );
    }

    // ---- Build Cytoscape elements ----
    const cyElements = [];
    // Compound parents: one per theorem so phi/not-phi move as a unit
    const seenTheorems = new Set();
    formulaNodes.forEach(function(node) {
      var tid = Number(node.theorem_id);
      if (!seenTheorems.has(tid)) {
        seenTheorems.add(tid);
        cyElements.push({ group: "nodes", data: { id: "t" + tid, isParent: true } });
      }
      cyElements.push({
        group: "nodes",
        data: {
          id: "n" + node.phi, parent: "t" + tid,
          phi: Number(node.phi), label: node.label,
          difficulty: Number(node.difficulty), truth: Boolean(node.truth),
          isFalseMember: Boolean(node.is_false_member),
        },
        position: { x: node.x, y: node.y },
      });
    });
    utilityEdges.forEach(function(edge, idx) {
      var w = Number(edge.weight);
      cyElements.push({
        group: "edges",
        data: {
          id: "e" + idx, source: "n" + edge.source, target: "n" + edge.target,
          weight: w, label: w.toFixed(2), isDep: w === 1.0,
        },
      });
    });

    // ---- Create Cytoscape with cola live-physics layout ----
    var cy = cytoscape({
      container: document.getElementById("cy"),
      elements: cyElements,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
      minZoom: 0.25,
      maxZoom: 3.0,
      style: [
        // Compound parents: invisible grouping container
        {
          selector: "node[?isParent]",
          style: {
            "shape": "roundrectangle",
            "background-opacity": 0,
            "border-width": 0,
            "padding": "6px",
            "label": "",
          },
        },
        // Formula child nodes
        {
          selector: "node[^isParent]",
          style: {
            "width": 46, "height": 46,
            "label": "data(label)",
            "text-valign": "center", "text-halign": "center",
            "font-size": 13, "font-weight": 700,
            "color": "#1f1a14",
            "text-outline-color": "rgba(255,250,242,0.8)", "text-outline-width": 1.5,
            "background-color": "#ccc",
            "border-width": 2, "border-color": "#53463a", "border-style": "solid",
          },
        },
        // Edges — slim and elegant
        {
          selector: "edge",
          style: {
            "width": 1.2, "line-color": "#b5a898",
            "target-arrow-color": "#b5a898", "target-arrow-shape": "vee",
            "curve-style": "bezier", "arrow-scale": 0.7, "opacity": 0.6,
            "label": "data(label)", "font-size": 9, "color": "#8a7e72",
            "text-background-color": "#fffaf2", "text-background-opacity": 0.8,
            "text-background-padding": "2px", "text-rotation": "autorotate",
          },
        },
        { selector: "edge[?isDep]", style: { "width": 2, "opacity": 0.75 } },
        { selector: "edge[!isDep]", style: { "line-style": "dashed", "line-dash-pattern": [6, 5] } },
      ],
      layout: {
        name: "cola",
        animate: true,
        infinite: true,
        fit: false,
        ungrabifyWhileSimulating: false,
        handleDisconnected: true,
        avoidOverlap: true,
        nodeSpacing: function() { return 30; },
        edgeLength: function(edge) {
          return edge.data("isDep") ? 200 : 180;
        },
        convergenceThreshold: 0.001,
        padding: 50,
      },
    });

    // Fit after initial settle
    setTimeout(function() { cy.fit(undefined, 50); }, 800);

    // ---- Overlay canvas for rings and arcs ----
    var overlay = document.getElementById("cyOverlay");
    var overlayCtx = overlay.getContext("2d");

    function resizeOverlay() {
      var rect = cy.container().getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      overlay.width = rect.width * dpr;
      overlay.height = rect.height * dpr;
      overlay.style.width = rect.width + "px";
      overlay.style.height = rect.height + "px";
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function drawOverlay() {
      resizeOverlay();
      var w = parseFloat(overlay.style.width);
      var h = parseFloat(overlay.style.height);
      overlayCtx.clearRect(0, 0, w, h);

      var frame = frames[currentFrame];
      var state = buildFrameState(frame);

      cy.nodes("[^isParent]").forEach(function(cyNode) {
        var phi = cyNode.data("phi");
        var rp = cyNode.renderedPosition();
        var rw = cyNode.renderedWidth() / 2;
        var node = formulaNodes.find(function(n) { return Number(n.phi) === phi; });
        if (!node) return;

        var isPublicResolved = state.publicResolved.has(phi);
        var isPrivateResolved = false;
        for (var a = 0; a < payload.num_agents; a++) {
          if (state.privateResolved[a].has(phi)) isPrivateResolved = true;
        }

        // Resolution ring (inner ring, radius = node edge + 4)
        var ringR = rw + 4;
        if (isPublicResolved) {
          overlayCtx.beginPath();
          overlayCtx.arc(rp.x, rp.y, ringR, 0, 2 * Math.PI);
          overlayCtx.lineWidth = 4;
          overlayCtx.strokeStyle = "#16a34a";
          overlayCtx.setLineDash([]);
          overlayCtx.stroke();
        } else if (isPrivateResolved) {
          overlayCtx.beginPath();
          overlayCtx.arc(rp.x, rp.y, ringR, 0, 2 * Math.PI);
          overlayCtx.lineWidth = 4;
          overlayCtx.strokeStyle = "#16a34a";
          overlayCtx.setLineDash([5, 4]);
          overlayCtx.stroke();
          overlayCtx.setLineDash([]);
        }

        // Agent touch arcs (outer ring, radius = node edge + 10)
        var touchAgents = (frame.touched[String(phi)] || []).map(Number);
        if (touchAgents.length > 0) {
          var arcR = rw + 10;
          var total = touchAgents.length;
          for (var i = 0; i < total; i++) {
            var startA = (-Math.PI / 2) + (i * 2 * Math.PI / total);
            var endA = (-Math.PI / 2) + ((i + 1) * 2 * Math.PI / total);
            overlayCtx.beginPath();
            overlayCtx.arc(rp.x, rp.y, arcR, startA, endA);
            overlayCtx.lineWidth = 3.5;
            overlayCtx.strokeStyle = agentColors[touchAgents[i] % agentColors.length];
            overlayCtx.stroke();
          }
        }
      });
    }

    // Redraw overlay whenever Cytoscape renders (pan, zoom, drag, layout tick)
    cy.on("render", drawOverlay);

    // ---- Tooltip ----
    cy.on("mouseover", "node[^isParent]", function(event) {
      var nd = event.target.data();
      var node = formulaNodes.find(function(n) { return Number(n.phi) === nd.phi; });
      if (!node) return;
      tooltip.innerHTML = tooltipHtml(node, buildFrameState(frames[currentFrame]));
      tooltip.style.display = "block";
    });
    cy.on("mousemove", "node[^isParent]", function(event) {
      var shellRect = document.getElementById("graphShell").getBoundingClientRect();
      var rp = event.renderedPosition || event.position;
      tooltip.style.left = (rp.x + 14) + "px";
      tooltip.style.top = (rp.y + 14) + "px";
    });
    cy.on("mouseout", "node[^isParent]", function() { tooltip.style.display = "none"; });

    // ---- renderGraph: update node styles per frame ----
    function renderGraph() {
      var frame = frames[currentFrame];
      var state = buildFrameState(frame);
      cy.batch(function() {
        cy.nodes("[^isParent]").forEach(function(cyNode) {
          var phi = cyNode.data("phi");
          var node = formulaNodes.find(function(n) { return Number(n.phi) === phi; });
          if (!node) return;

          var isPublicConcrete = state.publicConcrete.has(phi);
          var isPublicResolved = state.publicResolved.has(phi);
          var isPrivateConcrete = false;
          for (var a = 0; a < payload.num_agents; a++) {
            if (state.privateConcrete[a].has(phi)) isPrivateConcrete = true;
          }
          // Ghost = not concrete in any library (public or private)
          var isGhost = !isPublicConcrete && !isPrivateConcrete;

          // Shape: diamond for ghost, circle for concrete
          var shape = isGhost ? "diamond" : "ellipse";

          // Fill
          var dColor = difficultyColor(node.difficulty);
          var styles = { "shape": shape };

          if (isGhost) {
            styles["background-color"] = "#d4d0cb";
            styles["background-opacity"] = 0.45;
            styles["background-image"] = "none";
          } else if (isPrivateConcrete && !isPublicConcrete) {
            // Striped difficulty fill
            styles["background-color"] = dColor;
            styles["background-opacity"] = 1;
            styles["background-image"] = stripesSvg(dColor);
            styles["background-fit"] = "cover";
            styles["background-clip"] = "node";
          } else {
            // Public concrete — solid difficulty fill
            styles["background-color"] = dColor;
            styles["background-opacity"] = 1;
            styles["background-image"] = "none";
          }

          // Border: dashed exclusively for false members, solid otherwise
          styles["border-style"] = node.is_false_member ? "dashed" : "solid";
          styles["border-color"] = isGhost ? "#a39e96" : "#53463a";
          styles["border-width"] = 2;

          cyNode.style(styles);
        });
      });
      // Overlay is redrawn automatically via cy "render" event
    }

    function axisText(x, y, text, anchor = "middle") {
      return `<text x="${x}" y="${y}" fill="#655d52" font-size="12" text-anchor="${anchor}">${escapeHtml(text)}</text>`;
    }

    function renderChart(svgId, config) {
      const svg = document.getElementById(svgId);
      const width = 840;
      const height = 300;
      const pad = { left: 58, right: 20, top: 18, bottom: 42 };
      const current = currentFrame;
      const xMax = Math.max(config.maxX, 1);
      const activeSeries = [];
      const allYValues = [];

      config.series.forEach((item) => {
        const values = item.full ? item.values.slice() : item.values.slice(0, current + 1);
        activeSeries.push({ ...item, values });
        values.forEach((value) => allYValues.push(Number(value)));
      });
      if (!allYValues.length) {
        allYValues.push(0, 1);
      }

      let minY = Math.min(...allYValues);
      let maxY = Math.max(...allYValues);
      if (config.minY !== undefined) {
        minY = Math.min(minY, config.minY);
      }
      if (config.maxY !== undefined) {
        maxY = Math.max(maxY, config.maxY);
      }
      if (minY === maxY) {
        maxY += 1;
        minY -= 1;
      }

      function xFor(idx) {
        return pad.left + ((width - pad.left - pad.right) * idx / xMax);
      }

      function yFor(value) {
        return height - pad.bottom - ((height - pad.top - pad.bottom) * (value - minY) / (maxY - minY));
      }

      const grid = [];
      for (let i = 0; i <= 4; i += 1) {
        const yValue = minY + (maxY - minY) * i / 4;
        const y = yFor(yValue);
        grid.push(`<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#e3d8ca" stroke-width="1"></line>`);
        grid.push(axisText(pad.left - 8, y + 4, yValue.toFixed(2), "end"));
      }
      for (let i = 0; i <= 4; i += 1) {
        const xValue = Math.round(xMax * i / 4);
        const x = xFor(xValue);
        grid.push(`<line x1="${x}" y1="${pad.top}" x2="${x}" y2="${height - pad.bottom}" stroke="#efe6da" stroke-width="1"></line>`);
        grid.push(axisText(x, height - 14, String(xValue)));
      }

      const lines = activeSeries.map((item) => {
        if (!item.values.length) {
          return "";
        }
        const points = item.values.map((value, idx) => `${xFor(idx)},${yFor(Number(value))}`).join(" ");
        const dash = item.dashed ? `stroke-dasharray="7 5"` : "";
        return `<polyline fill="none" stroke="${item.color}" stroke-width="${item.width || 3}" ${dash} points="${points}"></polyline>`;
      }).join("");

      const cursorX = xFor(current);
      const cursor = `<line x1="${cursorX}" y1="${pad.top}" x2="${cursorX}" y2="${height - pad.bottom}" stroke="#1f1a14" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.45"></line>`;
      const axes = `
        <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#aa9d8e" stroke-width="1.5"></line>
        <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#aa9d8e" stroke-width="1.5"></line>
        ${axisText(width / 2, height - 4, config.xLabel)}
        <text x="16" y="${height / 2}" fill="#655d52" font-size="12" text-anchor="middle" transform="rotate(-90 16 ${height / 2})">${escapeHtml(config.yLabel)}</text>
      `;

      const legend = activeSeries.map((item, idx) => (
        `<g transform="translate(${pad.left + idx * 138}, ${pad.top - 2})">
          <line x1="0" y1="0" x2="26" y2="0" stroke="${item.color}" stroke-width="${item.width || 3}" ${item.dashed ? 'stroke-dasharray="7 5"' : ''}></line>
          <text x="32" y="4" fill="#655d52" font-size="12">${escapeHtml(item.label)}</text>
        </g>`
      )).join("");

      svg.innerHTML = `${grid.join("")}${axes}${cursor}${lines}<g>${legend}</g>`;
    }

    function listHtml(items) {
      if (!items || items.length === 0) {
        return '<ul class="contract-list empty"><li>none</li></ul>';
      }
      return `<ul class="contract-list">${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
    }

    function renderOffersTable() {
      const frame = frames[currentFrame];
      const body = document.querySelector("#offersTable tbody");
      const offers = frame.all_offers || [];
      if (offers.length === 0) {
        body.innerHTML = '<tr><td colspan="8" style="color:var(--muted);text-align:center">No offers</td></tr>';
        return;
      }
      body.innerHTML = offers.map(o => {
        const isAgent = o.poster < payload.num_agents;
        const posterLabel = isAgent
          ? `<span class="pill" style="border-color:${agentColors[o.poster]};color:${agentColors[o.poster]}">a_${o.poster}</span>`
          : `<span class="pill" style="border-color:var(--muted);color:var(--muted)">bounty</span>`;
        return `<tr>
          <td>${o.offer_id}</td>
          <td>${escapeHtml(o.target)}</td>
          <td>${o.side}</td>
          <td>${o.price.toFixed(2)}</td>
          <td>${o.quantity}</td>
          <td>${o.deadline}</td>
          <td>${o.loss.toFixed(2)}</td>
          <td>${posterLabel}</td>
        </tr>`;
      }).join("");
    }

    function renderTable() {
      const frame = frames[currentFrame];
      const body = document.querySelector("#agentTable tbody");
      const rows = [];
      for (let agentId = 0; agentId < payload.num_agents; agentId += 1) {
        rows.push(`
          <tr>
            <td><span class="pill" style="border-color:${agentColors[agentId]}; color:${agentColors[agentId]}">${escapeHtml(`a_${agentId}`)}</span></td>
            <td class="prev-action-col">${escapeHtml(frame.previous_actions[agentId] || "—")}</td>
            <td class="prev-result-col">${escapeHtml(frame.previous_results[agentId] || "") || "—"}</td>
            <td class="current-action current-action-col">${escapeHtml(frame.current_actions[agentId] || "—")}</td>
            <td>${Number(frame.cash[agentId]).toFixed(3)}</td>
            <td>${Number(frame.worst_case_balance[agentId]).toFixed(3)}</td>
            <td>${listHtml(frame.holdings[agentId])}</td>
            <td>${listHtml(frame.offers[agentId])}</td>
          </tr>
        `);
      }
      body.innerHTML = rows.join("");
    }

    function renderAllCharts() {
      renderChart("economicChart", {
        maxX: frames.length - 1,
        xLabel: "Timestep",
        yLabel: "Economic Return",
        series: [
          ...series.economic.map((values, agentId) => ({
            values,
            color: agentColors[agentId],
            label: `a_${agentId}`,
            width: 3,
          })),
          {
            values: series.economic_mean,
            color: "#111827",
            label: "mean",
            width: 2.5,
            dashed: true,
          },
        ],
      });

      renderChart("rmseChart", {
        maxX: frames.length - 1,
        xLabel: "Timestep",
        yLabel: "RMSE",
        minY: 0,
        series: series.rmse.map((values, agentId) => ({
          values,
          color: agentColors[agentId],
          label: `a_${agentId}`,
          width: 3,
        })),
      });

      const resolvedSeries = [
        {
          values: series.public_resolved,
          color: "#111827",
          label: "public",
          width: 3.5,
        },
        ...series.private_resolved.map((values, agentId) => ({
          values,
          color: agentColors[agentId],
          label: `a_${agentId}`,
          width: 2,
        })),
      ];
      if (series.oracle_public_resolved && series.oracle_public_resolved.length) {
        resolvedSeries.push({
          values: series.oracle_public_resolved,
          color: "#b45309",
          label: "oracle",
          width: 2.5,
          dashed: true,
          full: true,
        });
      }

      renderChart("resolvedChart", {
        maxX: Math.max(frames.length - 1, (series.oracle_public_resolved || []).length - 1),
        xLabel: "Timestep",
        yLabel: "Resolved Formulas",
        minY: 0,
        maxY: payload.f_size,
        series: resolvedSeries,
      });
    }

    function renderCurrentFrame() {
      const frame = frames[currentFrame];
      stepReadout.textContent = `frame ${currentFrame} / ${frames.length - 1} | t=${frame.timestep}`;
      stepRange.value = String(currentFrame);
      renderGraph();
      renderAllCharts();
      renderTable();
      renderOffersTable();
    }

    stepRange.addEventListener("input", (event) => {
      currentFrame = Number(event.target.value);
      renderCurrentFrame();
    });

    document.getElementById("togglePrevAction").addEventListener("change", (event) => {
      layoutRoot.classList.toggle("hidden-prev-action", !event.target.checked);
    });
    document.getElementById("togglePrevResult").addEventListener("change", (event) => {
      layoutRoot.classList.toggle("hidden-prev-result", !event.target.checked);
    });

    playPause.addEventListener("click", () => {
      if (autoplayTimer !== null) {
        window.clearInterval(autoplayTimer);
        autoplayTimer = null;
        playPause.textContent = "Autoplay";
        return;
      }
      playPause.textContent = "Pause";
      autoplayTimer = window.setInterval(() => {
        if (currentFrame >= frames.length - 1) {
          window.clearInterval(autoplayTimer);
          autoplayTimer = null;
          playPause.textContent = "Autoplay";
          return;
        }
        currentFrame += 1;
        renderCurrentFrame();
      }, 1000);
    });

    renderLegend();
    describeGraphAvailability();
    renderCurrentFrame();
  </script>
</body>
</html>
"""


def render_trajectory_viewer_html(payload: dict) -> str:
    page_title = html.escape(payload["title"])
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return (
        HTML_TEMPLATE
        .replace("__PAGE_TITLE__", page_title)
        .replace("__PAGE_HEADING__", page_title)
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def write_trajectory_viewer_html(
    output_path: str | Path,
    trajectory: dict,
    *,
    cfg: Any = None,
    oracle_series: Optional[List[float]] = None,
    seed_dir: Optional[str] = None,
    label: Optional[str] = None,
) -> Path:
    payload = build_viewer_payload(
        trajectory,
        cfg=cfg,
        oracle_series=oracle_series,
        seed_dir=seed_dir,
        label=label,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_trajectory_viewer_html(payload), encoding="utf-8")
    return out_path
