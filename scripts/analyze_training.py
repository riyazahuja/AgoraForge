#!/usr/bin/env python
"""Analyze MAPPO training results from TensorBoard event files.

Produces plots, a trajectory index, and a text summary.

Usage:
    python scripts/analyze_training.py --results_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.vamp.metadata import deserialize_config, load_run_metadata
from envs.vamp.oracle_solver import solve_public_resolution_oracle
from scripts.trajectory_viewer import write_trajectory_viewer_html

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: uv pip install matplotlib")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


AGENT_TAG_PATTERN = re.compile(r"online/(train|eval)_return_agent(\d+)$")
ECON_AGENT_TAG_PATTERN = re.compile(r"online/(train|eval)_economic_return_agent(\d+)$")
QUERY_TAG_PATTERN = re.compile(
    r"online/query_model_(mae|rmse|feasible_mae|feasible_rmse)_agent(\d+)$"
)
TRAJECTORY_JSON_PATTERN = re.compile(r"eval_epoch_(\d+)_thread_(\d+)\.json$")


def load_scalar(ea: EventAccumulator, tag: str):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_all_seeds(results_dir: str):
    logs_dir = os.path.join(results_dir, "logs")
    seed_dirs = sorted(
        [
            d
            for d in os.listdir(logs_dir)
            if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith("seed_")
        ]
    )

    if not seed_dirs:
        sys.exit(f"No seed directories found in {logs_dir}")

    all_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    trajectory_records = []
    tags_of_interest = {
        "online/eval_return",
        "online/eval_economic_return",
        "online/eval_total_return",
        "online/eval_total_economic_return",
        "online/random_baseline_return",
        "online/random_baseline_economic_return",
        "online/random_baseline_total_return",
        "online/random_baseline_total_economic_return",
        "online/eval_minus_random",
        "online/actor_loss",
        "online/critic_loss",
        "online/entropy",
        "online/confidence",
        "online/train_return",
        "online/train_economic_return",
        "online/train_total_return",
        "online/train_total_economic_return",
    }

    for sd in seed_dirs:
        path = os.path.join(logs_dir, sd)
        if EventAccumulator is not None:
            ea = EventAccumulator(path)
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            dynamic_agent_tags = {
                tag
                for tag in available
                if AGENT_TAG_PATTERN.match(tag) or ECON_AGENT_TAG_PATTERN.match(tag)
            }
            dynamic_query_tags = {
                tag for tag in available if QUERY_TAG_PATTERN.match(tag)
            }
            for tag in tags_of_interest | dynamic_agent_tags | dynamic_query_tags:
                if tag in available:
                    steps, values = load_scalar(ea, tag)
                    all_data[tag].append((steps, values))

        for trajectory_path in load_latest_eval_trajectory_paths(path):
            trajectory_records.append(
                {
                    "seed_dir": sd,
                    "json_path": trajectory_path,
                    "label": os.path.basename(trajectory_path),
                }
            )

    return all_data, seed_dirs, trajectory_records


def align_and_aggregate(series_list):
    if not series_list:
        return None, None, None

    ref_steps = series_list[0][0]
    for steps, _ in series_list[1:]:
        if len(steps) < len(ref_steps):
            ref_steps = steps

    aligned = []
    for _, values in series_list:
        aligned.append(values[: len(ref_steps)])

    aligned = np.array(aligned)
    return ref_steps, np.mean(aligned, axis=0), np.std(aligned, axis=0)


def plot_with_bands(ax, steps, mean, std, label, color):
    ax.plot(steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)


def plot_agent_returns(all_data, plots_dir):
    agent_tags = sorted(
        [tag for tag in all_data if AGENT_TAG_PATTERN.match(tag)],
        key=lambda tag: (
            AGENT_TAG_PATTERN.match(tag).group(1),
            int(AGENT_TAG_PATTERN.match(tag).group(2)),
        ),
    )
    if not agent_tags:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=False)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    for mode_idx, mode in enumerate(["train", "eval"]):
        ax = axes[mode_idx]
        found = False
        for color_idx, tag in enumerate(agent_tags):
            match = AGENT_TAG_PATTERN.match(tag)
            if match.group(1) != mode:
                continue
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is None:
                continue
            found = True
            agent_idx = int(match.group(2))
            plot_with_bands(
                ax,
                steps,
                mean,
                std,
                label=f"Agent {agent_idx}",
                color=colors[color_idx % len(colors)],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Return")
        ax.set_title(f"Per-Agent {mode.title()} Return")
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_agent_returns.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/per_agent_returns.png")


def plot_agent_economic_returns(all_data, plots_dir):
    agent_tags = sorted(
        [tag for tag in all_data if ECON_AGENT_TAG_PATTERN.match(tag)],
        key=lambda tag: (
            ECON_AGENT_TAG_PATTERN.match(tag).group(1),
            int(ECON_AGENT_TAG_PATTERN.match(tag).group(2)),
        ),
    )
    if not agent_tags:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=False)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    for mode_idx, mode in enumerate(["train", "eval"]):
        ax = axes[mode_idx]
        found = False
        for color_idx, tag in enumerate(agent_tags):
            match = ECON_AGENT_TAG_PATTERN.match(tag)
            if match.group(1) != mode:
                continue
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is None:
                continue
            found = True
            agent_idx = int(match.group(2))
            plot_with_bands(
                ax,
                steps,
                mean,
                std,
                label=f"Agent {agent_idx}",
                color=colors[color_idx % len(colors)],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Economic Return")
        ax.set_title(f"Per-Agent {mode.title()} Economic Return")
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_agent_economic_returns.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/per_agent_economic_returns.png")


def plot_query_model_quality(all_data, plots_dir):
    query_tags = sorted(
        [tag for tag in all_data if QUERY_TAG_PATTERN.match(tag)],
        key=lambda tag: (
            QUERY_TAG_PATTERN.match(tag).group(1),
            int(QUERY_TAG_PATTERN.match(tag).group(2)),
        ),
    )
    if not query_tags:
        return

    metrics = [
        ("mae", "All-Formula MAE"),
        ("rmse", "All-Formula RMSE"),
        ("feasible_mae", "Feasible-Only MAE"),
        ("feasible_rmse", "Feasible-Only RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    for ax, (metric_name, title) in zip(axes.flatten(), metrics):
        found = False
        for color_idx, tag in enumerate(query_tags):
            match = QUERY_TAG_PATTERN.match(tag)
            if match.group(1) != metric_name:
                continue
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is None:
                continue
            found = True
            agent_idx = int(match.group(2))
            plot_with_bands(
                ax,
                steps,
                mean,
                std,
                label=f"Agent {agent_idx}",
                color=colors[color_idx % len(colors)],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if found:
            ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "query_model_quality.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/query_model_quality.png")


def plot_mean_returns(all_data, plots_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    plot_specs = [
        (
            "online/train_return",
            "online/eval_return",
            "online/random_baseline_return",
            "Mean Shaped Return",
        ),
        (
            "online/train_economic_return",
            "online/eval_economic_return",
            "online/random_baseline_economic_return",
            "Mean Economic Return",
        ),
        (
            "online/train_total_return",
            "online/eval_total_return",
            "online/random_baseline_total_return",
            "Total Shaped Return",
        ),
        (
            "online/train_total_economic_return",
            "online/eval_total_economic_return",
            "online/random_baseline_total_economic_return",
            "Total Economic Return",
        ),
    ]

    for ax, (train_tag, eval_tag, random_tag, title) in zip(axes.flatten(), plot_specs):
        for tag, label, color in [
            (train_tag, "Train", "tab:blue"),
            (eval_tag, "Eval", "tab:orange"),
            (random_tag, "Random", "tab:red"),
        ]:
            if tag not in all_data:
                continue
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is None:
                continue
            plot_with_bands(ax, steps, mean, std, label, color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Return")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mean_returns.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/mean_returns.png")


def aggregate_sequence_series(series_list):
    if not series_list:
        return None, None, None
    min_len = min(len(series) for series in series_list)
    aligned = np.asarray([series[:min_len] for series in series_list], dtype=np.float64)
    steps = np.arange(min_len, dtype=np.int32)
    return steps, np.mean(aligned, axis=0), np.std(aligned, axis=0)


def aggregate_mean_std_series(mean_series_list, std_series_list):
    if not mean_series_list:
        return None, None, None
    min_len = min(len(series) for series in mean_series_list)
    aligned_means = np.asarray(
        [series[:min_len] for series in mean_series_list],
        dtype=np.float64,
    )
    if std_series_list:
        aligned_stds = np.asarray(
            [series[:min_len] for series in std_series_list],
            dtype=np.float64,
        )
    else:
        aligned_stds = np.zeros_like(aligned_means)
    steps = np.arange(min_len, dtype=np.int32)
    mean = np.mean(aligned_means, axis=0)
    second_moment = np.mean(aligned_stds**2 + aligned_means**2, axis=0)
    std = np.sqrt(np.maximum(second_moment - mean**2, 0.0))
    return steps, mean, std


def load_all_eval_trajectory_paths(seed_log_dir: str):
    by_epoch = load_eval_trajectory_paths_by_epoch(seed_log_dir)
    if not by_epoch:
        return []
    return sorted(p for paths in by_epoch.values() for p in paths)


def load_latest_eval_trajectory_paths(seed_log_dir: str):
    return load_all_eval_trajectory_paths(seed_log_dir)


def load_eval_trajectory_paths_by_epoch(seed_log_dir: str) -> dict[int, list[str]]:
    traj_dir = os.path.join(seed_log_dir, "trajectories")
    if not os.path.isdir(traj_dir):
        return {}

    by_epoch = defaultdict(list)
    for name in sorted(os.listdir(traj_dir)):
        match = TRAJECTORY_JSON_PATTERN.match(name)
        if not match:
            continue
        epoch = int(match.group(1))
        by_epoch[epoch].append(os.path.join(traj_dir, name))

    return {epoch: sorted(paths) for epoch, paths in by_epoch.items()}


def extract_public_resolved_counts(trajectory: dict) -> np.ndarray:
    counts = [len(trajectory["initial_state"]["public_library"]["resolved"])]
    counts.extend(
        len(step["state_after"]["public_library"]["resolved"])
        for step in trajectory.get("steps", [])
    )
    return np.asarray(counts, dtype=np.float64)


def select_highlight_epochs(epochs: list[int], max_highlights: int = 6) -> list[int]:
    if not epochs:
        return []
    if len(epochs) <= max_highlights:
        return list(epochs)

    positions = np.linspace(0, len(epochs) - 1, num=max_highlights)
    selected = []
    seen = set()
    for pos in positions:
        epoch = epochs[int(round(pos))]
        if epoch not in seen:
            selected.append(epoch)
            seen.add(epoch)
    return selected


def plot_public_resolved_vs_oracle(results_dir, seed_dirs, plots_dir):
    logs_dir = os.path.join(results_dir, "logs")
    realized_series_by_epoch = defaultdict(list)
    oracle_inputs_by_epoch = defaultdict(list)
    used_trajectories = 0
    skipped_oracle = []

    for seed_dir in seed_dirs:
        seed_log_dir = os.path.join(logs_dir, seed_dir)
        trajectory_paths_by_epoch = load_eval_trajectory_paths_by_epoch(seed_log_dir)
        if not trajectory_paths_by_epoch:
            continue

        cfg = None
        metadata_path = os.path.join(seed_log_dir, "run_metadata.json")
        if os.path.exists(metadata_path):
            metadata = load_run_metadata(metadata_path)
            cfg = deserialize_config(metadata["config"])
        else:
            skipped_oracle.append(seed_dir)

        for epoch, trajectory_paths in sorted(trajectory_paths_by_epoch.items()):
            for trajectory_path in trajectory_paths:
                with open(trajectory_path, "r", encoding="utf-8") as handle:
                    trajectory = json.load(handle)
                realized_series_by_epoch[epoch].append(
                    extract_public_resolved_counts(trajectory)
                )
                used_trajectories += 1
                if cfg is not None:
                    oracle_inputs_by_epoch[epoch].append(
                        (cfg, trajectory["initial_state"])
                    )

    if not realized_series_by_epoch:
        return None

    epochs = sorted(realized_series_by_epoch)
    latest_epoch = epochs[-1]
    observed_mean_by_epoch: dict[int, np.ndarray] = {}
    oracle_mean_by_epoch: dict[int, np.ndarray] = {}

    for epoch in epochs:
        real_steps, epoch_real_mean, _ = aggregate_sequence_series(
            realized_series_by_epoch[epoch]
        )
        if real_steps is None:
            continue
        observed_mean_by_epoch[epoch] = epoch_real_mean

        for cfg, initial_state in oracle_inputs_by_epoch.get(epoch, []):
            oracle = solve_public_resolution_oracle(cfg, initial_state)
            oracle_mean_by_epoch.setdefault(epoch, [])
            oracle_mean_by_epoch[epoch].append(
                np.asarray(oracle["expected_public_resolved_by_time"], dtype=np.float64)
            )

    oracle_available = False
    collapsed_oracle_mean_by_epoch: dict[int, np.ndarray] = {}
    for epoch, series_list in oracle_mean_by_epoch.items():
        oracle_steps, epoch_oracle_mean, _ = aggregate_mean_std_series(series_list, [])
        if oracle_steps is not None:
            oracle_available = True
            collapsed_oracle_mean_by_epoch[epoch] = epoch_oracle_mean

    plotted_epochs = sorted(observed_mean_by_epoch)
    max_len = max(
        max(len(series) for series in observed_mean_by_epoch.values()),
        max(
            (len(series) for series in collapsed_oracle_mean_by_epoch.values()),
            default=0,
        ),
    )

    timestep_grid, epoch_grid = np.meshgrid(
        np.arange(max_len, dtype=np.float64),
        np.asarray(plotted_epochs, dtype=np.float64),
    )
    observed_surface = np.full((len(plotted_epochs), max_len), np.nan, dtype=np.float64)
    oracle_surface = np.full((len(plotted_epochs), max_len), np.nan, dtype=np.float64)

    real_mean = observed_mean_by_epoch[latest_epoch]
    oracle_mean = None
    for row_idx, epoch in enumerate(plotted_epochs):
        observed_series = observed_mean_by_epoch[epoch]
        observed_surface[row_idx, : len(observed_series)] = observed_series
        oracle_series = collapsed_oracle_mean_by_epoch.get(epoch)
        if oracle_series is not None:
            oracle_surface[row_idx, : len(oracle_series)] = oracle_series
            if epoch == latest_epoch:
                oracle_mean = oracle_series

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    wire_rstride = max(1, len(plotted_epochs) // 10)
    wire_cstride = max(1, max_len // 18)

    ax.plot_surface(
        timestep_grid,
        epoch_grid,
        np.ma.masked_invalid(observed_surface),
        color="#2563eb",
        alpha=0.78,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.plot_wireframe(
        timestep_grid,
        epoch_grid,
        np.ma.masked_invalid(observed_surface),
        rstride=wire_rstride,
        cstride=wire_cstride,
        color="#1d4ed8",
        linewidth=0.5,
        alpha=0.28,
    )

    if oracle_available:
        ax.plot_surface(
            timestep_grid,
            epoch_grid,
            np.ma.masked_invalid(oracle_surface),
            color="#f59e0b",
            alpha=0.38,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.plot_wireframe(
            timestep_grid,
            epoch_grid,
            np.ma.masked_invalid(oracle_surface),
            rstride=wire_rstride,
            cstride=wire_cstride,
            color="#d97706",
            linewidth=0.45,
            alpha=0.18,
        )

    ax.set_xlabel("Timestep", labelpad=10)
    ax.set_ylabel("Eval Epoch", labelpad=12)
    ax.set_zlabel("Publicly Resolved Nodes", labelpad=10)
    ax.set_title("Public Resolution Surface Across Eval Epochs")
    ax.view_init(elev=24, azim=-128)
    ax.set_xlim(0, max_len - 1)
    ax.set_ylim(min(plotted_epochs), max(plotted_epochs))
    ax.set_zlim(bottom=0)
    ax.set_yticks(select_highlight_epochs(plotted_epochs, max_highlights=6))
    ax.xaxis.pane.set_alpha(0.06)
    ax.yaxis.pane.set_alpha(0.06)
    ax.zaxis.pane.set_alpha(0.06)
    ax.legend(
        handles=(
            [
                Patch(
                    facecolor="#2563eb",
                    edgecolor="#1d4ed8",
                    alpha=0.78,
                    label="Observed surface",
                ),
                Patch(
                    facecolor="#f59e0b",
                    edgecolor="#d97706",
                    alpha=0.38,
                    label="Oracle surface",
                ),
            ]
            if oracle_available
            else [
                Patch(
                    facecolor="#2563eb",
                    edgecolor="#1d4ed8",
                    alpha=0.78,
                    label="Observed surface",
                ),
            ]
        ),
        loc="upper left",
    )
    fig.tight_layout()
    out_path = os.path.join(plots_dir, "public_resolved_vs_oracle.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    return {
        "n_trajectories": int(used_trajectories),
        "n_epochs": int(len(epochs)),
        "latest_epoch": int(latest_epoch),
        "oracle_available": bool(oracle_available),
        "skipped_oracle_seeds": skipped_oracle,
        "observed_final_mean": float(real_mean[-1]),
        "oracle_final_mean": None if not oracle_available else float(oracle_mean[-1]),
    }


def _load_seed_config(seed_log_dir: str) -> Optional[object]:
    metadata_path = os.path.join(seed_log_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    metadata = load_run_metadata(metadata_path)
    return deserialize_config(metadata["config"])


def generate_trajectory_views(
    results_dir: str, trajectory_records: list[dict]
) -> list[dict]:
    generated = []
    logs_dir = os.path.join(results_dir, "logs")
    cfg_cache: dict[str, Optional[object]] = {}

    for record in trajectory_records:
        seed_dir = record["seed_dir"]
        if seed_dir not in cfg_cache:
            cfg_cache[seed_dir] = _load_seed_config(os.path.join(logs_dir, seed_dir))
        cfg = cfg_cache[seed_dir]

        json_path = record["json_path"]
        with open(json_path, "r", encoding="utf-8") as handle:
            trajectory = json.load(handle)

        oracle_series = None
        if cfg is not None:
            oracle = solve_public_resolution_oracle(cfg, trajectory["initial_state"])
            oracle_series = [
                float(v) for v in oracle.get("expected_public_resolved_by_time", [])
            ]

        json_stem = Path(json_path).stem
        html_path = os.path.join(
            os.path.dirname(json_path), f"{json_stem}_analysis.html"
        )
        write_trajectory_viewer_html(
            html_path,
            trajectory,
            cfg=cfg,
            oracle_series=oracle_series,
            seed_dir=seed_dir,
            label=f"Trajectory Viewer: {json_stem}",
        )
        generated.append(
            {
                "seed_dir": seed_dir,
                "json_path": json_path,
                "html_path": html_path,
                "label": os.path.basename(html_path),
            }
        )
        print(f"Saved {html_path}")

    return generated


def write_trajectory_index(results_dir, trajectory_artifacts):
    if not trajectory_artifacts:
        return

    out_path = os.path.join(results_dir, "trajectory_index.html")
    rows = "\n".join(
        f'<tr><td>{item["seed_dir"]}</td><td><a href="{os.path.relpath(item["html_path"], results_dir)}">{item["label"]}</a></td></tr>'
        for item in trajectory_artifacts
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Trajectory Artifact Index</title>
  <style>
    body {{ font-family: "IBM Plex Sans", sans-serif; margin: 24px; background: #f5f1e8; color: #1f1a14; }}
    table {{ border-collapse: collapse; width: 100%; background: #fffaf0; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #d6c7ad; text-align: left; }}
    a {{ color: #0f766e; }}
  </style>
</head>
<body>
  <h1>Trajectory Artifact Index</h1>
  <table>
    <thead><tr><th>Seed</th><th>Artifact</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/")
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_data, seed_dirs, trajectory_records = load_all_seeds(args.results_dir)
    n_seeds = len(seed_dirs)
    print(f"Loaded data from {n_seeds} seeds: {seed_dirs}")

    fig, ax = plt.subplots(figsize=(10, 5))
    if "online/eval_return" in all_data:
        steps, mean, std = align_and_aggregate(all_data["online/eval_return"])
        if steps is not None:
            plot_with_bands(ax, steps, mean, std, "Eval Return", "tab:blue")

    if "online/random_baseline_return" in all_data:
        r_steps, r_mean, _ = align_and_aggregate(
            all_data["online/random_baseline_return"]
        )
        if r_steps is not None:
            avg_random = np.mean(r_mean)
            ax.axhline(
                avg_random,
                color="tab:red",
                linestyle="--",
                label=f"Random baseline ({avg_random:.3f})",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return")
    ax.set_title("MAPPO Eval Return on VAMP")
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_return.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/eval_return.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (tag, title) in enumerate(
        [
            ("online/actor_loss", "Actor Loss"),
            ("online/critic_loss", "Critic Loss"),
        ]
    ):
        if tag in all_data:
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is not None:
                plot_with_bands(
                    axes[i],
                    steps,
                    mean,
                    std,
                    title,
                    "tab:orange" if i == 0 else "tab:green",
                )
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)
        if axes[i].lines:
            axes[i].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "losses.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/losses.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    if "online/entropy" in all_data:
        steps, mean, std = align_and_aggregate(all_data["online/entropy"])
        if steps is not None:
            plot_with_bands(ax, steps, mean, std, "Entropy", "tab:purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Policy Entropy over Training")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "entropy.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/entropy.png")

    plot_agent_returns(all_data, plots_dir)
    plot_agent_economic_returns(all_data, plots_dir)
    plot_query_model_quality(all_data, plots_dir)
    plot_mean_returns(all_data, plots_dir)
    public_resolution_summary = plot_public_resolved_vs_oracle(
        args.results_dir, seed_dirs, plots_dir
    )
    trajectory_artifacts = generate_trajectory_views(
        args.results_dir, trajectory_records
    )
    write_trajectory_index(args.results_dir, trajectory_artifacts)

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("MAPPO Training Analysis Summary")
    summary_lines.append(f"Seeds: {seed_dirs}")
    summary_lines.append("=" * 60)

    verdict = "INCONCLUSIVE"

    if "online/eval_return" in all_data:
        steps, mean, std = align_and_aggregate(all_data["online/eval_return"])
        if steps is not None and len(mean) >= 4:
            n = len(mean)
            q1 = n // 4
            first_q = np.mean(mean[:q1]) if q1 > 0 else mean[0]
            last_q = np.mean(mean[-q1:]) if q1 > 0 else mean[-1]
            improvement = last_q - first_q

            summary_lines.append("\nEval Return:")
            summary_lines.append(f"  First-quarter avg:  {first_q:.4f}")
            summary_lines.append(f"  Last-quarter avg:   {last_q:.4f}")
            summary_lines.append(f"  Improvement:        {improvement:+.4f}")
            summary_lines.append(f"  Best (any seed):    {np.max(mean):.4f}")

            x = np.arange(len(mean), dtype=float)
            slope = np.polyfit(x, mean, 1)[0]
            summary_lines.append(f"  Linear trend slope: {slope:.6f}")

            if improvement > 0.5 and slope > 0:
                verdict = "LEARNING"
            elif improvement <= 0.1 and slope <= 0:
                verdict = "NOT LEARNING"

    for tag_prefix, label in [
        ("online/train_return_agent", "Train"),
        ("online/eval_return_agent", "Eval"),
    ]:
        matched_tags = sorted(
            [tag for tag in all_data if tag.startswith(tag_prefix)],
            key=lambda tag: int(tag.rsplit("agent", 1)[1]),
        )
        if matched_tags:
            summary_lines.append(f"\n{label} Per-Agent Final Returns:")
            for tag in matched_tags:
                _, mean, _ = align_and_aggregate(all_data[tag])
                if mean is not None:
                    agent_idx = int(tag.rsplit("agent", 1)[1])
                    summary_lines.append(f"  Agent {agent_idx}: {mean[-1]:.4f}")

    for tag, label in [
        ("online/train_return", "Train Mean Shaped Return"),
        ("online/eval_return", "Eval Mean Shaped Return"),
        ("online/train_economic_return", "Train Mean Economic Return"),
        ("online/eval_economic_return", "Eval Mean Economic Return"),
        ("online/train_total_return", "Train Total Shaped Return"),
        ("online/eval_total_return", "Eval Total Shaped Return"),
        ("online/train_total_economic_return", "Train Total Economic Return"),
        ("online/eval_total_economic_return", "Eval Total Economic Return"),
    ]:
        if tag in all_data:
            _, mean, _ = align_and_aggregate(all_data[tag])
            if mean is not None:
                summary_lines.append(f"\n{label}:")
                summary_lines.append(f"  Final: {mean[-1]:.4f}")

    for tag_prefix, label in [
        ("online/train_economic_return_agent", "Train"),
        ("online/eval_economic_return_agent", "Eval"),
    ]:
        matched_tags = sorted(
            [tag for tag in all_data if tag.startswith(tag_prefix)],
            key=lambda tag: int(tag.rsplit("agent", 1)[1]),
        )
        if matched_tags:
            summary_lines.append(f"\n{label} Per-Agent Final Economic Returns:")
            for tag in matched_tags:
                _, mean, _ = align_and_aggregate(all_data[tag])
                if mean is not None:
                    agent_idx = int(tag.rsplit("agent", 1)[1])
                    summary_lines.append(f"  Agent {agent_idx}: {mean[-1]:.4f}")

    query_metric_labels = [
        ("online/query_model_mae_agent", "Query Model Final MAE"),
        ("online/query_model_rmse_agent", "Query Model Final RMSE"),
        ("online/query_model_feasible_mae_agent", "Query Model Final Feasible MAE"),
        ("online/query_model_feasible_rmse_agent", "Query Model Final Feasible RMSE"),
    ]
    for tag_prefix, label in query_metric_labels:
        matched_tags = sorted(
            [tag for tag in all_data if tag.startswith(tag_prefix)],
            key=lambda tag: int(tag.rsplit("agent", 1)[1]),
        )
        if matched_tags:
            summary_lines.append(f"\n{label}:")
            for tag in matched_tags:
                _, mean, _ = align_and_aggregate(all_data[tag])
                if mean is not None:
                    agent_idx = int(tag.rsplit("agent", 1)[1])
                    summary_lines.append(f"  Agent {agent_idx}: {mean[-1]:.4f}")

    if "online/random_baseline_return" in all_data:
        _, r_mean, _ = align_and_aggregate(all_data["online/random_baseline_return"])
        if r_mean is not None:
            summary_lines.append("\nRandom Baseline:")
            summary_lines.append(f"  Avg random return:  {np.mean(r_mean):.4f}")

    if "online/eval_minus_random" in all_data:
        _, gap_mean, _ = align_and_aggregate(all_data["online/eval_minus_random"])
        if gap_mean is not None:
            final_gap = np.mean(gap_mean[-max(1, len(gap_mean) // 4) :])
            summary_lines.append(
                f"  Final-quarter gap (eval - random): {final_gap:+.4f}"
            )
            if final_gap > 0.5 and verdict != "LEARNING":
                verdict = "LEARNING"

    if trajectory_artifacts:
        summary_lines.append("\nTrajectory Artifacts:")
        summary_lines.append(f"  HTML files indexed: {len(trajectory_artifacts)}")
        summary_lines.append("  Open results/trajectory_index.html to browse them")

    if public_resolution_summary is not None:
        summary_lines.append("\nPublic Resolution Trajectory:")
        summary_lines.append(
            f"  Trajectories used: {public_resolution_summary['n_trajectories']}"
        )
        summary_lines.append(
            f"  Eval epochs plotted: {public_resolution_summary['n_epochs']} (latest={public_resolution_summary['latest_epoch']})"
        )
        summary_lines.append(
            f"  Observed final public resolved: {public_resolution_summary['observed_final_mean']:.4f}"
        )
        if public_resolution_summary["oracle_available"]:
            summary_lines.append(
                f"  Oracle final public resolved: {public_resolution_summary['oracle_final_mean']:.4f}"
            )
        if public_resolution_summary["skipped_oracle_seeds"]:
            summary_lines.append(
                "  Oracle skipped for seeds without metadata: "
                + ", ".join(public_resolution_summary["skipped_oracle_seeds"])
            )

    summary_lines.append(f"\n{'=' * 60}")
    summary_lines.append(f"VERDICT: {verdict}")
    summary_lines.append(f"{'=' * 60}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(args.results_dir, "analysis_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
