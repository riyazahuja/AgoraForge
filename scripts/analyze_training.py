#!/usr/bin/env python
"""Analyze MAPPO training results from TensorBoard event files.

Produces plots, a trajectory index, and a text summary.

Usage:
    python scripts/analyze_training.py --results_dir results/
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: uv pip install matplotlib")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard is required: uv pip install tensorboard")


AGENT_TAG_PATTERN = re.compile(r"online/(train|eval)_return_agent(\d+)$")


def load_scalar(ea: EventAccumulator, tag: str):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_all_seeds(results_dir: str):
    logs_dir = os.path.join(results_dir, "logs")
    seed_dirs = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith("seed_")
    ])

    if not seed_dirs:
        sys.exit(f"No seed directories found in {logs_dir}")

    all_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    trajectory_artifacts = []
    tags_of_interest = {
        "online/eval_return",
        "online/random_baseline_return",
        "online/eval_minus_random",
        "online/actor_loss",
        "online/critic_loss",
        "online/entropy",
        "online/confidence",
        "online/train_return",
    }

    for sd in seed_dirs:
        path = os.path.join(logs_dir, sd)
        ea = EventAccumulator(path)
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        dynamic_agent_tags = {tag for tag in available if AGENT_TAG_PATTERN.match(tag)}
        for tag in tags_of_interest | dynamic_agent_tags:
            if tag in available:
                steps, values = load_scalar(ea, tag)
                all_data[tag].append((steps, values))

        traj_dir = os.path.join(path, "trajectories")
        if os.path.isdir(traj_dir):
            for name in sorted(os.listdir(traj_dir)):
                if name.endswith(".html"):
                    trajectory_artifacts.append({
                        'seed_dir': sd,
                        'html_path': os.path.join(traj_dir, name),
                        'label': name,
                    })

    return all_data, seed_dirs, trajectory_artifacts


def align_and_aggregate(series_list):
    if not series_list:
        return None, None, None

    ref_steps = series_list[0][0]
    for steps, _ in series_list[1:]:
        if len(steps) < len(ref_steps):
            ref_steps = steps

    aligned = []
    for _, values in series_list:
        aligned.append(values[:len(ref_steps)])

    aligned = np.array(aligned)
    return ref_steps, np.mean(aligned, axis=0), np.std(aligned, axis=0)


def plot_with_bands(ax, steps, mean, std, label, color):
    ax.plot(steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)


def plot_agent_returns(all_data, plots_dir):
    agent_tags = sorted(
        [tag for tag in all_data if AGENT_TAG_PATTERN.match(tag)],
        key=lambda tag: (AGENT_TAG_PATTERN.match(tag).group(1), int(AGENT_TAG_PATTERN.match(tag).group(2))),
    )
    if not agent_tags:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=False)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

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

    all_data, seed_dirs, trajectory_artifacts = load_all_seeds(args.results_dir)
    n_seeds = len(seed_dirs)
    print(f"Loaded data from {n_seeds} seeds: {seed_dirs}")

    fig, ax = plt.subplots(figsize=(10, 5))
    if "online/eval_return" in all_data:
        steps, mean, std = align_and_aggregate(all_data["online/eval_return"])
        if steps is not None:
            plot_with_bands(ax, steps, mean, std, "Eval Return", "tab:blue")

    if "online/random_baseline_return" in all_data:
        r_steps, r_mean, _ = align_and_aggregate(all_data["online/random_baseline_return"])
        if r_steps is not None:
            avg_random = np.mean(r_mean)
            ax.axhline(avg_random, color="tab:red", linestyle="--", label=f"Random baseline ({avg_random:.3f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return")
    ax.set_title("MAPPO Eval Return on VAMP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "eval_return.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plots_dir}/eval_return.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (tag, title) in enumerate([
        ("online/actor_loss", "Actor Loss"),
        ("online/critic_loss", "Critic Loss"),
    ]):
        if tag in all_data:
            steps, mean, std = align_and_aggregate(all_data[tag])
            if steps is not None:
                plot_with_bands(axes[i], steps, mean, std, title, "tab:orange" if i == 0 else "tab:green")
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

    for tag_prefix, label in [("online/train_return_agent", "Train"), ("online/eval_return_agent", "Eval")]:
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

    if "online/random_baseline_return" in all_data:
        _, r_mean, _ = align_and_aggregate(all_data["online/random_baseline_return"])
        if r_mean is not None:
            summary_lines.append("\nRandom Baseline:")
            summary_lines.append(f"  Avg random return:  {np.mean(r_mean):.4f}")

    if "online/eval_minus_random" in all_data:
        _, gap_mean, _ = align_and_aggregate(all_data["online/eval_minus_random"])
        if gap_mean is not None:
            final_gap = np.mean(gap_mean[-max(1, len(gap_mean)//4):])
            summary_lines.append(f"  Final-quarter gap (eval - random): {final_gap:+.4f}")
            if final_gap > 0.5 and verdict != "LEARNING":
                verdict = "LEARNING"

    if trajectory_artifacts:
        summary_lines.append("\nTrajectory Artifacts:")
        summary_lines.append(f"  HTML files indexed: {len(trajectory_artifacts)}")
        summary_lines.append("  Open results/trajectory_index.html to browse them")

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
