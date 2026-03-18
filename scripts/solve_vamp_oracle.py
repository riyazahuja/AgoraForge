#!/usr/bin/env python
"""Solve the reduced full-information VAMP oracle from a captured trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.vamp.metadata import deserialize_config, load_run_metadata
from envs.vamp.oracle_solver import solve_public_resolution_oracle


def parse_args():
    parser = argparse.ArgumentParser(description="Solve the reduced VAMP oracle from a trajectory snapshot")
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to run_metadata.json produced by training",
    )
    parser.add_argument(
        "--trajectory_json",
        type=str,
        required=True,
        help="Path to a captured eval trajectory JSON",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to write the oracle result as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = load_run_metadata(args.metadata_path)
    cfg = deserialize_config(metadata["config"])
    trajectory = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    result = solve_public_resolution_oracle(cfg, trajectory["initial_state"])

    text = json.dumps(result, indent=2)
    print(text)

    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
