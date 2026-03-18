"""Generate offline data for MADT pretraining.

Produces per-episode .pt files compatible with ReplayBuffer.load_offline_data().
Each file contains:
    episode = [  # n_agents lists
        [  # per step
            [global_obs_list, local_obs_list, [action], [reward], done_bool, avail_list, [v_value]],
            ...
        ],
        ...
    ]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.vamp.config import VampConfig
from envs.env import make_vamp_env


def parse_args():
    parser = argparse.ArgumentParser(description='Generate offline VAMP data')
    parser.add_argument('--num_theorems', type=int, default=4)
    parser.add_argument('--F_size', type=int, default=None)
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--max_timestep', type=int, default=100)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--policy', type=str, default='random',
                        choices=['random', 'prove_first', 'mixed'])
    parser.add_argument('--output_dir', type=str, default='data/offline/')
    return parser.parse_args()


def random_policy(avail: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.where(avail > 0)[0]
    return int(rng.choice(valid)) if len(valid) > 0 else 0


def prove_first_policy(avail: np.ndarray, encoder, rng: np.random.Generator) -> int:
    prove_mask = np.zeros_like(avail)
    prove_mask[encoder._prove_start:encoder._prove_end] = avail[encoder._prove_start:encoder._prove_end]
    valid = np.where(prove_mask > 0)[0]
    if len(valid) > 0:
        return int(rng.choice(valid))
    conj_mask = np.zeros_like(avail)
    conj_mask[encoder._conj_start:encoder._conj_end] = avail[encoder._conj_start:encoder._conj_end]
    valid = np.where(conj_mask > 0)[0]
    if len(valid) > 0:
        return int(rng.choice(valid))
    return random_policy(avail, rng)


def mixed_policy(avail: np.ndarray, encoder, rng: np.random.Generator) -> int:
    if rng.random() < 0.5:
        return prove_first_policy(avail, encoder, rng)
    return random_policy(avail, rng)


def main():
    args = parse_args()

    cfg = VampConfig(
        num_theorems=args.num_theorems,
        F_size=args.F_size,
        n_agents=args.n_agents,
        max_timestep=args.max_timestep,
    )
    env = make_vamp_env(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    encoder = env.encoder
    n_agents = cfg.n_agents

    policy_fn = {
        'random': lambda avail, _rng: random_policy(avail, _rng),
        'prove_first': lambda avail, _rng: prove_first_policy(avail, encoder, _rng),
        'mixed': lambda avail, _rng: mixed_policy(avail, encoder, _rng),
    }[args.policy]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.n_episodes} episodes with {args.policy} policy")
    print(f"n={cfg.num_theorems}, F={cfg.F_size}, N={args.n_agents}, T={args.max_timestep}")

    for ep in range(args.n_episodes):
        obs, share_obs, avail = env.reset()
        # obs shape: (n_agents, local_obs_dim)
        # share_obs shape: (n_agents, global_obs_dim)
        # avail shape: (n_agents, action_dim)

        # Per-agent step lists
        agent_trajectories = [[] for _ in range(n_agents)]
        done = False

        while not done:
            actions = np.zeros(n_agents, dtype=np.int64)
            for i in range(n_agents):
                actions[i] = policy_fn(avail[i], rng)

            next_obs, next_share_obs, rewards, dones, infos, next_avail = env.step(actions)
            # rewards shape: (n_agents, 1), dones shape: (n_agents, 1)

            for i in range(n_agents):
                step = [
                    share_obs[i].tolist(),       # global obs
                    obs[i].tolist(),             # local obs
                    [int(actions[i])],           # action
                    [float(rewards[i, 0])],      # reward
                    bool(dones[i, 0]),           # done
                    avail[i].tolist(),           # available actions
                    [0.0],                       # v_value placeholder
                ]
                agent_trajectories[i].append(step)

            done = bool(dones[0, 0])
            obs, share_obs, avail = next_obs, next_share_obs, next_avail

        torch.save(agent_trajectories, os.path.join(args.output_dir, f"{ep}.pt"))

        if (ep + 1) % 10 == 0:
            print(f"Generated {ep + 1}/{args.n_episodes} episodes")

    print(f"Saved {args.n_episodes} episodes to {args.output_dir}")


if __name__ == '__main__':
    main()
