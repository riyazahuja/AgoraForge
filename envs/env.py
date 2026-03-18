"""Environment wrapper: instantiates VampEnv with gym-like spaces."""

from __future__ import annotations

from typing import Optional

import numpy as np

from envs.vamp.config import VampConfig
from envs.vamp.vamp_env import VampEnv
from envs.vamp.formula_graph import FormulaGraph


def make_vamp_env(cfg: Optional[VampConfig] = None, seed: int = 0) -> VampEnv:
    """Create a VampEnv instance.

    Args:
        cfg: VampConfig. If None, uses defaults (random graph will be generated).
        seed: random seed.

    Returns:
        VampEnv instance ready for reset() and step().
    """
    if cfg is None:
        cfg = VampConfig()
    return VampEnv(cfg, seed=seed)


def _ensure_graph_populated(cfg: VampConfig, seed: int = 0) -> None:
    """Pre-populate config with a random graph if truth_map is None.

    This avoids a race condition where subprocess envs fork before the
    first env generates and mutates the shared config.
    """
    if cfg.truth_map is None:
        rng = np.random.default_rng(seed)
        graph = FormulaGraph.random(cfg.num_theorems, rng=rng)
        cfg.truth_map = graph.theorem_truth_map
        cfg.difficulty_map = graph.theorem_difficulty_map
        cfg.dependency_adj = graph.theorem_dependency_adj
        cfg.utility_weights = graph.theorem_utility_weights


def make_parallel_env(n_envs: int, cfg: Optional[VampConfig] = None, seed: int = 0):
    """Create parallel vectorized VampEnv instances.

    Uses ShareSubprocVecEnv from env_wrappers for multiprocessing.

    Args:
        n_envs: number of parallel environments
        cfg: VampConfig (shared across all envs)
        seed: base seed (each env gets seed + i)

    Returns:
        ShareSubprocVecEnv instance
    """
    from envs.env_wrappers import ShareSubprocVecEnv

    if cfg is None:
        cfg = VampConfig()

    # Pre-populate graph to avoid race condition in subprocesses
    _ensure_graph_populated(cfg, seed)

    def make_env_fn(env_seed):
        def _init():
            return make_vamp_env(cfg, seed=env_seed)
        return _init

    env_fns = [make_env_fn(seed + i) for i in range(n_envs)]
    return ShareSubprocVecEnv(env_fns)


class VampEnvWrapper:
    """Wrapper providing the interface RolloutWorker expects:
    env.real_env, env.n_threads, env.num_agents, env.max_timestep.
    """

    def __init__(self, n_threads: int, cfg: Optional[VampConfig] = None, seed: int = 0):
        if cfg is None:
            cfg = VampConfig()
        # Pre-populate graph before forking subprocesses
        _ensure_graph_populated(cfg, seed)
        self.real_env = make_parallel_env(n_threads, cfg, seed)
        self.num_agents = cfg.n_agents
        self.max_timestep = cfg.max_timestep
        self.n_threads = n_threads

    def close(self):
        self.real_env.close()
