"""Helpers for persisting and restoring VAMP run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from envs.vamp.config import VampConfig


def serialize_config(cfg: VampConfig) -> Dict[str, Any]:
    dependency_adj = {
        str(int(theorem_id)): sorted(int(dep) for dep in deps)
        for theorem_id, deps in (cfg.dependency_adj or {}).items()
    }
    utility_weights = [
        {"src": int(src), "dst": int(dst), "weight": float(weight)}
        for (src, dst), weight in sorted((cfg.utility_weights or {}).items())
    ]
    initial_resolved = {
        str(int(phi)): {
            "deps": sorted(int(dep) for dep in deps),
            "solve_time": int(solve_time),
            "solver": int(solver),
        }
        for phi, (deps, solve_time, solver) in (cfg.initial_resolved or {}).items()
    }
    return {
        "num_theorems": int(cfg.num_theorems),
        "F_size": int(cfg.F_size),
        "n_agents": int(cfg.n_agents),
        "truth_map": None if cfg.truth_map is None else [int(v) for v in cfg.truth_map.tolist()],
        "difficulty_map": None if cfg.difficulty_map is None else [float(v) for v in cfg.difficulty_map.tolist()],
        "dependency_adj": dependency_adj,
        "utility_weights": utility_weights,
        "initial_concrete": None if cfg.initial_concrete is None else sorted(int(v) for v in cfg.initial_concrete),
        "initial_resolved": initial_resolved,
        "initial_public_concrete_prob": float(cfg.initial_public_concrete_prob),
        "initial_cash": float(cfg.initial_cash),
        "gamma": float(cfg.gamma),
        "max_timestep": int(cfg.max_timestep),
        "kappa": float(cfg.kappa),
        "lambda_diff": float(cfg.lambda_diff),
        "alpha_util": float(cfg.alpha_util),
        "rho_0": [float(v) for v in cfg.rho_0.tolist()],
        "rho_1": [float(v) for v in cfg.rho_1.tolist()],
        "beta_conj": float(cfg.beta_conj),
        "eta_0": [float(v) for v in cfg.eta_0.tolist()],
        "eta_1": [float(v) for v in cfg.eta_1.tolist()],
        "kappa_conj_0": [float(v) for v in cfg.kappa_conj_0.tolist()],
        "kappa_conj_1": [float(v) for v in cfg.kappa_conj_1.tolist()],
        "phi_transform": str(cfg.phi_transform),
        "n_buckets": int(cfg.n_buckets),
        "horizon_H": int(cfg.horizon_H),
        "prior_a": float(cfg.prior_a),
        "prior_c": float(cfg.prior_c),
        "query_init_weight_std": float(cfg.query_init_weight_std),
        "query_global_lr": float(cfg.query_global_lr),
        "query_local_lr": float(cfg.query_local_lr),
        "query_private_truth_boost": float(cfg.query_private_truth_boost),
        "query_public_truth_boost": float(cfg.query_public_truth_boost),
        "budget_levels": [int(v) for v in cfg.budget_levels],
        "deadline_levels": [int(v) for v in cfg.deadline_levels],
        "loss_levels": [float(v) for v in cfg.loss_levels],
        "price_levels": [float(v) for v in cfg.price_levels],
        "max_offers": int(cfg.max_offers),
        "max_own_offers": int(cfg.max_own_offers),
        "operation_gas_fee": float(cfg.operation_gas_fee),
        "publish_resolution_bonus": float(cfg.publish_resolution_bonus),
        "bounty_quantity": int(cfg.bounty_quantity),
        "h_max": int(cfg.h_max),
    }


def _normalize_legacy_instance_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("num_theorems") is not None:
        return data

    legacy_f_size = int(data["F_size"])
    num_theorems = legacy_f_size // 2
    truth_map = np.asarray(data["truth_map"], dtype=np.int32)
    difficulty_map = np.asarray(data["difficulty_map"], dtype=np.float64)

    theorem_truth_map = np.zeros(num_theorems, dtype=np.int32)
    theorem_difficulty_map = np.zeros(num_theorems, dtype=np.float64)
    for theorem_id in range(num_theorems):
        lower_phi = theorem_id
        upper_phi = theorem_id + num_theorems
        theorem_truth_map[theorem_id] = 1 if truth_map[upper_phi] == 1 else 0
        theorem_difficulty_map[theorem_id] = max(
            float(difficulty_map[lower_phi]),
            float(difficulty_map[upper_phi]),
        )

    dependency_adj = {}
    for phi, deps in (data.get("dependency_adj") or {}).items():
        phi_int = int(phi)
        if phi_int >= num_theorems and truth_map[phi_int] == 1:
            dependency_adj[str(phi_int - num_theorems)] = sorted(int(dep) % num_theorems for dep in deps)
        elif phi_int < num_theorems and truth_map[phi_int] == 1:
            dependency_adj[str(phi_int)] = sorted(int(dep) % num_theorems for dep in deps)

    utility_weights = []
    for item in data.get("utility_weights") or []:
        src = int(item["src"])
        dst = int(item["dst"])
        if truth_map[src] == 1 and truth_map[dst] == 1:
            utility_weights.append(
                {
                    "src": src % num_theorems,
                    "dst": dst % num_theorems,
                    "weight": float(item["weight"]),
                }
            )

    normalized = dict(data)
    normalized["num_theorems"] = num_theorems
    normalized["truth_map"] = theorem_truth_map.tolist()
    normalized["difficulty_map"] = theorem_difficulty_map.tolist()
    normalized["dependency_adj"] = dependency_adj
    normalized["utility_weights"] = utility_weights
    return normalized


def deserialize_config(data: Dict[str, Any]) -> VampConfig:
    data = _normalize_legacy_instance_data(data)
    dependency_adj = {
        int(theorem_id): set(int(dep) for dep in deps)
        for theorem_id, deps in (data.get("dependency_adj") or {}).items()
    }
    utility_weights = {
        (int(item["src"]), int(item["dst"])): float(item["weight"])
        for item in data.get("utility_weights") or []
    }
    initial_resolved = {
        int(phi): (
            set(int(dep) for dep in info["deps"]),
            int(info["solve_time"]),
            int(info["solver"]),
        )
        for phi, info in (data.get("initial_resolved") or {}).items()
    }
    return VampConfig(
        num_theorems=int(data["num_theorems"]),
        F_size=int(data.get("F_size", 2 * int(data["num_theorems"]))),
        n_agents=int(data["n_agents"]),
        truth_map=None if data.get("truth_map") is None else np.asarray(data["truth_map"], dtype=np.int32),
        difficulty_map=None if data.get("difficulty_map") is None else np.asarray(data["difficulty_map"], dtype=np.float64),
        dependency_adj=dependency_adj or None,
        utility_weights=utility_weights or None,
        initial_concrete=None if data.get("initial_concrete") is None else set(int(v) for v in data["initial_concrete"]),
        initial_resolved=initial_resolved or None,
        initial_public_concrete_prob=float(data["initial_public_concrete_prob"]),
        initial_cash=float(data["initial_cash"]),
        gamma=float(data["gamma"]),
        max_timestep=int(data["max_timestep"]),
        kappa=float(data["kappa"]),
        lambda_diff=float(data["lambda_diff"]),
        alpha_util=float(data["alpha_util"]),
        rho_0=np.asarray(data["rho_0"], dtype=np.float64),
        rho_1=np.asarray(data["rho_1"], dtype=np.float64),
        beta_conj=float(data["beta_conj"]),
        eta_0=np.asarray(data["eta_0"], dtype=np.float64),
        eta_1=np.asarray(data["eta_1"], dtype=np.float64),
        kappa_conj_0=np.asarray(data["kappa_conj_0"], dtype=np.float64),
        kappa_conj_1=np.asarray(data["kappa_conj_1"], dtype=np.float64),
        phi_transform=str(data["phi_transform"]),
        n_buckets=int(data["n_buckets"]),
        horizon_H=int(data["horizon_H"]),
        prior_a=float(data["prior_a"]),
        prior_c=float(data["prior_c"]),
        query_init_weight_std=float(data["query_init_weight_std"]),
        query_global_lr=float(data["query_global_lr"]),
        query_local_lr=float(data["query_local_lr"]),
        query_private_truth_boost=float(data["query_private_truth_boost"]),
        query_public_truth_boost=float(data["query_public_truth_boost"]),
        budget_levels=[int(v) for v in data["budget_levels"]],
        deadline_levels=[int(v) for v in data["deadline_levels"]],
        loss_levels=[float(v) for v in data["loss_levels"]],
        price_levels=[float(v) for v in data["price_levels"]],
        max_offers=int(data["max_offers"]),
        max_own_offers=int(data["max_own_offers"]),
        operation_gas_fee=float(data["operation_gas_fee"]),
        publish_resolution_bonus=float(data["publish_resolution_bonus"]),
        bounty_quantity=int(data.get("bounty_quantity", 100)),
        h_max=int(data["h_max"]),
    )


def write_run_metadata(
    output_path: str | Path,
    *,
    args: Any,
    cfg: VampConfig,
    eval_seed_base: int,
    random_eval_seed_base: int,
    train_seed_base: int,
) -> None:
    payload = {
        "format_version": 2,
        "args": {key: value for key, value in vars(args).items()},
        "config": serialize_config(cfg),
        "seeds": {
            "base_seed": int(args.seed),
            "eval_seed_base": int(eval_seed_base),
            "random_eval_seed_base": int(random_eval_seed_base),
            "train_seed_base": int(train_seed_base),
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_metadata(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
