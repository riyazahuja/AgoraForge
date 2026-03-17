"""Main entry point: parse args -> build config -> create envs -> train MADT on VAMP.

Supports optional offline pretraining (if data dir provided) followed by
MAPPO RL self-play (always).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from envs.vamp.config import VampConfig
from envs.env import make_vamp_env, VampEnvWrapper
from models.gpt_model import GPT, GPTConfig
from framework.buffer import ReplayBuffer, StateActionReturnDataset
from framework.rollout import RolloutWorker
from framework.trainer import Trainer, TrainerConfig
from framework.trajectory_logging import summarize_query_model_diagnostics, write_trajectory_artifacts
from framework.utils import set_seed, unwrap_model, padding_ava


class NullSummaryWriter:

    def add_scalar(self, *args, **kwargs):
        del args, kwargs

    def add_text(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


def random_rollout(env, n_agents, action_dim):
    """Run episodes with uniform-random actions (within available action masks).

    Returns average return across all env threads (same metric as RolloutWorker.rollout).
    """
    _, _, available_actions = env.real_env.reset()
    available_actions = padding_ava(available_actions, action_dim)

    T_rewards = 0.0
    T_economic_rewards = 0.0
    T_shaping_rewards = 0.0
    episode_dones = np.zeros(env.n_threads)

    while True:
        action = np.zeros((env.n_threads, n_agents, 1), dtype=np.int64)
        for n in range(env.n_threads):
            for a in range(n_agents):
                avail = available_actions[n, a]
                valid_ids = np.where(avail == 1)[0]
                if len(valid_ids) == 0:
                    valid_ids = np.arange(avail.shape[0])
                action[n, a, 0] = np.random.choice(valid_ids)

        _, _, rewards, dones, infos, available_actions = env.real_env.step(action)
        available_actions = padding_ava(available_actions, action_dim)

        for n in range(env.n_threads):
            if not episode_dones[n]:
                T_rewards += np.mean(rewards[n])
                econ_step = 0.0
                shape_step = 0.0
                for a in range(n_agents):
                    econ_step += float(infos[n][a].get("economic_reward", rewards[n, a, 0]))
                    shape_step += float(infos[n][a].get("shaping_reward", 0.0))
                T_economic_rewards += econ_step / n_agents
                T_shaping_rewards += shape_step / n_agents
                if np.all(dones[n]):
                    episode_dones[n] = 1

        if np.all(episode_dones):
            break

    return (
        T_rewards / env.n_threads,
        T_economic_rewards / env.n_threads,
        T_shaping_rewards / env.n_threads,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='MADT training on VAMP environment')

    # Environment
    parser.add_argument('--F_size', type=int, default=8, help='Formula universe size')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--max_timestep', type=int, default=100, help='Episode length')
    parser.add_argument('--initial_cash', type=float, default=10.0, help='Starting cash per agent')
    parser.add_argument('--initial_public_concrete_prob', type=float, default=0.25,
                        help='Probability of seeding each negation pair as public concrete at reset')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Kernels
    parser.add_argument('--kappa', type=float, default=0.5, help='Budget concavity')
    parser.add_argument('--lambda_diff', type=float, default=1.0, help='Difficulty transform')
    parser.add_argument('--alpha_util', type=float, default=1.0, help='Utility sensitivity')
    parser.add_argument('--beta_conj', type=float, default=1.0, help='Conjecture mass exponent')

    # Query model
    parser.add_argument('--n_buckets', type=int, default=4, help='Query model buckets')
    parser.add_argument('--horizon_H', type=int, default=20, help='Query model horizon')
    parser.add_argument('--query_init_weight_std', type=float, default=2.0,
                        help='Stddev of the initial global feature weights for the query model')
    parser.add_argument('--query_global_lr', type=float, default=0.03,
                        help='Global feature-model learning rate for the query model')
    parser.add_argument('--query_local_lr', type=float, default=0.15,
                        help='Per-formula residual learning rate for the query model')
    parser.add_argument('--query_private_truth_boost', type=float, default=2.0,
                        help='Truth-belief boost after a private proof succeeds')
    parser.add_argument('--query_public_truth_boost', type=float, default=2.5,
                        help='Truth-belief boost after a public proof is observed')
    parser.add_argument('--operation_gas_fee', type=float, default=0.0,
                        help='Optional per-action gas fee applied to any non-noop action')
    parser.add_argument('--publish_resolution_bonus', type=float, default=0.0,
                        help='Optional zero-sum shaping bonus per newly public resolution')
    parser.add_argument('--target_init_prob', type=float, default=0.5,
                        help='Probability of seeding a public concrete theorem with a static bounty offer')
    parser.add_argument('--target_init_min_price', type=float, default=0.1,
                        help='Minimum offer price for static bounty initialization')
    parser.add_argument('--target_init_max_price', type=float, default=0.3,
                        help='Maximum offer price for static bounty initialization')
    parser.add_argument('--target_init_max_quantity', type=int, default=4,
                        help='Maximum quantity for static bounty offers at reset')
    parser.add_argument('--target_init_cash', type=float, default=100.0,
                        help='Starting cash for the static bounty agent used to seed targets')

    # Architecture
    parser.add_argument('--context_length', type=int, default=1, help='Transformer context length')
    parser.add_argument('--model_type', type=str, default='state_only',
                        choices=['state_only', 'state_action', 'rtgs_state_action'])
    parser.add_argument('--n_layer', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension')

    # Offline (optional)
    parser.add_argument('--offline_data_dir', type=str, default=None,
                        help='Directory with .pt episode files for offline pretraining')
    parser.add_argument('--offline_episode_num', type=int, default=100,
                        help='Number of offline episodes to load')
    parser.add_argument('--offline_epochs', type=int, default=5,
                        help='Number of offline training epochs')
    parser.add_argument('--offline_lr', type=float, default=5e-4,
                        help='Offline learning rate')
    parser.add_argument('--offline_batch_size', type=int, default=128,
                        help='Offline batch size')

    # Online (always)
    parser.add_argument('--online_buffer_size', type=int, default=4,
                        help='Total number of parallel envs for online training; split evenly across ranks in distributed mode')
    parser.add_argument('--online_epochs', type=int, default=100,
                        help='Number of online training epochs')
    parser.add_argument('--online_ppo_epochs', type=int, default=5,
                        help='PPO epochs per online update')
    parser.add_argument('--online_batch_size', type=int, default=128,
                        help='Per-rank PPO minibatch size')
    parser.add_argument('--online_lr', type=float, default=5e-4,
                        help='Online learning rate')
    parser.add_argument('--online_eval_interval', type=int, default=10,
                        help='Evaluate every N online epochs')
    parser.add_argument('--eval_episodes', type=int, default=4,
                        help='Number of eval episodes (= eval env threads)')

    # Visualization
    parser.add_argument('--capture_eval_episodes', type=int, default=1,
                        help='Number of eval threads to capture as trajectory artifacts (0 disables capture)')
    parser.add_argument('--trajectory_dir', type=str, default=None,
                        help='Directory for trajectory JSON/HTML artifacts (defaults to <log_dir>/trajectories)')

    # Distributed
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training via torchrun/torch.distributed')
    parser.add_argument('--backend', type=str, default=None,
                        help='torch.distributed backend (default: nccl on CUDA, else gloo)')

    # General
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--target_rtgs', type=float, default=5.0,
                        help='Target return-to-go for rollouts')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='TensorBoard log directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='Model checkpoint directory')

    return parser.parse_args()


def build_config(args) -> VampConfig:
    return VampConfig(
        F_size=args.F_size,
        n_agents=args.n_agents,
        max_timestep=args.max_timestep,
        initial_cash=args.initial_cash,
        initial_public_concrete_prob=args.initial_public_concrete_prob,
        gamma=args.gamma,
        kappa=args.kappa,
        lambda_diff=args.lambda_diff,
        alpha_util=args.alpha_util,
        beta_conj=args.beta_conj,
        n_buckets=args.n_buckets,
        horizon_H=args.horizon_H,
        query_init_weight_std=args.query_init_weight_std,
        query_global_lr=args.query_global_lr,
        query_local_lr=args.query_local_lr,
        query_private_truth_boost=args.query_private_truth_boost,
        query_public_truth_boost=args.query_public_truth_boost,
        operation_gas_fee=args.operation_gas_fee,
        publish_resolution_bonus=args.publish_resolution_bonus,
        target_init_prob=args.target_init_prob,
        target_init_min_price=args.target_init_min_price,
        target_init_max_price=args.target_init_max_price,
        target_init_max_quantity=args.target_init_max_quantity,
        target_init_cash=args.target_init_cash,
    )


def setup_runtime(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")

    if distributed and not dist.is_initialized():
        required_env = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
        missing = [name for name in required_env if name not in os.environ]
        if missing:
            raise RuntimeError(
                "Distributed training requires torchrun or an equivalent launcher. "
                f"Missing env vars: {', '.join(missing)}"
            )
        dist.init_process_group(backend=backend, init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    control_group = None
    if distributed and backend != "gloo":
        control_group = dist.new_group(backend="gloo")

    return {
        'distributed': distributed,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main': rank == 0,
        'device': device,
        'backend': backend,
        'control_group': control_group,
    }


def cleanup_runtime(runtime):
    if runtime['distributed'] and dist.is_initialized():
        try:
            if runtime['backend'] == "nccl" and runtime['device'].type == "cuda":
                dist.barrier(device_ids=[runtime['local_rank']])
            else:
                dist.barrier()
        except RuntimeError:
            pass
        dist.destroy_process_group()


def rank0_print(runtime, *args, **kwargs):
    if runtime['is_main']:
        print(*args, **kwargs)


def debug_print(runtime, *args, **kwargs):
    if os.environ.get("AGORA_DEBUG_DDP") == "1":
        print(f"[rank {runtime['rank']}]", *args, **kwargs, flush=True)


def broadcast_object(runtime, value):
    if not runtime['distributed']:
        return value
    payload = [value if runtime['is_main'] else None]
    kwargs = {}
    if runtime['control_group'] is not None:
        kwargs['group'] = runtime['control_group']
    dist.broadcast_object_list(payload, src=0, **kwargs)
    return payload[0]


def save_model_state(model, path):
    torch.save(unwrap_model(model).state_dict(), path)


def maybe_wrap_ddp(model, runtime):
    model = model.to(runtime['device'])
    if runtime['distributed']:
        if runtime['device'].type == "cuda":
            return DDP(
                model,
                device_ids=[runtime['local_rank']],
                output_device=runtime['local_rank'],
                find_unused_parameters=True,
            )
        return DDP(model, find_unused_parameters=True)
    return model


def build_models(args, cfg, local_obs_dim, global_obs_dim, action_dim):
    block_size = args.context_length * 3

    actor_config = GPTConfig(
        state_size=local_obs_dim,
        vocab_size=action_dim,
        block_size=block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_timestep=cfg.max_timestep,
    )
    critic_config = GPTConfig(
        state_size=global_obs_dim,
        vocab_size=action_dim,
        block_size=block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        max_timestep=cfg.max_timestep,
    )
    return GPT(actor_config, model_type='actor'), GPT(critic_config, model_type='critic')


def maybe_barrier(runtime):
    if runtime['distributed']:
        if runtime['backend'] == "nccl" and runtime['device'].type == "cuda":
            dist.barrier(device_ids=[runtime['local_rank']])
        else:
            dist.barrier()


def reduce_tensor(runtime, values, op="mean"):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=runtime['device'])
    if runtime['distributed']:
        if op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= runtime['world_size']
        elif op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"unsupported reduce op: {op}")
    return tensor


def reduce_scalar(runtime, value, op="mean"):
    return float(reduce_tensor(runtime, [value], op=op).item())


def get_local_env_threads(total_env_threads, runtime):
    if not runtime['distributed']:
        return total_env_threads
    if total_env_threads < runtime['world_size']:
        raise ValueError(
            f"online_buffer_size={total_env_threads} is smaller than world_size={runtime['world_size']}. "
            "Increase online_buffer_size or reduce the number of ranks."
        )
    if total_env_threads % runtime['world_size'] != 0:
        raise ValueError(
            f"online_buffer_size={total_env_threads} must be divisible by world_size={runtime['world_size']} "
            "so every rank performs the same number of rollout/training steps."
        )
    return total_env_threads // runtime['world_size']


def main():
    args = parse_args()
    runtime = setup_runtime(args)
    set_seed(args.seed + runtime['rank'])

    cfg = build_config(args)

    tmp_env = make_vamp_env(cfg, seed=args.seed)
    local_obs_dim = tmp_env.get_obs_size()
    global_obs_dim = tmp_env.get_state_size()
    action_dim = tmp_env.get_total_actions()
    del tmp_env

    rank0_print(runtime, f"VAMP: F={cfg.F_size}, N={cfg.n_agents}, T={cfg.max_timestep}")
    rank0_print(runtime, f"Dims: local_obs={local_obs_dim}, global_obs={global_obs_dim}, action={action_dim}")
    if runtime['distributed']:
        rank0_print(runtime, f"Distributed training enabled across {runtime['world_size']} ranks")

    local_online_buffer_size = get_local_env_threads(args.online_buffer_size, runtime)
    global_online_buffer_size = local_online_buffer_size * runtime['world_size']
    rank0_print(
        runtime,
        f"Online env threads: {local_online_buffer_size} per rank, {global_online_buffer_size} total",
    )

    actor, critic = build_models(args, cfg, local_obs_dim, global_obs_dim, action_dim)
    actor = maybe_wrap_ddp(actor, runtime)
    critic = maybe_wrap_ddp(critic, runtime)

    offline_trainer_config = TrainerConfig(
        max_epochs=args.offline_epochs,
        batch_size=args.offline_batch_size,
        learning_rate=args.offline_lr,
        mode="offline",
        device=runtime['device'],
        distributed=runtime['distributed'],
        rank=runtime['rank'],
        world_size=runtime['world_size'],
        use_distributed_sampler=runtime['distributed'],
    )
    online_trainer_config = TrainerConfig(
        max_epochs=args.online_ppo_epochs,
        batch_size=args.online_batch_size,
        learning_rate=args.online_lr,
        mode="online",
        device=runtime['device'],
        distributed=runtime['distributed'],
        rank=runtime['rank'],
        world_size=runtime['world_size'],
        use_distributed_sampler=False,
    )
    offline_trainer = Trainer(actor, critic, offline_trainer_config)
    online_trainer = Trainer(actor, critic, online_trainer_config)

    buffer = None
    rollout_worker = None
    train_env = None
    eval_env = None
    random_eval_env = None
    writer = None

    buffer = ReplayBuffer(args.context_length * 3, global_obs_dim, local_obs_dim, action_dim)
    rollout_worker = RolloutWorker(
        unwrap_model(actor),
        unwrap_model(critic),
        buffer,
        global_obs_dim,
        local_obs_dim,
        action_dim,
    )
    train_env = VampEnvWrapper(
        local_online_buffer_size,
        cfg,
        seed=args.seed + 2000 + runtime['rank'] * 10000,
    )

    if runtime['is_main']:
        eval_env = VampEnvWrapper(args.eval_episodes, cfg, seed=args.seed + 1000)
        random_eval_env = VampEnvWrapper(args.eval_episodes, cfg, seed=args.seed + 3000)

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir) if SummaryWriter is not None else NullSummaryWriter()
        if SummaryWriter is None:
            rank0_print(runtime, "tensorboard is not installed; continuing with a no-op SummaryWriter")

    best_eval_return = float('-inf')
    trajectory_dir = args.trajectory_dir or os.path.join(args.log_dir, "trajectories")

    try:
        if args.offline_data_dir is not None:
            rank0_print(runtime, f"\n=== Offline pretraining from {args.offline_data_dir} ===")
            if runtime['is_main']:
                buffer.reset()
                buffer.load_offline_data(
                    [args.offline_data_dir + "/"],
                    [args.offline_episode_num],
                )
                dataset = buffer.sample()
                rank0_print(runtime, f"Loaded {buffer.size} episodes into buffer")
                rank0_print(runtime, f"Dataset size: {len(dataset)}, max rtgs: {dataset.max_rtgs:.2f}")
                offline_payload = dataset.to_dict()
            else:
                offline_payload = None

            offline_payload = broadcast_object(runtime, offline_payload)
            dataset = StateActionReturnDataset.from_dict(offline_payload)
            actor_loss, critic_loss, _, _, _ = offline_trainer.train(dataset, train_critic=False)
            maybe_barrier(runtime)

            if runtime['is_main']:
                rank0_print(runtime, f"Offline training done. Actor loss: {actor_loss:.5f}")
                writer.add_scalar("offline/actor_loss", actor_loss, 0)

                save_model_state(actor, os.path.join(args.save_dir, "actor_offline.pt"))
                rank0_print(runtime, f"Saved offline checkpoint to {args.save_dir}/actor_offline.pt")

                eval_return, eval_win_rate, _, _, _, eval_economic_return, eval_shaping_return, _, _ = rollout_worker.rollout(
                    eval_env, args.target_rtgs, train=False, capture_threads=0
                )
                rank0_print(
                    runtime,
                    f"Post-offline eval: shaped_return={eval_return:.3f}, "
                    f"economic_return={eval_economic_return:.3f}, "
                    f"shaping_return={eval_shaping_return:.3f}, "
                    f"win_rate={eval_win_rate:.3f}",
                )
                writer.add_scalar("offline/eval_return", eval_return, 0)
                writer.add_scalar("offline/eval_economic_return", eval_economic_return, 0)
                writer.add_scalar("offline/eval_shaping_return", eval_shaping_return, 0)

        rank0_print(runtime, f"\n=== Online MAPPO training ({args.online_epochs} epochs) ===")
        epoch_iterator = tqdm(
            range(args.online_epochs),
            desc="online epochs",
            disable=not runtime['is_main'],
        )

        for epoch in epoch_iterator:
            epoch_start = time.perf_counter()
            buffer.reset(buffer_size=5000)
            rollout_start = time.perf_counter()
            (
                local_train_return,
                local_train_win_rate,
                local_steps,
                local_train_agent_returns,
                _,
                local_train_economic_return,
                local_train_shaping_return,
                local_train_agent_economic_returns,
                local_train_agent_shaping_returns,
            ) = rollout_worker.rollout(
                train_env, args.target_rtgs, train=True, capture_threads=0
            )
            dataset = buffer.sample()
            rollout_time = time.perf_counter() - rollout_start

            train_start = time.perf_counter()
            actor_loss, critic_loss, entropy, ratio, confidence = online_trainer.train(
                dataset, train_critic=True
            )
            train_time = time.perf_counter() - train_start
            maybe_barrier(runtime)

            train_return = reduce_scalar(runtime, local_train_return, op="mean")
            train_economic_return = reduce_scalar(runtime, local_train_economic_return, op="mean")
            train_shaping_return = reduce_scalar(runtime, local_train_shaping_return, op="mean")
            train_total_return = train_return * cfg.n_agents
            train_total_economic_return = train_economic_return * cfg.n_agents
            train_total_shaping_return = train_shaping_return * cfg.n_agents
            train_win_rate = reduce_scalar(runtime, local_train_win_rate, op="mean")
            steps = reduce_scalar(runtime, local_steps, op="sum")
            train_agent_returns = reduce_tensor(runtime, local_train_agent_returns, op="mean").cpu().tolist()
            train_agent_economic_returns = reduce_tensor(
                runtime, local_train_agent_economic_returns, op="mean"
            ).cpu().tolist()
            train_agent_shaping_returns = reduce_tensor(
                runtime, local_train_agent_shaping_returns, op="mean"
            ).cpu().tolist()
            rollout_time_max = reduce_scalar(runtime, rollout_time, op="max")
            train_time_max = reduce_scalar(runtime, train_time, op="max")
            samples_per_rank = len(dataset)
            train_samples_global = int(round(reduce_scalar(runtime, samples_per_rank, op="sum")))
            epoch_time = time.perf_counter() - epoch_start

            if runtime['is_main']:
                writer.add_scalar("online/train_return", train_return, epoch)
                writer.add_scalar("online/train_economic_return", train_economic_return, epoch)
                writer.add_scalar("online/train_shaping_return", train_shaping_return, epoch)
                writer.add_scalar("online/train_total_return", train_total_return, epoch)
                writer.add_scalar(
                    "online/train_total_economic_return",
                    train_total_economic_return,
                    epoch,
                )
                writer.add_scalar(
                    "online/train_total_shaping_return",
                    train_total_shaping_return,
                    epoch,
                )
                writer.add_scalar("online/train_win_rate", train_win_rate, epoch)
                writer.add_scalar("online/train_steps", steps, epoch)
                writer.add_scalar("online/train_samples_global", train_samples_global, epoch)
                writer.add_scalar("online/train_samples_per_rank", samples_per_rank, epoch)
                writer.add_scalar("online/rollout_time_s", rollout_time_max, epoch)
                writer.add_scalar("online/train_time_s", train_time_max, epoch)
                for a in range(cfg.n_agents):
                    writer.add_scalar(f"online/train_return_agent{a}", train_agent_returns[a], epoch)
                    writer.add_scalar(
                        f"online/train_economic_return_agent{a}",
                        train_agent_economic_returns[a],
                        epoch,
                    )
                    writer.add_scalar(
                        f"online/train_shaping_return_agent{a}",
                        train_agent_shaping_returns[a],
                        epoch,
                    )
                writer.add_scalar("online/actor_loss", actor_loss, epoch)
                writer.add_scalar("online/critic_loss", critic_loss, epoch)
                writer.add_scalar("online/entropy", entropy, epoch)
                writer.add_scalar("online/ratio", ratio, epoch)
                writer.add_scalar("online/confidence", confidence, epoch)

                postfix = {
                    'train_ret': f"{train_return:.3f}",
                    'train_econ': f"{train_economic_return:.3f}",
                    'train_shape': f"{train_shaping_return:.3f}",
                    'actor_loss': f"{actor_loss:.4f}",
                    'critic_loss': f"{critic_loss:.4f}",
                    'rollout_s': f"{rollout_time_max:.2f}",
                    'train_s': f"{train_time_max:.2f}",
                }

                if (epoch + 1) % args.online_eval_interval == 0:
                    debug_print(runtime, f"epoch {epoch}: eval start")
                    eval_start = time.perf_counter()
                    (
                        eval_return,
                        eval_win_rate,
                        _,
                        eval_agent_returns,
                        trajectories,
                        eval_economic_return,
                        eval_shaping_return,
                        eval_agent_economic_returns,
                        eval_agent_shaping_returns,
                    ) = rollout_worker.rollout(
                        eval_env,
                        args.target_rtgs,
                        train=False,
                        capture_threads=args.capture_eval_episodes,
                    )
                    debug_print(runtime, f"epoch {epoch}: eval rollout complete")
                    eval_time = time.perf_counter() - eval_start
                    writer.add_scalar("online/eval_return", eval_return, epoch)
                    writer.add_scalar("online/eval_economic_return", eval_economic_return, epoch)
                    writer.add_scalar("online/eval_shaping_return", eval_shaping_return, epoch)
                    writer.add_scalar("online/eval_total_return", eval_return * cfg.n_agents, epoch)
                    writer.add_scalar(
                        "online/eval_total_economic_return",
                        eval_economic_return * cfg.n_agents,
                        epoch,
                    )
                    writer.add_scalar(
                        "online/eval_total_shaping_return",
                        eval_shaping_return * cfg.n_agents,
                        epoch,
                    )
                    writer.add_scalar("online/eval_win_rate", eval_win_rate, epoch)
                    writer.add_scalar("online/eval_time_s", eval_time, epoch)
                    for a in range(cfg.n_agents):
                        writer.add_scalar(f"online/eval_return_agent{a}", eval_agent_returns[a], epoch)
                        writer.add_scalar(
                            f"online/eval_economic_return_agent{a}",
                            eval_agent_economic_returns[a],
                            epoch,
                        )
                        writer.add_scalar(
                            f"online/eval_shaping_return_agent{a}",
                            eval_agent_shaping_returns[a],
                            epoch,
                        )

                    debug_print(runtime, f"epoch {epoch}: random baseline start")
                    random_return, random_economic_return, random_shaping_return = random_rollout(
                        random_eval_env, cfg.n_agents, action_dim
                    )
                    debug_print(runtime, f"epoch {epoch}: random baseline complete")
                    writer.add_scalar("online/random_baseline_return", random_return, epoch)
                    writer.add_scalar("online/random_baseline_economic_return", random_economic_return, epoch)
                    writer.add_scalar("online/random_baseline_shaping_return", random_shaping_return, epoch)
                    writer.add_scalar("online/random_baseline_total_return", random_return * cfg.n_agents, epoch)
                    writer.add_scalar(
                        "online/random_baseline_total_economic_return",
                        random_economic_return * cfg.n_agents,
                        epoch,
                    )
                    writer.add_scalar(
                        "online/random_baseline_total_shaping_return",
                        random_shaping_return * cfg.n_agents,
                        epoch,
                    )
                    writer.add_scalar("online/eval_minus_random", eval_return - random_return, epoch)

                    if trajectories:
                        debug_print(runtime, f"epoch {epoch}: trajectory diagnostics start")
                        query_quality = summarize_query_model_diagnostics(trajectories)
                        for diag in query_quality:
                            agent_id = diag["agent_id"]
                            writer.add_scalar(
                                f"online/query_model_mae_agent{agent_id}",
                                diag["mae_all"],
                                epoch,
                            )
                            writer.add_scalar(
                                f"online/query_model_rmse_agent{agent_id}",
                                diag["rmse_all"],
                                epoch,
                            )
                            writer.add_scalar(
                                f"online/query_model_feasible_mae_agent{agent_id}",
                                diag["mae_feasible"],
                                epoch,
                            )
                            writer.add_scalar(
                                f"online/query_model_feasible_rmse_agent{agent_id}",
                                diag["rmse_feasible"],
                                epoch,
                            )
                        debug_print(runtime, f"epoch {epoch}: trajectory artifact write start")
                        artifacts = write_trajectory_artifacts(
                            trajectories,
                            output_dir=trajectory_dir,
                            split="eval",
                            epoch=epoch,
                        )
                        debug_print(runtime, f"epoch {epoch}: trajectory artifact write complete")
                        writer.add_text(
                            "online/trajectory_artifacts",
                            "\n".join(item['html'] for item in artifacts),
                            epoch,
                        )

                    postfix.update({
                        'eval_ret': f"{eval_return:.3f}",
                        'eval_econ': f"{eval_economic_return:.3f}",
                        'eval_shape': f"{eval_shaping_return:.3f}",
                        'random_ret': f"{random_return:.3f}",
                        'rand_econ': f"{random_economic_return:.3f}",
                        'eval_s': f"{eval_time:.2f}",
                    })
                    rank0_print(
                        runtime,
                        f"Epoch {epoch + 1}/{args.online_epochs}: "
                        f"train_ret={train_return:.3f} train_econ={train_economic_return:.3f} "
                        f"train_shape={train_shaping_return:.3f} "
                        f"eval_ret={eval_return:.3f} eval_econ={eval_economic_return:.3f} "
                        f"eval_shape={eval_shaping_return:.3f} "
                        f"random_ret={random_return:.3f} random_econ={random_economic_return:.3f} "
                        f"actor_loss={actor_loss:.5f} critic_loss={critic_loss:.5f} "
                        f"rollout_s={rollout_time_max:.2f} train_s={train_time_max:.2f} "
                        f"samples_global={train_samples_global}"
                    )

                    if eval_return > best_eval_return:
                        best_eval_return = eval_return
                        save_model_state(actor, os.path.join(args.save_dir, "actor_best.pt"))
                        save_model_state(critic, os.path.join(args.save_dir, "critic_best.pt"))
                        debug_print(runtime, f"epoch {epoch}: best checkpoint saved")

                writer.add_scalar("online/epoch_time_s", epoch_time, epoch)
                epoch_iterator.set_postfix(postfix)

            debug_print(runtime, f"epoch {epoch}: end-of-epoch barrier start")
            maybe_barrier(runtime)
            debug_print(runtime, f"epoch {epoch}: end-of-epoch barrier complete")

        if runtime['is_main']:
            save_model_state(actor, os.path.join(args.save_dir, "actor_final.pt"))
            save_model_state(critic, os.path.join(args.save_dir, "critic_final.pt"))
            rank0_print(runtime, f"\nTraining complete. Best eval return: {best_eval_return:.3f}")

    finally:
        if writer is not None:
            writer.close()
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        if random_eval_env is not None:
            random_eval_env.close()
        cleanup_runtime(runtime)


if __name__ == '__main__':
    main()
