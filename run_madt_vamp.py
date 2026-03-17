"""Main entry point: parse args -> build config -> create envs -> train MADT on VAMP.

Supports optional offline pretraining (if data dir provided) followed by
MAPPO RL self-play (always).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.vamp.config import VampConfig
from envs.env import make_vamp_env, VampEnvWrapper
from models.gpt_model import GPT, GPTConfig
from framework.buffer import ReplayBuffer
from framework.rollout import RolloutWorker
from framework.trainer import Trainer, TrainerConfig
from framework.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='MADT training on VAMP environment')

    # Environment
    parser.add_argument('--F_size', type=int, default=8, help='Formula universe size')
    parser.add_argument('--n_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--max_timestep', type=int, default=100, help='Episode length')
    parser.add_argument('--initial_cash', type=float, default=10.0, help='Starting cash per agent')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Kernels
    parser.add_argument('--kappa', type=float, default=0.5, help='Budget concavity')
    parser.add_argument('--lambda_diff', type=float, default=1.0, help='Difficulty transform')
    parser.add_argument('--alpha_util', type=float, default=1.0, help='Utility sensitivity')
    parser.add_argument('--beta_conj', type=float, default=1.0, help='Conjecture mass exponent')

    # Query model
    parser.add_argument('--n_buckets', type=int, default=4, help='Query model buckets')
    parser.add_argument('--horizon_H', type=int, default=20, help='Query model horizon')

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
                        help='Number of parallel envs for online training')
    parser.add_argument('--online_epochs', type=int, default=100,
                        help='Number of online training epochs')
    parser.add_argument('--online_ppo_epochs', type=int, default=5,
                        help='PPO epochs per online update')
    parser.add_argument('--online_lr', type=float, default=5e-4,
                        help='Online learning rate')
    parser.add_argument('--online_eval_interval', type=int, default=10,
                        help='Evaluate every N online epochs')
    parser.add_argument('--eval_episodes', type=int, default=4,
                        help='Number of eval episodes (= eval env threads)')

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
        gamma=args.gamma,
        kappa=args.kappa,
        lambda_diff=args.lambda_diff,
        alpha_util=args.alpha_util,
        beta_conj=args.beta_conj,
        n_buckets=args.n_buckets,
        horizon_H=args.horizon_H,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = build_config(args)

    # ── Extract dims from a temp env ──
    tmp_env = make_vamp_env(cfg, seed=args.seed)
    local_obs_dim = tmp_env.get_obs_size()
    global_obs_dim = tmp_env.get_state_size()
    action_dim = tmp_env.get_total_actions()
    del tmp_env

    print(f"VAMP: F={cfg.F_size}, N={cfg.n_agents}, T={cfg.max_timestep}")
    print(f"Dims: local_obs={local_obs_dim}, global_obs={global_obs_dim}, action={action_dim}")

    # block_size must always be context_length * 3 because the buffer/dataset
    # recovers context_length via block_size // 3 (regardless of model_type)
    block_size = args.context_length * 3

    # ── Create actor (local obs → action logits) ──
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
    actor = GPT(actor_config, model_type='actor')

    # ── Create critic (global obs → value) ──
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
    critic = GPT(critic_config, model_type='critic')

    # ── Buffer, RolloutWorker ──
    buffer = ReplayBuffer(block_size, global_obs_dim, local_obs_dim, action_dim)
    rollout_worker = RolloutWorker(actor, critic, buffer, global_obs_dim, local_obs_dim, action_dim)

    # ── Eval env ──
    eval_env = VampEnvWrapper(args.eval_episodes, cfg, seed=args.seed + 1000)

    # ── TensorBoard ──
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ════════════════════════════════════════════════════
    # Phase 1: Offline pretraining (optional)
    # ════════════════════════════════════════════════════
    if args.offline_data_dir is not None:
        print(f"\n=== Offline pretraining from {args.offline_data_dir} ===")
        buffer.reset()
        buffer.load_offline_data(
            [args.offline_data_dir + "/"],
            [args.offline_episode_num],
        )
        print(f"Loaded {buffer.size} episodes into buffer")

        dataset = buffer.sample()
        print(f"Dataset size: {len(dataset)}, max rtgs: {dataset.max_rtgs:.2f}")

        offline_trainer_config = TrainerConfig(
            max_epochs=args.offline_epochs,
            batch_size=args.offline_batch_size,
            learning_rate=args.offline_lr,
            mode="offline",
        )
        offline_trainer = Trainer(actor, critic, offline_trainer_config)
        actor_loss, critic_loss, _, _, _ = offline_trainer.train(dataset, train_critic=False)
        print(f"Offline training done. Actor loss: {actor_loss:.5f}")

        writer.add_scalar("offline/actor_loss", actor_loss, 0)

        # Save offline checkpoint
        torch.save(actor.state_dict(), os.path.join(args.save_dir, "actor_offline.pt"))
        print(f"Saved offline checkpoint to {args.save_dir}/actor_offline.pt")

        # Eval after offline
        eval_return, eval_win_rate, _ = rollout_worker.rollout(eval_env, args.target_rtgs, train=False)
        print(f"Post-offline eval: return={eval_return:.3f}, win_rate={eval_win_rate:.3f}")
        writer.add_scalar("offline/eval_return", eval_return, 0)

    # ════════════════════════════════════════════════════
    # Phase 2: Online MAPPO (always)
    # ════════════════════════════════════════════════════
    print(f"\n=== Online MAPPO training ({args.online_epochs} epochs) ===")

    train_env = VampEnvWrapper(args.online_buffer_size, cfg, seed=args.seed + 2000)

    online_trainer_config = TrainerConfig(
        max_epochs=args.online_ppo_epochs,
        batch_size=128,  # ignored for online (uses full buffer)
        learning_rate=args.online_lr,
        mode="online",
    )
    online_trainer = Trainer(actor, critic, online_trainer_config)

    best_eval_return = float('-inf')

    for epoch in range(args.online_epochs):
        # Collect rollouts
        buffer.reset(buffer_size=5000)
        train_return, train_win_rate, steps = rollout_worker.rollout(
            train_env, args.target_rtgs, train=True
        )

        # Train PPO
        dataset = buffer.sample()
        actor_loss, critic_loss, entropy, ratio, confidence = online_trainer.train(
            dataset, train_critic=True
        )

        writer.add_scalar("online/train_return", train_return, epoch)
        writer.add_scalar("online/train_win_rate", train_win_rate, epoch)
        writer.add_scalar("online/actor_loss", actor_loss, epoch)
        writer.add_scalar("online/critic_loss", critic_loss, epoch)
        writer.add_scalar("online/entropy", entropy, epoch)
        writer.add_scalar("online/ratio", ratio, epoch)
        writer.add_scalar("online/confidence", confidence, epoch)

        # Eval
        if (epoch + 1) % args.online_eval_interval == 0:
            eval_return, eval_win_rate, _ = rollout_worker.rollout(
                eval_env, args.target_rtgs, train=False
            )
            writer.add_scalar("online/eval_return", eval_return, epoch)
            writer.add_scalar("online/eval_win_rate", eval_win_rate, epoch)

            print(f"Epoch {epoch+1}/{args.online_epochs}: "
                  f"train_ret={train_return:.3f} eval_ret={eval_return:.3f} "
                  f"actor_loss={actor_loss:.5f} critic_loss={critic_loss:.5f}")

            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(actor.state_dict(), os.path.join(args.save_dir, "actor_best.pt"))
                torch.save(critic.state_dict(), os.path.join(args.save_dir, "critic_best.pt"))

    # Save final
    torch.save(actor.state_dict(), os.path.join(args.save_dir, "actor_final.pt"))
    torch.save(critic.state_dict(), os.path.join(args.save_dir, "critic_final.pt"))
    print(f"\nTraining complete. Best eval return: {best_eval_return:.3f}")

    writer.close()
    train_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
