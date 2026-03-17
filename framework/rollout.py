import numpy as np
import torch

from .utils import get_model_device, sample
from .utils import padding_obs, padding_ava


class RolloutWorker:

    def __init__(self, model, critic_model, buffer, global_obs_dim, local_obs_dim, action_dim):
        self.buffer = buffer
        self.model = model
        self.critic_model = critic_model
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.device = get_model_device(model)

    def rollout(self, env, ret, train=True, random_rate=0., capture_threads=0):
        del random_rate
        self.model.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0.0, 0.0, 0, np.zeros(env.n_threads)
        T_agent_rewards = np.zeros(env.num_agents)
        T_economic_rewards = 0.0
        T_shaping_rewards = 0.0
        T_agent_economic_rewards = np.zeros(env.num_agents)
        T_agent_shaping_rewards = np.zeros(env.num_agents)

        obs, share_obs, available_actions = env.real_env.reset()
        obs = padding_obs(obs, self.local_obs_dim)
        share_obs = padding_obs(share_obs, self.global_obs_dim)
        available_actions = padding_ava(available_actions, self.action_dim)

        capture_threads = min(int(capture_threads), env.n_threads)
        capture_indices = list(range(capture_threads))
        trajectories = []
        cumulative_returns = {
            n: np.zeros(env.num_agents, dtype=np.float64) for n in capture_indices
        }
        if capture_indices:
            snapshots = env.real_env.get_env_snapshots()
            for n in capture_indices:
                trajectories.append({
                    'thread_index': int(n),
                    'initial_state': snapshots[n],
                    'steps': [],
                })

        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret
        actions = np.zeros((env.n_threads, env.num_agents, 1, 1))
        timesteps = torch.zeros((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
        t = 0

        while True:
            sampled_action, v_value = sample(
                self.model,
                self.critic_model,
                state=global_states.view(-1, np.shape(global_states)[2], np.shape(global_states)[3]),
                obs=local_obss.view(-1, np.shape(local_obss)[2], np.shape(local_obss)[3]),
                sample=train,
                actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(
                    -1, np.shape(actions)[2], np.shape(actions)[3]
                ),
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(
                    -1, np.shape(rtgs)[2], np.shape(rtgs)[3]
                ),
                timesteps=timesteps.to(self.device),
                available_actions=torch.from_numpy(available_actions).view(
                    -1, np.shape(available_actions)[-1]
                ),
            )

            action = sampled_action.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
            action_indices = action[..., 0]

            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            if capture_indices:
                pre_snapshots = env.real_env.get_env_snapshots()
                action_descriptions = env.real_env.describe_actions(action_indices)

            obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step(action)
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
            available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            if capture_indices:
                post_snapshots = env.real_env.get_env_snapshots()
                for item in trajectories:
                    n = item['thread_index']
                    if episode_dones[n]:
                        continue
                    cumulative_returns[n] += rewards[n, :, 0]
                    item['steps'].append({
                        'step_index': int(len(item['steps'])),
                        'action_indices': [int(v) for v in action_indices[n].tolist()],
                        'actions': action_descriptions[n],
                        'rewards': [float(v) for v in rewards[n, :, 0].tolist()],
                        'cumulative_returns': [float(v) for v in cumulative_returns[n].tolist()],
                        'done': bool(np.all(dones[n])),
                        'state_before': pre_snapshots[n],
                        'state_after': post_snapshots[n],
                    })

            if train:
                v_value = v_value.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                self.buffer.insert(cur_global_obs, cur_local_obs, action, rewards, dones, cur_ava, v_value)

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    economic_step = 0.0
                    shaping_step = 0.0
                    for a in range(env.num_agents):
                        T_agent_rewards[a] += rewards[n, a, 0]
                        economic_reward = float(infos[n][a].get('economic_reward', rewards[n, a, 0]))
                        shaping_reward = float(infos[n][a].get('shaping_reward', 0.0))
                        T_agent_economic_rewards[a] += economic_reward
                        T_agent_shaping_rewards[a] += shaping_reward
                        economic_step += economic_reward
                        shaping_step += shaping_reward
                    T_economic_rewards += economic_step / env.num_agents
                    T_shaping_rewards += shaping_step / env.num_agents
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        if infos[n][0].get('won', False):
                            T_wins += 1.0
            if np.all(episode_dones):
                break

            rtgs = np.concatenate([rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2)
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
            global_states = torch.cat([global_states, global_state], dim=2)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            actions = np.concatenate([actions, np.expand_dims(action, axis=2)], axis=2)
            timestep = t * torch.ones((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
            timesteps = torch.cat([timesteps, timestep], dim=1)

        aver_return = T_rewards / env.n_threads
        aver_win_rate = T_wins / env.n_threads
        aver_agent_returns = T_agent_rewards / env.n_threads
        aver_economic_return = T_economic_rewards / env.n_threads
        aver_shaping_return = T_shaping_rewards / env.n_threads
        aver_agent_economic_returns = T_agent_economic_rewards / env.n_threads
        aver_agent_shaping_returns = T_agent_shaping_rewards / env.n_threads
        self.model.train(True)
        self.critic_model.train(True)
        return (
            aver_return,
            aver_win_rate,
            steps,
            aver_agent_returns,
            trajectories,
            aver_economic_return,
            aver_shaping_return,
            aver_agent_economic_returns,
            aver_agent_shaping_returns,
        )
