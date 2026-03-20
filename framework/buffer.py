import torch
import numpy as np
import copy, glob
from torch.utils.data import Dataset
from .utils import padding_obs, padding_ava


class StateActionReturnDataset(Dataset):

    def __init__(self, global_state, local_obs, block_size, actions, done_idxs, rewards, avas, v_values, rtgs, rets,
                 advs, timesteps):
        self.block_size = block_size
        self.done_idxs = done_idxs
        self._done_idxs_arr = np.array(done_idxs)

        # Pre-tensorize all arrays for fast __getitem__
        self.global_state = torch.tensor(np.array(global_state), dtype=torch.float32)
        self.local_obs = torch.tensor(np.array(local_obs), dtype=torch.float32)
        self.actions = torch.tensor(np.array(actions), dtype=torch.long)
        self.rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        self.avas = torch.tensor(np.array(avas), dtype=torch.long)
        self.v_values = torch.tensor(np.array(v_values), dtype=torch.float32)
        self.rtgs = torch.tensor(np.array(rtgs), dtype=torch.float32)
        self.rets = torch.tensor(np.array(rets), dtype=torch.float32)
        self.advs = torch.tensor(np.array(advs), dtype=torch.float32)
        self.timesteps = torch.tensor(np.array(timesteps), dtype=torch.int64)

    def __len__(self):
        return len(self.global_state)

    def stats(self):
        done_arr = self._done_idxs_arr
        print("max episode length: ", max(done_arr[1:] - done_arr[:-1]))
        print("min episode length: ", min(done_arr[1:] - done_arr[:-1]))
        print("max rtgs: ", self.rtgs.max().item())
        print("aver episode rtgs: ", np.mean([self.rtgs[i].item() for i in self.done_idxs[:-1]]))

    @property
    def max_rtgs(self):
        return self.rtgs.max().item()

    def to_dict(self):
        return {
            'global_state': self.global_state.numpy().tolist(),
            'local_obs': self.local_obs.numpy().tolist(),
            'block_size': int(self.block_size),
            'actions': self.actions.numpy().tolist(),
            'done_idxs': copy.deepcopy(self.done_idxs),
            'rewards': self.rewards.numpy().tolist(),
            'avas': self.avas.numpy().tolist(),
            'v_values': self.v_values.numpy().tolist(),
            'rtgs': self.rtgs.numpy().tolist(),
            'rets': self.rets.numpy().tolist(),
            'advs': self.advs.numpy().tolist(),
            'timesteps': self.timesteps.numpy().tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            global_state=payload['global_state'],
            local_obs=payload['local_obs'],
            block_size=payload['block_size'],
            actions=payload['actions'],
            done_idxs=payload['done_idxs'],
            rewards=payload['rewards'],
            avas=payload['avas'],
            v_values=payload['v_values'],
            rtgs=payload['rtgs'],
            rets=payload['rets'],
            advs=payload['advs'],
            timesteps=payload['timesteps'],
        )

    def __getitem__(self, idx):
        context_length = self.block_size // 3
        done_idx = idx + context_length
        # Binary search for the next done boundary after idx
        pos = np.searchsorted(self._done_idxs_arr, idx, side='right')
        if pos < len(self._done_idxs_arr):
            done_idx = min(int(self._done_idxs_arr[pos]), done_idx)
        # Clamp idx to episode start (don't go before the previous done boundary)
        ep_start = int(self._done_idxs_arr[pos - 1]) if pos > 0 else 0
        idx = max(ep_start, done_idx - context_length)

        states = self.global_state[idx:done_idx]
        obss = self.local_obs[idx:done_idx]

        at_done = pos < len(self._done_idxs_arr) and done_idx == int(self._done_idxs_arr[pos])
        if at_done:
            zeros_s = torch.zeros_like(self.global_state[idx:idx+1])
            next_states = torch.cat([self.global_state[idx+1:done_idx], zeros_s], dim=0)
            zeros_r = torch.zeros_like(self.rtgs[idx:idx+1])
            next_rtgs = torch.cat([self.rtgs[idx+1:done_idx], zeros_r], dim=0)
        else:
            next_states = self.global_state[idx+1:done_idx+1]
            next_rtgs = self.rtgs[idx+1:done_idx+1]

        is_episode_start = (idx == 0) or (pos > 0 and idx == int(self._done_idxs_arr[pos - 1]))
        if is_episode_start:
            zero_act = torch.zeros_like(self.actions[idx:idx+1])
            pre_actions = torch.cat([zero_act, self.actions[idx:done_idx-1]], dim=0)
        else:
            pre_actions = self.actions[idx-1:done_idx-1]
        actions = self.actions[idx:done_idx]

        rewards = self.rewards[idx:done_idx]
        avas = self.avas[idx:done_idx]
        v_values = self.v_values[idx:done_idx]
        rtgs = self.rtgs[idx:done_idx]
        rets = self.rets[idx:done_idx]
        advs = self.advs[idx:done_idx]
        timesteps = self.timesteps[idx:done_idx]

        dones = torch.zeros_like(rewards)
        if at_done:
            dones[-1][0] = 1

        return states, obss, actions, rewards, avas, v_values, rtgs, rets, advs, timesteps, pre_actions, next_states, next_rtgs, dones


class ReplayBuffer:

    def __init__(self, block_size, global_obs_dim, local_obs_dim, action_dim):
        self.block_size = block_size
        self.buffer_size = 5000
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.data = []
        self.episodes = []
        self.episode_dones = []
        self.gamma = 0.99
        self.gae_lambda = 0.95

    @property
    def size(self):
        return len(self.data)

    def insert(self, global_obs, local_obs, action, reward, done, available_actions, v_value):
        n_threads, n_agents = np.shape(reward)[0], np.shape(reward)[1]
        for n in range(n_threads):
            if len(self.episodes) < n + 1:
                self.episodes.append([])
                self.episode_dones.append(False)
            if not self.episode_dones[n]:
                for i in range(n_agents):
                    if len(self.episodes[n]) < i + 1:
                        self.episodes[n].append([])
                    step = [global_obs[n][i].tolist(), local_obs[n][i].tolist(), action[n][i].tolist(),
                            reward[n][i].tolist(), done[n][i], available_actions[n][i].tolist(), v_value[n][i].tolist()]
                    self.episodes[n][i].append(step)
                if np.all(done[n]):
                    self.episode_dones[n] = True
                    if self.size > self.buffer_size:
                        raise NotImplementedError
                    if self.size == self.buffer_size:
                        del self.data[0]
                    self.data.append(copy.deepcopy(self.episodes[n]))
        if np.all(self.episode_dones):
            self.episodes = []
            self.episode_dones = []

    def reset(self, num_keep=0, buffer_size=5000):
        self.buffer_size = buffer_size
        if num_keep == 0:
            self.data = []
        elif self.size >= num_keep:
            keep_idx = np.random.randint(0, self.size, num_keep)
            self.data = [self.data[idx] for idx in keep_idx]

    def load_offline_data(self, data_dir, offline_episode_num, max_epi_length=400):
        for j in range(len(data_dir)):
            path_files = glob.glob(pathname=data_dir[j] + "*")
            for i in range(offline_episode_num[j]):
                episode = torch.load(path_files[i], weights_only=False)
                for agent_trajectory in episode:
                    for step in agent_trajectory:
                        step[0] = padding_obs(step[0], self.global_obs_dim)
                        step[1] = padding_obs(step[1], self.local_obs_dim)
                        step[5] = padding_ava(step[5], self.action_dim)
                self.data.append(episode)

    def sample(self):
        global_states = []
        local_obss = []
        actions = []
        rewards = []
        avas = []
        v_values = []
        rtgs = []
        rets = []
        done_idxs = []
        time_steps = []
        advs = []

        for episode_idx in range(self.size):
            episode = self.get_episode(episode_idx)
            if episode is None:
                continue
            for agent_trajectory in episode:
                time_step = 0
                for step in agent_trajectory:
                    g, o, a, r, d, ava, v, rtg, ret, adv = step
                    global_states.append(g)
                    local_obss.append(o)
                    actions.append(a)
                    rewards.append(r)
                    avas.append(ava)
                    v_values.append(v)
                    rtgs.append(rtg)
                    rets.append(ret)
                    advs.append(adv)
                    time_steps.append([time_step])
                    time_step += 1
                done_idxs.append(len(global_states))

        dataset = StateActionReturnDataset(global_states, local_obss, self.block_size, actions, done_idxs, rewards,
                                           avas, v_values, rtgs, rets, advs, time_steps)
        return dataset

    def get_episode(self, index):
        episode = copy.deepcopy(self.data[index])

        for agent_trajectory in episode:
            rtg = 0.
            ret = 0.
            adv = 0.
            for i in reversed(range(len(agent_trajectory))):
                if len(agent_trajectory[i]) == 6:
                    agent_trajectory[i].append([0.])
                elif len(agent_trajectory[i]) == 7:
                    pass
                else:
                    raise NotImplementedError

                reward = agent_trajectory[i][3][0]
                rtg += reward
                agent_trajectory[i].append([rtg])

                if i == len(agent_trajectory) - 1:
                    next_v = 0.
                else:
                    next_v = agent_trajectory[i + 1][6][0]
                v = agent_trajectory[i][6][0]
                delta = reward + self.gamma * next_v - v
                adv = delta + self.gamma * self.gae_lambda * adv

                ret = reward + self.gamma * ret

                agent_trajectory[i].append([ret])
                agent_trajectory[i].append([adv])

        for i in range(len(episode)):
            end_idx = 0
            for step in episode[i]:
                if step[4]:
                    break
                else:
                    end_idx += 1
            episode[i] = episode[i][0:end_idx + 1]
        return episode
