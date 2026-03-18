import unittest

import numpy as np
import torch

from envs.env_wrappers import _attach_terminal_snapshot
from framework.rollout import RolloutWorker


class _DummyActor(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states, pre_actions=None, rtgs=None, timesteps=None):
        batch_size, seq_len = states.shape[0], states.shape[1]
        return torch.zeros(batch_size, seq_len, self.action_dim, device=states.device)

    def get_block_size(self):
        return 3


class _DummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, states, pre_actions=None, rtgs=None, timesteps=None):
        batch_size, seq_len = states.shape[0], states.shape[1]
        return torch.zeros(batch_size, seq_len, 1, device=states.device)

    def get_block_size(self):
        return 3


class _FakeBuffer:
    def insert(self, *args, **kwargs):
        raise AssertionError("buffer.insert should not be called when train=False")


class _FakeRealEnv:
    def __init__(self):
        self._reset_snapshot = self._snapshot(timestep=0, resolved=[])
        self._current_snapshot = self._reset_snapshot

    def _snapshot(self, timestep, resolved):
        return {
            "timestep": timestep,
            "jobs": [None],
            "query_responses": [None],
            "query_model_quality": [{"agent_id": 0, "mae_all": 0.0, "rmse_all": 0.0, "mae_feasible": 0.0, "rmse_feasible": 0.0}],
            "agents": [
                {
                    "agent_id": 0,
                    "cash": 10.0,
                    "worst_case_balance": 10.0,
                    "positions": [],
                    "library": {"concrete": [], "resolved": []},
                    "cumulative_proof": [0.0],
                    "cumulative_conj": [0.0],
                }
            ],
            "public_library": {
                "concrete": [],
                "resolved": [{"formula": formula, "deps": [], "solve_time": timestep, "solver": 0} for formula in resolved],
            },
            "market_summary": {
                "tracked_agent_cash_total": 10.0,
                "system_cash_total": 10.0,
                "bounty_agent_cash": 0.0,
            },
            "offers": [],
        }

    def reset(self):
        self._current_snapshot = self._reset_snapshot
        obs = np.zeros((1, 1, 1), dtype=np.float32)
        share_obs = np.zeros((1, 1, 1), dtype=np.float32)
        available_actions = np.ones((1, 1, 2), dtype=np.float32)
        return obs, share_obs, available_actions

    def step(self, action):
        del action
        terminal_snapshot = self._snapshot(timestep=1, resolved=[3])
        self._current_snapshot = self._reset_snapshot
        obs = np.zeros((1, 1, 1), dtype=np.float32)
        share_obs = np.zeros((1, 1, 1), dtype=np.float32)
        rewards = np.zeros((1, 1, 1), dtype=np.float32)
        dones = np.ones((1, 1, 1), dtype=np.float32)
        infos = [[{
            "won": False,
            "economic_reward": 0.0,
            "shaping_reward": 0.0,
            "terminal_snapshot": terminal_snapshot,
        }]]
        available_actions = np.ones((1, 1, 2), dtype=np.float32)
        return obs, share_obs, rewards, dones, infos, available_actions

    def get_env_snapshots(self):
        return [self._current_snapshot]

    def describe_actions(self, actions):
        del actions
        return [[{"type": "noop", "formula": None, "budget": None, "deadline": None, "loss": None, "side": None, "price": None, "offer_slot": None}]]


class _FakeEnv:
    def __init__(self):
        self.real_env = _FakeRealEnv()
        self.n_threads = 1
        self.num_agents = 1
        self.max_timestep = 1


class TerminalSnapshotRegressionTests(unittest.TestCase):
    def test_attach_terminal_snapshot_copies_info_dicts(self):
        info = [{"economic_reward": 1.0}, {"economic_reward": 2.0}]
        snapshot = {"timestep": 5}

        augmented = _attach_terminal_snapshot(info, snapshot)

        self.assertEqual(snapshot, augmented[0]["terminal_snapshot"])
        self.assertEqual(snapshot, augmented[1]["terminal_snapshot"])
        self.assertNotIn("terminal_snapshot", info[0])

    def test_rollout_uses_terminal_snapshot_for_done_step(self):
        worker = RolloutWorker(
            _DummyActor(action_dim=2),
            _DummyCritic(),
            _FakeBuffer(),
            global_obs_dim=1,
            local_obs_dim=1,
            action_dim=2,
        )

        _, _, _, _, trajectories, _, _, _, _ = worker.rollout(
            _FakeEnv(),
            ret=0.0,
            train=False,
            capture_threads=1,
        )

        final_step = trajectories[0]["steps"][-1]
        self.assertTrue(final_step["done"])
        self.assertEqual(1, final_step["state_after"]["timestep"])
        self.assertEqual([3], [item["formula"] for item in final_step["state_after"]["public_library"]["resolved"]])


if __name__ == "__main__":
    unittest.main()
