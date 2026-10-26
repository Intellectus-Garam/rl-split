"""Learner-side VecEnv that forwards every step to the Worker (central inference)."""

from __future__ import annotations

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from rl_split.comm import types as msg
from rl_split.comm.convert import flat, log_from_json, to_torch
from rl_split.session import CtrlSender, wait_worker

OBS_TIMEOUT_S = 120.0


class RemoteEnv(VecEnv):
    """Looks like RslRlVecEnvWrapper to OnPolicyRunner; the simulation runs on the Worker."""

    def __init__(self, node, cfg, device: str) -> None:
        self.cfg = cfg.env  # stored by the rsl_rl Logger
        self._config = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = cfg.action_dim
        self.max_episode_length = cfg.env["max_episode_length"]
        self.device = device
        # OnPolicyRunner overwrites this when init_at_random_ep_len is set, but it would not
        # reach the simulation. The Worker randomizes its own buffer instead (see start()).
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        self._node = node
        self._obs_in = node.inbox("central_obs")
        node.writer("central_action")
        node.writer("ctrl")
        self._ctrl = CtrlSender(node)
        self._step_id = 0
        self._obs = TensorDict(
            {cfg.env["obs_group"]: torch.zeros(self.num_envs, cfg.obs_dim, device=device)},
            batch_size=[self.num_envs],
        )

    def start(self, init_at_random_ep_len: bool) -> None:
        """Handshake with the Worker and receive the initial observation (step 0)."""
        wait_worker(self._node)
        self._ctrl.send("start", init_at_random_ep_len=init_at_random_ep_len)
        self._obs, *_ = self._receive(0)
        print("[learner] received initial observation")

    def _receive(self, step_id: int):
        sample = self._obs_in.get(timeout=OBS_TIMEOUT_S)
        if sample is None:
            raise TimeoutError(f"no observation for step {step_id} in {OBS_TIMEOUT_S}s")
        if sample.step_id != step_id:
            raise RuntimeError(f"expected observation {step_id}, got {sample.step_id}")
        n, d = self.num_envs, self.device
        obs = TensorDict(
            {self._config.env["obs_group"]: to_torch(sample.obs, np.float32, (n, self._config.obs_dim), d)},
            batch_size=[n],
        )
        rewards = to_torch(sample.rewards, np.float32, (n,), d)
        dones = to_torch(sample.dones, np.uint8, (n,), d).long()
        extras = {
            "time_outs": to_torch(sample.time_outs, np.uint8, (n,), d).bool(),
            "log": log_from_json(sample.log_json),
        }
        return obs, rewards, dones, extras

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        self._node.write("central_action", msg.CentralAction(step_id=self._step_id, actions=flat(actions)))
        self._step_id += 1
        self._obs, rewards, dones, extras = self._receive(self._step_id)
        return self._obs, rewards, dones, extras

    def close(self) -> None:
        self._ctrl.stop()
