"""Learner-side VecEnv that forwards every step to the Workers (central inference)."""

from __future__ import annotations

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from rl_split.comm import types as msg
from rl_split.comm.convert import cat_torch, flat, fmt_bytes, merge_logs, payload_bytes
from rl_split.comm.node import per_worker
from rl_split.session import CtrlSender, wait_workers, worker_rows

OBS_TIMEOUT_S = 120.0


class RemoteEnv(VecEnv):
    """Looks like RslRlVecEnvWrapper to OnPolicyRunner; the simulation runs on the Workers.

    The Workers' envs are stacked in worker_id order into one batch of num_workers * env.num_envs.
    ctrl is owned by the caller, which sends "stop" on exit.
    """

    def __init__(self, node, cfg, num_workers: int, ctrl: CtrlSender, device: str) -> None:
        self.cfg = cfg.env  # stored by the rsl_rl Logger
        self._config = cfg
        self._num_workers = num_workers
        self.num_envs = num_workers * cfg.num_envs
        self.num_actions = cfg.action_dim
        self.max_episode_length = cfg.env["max_episode_length"]
        self.device = device
        # OnPolicyRunner overwrites this when init_at_random_ep_len is set, but it would not
        # reach the simulation. The Worker randomizes its own buffer instead (see start()).
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        self._node = node
        self._obs_in = [node.inbox(per_worker("central_obs", w)) for w in range(num_workers)]
        for w in range(num_workers):
            node.writer(per_worker("central_action", w))
        self._ctrl = ctrl
        self._step_id = 0
        self._obs = TensorDict(
            {cfg.env["obs_group"]: torch.zeros(self.num_envs, cfg.obs_dim, device=device)},
            batch_size=[self.num_envs],
        )

    def start(self, init_at_random_ep_len: bool) -> None:
        """Handshake with the Workers and receive the initial observation (step 0)."""
        wait_workers(self._node, self._num_workers)
        self._ctrl.send("start", init_at_random_ep_len=init_at_random_ep_len)
        self._obs, *_ = self._receive(0)

    def _receive(self, step_id: int):
        samples = []
        for w, inbox in enumerate(self._obs_in):
            s = inbox.get(timeout=OBS_TIMEOUT_S)
            if s is None:
                raise TimeoutError(f"no observation for step {step_id} from worker {w} in {OBS_TIMEOUT_S}s")
            if (s.worker_id, s.step_id) != (w, step_id):
                raise RuntimeError(f"expected observation {step_id} from worker {w}, got {s.step_id} from worker {s.worker_id}")
            samples.append(s)
        self._node.check_single_worker()
        size = sum(payload_bytes(s) for s in samples)
        print(f"[learner] sub central_obs step {step_id} from {len(samples)} worker(s) ({fmt_bytes(size)})")
        n, d, k = self.num_envs, self.device, self._config.num_envs
        shapes = [k] * len(samples)
        obs = TensorDict(
            {
                self._config.env["obs_group"]: cat_torch(
                    [s.obs for s in samples], np.float32, [(k, self._config.obs_dim)] * len(samples), d
                )
            },
            batch_size=[n],
        )
        rewards = cat_torch([s.rewards for s in samples], np.float32, shapes, d)
        dones = cat_torch([s.dones for s in samples], np.uint8, shapes, d).long()
        extras = {
            "time_outs": cat_torch([s.time_outs for s in samples], np.uint8, shapes, d).bool(),
            "log": merge_logs([s.log_json for s in samples]),
        }
        return obs, rewards, dones, extras

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        size = 0
        for w in range(self._num_workers):
            rows = worker_rows(w, self._config.num_envs)
            sample = msg.CentralAction(worker_id=w, step_id=self._step_id, actions=flat(actions[rows]))
            self._node.write(per_worker("central_action", w), sample)
            size += payload_bytes(sample)
        print(f"[learner] pub central_action step {self._step_id} to {self._num_workers} worker(s) ({fmt_bytes(size)})")
        self._step_id += 1
        self._obs, rewards, dones, extras = self._receive(self._step_id)
        return self._obs, rewards, dones, extras

    def close(self) -> None:
        pass  # "stop" is sent by the learner script, see rl_split/session.py
