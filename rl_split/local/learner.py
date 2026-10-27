"""Learner side of local inference: fill RolloutStorage from streamed transitions and run PPO updates."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from rl_split.comm import types as msg
from rl_split.comm.convert import fmt_bytes, log_from_json, payload_bytes, state_to_msg, to_torch
from rl_split.session import CtrlSender, wait_worker

TRANSITION_TIMEOUT_S = 120.0


class SpecEnv(VecEnv):
    """Shape-only env so OnPolicyRunner can build PPO and its storage without a simulator."""

    def __init__(self, cfg, device: str) -> None:
        self.cfg = cfg.env
        self.num_envs = cfg.num_envs
        self.num_actions = cfg.action_dim
        self.max_episode_length = cfg.env["max_episode_length"]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.device = device
        self._obs = TensorDict(
            {cfg.env["obs_group"]: torch.zeros(self.num_envs, cfg.obs_dim, device=device)},
            batch_size=[self.num_envs],
        )

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions):
        raise RuntimeError("the simulation runs on the worker")


class LocalLearnerRunner(OnPolicyRunner):
    """OnPolicyRunner whose rollout phase reads transitions from DDS instead of stepping an env.

    Everything else (PPO construction, update, Logger, save/load and the checkpoint format) is inherited.
    """

    def __init__(self, node, cfg, log_dir: str | None, device: str) -> None:
        super().__init__(SpecEnv(cfg, device), cfg.agent(), log_dir=log_dir, device=device)
        self._node = node
        self._config = cfg
        self._transitions = node.inbox("local_transition")
        node.writer("local_weights")
        node.writer("ctrl")
        self._ctrl = CtrlSender(node)

    def _publish_weights(self, version: int) -> None:
        names, data = state_to_msg({"actor": self.alg._raw_actor, "critic": self.alg._raw_critic})
        sample = msg.LocalWeights(version=version, names=names, data=data)
        self._node.write("local_weights", sample)
        print(f"[learner] pub local_weights v{version} ({fmt_bytes(payload_bytes(sample))})")

    def _receive(self, it: int, step: int):
        t = self._transitions.get(timeout=TRANSITION_TIMEOUT_S)
        if t is None:
            raise TimeoutError(f"no transition {it}/{step} in {TRANSITION_TIMEOUT_S}s")
        if (t.iter, t.step_idx, t.policy_version) != (it, step, it):
            raise RuntimeError(f"expected transition {it}/{step} (policy {it}), got {t.iter}/{t.step_idx} (policy {t.policy_version})")
        print(f"[learner] sub local_transition {it}/{step} policy v{t.policy_version} ({fmt_bytes(payload_bytes(t))})")
        return t

    def _to_transition(self, t) -> tuple[RolloutStorage.Transition, torch.Tensor, torch.Tensor, dict]:
        n, a, d = self._config.num_envs, self._config.action_dim, self.device
        group = self._config.env["obs_group"]
        tr = RolloutStorage.Transition()
        tr.observations = TensorDict(
            {group: to_torch(t.obs, np.float32, (n, self._config.obs_dim), d)}, batch_size=[n]
        )
        tr.actions = to_torch(t.actions, np.float32, (n, a), d)
        tr.rewards = to_torch(t.rewards, np.float32, (n,), d)
        tr.dones = to_torch(t.dones, np.uint8, (n,), d).long()
        tr.values = to_torch(t.values, np.float32, (n, 1), d)
        tr.actions_log_prob = to_torch(t.actions_log_prob, np.float32, (n,), d)
        tr.distribution_params = tuple(to_torch(t.dist_params, np.float32, (t.num_dist_params, n, a), d).unbind(0))
        raw_rewards = to_torch(t.raw_rewards, np.float32, (n,), d)
        return tr, raw_rewards, tr.dones, {"log": log_from_json(t.log_json)}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._publish_weights(self.current_learning_iteration)
        wait_worker(self._node)
        self._ctrl.send(
            "start",
            num_iterations=num_learning_iterations,
            start_iter=self.current_learning_iteration,
            init_at_random_ep_len=init_at_random_ep_len,
        )
        self.alg.train_mode()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        num_steps = self.cfg["num_steps_per_env"]
        group = self._config.env["obs_group"]
        try:
            for it in range(start_it, total_it):
                start = time.time()
                # Rollout: the Worker collects with weights version `it` and streams each step.
                with torch.inference_mode():
                    for step in range(num_steps):
                        t = self._receive(it, step)
                        tr, raw_rewards, dones, extras = self._to_transition(t)
                        self.alg.storage.add_transition(tr)
                        self.logger.process_env_step(raw_rewards, dones, extras, None)
                    last_obs = TensorDict(
                        {group: to_torch(t.last_obs, np.float32, (self._config.num_envs, self._config.obs_dim), self.device)},
                        batch_size=[self._config.num_envs],
                    )
                    stop = time.time()
                    collect_time = stop - start
                    start = stop
                    self.alg.compute_returns(last_obs)

                loss_dict = self.alg.update()
                self._publish_weights(it + 1)
                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it

                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collect_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                )
                if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        finally:
            self._ctrl.stop()

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
