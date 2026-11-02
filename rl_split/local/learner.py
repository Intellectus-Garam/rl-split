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
from rl_split.comm.convert import cat_torch, fmt_bytes, merge_logs, payload_bytes, state_to_msg
from rl_split.comm.node import per_worker
from rl_split.session import CtrlSender, wait_workers

TRANSITION_TIMEOUT_S = 120.0


class SpecEnv(VecEnv):
    """Shape-only env so OnPolicyRunner can build PPO and its storage without a simulator."""

    def __init__(self, cfg, num_envs: int, device: str) -> None:
        self.cfg = cfg.env
        self.num_envs = num_envs
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

    The Workers' envs are stacked in worker_id order into one batch of num_workers * env.num_envs.
    Everything else (PPO construction, update, Logger, save/load and the checkpoint format) is inherited.
    """

    def __init__(self, node, cfg, num_workers: int, agent: dict, log_dir: str | None, device: str) -> None:
        super().__init__(SpecEnv(cfg, num_workers * cfg.num_envs, device), agent, log_dir=log_dir, device=device)
        self._node = node
        self._config = cfg
        self._num_workers = num_workers
        self._transitions = [node.inbox(per_worker("local_transition", w)) for w in range(num_workers)]
        node.writer("local_weights", num_readers=num_workers)
        node.writer("ctrl", num_readers=num_workers)
        self._ctrl = CtrlSender(node)

    def _publish_weights(self, version: int) -> None:
        names, data = state_to_msg({"actor": self.alg._raw_actor, "critic": self.alg._raw_critic})
        sample = msg.LocalWeights(version=version, names=names, data=data)
        self._node.write("local_weights", sample)
        print(f"[learner] pub local_weights v{version} ({fmt_bytes(payload_bytes(sample))})")

    def start(self, num_iterations: int, init_at_random_ep_len: bool) -> None:
        """Publish the first weights and start every Worker."""
        self._publish_weights(self.current_learning_iteration)
        wait_workers(self._node, self._num_workers)
        self._ctrl.send(
            "start",
            num_iterations=num_iterations,
            start_iter=self.current_learning_iteration,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def _receive(self, it: int, step: int) -> list:
        samples = []
        for w, inbox in enumerate(self._transitions):
            t = inbox.get(timeout=TRANSITION_TIMEOUT_S)
            if t is None:
                raise TimeoutError(f"no transition {it}/{step} from worker {w} in {TRANSITION_TIMEOUT_S}s")
            if (t.worker_id, t.iter, t.step_idx, t.policy_version) != (w, it, step, it):
                raise RuntimeError(
                    f"expected transition {it}/{step} (policy {it}) from worker {w}, "
                    f"got {t.iter}/{t.step_idx} (policy {t.policy_version}) from worker {t.worker_id}"
                )
            samples.append(t)
        self._node.check_single_worker()
        size = sum(payload_bytes(t) for t in samples)
        print(f"[learner] sub local_transition {it}/{step} policy v{it} from {len(samples)} worker(s) ({fmt_bytes(size)})")
        return samples

    def _cat(self, samples, name: str, dtype, shape, axis: int = 0) -> torch.Tensor:
        """Field `name` of every Worker stacked on the env axis. shape(n) gives one Worker's shape for n envs."""
        fields = [getattr(t, name) for t in samples]
        return cat_torch(fields, dtype, [shape(self._config.num_envs)] * len(fields), self.device, axis)

    def _to_transition(self, samples) -> tuple[RolloutStorage.Transition, torch.Tensor, torch.Tensor, dict]:
        n, a, obs_dim = self.env.num_envs, self._config.action_dim, self._config.obs_dim
        k = samples[0].num_dist_params
        group = self._config.env["obs_group"]
        tr = RolloutStorage.Transition()
        tr.observations = TensorDict({group: self._cat(samples, "obs", np.float32, lambda m: (m, obs_dim))}, batch_size=[n])
        tr.actions = self._cat(samples, "actions", np.float32, lambda m: (m, a))
        tr.rewards = self._cat(samples, "rewards", np.float32, lambda m: (m,))
        tr.dones = self._cat(samples, "dones", np.uint8, lambda m: (m,)).long()
        tr.values = self._cat(samples, "values", np.float32, lambda m: (m, 1))
        tr.actions_log_prob = self._cat(samples, "actions_log_prob", np.float32, lambda m: (m,))
        tr.distribution_params = tuple(self._cat(samples, "dist_params", np.float32, lambda m: (k, m, a), axis=1).unbind(0))
        raw_rewards = self._cat(samples, "raw_rewards", np.float32, lambda m: (m,))
        log = merge_logs([t.log_json for t in samples])
        return tr, raw_rewards, tr.dones, {"log": log}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self.start(num_learning_iterations, init_at_random_ep_len)
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
                        samples = self._receive(it, step)
                        tr, raw_rewards, dones, extras = self._to_transition(samples)
                        self.alg.storage.add_transition(tr)
                        self.logger.process_env_step(raw_rewards, dones, extras, None)
                    obs_dim = self._config.obs_dim
                    last_obs = TensorDict(
                        {group: self._cat(samples, "last_obs", np.float32, lambda m: (m, obs_dim))},
                        batch_size=[self.env.num_envs],
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
