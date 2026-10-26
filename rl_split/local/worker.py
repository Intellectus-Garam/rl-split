"""Worker side of local inference: collect rollouts with a policy copy and stream them to the Learner."""

from __future__ import annotations

import time

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from rl_split.comm import types as msg
from rl_split.comm.convert import flat, load_state_from_msg, log_to_json
from rl_split.comm.sender import Sender
from rl_split.session import stop_requested, wait_start

WEIGHTS_TIMEOUT_S = 600.0


class StreamingStorage:
    """Stands in for RolloutStorage on the Worker: PPO.process_env_step hands each
    transition here, and it is published instead of stored."""

    def __init__(self, sender: Sender, obs_group: str) -> None:
        self._sender = sender
        self._obs_group = obs_group
        self.iter = 0
        self.step_idx = 0
        self.policy_version = 0
        self._raw_rewards = None
        self._log_json = ""
        self._last_obs = None

    def stage(self, raw_rewards, extras, last_obs=None) -> None:
        """Data PPO does not keep on the transition: env rewards before bootstrapping, logs, bootstrap obs."""
        self._raw_rewards = flat(raw_rewards)
        self._log_json = log_to_json(extras)
        self._last_obs = None if last_obs is None else flat(last_obs[self._obs_group])

    def add_transition(self, t) -> None:
        # Copy to host now: the env reuses its observation buffers on the next step.
        sample = dict(
            iter=self.iter,
            step_idx=self.step_idx,
            policy_version=self.policy_version,
            obs=flat(t.observations[self._obs_group]),
            actions=flat(t.actions),
            rewards=flat(t.rewards),
            raw_rewards=self._raw_rewards,
            dones=flat(t.dones, np.uint8),
            values=flat(t.values),
            actions_log_prob=flat(t.actions_log_prob),
            dist_params=np.concatenate([flat(p) for p in t.distribution_params]),
            num_dist_params=len(t.distribution_params),
            log_json=self._log_json,
            last_obs=np.zeros(0, np.float32) if self._last_obs is None else self._last_obs,
        )
        self._sender.put("local_transition", lambda: msg.LocalTransition(**sample))
        self.step_idx += 1

    def clear(self) -> None:
        self.step_idx = 0


def _wait_weights(inbox, alg, version: int) -> None:
    deadline = time.monotonic() + WEIGHTS_TIMEOUT_S
    while True:
        w = inbox.get(timeout=max(0.0, deadline - time.monotonic()))
        if w is None:
            raise TimeoutError(f"no weights version {version} in {WEIGHTS_TIMEOUT_S}s")
        if w.version == version:
            load_state_from_msg({"actor": alg._raw_actor, "critic": alg._raw_critic}, w.names, w.data)
            return
        if w.version > version:
            raise RuntimeError(f"expected weights {version}, got {w.version}")


def run(env, node, cfg, device: str) -> None:
    obs_group = cfg.env["obs_group"]
    ctrl = node.inbox("ctrl")
    weights_in = node.inbox("local_weights")
    node.writer("local_transition")

    args = wait_start(ctrl)
    node.wait_matched(timeout=60.0)
    num_iterations = args["num_iterations"]
    start_iter = args.get("start_iter", 0)

    # Same construction as OnPolicyRunner, so the Transition fields match the Learner's storage.
    agent_cfg = cfg.agent()
    agent_cfg["multi_gpu"] = None
    obs = env.get_observations().to(device)
    alg = PPO.construct_algorithm(obs, env, agent_cfg, device)
    sender = Sender(node)
    storage = StreamingStorage(sender, obs_group)
    alg.storage = storage
    alg.train_mode()

    if args.get("init_at_random_ep_len"):
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = env.get_observations().to(device)
    print(f"[worker] started (local), {num_iterations} iterations")

    for it in range(start_iter, start_iter + num_iterations):
        t0 = time.perf_counter()
        _wait_weights(weights_in, alg, version=it)
        t1 = time.perf_counter()
        storage.iter, storage.policy_version = it, it
        storage.clear()
        with torch.inference_mode():
            for step in range(cfg.num_steps):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                storage.stage(rewards, extras, last_obs=obs if step == cfg.num_steps - 1 else None)
                alg.process_env_step(obs, rewards, dones, extras)
        t2 = time.perf_counter()
        print(f"[worker] iter {it}: waited for weights {t1 - t0:.2f}s, collected {t2 - t1:.2f}s")
        if stop_requested(ctrl):
            break

    sender.flush()
    print("[worker] all transitions sent, waiting for stop")
    while not stop_requested(ctrl):
        time.sleep(0.1)
    sender.close()
    print("[worker] stop received")
