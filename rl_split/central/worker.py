"""Worker side of central inference: step the env with actions received from the Learner."""

from __future__ import annotations

import time

import numpy as np
import torch

from rl_split.comm import types as msg
from rl_split.comm.convert import flat, log_to_json, to_torch
from rl_split.session import stop_requested, wait_start


def _obs_msg(step_id: int, obs, rewards=None, dones=None, time_outs=None, log_json: str = "") -> msg.CentralObs:
    # The initial observation (step 0) carries zeros so the Learner can parse every message the same way.
    zeros = torch.zeros(obs.shape[0])
    return msg.CentralObs(
        step_id=step_id,
        obs=flat(obs),
        rewards=flat(zeros if rewards is None else rewards),
        dones=flat(zeros if dones is None else dones, np.uint8),
        time_outs=flat(zeros if time_outs is None else time_outs, np.uint8),
        log_json=log_json,
    )


def run(env, node, cfg) -> None:
    obs_group = cfg.env["obs_group"]
    ctrl = node.inbox("ctrl")
    actions_in = node.inbox("central_action")
    node.writer("central_obs")

    args = wait_start(ctrl)
    node.wait_matched(timeout=60.0)
    if args.get("init_at_random_ep_len"):
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    node.write("central_obs", _obs_msg(0, env.get_observations()[obs_group]))
    print("[worker] started (central)")

    step_id = 0
    t_env = t_wait = 0.0
    while True:
        t0 = time.perf_counter()
        action = actions_in.get(timeout=1.0)
        if action is None:
            if stop_requested(ctrl):
                break
            continue
        if action.step_id != step_id:
            raise RuntimeError(f"expected action {step_id}, got {action.step_id}")
        t1 = time.perf_counter()
        actions = to_torch(action.actions, np.float32, (env.num_envs, env.num_actions), env.device)
        with torch.inference_mode():
            obs, rewards, dones, extras = env.step(actions)
        step_id += 1
        node.write(
            "central_obs",
            _obs_msg(step_id, obs[obs_group], rewards, dones, extras["time_outs"], log_to_json(extras)),
        )
        t_wait += t1 - t0
        t_env += time.perf_counter() - t1
        if step_id % cfg.num_steps == 0:
            print(f"[worker] step {step_id}: env+send {t_env:.2f}s, waiting for actions {t_wait:.2f}s")
            t_env = t_wait = 0.0
    print("[worker] stop received")
