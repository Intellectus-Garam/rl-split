"""Worker side of central inference: step the env with actions received from the Learner."""

from __future__ import annotations

import time

import numpy as np
import torch

from rl_split.comm import types as msg
from rl_split.comm.convert import flat, fmt_bytes, log_to_json, payload_bytes, to_torch
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
    sample = _obs_msg(0, env.get_observations()[obs_group])
    node.write("central_obs", sample)
    print(f"[worker] started (central), sent obs 0 ({fmt_bytes(payload_bytes(sample))})")

    step_id = 0
    t0 = time.perf_counter()  # reset only once an action arrives, so waits longer than the 1s poll are counted fully
    while True:
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
        sample = _obs_msg(step_id, obs[obs_group], rewards, dones, extras["time_outs"], log_to_json(extras))
        node.write("central_obs", sample)
        print(
            f"[worker] step {step_id}: waited {t1 - t0:.3f}s, received action {step_id - 1} ({fmt_bytes(payload_bytes(action))}), "
            f"env+send {time.perf_counter() - t1:.3f}s, sent obs {step_id} ({fmt_bytes(payload_bytes(sample))})"
        )
        t0 = time.perf_counter()
    print("[worker] received ctrl 'stop'")
