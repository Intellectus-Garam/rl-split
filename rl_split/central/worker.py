"""Worker side of central inference: step the env with actions received from the Learner."""

from __future__ import annotations

import numpy as np
import torch

from rl_split.comm import types as msg
from rl_split.comm.convert import flat, fmt_bytes, log_to_json, payload_bytes, to_torch
from rl_split.comm.node import per_worker
from rl_split.session import StopRequested, check_stop, wait_start


def _obs_msg(
    worker_id: int, step_id: int, obs, rewards=None, dones=None, time_outs=None, log_json: str = ""
) -> msg.CentralObs:
    # The initial observation (step 0) carries zeros so the Learner can parse every message the same way.
    zeros = torch.zeros(obs.shape[0])
    return msg.CentralObs(
        worker_id=worker_id,
        step_id=step_id,
        obs=flat(obs),
        rewards=flat(zeros if rewards is None else rewards),
        dones=flat(zeros if dones is None else dones, np.uint8),
        time_outs=flat(zeros if time_outs is None else time_outs, np.uint8),
        log_json=log_json,
    )


def run(env, node, cfg, worker_id: int, keep_alive: bool = False) -> None:
    """Serve the env until ctrl "stop", which may come at any point (also before "start").

    keep_alive: after "stop", wait for the next "start" instead of returning.
    """
    ctrl = node.inbox("ctrl")
    actions_in = node.inbox(per_worker("central_action", worker_id))
    node.writer(per_worker("central_obs", worker_id))
    while True:
        try:
            _serve(env, node, cfg, worker_id, ctrl, actions_in)
        except StopRequested:
            print("[worker] stopped by the learner")
        if not keep_alive:
            return
        # Actions the stopped Learner sent after the last step would be read as the next session's.
        print(f"[worker] dropped {actions_in.drain()} leftover action(s), waiting for the next learner")


def _serve(env, node, cfg, worker_id: int, ctrl, actions_in) -> None:
    obs_group = cfg.env["obs_group"]
    obs_topic = per_worker("central_obs", worker_id)

    args = wait_start(ctrl)
    node.wait_matched(timeout=60.0)
    if args.get("init_at_random_ep_len"):
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    print(f"[worker] started (central), worker {worker_id}")
    sample = _obs_msg(worker_id, 0, env.get_observations()[obs_group])
    node.write(obs_topic, sample)
    print(f"[worker] pub central_obs step 0 ({fmt_bytes(payload_bytes(sample))}), initial observation")

    step_id = 0
    while True:
        action = actions_in.get(timeout=1.0)
        check_stop(ctrl)
        if action is None:
            continue
        if (action.worker_id, action.step_id) != (worker_id, step_id):
            raise RuntimeError(
                f"expected action {step_id} for worker {worker_id}, got {action.step_id} for worker {action.worker_id}"
            )
        print(f"[worker] sub central_action step {action.step_id} ({fmt_bytes(payload_bytes(action))})")
        actions = to_torch(action.actions, np.float32, (env.num_envs, env.num_actions), env.device)
        with torch.inference_mode():
            obs, rewards, dones, extras = env.step(actions)
        step_id += 1
        sample = _obs_msg(worker_id, step_id, obs[obs_group], rewards, dones, extras["time_outs"], log_to_json(extras))
        node.write(obs_topic, sample)
        print(f"[worker] pub central_obs step {step_id} ({fmt_bytes(payload_bytes(sample))})")
