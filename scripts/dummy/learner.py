"""Dummy Learner: no training, same DDS pub/sub scenario as scripts/learner.py.

Uses the real Learner pieces (RemoteEnv for central, LocalLearnerRunner's publish/receive
for local) but skips policy inference and the PPO update (a sleep of --update_time stands in for
it): central sends zero actions every step and waits for each obs; local publishes
weights, waits for the iteration's transitions, and publishes the next weights.
Message sizes and order match a real run.
Use it with scripts/dummy/worker.py to test DDS without training.

    PYTHONPATH=<path-to-int2DDS>/python python scripts/dummy/learner.py --mode central [--num_workers 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rl_split import timestamp

timestamp.install()

import torch

from rl_split.comm.node import Node
from rl_split.config import DEFAULT_CONFIG, Config
from rl_split.session import CtrlSender, exit_on_signals


def run_central(node, cfg, num_workers: int, ctrl, device: str, num_iterations: int, update_time: float) -> None:
    from rl_split.central.remote_env import RemoteEnv

    env = RemoteEnv(node, cfg, num_workers, ctrl, device)
    env.start(init_at_random_ep_len=False)
    actions = torch.zeros(env.num_envs, env.num_actions, device=device)
    for _ in range(num_iterations):
        for _ in range(cfg.num_steps):
            env.step(actions)
        time.sleep(update_time)  # instead of compute_returns + alg.update()


def run_local(node, cfg, num_workers: int, ctrl, device: str, num_iterations: int, update_time: float) -> None:
    from rl_split.local.learner import LocalLearnerRunner

    # Builds PPO only so the weights message has the real names and size; it is never updated.
    agent = cfg.agent()
    agent["device"] = device
    runner = LocalLearnerRunner(node, cfg, num_workers, ctrl, agent, log_dir=None, device=device)
    runner.start(num_iterations, init_at_random_ep_len=False)
    for it in range(num_iterations):
        for step in range(cfg.num_steps):
            runner._to_transition(runner._receive(it, step))  # also checks that the Workers' fields stack
        time.sleep(update_time)  # instead of compute_returns + alg.update()
        runner._publish_weights(it + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-split dummy learner (DDS only)")
    parser.add_argument("--mode", choices=["central", "local"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None, help="Tensor device (default: agent.device from the config).")
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--update_time", type=float, default=2.0, help="Seconds to sleep in place of the PPO update.")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Workers to wait for; they must use --worker_id 0 .. num_workers-1."
    )
    args = parser.parse_args()

    cfg = Config(args.config)
    device = args.device or cfg.agent()["device"]

    exit_on_signals()
    node = Node(cfg.dds, "learner")
    ctrl = CtrlSender(node, args.num_workers)
    try:
        if args.mode == "central":
            run_central(node, cfg, args.num_workers, ctrl, device, args.max_iterations, args.update_time)
        else:
            run_local(node, cfg, args.num_workers, ctrl, device, args.max_iterations, args.update_time)
    finally:
        ctrl.stop()
        node.close()


if __name__ == "__main__":
    main()
