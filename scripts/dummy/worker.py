"""Dummy Worker: no simulation, same DDS pub/sub scenario as scripts/worker.py.

Runs the real Worker loops (rl_split.central.worker / rl_split.local.worker) against an
env that returns zero tensors of the configured shapes, so every message has the real
size and order: central streams obs every step and waits for each action; local waits
for weights, streams the iteration's transitions, and waits for the next weights.
Use it with scripts/dummy/learner.py to test DDS without IsaacLab or training.

    PYTHONPATH=<path-to-int2DDS>/python python scripts/dummy/worker.py --mode central
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
from tensordict import TensorDict

from rl_split.comm.node import Node
from rl_split.config import DEFAULT_CONFIG, Config


class DummyEnv:
    """The parts of RslRlVecEnvWrapper the Worker loops use, returning zeros."""

    def __init__(self, cfg, device: str, step_time: float) -> None:
        n = cfg.num_envs
        self.num_envs = n
        self.num_actions = cfg.action_dim
        self.max_episode_length = cfg.env["max_episode_length"]
        self.device = device
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
        self._step_time = step_time
        self._obs = TensorDict({cfg.env["obs_group"]: torch.zeros(n, cfg.obs_dim, device=device)}, batch_size=[n])
        self._rewards = torch.zeros(n, device=device)
        self._dones = torch.zeros(n, dtype=torch.long, device=device)
        self._extras = {"time_outs": torch.zeros(n, dtype=torch.bool, device=device), "log": {}}

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor):
        if self._step_time > 0:
            time.sleep(self._step_time)
        return self._obs, self._rewards, self._dones, self._extras


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-split dummy worker (DDS only)")
    parser.add_argument("--mode", choices=["central", "local"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None, help="Tensor device (default: agent.device from the config).")
    parser.add_argument("--step_time", type=float, default=0.0, help="Seconds to sleep per env step, to mimic the simulation.")
    args = parser.parse_args()

    cfg = Config(args.config)
    device = args.device or cfg.agent()["device"]
    env = DummyEnv(cfg, device, args.step_time)

    node = Node(cfg.dds, "worker")
    try:
        if args.mode == "central":
            from rl_split.central.worker import run

            run(env, node, cfg)
        else:
            from rl_split.local.worker import run

            run(env, node, cfg, device=device)
    finally:
        node.close()


if __name__ == "__main__":
    main()
