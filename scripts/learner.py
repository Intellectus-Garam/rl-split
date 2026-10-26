"""Learner (training PC): PPO with rsl_rl, data from the Worker over DDS. Does not import IsaacLab.

    PYTHONPATH=<path-to-int2DDS>/python python scripts/learner.py --mode central
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rsl_rl.runners import OnPolicyRunner

from rl_split.comm.node import Node
from rl_split.config import DEFAULT_CONFIG, Config


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-split learner")
    parser.add_argument("--mode", choices=["central", "local"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None, help="Training device (default: agent.device from the config).")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--log_root", default="logs/rsl_rl")
    args = parser.parse_args()

    cfg = Config(args.config)
    agent = cfg.agent()
    device = args.device or agent["device"]
    num_iterations = args.max_iterations or agent["max_iterations"]
    torch.manual_seed(agent["seed"])

    # Same layout as IsaacLab (logs/rsl_rl/<experiment>/<time>) so `play` finds the checkpoints.
    log_dir = Path(args.log_root) / agent["experiment_name"] / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.mode}"
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, log_dir / "params" / cfg.path.name)
    print(f"[learner] logging to {log_dir}")

    node = Node(cfg.dds, "learner")
    try:
        if args.mode == "central":
            from rl_split.central.remote_env import RemoteEnv

            env = RemoteEnv(node, cfg, device)
            try:
                env.start(init_at_random_ep_len=agent["init_at_random_ep_len"])
                agent["device"] = device
                runner = OnPolicyRunner(env, agent, log_dir=str(log_dir), device=device)
                # Episode lengths were already randomized on the Worker in start().
                runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=False)
            finally:
                env.close()
        else:
            from rl_split.local.learner import LocalLearnerRunner

            runner = LocalLearnerRunner(node, cfg, log_dir=str(log_dir), device=device)
            runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=agent["init_at_random_ep_len"])
    finally:
        node.close()


if __name__ == "__main__":
    main()
