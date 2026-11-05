"""Learner (training PC): PPO with rsl_rl, data from the Workers over DDS. Does not import IsaacLab.

    PYTHONPATH=<path-to-int2DDS>/python python scripts/learner.py --mode central [--num_workers 2]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_split import timestamp

timestamp.install()

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from rl_split.comm.node import Node
from rl_split.config import DEFAULT_CONFIG, Config
from rl_split.session import CtrlSender, exit_on_signals


def set_mini_batches(agent: dict, cfg: Config, num_workers: int, override: int | None) -> dict:
    """Set agent num_mini_batches for num_workers Workers and return what was used (also saved next to the config)."""
    total_envs = num_workers * cfg.num_envs
    num = override if override is not None else cfg.scaled_mini_batches(total_envs)
    agent["algorithm"]["num_mini_batches"] = num
    batch = total_envs * cfg.num_steps
    return {
        "num_workers": num_workers,
        "num_envs_per_worker": cfg.num_envs,
        "total_envs": total_envs,
        "num_mini_batches": num,
        "num_mini_batches_source": "--num_mini_batches" if override is not None else "scaled from config",
        "mini_batch_size": batch // num,
    }


def latest_checkpoint(log_dir: Path) -> tuple[Path, int] | None:
    """The model_<n>.pt with the largest n in log_dir, and n (the last finished iteration)."""
    found = [(p, int(p.stem.removeprefix("model_"))) for p in log_dir.glob("model_*.pt")]
    return max(found, key=lambda f: f[1]) if found else None


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-split learner")
    parser.add_argument("--mode", choices=["central", "local"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None, help="Training device (default: agent.device from the config).")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Workers to wait for; they must use --worker_id 0 .. num_workers-1."
    )
    parser.add_argument(
        "--num_mini_batches",
        type=int,
        default=None,
        help="PPO mini-batches per epoch (default: the config's value times num_workers, "
        "which keeps the config's mini-batch size).",
    )
    parser.add_argument("--log_root", default="logs/rsl_rl")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Fixed log directory for this run (default: <log_root>/<experiment>/<time>_<mode>). "
        "Pass the same path on restart to keep logging and checkpoints in one place.",
    )
    args = parser.parse_args()

    cfg = Config(args.config)
    agent = cfg.agent()
    device = args.device or agent["device"]
    num_iterations = args.max_iterations or agent["max_iterations"]
    torch.manual_seed(agent["seed"])

    # Same layout as IsaacLab (logs/rsl_rl/<experiment>/<time>) so `play` finds the checkpoints.
    if args.run_dir:
        log_dir = Path(args.run_dir)
    else:
        log_dir = Path(args.log_root) / agent["experiment_name"] / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.mode}"
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.path, log_dir / "params" / cfg.path.name)
    run_info = set_mini_batches(agent, cfg, args.num_workers, args.num_mini_batches)
    (log_dir / "params" / "run.yaml").write_text(yaml.safe_dump(run_info, sort_keys=False))
    print(f"[learner] logging to {log_dir}")
    print(f"[learner] {run_info}")

    # Resume from the latest checkpoint in log_dir (only possible with --run_dir) and train the rest of num_iterations.
    checkpoint = latest_checkpoint(log_dir)
    start_it = checkpoint[1] + 1 if checkpoint else 0
    remaining = num_iterations - start_it
    if remaining <= 0:
        print(f"[learner] {checkpoint[0]} already covers {num_iterations} iterations, nothing to do")
        return

    exit_on_signals()
    node = Node(cfg.dds, "learner")
    ctrl = CtrlSender(node, args.num_workers)
    # Whatever ends the Learner (end of training, an error, Ctrl+C, SIGTERM/SIGHUP), the Workers get "stop".
    try:
        agent["device"] = device
        if args.mode == "central":
            from rl_split.central.remote_env import RemoteEnv

            env = RemoteEnv(node, cfg, args.num_workers, ctrl, device)
            env.start(init_at_random_ep_len=agent["init_at_random_ep_len"])
            runner = OnPolicyRunner(env, agent, log_dir=str(log_dir), device=device)
            # Episode lengths were already randomized on the Worker in start().
            init_at_random_ep_len = False
        else:
            from rl_split.local.learner import LocalLearnerRunner

            runner = LocalLearnerRunner(node, cfg, args.num_workers, ctrl, agent, log_dir=str(log_dir), device=device)
            init_at_random_ep_len = agent["init_at_random_ep_len"]
        if checkpoint:
            runner.load(str(checkpoint[0]), map_location=device)
            # The checkpoint's iteration is finished, so continue from the next one.
            runner.current_learning_iteration = start_it
            print(f"[learner] resumed from {checkpoint[0]}, iterations {start_it} .. {num_iterations - 1}")
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=init_at_random_ep_len)
    finally:
        ctrl.stop()
        node.close()


if __name__ == "__main__":
    main()
