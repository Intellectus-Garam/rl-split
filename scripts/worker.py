"""Worker (simulation PC): run the IsaacLab env and serve it to the Learner over DDS.

    PYTHONPATH=<path-to-int2DDS>/python .venv/bin/python scripts/worker.py --mode central
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_rl.entrypoints.common import (
    add_common_train_args,
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    set_hydra_args,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from rl_split.config import DEFAULT_CONFIG, Config


def main() -> None:
    parser = argparse.ArgumentParser(description="rl-split worker")
    parser.add_argument("--mode", choices=["central", "local"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    add_common_train_args(
        parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Agent config entry point.", include_distributed=False
    )
    add_launcher_args(parser)
    # Leftover tokens (e.g. physics=...) go to Hydra, which reads sys.argv in resolve_task_config.
    args, hydra_args = setup_preset_cli(parser)
    set_hydra_args(hydra_args)

    cfg = Config(args.config)
    agent = cfg.agent()
    if args.num_envs is not None and args.num_envs != cfg.num_envs:
        parser.error(f"--num_envs must come from the config ({cfg.num_envs}); edit {cfg.path} instead")
    args.task = args.task or cfg.env["task"]
    args.num_envs = cfg.num_envs

    env_cfg, _ = resolve_task_config(args.task, args.agent)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        apply_env_overrides(args, env_cfg)
        env_cfg.seed = agent["seed"] if args.seed is None else args.seed
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])

        obs = env.get_observations()
        if cfg.env["obs_group"] not in obs.keys():
            raise RuntimeError(f"env has no observation group {cfg.env['obs_group']!r}: {list(obs.keys())}")
        if obs[cfg.env["obs_group"]].shape[1] != cfg.obs_dim:
            raise RuntimeError(f"obs dim {obs[cfg.env['obs_group']].shape[1]} != config {cfg.obs_dim}")

        # Imported after the simulation is up, like the IsaacLab entrypoints do with env modules.
        from rl_split.comm.node import Node

        node = Node(cfg.dds, "worker")
        cleanup.callback(node.close)
        if args.mode == "central":
            from rl_split.central.worker import run

            run(env, node, cfg)
        else:
            from rl_split.local.worker import run

            run(env, node, cfg, device=agent["device"])


if __name__ == "__main__":
    main()
