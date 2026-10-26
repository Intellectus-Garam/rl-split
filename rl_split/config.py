"""Shared run configuration (configs/*.yaml)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "go2_rough.yaml"


class Config:
    """Parsed config file."""

    def __init__(self, path: str | Path = DEFAULT_CONFIG) -> None:
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text())
        self.env: dict = data["env"]
        self.dds: dict = data["dds"]
        self._agent: dict = data["agent"]

    def agent(self) -> dict:
        """A fresh copy of the rsl_rl runner config (rsl_rl mutates it in place)."""
        return copy.deepcopy(self._agent)

    @property
    def num_envs(self) -> int:
        return self.env["num_envs"]

    @property
    def obs_dim(self) -> int:
        return self.env["obs_dim"]

    @property
    def action_dim(self) -> int:
        return self.env["action_dim"]

    @property
    def num_steps(self) -> int:
        return self._agent["num_steps_per_env"]
