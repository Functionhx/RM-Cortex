"""Reproducible configuration for the pure-Torch MAPPO baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class MAPPOConfig:
    """Training hyperparameters with conservative, runnable defaults."""

    seed: int = 7
    device: str = "auto"
    num_envs: int = 64
    rollout_steps: int = 64
    total_updates: int = 200
    update_epochs: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_dim: int = 128
    role_embedding_dim: int = 16
    checkpoint_interval: int = 10
    output_dir: str = "runs/phase1_mappo"

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self) -> None:
        integer_positive = {
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "total_updates": self.total_updates,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "hidden_dim": self.hidden_dim,
            "role_embedding_dim": self.role_embedding_dim,
        }
        for name, value in integer_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in [0, 1)")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "MAPPOConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("MAPPO config must be a JSON object")
        config = cls(**values)
        config.validate()
        return config
