"""Explicit random samples shared across simulation backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RandomTape:
    """A deterministic, finite stream of per-environment uniform samples.

    Backend parity tests pass the same tensor to both backends. This avoids
    assuming that CPU and CUDA generators produce identical streams.
    """

    values: Tensor
    cursor: int = 0

    @classmethod
    def from_seed(
        cls,
        num_envs: int,
        num_draws: int,
        *,
        seed: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "RandomTape":
        device = torch.device(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        values = torch.rand(
            (num_envs, num_draws),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return cls(values)

    def take(self, count: int) -> Tensor:
        if count < 0:
            raise ValueError("count cannot be negative")
        end = self.cursor + count
        if end > self.values.shape[1]:
            raise RuntimeError("random tape exhausted")
        result = self.values[:, self.cursor : end]
        self.cursor = end
        return result

    def clone(self) -> "RandomTape":
        return RandomTape(self.values.clone(), self.cursor)
