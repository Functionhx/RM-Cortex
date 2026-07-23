"""Small tensor helpers with rule-specific numerical semantics."""

from __future__ import annotations

import torch
from torch import Tensor


def round_half_up(value: Tensor) -> Tensor:
    """Round non-negative rule values using conventional half-up semantics."""

    return torch.floor(value + 0.5)


def gather_unit(values: Tensor, unit_index: Tensor) -> Tensor:
    """Gather ``values[env, unit]`` with an arbitrary per-env index shape."""

    if values.ndim != 2:
        raise ValueError("gather_unit expects [env, unit] values")
    return torch.gather(values, 1, unit_index.reshape(values.shape[0], -1)).reshape(
        unit_index.shape
    )
