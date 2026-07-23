"""10 Hz launcher heat state machine."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.schema import HeatLock
from rm_referee.state import GameState


def apply_heat(state: GameState, dt: float) -> None:
    if abs(dt - constants.REFEREE_DT_S) > 1.0e-9:
        raise ValueError("heat must be settled at the official 10 Hz tick")

    applicable = state.heat_limit > 0
    q2 = state.heat_limit + torch.tensor(
        constants.HEAT_Q2_OFFSET,
        device=state.device,
        dtype=state.dtype,
    )
    permanent = applicable & (state.heat >= q2)
    temporary = applicable & (state.heat > state.heat_limit) & ~permanent

    state.heat_lock.copy_(
        torch.where(
            permanent,
            torch.full_like(state.heat_lock, HeatLock.PERMANENT),
            state.heat_lock,
        )
    )
    state.heat_lock.copy_(
        torch.where(
            temporary & (state.heat_lock != HeatLock.PERMANENT),
            torch.full_like(state.heat_lock, HeatLock.TEMPORARY),
            state.heat_lock,
        )
    )

    state.heat.sub_(state.cooling_per_s * dt).clamp_(min=0)
    cooled_to_zero = (state.heat <= 0) & (state.heat_lock == HeatLock.TEMPORARY)
    state.heat_lock.copy_(
        torch.where(
            cooled_to_zero,
            torch.full_like(state.heat_lock, HeatLock.NONE),
            state.heat_lock,
        )
    )
    state.video_disabled.copy_((state.heat_lock != HeatLock.NONE).any(dim=-1))
