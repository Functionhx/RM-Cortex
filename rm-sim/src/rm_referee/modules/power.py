"""10 Hz chassis power buffer and chassis-energy bookkeeping."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, Team, chassis_energy_mask, chassis_power_mask, slot
from rm_referee.state import GameState


def apply_power(state: GameState, inputs: RuleInputs, dt: float) -> None:
    if abs(dt - constants.REFEREE_DT_S) > 1.0e-9:
        raise ValueError("power must be settled at the official 10 Hz tick")

    active = ~state.done
    power_applicable = chassis_power_mask(state.device).unsqueeze(0)
    energy_applicable = chassis_energy_mask(state.device).unsqueeze(0)
    previously_disabled = state.chassis_disabled_s > 0
    state.chassis_disabled_s.copy_(
        torch.clamp(state.chassis_disabled_s - dt * active[:, None], min=0)
    )

    hero_slots = torch.tensor(
        [slot(Team.RED, Role.HERO), slot(Team.BLUE, Role.HERO)],
        device=state.device,
    )
    role_chassis_off = state.controller_offline.clone()
    role_chassis_off[:, hero_slots] |= state.hero_deployed
    measured = torch.clamp(inputs.chassis_power_w, min=0)
    measured = torch.where(
        active[:, None] & power_applicable & ~previously_disabled & ~role_chassis_off,
        measured,
        torch.zeros_like(measured),
    )

    effective_limit = state.power_limit_w
    effective_limit = torch.where(
        energy_applicable & (state.chassis_energy <= 0),
        torch.full_like(effective_limit, constants.CHASSIS_ENERGY_EMPTY_POWER_W),
        effective_limit,
    )
    boosted_limit = torch.clamp(
        state.power_limit_w * constants.CHASSIS_ENERGY_BOOST_MULTIPLIER,
        max=constants.CHASSIS_POWER_ABSOLUTE_MAX_W,
    )
    effective_limit = torch.where(
        energy_applicable & (state.chassis_energy >= constants.CHASSIS_ENERGY_BOOST_THRESHOLD),
        boosted_limit,
        effective_limit,
    )

    raw_buffer = state.power_buffer_j - (measured - effective_limit) * dt
    can_settle = active[:, None] & power_applicable & ~previously_disabled & ~role_chassis_off
    trigger = can_settle & (raw_buffer <= 0)
    state.power_buffer_j.copy_(
        torch.where(
            can_settle,
            torch.clamp(raw_buffer, min=0, max=constants.POWER_BUFFER_MAX_J),
            state.power_buffer_j,
        )
    )
    state.chassis_disabled_s.copy_(
        torch.where(
            trigger,
            torch.full_like(state.chassis_disabled_s, constants.CHASSIS_POWER_OFF_S),
            state.chassis_disabled_s,
        )
    )

    energy_used = measured * dt
    wireless_delta = torch.clamp(inputs.supercap_input_w - measured, min=0) * dt
    energy_charged = (
        wireless_delta
        * constants.CHASSIS_WIRELESS_CHARGE_MULTIPLIER
        * inputs.in_supply_zone.to(state.dtype)
    )
    state.chassis_energy.copy_(
        torch.where(
            energy_applicable,
            torch.clamp(
                state.chassis_energy - energy_used + energy_charged,
                min=0,
                max=constants.CHASSIS_ENERGY_MAX,
            ),
            state.chassis_energy,
        )
    )
