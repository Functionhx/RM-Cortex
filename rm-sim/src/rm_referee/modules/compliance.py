"""Muzzle-velocity and referee-module status rules."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, Team, Weapon, ground_robot_mask, slot, unit_roles
from rm_referee.state import GameState
from rm_referee.tensor_ops import round_half_up


def apply_compliance(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active = (~state.done)[:, None]
    ground = ground_robot_mask(state.device)[None, :]
    state.controller_offline.copy_(inputs.controller_offline)
    state.controller_offline_s.copy_(
        torch.where(
            state.controller_offline & active,
            state.controller_offline_s + dt,
            torch.zeros_like(state.controller_offline_s),
        )
    )
    controller_settlements = torch.floor(
        state.controller_offline_s / constants.MODULE_STATUS_SETTLE_S
    ) - torch.floor(
        torch.clamp(state.controller_offline_s - dt, min=0) / constants.MODULE_STATUS_SETTLE_S
    )
    controller_damage = round_half_up(
        state.max_hp
        * constants.CONTROLLER_OFFLINE_HP_FRACTION_PER_S
        * constants.MODULE_STATUS_SETTLE_S
        * controller_settlements
    )
    controller_damage = torch.where(
        active & ground & state.alive & state.controller_offline,
        torch.minimum(controller_damage, state.hp),
        torch.zeros_like(controller_damage),
    )
    state.hp.sub_(controller_damage)
    events.damage_taken.add_(controller_damage)

    module_active = (
        active
        & ground
        & state.alive
        & ~state.controller_offline
        & (inputs.offline_module_count > 0)
    )
    accumulated = torch.where(
        module_active,
        state.offline_module_accumulator_s + dt,
        torch.zeros_like(state.offline_module_accumulator_s),
    )
    settlements = torch.floor(accumulated / constants.MODULE_STATUS_SETTLE_S).to(torch.long)
    state.offline_module_accumulator_s.copy_(
        accumulated - settlements.to(state.dtype) * constants.MODULE_STATUS_SETTLE_S
    )
    module_damage = (
        settlements * inputs.offline_module_count * int(constants.OFFLINE_MODULE_DAMAGE)
    ).to(state.dtype)
    module_damage = torch.minimum(module_damage, state.hp)
    state.hp.sub_(module_damage)
    events.damage_taken.add_(module_damage)
    state.speed_module_offline.copy_(inputs.speed_module_offline)

    fired = events.shots_fired > 0
    measured = inputs.muzzle_velocity_mps
    limit = state.muzzle_velocity_limit_mps
    roles = unit_roles(state.device)
    is_17 = torch.zeros_like(fired)
    is_17[:, :, Weapon.MM17] = True
    is_42 = torch.zeros_like(fired)
    is_42[:, :, Weapon.MM42] = True
    delta = measured - limit

    short_17 = fired & is_17 & (delta > 0) & (delta < 5)
    long_17 = fired & is_17 & (delta >= 5) & (delta < 10)
    permanent_17 = fired & is_17 & (delta >= 10)

    hero_slots = torch.tensor(
        [slot(Team.RED, Role.HERO), slot(Team.BLUE, Role.HERO)],
        device=state.device,
    )
    deployed_by_unit = torch.zeros_like(state.alive)
    deployed_by_unit[:, hero_slots] = state.hero_deployed
    non_deployed_42 = fired & is_42 & ~deployed_by_unit[:, :, None]
    deployed_42 = fired & is_42 & deployed_by_unit[:, :, None]
    short_42 = non_deployed_42 & (measured > limit) & (measured <= 1.1 * limit)
    long_42 = non_deployed_42 & (measured > 1.1 * limit) & (measured <= 1.2 * limit)
    permanent_42 = non_deployed_42 & (measured > 1.2 * limit)
    short_42 |= deployed_42 & (measured > limit) & (measured <= 18.0)
    permanent_42 |= deployed_42 & (measured > 18.0)

    short = short_17 | short_42
    long = long_17 | long_42
    permanent = permanent_17 | permanent_42
    state.velocity_lock_s.copy_(
        torch.where(
            short,
            torch.maximum(
                state.velocity_lock_s,
                torch.full_like(state.velocity_lock_s, constants.VELOCITY_LOCK_SHORT_S),
            ),
            state.velocity_lock_s,
        )
    )
    state.velocity_lock_s.copy_(
        torch.where(
            long,
            torch.maximum(
                state.velocity_lock_s,
                torch.full_like(state.velocity_lock_s, constants.VELOCITY_LOCK_LONG_S),
            ),
            state.velocity_lock_s,
        )
    )
    state.velocity_permanent_lock |= permanent

    applicable_roles = (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
        | (roles == Role.SENTRY)
    )
    state.speed_module_offline &= applicable_roles[None, :, None]
