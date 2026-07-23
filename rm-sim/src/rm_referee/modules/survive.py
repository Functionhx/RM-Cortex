"""Out-of-combat tracking, healing, and both respawn paths."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import HeatLock, Role, Team, ground_robot_mask, slot, unit_roles
from rm_referee.state import GameState
from rm_referee.tensor_ops import round_half_up


def _team_affordable(
    requested: torch.Tensor,
    cost: torch.Tensor,
    coin: torch.Tensor,
) -> torch.Tensor:
    shaped_request = requested.view(
        requested.shape[0],
        constants.TEAM_COUNT,
        constants.ROLES_PER_TEAM,
    )
    shaped_cost = cost.view_as(shaped_request)
    running_cost = torch.cumsum(
        shaped_request.to(torch.long) * shaped_cost,
        dim=-1,
    )
    return (shaped_request & (running_cost <= coin[:, :, None])).reshape_as(requested)


def apply_survive(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active = ~state.done
    roles = unit_roles(state.device)
    ground = ground_robot_mask(state.device).unsqueeze(0)
    alive_before = state.alive.clone()
    damageable = ground | (((roles == Role.BASE) | (roles == Role.OUTPOST))[None, :])
    new_death = alive_before & (state.hp <= 0) & damageable
    new_ground_death = new_death & ground
    state.alive &= ~new_death
    events.deaths |= new_death

    damage_to_target = events.damage_by_source_target
    best_source_damage, best_source = damage_to_target.max(dim=1)
    events.kill_source.copy_(
        torch.where(
            new_death & (best_source_damage > 0),
            best_source,
            events.kill_source,
        )
    )

    state.heat.copy_(torch.where(new_ground_death[:, :, None], 0.0, state.heat))
    temporary_lock = new_ground_death[:, :, None] & (state.heat_lock == HeatLock.TEMPORARY)
    state.heat_lock.copy_(
        torch.where(
            temporary_lock,
            torch.full_like(state.heat_lock, HeatLock.NONE),
            state.heat_lock,
        )
    )
    state.power_buffer_j.copy_(
        torch.where(
            new_ground_death,
            torch.full_like(state.power_buffer_j, constants.POWER_BUFFER_MAX_J),
            state.power_buffer_j,
        )
    )
    required = round_half_up(
        constants.RESPAWN_BASE_PROGRESS
        + state.elapsed_s[:, None] / constants.RESPAWN_ELAPSED_DIVISOR
        + constants.RESPAWN_IMMEDIATE_PENALTY * state.immediate_respawn_count.to(state.dtype)
    )
    state.respawn_required.copy_(torch.where(new_ground_death, required, state.respawn_required))
    state.respawn_progress.copy_(
        torch.where(
            new_ground_death,
            torch.zeros_like(state.respawn_progress),
            state.respawn_progress,
        )
    )
    state.weak &= ~new_ground_death
    state.invulnerable_s.copy_(
        torch.where(
            new_ground_death,
            torch.zeros_like(state.invulnerable_s),
            state.invulnerable_s,
        )
    )
    state.remote_heal_active &= ~new_ground_death
    state.remote_heal_progress_s.copy_(
        torch.where(new_ground_death, 0.0, state.remote_heal_progress_s)
    )

    state.dead_s.copy_(
        torch.where(
            active[:, None] & ~state.alive & ground,
            state.dead_s + dt,
            torch.zeros_like(state.dead_s),
        )
    )
    state.invulnerable_s.copy_(torch.clamp(state.invulnerable_s - dt * active[:, None], min=0))

    immediate_cost = (
        torch.ceil(state.elapsed_s[:, None] / 60.0).to(torch.long)
        * constants.IMMEDIATE_RESPAWN_COST_PER_MINUTE
        + state.level * constants.IMMEDIATE_RESPAWN_COST_PER_LEVEL
    )
    immediate_requested = (
        active[:, None]
        & inputs.immediate_respawn_request
        & ~state.alive
        & ground
        & (state.respawn_required > 0)
        & ~state.controller_offline
    )
    immediate = _team_affordable(
        immediate_requested,
        immediate_cost,
        state.team_coin,
    )
    immediate_team_cost = (
        (immediate.to(torch.long) * immediate_cost)
        .view(state.num_envs, constants.TEAM_COUNT, constants.ROLES_PER_TEAM)
        .sum(dim=-1)
    )
    state.team_coin.sub_(immediate_team_cost)
    state.alive |= immediate
    state.hp.copy_(torch.where(immediate, state.max_hp, state.hp))
    state.invulnerable_s.copy_(
        torch.where(
            immediate,
            torch.full_like(
                state.invulnerable_s,
                constants.IMMEDIATE_RESPAWN_INVULNERABLE_S,
            ),
            state.invulnerable_s,
        )
    )
    state.immediate_power_boost_s.copy_(
        torch.where(
            immediate,
            torch.full_like(
                state.immediate_power_boost_s,
                constants.IMMEDIATE_RESPAWN_POWER_BOOST_S,
            ),
            state.immediate_power_boost_s,
        )
    )
    state.immediate_respawn_count.add_(immediate.to(torch.long))
    state.respawn_progress.copy_(torch.where(immediate, 0.0, state.respawn_progress))
    state.respawn_required.copy_(torch.where(immediate, 0.0, state.respawn_required))
    state.dead_s.copy_(torch.where(immediate, 0.0, state.dead_s))
    events.immediate_respawns.copy_(immediate)
    events.respawns |= immediate

    activity = (events.shots_fired.sum(dim=-1) > 0) | (events.damage_taken > 0)
    state.out_of_combat_s.copy_(
        torch.where(
            activity,
            torch.zeros_like(state.out_of_combat_s),
            torch.where(
                active[:, None] & state.alive & ground & ~state.controller_offline,
                state.out_of_combat_s + dt,
                state.out_of_combat_s,
            ),
        )
    )

    remote_role = (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )
    remote_cost = torch.full_like(
        state.immediate_respawn_count,
        constants.REMOTE_HEAL_BASE_COST,
    )
    remote_cost += (
        torch.ceil(state.elapsed_s[:, None] / 60.0 * constants.REMOTE_HEAL_COST_PER_MINUTE)
    ).to(torch.long)
    remote_requested = (
        active[:, None]
        & inputs.remote_heal_request
        & state.alive
        & remote_role[None, :]
        & ~state.remote_heal_active
        & ~state.controller_offline
        & (state.out_of_combat_s >= constants.OUT_OF_COMBAT_S)
        & (state.hp < state.max_hp)
    )
    remote_start = _team_affordable(
        remote_requested,
        remote_cost,
        state.team_coin,
    )
    remote_team_cost = (
        (remote_start.to(torch.long) * remote_cost)
        .view(state.num_envs, constants.TEAM_COUNT, constants.ROLES_PER_TEAM)
        .sum(dim=-1)
    )
    state.team_coin.sub_(remote_team_cost)
    state.remote_heal_active |= remote_start
    state.remote_heal_progress_s.copy_(torch.where(remote_start, 0.0, state.remote_heal_progress_s))
    state.remote_heal_progress_s.add_(state.remote_heal_active.to(state.dtype) * dt)
    remote_complete = state.remote_heal_active & (
        state.remote_heal_progress_s >= constants.REMOTE_TRANSACTION_CONFIRM_S
    )
    remote_heal = round_half_up(state.max_hp * constants.REMOTE_HEAL_FRACTION)
    state.hp.copy_(
        torch.where(
            remote_complete,
            torch.minimum(state.max_hp, state.hp + remote_heal),
            state.hp,
        )
    )
    state.remote_heal_active &= ~remote_complete
    state.remote_heal_progress_s.copy_(
        torch.where(remote_complete, 0.0, state.remote_heal_progress_s)
    )
    events.remote_heals.copy_(remote_complete)

    late_out_of_combat = (state.elapsed_s[:, None] >= constants.LATE_HEAL_START_S) & (
        state.out_of_combat_s >= constants.OUT_OF_COMBAT_S
    )
    heal_fraction = torch.where(
        late_out_of_combat,
        torch.full_like(state.hp, constants.LATE_SUPPLY_HEAL_FRACTION_PER_S),
        torch.full_like(state.hp, constants.SUPPLY_HEAL_FRACTION_PER_S),
    )
    heal = state.max_hp * heal_fraction * dt
    can_heal = (
        active[:, None] & state.alive & ground & inputs.in_supply_zone & ~state.controller_offline
    )
    state.hp.copy_(
        torch.where(
            can_heal,
            torch.minimum(state.max_hp, state.hp + heal),
            state.hp,
        )
    )

    base_slots = torch.tensor(
        [slot(Team.RED, Role.BASE), slot(Team.BLUE, Role.BASE)],
        device=state.device,
    )
    base_hp = state.hp[:, base_slots]
    own_base_hp = base_hp.repeat_interleave(constants.ROLES_PER_TEAM, dim=1)
    fast = inputs.in_supply_zone | (own_base_hp < constants.BASE_ARMOR_DEPLOY_HP)
    rate = torch.where(
        fast,
        torch.full_like(state.hp, constants.RESPAWN_RATE_FAST),
        torch.full_like(state.hp, constants.RESPAWN_RATE_NORMAL),
    )
    can_progress = active[:, None] & ~state.alive & ground & ~state.controller_offline
    state.respawn_progress.add_(torch.where(can_progress, rate * dt, 0.0))
    respawn = (
        can_progress
        & (state.respawn_progress >= state.respawn_required)
        & (state.respawn_required > 0)
    )
    state.alive |= respawn
    state.hp.copy_(
        torch.where(
            respawn,
            state.max_hp * constants.READ_RESPAWN_HP_FRACTION,
            state.hp,
        )
    )
    state.invulnerable_s.copy_(
        torch.where(
            respawn,
            torch.full_like(
                state.invulnerable_s,
                constants.READ_RESPAWN_INVULNERABLE_S,
            ),
            state.invulnerable_s,
        )
    )
    state.weak |= respawn
    state.respawn_progress.copy_(
        torch.where(respawn, torch.zeros_like(state.respawn_progress), state.respawn_progress)
    )
    state.respawn_required.copy_(
        torch.where(respawn, torch.zeros_like(state.respawn_required), state.respawn_required)
    )
    state.dead_s.copy_(torch.where(respawn, 0.0, state.dead_s))
    events.respawns |= respawn

    release_weak = state.weak & inputs.respawn_contact
    elapsed_invulnerable = constants.READ_RESPAWN_INVULNERABLE_S - state.invulnerable_s
    minimum_remaining = torch.clamp(
        constants.READ_RESPAWN_MIN_INVULNERABLE_S - elapsed_invulnerable,
        min=0,
    )
    state.invulnerable_s.copy_(torch.where(release_weak, minimum_remaining, state.invulnerable_s))
    state.weak &= ~release_weak
