"""Out-of-combat tracking, supply healing, and read-bar respawn."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import HeatLock, Role, Team, ground_robot_mask, slot, unit_teams
from rm_referee.state import GameState
from rm_referee.tensor_ops import round_half_up


def apply_survive(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active = ~state.done
    ground = ground_robot_mask(state.device).unsqueeze(0)
    new_death = events.deaths & ground

    state.heat.copy_(torch.where(new_death[:, :, None], 0.0, state.heat))
    temporary_lock = new_death[:, :, None] & (state.heat_lock == HeatLock.TEMPORARY)
    state.heat_lock.copy_(
        torch.where(
            temporary_lock,
            torch.full_like(state.heat_lock, HeatLock.NONE),
            state.heat_lock,
        )
    )
    state.video_disabled.copy_((state.heat_lock != HeatLock.NONE).any(dim=-1))
    state.power_buffer_j.copy_(
        torch.where(
            new_death,
            torch.full_like(state.power_buffer_j, constants.POWER_BUFFER_MAX_J),
            state.power_buffer_j,
        )
    )
    required = round_half_up(
        constants.RESPAWN_BASE_PROGRESS
        + state.elapsed_s[:, None] / constants.RESPAWN_ELAPSED_DIVISOR
        + constants.RESPAWN_IMMEDIATE_PENALTY * state.immediate_respawn_count.to(state.dtype)
    )
    state.respawn_required.copy_(torch.where(new_death, required, state.respawn_required))
    state.respawn_progress.copy_(
        torch.where(new_death, torch.zeros_like(state.respawn_progress), state.respawn_progress)
    )
    state.weak &= ~new_death
    state.invulnerable_s.copy_(
        torch.where(new_death, torch.zeros_like(state.invulnerable_s), state.invulnerable_s)
    )

    activity = (events.shots_fired.sum(dim=-1) > 0) | (events.damage_taken > 0)
    state.out_of_combat_s.copy_(
        torch.where(
            activity,
            torch.zeros_like(state.out_of_combat_s),
            torch.where(
                active[:, None] & state.alive & ground,
                state.out_of_combat_s + dt,
                state.out_of_combat_s,
            ),
        )
    )
    state.invulnerable_s.copy_(torch.clamp(state.invulnerable_s - dt * active[:, None], min=0))

    late_out_of_combat = (state.elapsed_s[:, None] >= constants.LATE_HEAL_START_S) & (
        state.out_of_combat_s >= constants.OUT_OF_COMBAT_S
    )
    heal_fraction = torch.where(
        late_out_of_combat,
        torch.full_like(state.hp, constants.LATE_SUPPLY_HEAL_FRACTION_PER_S),
        torch.full_like(state.hp, constants.SUPPLY_HEAL_FRACTION_PER_S),
    )
    heal = state.max_hp * heal_fraction * dt
    can_heal = active[:, None] & state.alive & ground & inputs.in_supply_zone
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
    own_team = unit_teams(state.device).view(1, constants.UNIT_COUNT)
    own_base_hp = torch.gather(
        base_hp,
        1,
        own_team.expand(state.num_envs, -1),
    )
    fast = inputs.in_supply_zone | (own_base_hp < constants.BASE_ARMOR_DEPLOY_HP)
    rate = torch.where(
        fast,
        torch.full_like(state.hp, constants.RESPAWN_RATE_FAST),
        torch.full_like(state.hp, constants.RESPAWN_RATE_NORMAL),
    )
    can_progress = active[:, None] & ~state.alive & ground
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
    events.respawns.copy_(respawn)

    release_weak = state.weak & inputs.respawn_contact
    elapsed_invulnerable = constants.READ_RESPAWN_INVULNERABLE_S - state.invulnerable_s
    minimum_remaining = torch.clamp(
        constants.READ_RESPAWN_MIN_INVULNERABLE_S - elapsed_invulnerable,
        min=0,
    )
    state.invulnerable_s.copy_(torch.where(release_weak, minimum_remaining, state.invulnerable_s))
    state.weak &= ~release_weak
