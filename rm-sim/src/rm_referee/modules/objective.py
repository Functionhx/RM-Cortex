"""Building state and the ordered match outcome ladder."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, Team, Winner, ground_robot_mask, slot, unit_roles
from rm_referee.state import GameState


def _assign_metric(
    winner: torch.Tensor,
    unresolved: torch.Tensor,
    red: torch.Tensor,
    blue: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    red_wins = unresolved & (red > blue)
    blue_wins = unresolved & (blue > red)
    winner = torch.where(
        red_wins,
        torch.full_like(winner, Winner.RED),
        winner,
    )
    winner = torch.where(
        blue_wins,
        torch.full_like(winner, Winner.BLUE),
        winner,
    )
    return winner, unresolved & ~(red_wins | blue_wins)


def apply_objective(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    base_slots = torch.tensor(
        [slot(Team.RED, Role.BASE), slot(Team.BLUE, Role.BASE)],
        device=state.device,
    )
    outpost_slots = torch.tensor(
        [slot(Team.RED, Role.OUTPOST), slot(Team.BLUE, Role.OUTPOST)],
        device=state.device,
    )
    base_hp = state.hp[:, base_slots]
    outpost_hp = state.hp[:, outpost_slots]

    state.base_armor_deployed |= base_hp <= constants.BASE_ARMOR_DEPLOY_HP
    state.outpost_destroyed_once |= outpost_hp <= 0

    milestone = torch.floor(state.base_hp_lost / constants.OUTPOST_REBUILD_DAMAGE_STEP).to(
        torch.long
    )
    new_charges = torch.clamp(milestone - state.outpost_rebuild_awarded, min=0)
    state.outpost_rebuild_charges.add_(new_charges)
    state.outpost_rebuild_awarded.copy_(torch.maximum(state.outpost_rebuild_awarded, milestone))

    roles = unit_roles(state.device)
    rebuild_role = (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )
    own_outpost_dead = (~state.alive[:, outpost_slots]).repeat_interleave(
        constants.ROLES_PER_TEAM,
        dim=1,
    )
    own_charge = (state.outpost_rebuild_charges > 0).repeat_interleave(
        constants.ROLES_PER_TEAM,
        dim=1,
    )
    rebuilding = (
        (~state.done)[:, None]
        & (state.elapsed_s[:, None] < constants.OUTPOST_REBUILD_DEADLINE_S)
        & own_outpost_dead
        & own_charge
        & state.alive
        & rebuild_role[None, :]
        & ~state.weak
        & inputs.in_outpost_zone
        & inputs.outpost_rebuild_request
    )
    state.outpost_rebuild_progress_s.copy_(
        torch.where(
            rebuilding,
            state.outpost_rebuild_progress_s + dt,
            torch.zeros_like(state.outpost_rebuild_progress_s),
        )
    )
    required = torch.where(
        roles == Role.ENGINEER,
        torch.full(
            (constants.UNIT_COUNT,),
            constants.OUTPOST_REBUILD_ENGINEER_S,
            device=state.device,
            dtype=state.dtype,
        ),
        torch.full(
            (constants.UNIT_COUNT,),
            constants.OUTPOST_REBUILD_NORMAL_S,
            device=state.device,
            dtype=state.dtype,
        ),
    )
    finished = (state.outpost_rebuild_progress_s >= required[None, :]).view(
        state.num_envs,
        constants.TEAM_COUNT,
        constants.ROLES_PER_TEAM,
    )
    first_finished = finished & (torch.cumsum(finished.to(torch.long), dim=-1) == 1)
    rebuilt = first_finished.any(dim=-1) & (state.outpost_rebuild_charges > 0)
    state.outpost_rebuild_charges.sub_(rebuilt.to(torch.long))
    state.hp[:, outpost_slots] = torch.where(
        rebuilt,
        torch.full_like(outpost_hp, constants.OUTPOST_REBUILD_HP),
        state.hp[:, outpost_slots],
    )
    state.alive[:, outpost_slots] |= rebuilt
    state.outpost_rebuild_progress_s.copy_(
        torch.where(
            rebuilt.repeat_interleave(constants.ROLES_PER_TEAM, dim=1),
            0.0,
            state.outpost_rebuild_progress_s,
        )
    )

    stop_rotation = (
        state.outpost_destroyed_once
        | state.base_armor_deployed.flip(dims=(1,))
        | (state.elapsed_s[:, None] >= constants.OUTPOST_ROTATION_STOP_S)
    )
    state.outpost_rotation_stopped |= stop_rotation
    ramp = torch.clamp(
        state.elapsed_s[:, None] / constants.OUTPOST_ROTATION_RAMP_S,
        max=1.0,
    )
    speed = constants.OUTPOST_ROTATION_SPEED_RAD_S * ramp * state.outpost_rotation_direction
    state.outpost_speed_rad_s.copy_(torch.where(state.outpost_rotation_stopped, 0.0, speed))
    raw_angle = state.outpost_angle_rad + state.outpost_speed_rad_s * dt
    state.outpost_angle_rad.copy_(torch.atan2(torch.sin(raw_angle), torch.cos(raw_angle)))

    newly_ended = (~state.done) & (
        (base_hp <= 0).any(dim=1) | (state.elapsed_s >= constants.MATCH_DURATION_S)
    )
    winner = torch.full_like(state.winner, Winner.UNDECIDED)
    unresolved = newly_ended.clone()

    winner, unresolved = _assign_metric(
        winner,
        unresolved,
        base_hp[:, Team.RED],
        base_hp[:, Team.BLUE],
    )

    red_destroyed = state.outpost_destroyed_once[:, Team.RED]
    blue_destroyed = state.outpost_destroyed_once[:, Team.BLUE]
    only_red_destroyed = unresolved & red_destroyed & ~blue_destroyed
    only_blue_destroyed = unresolved & ~red_destroyed & blue_destroyed
    winner = torch.where(
        only_red_destroyed,
        torch.full_like(winner, Winner.BLUE),
        winner,
    )
    winner = torch.where(
        only_blue_destroyed,
        torch.full_like(winner, Winner.RED),
        winner,
    )
    unresolved &= ~(only_red_destroyed | only_blue_destroyed)

    neither_destroyed = unresolved & ~red_destroyed & ~blue_destroyed
    outpost_winner, still_tied = _assign_metric(
        winner,
        neither_destroyed,
        outpost_hp[:, Team.RED],
        outpost_hp[:, Team.BLUE],
    )
    winner = torch.where(neither_destroyed, outpost_winner, winner)
    unresolved = (unresolved & ~neither_destroyed) | still_tied

    winner, unresolved = _assign_metric(
        winner,
        unresolved,
        state.team_damage[:, Team.RED],
        state.team_damage[:, Team.BLUE],
    )

    ground = ground_robot_mask(state.device)
    red_ground = ground.clone()
    red_ground[constants.ROLES_PER_TEAM :] = False
    blue_ground = ground.clone()
    blue_ground[: constants.ROLES_PER_TEAM] = False
    remaining_red = (state.hp * state.alive.to(state.dtype) * red_ground.unsqueeze(0)).sum(dim=1)
    remaining_blue = (state.hp * state.alive.to(state.dtype) * blue_ground.unsqueeze(0)).sum(dim=1)
    winner, unresolved = _assign_metric(
        winner,
        unresolved,
        remaining_red,
        remaining_blue,
    )
    winner = torch.where(
        unresolved,
        torch.full_like(winner, Winner.DRAW),
        winner,
    )

    state.winner.copy_(torch.where(newly_ended, winner, state.winner))
    state.done |= newly_ended
    events.match_ended.copy_(newly_ended)
