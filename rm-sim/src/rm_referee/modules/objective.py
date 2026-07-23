"""Building state and the ordered match outcome ladder."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.schema import Role, Team, Winner, ground_robot_mask, slot
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


def apply_objective(state: GameState, events: RefereeEvents) -> None:
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
