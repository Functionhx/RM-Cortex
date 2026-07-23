"""Deterministic baseline opponent for smoke tests and evaluation."""

from __future__ import annotations

import math

import torch

from rm_referee import constants
from rm_referee.schema import Role, Team, unit_roles
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.geometry import NO_TARGET, RUNE_TARGET
from rm_world.kinematics import KinematicState


MOBILE_UNIT_SLOTS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13)
TERRAIN_DEMO_PHASE_EDGES_S = (6.0, 13.0, 22.0)
TERRAIN_DEMO_RED_ROUTES = (
    (
        (-6.0, -2.9),
        (-6.0, -4.8),
        (-5.5, -5.4),
        (-6.8, 2.8),
        (-5.0, 1.2),
        (-6.0, 3.0),
    ),
    (
        (-2.6, -3.4),
        (-2.7, -5.2),
        (-1.0, -4.7),
        (-2.5, 4.8),
        (0.0, 0.0),
        (-2.8, 3.2),
    ),
    (
        (2.6, -3.4),
        (1.5, -3.2),
        (3.0, -3.0),
        (2.0, 3.5),
        (5.0, -1.2),
        (2.8, 3.2),
    ),
    (
        (6.0, -1.0),
        (5.0, -1.8),
        (6.0, -1.5),
        (6.0, 1.5),
        (8.0, 0.0),
        (5.5, 1.0),
    ),
)


def terrain_demo_targets(
    elapsed_s: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return center-symmetric waypoints for the visualization demos."""

    phase = sum(elapsed_s >= edge for edge in TERRAIN_DEMO_PHASE_EDGES_S)
    red = torch.tensor(
        TERRAIN_DEMO_RED_ROUTES[phase],
        device=device,
        dtype=dtype,
    )
    return torch.cat((red, -red), dim=0)


class ScriptedOpponent:
    """A small objective-first policy, intentionally free of hidden state."""

    def __init__(self, *, cruise_speed_mps: float = 2.0, fire_range_m: float = 12.0) -> None:
        self.cruise_speed_mps = cruise_speed_mps
        self.fire_range_m = fire_range_m

    def act(
        self,
        game: GameState,
        world: KinematicState,
        *,
        team: Team | int | None = None,
    ) -> WorldActions:
        actions = WorldActions.zeros(game)
        roles = unit_roles(game.device)
        unit_team = torch.arange(constants.UNIT_COUNT, device=game.device) // (
            constants.ROLES_PER_TEAM
        )
        controlled = torch.ones(
            constants.UNIT_COUNT,
            device=game.device,
            dtype=torch.bool,
        )
        if team is not None:
            controlled = unit_team == int(team)

        enemy_outpost_alive = game.alive[:, (15, 7)][:, unit_team]
        target_choice = torch.where(
            enemy_outpost_alive,
            torch.full(
                (game.num_envs, constants.UNIT_COUNT),
                6,
                device=game.device,
                dtype=torch.long,
            ),
            torch.full(
                (game.num_envs, constants.UNIT_COUNT),
                5,
                device=game.device,
                dtype=torch.long,
            ),
        )
        enemy_objective_slot = torch.where(
            enemy_outpost_alive,
            (1 - unit_team)[None, :] * constants.ROLES_PER_TEAM + Role.OUTPOST,
            (1 - unit_team)[None, :] * constants.ROLES_PER_TEAM + Role.BASE,
        )
        objective_xy = torch.gather(
            world.position_xy,
            1,
            enemy_objective_slot[:, :, None].expand(-1, -1, 2),
        )
        delta = objective_xy - world.position_xy
        distance = torch.linalg.vector_norm(delta, dim=-1)
        desired_world = (
            delta / torch.clamp(distance[:, :, None], min=1.0e-6) * self.cruise_speed_mps
        )
        cosine = torch.cos(world.yaw)
        sine = torch.sin(world.yaw)
        body_velocity = torch.stack(
            (
                cosine * desired_world[..., 0] + sine * desired_world[..., 1],
                -sine * desired_world[..., 0] + cosine * desired_world[..., 1],
            ),
            dim=-1,
        )
        movable_role = (roles != Role.BASE) & (roles != Role.OUTPOST)
        actions.body_velocity_xy.copy_(
            torch.where(
                (controlled & movable_role)[None, :, None],
                body_velocity,
                actions.body_velocity_xy,
            )
        )
        desired_yaw = torch.atan2(delta[..., 1], delta[..., 0])
        yaw_error = torch.atan2(
            torch.sin(desired_yaw - world.yaw),
            torch.cos(desired_yaw - world.yaw),
        )
        actions.yaw_rate.copy_(
            torch.where(
                (controlled & movable_role)[None, :],
                torch.clamp(yaw_error * 3.0, min=-math.pi, max=math.pi),
                actions.yaw_rate,
            )
        )

        launcher_role = (
            (roles == Role.HERO)
            | (roles == Role.INFANTRY_3)
            | (roles == Role.INFANTRY_4)
            | (roles == Role.AERIAL)
            | (roles == Role.SENTRY)
        )
        rune_active = game.rune_mode[:, unit_team] != 0
        target_choice = torch.where(
            rune_active & launcher_role[None, :],
            torch.full_like(target_choice, RUNE_TARGET),
            target_choice,
        )
        actions.target.copy_(
            torch.where(
                controlled[None, :],
                target_choice,
                torch.full_like(target_choice, NO_TARGET),
            )
        )
        actions.fire.copy_(
            controlled[None, :]
            & launcher_role[None, :]
            & game.alive
            & ((distance <= self.fire_range_m) | rune_active)
        )

        controlled_teams = torch.ones(
            constants.TEAM_COUNT,
            device=game.device,
            dtype=torch.bool,
        )
        if team is not None:
            controlled_teams = torch.arange(constants.TEAM_COUNT, device=game.device) == int(team)
        actions.aerial_support.copy_(controlled_teams[None, :] & (game.aerial_support_bank_s > 0))
        radar_target = torch.tensor(
            [constants.ROLES_PER_TEAM + Role.INFANTRY_3, Role.INFANTRY_3],
            device=game.device,
            dtype=torch.long,
        )
        actions.radar_target.copy_(
            torch.where(
                controlled_teams[None, :],
                radar_target[None, :],
                torch.full_like(actions.radar_target, -1),
            )
        )
        actions.radar_report_xy.copy_(
            torch.gather(
                world.position_xy,
                1,
                radar_target[None, :, None].expand(game.num_envs, -1, 2),
            )
        )
        return actions
