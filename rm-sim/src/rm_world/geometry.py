"""Target resolution, analytic armor selection, and calibrated hit model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, unit_roles, unit_teams
from rm_referee.state import GameState
from rm_world.arena import ArenaGeometry
from rm_world.kinematics import KinematicState


TARGET_ROLE_CHOICES = (
    Role.HERO,
    Role.ENGINEER,
    Role.INFANTRY_3,
    Role.INFANTRY_4,
    Role.SENTRY,
    Role.BASE,
    Role.OUTPOST,
)
RUNE_TARGET = 7
NO_TARGET = 8


@dataclass(frozen=True)
class HitModelConfig:
    """Calibratable ``[SIM]`` values; none are official rule constants."""

    base_accuracy: float = 0.96
    range_scale_m: float = 18.0
    motion_scale_mps: float = 5.0
    projection_floor: float = 0.30
    calibration: float = 1.0
    critical_probability: float = 0.02
    dart_hit_probability: float = 0.65


@dataclass
class ArmorSolution:
    target_slot: Tensor
    armor_index: Tensor
    distance_m: Tensor
    projected_fraction: Tensor
    target_speed_mps: Tensor
    line_of_sight: Tensor
    target_valid: Tensor


class HitModel:
    def __init__(self, config: HitModelConfig | None = None) -> None:
        self.config = config or HitModelConfig()

    def probability(self, solution: ArmorSolution) -> Tensor:
        range_factor = torch.exp(-torch.square(solution.distance_m / self.config.range_scale_m))
        motion_factor = torch.exp(-solution.target_speed_mps / self.config.motion_scale_mps)
        projection = (
            self.config.projection_floor
            + (1.0 - self.config.projection_floor) * solution.projected_fraction
        )
        probability = (
            self.config.base_accuracy
            * self.config.calibration
            * range_factor
            * motion_factor
            * projection
        )
        return torch.where(
            solution.target_valid & solution.line_of_sight,
            torch.clamp(probability, min=0.0, max=1.0),
            torch.zeros_like(probability),
        )


def resolve_target_slots(target_choice: Tensor) -> Tensor:
    """Map each unit's nine-class target choice to an absolute unit slot."""

    source_teams = unit_teams(target_choice.device)
    enemy_team = 1 - source_teams
    role_choice = torch.tensor(
        tuple(int(role) for role in TARGET_ROLE_CHOICES),
        device=target_choice.device,
        dtype=torch.long,
    )
    safe_choice = torch.clamp(target_choice, min=0, max=len(TARGET_ROLE_CHOICES) - 1)
    selected_role = role_choice[safe_choice]
    team_shape = (1, constants.UNIT_COUNT) + (1,) * (target_choice.ndim - 2)
    resolved = enemy_team.view(team_shape) * constants.ROLES_PER_TEAM + selected_role
    return torch.where(
        target_choice < len(TARGET_ROLE_CHOICES),
        resolved,
        torch.full_like(resolved, -1),
    )


def solve_armor_geometry(
    world: KinematicState,
    game: GameState,
    target_choice: Tensor,
    arena: ArenaGeometry,
) -> ArmorSolution:
    target_slot = resolve_target_slots(target_choice)
    target_safe = torch.clamp(target_slot, min=0)
    target_xy = torch.gather(
        world.position_xy,
        1,
        target_safe[:, :, None].expand(-1, -1, 2),
    )
    target_yaw = torch.gather(world.yaw, 1, target_safe)
    target_velocity = torch.gather(
        world.velocity_xy,
        1,
        target_safe[:, :, None].expand(-1, -1, 2),
    )
    to_source = world.position_xy - target_xy
    distance = torch.linalg.vector_norm(to_source, dim=-1)
    direction = to_source / torch.clamp(distance[:, :, None], min=1.0e-6)

    target_roles = unit_roles(game.device)[target_safe]
    armor_count_table = torch.tensor(
        constants.ARMOR_COUNT_BY_ROLE,
        device=game.device,
        dtype=torch.long,
    )
    armor_count = armor_count_table[target_roles]
    armor = torch.arange(constants.MAX_ARMOR_COUNT, device=game.device)
    armor_angle = target_yaw[:, :, None] + (
        2.0 * math.pi * armor[None, None, :] / torch.clamp(armor_count[:, :, None], min=1)
    )
    normal = torch.stack((torch.cos(armor_angle), torch.sin(armor_angle)), dim=-1)
    projected = (normal * direction[:, :, None, :]).sum(dim=-1)
    armor_valid = armor[None, None, :] < armor_count[:, :, None]
    projected = torch.where(
        armor_valid,
        projected,
        torch.full_like(projected, -torch.inf),
    )
    projected_fraction, armor_index = projected.max(dim=-1)

    source_teams = unit_teams(game.device)
    target_teams = unit_teams(game.device)[target_safe]
    target_alive = torch.gather(game.alive, 1, target_safe)
    target_valid = (
        (target_slot >= 0)
        & target_alive
        & (target_teams != source_teams[None, :])
        & (target_roles != Role.AERIAL)
        & (armor_count > 0)
    )
    los = arena.line_of_sight(world.position_xy, target_xy)
    return ArmorSolution(
        target_slot=target_slot,
        armor_index=armor_index,
        distance_m=distance,
        projected_fraction=torch.clamp(projected_fraction, min=0, max=1),
        target_speed_mps=torch.linalg.vector_norm(target_velocity, dim=-1),
        line_of_sight=los,
        target_valid=target_valid,
    )
