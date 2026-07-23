"""Batched 2.5D kinematics for the Phase 1 Torch backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, unit_roles
from rm_referee.state import GameState


@dataclass(frozen=True)
class KinematicConfig:
    """Explicit ``[SIM]`` parameters, not official rule constants."""

    field_length_m: float = 28.0
    field_width_m: float = 15.0
    max_linear_speed_mps: float = 3.0
    max_yaw_rate_rad_s: float = 2.0 * math.pi


@dataclass
class KinematicState:
    position_xy: Tensor
    yaw: Tensor
    velocity_xy: Tensor
    yaw_rate: Tensor

    @classmethod
    def zeros(cls, game: GameState) -> "KinematicState":
        env_unit = (game.num_envs, constants.UNIT_COUNT)
        return cls(
            position_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            yaw=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            velocity_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            yaw_rate=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
        )

    def clone(self) -> "KinematicState":
        return KinematicState(
            self.position_xy.clone(),
            self.yaw.clone(),
            self.velocity_xy.clone(),
            self.yaw_rate.clone(),
        )


@dataclass
class KinematicCommands:
    body_velocity_xy: Tensor
    yaw_rate: Tensor

    @classmethod
    def zeros(cls, game: GameState) -> "KinematicCommands":
        env_unit = (game.num_envs, constants.UNIT_COUNT)
        return cls(
            body_velocity_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            yaw_rate=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
        )


class KinematicWorld:
    def __init__(self, config: KinematicConfig | None = None) -> None:
        self.config = config or KinematicConfig()

    def step(
        self,
        world: KinematicState,
        game: GameState,
        commands: KinematicCommands,
        *,
        dt: float = 1.0 / 60.0,
    ) -> KinematicState:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if commands.body_velocity_xy.shape != world.position_xy.shape:
            raise ValueError("velocity command shape differs from world state")

        speed = torch.linalg.vector_norm(commands.body_velocity_xy, dim=-1, keepdim=True)
        scale = torch.clamp(
            self.config.max_linear_speed_mps / torch.clamp(speed, min=1.0e-9),
            max=1.0,
        )
        body_velocity = commands.body_velocity_xy * scale
        yaw_rate = torch.clamp(
            commands.yaw_rate,
            min=-self.config.max_yaw_rate_rad_s,
            max=self.config.max_yaw_rate_rad_s,
        )

        cosine = torch.cos(world.yaw)
        sine = torch.sin(world.yaw)
        world_velocity = torch.stack(
            (
                cosine * body_velocity[..., 0] - sine * body_velocity[..., 1],
                sine * body_velocity[..., 0] + cosine * body_velocity[..., 1],
            ),
            dim=-1,
        )
        roles = unit_roles(game.device)
        movable_role = (roles != Role.BASE) & (roles != Role.OUTPOST)
        movable = game.alive & movable_role.unsqueeze(0) & ~game.done[:, None]
        world_velocity = torch.where(
            movable[:, :, None],
            world_velocity,
            torch.zeros_like(world_velocity),
        )
        yaw_rate = torch.where(movable, yaw_rate, torch.zeros_like(yaw_rate))

        result = world.clone()
        result.position_xy.add_(world_velocity * dt)
        result.position_xy[..., 0].clamp_(
            min=-self.config.field_length_m / 2,
            max=self.config.field_length_m / 2,
        )
        result.position_xy[..., 1].clamp_(
            min=-self.config.field_width_m / 2,
            max=self.config.field_width_m / 2,
        )
        raw_yaw = result.yaw + yaw_rate * dt
        result.yaw.copy_(torch.atan2(torch.sin(raw_yaw), torch.cos(raw_yaw)))
        result.velocity_xy.copy_(world_velocity)
        result.yaw_rate.copy_(yaw_rate)
        return result
