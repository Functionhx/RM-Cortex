"""Batched 2.5D kinematics for the Phase 1 Torch backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Team, slot, unit_roles
from rm_referee.state import GameState
from rm_world.arena import ArenaConfig, ArenaGeometry


@dataclass(frozen=True)
class KinematicConfig:
    """Explicit ``[SIM]`` parameters, not official rule constants."""

    field_length_m: float = 28.0
    field_width_m: float = 15.0
    max_linear_speed_mps: float = 3.0
    max_vertical_speed_mps: float = 2.0
    max_yaw_rate_rad_s: float = 2.0 * math.pi
    aerial_min_z_m: float = 0.0
    aerial_max_z_m: float = 5.0
    robot_radius_m: float = 0.40
    boundary_margin_m: float = 0.0
    enable_static_collisions: bool = False
    enable_unit_collisions: bool = False
    collision_iterations: int = 4
    collision_clearance_m: float = 0.05


@dataclass
class KinematicState:
    position_xy: Tensor
    position_z: Tensor
    yaw: Tensor
    velocity_xy: Tensor
    vertical_velocity: Tensor
    yaw_rate: Tensor
    terrain_contact: Tensor
    terrain_crossed: Tensor

    @classmethod
    def zeros(cls, game: GameState) -> "KinematicState":
        env_unit = (game.num_envs, constants.UNIT_COUNT)
        return cls(
            position_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            position_z=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            yaw=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            velocity_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            vertical_velocity=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            yaw_rate=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            terrain_contact=torch.zeros(
                (*env_unit, constants.TERRAIN_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
            terrain_crossed=torch.zeros(
                (*env_unit, constants.TERRAIN_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
        )

    @classmethod
    def spawn(
        cls,
        game: GameState,
        arena: ArenaGeometry | None = None,
    ) -> "KinematicState":
        arena = arena or ArenaGeometry()
        state = cls.zeros(game)
        state.position_xy.copy_(arena.spawn_positions(game))
        roles = unit_roles(game.device)
        blue = torch.arange(constants.UNIT_COUNT, device=game.device) >= constants.ROLES_PER_TEAM
        state.yaw[:, blue] = math.pi
        state.position_z.copy_(
            torch.where(
                roles[None, :] == Role.AERIAL,
                torch.zeros_like(state.position_z),
                arena.terrain_height(state.position_xy),
            )
        )
        state.terrain_contact.copy_(arena.terrain_contacts(state.position_xy))
        return state

    def clone(self) -> "KinematicState":
        return KinematicState(
            self.position_xy.clone(),
            self.position_z.clone(),
            self.yaw.clone(),
            self.velocity_xy.clone(),
            self.vertical_velocity.clone(),
            self.yaw_rate.clone(),
            self.terrain_contact.clone(),
            self.terrain_crossed.clone(),
        )


@dataclass
class KinematicCommands:
    body_velocity_xy: Tensor
    yaw_rate: Tensor
    vertical_velocity: Tensor

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
            vertical_velocity=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
        )


class KinematicWorld:
    def __init__(
        self,
        config: KinematicConfig | None = None,
        arena: ArenaGeometry | None = None,
    ) -> None:
        self.config = config or KinematicConfig()
        if self.config.collision_iterations <= 0:
            raise ValueError("collision_iterations must be positive")
        if self.config.collision_clearance_m < 0:
            raise ValueError("collision_clearance_m must be non-negative")
        self.arena = arena or ArenaGeometry(
            ArenaConfig(
                field_length_m=self.config.field_length_m,
                field_width_m=self.config.field_width_m,
                robot_radius_m=self.config.robot_radius_m,
            )
        )

    def _resolve_unit_collisions(
        self,
        position_xy: Tensor,
        game: GameState,
        movable: Tensor,
    ) -> Tensor:
        """Project ground-unit circles apart with fixed-iteration PBD.

        The fallback normals handle exact coincident centers deterministically.
        Destroyed robots and structures remain physical, but only movable units
        receive corrections.
        """

        roles = unit_roles(game.device)
        radius = torch.where(
            roles == Role.BASE,
            torch.full(
                (constants.UNIT_COUNT,),
                0.95,
                device=game.device,
                dtype=game.dtype,
            ),
            torch.where(
                roles == Role.OUTPOST,
                torch.full(
                    (constants.UNIT_COUNT,),
                    0.40,
                    device=game.device,
                    dtype=game.dtype,
                ),
                torch.full(
                    (constants.UNIT_COUNT,),
                    self.config.robot_radius_m,
                    device=game.device,
                    dtype=game.dtype,
                ),
            ),
        )
        collidable = roles[None, :] != Role.AERIAL
        pair_radius = radius[None, :, None] + radius[None, None, :]
        identity = torch.eye(
            constants.UNIT_COUNT,
            device=game.device,
            dtype=torch.bool,
        )[None, :, :]
        indices = torch.arange(constants.UNIT_COUNT, device=game.device)
        index_i = indices[:, None]
        index_j = indices[None, :]
        lower = torch.minimum(index_i, index_j)
        upper = torch.maximum(index_i, index_j)
        fallback_angle = (lower * constants.UNIT_COUNT + upper).to(game.dtype) * 2.39996323
        fallback_direction = torch.stack(
            (torch.cos(fallback_angle), torch.sin(fallback_angle)),
            dim=-1,
        )
        antisymmetric_sign = torch.where(
            index_i < index_j,
            torch.ones_like(fallback_angle),
            -torch.ones_like(fallback_angle),
        )
        fallback_direction *= antisymmetric_sign[..., None]
        movable_i = movable[:, :, None].to(game.dtype)
        movable_j = movable[:, None, :].to(game.dtype)
        share = movable_i / torch.clamp(movable_i + movable_j, min=1.0)
        resolved = position_xy
        for _ in range(self.config.collision_iterations):
            delta = resolved[:, :, None, :] - resolved[:, None, :, :]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            overlap = torch.clamp(
                pair_radius + self.config.collision_clearance_m - distance,
                min=0,
            )
            pair_active = (
                collidable[:, :, None] & collidable[:, None, :] & ~identity & (overlap > 0)
            )
            direction = torch.where(
                (distance > 1.0e-6)[..., None],
                delta / torch.clamp(distance[..., None], min=1.0e-6),
                fallback_direction[None, ...],
            )
            displacement = (
                direction
                * overlap[..., None]
                * pair_active[..., None].to(game.dtype)
                * share[..., None]
            ).sum(dim=2)
            resolved = torch.where(
                movable[..., None],
                resolved + displacement,
                resolved,
            )
            resolved[..., 0].clamp_(
                min=-self.config.field_length_m / 2 + self.config.boundary_margin_m,
                max=self.config.field_length_m / 2 - self.config.boundary_margin_m,
            )
            resolved[..., 1].clamp_(
                min=-self.config.field_width_m / 2 + self.config.boundary_margin_m,
                max=self.config.field_width_m / 2 - self.config.boundary_margin_m,
            )
        return resolved

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
        vertical_velocity = torch.clamp(
            commands.vertical_velocity,
            min=-self.config.max_vertical_speed_mps,
            max=self.config.max_vertical_speed_mps,
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
        movable = self.arena.movable_mask(game) & ~game.done[:, None]
        hero_slots = torch.tensor(
            [slot(Team.RED, Role.HERO), slot(Team.BLUE, Role.HERO)],
            device=game.device,
        )
        movable[:, hero_slots] &= ~game.hero_deployed
        movable &= game.chassis_disabled_s <= 0
        world_velocity = torch.where(
            movable[:, :, None],
            world_velocity,
            torch.zeros_like(world_velocity),
        )
        yaw_rate = torch.where(movable, yaw_rate, torch.zeros_like(yaw_rate))
        aerial = roles == Role.AERIAL
        vertical_velocity = torch.where(
            movable & aerial[None, :],
            vertical_velocity,
            torch.zeros_like(vertical_velocity),
        )

        result = world.clone()
        candidate_xy = world.position_xy + world_velocity * dt
        candidate_xy[..., 0].clamp_(
            min=-self.config.field_length_m / 2 + self.config.boundary_margin_m,
            max=self.config.field_length_m / 2 - self.config.boundary_margin_m,
        )
        candidate_xy[..., 1].clamp_(
            min=-self.config.field_width_m / 2 + self.config.boundary_margin_m,
            max=self.config.field_width_m / 2 - self.config.boundary_margin_m,
        )
        if self.config.enable_static_collisions:
            projected = self.arena.project_out_of_obstacles(
                candidate_xy,
                self.config.robot_radius_m,
            )
            ground_movable = movable & ~aerial[None, :]
            candidate_xy = torch.where(
                ground_movable[:, :, None],
                projected,
                candidate_xy,
            )
        if self.config.enable_unit_collisions:
            candidate_xy = self._resolve_unit_collisions(candidate_xy, game, movable)
        if self.config.enable_static_collisions:
            projected = self.arena.project_out_of_obstacles(
                candidate_xy,
                self.config.robot_radius_m,
            )
            ground_movable = movable & ~aerial[None, :]
            candidate_xy = torch.where(
                ground_movable[:, :, None],
                projected,
                candidate_xy,
            )
        if self.config.enable_unit_collisions or self.config.enable_static_collisions:
            candidate_xy[..., 0].clamp_(
                min=-self.config.field_length_m / 2 + self.config.boundary_margin_m,
                max=self.config.field_length_m / 2 - self.config.boundary_margin_m,
            )
            candidate_xy[..., 1].clamp_(
                min=-self.config.field_width_m / 2 + self.config.boundary_margin_m,
                max=self.config.field_width_m / 2 - self.config.boundary_margin_m,
            )
        result.position_xy.copy_(torch.where(movable[:, :, None], candidate_xy, world.position_xy))

        ground_height = self.arena.terrain_height(result.position_xy)
        result.position_z.copy_(
            torch.where(
                aerial[None, :],
                torch.clamp(
                    world.position_z + vertical_velocity * dt,
                    min=self.config.aerial_min_z_m,
                    max=self.config.aerial_max_z_m,
                ),
                ground_height,
            )
        )
        raw_yaw = result.yaw + yaw_rate * dt
        result.yaw.copy_(torch.atan2(torch.sin(raw_yaw), torch.cos(raw_yaw)))
        outpost_slots = torch.tensor(
            [slot(Team.RED, Role.OUTPOST), slot(Team.BLUE, Role.OUTPOST)],
            device=game.device,
        )
        result.yaw[:, outpost_slots] = game.outpost_angle_rad
        result.velocity_xy.copy_((result.position_xy - world.position_xy) / dt)
        result.vertical_velocity.copy_(vertical_velocity)
        result.yaw_rate.copy_(yaw_rate)
        new_terrain_contact = self.arena.terrain_contacts(result.position_xy)
        result.terrain_crossed.copy_(new_terrain_contact & ~world.terrain_contact)
        result.terrain_contact.copy_(new_terrain_contact)
        return result
