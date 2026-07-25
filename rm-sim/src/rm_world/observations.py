"""Oracle entity tensors, visibility metadata, and deterministic action masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Weapon, unit_roles, unit_teams, weapon_capability
from rm_referee.state import GameState
from rm_world.arena import ArenaGeometry
from rm_world.geometry import NO_TARGET, RUNE_TARGET, resolve_target_slots
from rm_world.kinematics import KinematicState


ENTITY_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "distance_xy",
    "target_yaw_sin",
    "target_yaw_cos",
    "target_velocity_x",
    "target_velocity_y",
    "target_hp_fraction",
    "target_alive",
    "target_level_fraction",
    "target_xp_fraction",
    "target_heat_17mm_fraction",
    "target_heat_42mm_fraction",
    "target_ammo_17mm_fraction",
    "target_ammo_42mm_fraction",
    "target_chassis_energy_fraction",
    "target_power_buffer_fraction",
    "target_weak",
    "target_invulnerable_fraction",
    "target_effective_vulnerability",
    "observer_radar_progress",
    "observer_radar_truth_visible",
    "same_team",
    "target_role_hero",
    "target_role_engineer",
    "target_role_infantry_3",
    "target_role_infantry_4",
    "target_role_aerial",
    "target_role_sentry",
    "target_role_base",
    "target_role_outpost",
)
ENTITY_DIM = len(ENTITY_FEATURE_NAMES)


@dataclass
class WorldObservation:
    agents: Tensor
    entities: Tensor
    entity_mask: Tensor
    target_mask: Tensor
    fire_mask: Tensor
    central: Tensor


class ObservationBuilder:
    """Build full-truth entity observations and a privileged critic state.

    ``entities`` always contains oracle state; ``entity_mask`` separately records
    which observer-target pairs are legally visible.
    """

    def __init__(
        self,
        arena: ArenaGeometry | None = None,
        *,
        local_sensor_range_m: float = 12.0,
    ) -> None:
        self.arena = arena or ArenaGeometry()
        self.local_sensor_range_m = local_sensor_range_m

    @property
    def agent_dim(self) -> int:
        return 34

    @property
    def entity_dim(self) -> int:
        return ENTITY_DIM

    def build(self, game: GameState, world: KinematicState) -> WorldObservation:
        roles = unit_roles(game.device)
        teams = unit_teams(game.device)
        role_one_hot = torch.nn.functional.one_hot(
            roles,
            constants.ROLES_PER_TEAM,
        ).to(game.dtype)
        hp_fraction = game.hp / torch.clamp(game.max_hp, min=1)
        heat_fraction = game.heat / torch.clamp(game.heat_limit, min=1)
        own_coin = game.team_coin[:, teams].to(game.dtype) / 2000.0
        own_level_cap = game.team_level_cap[:, teams].to(game.dtype) / 10.0
        own_base_hp = game.hp[:, (6, 14)][:, teams] / 5000.0
        own_outpost_hp = game.hp[:, (7, 15)][:, teams] / 1500.0
        enemy_base_hp = game.hp[:, (14, 6)][:, teams] / 5000.0
        enemy_outpost_hp = game.hp[:, (15, 7)][:, teams] / 1500.0
        agent_parts = (
            world.position_xy[..., 0:1] / 14.0,
            world.position_xy[..., 1:2] / 7.5,
            world.position_z[..., None] / 5.0,
            torch.sin(world.yaw)[..., None],
            torch.cos(world.yaw)[..., None],
            world.velocity_xy / 3.0,
            hp_fraction[..., None],
            game.alive.to(game.dtype)[..., None],
            game.level.to(game.dtype)[..., None] / 10.0,
            game.xp[..., None] / 5000.0,
            heat_fraction,
            torch.clamp(game.ammo.to(game.dtype) / 750.0, max=1.0),
            game.chassis_energy[..., None] / constants.CHASSIS_ENERGY_MAX,
            game.power_buffer_j[..., None] / constants.POWER_BUFFER_MAX_J,
            game.weak.to(game.dtype)[..., None],
            torch.clamp(game.invulnerable_s[..., None] / 30.0, max=1.0),
            own_coin[..., None],
            own_level_cap[..., None],
            own_base_hp[..., None],
            own_outpost_hp[..., None],
            enemy_base_hp[..., None],
            enemy_outpost_hp[..., None],
            ((constants.MATCH_DURATION_S - game.elapsed_s) / constants.MATCH_DURATION_S)[
                :, None, None
            ].expand(-1, constants.UNIT_COUNT, -1),
            role_one_hot[None, :, :].expand(game.num_envs, -1, -1),
        )
        agents = torch.cat(agent_parts, dim=-1)

        observer_xy = world.position_xy[:, :, None, :]
        target_xy = world.position_xy[:, None, :, :]
        relative = target_xy - observer_xy
        distance = torch.linalg.vector_norm(relative, dim=-1)
        los = self.arena.line_of_sight(
            observer_xy.expand(-1, -1, constants.UNIT_COUNT, -1),
            target_xy.expand(-1, constants.UNIT_COUNT, -1, -1),
        )
        same_team = teams[:, None] == teams[None, :]
        target_is_building = (roles == Role.BASE) | (roles == Role.OUTPOST)
        # V1.5.0 pp. 117–119: radar-confirmed truth is scoped to the observing team.
        radar_visible = game.radar_truth_visible[:, teams, :]
        visible = (
            same_team[None, :, :]
            | target_is_building[None, None, :]
            | ((distance <= self.local_sensor_range_m) & los)
            | radar_visible
        )
        visible &= game.alive[:, None, :] | target_is_building[None, None, :]

        target_role = torch.nn.functional.one_hot(
            roles,
            constants.ROLES_PER_TEAM,
        ).to(game.dtype)
        target_yaw = world.yaw[:, None, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
        )
        target_velocity = world.velocity_xy[:, None, :, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
            -1,
        )
        target_hp = hp_fraction[:, None, :].expand(-1, constants.UNIT_COUNT, -1)
        target_alive = game.alive[:, None, :].expand(-1, constants.UNIT_COUNT, -1)
        target_level = game.level[:, None, :].expand(-1, constants.UNIT_COUNT, -1)
        target_xp = game.xp[:, None, :].expand(-1, constants.UNIT_COUNT, -1)
        target_heat = heat_fraction[:, None, :, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
            -1,
        )
        target_ammo = game.ammo[:, None, :, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
            -1,
        )
        target_chassis_energy = game.chassis_energy[:, None, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
        )
        target_power_buffer = game.power_buffer_j[:, None, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
        )
        target_weak = game.weak[:, None, :].expand(-1, constants.UNIT_COUNT, -1)
        target_invulnerable = game.invulnerable_s[:, None, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
        )
        effective_vulnerability = torch.maximum(
            game.vulnerability_fraction,
            game.radar_vulnerability_fraction,
        )
        target_vulnerability = effective_vulnerability[:, None, :].expand(
            -1,
            constants.UNIT_COUNT,
            -1,
        )
        observer_radar_progress = game.radar_p[:, teams, :]
        entity_parts = (
            relative[..., 0:1] / 28.0,
            relative[..., 1:2] / 15.0,
            (world.position_z[:, None, :] - world.position_z[:, :, None])[..., None] / 5.0,
            (distance / 31.0)[..., None],
            torch.sin(target_yaw)[..., None],
            torch.cos(target_yaw)[..., None],
            target_velocity / 3.0,
            target_hp[..., None],
            target_alive.to(game.dtype)[..., None],
            target_level.to(game.dtype)[..., None] / 10.0,
            target_xp[..., None] / 5000.0,
            target_heat,
            torch.clamp(target_ammo.to(game.dtype) / 750.0, max=1.0),
            target_chassis_energy[..., None] / constants.CHASSIS_ENERGY_MAX,
            target_power_buffer[..., None] / constants.POWER_BUFFER_MAX_J,
            target_weak.to(game.dtype)[..., None],
            torch.clamp(target_invulnerable[..., None] / 30.0, max=1.0),
            target_vulnerability[..., None],
            (observer_radar_progress / 150.0)[..., None],
            radar_visible.to(game.dtype)[..., None],
            same_team[None, :, :, None].to(game.dtype).expand(game.num_envs, -1, -1, -1),
            target_role[None, None, :, :].expand(
                game.num_envs,
                constants.UNIT_COUNT,
                -1,
                -1,
            ),
        )
        entities = torch.cat(entity_parts, dim=-1)

        choice = (
            torch.arange(9, device=game.device)
            .view(1, 1, 9)
            .expand(
                game.num_envs,
                constants.UNIT_COUNT,
                -1,
            )
        )
        target_slots = resolve_target_slots(choice)
        safe_target = torch.clamp(target_slots, min=0)
        target_alive_by_choice = torch.gather(
            game.alive[:, None, :].expand(-1, constants.UNIT_COUNT, -1),
            2,
            safe_target,
        )
        target_mask = (choice < RUNE_TARGET) & target_alive_by_choice
        outpost_alive = target_mask[:, :, 6]
        target_mask[:, :, 5] &= ~outpost_alive
        own_rune_active = (game.rune_mode[:, teams] != 0) & (
            game.rune_trigger_remaining_s[:, teams] > 0
        )
        target_mask[:, :, RUNE_TARGET] = own_rune_active
        target_mask[:, :, NO_TARGET] = True

        weapon = torch.where(
            roles == Role.HERO,
            torch.full(
                (constants.UNIT_COUNT,),
                Weapon.MM42,
                device=game.device,
                dtype=torch.long,
            ),
            torch.zeros(constants.UNIT_COUNT, device=game.device, dtype=torch.long),
        )
        launcher = weapon_capability(game.device)[
            torch.arange(constants.UNIT_COUNT, device=game.device),
            weapon,
        ]
        fire_mask = (
            game.alive
            & launcher[None, :]
            & ~game.weak
            & ~game.controller_offline
            & (
                game.heat_lock.gather(
                    2, weapon[None, :, None].expand(game.num_envs, -1, 1)
                ).squeeze(-1)
                == 0
            )
            & (
                game.velocity_lock_s.gather(
                    2,
                    weapon[None, :, None].expand(game.num_envs, -1, 1),
                ).squeeze(-1)
                <= 0
            )
        )

        central = torch.cat(
            (
                world.position_xy.reshape(game.num_envs, -1) / 14.0,
                world.position_z / 5.0,
                hp_fraction,
                game.alive.to(game.dtype),
                game.ammo.to(game.dtype).reshape(game.num_envs, -1) / 750.0,
                game.team_coin.to(game.dtype) / 2000.0,
                game.team_damage / 10000.0,
                game.elapsed_s[:, None] / constants.MATCH_DURATION_S,
            ),
            dim=-1,
        )
        return WorldObservation(
            agents=agents,
            entities=entities,
            entity_mask=visible,
            target_mask=target_mask,
            fire_mask=fire_mask,
            central=central,
        )
