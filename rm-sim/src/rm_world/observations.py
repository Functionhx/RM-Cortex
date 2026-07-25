"""Oracle or belief entity tensors with deterministic action masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Weapon, unit_roles, unit_teams, weapon_capability
from rm_referee.state import GameState
from rm_world.arena import ArenaGeometry
from rm_world.belief import (
    ENTITY_SOURCE_COUNT,
    TARGET_ALIVE_INDEX,
    BeliefConfig,
    EntityBeliefTracker,
    EntityBeliefView,
    EntitySource,
    build_target_state,
)
from rm_world.geometry import NO_TARGET, RUNE_TARGET, resolve_target_slots
from rm_world.kinematics import KinematicState


BASE_ENTITY_FEATURE_NAMES = (
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
COVARIANCE_FEATURE_NAMES = (
    "covariance_x_x",
    "covariance_x_y",
    "covariance_x_vx",
    "covariance_x_vy",
    "covariance_y_y",
    "covariance_y_vx",
    "covariance_y_vy",
    "covariance_vx_vx",
    "covariance_vx_vy",
    "covariance_vy_vy",
)
SOURCE_FEATURE_NAMES = (
    "source_oracle",
    "source_shared",
    "source_local",
    "source_radar_confirmed",
    "source_radar_estimate",
    "source_prediction",
    "source_prior",
)
ENTITY_FEATURE_NAMES = (
    *BASE_ENTITY_FEATURE_NAMES,
    "position_observation_age_fraction",
    "state_observation_age_fraction",
    *COVARIANCE_FEATURE_NAMES,
    *SOURCE_FEATURE_NAMES,
)
ENTITY_DIM = len(ENTITY_FEATURE_NAMES)


@dataclass
class WorldObservation:
    agents: Tensor
    entities: Tensor
    entity_mask: Tensor
    entity_covariance: Tensor
    entity_position_age_s: Tensor
    entity_state_age_s: Tensor
    entity_source: Tensor
    target_mask: Tensor
    fire_mask: Tensor
    central: Tensor


@dataclass(frozen=True)
class VisibilityMasks:
    local_by_observer: Tensor
    local_by_team: Tensor
    shared_by_team: Tensor
    radar_confirmed_by_team: Tensor
    visible_by_observer: Tensor


class ObservationBuilder:
    """Build oracle or state-estimated actor observations."""

    def __init__(
        self,
        arena: ArenaGeometry | None = None,
        *,
        local_sensor_range_m: float = 12.0,
        mode: str = "oracle",
        belief_config: BeliefConfig | None = None,
    ) -> None:
        if mode not in {"oracle", "belief"}:
            raise ValueError("observation mode must be 'oracle' or 'belief'")
        self.arena = arena or ArenaGeometry()
        self.local_sensor_range_m = local_sensor_range_m
        self.mode = mode
        self.belief_config = belief_config or BeliefConfig()
        self.belief_config.validate()

    @property
    def agent_dim(self) -> int:
        return 34

    @property
    def entity_dim(self) -> int:
        return ENTITY_DIM

    def visibility(self, game: GameState, world: KinematicState) -> VisibilityMasks:
        """Return team-shared local, radar, and static knowledge masks."""

        roles = unit_roles(game.device)
        teams = unit_teams(game.device)
        observer_xy = world.position_xy[:, :, None, :]
        target_xy = world.position_xy[:, None, :, :]
        distance = torch.linalg.vector_norm(target_xy - observer_xy, dim=-1)
        los = self.arena.line_of_sight(
            observer_xy.expand(-1, -1, constants.UNIT_COUNT, -1),
            target_xy.expand(-1, constants.UNIT_COUNT, -1, -1),
        )
        observer_mobile = (
            (roles <= Role.SENTRY)[None, :]
            & game.alive
            & ~game.video_disabled
            & ~game.controller_offline
        )
        target_mobile = roles <= Role.SENTRY
        local_by_observer = (
            (distance <= self.local_sensor_range_m)
            & los
            & observer_mobile[:, :, None]
            & target_mobile[None, None, :]
        )
        local_by_team = local_by_observer.view(
            game.num_envs,
            constants.TEAM_COUNT,
            constants.ROLES_PER_TEAM,
            constants.UNIT_COUNT,
        ).any(dim=2)
        shared_by_team = EntityBeliefTracker.shared_mask(game.device)
        radar_confirmed_by_team = game.radar_truth_visible & target_mobile[None, None, :]
        known_by_team = shared_by_team[None, :, :] | local_by_team | radar_confirmed_by_team
        return VisibilityMasks(
            local_by_observer=local_by_observer,
            local_by_team=local_by_team,
            shared_by_team=shared_by_team,
            radar_confirmed_by_team=radar_confirmed_by_team,
            visible_by_observer=known_by_team[:, teams, :],
        )

    @staticmethod
    def _pack_covariance(covariance: Tensor) -> Tensor:
        scale = torch.tensor(
            (14.0, 7.5, 3.0, 3.0),
            device=covariance.device,
            dtype=covariance.dtype,
        )
        normalized = covariance / (scale[:, None] * scale[None, :])
        pairs = (
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        )
        return torch.stack(
            tuple(normalized[..., row, column] for row, column in pairs),
            dim=-1,
        )

    def build(
        self,
        game: GameState,
        world: KinematicState,
        *,
        belief: EntityBeliefView | None = None,
        visibility: VisibilityMasks | None = None,
    ) -> WorldObservation:
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

        visibility = visibility or self.visibility(game, world)
        same_team = teams[:, None] == teams[None, :]
        target_role = torch.nn.functional.one_hot(
            roles,
            constants.ROLES_PER_TEAM,
        ).to(game.dtype)
        if self.mode == "belief":
            if belief is None:
                raise ValueError("belief observation mode requires an EntityBeliefView")
            belief_view = belief
        else:
            truth_mean = torch.cat((world.position_xy, world.velocity_xy), dim=-1)
            truth_state = build_target_state(game, world)
            team_shape = (
                game.num_envs,
                constants.TEAM_COUNT,
                constants.UNIT_COUNT,
            )
            belief_view = EntityBeliefView(
                mean=truth_mean[:, None, :, :].expand(-1, constants.TEAM_COUNT, -1, -1),
                covariance=torch.zeros(
                    (*team_shape, 4, 4),
                    device=game.device,
                    dtype=game.dtype,
                ),
                target_state=truth_state[:, None, :, :].expand(
                    -1,
                    constants.TEAM_COUNT,
                    -1,
                    -1,
                ),
                valid=torch.ones(team_shape, device=game.device, dtype=torch.bool),
                position_age_s=torch.zeros(
                    team_shape,
                    device=game.device,
                    dtype=game.dtype,
                ),
                state_age_s=torch.zeros(
                    team_shape,
                    device=game.device,
                    dtype=game.dtype,
                ),
                source=torch.full(
                    team_shape,
                    EntitySource.ORACLE,
                    device=game.device,
                    dtype=torch.int8,
                ),
            )

        target_mean = belief_view.mean[:, teams, :, :]
        target_state = belief_view.target_state[:, teams, :, :]
        covariance = belief_view.covariance[:, teams, :, :, :]
        position_age_s = belief_view.position_age_s[:, teams, :]
        state_age_s = belief_view.state_age_s[:, teams, :]
        source = belief_view.source[:, teams, :]
        observer_xy = world.position_xy[:, :, None, :]
        relative = target_mean[..., :2] - observer_xy
        distance = torch.linalg.vector_norm(relative, dim=-1)
        observer_radar_progress = game.radar_p[:, teams, :]
        base_entity_parts = (
            relative[..., 0:1] / 28.0,
            relative[..., 1:2] / 15.0,
            (target_state[..., 0] - world.position_z[:, :, None])[..., None] / 5.0,
            (distance / 31.0)[..., None],
            target_state[..., 1:3],
            target_mean[..., 2:4] / 3.0,
            target_state[..., 3:],
            (observer_radar_progress / 150.0)[..., None],
            visibility.radar_confirmed_by_team[:, teams, :, None].to(game.dtype),
            same_team[None, :, :, None].to(game.dtype).expand(game.num_envs, -1, -1, -1),
            target_role[None, None, :, :].expand(
                game.num_envs,
                constants.UNIT_COUNT,
                -1,
                -1,
            ),
        )
        base_entities = torch.cat(base_entity_parts, dim=-1)
        if base_entities.shape[-1] != len(BASE_ENTITY_FEATURE_NAMES):
            raise RuntimeError("base entity feature schema is inconsistent")
        source_one_hot = torch.nn.functional.one_hot(
            source.to(torch.long),
            ENTITY_SOURCE_COUNT,
        ).to(game.dtype)
        entities = torch.cat(
            (
                base_entities,
                torch.clamp(
                    position_age_s / self.belief_config.age_feature_horizon_s,
                    max=1.0,
                )[..., None],
                torch.clamp(
                    state_age_s / self.belief_config.age_feature_horizon_s,
                    max=1.0,
                )[..., None],
                self._pack_covariance(covariance),
                source_one_hot,
            ),
            dim=-1,
        )
        if entities.shape[-1] != ENTITY_DIM:
            raise RuntimeError("entity feature schema is inconsistent")

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
            (target_state[..., TARGET_ALIVE_INDEX] >= 0.5),
            2,
            safe_target,
        )
        target_visible_by_choice = torch.gather(
            visibility.visible_by_observer,
            2,
            safe_target,
        )
        targetable = (
            target_alive_by_choice
            if self.mode == "oracle"
            else target_alive_by_choice & target_visible_by_choice
        )
        target_mask = (choice < RUNE_TARGET) & targetable
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
            entity_mask=visibility.visible_by_observer,
            entity_covariance=covariance,
            entity_position_age_s=position_age_s,
            entity_state_age_s=state_age_s,
            entity_source=source,
            target_mask=target_mask,
            fire_mask=fire_mask,
            central=central,
        )
