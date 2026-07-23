"""Conversion from high-level world actions to normalized referee facts."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.inputs import RuleInputs
from rm_referee.random_tape import RandomTape
from rm_referee.schema import (
    RadarQuality,
    Role,
    Team,
    Weapon,
    Zone,
    slot,
    unit_roles,
    weapon_capability,
)
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.arena import ArenaGeometry
from rm_world.geometry import (
    NO_TARGET,
    RUNE_TARGET,
    HitModel,
    solve_armor_geometry,
)
from rm_world.kinematics import KinematicCommands, KinematicState


class TorchRuleBackend:
    """Generate LOS, armor, zone, and command facts without game-rule duplication."""

    def __init__(
        self,
        arena: ArenaGeometry | None = None,
        hit_model: HitModel | None = None,
    ) -> None:
        self.arena = arena or ArenaGeometry()
        self.hit_model = hit_model or HitModel()

    def kinematic_commands(
        self,
        game: GameState,
        actions: WorldActions,
    ) -> KinematicCommands:
        actions.validate(game)
        return KinematicCommands(
            body_velocity_xy=actions.body_velocity_xy,
            yaw_rate=actions.yaw_rate,
            vertical_velocity=actions.vertical_velocity,
        )

    def _pack_hits(
        self,
        inputs: RuleInputs,
        solution_target: torch.Tensor,
        armor_index: torch.Tensor,
        weapon: torch.Tensor,
        hit: torch.Tensor,
        critical: torch.Tensor,
        dt: float,
    ) -> None:
        num_envs = inputs.shots_fired.shape[0]
        source_index = torch.arange(constants.UNIT_COUNT, device=hit.device)
        target_safe = torch.clamp(solution_target, min=0)
        key = (
            target_safe * constants.MAX_ARMOR_COUNT * constants.WEAPON_COUNT
            + armor_index * constants.WEAPON_COUNT
            + weapon
        )
        same_key = key[:, :, None] == key[:, None, :]
        previous_source = source_index[None, :, None] > source_index[None, None, :]
        rank = (same_key & previous_source & hit[:, None, :]).sum(dim=-1).to(torch.long)
        hit &= rank < constants.MAX_HIT_CANDIDATES
        destination = key * constants.MAX_HIT_CANDIDATES + torch.clamp(
            rank,
            max=constants.MAX_HIT_CANDIDATES - 1,
        )
        total = (
            constants.UNIT_COUNT
            * constants.MAX_ARMOR_COUNT
            * constants.WEAPON_COUNT
            * constants.MAX_HIT_CANDIDATES
        )
        source_value = torch.where(
            hit,
            source_index[None, :].expand(num_envs, -1),
            torch.full_like(destination, -1),
        )
        source_flat = torch.full(
            (num_envs, total),
            -1,
            device=hit.device,
            dtype=torch.long,
        )
        source_flat.scatter_reduce_(
            1,
            destination,
            source_value,
            reduce="amax",
            include_self=True,
        )
        offset = ((source_index.to(inputs.shots_fired.dtype) + 0.5) / constants.UNIT_COUNT * dt).to(
            inputs.hits.time_offset_s.dtype
        )
        time_value = torch.where(
            hit,
            offset[None, :],
            torch.zeros(
                (num_envs, constants.UNIT_COUNT),
                device=hit.device,
                dtype=inputs.hits.time_offset_s.dtype,
            ),
        )
        time_flat = torch.zeros(
            (num_envs, total),
            device=hit.device,
            dtype=inputs.hits.time_offset_s.dtype,
        )
        time_flat.scatter_reduce_(
            1,
            destination,
            time_value,
            reduce="amax",
            include_self=True,
        )
        critical_flat = torch.zeros(
            (num_envs, total),
            device=hit.device,
            dtype=torch.long,
        )
        critical_flat.scatter_reduce_(
            1,
            destination,
            (critical & hit).to(torch.long),
            reduce="amax",
            include_self=True,
        )
        inputs.hits.source.copy_(source_flat.reshape_as(inputs.hits.source))
        inputs.hits.time_offset_s.copy_(time_flat.reshape_as(inputs.hits.time_offset_s))
        inputs.hits.critical.copy_(critical_flat.reshape_as(inputs.hits.critical).to(torch.bool))

    def rule_inputs(
        self,
        world: KinematicState,
        state: GameState,
        actions: WorldActions,
        random_tape: RandomTape,
        dt: float,
    ) -> RuleInputs:
        actions.validate(state)
        inputs = RuleInputs.empty(state)
        roles = unit_roles(state.device)
        capability = weapon_capability(state.device)
        weapon = torch.where(
            roles == Role.HERO,
            torch.full(
                (constants.UNIT_COUNT,),
                Weapon.MM42,
                device=state.device,
                dtype=torch.long,
            ),
            torch.zeros(constants.UNIT_COUNT, device=state.device, dtype=torch.long),
        )
        can_command_fire = capability[
            torch.arange(constants.UNIT_COUNT, device=state.device),
            weapon,
        ]
        shot = actions.fire & can_command_fire[None, :] & (actions.target != NO_TARGET)
        shot_one_hot = functional.one_hot(
            weapon,
            constants.WEAPON_COUNT,
        ).to(torch.long)
        inputs.shots_fired.copy_(shot[:, :, None].to(torch.long) * shot_one_hot[None, :, :])
        inputs.muzzle_velocity_mps.copy_(
            state.muzzle_velocity_limit_mps
            * actions.muzzle_speed_scale[:, :, None]
            * inputs.shots_fired.to(state.dtype)
        )

        solution = solve_armor_geometry(world, state, actions.target, self.arena)
        draws = random_tape.take(constants.UNIT_COUNT * 2).reshape(
            state.num_envs,
            constants.UNIT_COUNT,
            2,
        )
        probability = self.hit_model.probability(solution)
        normal_target = actions.target < RUNE_TARGET
        hit = shot & normal_target & solution.target_valid & (draws[:, :, 0] < probability)
        critical = draws[:, :, 1] < self.hit_model.config.critical_probability
        self._pack_hits(
            inputs,
            solution.target_slot,
            solution.armor_index,
            weapon[None, :].expand(state.num_envs, -1),
            hit,
            critical,
            dt,
        )

        speed = torch.linalg.vector_norm(world.velocity_xy, dim=-1)
        chassis_roles = (
            (roles == Role.HERO)
            | (roles == Role.ENGINEER)
            | (roles == Role.INFANTRY_3)
            | (roles == Role.INFANTRY_4)
            | (roles == Role.SENTRY)
        )
        inputs.chassis_power_w.copy_(
            torch.where(
                chassis_roles[None, :] & state.alive,
                5.0 + 8.0 * torch.square(speed),
                torch.zeros_like(speed),
            )
        )
        inputs.supercap_input_w.copy_(actions.supercap_input_w)

        occupancy = self.arena.zone_occupancy(world.position_xy)
        inputs.zone_occupancy.copy_(occupancy)
        inputs.in_supply_zone.copy_(occupancy[:, :, Zone.SUPPLY])
        inputs.in_ammo_zone.copy_(
            occupancy[:, :, Zone.SUPPLY]
            | occupancy[:, :, Zone.BASE]
            | occupancy[:, :, Zone.OUTPOST]
        )
        inputs.in_outpost_zone.copy_(occupancy[:, :, Zone.OUTPOST])
        inputs.respawn_contact.copy_(
            occupancy[:, :, Zone.SUPPLY]
            | occupancy[:, :, Zone.BASE]
            | occupancy[:, :, Zone.OUTPOST]
        )
        inputs.in_hero_deploy_zone.copy_(self.arena.hero_deploy_zone(world.position_xy))
        inputs.terrain_crossed.copy_(world.terrain_crossed)

        inputs.purchase_unit.copy_(actions.purchase_unit)
        inputs.purchase_weapon.copy_(actions.purchase_weapon)
        inputs.purchase_kind.copy_(actions.purchase_kind)
        inputs.remote_heal_request.copy_(actions.remote_heal)
        inputs.immediate_respawn_request.copy_(actions.immediate_respawn)
        inputs.sentry_claim_ammo.copy_(actions.sentry_claim_ammo)
        inputs.hero_deploy_request.copy_(actions.hero_deploy)
        inputs.sentry_stance_request.copy_(actions.sentry_stance)
        inputs.sentry_operator_command.copy_(actions.sentry_operator_command)
        inputs.aerial_support_request.copy_(actions.aerial_support)
        aerial_slots = torch.tensor(
            [slot(Team.RED, Role.AERIAL), slot(Team.BLUE, Role.AERIAL)],
            device=state.device,
        )
        inputs.aerial_on_pad.copy_(world.position_z[:, aerial_slots] <= 0.05)
        inputs.radar_illuminating.copy_(actions.radar_illuminate)
        inputs.outpost_rebuild_request.copy_(actions.rebuild_outpost)
        inputs.tech_complete_level.copy_(actions.tech_complete_level)
        inputs.rune_trigger.copy_(actions.rune_trigger)

        rune_shot = shot & (actions.target == RUNE_TARGET)
        rune_success = rune_shot & (draws[:, :, 0] < 0.85)
        rune_success_by_team = rune_success.view(
            state.num_envs,
            constants.TEAM_COUNT,
            constants.ROLES_PER_TEAM,
        )
        rune_hits = torch.clamp(rune_success_by_team.sum(dim=-1), max=2)
        ring = torch.floor(draws[:, :, 1] * 10.0).to(torch.long) + 1
        ring_by_team = (
            (ring * rune_success.to(torch.long))
            .view(
                state.num_envs,
                constants.TEAM_COUNT,
                constants.ROLES_PER_TEAM,
            )
            .sum(dim=-1)
        )
        inputs.rune_group_hits.copy_(rune_hits)
        inputs.rune_group_correct.copy_(rune_hits > 0)
        inputs.rune_group_ring_sum.copy_(ring_by_team.to(state.dtype))

        inputs.dart_open_gate.copy_(actions.dart_open_gate)
        inputs.dart_close_gate.copy_(actions.dart_close_gate)
        inputs.dart_fire_target.copy_(actions.dart_target)
        dart_draw = random_tape.take(constants.TEAM_COUNT)
        inputs.dart_hit.copy_(
            (actions.dart_target >= 0) & (dart_draw < self.hit_model.config.dart_hit_probability)
        )

        radar_target = actions.radar_target
        radar_safe = torch.clamp(radar_target, min=0, max=constants.UNIT_COUNT - 1)
        actual_xy = torch.gather(
            world.position_xy,
            1,
            radar_safe[:, :, None].expand(-1, -1, 2),
        )
        radar_error = torch.linalg.vector_norm(
            actions.radar_report_xy - actual_xy,
            dim=-1,
        )
        radar_quality = torch.where(
            radar_error < 0.8,
            torch.full_like(radar_safe, RadarQuality.ACCURATE),
            torch.where(
                radar_error < 1.6,
                torch.full_like(radar_safe, RadarQuality.HALF_ACCURATE),
                torch.full_like(radar_safe, RadarQuality.WRONG),
            ),
        )
        radar_valid = radar_target >= 0
        inputs.radar_update.scatter_(
            2,
            radar_safe[:, :, None],
            radar_valid[:, :, None],
        )
        inputs.radar_quality.scatter_(
            2,
            radar_safe[:, :, None],
            radar_quality[:, :, None].to(torch.int8),
        )
        inputs.radar_double_request.copy_(actions.radar_double)
        inputs.radar_key_solved.copy_(actions.radar_key_solved)
        return inputs
