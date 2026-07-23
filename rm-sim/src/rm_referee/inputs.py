"""Normalized inputs produced by Torch or Isaac backends."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import (
    DartTarget,
    PurchaseKind,
    RadarQuality,
    RuneMode,
    SentryStance,
)
from rm_referee.state import GameState


@dataclass
class HitCandidates:
    """Chronological hit candidates grouped by target armor and weapon.

    Shapes are ``[env, target, armor, weapon, candidate]``. A source of ``-1``
    marks an unused candidate. Backends must sort valid candidates by
    ``time_offset_s`` and raise on overflow; the fixed capacity is an interface
    bound, not a weapon fire-rate rule.
    """

    source: Tensor
    time_offset_s: Tensor
    critical: Tensor

    @classmethod
    def empty(cls, state: GameState) -> "HitCandidates":
        shape = (
            state.num_envs,
            constants.UNIT_COUNT,
            constants.MAX_ARMOR_COUNT,
            constants.WEAPON_COUNT,
            constants.MAX_HIT_CANDIDATES,
        )
        return cls(
            source=torch.full(shape, -1, device=state.device, dtype=torch.long),
            time_offset_s=torch.zeros(shape, device=state.device, dtype=state.dtype),
            critical=torch.zeros(shape, device=state.device, dtype=torch.bool),
        )

    def validate(self, dt: float) -> None:
        expected_tail = (
            constants.UNIT_COUNT,
            constants.MAX_ARMOR_COUNT,
            constants.WEAPON_COUNT,
            constants.MAX_HIT_CANDIDATES,
        )
        if self.source.shape[1:] != expected_tail:
            raise ValueError("hit candidate tensor has an invalid shape")
        valid = self.source >= 0
        if torch.any(self.source[valid] >= constants.UNIT_COUNT):
            raise ValueError("hit source is outside the unit schema")
        if torch.any(self.time_offset_s[valid] < 0) or torch.any(self.time_offset_s[valid] >= dt):
            raise ValueError("hit offsets must be within [0, dt)")
        ordered = torch.where(
            valid,
            self.time_offset_s,
            torch.full_like(self.time_offset_s, torch.inf),
        )
        if torch.any(ordered[..., 1:] < ordered[..., :-1]):
            raise ValueError("hit candidates must be chronological and packed")

    def clone(self) -> "HitCandidates":
        """Return a value copy of the candidate tensors."""

        return HitCandidates(
            source=self.source.clone(),
            time_offset_s=self.time_offset_s.clone(),
            critical=self.critical.clone(),
        )


@dataclass
class RuleInputs:
    """All exogenous facts consumed by one referee tick."""

    shots_fired: Tensor
    hits: HitCandidates
    chassis_power_w: Tensor
    supercap_input_w: Tensor
    muzzle_velocity_mps: Tensor

    in_supply_zone: Tensor
    in_ammo_zone: Tensor
    in_hero_deploy_zone: Tensor
    in_outpost_zone: Tensor
    respawn_contact: Tensor
    zone_occupancy: Tensor
    terrain_crossed: Tensor

    purchase_unit: Tensor
    purchase_weapon: Tensor
    purchase_kind: Tensor
    purchase_ready: Tensor
    remote_heal_request: Tensor
    immediate_respawn_request: Tensor
    sentry_claim_ammo: Tensor

    hero_deploy_request: Tensor
    sentry_stance_request: Tensor
    sentry_operator_command: Tensor
    aerial_support_request: Tensor
    aerial_on_pad: Tensor
    radar_illuminating: Tensor
    outpost_rebuild_request: Tensor

    controller_offline: Tensor
    offline_module_count: Tensor
    speed_module_offline: Tensor
    localization_module_offline: Tensor
    aerial_laser_module_offline: Tensor

    tech_complete_level: Tensor
    rune_trigger: Tensor
    rune_group_correct: Tensor
    rune_group_hits: Tensor
    rune_group_ring_sum: Tensor

    dart_open_gate: Tensor
    dart_close_gate: Tensor
    dart_fire_target: Tensor
    dart_hit: Tensor

    radar_quality: Tensor
    radar_update: Tensor
    radar_double_request: Tensor
    radar_key_solved: Tensor

    @classmethod
    def empty(cls, state: GameState) -> "RuleInputs":
        env_unit = (state.num_envs, constants.UNIT_COUNT)
        env_team = (state.num_envs, constants.TEAM_COUNT)
        env_weapon = (*env_unit, constants.WEAPON_COUNT)
        return cls(
            shots_fired=torch.zeros(env_weapon, device=state.device, dtype=torch.long),
            hits=HitCandidates.empty(state),
            chassis_power_w=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            supercap_input_w=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            muzzle_velocity_mps=torch.zeros(env_weapon, device=state.device, dtype=state.dtype),
            in_supply_zone=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            in_ammo_zone=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            in_hero_deploy_zone=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            in_outpost_zone=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            respawn_contact=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            zone_occupancy=torch.zeros(
                (*env_unit, constants.ZONE_COUNT),
                device=state.device,
                dtype=torch.bool,
            ),
            terrain_crossed=torch.zeros(
                (*env_unit, constants.TERRAIN_COUNT),
                device=state.device,
                dtype=torch.bool,
            ),
            purchase_unit=torch.full(env_team, -1, device=state.device, dtype=torch.long),
            purchase_weapon=torch.zeros(env_team, device=state.device, dtype=torch.long),
            purchase_kind=torch.full(
                env_team,
                PurchaseKind.NONE,
                device=state.device,
                dtype=torch.int8,
            ),
            purchase_ready=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            remote_heal_request=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            immediate_respawn_request=torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.bool,
            ),
            sentry_claim_ammo=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            hero_deploy_request=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            sentry_stance_request=torch.full(
                env_team,
                -1,
                device=state.device,
                dtype=torch.int8,
            ),
            sentry_operator_command=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            aerial_support_request=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            aerial_on_pad=torch.ones(env_team, device=state.device, dtype=torch.bool),
            radar_illuminating=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            outpost_rebuild_request=torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.bool,
            ),
            controller_offline=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            offline_module_count=torch.zeros(env_unit, device=state.device, dtype=torch.long),
            speed_module_offline=torch.zeros(
                env_weapon,
                device=state.device,
                dtype=torch.bool,
            ),
            localization_module_offline=torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.bool,
            ),
            aerial_laser_module_offline=torch.zeros(
                env_team,
                device=state.device,
                dtype=torch.bool,
            ),
            tech_complete_level=torch.zeros(env_team, device=state.device, dtype=torch.int8),
            rune_trigger=torch.full(
                env_team,
                RuneMode.IDLE,
                device=state.device,
                dtype=torch.int8,
            ),
            rune_group_correct=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            rune_group_hits=torch.zeros(env_team, device=state.device, dtype=torch.long),
            rune_group_ring_sum=torch.zeros(env_team, device=state.device, dtype=state.dtype),
            dart_open_gate=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            dart_close_gate=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            dart_fire_target=torch.full(
                env_team,
                -1,
                device=state.device,
                dtype=torch.int8,
            ),
            dart_hit=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            radar_quality=torch.full(
                (state.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                RadarQuality.UNKNOWN,
                device=state.device,
                dtype=torch.int8,
            ),
            radar_update=torch.zeros(
                (state.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT),
                device=state.device,
                dtype=torch.bool,
            ),
            radar_double_request=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            radar_key_solved=torch.zeros(env_team, device=state.device, dtype=torch.bool),
        )

    def validate(self, state: GameState, dt: float) -> None:
        env_unit = (state.num_envs, constants.UNIT_COUNT)
        env_team = (state.num_envs, constants.TEAM_COUNT)
        env_weapon = (*env_unit, constants.WEAPON_COUNT)
        if self.shots_fired.shape != env_weapon:
            raise ValueError("shots_fired must have shape [env, unit, weapon]")
        if torch.any(self.shots_fired < 0):
            raise ValueError("shots_fired cannot be negative")
        if self.chassis_power_w.shape != env_unit:
            raise ValueError("chassis_power_w must have shape [env, unit]")
        if torch.any(self.chassis_power_w < 0) or torch.any(self.supercap_input_w < 0):
            raise ValueError("power inputs cannot be negative")
        if torch.any(self.muzzle_velocity_mps < 0):
            raise ValueError("muzzle velocity cannot be negative")
        if torch.any(self.offline_module_count < 0):
            raise ValueError("offline module count cannot be negative")
        self.hits.validate(dt)

        valid_source = self.hits.source >= 0
        for weapon in range(constants.WEAPON_COUNT):
            source = self.hits.source[:, :, :, weapon, :]
            source_safe = torch.clamp(source, min=0)
            candidate_count = torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.long,
            )
            candidate_count.scatter_add_(
                1,
                source_safe.reshape(state.num_envs, -1),
                valid_source[:, :, :, weapon, :].reshape(state.num_envs, -1).to(torch.long),
            )
            if torch.any(candidate_count > self.shots_fired[:, :, weapon]):
                raise ValueError("hit candidates exceed detected shots for a source")

        valid_kinds = (
            (self.purchase_kind == PurchaseKind.NONE)
            | (self.purchase_kind == PurchaseKind.LOCAL)
            | (self.purchase_kind == PurchaseKind.REMOTE)
        )
        if not torch.all(valid_kinds):
            raise ValueError("unknown purchase kind")
        if torch.any(self.purchase_unit < -1) or torch.any(
            self.purchase_unit >= constants.UNIT_COUNT
        ):
            raise ValueError("purchase unit is outside the unit schema")
        if torch.any(self.purchase_weapon < 0) or torch.any(
            self.purchase_weapon >= constants.WEAPON_COUNT
        ):
            raise ValueError("purchase weapon is outside the weapon schema")
        if torch.any(self.sentry_stance_request < -1) or torch.any(
            self.sentry_stance_request > SentryStance.MOBILE
        ):
            raise ValueError("unknown sentry stance")
        if torch.any(self.tech_complete_level < 0) or torch.any(
            self.tech_complete_level > constants.TECH_LEVEL_COUNT
        ):
            raise ValueError("unknown technology-core level")
        if torch.any(self.rune_trigger < RuneMode.IDLE) or torch.any(
            self.rune_trigger > RuneMode.LARGE
        ):
            raise ValueError("unknown energy mechanism mode")
        if torch.any(self.rune_group_hits < 0) or torch.any(self.rune_group_hits > 2):
            raise ValueError("a rune group can accept zero, one, or two hits")
        if torch.any(self.dart_fire_target < -1) or torch.any(
            self.dart_fire_target > DartTarget.BASE_TERMINAL_MOVING
        ):
            raise ValueError("unknown dart target")

        quality_is_valid = (
            (self.radar_quality == RadarQuality.WRONG)
            | (self.radar_quality == RadarQuality.HALF_ACCURATE)
            | (self.radar_quality == RadarQuality.ACCURATE)
        )
        if torch.any(self.radar_update & ~quality_is_valid):
            raise ValueError("updated radar quality must be wrong, half-accurate, or accurate")
        if self.zone_occupancy.shape != (*env_unit, constants.ZONE_COUNT):
            raise ValueError("zone_occupancy has an invalid shape")
        if self.terrain_crossed.shape != (*env_unit, constants.TERRAIN_COUNT):
            raise ValueError("terrain_crossed has an invalid shape")
        if self.tech_complete_level.shape != env_team:
            raise ValueError("team command tensors must have shape [env, team]")

    def clone(self) -> "RuleInputs":
        """Copy a normalized input frame without sharing mutable tensors."""

        values: dict[str, Tensor | HitCandidates] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.clone()
        return RuleInputs(**values)  # type: ignore[arg-type]
