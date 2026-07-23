"""Normalized inputs produced by Torch or Isaac backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import PurchaseKind, RadarQuality
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


@dataclass
class RuleInputs:
    """All exogenous facts consumed by one referee tick."""

    shots_fired: Tensor
    hits: HitCandidates
    chassis_power_w: Tensor
    in_supply_zone: Tensor
    respawn_contact: Tensor

    purchase_unit: Tensor
    purchase_weapon: Tensor
    purchase_kind: Tensor
    purchase_ready: Tensor

    radar_quality: Tensor
    radar_update: Tensor

    @classmethod
    def empty(cls, state: GameState) -> "RuleInputs":
        env_unit = (state.num_envs, constants.UNIT_COUNT)
        env_team = (state.num_envs, constants.TEAM_COUNT)
        return cls(
            shots_fired=torch.zeros(
                (*env_unit, constants.WEAPON_COUNT),
                device=state.device,
                dtype=torch.long,
            ),
            hits=HitCandidates.empty(state),
            chassis_power_w=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            in_supply_zone=torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.bool,
            ),
            respawn_contact=torch.zeros(
                env_unit,
                device=state.device,
                dtype=torch.bool,
            ),
            purchase_unit=torch.full(
                env_team,
                -1,
                device=state.device,
                dtype=torch.long,
            ),
            purchase_weapon=torch.zeros(
                env_team,
                device=state.device,
                dtype=torch.long,
            ),
            purchase_kind=torch.full(
                env_team,
                PurchaseKind.NONE,
                device=state.device,
                dtype=torch.int8,
            ),
            purchase_ready=torch.zeros(
                env_team,
                device=state.device,
                dtype=torch.bool,
            ),
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
        )

    def validate(self, state: GameState, dt: float) -> None:
        if self.shots_fired.shape != (
            state.num_envs,
            constants.UNIT_COUNT,
            constants.WEAPON_COUNT,
        ):
            raise ValueError("shots_fired must have shape [env, unit, weapon]")
        if torch.any(self.shots_fired < 0):
            raise ValueError("shots_fired cannot be negative")
        if self.chassis_power_w.shape != (state.num_envs, constants.UNIT_COUNT):
            raise ValueError("chassis_power_w must have shape [env, unit]")
        if torch.any(self.chassis_power_w < 0):
            raise ValueError("chassis power cannot be negative")
        self.hits.validate(dt)
        valid_source = self.hits.source >= 0
        for weapon in range(constants.WEAPON_COUNT):
            source = self.hits.source[:, :, :, weapon, :]
            source_safe = torch.clamp(source, min=0)
            candidate_count = torch.zeros(
                (state.num_envs, constants.UNIT_COUNT),
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

        quality_is_valid = (
            (self.radar_quality == RadarQuality.WRONG)
            | (self.radar_quality == RadarQuality.HALF_ACCURATE)
            | (self.radar_quality == RadarQuality.ACCURATE)
        )
        if torch.any(self.radar_update & ~quality_is_valid):
            raise ValueError("updated radar quality must be wrong, half-accurate, or accurate")
