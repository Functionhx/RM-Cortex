"""Heterogeneous high-level actions for the Torch world."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import PurchaseKind, RuneMode
from rm_referee.state import GameState


@dataclass
class WorldActions:
    """One policy command frame.

    Unit tensors use the stable 16-slot schema. Team tensors hold low-frequency
    hero, sentry, aerial, dart, radar, and technology commands.
    """

    body_velocity_xy: Tensor
    yaw_rate: Tensor
    vertical_velocity: Tensor
    target: Tensor
    fire: Tensor
    muzzle_speed_scale: Tensor
    supercap_input_w: Tensor

    purchase_unit: Tensor
    purchase_weapon: Tensor
    purchase_kind: Tensor
    remote_heal: Tensor
    immediate_respawn: Tensor
    sentry_claim_ammo: Tensor

    hero_deploy: Tensor
    sentry_stance: Tensor
    sentry_operator_command: Tensor
    aerial_support: Tensor
    rebuild_outpost: Tensor

    tech_complete_level: Tensor
    rune_trigger: Tensor
    dart_open_gate: Tensor
    dart_close_gate: Tensor
    dart_target: Tensor

    radar_target: Tensor
    radar_report_xy: Tensor
    radar_illuminate: Tensor
    radar_double: Tensor
    radar_key_solved: Tensor

    @classmethod
    def zeros(cls, game: GameState) -> "WorldActions":
        env_unit = (game.num_envs, constants.UNIT_COUNT)
        env_team = (game.num_envs, constants.TEAM_COUNT)
        return cls(
            body_velocity_xy=torch.zeros(
                (*env_unit, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            yaw_rate=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            vertical_velocity=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            target=torch.full(env_unit, 8, device=game.device, dtype=torch.long),
            fire=torch.zeros(env_unit, device=game.device, dtype=torch.bool),
            muzzle_speed_scale=torch.ones(env_unit, device=game.device, dtype=game.dtype),
            supercap_input_w=torch.zeros(env_unit, device=game.device, dtype=game.dtype),
            purchase_unit=torch.full(env_team, -1, device=game.device, dtype=torch.long),
            purchase_weapon=torch.zeros(env_team, device=game.device, dtype=torch.long),
            purchase_kind=torch.full(
                env_team,
                PurchaseKind.NONE,
                device=game.device,
                dtype=torch.int8,
            ),
            remote_heal=torch.zeros(env_unit, device=game.device, dtype=torch.bool),
            immediate_respawn=torch.zeros(env_unit, device=game.device, dtype=torch.bool),
            sentry_claim_ammo=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            hero_deploy=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            sentry_stance=torch.full(env_team, -1, device=game.device, dtype=torch.int8),
            sentry_operator_command=torch.zeros(
                env_team,
                device=game.device,
                dtype=torch.bool,
            ),
            aerial_support=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            rebuild_outpost=torch.zeros(env_unit, device=game.device, dtype=torch.bool),
            tech_complete_level=torch.zeros(env_team, device=game.device, dtype=torch.int8),
            rune_trigger=torch.full(
                env_team,
                RuneMode.IDLE,
                device=game.device,
                dtype=torch.int8,
            ),
            dart_open_gate=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            dart_close_gate=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            dart_target=torch.full(env_team, -1, device=game.device, dtype=torch.int8),
            radar_target=torch.full(env_team, -1, device=game.device, dtype=torch.long),
            radar_report_xy=torch.zeros(
                (*env_team, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            radar_illuminate=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            radar_double=torch.zeros(env_team, device=game.device, dtype=torch.bool),
            radar_key_solved=torch.zeros(env_team, device=game.device, dtype=torch.bool),
        )

    def validate(self, game: GameState) -> None:
        env_unit = (game.num_envs, constants.UNIT_COUNT)
        env_team = (game.num_envs, constants.TEAM_COUNT)
        if self.body_velocity_xy.shape != (*env_unit, 2):
            raise ValueError("body_velocity_xy must have shape [env, unit, 2]")
        if self.target.shape != env_unit or self.fire.shape != env_unit:
            raise ValueError("target and fire must have shape [env, unit]")
        if torch.any(self.target < 0) or torch.any(self.target > 8):
            raise ValueError("target values must use the nine-class action schema")
        if self.radar_target.shape != env_team:
            raise ValueError("radar_target must have shape [env, team]")
        if torch.any(self.radar_target < -1) or torch.any(
            self.radar_target >= constants.UNIT_COUNT
        ):
            raise ValueError("radar_target must be -1 or a valid unit slot")
        if self.radar_report_xy.shape != (*env_team, 2):
            raise ValueError("radar_report_xy must have shape [env, team, 2]")
        if not torch.all(torch.isfinite(self.radar_report_xy)):
            raise ValueError("radar_report_xy must contain finite coordinates")
