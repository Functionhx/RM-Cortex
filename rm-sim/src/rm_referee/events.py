"""Tensor event stream emitted by one referee tick."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.state import GameState


@dataclass
class RefereeEvents:
    shots_fired: Tensor
    overfire: Tensor
    hit_accepted: Tensor
    damage_taken: Tensor
    damage_dealt: Tensor
    damage_by_source_target: Tensor
    deaths: Tensor
    kill_source: Tensor
    respawns: Tensor
    immediate_respawns: Tensor
    remote_heals: Tensor
    purchased_rounds: Tensor
    sentry_claimed_rounds: Tensor
    scheduled_coin: Tensor
    tech_completed: Tensor
    rune_activated: Tensor
    dart_fired: Tensor
    dart_hits: Tensor
    xp_gained: Tensor
    match_ended: Tensor

    @classmethod
    def empty(cls, state: GameState) -> "RefereeEvents":
        env_unit = (state.num_envs, constants.UNIT_COUNT)
        env_team = (state.num_envs, constants.TEAM_COUNT)
        hit_shape = (
            state.num_envs,
            constants.UNIT_COUNT,
            constants.MAX_ARMOR_COUNT,
            constants.WEAPON_COUNT,
            constants.MAX_HIT_CANDIDATES,
        )
        return cls(
            shots_fired=torch.zeros(
                (*env_unit, constants.WEAPON_COUNT),
                device=state.device,
                dtype=torch.long,
            ),
            overfire=torch.zeros(
                (*env_unit, constants.WEAPON_COUNT),
                device=state.device,
                dtype=torch.long,
            ),
            hit_accepted=torch.zeros(hit_shape, device=state.device, dtype=torch.bool),
            damage_taken=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            damage_dealt=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            damage_by_source_target=torch.zeros(
                (state.num_envs, constants.UNIT_COUNT, constants.UNIT_COUNT),
                device=state.device,
                dtype=state.dtype,
            ),
            deaths=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            kill_source=torch.full(env_unit, -1, device=state.device, dtype=torch.long),
            respawns=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            immediate_respawns=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            remote_heals=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            purchased_rounds=torch.zeros(env_team, device=state.device, dtype=torch.long),
            sentry_claimed_rounds=torch.zeros(env_team, device=state.device, dtype=torch.long),
            scheduled_coin=torch.zeros(
                (state.num_envs,),
                device=state.device,
                dtype=torch.long,
            ),
            tech_completed=torch.zeros(env_team, device=state.device, dtype=torch.int8),
            rune_activated=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            dart_fired=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            dart_hits=torch.zeros(env_team, device=state.device, dtype=torch.bool),
            xp_gained=torch.zeros(env_unit, device=state.device, dtype=state.dtype),
            match_ended=torch.zeros(
                (state.num_envs,),
                device=state.device,
                dtype=torch.bool,
            ),
        )
