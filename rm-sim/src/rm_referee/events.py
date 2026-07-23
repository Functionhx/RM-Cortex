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
    deaths: Tensor
    respawns: Tensor
    purchased_rounds: Tensor
    scheduled_coin: Tensor
    match_ended: Tensor

    @classmethod
    def empty(cls, state: GameState) -> "RefereeEvents":
        env_unit = (state.num_envs, constants.UNIT_COUNT)
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
            deaths=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            respawns=torch.zeros(env_unit, device=state.device, dtype=torch.bool),
            purchased_rounds=torch.zeros(
                (state.num_envs, constants.TEAM_COUNT),
                device=state.device,
                dtype=torch.long,
            ),
            scheduled_coin=torch.zeros(
                (state.num_envs,),
                device=state.device,
                dtype=torch.long,
            ),
            match_ended=torch.zeros(
                (state.num_envs,),
                device=state.device,
                dtype=torch.bool,
            ),
        )
