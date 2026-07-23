"""Radar mark recurrence from manual table 5-21."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.inputs import RuleInputs
from rm_referee.schema import RadarQuality, ground_robot_mask, unit_teams
from rm_referee.state import GameState


def apply_radar(state: GameState, inputs: RuleInputs) -> None:
    active_update = inputs.radar_update & (~state.done)[:, None, None]
    quality = inputs.radar_quality
    step = torch.where(
        quality == RadarQuality.ACCURATE,
        torch.ones_like(state.radar_x),
        torch.where(
            quality == RadarQuality.HALF_ACCURATE,
            torch.full_like(state.radar_x, 0.5),
            torch.full_like(state.radar_x, -0.8),
        ),
    )
    same_direction = quality == state.radar_last_quality
    new_x = torch.where(same_direction, state.radar_x + step, step)
    state.radar_x.copy_(torch.where(active_update, new_x, state.radar_x))
    state.radar_p.copy_(
        torch.where(
            active_update,
            torch.clamp(state.radar_p + new_x, min=0, max=150),
            state.radar_p,
        )
    )
    state.radar_last_quality.copy_(torch.where(active_update, quality, state.radar_last_quality))

    target_team = unit_teams(state.device)
    enemy_observer = 1 - target_team
    target = torch.arange(constants.UNIT_COUNT, device=state.device)
    enemy_p = state.radar_p[:, enemy_observer, target]
    vulnerability = torch.where(
        enemy_p >= 120,
        torch.full_like(enemy_p, 0.20),
        torch.where(
            enemy_p >= 100,
            torch.full_like(enemy_p, 0.15),
            torch.zeros_like(enemy_p),
        ),
    )
    ground = ground_robot_mask(state.device).unsqueeze(0)
    state.radar_vulnerability_fraction.copy_(torch.where(ground, vulnerability, 0.0))
