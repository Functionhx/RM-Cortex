"""Radar marking, interference, no-data timing, and double vulnerability."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.inputs import RuleInputs
from rm_referee.schema import RadarQuality, Role, ground_robot_mask, unit_roles, unit_teams
from rm_referee.state import GameState


def apply_radar(state: GameState, inputs: RuleInputs, dt: float) -> None:
    active = (~state.done)[:, None, None]
    roles = unit_roles(state.device)
    radar_target = (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
        | (roles == Role.SENTRY)
    )
    received = inputs.radar_update & active & radar_target[None, None, :]
    state.radar_report_xy.copy_(
        torch.where(
            received[..., None],
            inputs.radar_report_xy,
            state.radar_report_xy,
        )
    )
    state.radar_report_age_s.copy_(
        torch.where(
            received,
            torch.zeros_like(state.radar_report_age_s),
            state.radar_report_age_s + dt * (state.radar_report_valid & active).to(state.dtype),
        )
    )
    state.radar_report_valid.logical_or_(received)
    state.radar_no_data_s.copy_(
        torch.where(
            received,
            torch.zeros_like(state.radar_no_data_s),
            state.radar_no_data_s + dt * active,
        )
    )
    no_data_update = (
        (state.radar_no_data_s >= constants.RADAR_NO_DATA_S) & active & radar_target[None, None, :]
    )
    state.radar_no_data_s.copy_(
        torch.where(
            no_data_update,
            state.radar_no_data_s - constants.RADAR_NO_DATA_S,
            state.radar_no_data_s,
        )
    )
    update = received | no_data_update
    quality = torch.where(
        no_data_update,
        torch.full_like(inputs.radar_quality, RadarQuality.WRONG),
        inputs.radar_quality,
    )

    target_teams = unit_teams(state.device)
    observer = torch.arange(constants.TEAM_COUNT, device=state.device).view(1, -1, 1)
    is_own_target = observer == target_teams.view(1, 1, -1)
    localization_offline = inputs.localization_module_offline[:, None, :]
    offline_quality = torch.where(
        is_own_target,
        torch.full_like(quality, RadarQuality.WRONG),
        torch.full_like(quality, RadarQuality.ACCURATE),
    )
    quality = torch.where(localization_offline & update, offline_quality, quality)

    step = torch.where(
        quality == RadarQuality.ACCURATE,
        torch.ones_like(state.radar_x),
        torch.where(
            quality == RadarQuality.HALF_ACCURATE,
            torch.full_like(state.radar_x, 0.5),
            torch.full_like(state.radar_x, -0.8),
        ),
    )
    same_positive_direction = (
        (quality == RadarQuality.ACCURATE) | (quality == RadarQuality.HALF_ACCURATE)
    ) & (
        (state.radar_last_quality == RadarQuality.ACCURATE)
        | (state.radar_last_quality == RadarQuality.HALF_ACCURATE)
    )
    same_wrong_direction = (quality == RadarQuality.WRONG) & (
        state.radar_last_quality == RadarQuality.WRONG
    )
    same_direction = same_positive_direction | same_wrong_direction
    new_x = torch.where(same_direction, state.radar_x + step, step)
    state.radar_x.copy_(torch.where(update, new_x, state.radar_x))
    state.radar_p.copy_(
        torch.where(
            update,
            torch.clamp(state.radar_p + new_x, min=0, max=150),
            state.radar_p,
        )
    )
    state.radar_last_quality.copy_(torch.where(update, quality, state.radar_last_quality))

    solved = inputs.radar_key_solved & (~state.done)[:, None]
    state.radar_interference_level.copy_(
        torch.where(
            solved,
            torch.clamp(state.radar_interference_level + 1, max=3),
            state.radar_interference_level,
        )
    )

    own_threshold = state.radar_p >= 50
    enemy_threshold = state.radar_p >= 100
    state.radar_truth_visible.copy_(
        torch.where(is_own_target, own_threshold, enemy_threshold) & radar_target[None, None, :]
    )

    target = torch.arange(constants.UNIT_COUNT, device=state.device)
    enemy_observer = 1 - target_teams
    enemy_p = state.radar_p[:, enemy_observer, target]
    interference_blocked = (
        state.radar_interference_level[:, target_teams]
        > state.radar_interference_level[:, enemy_observer]
    )
    base_vulnerability = torch.where(
        enemy_p >= 120,
        torch.full_like(enemy_p, 0.20),
        torch.where(
            enemy_p >= 100,
            torch.full_like(enemy_p, 0.15),
            torch.zeros_like(enemy_p),
        ),
    )
    base_vulnerability = torch.where(
        interference_blocked,
        torch.zeros_like(base_vulnerability),
        base_vulnerability,
    )
    ground = ground_robot_mask(state.device).unsqueeze(0)
    base_vulnerability = torch.where(ground, base_vulnerability, 0.0)

    vulnerable_by_target_team = (
        (base_vulnerability > 0)
        .view(state.num_envs, constants.TEAM_COUNT, constants.ROLES_PER_TEAM)
        .any(dim=-1)
    )
    observer_has_vulnerability = vulnerable_by_target_team.flip(dims=(1,))
    state.radar_vulnerability_progress_s.add_(observer_has_vulnerability.to(state.dtype) * dt)
    total_available = state.radar_double_charges + state.radar_double_uses
    earned = (state.radar_vulnerability_progress_s >= constants.RADAR_DOUBLE_PROGRESS_S) & (
        total_available < constants.RADAR_DOUBLE_MAX_USES
    )
    state.radar_double_charges.add_(earned.to(torch.long))
    state.radar_vulnerability_progress_s.copy_(
        torch.where(
            earned,
            state.radar_vulnerability_progress_s - constants.RADAR_DOUBLE_PROGRESS_S,
            state.radar_vulnerability_progress_s,
        )
    )

    state.radar_double_s.sub_(dt * (~state.done)[:, None]).clamp_(min=0)
    activate_double = (
        inputs.radar_double_request
        & (state.radar_double_charges > 0)
        & (state.radar_double_s <= 0)
        & (~state.done)[:, None]
    )
    state.radar_double_charges.sub_(activate_double.to(torch.long))
    state.radar_double_uses.add_(activate_double.to(torch.long))
    state.radar_double_s.copy_(
        torch.where(
            activate_double,
            torch.full_like(
                state.radar_double_s,
                constants.RADAR_DOUBLE_DURATION_S,
            ),
            state.radar_double_s,
        )
    )
    target_double = state.radar_double_s[:, enemy_observer] > 0
    state.radar_vulnerability_fraction.copy_(
        torch.where(target_double, base_vulnerability * 2.0, base_vulnerability)
    )
