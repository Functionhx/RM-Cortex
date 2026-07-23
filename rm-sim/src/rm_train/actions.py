"""Decode heterogeneous policy heads into the shared world action schema."""

from __future__ import annotations

from dataclasses import fields
import math

import torch

from rm_referee import constants
from rm_referee.schema import Role, RuneMode, Team, slot, unit_roles
from rm_referee.state import GameState
from rm_train.policy import PolicyAction
from rm_world.actions import WorldActions
from rm_world.kinematics import KinematicState


def decode_policy_actions(
    game: GameState,
    world: KinematicState,
    policy: PolicyAction,
) -> WorldActions:
    """Apply role-specific meanings without introducing referee rules."""

    expected = (game.num_envs, constants.UNIT_COUNT)
    if policy.target.shape != expected or policy.motion.shape != (*expected, 4):
        raise ValueError("policy actions must use [env, 16, ...] agent layout")

    actions = WorldActions.zeros(game)
    roles = unit_roles(game.device)
    robot = roles <= Role.SENTRY
    ground = robot & (roles != Role.AERIAL)
    motion = torch.tanh(policy.motion)
    actions.body_velocity_xy.copy_(motion[..., :2] * 3.0 * robot[None, :, None])
    actions.yaw_rate.copy_(motion[..., 2] * (2.0 * math.pi) * robot[None, :])
    actions.vertical_velocity.copy_(motion[..., 3] * 2.0 * (roles == Role.AERIAL)[None, :])
    actions.supercap_input_w.copy_(torch.sigmoid(policy.motion[..., 3]) * 25.0 * ground[None, :])
    actions.target.copy_(policy.target)
    actions.fire.copy_(policy.fire)
    actions.remote_heal.copy_(policy.common[..., 0] & ground[None, :])
    actions.immediate_respawn.copy_(policy.common[..., 1] & ground[None, :])

    teams = range(constants.TEAM_COUNT)
    hero_slots = [slot(team, Role.HERO) for team in teams]
    engineer_slots = [slot(team, Role.ENGINEER) for team in teams]
    aerial_slots = [slot(team, Role.AERIAL) for team in teams]
    sentry_slots = [slot(team, Role.SENTRY) for team in teams]
    dart_slots = [slot(team, Role.BASE) for team in teams]
    radar_slots = [slot(team, Role.OUTPOST) for team in teams]

    actions.hero_deploy.copy_(policy.special[:, hero_slots, 0])
    actions.rebuild_outpost[:, engineer_slots] = policy.special[:, engineer_slots, 0]
    requested_level = policy.mode[:, engineer_slots].to(torch.int8)
    actions.tech_complete_level.copy_(
        torch.where(
            policy.special[:, engineer_slots, 1],
            requested_level,
            torch.zeros_like(requested_level),
        )
    )
    rune_mode = torch.where(
        policy.mode[:, engineer_slots] >= 3,
        torch.full_like(requested_level, RuneMode.LARGE),
        torch.full_like(requested_level, RuneMode.SMALL),
    )
    actions.rune_trigger.copy_(
        torch.where(
            policy.special[:, engineer_slots, 2],
            rune_mode,
            torch.full_like(requested_level, RuneMode.IDLE),
        )
    )
    actions.aerial_support.copy_(policy.special[:, aerial_slots, 0])
    actions.sentry_claim_ammo.copy_(policy.special[:, sentry_slots, 0])
    actions.sentry_operator_command.copy_(policy.special[:, sentry_slots, 1])
    actions.sentry_stance.copy_(
        policy.mode[:, sentry_slots].remainder(constants.SENTRY_STANCE_COUNT).to(torch.int8)
    )

    actions.dart_open_gate.copy_(policy.special[:, dart_slots, 0])
    actions.dart_close_gate.copy_(policy.special[:, dart_slots, 1])
    actions.dart_target.copy_(
        torch.where(
            policy.special[:, dart_slots, 2],
            policy.mode[:, dart_slots].to(torch.int8),
            torch.full_like(policy.mode[:, dart_slots], -1, dtype=torch.int8),
        )
    )

    actions.radar_target.copy_(policy.radar_target[:, radar_slots])
    target_xy = torch.gather(
        world.position_xy,
        1,
        actions.radar_target[:, :, None].expand(-1, -1, 2),
    )
    actions.radar_report_xy.copy_(target_xy + torch.tanh(policy.radar_offset[:, radar_slots]) * 2.0)
    actions.radar_illuminate.copy_(policy.special[:, radar_slots, 0])
    actions.radar_double.copy_(policy.special[:, radar_slots, 1])
    actions.radar_key_solved.copy_(policy.special[:, radar_slots, 2])
    return actions


def replace_team_actions(
    destination: WorldActions,
    source: WorldActions,
    team: Team | int,
) -> WorldActions:
    """Replace one team's unit and team commands, in place."""

    team_index = int(team)
    unit_slice = slice(
        team_index * constants.ROLES_PER_TEAM,
        (team_index + 1) * constants.ROLES_PER_TEAM,
    )
    team_fields = {
        "purchase_unit",
        "purchase_weapon",
        "purchase_kind",
        "sentry_claim_ammo",
        "hero_deploy",
        "sentry_stance",
        "sentry_operator_command",
        "aerial_support",
        "tech_complete_level",
        "rune_trigger",
        "dart_open_gate",
        "dart_close_gate",
        "dart_target",
        "radar_target",
        "radar_report_xy",
        "radar_illuminate",
        "radar_double",
        "radar_key_solved",
    }
    for field in fields(WorldActions):
        destination_value = getattr(destination, field.name)
        source_value = getattr(source, field.name)
        if field.name in team_fields:
            destination_value[:, team_index] = source_value[:, team_index]
        else:
            destination_value[:, unit_slice] = source_value[:, unit_slice]
    return destination
