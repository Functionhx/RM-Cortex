"""Experience awards, level caps, and official performance tables."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.schema import (
    HeroProfile,
    InfantryChassisProfile,
    InfantryLauncherProfile,
    Role,
    unit_roles,
    unit_teams,
)
from rm_referee.state import GameState


def _table(values: tuple[int, ...], state: GameState) -> torch.Tensor:
    return torch.tensor(values, device=state.device, dtype=state.dtype)


def apply_upgrade(state: GameState, events: RefereeEvents) -> None:
    roles = unit_roles(state.device)
    teams = unit_teams(state.device)
    upgradable = (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
    )

    shot_count = events.shots_fired.sum(dim=-1).to(state.dtype)
    shot_xp_per_round = torch.where(
        roles == Role.HERO,
        torch.full((constants.UNIT_COUNT,), 10.0, device=state.device, dtype=state.dtype),
        torch.where(
            upgradable,
            torch.ones(constants.UNIT_COUNT, device=state.device, dtype=state.dtype),
            torch.zeros(constants.UNIT_COUNT, device=state.device, dtype=state.dtype),
        ),
    )
    events.xp_gained.add_(shot_count * shot_xp_per_round[None, :])

    target_roles = roles.view(1, 1, constants.UNIT_COUNT)
    damage = events.damage_by_source_target
    robot_target = (
        (target_roles == Role.HERO)
        | (target_roles == Role.ENGINEER)
        | (target_roles == Role.INFANTRY_3)
        | (target_roles == Role.INFANTRY_4)
        | (target_roles == Role.SENTRY)
    )
    outpost_target = target_roles == Role.OUTPOST
    base_target = target_roles == Role.BASE
    damage_xp = torch.where(robot_target, damage * 4.0, torch.zeros_like(damage))
    damage_xp += torch.where(outpost_target, damage * 2.0, torch.zeros_like(damage))
    damage_xp += torch.where(base_target, torch.ceil(damage / 2.0), torch.zeros_like(damage))
    events.xp_gained.add_(damage_xp.sum(dim=-1))

    valid_kill = events.kill_source >= 0
    kill_source = torch.clamp(events.kill_source, min=0)
    source_level = torch.gather(state.level, 1, kill_source)
    target_level = state.level.clone()
    target_level[:, (roles == Role.ENGINEER) | (roles == Role.SENTRY)] = 1
    target_is_robot = (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
        | (roles == Role.SENTRY)
    )
    level_gap = torch.clamp(target_level - source_level, min=0).to(state.dtype)
    kill_xp = 50.0 * target_level.to(state.dtype) * (1.0 + 0.2 * level_gap)
    source_upgradable = upgradable[kill_source]
    kill_xp = torch.where(
        valid_kill & target_is_robot[None, :] & source_upgradable,
        kill_xp,
        torch.zeros_like(kill_xp),
    )
    kill_scatter = torch.zeros_like(state.xp)
    kill_scatter.scatter_add_(1, kill_source, kill_xp)
    events.xp_gained.add_(kill_scatter)

    base_targets = roles == Role.BASE
    deployed_damage = (
        (damage[:, :, base_targets] > 0).any(dim=-1)
        & (roles[None, :] == Role.HERO)
        & state.hero_deployed[:, teams]
    )
    events.xp_gained.add_(deployed_damage.to(state.dtype) * 100.0)

    raw_gain = torch.where(
        upgradable[None, :],
        events.xp_gained,
        torch.zeros_like(events.xp_gained),
    )
    rune_active = state.rune_small_buff_s > 0
    gain_by_team = raw_gain.view(
        state.num_envs,
        constants.TEAM_COUNT,
        constants.ROLES_PER_TEAM,
    )
    team_gain = gain_by_team.sum(dim=-1)
    bonus_remaining = torch.clamp(
        constants.RUNE_SMALL_BONUS_XP_CAP - state.rune_small_bonus_xp,
        min=0,
    )
    team_bonus = torch.where(
        rune_active,
        torch.minimum(team_gain, bonus_remaining),
        torch.zeros_like(team_gain),
    )
    bonus_scale = torch.where(
        team_gain > 0,
        team_bonus / torch.clamp(team_gain, min=1.0e-9),
        torch.zeros_like(team_gain),
    )
    bonus = gain_by_team * bonus_scale[:, :, None]
    raw_gain.add_(bonus.reshape_as(raw_gain))
    state.rune_small_bonus_xp.add_(team_bonus)

    cap = state.team_level_cap[:, teams]
    eligible = upgradable[None, :] & (state.level < cap)
    thresholds = torch.tensor(
        constants.XP_THRESHOLDS,
        device=state.device,
        dtype=state.dtype,
    )
    cap_threshold = thresholds[torch.clamp(cap - 1, min=0)]
    old_xp = state.xp.clone()
    state.xp.copy_(
        torch.where(
            eligible,
            torch.minimum(state.xp + raw_gain, cap_threshold),
            state.xp,
        )
    )
    computed_level = (state.xp[:, :, None] >= thresholds.view(1, 1, -1)).sum(dim=-1)
    state.level.copy_(
        torch.where(
            upgradable[None, :],
            torch.minimum(computed_level, cap),
            state.level,
        )
    )
    events.xp_gained.copy_(state.xp - old_xp)

    level_index = torch.clamp(state.level - 1, min=0, max=9)
    old_max_hp = state.max_hp.clone()
    hero = roles == Role.HERO
    infantry = (roles == Role.INFANTRY_3) | (roles == Role.INFANTRY_4)
    aerial = roles == Role.AERIAL

    hero_close = state.hero_profile == HeroProfile.CLOSE
    hero_hp = torch.where(
        hero_close,
        _table(constants.HERO_CLOSE_HP, state)[level_index],
        _table(constants.HERO_REMOTE_HP, state)[level_index],
    )
    hero_power = torch.where(
        hero_close,
        _table(constants.HERO_CLOSE_POWER_W, state)[level_index],
        _table(constants.HERO_REMOTE_POWER_W, state)[level_index],
    )
    hero_heat = torch.where(
        hero_close,
        _table(constants.HERO_CLOSE_HEAT_42, state)[level_index],
        _table(constants.HERO_REMOTE_HEAT_42, state)[level_index],
    )
    hero_cooling = torch.where(
        hero_close,
        _table(constants.HERO_CLOSE_COOLING_42, state)[level_index],
        _table(constants.HERO_REMOTE_COOLING_42, state)[level_index],
    )

    infantry_power_profile = state.infantry_chassis_profile == InfantryChassisProfile.POWER
    infantry_hp = torch.where(
        infantry_power_profile,
        _table(constants.INFANTRY_POWER_HP, state)[level_index],
        _table(constants.INFANTRY_HP_HP, state)[level_index],
    )
    infantry_power = torch.where(
        infantry_power_profile,
        _table(constants.INFANTRY_POWER_W, state)[level_index],
        _table(constants.INFANTRY_HP_POWER_W, state)[level_index],
    )
    infantry_burst = state.infantry_launcher_profile == InfantryLauncherProfile.BURST
    infantry_heat = torch.where(
        infantry_burst,
        _table(constants.INFANTRY_BURST_HEAT_17, state)[level_index],
        _table(constants.INFANTRY_COOL_HEAT_17, state)[level_index],
    )
    infantry_cooling = torch.where(
        infantry_burst,
        _table(constants.INFANTRY_BURST_COOLING_17, state)[level_index],
        _table(constants.INFANTRY_COOL_COOLING_17, state)[level_index],
    )

    state.max_hp.copy_(
        torch.where(
            hero[None, :],
            hero_hp,
            torch.where(infantry[None, :], infantry_hp, state.max_hp),
        )
    )
    hp_gain = torch.clamp(state.max_hp - old_max_hp, min=0)
    state.hp.copy_(torch.minimum(state.hp + hp_gain, state.max_hp))
    state.base_power_limit_w.copy_(
        torch.where(
            hero[None, :],
            hero_power,
            torch.where(infantry[None, :], infantry_power, state.base_power_limit_w),
        )
    )
    state.heat_limit[:, :, 1].copy_(
        torch.where(hero[None, :], hero_heat, state.heat_limit[:, :, 1])
    )
    state.base_cooling_per_s[:, :, 1].copy_(
        torch.where(hero[None, :], hero_cooling, state.base_cooling_per_s[:, :, 1])
    )
    state.heat_limit[:, :, 0].copy_(
        torch.where(
            infantry[None, :],
            infantry_heat,
            torch.where(
                aerial[None, :],
                _table(constants.AERIAL_HEAT_17, state)[level_index],
                state.heat_limit[:, :, 0],
            ),
        )
    )
    state.base_cooling_per_s[:, :, 0].copy_(
        torch.where(
            infantry[None, :],
            infantry_cooling,
            torch.where(
                aerial[None, :],
                _table(constants.AERIAL_COOLING_17, state)[level_index],
                state.base_cooling_per_s[:, :, 0],
            ),
        )
    )
