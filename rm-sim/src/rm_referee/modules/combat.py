"""Projectile bookkeeping, armor detection, and attack damage."""

from __future__ import annotations

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import HeatLock, Role, Team, Weapon, slot, unit_roles, unit_teams
from rm_referee.state import GameState
from rm_referee.tensor_ops import gather_unit, round_half_up
from rm_referee.schema import weapon_capability


def _raw_damage(
    roles: Tensor,
    armor_index: Tensor,
    weapon_index: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    is_17 = weapon_index == Weapon.MM17
    is_42 = weapon_index == Weapon.MM42
    is_base = roles == Role.BASE
    is_outpost = roles == Role.OUTPOST
    is_aerial = roles == Role.AERIAL
    is_base_front = is_base & (armor_index == 0)

    damage = torch.where(
        is_17,
        torch.full_like(weapon_index, constants.DAMAGE_ROBOT[Weapon.MM17], dtype=dtype),
        torch.zeros_like(weapon_index, dtype=dtype),
    )
    damage = torch.where(
        is_42,
        torch.full_like(damage, constants.DAMAGE_ROBOT[Weapon.MM42]),
        damage,
    )
    damage = torch.where(
        is_base & is_17,
        torch.full_like(damage, constants.DAMAGE_BASE_OTHER[Weapon.MM17]),
        damage,
    )
    damage = torch.where(
        is_base_front & is_17,
        torch.full_like(damage, constants.DAMAGE_BASE_FRONT[Weapon.MM17]),
        damage,
    )
    damage = torch.where(
        is_base & is_42,
        torch.full_like(damage, constants.DAMAGE_BASE_OTHER[Weapon.MM42]),
        damage,
    )
    damage = torch.where(
        is_outpost & is_17,
        torch.full_like(damage, constants.DAMAGE_OUTPOST[Weapon.MM17]),
        damage,
    )
    damage = torch.where(
        is_outpost & is_42,
        torch.full_like(damage, constants.DAMAGE_OUTPOST[Weapon.MM42]),
        damage,
    )
    return torch.where(is_aerial, torch.zeros_like(damage), damage)


def apply_combat(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    """Apply one deterministic combat tick in place.

    The fixed candidate loop is chronological per target armor. Every
    iteration is vectorized across all environments, targets, armors, and
    weapons; there is no per-environment or per-unit Python loop.
    """

    active = ~state.done
    capability = weapon_capability(state.device).unsqueeze(0)
    can_fire = (
        active[:, None, None]
        & state.alive[:, :, None]
        & ~state.weak[:, :, None]
        & capability
        & (state.heat_lock == HeatLock.NONE)
    )
    shots = torch.where(can_fire, inputs.shots_fired, torch.zeros_like(inputs.shots_fired))
    events.shots_fired.copy_(shots)

    ammo_before = state.ammo.clone()
    events.overfire.copy_(torch.clamp(shots - ammo_before, min=0))
    state.ammo.copy_(torch.clamp(ammo_before - shots, min=0))

    heat_per_shot = torch.tensor(
        constants.HEAT_PER_SHOT,
        device=state.device,
        dtype=state.dtype,
    )
    state.heat.add_(shots.to(state.dtype) * heat_per_shot)

    hero_slots = torch.tensor(
        [slot(Team.RED, Role.HERO), slot(Team.BLUE, Role.HERO)],
        device=state.device,
    )
    hero_shots = shots[:, hero_slots, Weapon.MM42]
    hero_overfire = events.overfire[:, hero_slots, Weapon.MM42] > 0
    hero_alive = state.alive[:, hero_slots]
    hero_ammo = ammo_before[:, hero_slots, Weapon.MM42]
    state.hero_42_invalid_block &= ~(hero_alive & (hero_ammo > 0))
    state.hero_42_invalid_block |= hero_overfire
    state.hero_42_silence_s.copy_(
        torch.where(
            hero_shots > 0,
            torch.zeros_like(state.hero_42_silence_s),
            state.hero_42_silence_s + dt * active[:, None],
        )
    )

    num_envs = state.num_envs
    roles = unit_roles(state.device)
    teams = unit_teams(state.device)
    target_role = roles.view(1, constants.UNIT_COUNT, 1, 1)
    target_team = teams.view(1, constants.UNIT_COUNT)
    armor_index = torch.arange(
        constants.MAX_ARMOR_COUNT,
        device=state.device,
    ).view(1, 1, constants.MAX_ARMOR_COUNT, 1)
    weapon_index = torch.arange(
        constants.WEAPON_COUNT,
        device=state.device,
    ).view(1, 1, 1, constants.WEAPON_COUNT)
    armor_count = torch.tensor(
        constants.ARMOR_COUNT_BY_ROLE * constants.TEAM_COUNT,
        device=state.device,
    ).view(1, constants.UNIT_COUNT, 1, 1)
    armor_valid = armor_index < armor_count

    outpost_slots = torch.tensor(
        [slot(Team.RED, Role.OUTPOST), slot(Team.BLUE, Role.OUTPOST)],
        device=state.device,
    )
    outpost_alive = state.alive[:, outpost_slots] & (state.hp[:, outpost_slots] > 0)
    target_outpost_alive = torch.gather(
        outpost_alive,
        1,
        target_team.expand(num_envs, -1),
    )
    base_protected = (target_role == Role.BASE) & target_outpost_alive[:, :, None, None]

    raw_damage = _raw_damage(
        target_role,
        armor_index,
        weapon_index,
        dtype=state.dtype,
    )
    intervals = torch.tensor(
        constants.ARMOR_REFRACTORY_S,
        device=state.device,
        dtype=state.dtype,
    ).view(1, 1, 1, constants.WEAPON_COUNT)
    env_index = torch.arange(num_envs, device=state.device).view(num_envs, 1, 1, 1)
    weapon_grid = weapon_index.expand(
        num_envs,
        constants.UNIT_COUNT,
        constants.MAX_ARMOR_COUNT,
        constants.WEAPON_COUNT,
    )

    alive_before = state.alive.clone()

    for candidate in range(constants.MAX_HIT_CANDIDATES):
        source = inputs.hits.source[..., candidate]
        source_safe = torch.clamp(source, min=0)
        valid = source >= 0
        source_fired = (
            events.shots_fired[
                env_index,
                source_safe,
                weapon_grid,
            ]
            > 0
        )
        source_alive = gather_unit(state.alive, source_safe)
        source_team = teams[source_safe]

        event_time = (
            state.elapsed_s[:, None, None, None] + inputs.hits.time_offset_s[..., candidate]
        )
        refractory_ok = (
            event_time - state.armor_last_hit_s >= intervals - torch.finfo(state.dtype).eps
        )
        target_eligible = (
            active[:, None, None, None]
            & state.alive[:, :, None, None]
            & (state.invulnerable_s[:, :, None, None] <= 0)
            & armor_valid
            & ~base_protected
        )
        invalid_42 = (weapon_grid == Weapon.MM42) & state.hero_42_invalid_block[
            env_index, source_team
        ]
        accepted = (
            valid & source_fired & source_alive & refractory_ok & target_eligible & ~invalid_42
        )
        events.hit_accepted[..., candidate].copy_(accepted)
        state.armor_last_hit_s.copy_(torch.where(accepted, event_time, state.armor_last_hit_s))

        source_attack = gather_unit(state.attack_multiplier, source_safe)
        critical_multiplier = torch.where(
            inputs.hits.critical[..., candidate],
            torch.full_like(source_attack, constants.CRITICAL_ATTACK_MULTIPLIER),
            torch.ones_like(source_attack),
        )
        attack_multiplier = torch.maximum(source_attack, critical_multiplier)
        defense_factor = torch.clamp(
            1.0
            - state.defense_fraction[:, :, None, None]
            + state.vulnerability_fraction[:, :, None, None]
            + state.radar_vulnerability_fraction[:, :, None, None],
            min=0.0,
        )
        event_damage = round_half_up(raw_damage * attack_multiplier * defense_factor)
        event_damage = torch.where(accepted, event_damage, torch.zeros_like(event_damage))
        target_damage = event_damage.sum(dim=(2, 3))

        shield_by_target = torch.zeros_like(state.hp)
        base_slots = torch.tensor(
            [slot(Team.RED, Role.BASE), slot(Team.BLUE, Role.BASE)],
            device=state.device,
        )
        shield_by_target[:, base_slots] = state.base_shield
        effective_damage = torch.minimum(target_damage, state.hp + shield_by_target)
        shield_absorbed = torch.minimum(effective_damage, shield_by_target)
        hp_damage = effective_damage - shield_absorbed

        state.hp.sub_(hp_damage).clamp_(min=0)
        state.base_shield.sub_(shield_absorbed[:, base_slots]).clamp_(min=0)
        events.damage_taken.add_(effective_damage)

        scale = torch.where(
            target_damage > 0,
            effective_damage / torch.clamp(target_damage, min=1),
            torch.zeros_like(target_damage),
        )
        attributed = event_damage * scale[:, :, None, None]
        team_add = torch.zeros(
            (num_envs, constants.TEAM_COUNT),
            device=state.device,
            dtype=state.dtype,
        )
        team_add.scatter_add_(
            1,
            source_team.reshape(num_envs, -1),
            attributed.reshape(num_envs, -1),
        )
        state.team_damage.add_(team_add)

    deaths = alive_before & (state.hp <= 0)
    state.alive &= ~deaths
    events.deaths.copy_(deaths)
