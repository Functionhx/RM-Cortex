"""Field-zone, traversal, and composed modifier rules."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, Team, Terrain, Zone, slot, unit_roles, unit_teams
from rm_referee.state import GameState


def apply_field_buffs(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    """Update occupancy linger, terrain rewards, and zone-specific state."""

    active_match = (~state.done)[:, None]
    roles = unit_roles(state.device)
    ground = (
        (roles == Role.HERO)
        | (roles == Role.ENGINEER)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )
    occupancy = inputs.zone_occupancy.clone()
    occupancy[:, :, Zone.SUPPLY] |= inputs.in_supply_zone
    occupancy &= (
        active_match[:, :, None]
        & state.alive[:, :, None]
        & ground[None, :, None]
        & ~state.weak[:, :, None]
    )

    was_active = state.zone_linger_s > 0
    state.zone_linger_s.copy_(
        torch.where(
            occupancy,
            torch.full_like(state.zone_linger_s, constants.ZONE_LOSS_DELAY_S),
            torch.clamp(state.zone_linger_s - dt, min=0),
        )
    )
    state.zone_linger_s.copy_(
        torch.where(
            state.alive[:, :, None],
            state.zone_linger_s,
            torch.zeros_like(state.zone_linger_s),
        )
    )
    zone_active = state.zone_linger_s > 0

    defense = torch.zeros_like(state.hp)
    defense = torch.maximum(
        defense,
        zone_active[:, :, Zone.BASE].to(state.dtype) * 0.50,
    )
    central_eligible = (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.SENTRY)
    )
    defense = torch.maximum(
        defense,
        zone_active[:, :, Zone.CENTRAL_HIGH].to(state.dtype)
        * central_eligible[None, :].to(state.dtype)
        * 0.25,
    )
    defense = torch.maximum(
        defense,
        zone_active[:, :, Zone.TRAPEZOID_HIGH].to(state.dtype) * 0.50,
    )
    defense = torch.maximum(
        defense,
        zone_active[:, :, Zone.OUTPOST].to(state.dtype) * 0.25,
    )
    defense = torch.maximum(
        defense,
        zone_active[:, :, Zone.OWN_FORTRESS].to(state.dtype) * 0.50,
    )
    state.zone_defense_fraction.copy_(defense)

    base_slots = torch.tensor(
        [slot(Team.RED, Role.BASE), slot(Team.BLUE, Role.BASE)],
        device=state.device,
    )
    own_teams = unit_teams(state.device)
    base_delta = state.max_hp[:, base_slots] - state.hp[:, base_slots]
    own_delta = base_delta[:, own_teams]
    fortress_cooling = torch.clamp(torch.floor(own_delta / 40.0), max=75.0)
    own_fortress = zone_active[:, :, Zone.OWN_FORTRESS]
    state.zone_cooling_bonus_per_s.zero_()
    state.zone_cooling_bonus_per_s.copy_(
        torch.where(
            own_fortress[:, :, None],
            fortress_cooling[:, :, None].expand_as(state.zone_cooling_bonus_per_s),
            state.zone_cooling_bonus_per_s,
        )
    )

    reserve_cap = torch.clamp(
        100 + 2 * torch.floor(own_delta / 15.0).to(torch.long),
        max=500,
    )
    entered_fortress = own_fortress & ~was_active[:, :, Zone.OWN_FORTRESS]
    cap_increase = torch.clamp(reserve_cap - state.fortress_reserve_cap, min=0)
    state.fortress_reserve_ammo.add_(
        torch.where(
            entered_fortress,
            reserve_cap,
            torch.where(own_fortress, cap_increase, torch.zeros_like(cap_increase)),
        )
    )
    state.fortress_reserve_cap.copy_(
        torch.where(own_fortress, reserve_cap, torch.zeros_like(reserve_cap))
    )
    state.fortress_reserve_ammo.copy_(
        torch.where(
            own_fortress,
            torch.minimum(state.fortress_reserve_ammo, reserve_cap),
            torch.zeros_like(state.fortress_reserve_ammo),
        )
    )

    engineer = roles == Role.ENGINEER
    assembly = zone_active[:, :, Zone.ASSEMBLY] & engineer[None, :]
    assembly_has_time = state.assembly_invulnerable_used_s < constants.ASSEMBLY_INVULNERABLE_CAP_S
    assembly_invulnerable = assembly & assembly_has_time
    state.assembly_invulnerable_used_s.add_(assembly_invulnerable.to(state.dtype) * dt).clamp_(
        max=constants.ASSEMBLY_INVULNERABLE_CAP_S
    )
    supply_engineer = zone_active[:, :, Zone.SUPPLY] & engineer[None, :]
    zone_invulnerable = assembly_invulnerable | supply_engineer
    state.invulnerable_s.copy_(
        torch.where(
            zone_invulnerable,
            torch.maximum(state.invulnerable_s, torch.full_like(state.invulnerable_s, 2 * dt)),
            state.invulnerable_s,
        )
    )

    terrain_crossed = (
        inputs.terrain_crossed
        & active_match[:, :, None]
        & state.alive[:, :, None]
        & ground[None, :, None]
    )
    state.terrain_buff_s.sub_(dt).clamp_(min=0)
    state.terrain_group_enhanced_s.sub_(dt).clamp_(min=0)
    state.terrain_road_cooldown_s.sub_(dt).clamp_(min=0)
    state.tunnel_cooling_s.sub_(dt).clamp_(min=0)
    terrain_crossed[:, :, Terrain.ROAD] &= state.terrain_road_cooldown_s <= 0

    first = terrain_crossed & ~state.terrain_seen
    events.xp_gained.add_(first.sum(dim=-1).to(state.dtype) * constants.TERRAIN_FIRST_XP)
    state.terrain_seen |= terrain_crossed

    grouped_before = (state.terrain_buff_s[:, :, :3] > 0).any(dim=-1)
    grouped_cross = terrain_crossed[:, :, :3].any(dim=-1)
    durations = torch.tensor(
        constants.TERRAIN_DEFENSE_DURATION_S,
        device=state.device,
        dtype=state.dtype,
    )
    state.terrain_buff_s.copy_(
        torch.where(
            terrain_crossed,
            torch.maximum(
                state.terrain_buff_s,
                durations.view(1, 1, constants.TERRAIN_COUNT),
            ),
            state.terrain_buff_s,
        )
    )
    new_group_duration = torch.where(
        terrain_crossed[:, :, :3],
        durations[:3].view(1, 1, 3),
        torch.zeros_like(state.terrain_buff_s[:, :, :3]),
    ).amax(dim=-1)
    enhanced = grouped_before & grouped_cross
    state.terrain_group_enhanced_s.copy_(
        torch.where(
            enhanced,
            torch.maximum(
                state.terrain_group_enhanced_s,
                torch.maximum(
                    state.terrain_buff_s[:, :, :3].amax(dim=-1),
                    new_group_duration,
                ),
            ),
            state.terrain_group_enhanced_s,
        )
    )
    road_cross = terrain_crossed[:, :, Terrain.ROAD]
    state.terrain_road_cooldown_s.copy_(
        torch.where(
            road_cross,
            torch.full_like(state.terrain_road_cooldown_s, 15.0),
            state.terrain_road_cooldown_s,
        )
    )
    tunnel_cross = terrain_crossed[:, :, Terrain.TUNNEL]
    state.tunnel_cooling_s.copy_(
        torch.where(
            tunnel_cross,
            torch.full_like(
                state.tunnel_cooling_s,
                constants.TUNNEL_COOLING_DURATION_S,
            ),
            state.tunnel_cooling_s,
        )
    )
    grouped_active = (state.terrain_buff_s[:, :, :3] > 0).any(dim=-1)
    group_defense = torch.where(
        grouped_active,
        torch.where(
            state.terrain_group_enhanced_s > 0,
            torch.full_like(state.hp, 0.50),
            torch.full_like(state.hp, 0.25),
        ),
        torch.zeros_like(state.hp),
    )
    tunnel_defense = torch.where(
        state.terrain_buff_s[:, :, Terrain.TUNNEL] > 0,
        torch.full_like(state.hp, 0.50),
        torch.zeros_like(state.hp),
    )
    state.terrain_defense_fraction.copy_(torch.maximum(group_defense, tunnel_defense))

    dead = ~state.alive
    state.terrain_buff_s.copy_(torch.where(dead[:, :, None], 0.0, state.terrain_buff_s))
    state.terrain_group_enhanced_s.copy_(torch.where(dead, 0.0, state.terrain_group_enhanced_s))
    state.tunnel_cooling_s.copy_(torch.where(dead, 0.0, state.tunnel_cooling_s))

    enemy_fortress = zone_active[:, :, Zone.ENEMY_FORTRESS]
    fortress_roles = (
        (roles == Role.INFANTRY_3) | (roles == Role.INFANTRY_4) | (roles == Role.SENTRY)
    )
    enemy_outpost_destroyed = state.outpost_destroyed_once[:, 1 - own_teams]
    enemy_base_deployed = state.base_armor_deployed[:, 1 - own_teams]
    capture_active = (
        enemy_fortress
        & fortress_roles[None, :]
        & (state.elapsed_s[:, None] >= 180.0)
        & enemy_outpost_destroyed
        & ~enemy_base_deployed
    )
    state.enemy_fortress_progress_s.add_(capture_active.to(state.dtype) * dt)
    state.enemy_fortress_hold_s.copy_(
        torch.where(
            capture_active,
            torch.full_like(state.enemy_fortress_hold_s, 3.0),
            torch.clamp(state.enemy_fortress_hold_s - dt, min=0),
        )
    )
    state.enemy_fortress_progress_s.copy_(
        torch.where(
            capture_active | (state.enemy_fortress_hold_s > 0),
            state.enemy_fortress_progress_s,
            torch.zeros_like(state.enemy_fortress_progress_s),
        )
    )
    captured = (state.enemy_fortress_progress_s >= constants.ENEMY_FORTRESS_CAPTURE_S).view(
        state.num_envs, constants.TEAM_COUNT, constants.ROLES_PER_TEAM
    )
    captured_by_team = captured.any(dim=-1)
    state.base_armor_deployed |= captured_by_team.flip(dims=(1,))
    state.zone_vulnerability_fraction.copy_(
        torch.where(
            capture_active & ~enemy_base_deployed,
            torch.ones_like(state.hp),
            torch.zeros_like(state.hp),
        )
    )


def compose_modifiers(state: GameState) -> None:
    """Compose same-type effects using the official maximum-effect rule."""

    teams = unit_teams(state.device)
    tech_defense = state.tech_defense_fraction[:, teams]
    rune_defense = state.rune_defense_fraction[:, teams]
    state.defense_fraction.copy_(
        torch.maximum(
            torch.maximum(tech_defense, rune_defense),
            torch.maximum(
                torch.maximum(state.role_defense_fraction, state.zone_defense_fraction),
                state.terrain_defense_fraction,
            ),
        )
    )
    state.vulnerability_fraction.copy_(
        torch.maximum(
            state.role_vulnerability_fraction,
            state.zone_vulnerability_fraction,
        )
    )
    state.attack_multiplier.copy_(
        torch.maximum(
            torch.ones_like(state.attack_multiplier),
            state.rune_attack_multiplier[:, teams],
        )
    )

    rune_cooling = state.base_cooling_per_s * state.rune_cooling_multiplier[:, teams, None]
    tunnel_cooling = state.base_cooling_per_s * torch.where(
        state.tunnel_cooling_s[:, :, None] > 0,
        torch.full_like(state.base_cooling_per_s, constants.TUNNEL_COOLING_MULTIPLIER),
        torch.ones_like(state.base_cooling_per_s),
    )
    additive_cooling = state.base_cooling_per_s + state.zone_cooling_bonus_per_s
    pre_stance_cooling = torch.maximum(
        torch.maximum(rune_cooling, tunnel_cooling),
        additive_cooling,
    )
    state.cooling_per_s.copy_(pre_stance_cooling * state.role_cooling_multiplier[:, :, None])

    effective_power = state.base_power_limit_w * state.role_power_multiplier
    boosted = torch.clamp(
        state.base_power_limit_w * 2.0,
        max=constants.CHASSIS_POWER_ABSOLUTE_MAX_W,
    )
    effective_power = torch.where(
        state.immediate_power_boost_s > 0,
        boosted,
        effective_power,
    )
    state.power_limit_w.copy_(effective_power)
