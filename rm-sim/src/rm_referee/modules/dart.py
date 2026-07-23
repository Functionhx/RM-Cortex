"""Dart gate, detection window, and direct-damage rules."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import (
    DartGateState,
    DartTarget,
    Role,
    Team,
    ground_robot_mask,
    slot,
    unit_teams,
)
from rm_referee.state import GameState
from rm_referee.tensor_ops import round_half_up


def apply_dart(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active = (~state.done)[:, None]
    thresholds = torch.tensor(
        constants.DART_GATE_SCHEDULE_S,
        device=state.device,
        dtype=state.dtype,
    )
    opportunities = (
        (state.elapsed_s[:, None] < thresholds)
        & (state.elapsed_s[:, None] + dt >= thresholds)
        & (~state.done)[:, None]
    ).sum(dim=1)
    state.dart_gate_opportunities.add_(opportunities[:, None])

    state.dart_detection_s.sub_(dt * active).clamp_(min=0)
    state.dart_detection_block_s.sub_(dt * active).clamp_(min=0)
    state.dart_zone_disabled_s.sub_(dt * active).clamp_(min=0)
    timed_state = (
        (state.dart_gate_state == DartGateState.OPENING)
        | (state.dart_gate_state == DartGateState.OPEN)
        | (state.dart_gate_state == DartGateState.COOLDOWN)
    )
    state.dart_gate_timer_s.copy_(
        torch.where(
            timed_state & active,
            torch.clamp(state.dart_gate_timer_s - dt, min=0),
            state.dart_gate_timer_s,
        )
    )

    start_opening = (
        active
        & inputs.dart_open_gate
        & (state.dart_gate_state == DartGateState.CLOSED)
        & (state.dart_gate_opportunities > 0)
        & (state.dart_rounds > 0)
    )
    state.dart_gate_opportunities.sub_(start_opening.to(torch.long))
    state.dart_gate_state.copy_(
        torch.where(
            start_opening,
            torch.full_like(state.dart_gate_state, DartGateState.OPENING),
            state.dart_gate_state,
        )
    )
    state.dart_gate_timer_s.copy_(
        torch.where(
            start_opening,
            torch.full_like(
                state.dart_gate_timer_s,
                constants.DART_GATE_OPENING_S,
            ),
            state.dart_gate_timer_s,
        )
    )

    fully_open = (state.dart_gate_state == DartGateState.OPENING) & (state.dart_gate_timer_s <= 0)
    state.dart_gate_state.copy_(
        torch.where(
            fully_open,
            torch.full_like(state.dart_gate_state, DartGateState.OPEN),
            state.dart_gate_state,
        )
    )
    state.dart_gate_timer_s.copy_(
        torch.where(
            fully_open,
            torch.full_like(
                state.dart_gate_timer_s,
                constants.DART_FIRE_WINDOW_S,
            ),
            state.dart_gate_timer_s,
        )
    )
    state.dart_detection_s.copy_(
        torch.where(
            fully_open,
            torch.full_like(
                state.dart_detection_s,
                constants.DART_DETECTION_WINDOW_S,
            ),
            state.dart_detection_s,
        )
    )

    close = (state.dart_gate_state == DartGateState.OPEN) & (
        inputs.dart_close_gate | (state.dart_gate_timer_s <= 0)
    )
    state.dart_gate_state.copy_(
        torch.where(
            close,
            torch.full_like(state.dart_gate_state, DartGateState.COOLDOWN),
            state.dart_gate_state,
        )
    )
    state.dart_gate_timer_s.copy_(
        torch.where(
            close,
            torch.full_like(
                state.dart_gate_timer_s,
                constants.DART_GATE_COOLDOWN_S,
            ),
            state.dart_gate_timer_s,
        )
    )
    cooldown_done = (state.dart_gate_state == DartGateState.COOLDOWN) & (
        state.dart_gate_timer_s <= 0
    )
    state.dart_gate_state.copy_(
        torch.where(
            cooldown_done,
            torch.full_like(state.dart_gate_state, DartGateState.CLOSED),
            state.dart_gate_state,
        )
    )

    target = inputs.dart_fire_target.to(torch.long)
    target_safe = torch.clamp(target, min=0)
    target_outpost_slots = torch.tensor(
        [slot(Team.BLUE, Role.OUTPOST), slot(Team.RED, Role.OUTPOST)],
        device=state.device,
    )
    enemy_outpost_alive = state.alive[:, target_outpost_slots]
    target_allowed = torch.where(
        enemy_outpost_alive,
        target == DartTarget.OUTPOST,
        target > DartTarget.OUTPOST,
    )
    fired = (
        active
        & (state.dart_gate_state == DartGateState.OPEN)
        & (target >= 0)
        & target_allowed
        & (state.dart_rounds > 0)
    )
    state.dart_rounds.sub_(fired.to(torch.long))
    events.dart_fired.copy_(fired)
    recognized = (
        fired & inputs.dart_hit & (state.dart_detection_s > 0) & (state.dart_detection_block_s <= 0)
    )
    events.dart_hits.copy_(recognized)
    state.dart_detection_block_s.copy_(
        torch.where(
            recognized,
            torch.full_like(
                state.dart_detection_block_s,
                constants.DART_DETECTION_BLOCK_S,
            ),
            state.dart_detection_block_s,
        )
    )
    state.dart_zone_disabled_s.copy_(
        torch.where(
            recognized.flip(dims=(1,)),
            torch.full_like(
                state.dart_zone_disabled_s,
                constants.DART_ZONE_DISABLE_S,
            ),
            state.dart_zone_disabled_s,
        )
    )

    target_one_hot = torch.nn.functional.one_hot(
        target_safe,
        constants.DART_TARGET_COUNT,
    ).to(torch.long)
    state.dart_hits_by_target.add_(target_one_hot * recognized[:, :, None].to(torch.long))
    damage_table = torch.tensor(
        constants.DART_DAMAGE,
        device=state.device,
        dtype=state.dtype,
    )
    direct_damage = torch.where(
        recognized,
        damage_table[target_safe],
        torch.zeros_like(state.dart_detection_s),
    )
    target_is_base = target > DartTarget.OUTPOST
    shield_available = state.base_shield.flip(dims=(1,))
    shield_absorbed = torch.where(
        target_is_base,
        torch.minimum(direct_damage, shield_available),
        torch.zeros_like(direct_damage),
    )
    hp_damage_requested = direct_damage - shield_absorbed
    target_base_slots = torch.tensor(
        [slot(Team.BLUE, Role.BASE), slot(Team.RED, Role.BASE)],
        device=state.device,
    )
    target_slots = torch.where(target_is_base, target_base_slots, target_outpost_slots)
    target_hp = torch.gather(state.hp, 1, target_slots)
    hp_damage = torch.minimum(hp_damage_requested, target_hp)
    state.hp.scatter_(
        1,
        target_slots,
        target_hp - hp_damage,
    )
    state.base_shield.sub_(shield_absorbed.flip(dims=(1,))).clamp_(min=0)
    state.base_hp_lost.add_(torch.where(target_is_base, hp_damage, 0.0).flip(dims=(1,)))
    effective_direct = shield_absorbed + hp_damage
    direct_by_unit = torch.zeros_like(state.hp)
    direct_by_unit.scatter_add_(1, target_slots, effective_direct)
    events.damage_taken.add_(direct_by_unit)

    ground_fraction_table = torch.tensor(
        constants.DART_GROUND_HP_FRACTION,
        device=state.device,
        dtype=state.dtype,
    )
    ground_fraction = torch.where(
        recognized,
        ground_fraction_table[target_safe],
        torch.zeros_like(state.dart_detection_s),
    )
    teams = unit_teams(state.device)
    fraction_by_unit = ground_fraction[:, 1 - teams]
    ground_damage = round_half_up(state.max_hp * fraction_by_unit)
    ground_damage = torch.where(
        ground_robot_mask(state.device)[None, :] & state.alive & (state.invulnerable_s <= 0),
        torch.minimum(ground_damage, state.hp),
        torch.zeros_like(ground_damage),
    )
    state.hp.sub_(ground_damage)
    events.damage_taken.add_(ground_damage)

    ground_damage_by_target_team = ground_damage.view(
        state.num_envs,
        constants.TEAM_COUNT,
        constants.ROLES_PER_TEAM,
    ).sum(dim=-1)
    state.team_damage.add_(effective_direct + ground_damage_by_target_team.flip(dims=(1,)))

    moving_hit = recognized & (
        (target == DartTarget.BASE_RANDOM_MOVING) | (target == DartTarget.BASE_TERMINAL_MOVING)
    )
    four_fixed_hits = (state.dart_hits_by_target[:, :, DartTarget.BASE_FIXED] >= 4) | (
        state.dart_hits_by_target[:, :, DartTarget.BASE_RANDOM_FIXED] >= 4
    )
    expand_enemy = moving_hit | four_fixed_hits
    state.base_armor_deployed |= expand_enemy.flip(dims=(1,))
