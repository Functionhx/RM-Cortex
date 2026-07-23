from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import (
    DartGateState,
    DartTarget,
    Role,
    RuneMode,
    Team,
    slot,
)


def test_technology_levels_apply_income_cap_defense_and_base_shield() -> None:
    state = GameState.create(1)
    state.elapsed_s[0] = 180
    state.tech_completion_count[0, Team.RED, :3] = 1
    state.tech_energy_units_remaining[0] = 2
    base = slot(Team.RED, Role.BASE)
    state.hp[0, base] = 4000
    state.base_shield[0, Team.RED] = 0
    inputs = RuleInputs.empty(state)
    inputs.tech_complete_level[0, Team.RED] = 4

    next_state, events = Referee().step(state, inputs)

    assert events.tech_completed[0, Team.RED] == 4
    assert next_state.team_level_cap[0, Team.RED] == 10
    assert next_state.tech_defense_fraction[0, Team.RED] == pytest.approx(0.5)
    assert next_state.hp[0, base] == 5000
    assert next_state.base_shield[0, Team.RED] == 1000
    assert next_state.tech_income_per_10s[0, Team.RED] == 50


def test_small_rune_activates_after_five_correct_groups() -> None:
    state = GameState.create(1)
    referee = Referee()
    trigger = RuleInputs.empty(state)
    trigger.rune_trigger[0, Team.RED] = RuneMode.SMALL
    state, _ = referee.step(state, trigger)

    for _ in range(5):
        hit = RuleInputs.empty(state)
        hit.rune_group_correct[0, Team.RED] = True
        hit.rune_group_hits[0, Team.RED] = 1
        hit.rune_group_ring_sum[0, Team.RED] = 5
        state, events = referee.step(state, hit)

    assert events.rune_activated[0, Team.RED]
    assert state.rune_small_buff_s[0, Team.RED] == pytest.approx(45)
    assert state.rune_defense_fraction[0, Team.RED] == pytest.approx(0.25)


def test_large_rune_uses_ring_and_arm_tables_and_shares_xp() -> None:
    state = GameState.create(1)
    state.elapsed_s[0] = 180
    state.rune_large_opportunities[0, Team.RED] = 1
    referee = Referee()
    trigger = RuleInputs.empty(state)
    trigger.rune_trigger[0, Team.RED] = RuneMode.LARGE
    state, _ = referee.step(state, trigger)

    for _ in range(5):
        hit = RuleInputs.empty(state)
        hit.rune_group_correct[0, Team.RED] = True
        hit.rune_group_hits[0, Team.RED] = 2
        hit.rune_group_ring_sum[0, Team.RED] = 20
        state, events = referee.step(state, hit)

    assert events.rune_activated[0, Team.RED]
    assert state.rune_buff_s[0, Team.RED] == pytest.approx(60)
    assert state.rune_attack_multiplier[0, Team.RED] == pytest.approx(3)
    assert state.rune_defense_fraction[0, Team.RED] == pytest.approx(0.5)
    assert state.rune_cooling_multiplier[0, Team.RED] == pytest.approx(5)
    assert events.xp_gained[0, slot(Team.RED, Role.HERO)] == pytest.approx(250)


def test_dart_outpost_and_moving_base_damage() -> None:
    state = GameState.create(1)
    state.dart_gate_state[0, Team.RED] = DartGateState.OPEN
    state.dart_gate_timer_s[0, Team.RED] = 30
    state.dart_detection_s[0, Team.RED] = 40
    outpost = slot(Team.BLUE, Role.OUTPOST)
    fire = RuleInputs.empty(state)
    fire.dart_fire_target[0, Team.RED] = DartTarget.OUTPOST
    fire.dart_hit[0, Team.RED] = True

    state, events = Referee().step(state, fire)
    assert events.dart_hits[0, Team.RED]
    assert state.hp[0, outpost] == 750

    state.hp[0, outpost] = 0
    state.alive[0, outpost] = False
    state.outpost_destroyed_once[0, Team.BLUE] = True
    state.dart_gate_state[0, Team.RED] = DartGateState.OPEN
    state.dart_gate_timer_s[0, Team.RED] = 30
    state.dart_detection_s[0, Team.RED] = 40
    state.dart_detection_block_s[0, Team.RED] = 0
    base = slot(Team.BLUE, Role.BASE)
    infantry = slot(Team.BLUE, Role.INFANTRY_3)
    moving = RuleInputs.empty(state)
    moving.dart_fire_target[0, Team.RED] = DartTarget.BASE_RANDOM_MOVING
    moving.dart_hit[0, Team.RED] = True

    state, _ = Referee().step(state, moving)

    assert state.base_shield[0, Team.BLUE] == 0
    assert state.hp[0, base] == 4525
    assert state.hp[0, infantry] == 180
    assert state.base_armor_deployed[0, Team.BLUE]
