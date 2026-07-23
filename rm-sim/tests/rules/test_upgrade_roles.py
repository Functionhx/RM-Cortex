from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, SentryStance, Team, Weapon, slot


def test_level_up_applies_official_remote_hero_table() -> None:
    state = GameState.create(1)
    hero = slot(Team.RED, Role.HERO)
    state.xp[0, hero] = 540
    state.ammo[0, hero, Weapon.MM42] = 1
    inputs = RuleInputs.empty(state)
    inputs.shots_fired[0, hero, Weapon.MM42] = 1

    next_state, events = Referee().step(state, inputs)

    assert events.xp_gained[0, hero] == 10
    assert next_state.level[0, hero] == 2
    assert next_state.max_hp[0, hero] == 165
    assert next_state.hp[0, hero] == 165
    assert next_state.base_power_limit_w[0, hero] == 55
    assert next_state.heat_limit[0, hero, Weapon.MM42] == 102
    assert next_state.base_cooling_per_s[0, hero, Weapon.MM42] == 23


def test_remote_hero_deploys_after_two_second_confirmation() -> None:
    state = GameState.create(1)
    hero = slot(Team.RED, Role.HERO)
    referee = Referee()

    for _ in range(21):
        inputs = RuleInputs.empty(state)
        inputs.hero_deploy_request[0, Team.RED] = True
        inputs.in_hero_deploy_zone[0, hero] = True
        state, _ = referee.step(state, inputs)

    assert state.hero_deployed[0, Team.RED]
    assert state.defense_fraction[0, hero] == pytest.approx(0.25)
    assert state.muzzle_velocity_limit_mps[0, hero, Weapon.MM42] == pytest.approx(16.5)
    assert state.video_disabled[0, hero]


def test_sentry_stance_degrades_after_three_minutes_in_that_stance() -> None:
    state = GameState.create(1)
    sentry = slot(Team.RED, Role.SENTRY)
    state.sentry_stance[0, Team.RED] = SentryStance.OFFENSIVE
    state.sentry_stance_time_s[0, Team.RED, SentryStance.OFFENSIVE] = 180.1

    next_state, _ = Referee().step(state, RuleInputs.empty(state))

    assert next_state.role_cooling_multiplier[0, sentry] == pytest.approx(2)
    assert next_state.role_power_multiplier[0, sentry] == pytest.approx(0.5)
    assert next_state.vulnerability_fraction[0, sentry] == pytest.approx(0.25)


def test_radar_countermeasure_locks_aerial_after_triangular_progress() -> None:
    state = GameState.create(1)
    referee = Referee()

    for _ in range(10):
        inputs = RuleInputs.empty(state)
        inputs.aerial_support_request[0, Team.BLUE] = True
        inputs.aerial_on_pad[0, Team.BLUE] = False
        inputs.radar_illuminating[0, Team.RED] = True
        state, _ = referee.step(state, inputs)

    aerial = slot(Team.BLUE, Role.AERIAL)
    assert state.alive[0, aerial]
    assert state.aerial_counter_uses[0, Team.BLUE] == 1
    assert state.aerial_counter_lock_s[0, Team.BLUE] == pytest.approx(45)


def test_aerial_support_bank_can_pause_and_receives_minute_grant() -> None:
    state = GameState.create(1)
    referee = Referee()
    aerial = slot(Team.RED, Role.AERIAL)
    assert state.aerial_support_bank_s[0, Team.RED] == 30
    assert state.ammo[0, aerial, Weapon.MM17] == 750

    active = RuleInputs.empty(state)
    active.aerial_support_request[0, Team.RED] = True
    active.aerial_on_pad[0, Team.RED] = False
    state, _ = referee.step(state, active)
    assert state.alive[0, aerial]
    assert state.aerial_support_bank_s[0, Team.RED] == pytest.approx(29.9)

    paused_bank = state.aerial_support_bank_s[0, Team.RED].item()
    state, _ = referee.step(state, RuleInputs.empty(state))
    assert not state.alive[0, aerial]
    assert state.aerial_support_bank_s[0, Team.RED] == pytest.approx(paused_bank)

    state.elapsed_s[0] = 59.95
    state, _ = referee.step(state, RuleInputs.empty(state))
    assert state.aerial_support_bank_s[0, Team.RED] == pytest.approx(
        paused_bank + 20,
    )


def test_aerial_can_only_fire_during_support_and_off_the_pad() -> None:
    state = GameState.create(1)
    referee = Referee()
    aerial = slot(Team.RED, Role.AERIAL)

    unsupported = RuleInputs.empty(state)
    unsupported.aerial_on_pad[0, Team.RED] = False
    unsupported.shots_fired[0, aerial, Weapon.MM17] = 1
    state, events = referee.step(state, unsupported)
    assert events.shots_fired[0, aerial, Weapon.MM17] == 0
    assert state.ammo[0, aerial, Weapon.MM17] == 750

    on_pad = RuleInputs.empty(state)
    on_pad.aerial_support_request[0, Team.RED] = True
    on_pad.aerial_on_pad[0, Team.RED] = True
    on_pad.shots_fired[0, aerial, Weapon.MM17] = 1
    state, events = referee.step(state, on_pad)
    assert events.shots_fired[0, aerial, Weapon.MM17] == 0
    assert state.ammo[0, aerial, Weapon.MM17] == 750

    airborne = RuleInputs.empty(state)
    airborne.aerial_support_request[0, Team.RED] = True
    airborne.aerial_on_pad[0, Team.RED] = False
    airborne.shots_fired[0, aerial, Weapon.MM17] = 1
    state, events = referee.step(state, airborne)
    assert events.shots_fired[0, aerial, Weapon.MM17] == 1
    assert state.ammo[0, aerial, Weapon.MM17] == 749


def test_radar_countermeasure_blocks_aerial_fire_during_support() -> None:
    state = GameState.create(1)
    referee = Referee()
    aerial = slot(Team.BLUE, Role.AERIAL)
    state.aerial_counter_lock_s[0, Team.BLUE] = 45

    locked = RuleInputs.empty(state)
    locked.aerial_support_request[0, Team.BLUE] = True
    locked.aerial_on_pad[0, Team.BLUE] = False
    locked.shots_fired[0, aerial, Weapon.MM17] = 1

    state, events = referee.step(state, locked)

    assert state.alive[0, aerial]
    assert events.shots_fired[0, aerial, Weapon.MM17] == 0
    assert state.ammo[0, aerial, Weapon.MM17] == 750


def test_aerial_paid_support_costs_one_coin_per_second_then_stops() -> None:
    state = GameState.create(1)
    referee = Referee()
    state.aerial_support_bank_s[0, Team.RED] = 0
    state.team_coin[0, Team.RED] = 1

    for _ in range(10):
        paid = RuleInputs.empty(state)
        paid.aerial_support_request[0, Team.RED] = True
        paid.aerial_on_pad[0, Team.RED] = False
        state, _ = referee.step(state, paid)

    assert state.aerial_support_active[0, Team.RED]
    assert state.team_coin[0, Team.RED] == 0

    no_coin = RuleInputs.empty(state)
    no_coin.aerial_support_request[0, Team.RED] = True
    no_coin.aerial_on_pad[0, Team.RED] = False
    state, _ = referee.step(state, no_coin)

    assert not state.aerial_support_active[0, Team.RED]
