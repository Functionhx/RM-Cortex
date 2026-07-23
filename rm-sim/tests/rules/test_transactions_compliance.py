from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import PurchaseKind, Role, Team, Weapon, slot


def test_remote_ammo_purchase_settles_after_six_seconds() -> None:
    state = GameState.create(1)
    state.team_coin[0, Team.RED] = 500
    hero = slot(Team.RED, Role.HERO)
    referee = Referee()

    request = RuleInputs.empty(state)
    request.purchase_unit[0, Team.RED] = hero
    request.purchase_weapon[0, Team.RED] = Weapon.MM42
    request.purchase_kind[0, Team.RED] = PurchaseKind.REMOTE
    state, _ = referee.step(state, request)
    assert state.team_coin[0, Team.RED] == 350
    assert state.ammo[0, hero, Weapon.MM42] == 0

    for _ in range(60):
        state, events = referee.step(state, RuleInputs.empty(state))
        if events.purchased_rounds[0, Team.RED]:
            break

    assert state.ammo[0, hero, Weapon.MM42] == 10


def test_remote_heal_and_immediate_respawn_charge_official_costs() -> None:
    state = GameState.create(1)
    hero = slot(Team.RED, Role.HERO)
    state.team_coin[0, Team.RED] = 500
    state.hp[0, hero] = 50
    referee = Referee()

    request = RuleInputs.empty(state)
    request.remote_heal_request[0, hero] = True
    state, _ = referee.step(state, request)
    assert state.team_coin[0, Team.RED] == 450

    for _ in range(60):
        state, events = referee.step(state, RuleInputs.empty(state))
        if events.remote_heals[0, hero]:
            break
    assert state.hp[0, hero] == pytest.approx(140)

    state.alive[0, hero] = False
    state.hp[0, hero] = 0
    state.respawn_required[0, hero] = 10
    immediate = RuleInputs.empty(state)
    immediate.immediate_respawn_request[0, hero] = True
    state, events = referee.step(state, immediate)

    assert events.immediate_respawns[0, hero]
    assert state.hp[0, hero] == state.max_hp[0, hero]
    assert state.immediate_respawn_count[0, hero] == 1
    assert state.immediate_power_boost_s[0, hero] == pytest.approx(4)


def test_muzzle_velocity_and_offline_modules_use_independent_locks_and_damage() -> None:
    state = GameState.create(1)
    infantry = slot(Team.RED, Role.INFANTRY_3)
    state.ammo[0, infantry, Weapon.MM17] = 1
    fire = RuleInputs.empty(state)
    fire.shots_fired[0, infantry, Weapon.MM17] = 1
    fire.muzzle_velocity_mps[0, infantry, Weapon.MM17] = 29

    state, _ = Referee().step(state, fire)
    assert state.velocity_lock_s[0, infantry, Weapon.MM17] == pytest.approx(15)

    hp_before = state.hp[0, infantry].item()
    for _ in range(5):
        offline = RuleInputs.empty(state)
        offline.offline_module_count[0, infantry] = 2
        state, _ = Referee().step(state, offline)
    assert state.hp[0, infantry] == pytest.approx(hp_before - 8)


def test_wireless_charging_uses_eight_to_one_energy_conversion() -> None:
    state = GameState.create(1)
    infantry = slot(Team.RED, Role.INFANTRY_3)
    state.chassis_energy[0, infantry] = 1000
    inputs = RuleInputs.empty(state)
    inputs.chassis_power_w[0, infantry] = 50
    inputs.supercap_input_w[0, infantry] = 100
    inputs.in_supply_zone[0, infantry] = True

    next_state, _ = Referee().step(state, inputs)

    assert next_state.chassis_energy[0, infantry] == pytest.approx(1035)
