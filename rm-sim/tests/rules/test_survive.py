from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, Team, Weapon, slot


def test_read_bar_respawn_returns_with_ten_percent_hp(
    state: GameState,
    referee: Referee,
) -> None:
    source = slot(Team.BLUE, Role.INFANTRY_3)
    target = slot(Team.RED, Role.INFANTRY_3)
    state.ammo[0, source, Weapon.MM17] = 1
    state.hp[0, target] = 20
    hit = RuleInputs.empty(state)
    hit.shots_fired[0, source, Weapon.MM17] = 1
    hit.hits.source[0, target, 0, Weapon.MM17, 0] = source
    hit.hits.time_offset_s[0, target, 0, Weapon.MM17, 0] = 0

    current, events = referee.step(state, hit)
    assert events.deaths[0, target]
    assert not current.alive[0, target]
    assert current.respawn_required[0, target] == 10

    respawned = False
    for _ in range(100):
        current, events = referee.step(current, RuleInputs.empty(current))
        if events.respawns[0, target]:
            respawned = True
            break

    assert respawned
    assert current.hp[0, target] == pytest.approx(current.max_hp[0, target] * 0.10)
    assert current.invulnerable_s[0, target] == pytest.approx(30)
    assert current.weak[0, target]


def test_supply_zone_heals_ten_percent_per_second(
    state: GameState,
    referee: Referee,
) -> None:
    target = slot(Team.RED, Role.ENGINEER)
    state.hp[0, target] = 100
    inputs = RuleInputs.empty(state)
    inputs.in_supply_zone[0, target] = True

    next_state, _ = referee.step(state, inputs)

    assert next_state.hp[0, target] == pytest.approx(102.5)
