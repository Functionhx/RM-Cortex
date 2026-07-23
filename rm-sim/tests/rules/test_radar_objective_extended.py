from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import RadarQuality, Role, Team, slot


def test_radar_generates_wrong_update_after_half_second_without_data() -> None:
    state = GameState.create(1)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    state.radar_x[0, Team.RED, target] = 1
    state.radar_p[0, Team.RED, target] = 10
    state.radar_last_quality[0, Team.RED, target] = RadarQuality.ACCURATE

    for _ in range(5):
        state, _ = Referee().step(state, RuleInputs.empty(state))

    assert state.radar_x[0, Team.RED, target] == pytest.approx(-0.8)
    assert state.radar_p[0, Team.RED, target] == pytest.approx(9.2)


def test_harder_interference_suppresses_radar_vulnerability() -> None:
    state = GameState.create(1)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    state.radar_p[0, Team.RED, target] = 120
    state.radar_interference_level[0, Team.BLUE] = 2
    state.radar_interference_level[0, Team.RED] = 1

    next_state, _ = Referee().step(state, RuleInputs.empty(state))

    assert next_state.radar_vulnerability_fraction[0, target] == 0


def test_double_radar_vulnerability_uses_accumulated_opportunity() -> None:
    state = GameState.create(1)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    state.radar_p[0, Team.RED, target] = 120
    state.radar_vulnerability_progress_s[0, Team.RED] = 59.9
    state, _ = Referee().step(state, RuleInputs.empty(state))
    assert state.radar_double_charges[0, Team.RED] == 1

    activate = RuleInputs.empty(state)
    activate.radar_double_request[0, Team.RED] = True
    state, _ = Referee().step(state, activate)

    assert state.radar_double_s[0, Team.RED] == pytest.approx(30)
    assert state.radar_vulnerability_fraction[0, target] == pytest.approx(0.4)


def test_engineer_rebuilds_outpost_in_five_seconds_with_a_charge() -> None:
    state = GameState.create(1)
    outpost = slot(Team.RED, Role.OUTPOST)
    engineer = slot(Team.RED, Role.ENGINEER)
    state.hp[0, outpost] = 0
    state.alive[0, outpost] = False
    state.outpost_destroyed_once[0, Team.RED] = True
    state.base_hp_lost[0, Team.RED] = 1000
    referee = Referee()

    for _ in range(51):
        inputs = RuleInputs.empty(state)
        inputs.in_outpost_zone[0, engineer] = True
        inputs.outpost_rebuild_request[0, engineer] = True
        state, _ = referee.step(state, inputs)
        if state.alive[0, outpost]:
            break

    assert state.alive[0, outpost]
    assert state.hp[0, outpost] == 750
    assert state.outpost_rebuild_charges[0, Team.RED] == 0
