from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import RadarQuality, Role, Team, slot


def _radar_update(state: GameState, quality: RadarQuality) -> RuleInputs:
    inputs = RuleInputs.empty(state)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    inputs.radar_update[0, Team.RED, target] = True
    inputs.radar_quality[0, Team.RED, target] = quality
    return inputs


def test_radar_updates_x_then_accumulates_p(
    state: GameState,
    referee: Referee,
) -> None:
    target = slot(Team.BLUE, Role.INFANTRY_3)
    current = state
    expected = [(1, 1), (2, 3), (3, 6)]

    for expected_x, expected_p in expected:
        current, _ = referee.step(
            current,
            _radar_update(current, RadarQuality.ACCURATE),
        )
        assert current.radar_x[0, Team.RED, target] == pytest.approx(expected_x)
        assert current.radar_p[0, Team.RED, target] == pytest.approx(expected_p)

    current, _ = referee.step(
        current,
        _radar_update(current, RadarQuality.WRONG),
    )
    assert current.radar_x[0, Team.RED, target] == pytest.approx(-0.8)
    assert current.radar_p[0, Team.RED, target] == pytest.approx(5.2)


def test_radar_vulnerability_never_applies_to_aerial(
    state: GameState,
    referee: Referee,
) -> None:
    ground = slot(Team.BLUE, Role.INFANTRY_3)
    aerial = slot(Team.BLUE, Role.AERIAL)
    state.radar_p[0, Team.RED, ground] = 120
    state.radar_p[0, Team.RED, aerial] = 150

    next_state, _ = referee.step(state, RuleInputs.empty(state))

    assert next_state.radar_vulnerability_fraction[0, ground] == pytest.approx(0.20)
    assert next_state.radar_vulnerability_fraction[0, aerial] == 0
