from __future__ import annotations

import pytest
import torch

from rm_referee import constants
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


def test_radar_report_tensors_have_stable_shapes_and_clone_by_value() -> None:
    state = GameState.create(2)
    inputs = RuleInputs.empty(state)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    inputs.radar_report_xy[0, Team.RED, target] = torch.tensor((3.25, -1.5))

    inputs_copy = inputs.clone()
    state_copy = state.clone()
    inputs_copy.radar_report_xy[0, Team.RED, target, 0] = 99
    state_copy.radar_report_valid[0, Team.RED, target] = True

    expected_shape = (2, constants.TEAM_COUNT, constants.UNIT_COUNT)
    assert inputs.radar_report_xy.shape == (*expected_shape, 2)
    assert state.radar_report_xy.shape == (*expected_shape, 2)
    assert state.radar_report_valid.shape == expected_shape
    assert state.radar_report_age_s.shape == expected_shape
    assert inputs.radar_report_xy[0, Team.RED, target, 0] == pytest.approx(3.25)
    assert not state.radar_report_valid[0, Team.RED, target]


def test_real_radar_report_persists_coordinates_and_tracks_age(
    state: GameState,
    referee: Referee,
) -> None:
    target = slot(Team.BLUE, Role.INFANTRY_3)
    report = _radar_update(state, RadarQuality.ACCURATE)
    report.radar_report_xy[0, Team.RED, target] = torch.tensor((4.5, -2.25))

    current, _ = referee.step(state, report)

    assert current.radar_report_valid[0, Team.RED, target]
    assert current.radar_report_age_s[0, Team.RED, target] == 0
    assert torch.equal(
        current.radar_report_xy[0, Team.RED, target],
        torch.tensor((4.5, -2.25)),
    )

    finished = current.clone()
    finished.done[0] = True
    finished, _ = referee.step(finished, RuleInputs.empty(finished))
    assert finished.radar_report_age_s[0, Team.RED, target] == 0

    for _ in range(5):
        current, _ = referee.step(current, RuleInputs.empty(current))

    assert current.radar_last_quality[0, Team.RED, target] == RadarQuality.WRONG
    assert current.radar_report_age_s[0, Team.RED, target] == pytest.approx(0.5)
    assert torch.equal(
        current.radar_report_xy[0, Team.RED, target],
        torch.tensor((4.5, -2.25)),
    )

    replacement = _radar_update(current, RadarQuality.HALF_ACCURATE)
    replacement.radar_report_xy[0, Team.RED, target] = torch.tensor((-3.0, 6.0))
    current, _ = referee.step(current, replacement)

    assert current.radar_report_age_s[0, Team.RED, target] == 0
    assert torch.equal(
        current.radar_report_xy[0, Team.RED, target],
        torch.tensor((-3.0, 6.0)),
    )


def test_no_data_quality_event_does_not_create_a_radar_report(
    state: GameState,
    referee: Referee,
) -> None:
    target = slot(Team.BLUE, Role.INFANTRY_3)
    current = state

    for _ in range(5):
        current, _ = referee.step(current, RuleInputs.empty(current))

    assert current.radar_last_quality[0, Team.RED, target] == RadarQuality.WRONG
    assert not current.radar_report_valid[0, Team.RED, target]
    assert current.radar_report_age_s[0, Team.RED, target] == 0
    assert torch.equal(
        current.radar_report_xy[0, Team.RED, target],
        torch.zeros(2),
    )


def test_radar_truth_visibility_uses_own_and_enemy_thresholds(
    state: GameState,
    referee: Referee,
) -> None:
    own_target = slot(Team.RED, Role.HERO)
    enemy_target = slot(Team.BLUE, Role.HERO)
    state.radar_p[0, Team.RED, own_target] = 49
    state.radar_p[0, Team.RED, enemy_target] = 99

    below_threshold, _ = referee.step(state, RuleInputs.empty(state))

    assert not below_threshold.radar_truth_visible[0, Team.RED, own_target]
    assert not below_threshold.radar_truth_visible[0, Team.RED, enemy_target]

    below_threshold.radar_p[0, Team.RED, own_target] = 50
    below_threshold.radar_p[0, Team.RED, enemy_target] = 100
    at_threshold, _ = referee.step(
        below_threshold,
        RuleInputs.empty(below_threshold),
    )

    assert at_threshold.radar_truth_visible[0, Team.RED, own_target]
    assert at_threshold.radar_truth_visible[0, Team.RED, enemy_target]
    assert at_threshold.radar_vulnerability_fraction[0, enemy_target] == pytest.approx(0.15)


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
