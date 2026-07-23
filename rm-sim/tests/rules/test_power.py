from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, Team, slot


def test_power_buffer_exhaustion_disables_chassis_without_hp_loss(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    engineer = slot(Team.RED, Role.ENGINEER)
    state.power_buffer_j[0, engineer] = 1
    inputs.chassis_power_w[0, engineer] = 130
    hp_before = state.hp[0, engineer].item()

    next_state, _ = referee.step(state, inputs)

    assert next_state.power_buffer_j[0, engineer] == pytest.approx(0)
    assert next_state.chassis_disabled_s[0, engineer] == pytest.approx(5)
    assert next_state.hp[0, engineer] == hp_before


def test_power_below_limit_recharges_buffer(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    engineer = slot(Team.RED, Role.ENGINEER)
    state.power_buffer_j[0, engineer] = 20
    inputs.chassis_power_w[0, engineer] = 100

    next_state, _ = referee.step(state, inputs)

    assert next_state.power_buffer_j[0, engineer] == pytest.approx(22)
