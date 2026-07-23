from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs


@pytest.fixture
def state() -> GameState:
    return GameState.create(1)


@pytest.fixture
def referee() -> Referee:
    return Referee()


@pytest.fixture
def inputs(state: GameState) -> RuleInputs:
    return RuleInputs.empty(state)
