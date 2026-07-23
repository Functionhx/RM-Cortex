from __future__ import annotations

from dataclasses import fields
import sys

import pytest
import torch

from rm_referee import GameState, RandomTape, Referee, RuleInputs


def test_referee_is_deterministic_for_identical_normalized_inputs() -> None:
    state = GameState.create(4)
    inputs = RuleInputs.empty(state)
    left, _ = Referee().step(state, inputs)
    right, _ = Referee().step(state, inputs)

    for field in fields(GameState):
        assert torch.equal(getattr(left, field.name), getattr(right, field.name))


def test_random_tape_can_be_shared_exactly_between_backends() -> None:
    tape = RandomTape.from_seed(3, 8, seed=7)
    copy = tape.clone()

    assert torch.equal(tape.take(4), copy.take(4))
    assert torch.equal(tape.take(4), copy.take(4))


def test_importing_referee_does_not_import_isaac_or_omniverse() -> None:
    forbidden = [
        name
        for name in sys.modules
        if name == "isaaclab" or name.startswith("isaaclab.") or name.startswith("omni.")
    ]
    assert forbidden == []


def test_normalized_hits_cannot_exceed_detected_source_shots() -> None:
    state = GameState.create(1)
    inputs = RuleInputs.empty(state)
    inputs.hits.source[0, 10, 0, 0, 0] = 2

    with pytest.raises(ValueError, match="exceed detected shots"):
        Referee().step(state, inputs)
