from __future__ import annotations

from dataclasses import fields
import sys

import pytest
import torch

from rm_referee import GameState, RandomTape, Referee, RuleInputs
from rm_referee.schema import DartTarget, Role, Team, slot
from rm_world.backend import mask_intermediate_policy_pulses


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


def test_intermediate_tick_masks_only_policy_rate_pulses() -> None:
    state = GameState.create(1)
    inputs = RuleInputs.empty(state)
    radar_target = slot(Team.BLUE, Role.HERO)
    inputs.radar_update[0, Team.RED, radar_target] = True
    inputs.radar_report_xy[0, Team.RED, radar_target] = torch.tensor((3.0, -2.0))
    inputs.radar_key_solved[0, Team.RED] = True
    inputs.dart_fire_target[0, Team.RED] = DartTarget.OUTPOST
    inputs.dart_open_gate[0, Team.RED] = True
    inputs.dart_close_gate[0, Team.BLUE] = True
    inputs.radar_illuminating[0, Team.RED] = True
    inputs.radar_double_request[0, Team.BLUE] = True

    mask_intermediate_policy_pulses(inputs)

    assert not inputs.radar_update.any()
    assert not inputs.radar_report_xy.any()
    assert not inputs.radar_key_solved.any()
    assert torch.equal(inputs.dart_fire_target, torch.full((1, 2), -1))
    assert inputs.dart_open_gate[0, Team.RED]
    assert inputs.dart_close_gate[0, Team.BLUE]
    assert inputs.radar_illuminating[0, Team.RED]
    assert inputs.radar_double_request[0, Team.BLUE]


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
