from __future__ import annotations

import torch

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, Team, Weapon, slot


def test_state_invariants_hold_for_batched_empty_rollout() -> None:
    state = GameState.create(128)
    referee = Referee()

    for _ in range(32):
        inputs = RuleInputs.empty(state)
        inputs.chassis_power_w.uniform_(0, 220)
        state, _ = referee.step(state, inputs)

    state.validate()
    assert torch.all(state.hp >= 0)
    assert torch.all(state.hp <= state.max_hp)
    assert torch.all(state.power_buffer_j >= 0)
    assert torch.all(state.power_buffer_j <= 60)


def test_ammo_and_heat_follow_all_detected_shots_even_without_hits() -> None:
    state = GameState.create(1)
    shooter = slot(Team.RED, Role.SENTRY)
    state.ammo[0, shooter, Weapon.MM17] = 300
    inputs = RuleInputs.empty(state)
    inputs.shots_fired[0, shooter, Weapon.MM17] = 3

    next_state, events = Referee().step(state, inputs)

    assert events.shots_fired[0, shooter, Weapon.MM17] == 3
    assert next_state.ammo[0, shooter, Weapon.MM17] == 297
    assert next_state.heat[0, shooter, Weapon.MM17] > 0
