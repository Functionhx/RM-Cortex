from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import HeatLock, Role, Team, Weapon, slot


def test_q1_equal_q2_is_permanent_lock_without_hp_loss(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    infantry = slot(Team.RED, Role.INFANTRY_3)
    state.ammo[0, infantry, Weapon.MM17] = 1
    state.heat[0, infantry, Weapon.MM17] = 130
    hp_before = state.hp.clone()
    inputs.shots_fired[0, infantry, Weapon.MM17] = 1

    next_state, _ = referee.step(state, inputs)

    assert next_state.heat_lock[0, infantry, Weapon.MM17] == HeatLock.PERMANENT
    assert next_state.hp.tolist() == hp_before.tolist()


def test_temporary_heat_lock_only_releases_at_zero(
    state: GameState,
    referee: Referee,
) -> None:
    infantry = slot(Team.RED, Role.INFANTRY_3)
    state.heat[0, infantry, Weapon.MM17] = 41

    locked, _ = referee.step(state, RuleInputs.empty(state))
    assert locked.heat_lock[0, infantry, Weapon.MM17] == HeatLock.TEMPORARY
    assert locked.video_disabled[0, infantry]

    locked.heat[0, infantry, Weapon.MM17] = 0.5
    unlocked, _ = referee.step(locked, RuleInputs.empty(locked))
    assert unlocked.heat[0, infantry, Weapon.MM17] == pytest.approx(0)
    assert unlocked.heat_lock[0, infantry, Weapon.MM17] == HeatLock.NONE
    assert not unlocked.video_disabled[0, infantry]
