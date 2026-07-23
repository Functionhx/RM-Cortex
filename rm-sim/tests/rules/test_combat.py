from __future__ import annotations

import pytest

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, Team, Weapon, slot


def _hit(
    inputs: RuleInputs,
    *,
    source: int,
    target: int,
    armor: int,
    weapon: Weapon,
    candidate: int,
    offset: float,
    critical: bool = False,
) -> None:
    inputs.hits.source[0, target, armor, weapon, candidate] = source
    inputs.hits.time_offset_s[0, target, armor, weapon, candidate] = offset
    inputs.hits.critical[0, target, armor, weapon, candidate] = critical


def test_armor_refractory_does_not_truncate_shots_heat_or_ammo(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    source = slot(Team.RED, Role.INFANTRY_3)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    state.ammo[0, source, Weapon.MM17] = 10
    inputs.shots_fired[0, source, Weapon.MM17] = 3
    _hit(
        inputs, source=source, target=target, armor=0, weapon=Weapon.MM17, candidate=0, offset=0.00
    )
    _hit(
        inputs, source=source, target=target, armor=0, weapon=Weapon.MM17, candidate=1, offset=0.04
    )
    _hit(
        inputs, source=source, target=target, armor=0, weapon=Weapon.MM17, candidate=2, offset=0.06
    )

    next_state, events = referee.step(state, inputs)

    assert events.hit_accepted[0, target, 0, Weapon.MM17].tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert next_state.hp[0, target] == pytest.approx(160)
    assert next_state.ammo[0, source, Weapon.MM17] == 7
    assert next_state.heat[0, source, Weapon.MM17] == pytest.approx(28.8)


def test_base_is_invulnerable_while_its_outpost_is_alive(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    source = slot(Team.RED, Role.INFANTRY_3)
    target = slot(Team.BLUE, Role.BASE)
    state.ammo[0, source, Weapon.MM17] = 1
    inputs.shots_fired[0, source, Weapon.MM17] = 1
    _hit(inputs, source=source, target=target, armor=0, weapon=Weapon.MM17, candidate=0, offset=0)

    next_state, events = referee.step(state, inputs)

    assert next_state.hp[0, target] == 5000
    assert next_state.base_shield[0, Team.BLUE] == 150
    assert not events.hit_accepted[0, target, 0, Weapon.MM17, 0]


def test_critical_hit_uses_max_attack_multiplier_and_half_up_rounding(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    source = slot(Team.RED, Role.INFANTRY_3)
    target = slot(Team.BLUE, Role.OUTPOST)
    state.ammo[0, source, Weapon.MM17] = 1
    inputs.shots_fired[0, source, Weapon.MM17] = 1
    _hit(
        inputs,
        source=source,
        target=target,
        armor=0,
        weapon=Weapon.MM17,
        candidate=0,
        offset=0,
        critical=True,
    )

    next_state, _ = referee.step(state, inputs)

    assert next_state.hp[0, target] == pytest.approx(1470)


def test_42mm_overfire_is_not_hp_penalty_and_blocks_damage(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    source = slot(Team.RED, Role.HERO)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    inputs.shots_fired[0, source, Weapon.MM42] = 1
    _hit(inputs, source=source, target=target, armor=0, weapon=Weapon.MM42, candidate=0, offset=0)
    hero_hp = state.hp[0, source].item()

    blocked, events = referee.step(state, inputs)

    assert events.overfire[0, source, Weapon.MM42] == 1
    assert blocked.hp[0, source] == hero_hp
    assert blocked.hp[0, target] == state.hp[0, target]
    assert blocked.hero_42_invalid_block[0, Team.RED]


def test_aerial_is_not_a_normal_damage_target(
    state: GameState,
    referee: Referee,
    inputs: RuleInputs,
) -> None:
    source = slot(Team.RED, Role.INFANTRY_3)
    target = slot(Team.BLUE, Role.AERIAL)
    state.alive[0, target] = True
    state.ammo[0, source, Weapon.MM17] = 1
    inputs.shots_fired[0, source, Weapon.MM17] = 1
    _hit(inputs, source=source, target=target, armor=0, weapon=Weapon.MM17, candidate=0, offset=0)

    next_state, events = referee.step(state, inputs)

    assert next_state.hp[0, target] == 0
    assert not events.hit_accepted[0, target, 0, Weapon.MM17, 0]
