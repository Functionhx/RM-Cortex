from __future__ import annotations

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import Role, Team, Winner, slot


def test_base_armor_deployment_does_not_add_defense(
    state: GameState,
    referee: Referee,
) -> None:
    base = slot(Team.BLUE, Role.BASE)
    state.hp[0, base] = 2000

    next_state, _ = referee.step(state, RuleInputs.empty(state))

    assert next_state.base_armor_deployed[0, Team.BLUE]
    assert next_state.defense_fraction[0, base] == 0


def test_timeout_uses_outpost_hp_when_neither_was_destroyed(
    state: GameState,
    referee: Referee,
) -> None:
    state.elapsed_s.fill_(419.9)
    state.hp[0, slot(Team.RED, Role.OUTPOST)] = 1000
    state.hp[0, slot(Team.BLUE, Role.OUTPOST)] = 900

    next_state, events = referee.step(state, RuleInputs.empty(state))

    assert events.match_ended[0]
    assert next_state.winner[0] == Winner.RED


def test_simultaneous_zero_bases_continue_through_damage_tiebreak(
    state: GameState,
    referee: Referee,
) -> None:
    state.hp[0, slot(Team.RED, Role.BASE)] = 0
    state.hp[0, slot(Team.BLUE, Role.BASE)] = 0
    state.alive[0, slot(Team.RED, Role.BASE)] = False
    state.alive[0, slot(Team.BLUE, Role.BASE)] = False
    state.outpost_destroyed_once.fill_(True)
    state.team_damage[0, Team.RED] = 100
    state.team_damage[0, Team.BLUE] = 80

    next_state, _ = referee.step(state, RuleInputs.empty(state))

    assert next_state.winner[0] == Winner.RED
