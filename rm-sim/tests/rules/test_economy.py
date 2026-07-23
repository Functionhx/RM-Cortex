from __future__ import annotations

from rm_referee import GameState, Referee, RuleInputs
from rm_referee.schema import PurchaseKind, Role, Team, Weapon, slot


def _remote_42_purchase(state: GameState) -> RuleInputs:
    inputs = RuleInputs.empty(state)
    inputs.purchase_unit[0, Team.RED] = slot(Team.RED, Role.HERO)
    inputs.purchase_weapon[0, Team.RED] = Weapon.MM42
    inputs.purchase_kind[0, Team.RED] = PurchaseKind.REMOTE
    inputs.purchase_ready[0, Team.RED] = True
    return inputs


def test_42mm_purchase_cap_is_100_per_team_and_ammo_is_per_robot(
    state: GameState,
    referee: Referee,
) -> None:
    state.team_coin.fill_(10_000)
    hero = slot(Team.RED, Role.HERO)

    current = state
    for _ in range(10):
        current, _ = referee.step(current, _remote_42_purchase(current))

    assert current.team_purchased_ammo[0, Team.RED, Weapon.MM42] == 100
    assert current.ammo[0, hero, Weapon.MM42] == 100
    assert current.ammo[0, slot(Team.BLUE, Role.HERO), Weapon.MM42] == 0

    rejected, events = referee.step(current, _remote_42_purchase(current))
    assert events.purchased_rounds[0, Team.RED] == 0
    assert rejected.ammo[0, hero, Weapon.MM42] == 100


def test_scheduled_coin_is_awarded_once_when_crossing_boundary(
    state: GameState,
    referee: Referee,
) -> None:
    state.elapsed_s.fill_(59.9)
    before = state.team_coin.clone()

    crossed, events = referee.step(state, RuleInputs.empty(state))
    after, second_events = referee.step(crossed, RuleInputs.empty(crossed))

    assert events.scheduled_coin.item() == 50
    assert (crossed.team_coin == before + 50).all()
    assert second_events.scheduled_coin.item() == 0
    assert (after.team_coin == crossed.team_coin).all()
