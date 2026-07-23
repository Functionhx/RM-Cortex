"""Scheduled coins and one normalized ammunition transaction per team/tick."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import PurchaseKind, unit_teams, weapon_capability
from rm_referee.state import GameState


def apply_economy(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active = ~state.done
    next_elapsed = state.elapsed_s + dt
    thresholds = torch.tensor(
        constants.COIN_SCHEDULE_S,
        device=state.device,
        dtype=state.dtype,
    )
    amounts = torch.tensor(
        constants.COIN_SCHEDULE_AMOUNT,
        device=state.device,
        dtype=torch.long,
    )
    crossed = (
        (state.elapsed_s[:, None] < thresholds)
        & (next_elapsed[:, None] >= thresholds)
        & active[:, None]
    )
    scheduled = (crossed.to(torch.long) * amounts).sum(dim=1)
    state.team_coin.add_(scheduled[:, None])
    events.scheduled_coin.copy_(scheduled)

    unit = inputs.purchase_unit
    unit_safe = torch.clamp(unit, min=0, max=constants.UNIT_COUNT - 1)
    weapon = torch.clamp(
        inputs.purchase_weapon,
        min=0,
        max=constants.WEAPON_COUNT - 1,
    )
    kind = inputs.purchase_kind

    team_index = torch.arange(
        constants.TEAM_COUNT,
        device=state.device,
    ).view(1, constants.TEAM_COUNT)
    belongs_to_team = unit_teams(state.device)[unit_safe] == team_index
    capability = weapon_capability(state.device)[unit_safe, weapon]

    local_rounds = torch.tensor(
        constants.LOCAL_PURCHASE_ROUNDS,
        device=state.device,
        dtype=torch.long,
    )[weapon]
    remote_rounds = torch.tensor(
        constants.REMOTE_PURCHASE_ROUNDS,
        device=state.device,
        dtype=torch.long,
    )[weapon]
    rounds = torch.where(kind == PurchaseKind.LOCAL, local_rounds, remote_rounds)
    cost = torch.where(
        kind == PurchaseKind.LOCAL,
        torch.full_like(rounds, constants.LOCAL_PURCHASE_COST),
        torch.full_like(rounds, constants.REMOTE_PURCHASE_COST),
    )
    request_ready = (kind == PurchaseKind.LOCAL) | (
        (kind == PurchaseKind.REMOTE) & inputs.purchase_ready
    )

    purchased = torch.gather(
        state.team_purchased_ammo,
        2,
        weapon.unsqueeze(-1),
    ).squeeze(-1)
    cap = torch.tensor(
        constants.PURCHASE_CAP_PER_TEAM,
        device=state.device,
        dtype=torch.long,
    )[weapon]
    grant = (
        active[:, None]
        & (kind != PurchaseKind.NONE)
        & request_ready
        & (unit >= 0)
        & belongs_to_team
        & capability
        & (state.team_coin >= cost)
        & (purchased + rounds <= cap)
    )
    granted_rounds = torch.where(grant, rounds, torch.zeros_like(rounds))
    granted_cost = torch.where(grant, cost, torch.zeros_like(cost))

    state.team_coin.sub_(granted_cost)
    weapon_one_hot = functional.one_hot(
        weapon,
        constants.WEAPON_COUNT,
    ).to(torch.long)
    state.team_purchased_ammo.add_(weapon_one_hot * granted_rounds.unsqueeze(-1))

    unit_one_hot = functional.one_hot(
        unit_safe,
        constants.UNIT_COUNT,
    ).to(torch.long)
    ammo_add = (
        unit_one_hot[:, :, :, None]
        * weapon_one_hot[:, :, None, :]
        * granted_rounds[:, :, None, None]
    ).sum(dim=1)
    state.ammo.add_(ammo_add)
    events.purchased_rounds.copy_(granted_rounds)
