"""Coins, delayed ammunition exchange, and sentry supply rounds."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import PurchaseKind, Role, Team, Weapon, slot, unit_teams
from rm_referee.schema import weapon_capability
from rm_referee.state import GameState


def _rounds_and_cost(
    weapon: torch.Tensor,
    kind: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_rounds = torch.tensor(
        constants.LOCAL_PURCHASE_ROUNDS,
        device=device,
        dtype=torch.long,
    )[weapon]
    remote_rounds = torch.tensor(
        constants.REMOTE_PURCHASE_ROUNDS,
        device=device,
        dtype=torch.long,
    )[weapon]
    rounds = torch.where(kind == PurchaseKind.LOCAL, local_rounds, remote_rounds)
    cost = torch.where(
        kind == PurchaseKind.LOCAL,
        torch.full_like(rounds, constants.LOCAL_PURCHASE_COST),
        torch.full_like(rounds, constants.REMOTE_PURCHASE_COST),
    )
    return rounds, cost


def _grant_ammo(
    state: GameState,
    unit: torch.Tensor,
    weapon: torch.Tensor,
    rounds: torch.Tensor,
) -> None:
    weapon_one_hot = functional.one_hot(
        weapon,
        constants.WEAPON_COUNT,
    ).to(torch.long)
    state.team_purchased_ammo.add_(weapon_one_hot * rounds.unsqueeze(-1))
    unit_one_hot = functional.one_hot(
        unit,
        constants.UNIT_COUNT,
    ).to(torch.long)
    ammo_add = (
        unit_one_hot[:, :, :, None] * weapon_one_hot[:, :, None, :] * rounds[:, :, None, None]
    ).sum(dim=1)
    state.ammo.add_(ammo_add)


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
    state.team_total_coin_earned.add_(scheduled[:, None])
    events.scheduled_coin.copy_(scheduled)

    core_ticks = (
        torch.floor((next_elapsed + 1.0e-6) / 10.0) - torch.floor((state.elapsed_s + 1.0e-6) / 10.0)
    ).to(torch.long)
    core_income = state.tech_income_per_10s * core_ticks[:, None] * active[:, None]
    state.team_coin.add_(core_income)
    state.team_total_coin_earned.add_(core_income)

    minute_crossings = crossed.to(torch.long).sum(dim=1)
    state.sentry_supply_rounds.add_(
        minute_crossings[:, None] * constants.SENTRY_SUPPLY_ROUNDS_PER_MINUTE
    )
    sentry_slots = torch.tensor(
        [slot(Team.RED, Role.SENTRY), slot(Team.BLUE, Role.SENTRY)],
        device=state.device,
    )
    sentry_can_claim = (
        active[:, None]
        & inputs.sentry_claim_ammo
        & inputs.in_supply_zone[:, sentry_slots]
        & state.alive[:, sentry_slots]
    )
    claimed = torch.where(
        sentry_can_claim,
        state.sentry_supply_rounds,
        torch.zeros_like(state.sentry_supply_rounds),
    )
    state.ammo[:, sentry_slots, Weapon.MM17].add_(claimed)
    state.sentry_supply_rounds.sub_(claimed)
    events.sentry_claimed_rounds.copy_(claimed)

    unit = inputs.purchase_unit
    unit_safe = torch.clamp(unit, min=0, max=constants.UNIT_COUNT - 1)
    weapon = inputs.purchase_weapon.to(torch.long)
    kind = inputs.purchase_kind
    team_index = torch.arange(
        constants.TEAM_COUNT,
        device=state.device,
    ).view(1, constants.TEAM_COUNT)
    belongs_to_team = unit_teams(state.device)[unit_safe] == team_index
    capability = weapon_capability(state.device)[unit_safe, weapon]
    rounds, cost = _rounds_and_cost(weapon, kind, device=state.device)
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
    target_out_of_combat = (
        torch.gather(
            state.out_of_combat_s,
            1,
            unit_safe,
        )
        >= constants.OUT_OF_COMBAT_S
    )
    target_in_zone = torch.gather(
        inputs.in_ammo_zone | inputs.in_supply_zone,
        1,
        unit_safe,
    )
    no_pending = state.pending_purchase_kind == PurchaseKind.NONE
    common = (
        active[:, None]
        & no_pending
        & (unit >= 0)
        & belongs_to_team
        & capability
        & (state.team_coin >= cost)
        & (purchased + rounds <= cap)
    )

    direct = common & (
        ((kind == PurchaseKind.LOCAL) & target_in_zone)
        | ((kind == PurchaseKind.REMOTE) & inputs.purchase_ready & target_out_of_combat)
    )
    direct_rounds = torch.where(direct, rounds, torch.zeros_like(rounds))
    direct_cost = torch.where(direct, cost, torch.zeros_like(cost))
    state.team_coin.sub_(direct_cost)
    _grant_ammo(state, unit_safe, weapon, direct_rounds)

    start_pending = (
        common & (kind == PurchaseKind.REMOTE) & ~inputs.purchase_ready & target_out_of_combat
    )
    state.team_coin.sub_(torch.where(start_pending, cost, torch.zeros_like(cost)))
    state.pending_purchase_unit.copy_(
        torch.where(start_pending, unit_safe, state.pending_purchase_unit)
    )
    state.pending_purchase_weapon.copy_(
        torch.where(start_pending, weapon, state.pending_purchase_weapon)
    )
    state.pending_purchase_kind.copy_(
        torch.where(
            start_pending,
            torch.full_like(state.pending_purchase_kind, PurchaseKind.REMOTE),
            state.pending_purchase_kind,
        )
    )
    state.pending_purchase_progress_s.copy_(
        torch.where(
            start_pending,
            torch.zeros_like(state.pending_purchase_progress_s),
            state.pending_purchase_progress_s,
        )
    )

    pending = state.pending_purchase_kind == PurchaseKind.REMOTE
    state.pending_purchase_progress_s.add_(pending.to(state.dtype) * dt)
    complete = pending & (
        state.pending_purchase_progress_s >= constants.REMOTE_TRANSACTION_CONFIRM_S
    )
    pending_weapon = state.pending_purchase_weapon
    pending_unit = torch.clamp(state.pending_purchase_unit, min=0)
    pending_rounds, _ = _rounds_and_cost(
        pending_weapon,
        state.pending_purchase_kind,
        device=state.device,
    )
    granted_pending = torch.where(
        complete,
        pending_rounds,
        torch.zeros_like(pending_rounds),
    )
    _grant_ammo(state, pending_unit, pending_weapon, granted_pending)
    events.purchased_rounds.copy_(direct_rounds + granted_pending)

    state.pending_purchase_unit.copy_(torch.where(complete, -1, state.pending_purchase_unit))
    state.pending_purchase_weapon.copy_(torch.where(complete, 0, state.pending_purchase_weapon))
    state.pending_purchase_kind.copy_(
        torch.where(
            complete,
            torch.full_like(state.pending_purchase_kind, PurchaseKind.NONE),
            state.pending_purchase_kind,
        )
    )
    state.pending_purchase_progress_s.copy_(
        torch.where(complete, 0.0, state.pending_purchase_progress_s)
    )
