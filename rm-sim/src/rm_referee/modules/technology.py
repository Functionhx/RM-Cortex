"""Technology-core rewards and permanent team upgrades."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, Team, slot
from rm_referee.state import GameState


def apply_technology(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
) -> None:
    requested = inputs.tech_complete_level.to(torch.long)
    requested_index = torch.clamp(requested - 1, min=0)
    existing = torch.gather(
        state.tech_completion_count,
        2,
        requested_index.unsqueeze(-1),
    ).squeeze(-1)
    opened_at = torch.tensor(
        constants.TECH_OPEN_S,
        device=state.device,
        dtype=state.dtype,
    )[requested_index]
    previous_index = torch.clamp(requested_index - 1, min=0)
    previous_count = torch.gather(
        state.tech_completion_count,
        2,
        previous_index.unsqueeze(-1),
    ).squeeze(-1)
    sequence_ok = (requested == 1) | (previous_count > 0)
    level4_once = (requested != 4) | (existing == 0)
    energy_cost = torch.tensor(
        constants.TECH_ENERGY_UNIT_COST,
        device=state.device,
        dtype=torch.long,
    )[requested_index]
    valid = (
        (~state.done)[:, None]
        & (requested > 0)
        & (state.elapsed_s[:, None] >= opened_at)
        & sequence_ok
        & level4_once
    )

    # A shared resource shortage is resolved deterministically in team order.
    red_valid = valid[:, Team.RED] & (energy_cost[:, Team.RED] <= state.tech_energy_units_remaining)
    remaining = state.tech_energy_units_remaining - torch.where(
        red_valid,
        energy_cost[:, Team.RED],
        torch.zeros_like(energy_cost[:, Team.RED]),
    )
    blue_valid = valid[:, Team.BLUE] & (energy_cost[:, Team.BLUE] <= remaining)
    valid = torch.stack((red_valid, blue_valid), dim=1)
    consumed = torch.where(valid, energy_cost, torch.zeros_like(energy_cost)).sum(dim=1)
    state.tech_energy_units_remaining.sub_(consumed)

    first = valid & (existing == 0)
    repeat = valid & ~first
    first_income = torch.tensor(
        constants.TECH_FIRST_INCOME_PER_10S,
        device=state.device,
        dtype=torch.long,
    )[requested_index]
    repeat_income = torch.tensor(
        constants.TECH_REPEAT_INCOME_PER_10S,
        device=state.device,
        dtype=torch.long,
    )[requested_index]
    state.tech_income_per_10s.add_(
        torch.where(
            first,
            first_income,
            torch.where(repeat, repeat_income, torch.zeros_like(first_income)),
        )
    )
    completion_one_hot = functional.one_hot(
        requested_index,
        constants.TECH_LEVEL_COUNT,
    ).to(torch.long)
    state.tech_completion_count.add_(completion_one_hot * valid[:, :, None])
    events.tech_completed.copy_(torch.where(valid, requested, torch.zeros_like(requested)))

    cap = torch.tensor(
        constants.TECH_LEVEL_CAP,
        device=state.device,
        dtype=torch.long,
    )[requested_index]
    state.team_level_cap.copy_(
        torch.where(first, torch.maximum(state.team_level_cap, cap), state.team_level_cap)
    )
    defense = torch.tensor(
        constants.TECH_DEFENSE,
        device=state.device,
        dtype=state.dtype,
    )[requested_index]
    state.tech_defense_fraction.copy_(
        torch.where(
            first,
            torch.maximum(state.tech_defense_fraction, defense),
            state.tech_defense_fraction,
        )
    )

    level4 = first & (requested == 4)
    base_slots = torch.tensor(
        [slot(Team.RED, Role.BASE), slot(Team.BLUE, Role.BASE)],
        device=state.device,
    )
    base_hp = state.hp[:, base_slots]
    base_max = state.max_hp[:, base_slots]
    healed = base_hp + level4.to(state.dtype) * constants.TECH_LEVEL4_BASE_HP_BONUS
    overflow = torch.clamp(healed - base_max, min=0)
    state.hp[:, base_slots] = torch.minimum(healed, base_max)
    state.base_shield.add_(overflow)
