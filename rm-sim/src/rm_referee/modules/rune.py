"""Small and large energy-mechanism activation state machines."""

from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.schema import Role, RuneMode, unit_roles
from rm_referee.state import GameState


def _crossed_schedule(
    state: GameState,
    schedule: tuple[float, ...],
    dt: float,
) -> torch.Tensor:
    thresholds = torch.tensor(schedule, device=state.device, dtype=state.dtype)
    return (
        (state.elapsed_s[:, None] < thresholds)
        & (state.elapsed_s[:, None] + dt >= thresholds)
        & (~state.done)[:, None]
    ).sum(dim=1)


def apply_rune(
    state: GameState,
    inputs: RuleInputs,
    events: RefereeEvents,
    dt: float,
) -> None:
    active_match = (~state.done)[:, None]
    state.rune_small_opportunities.add_(
        _crossed_schedule(state, constants.RUNE_SMALL_SCHEDULE_S[1:], dt)[:, None]
    )
    state.rune_large_opportunities.add_(
        _crossed_schedule(state, constants.RUNE_LARGE_SCHEDULE_S, dt)[:, None]
    )

    state.rune_buff_s.sub_(dt * active_match).clamp_(min=0)
    state.rune_small_buff_s.sub_(dt * active_match).clamp_(min=0)
    buff_expired = state.rune_buff_s <= 0
    state.rune_attack_multiplier.copy_(torch.where(buff_expired, 1.0, state.rune_attack_multiplier))
    state.rune_defense_fraction.copy_(torch.where(buff_expired, 0.0, state.rune_defense_fraction))
    state.rune_cooling_multiplier.copy_(
        torch.where(buff_expired, 1.0, state.rune_cooling_multiplier)
    )

    request = inputs.rune_trigger
    can_trigger = active_match & (state.rune_mode == RuneMode.IDLE) & (state.rune_buff_s <= 0)
    start_small = (
        can_trigger
        & (request == RuneMode.SMALL)
        & (state.elapsed_s[:, None] < 180.0)
        & (state.rune_small_opportunities > 0)
    )
    start_large = (
        can_trigger
        & (request == RuneMode.LARGE)
        & (state.elapsed_s[:, None] >= 180.0)
        & (state.rune_large_opportunities > 0)
    )
    started = start_small | start_large
    state.rune_small_opportunities.sub_(start_small.to(torch.long))
    state.rune_large_opportunities.sub_(start_large.to(torch.long))
    state.rune_mode.copy_(
        torch.where(
            start_small,
            torch.full_like(state.rune_mode, RuneMode.SMALL),
            torch.where(
                start_large,
                torch.full_like(state.rune_mode, RuneMode.LARGE),
                state.rune_mode,
            ),
        )
    )
    state.rune_trigger_remaining_s.copy_(
        torch.where(
            started,
            torch.full_like(
                state.rune_trigger_remaining_s,
                constants.RUNE_TRIGGER_WINDOW_S,
            ),
            torch.clamp(
                state.rune_trigger_remaining_s - dt * active_match,
                min=0,
            ),
        )
    )
    state.rune_group_remaining_s.copy_(
        torch.where(
            started,
            torch.full_like(
                state.rune_group_remaining_s,
                constants.RUNE_GROUP_WINDOW_S,
            ),
            torch.clamp(
                state.rune_group_remaining_s - dt * active_match,
                min=0,
            ),
        )
    )
    state.rune_groups.copy_(torch.where(started, 0, state.rune_groups))
    state.rune_arms.copy_(torch.where(started, 0, state.rune_arms))
    state.rune_ring_sum.copy_(torch.where(started, 0.0, state.rune_ring_sum))
    state.rune_ring_count.copy_(torch.where(started, 0, state.rune_ring_count))

    activating = state.rune_mode != RuneMode.IDLE
    submitted = activating & (inputs.rune_group_hits > 0)
    small_hit_count_ok = (state.rune_mode != RuneMode.SMALL) | (inputs.rune_group_hits == 1)
    group_success = submitted & inputs.rune_group_correct & small_hit_count_ok
    group_failure = submitted & ~group_success
    timed_out_group = (
        activating & (state.rune_group_remaining_s <= 0) & (state.rune_trigger_remaining_s > 0)
    )
    reset_progress = group_failure | timed_out_group
    state.rune_groups.copy_(torch.where(reset_progress, 0, state.rune_groups))
    state.rune_arms.copy_(torch.where(reset_progress, 0, state.rune_arms))
    state.rune_ring_sum.copy_(torch.where(reset_progress, 0.0, state.rune_ring_sum))
    state.rune_ring_count.copy_(torch.where(reset_progress, 0, state.rune_ring_count))
    state.rune_group_remaining_s.copy_(
        torch.where(
            reset_progress | group_success,
            torch.full_like(
                state.rune_group_remaining_s,
                constants.RUNE_GROUP_WINDOW_S,
            ),
            state.rune_group_remaining_s,
        )
    )

    state.rune_groups.add_(group_success.to(torch.long))
    state.rune_arms.add_(
        torch.where(
            group_success,
            inputs.rune_group_hits,
            torch.zeros_like(inputs.rune_group_hits),
        )
    )
    state.rune_ring_sum.add_(
        torch.where(
            group_success,
            inputs.rune_group_ring_sum,
            torch.zeros_like(inputs.rune_group_ring_sum),
        )
    )
    state.rune_ring_count.add_(
        torch.where(
            group_success,
            inputs.rune_group_hits,
            torch.zeros_like(inputs.rune_group_hits),
        )
    )

    activated = activating & (state.rune_groups >= constants.RUNE_REQUIRED_GROUPS)
    small_activated = activated & (state.rune_mode == RuneMode.SMALL)
    large_activated = activated & (state.rune_mode == RuneMode.LARGE)
    average_ring = state.rune_ring_sum / torch.clamp(
        state.rune_ring_count.to(state.dtype),
        min=1,
    )
    large_attack = torch.where(
        average_ring <= 7,
        torch.full_like(average_ring, 1.5),
        torch.where(
            average_ring <= 9,
            torch.full_like(average_ring, 2.0),
            torch.full_like(average_ring, 3.0),
        ),
    )
    large_defense = torch.where(
        average_ring <= 9,
        torch.full_like(average_ring, 0.25),
        torch.full_like(average_ring, 0.50),
    )
    large_cooling = torch.where(
        average_ring <= 3,
        torch.ones_like(average_ring),
        torch.where(
            average_ring <= 8,
            torch.full_like(average_ring, 2.0),
            torch.where(
                average_ring <= 9,
                torch.full_like(average_ring, 3.0),
                torch.full_like(average_ring, 5.0),
            ),
        ),
    )
    duration_table = torch.tensor(
        constants.RUNE_LARGE_DURATION_S,
        device=state.device,
        dtype=state.dtype,
    )
    duration_index = torch.clamp(state.rune_arms, min=5, max=10) - 5
    large_duration = duration_table[duration_index]
    state.rune_buff_s.copy_(
        torch.where(
            small_activated,
            torch.full_like(state.rune_buff_s, constants.RUNE_SMALL_BUFF_S),
            torch.where(large_activated, large_duration, state.rune_buff_s),
        )
    )
    state.rune_small_buff_s.copy_(
        torch.where(
            small_activated,
            torch.full_like(
                state.rune_small_buff_s,
                constants.RUNE_SMALL_BUFF_S,
            ),
            state.rune_small_buff_s,
        )
    )
    state.rune_small_bonus_xp.copy_(torch.where(small_activated, 0.0, state.rune_small_bonus_xp))
    state.rune_attack_multiplier.copy_(
        torch.where(
            small_activated,
            1.0,
            torch.where(large_activated, large_attack, state.rune_attack_multiplier),
        )
    )
    state.rune_defense_fraction.copy_(
        torch.where(
            small_activated,
            torch.full_like(
                state.rune_defense_fraction,
                constants.RUNE_SMALL_DEFENSE,
            ),
            torch.where(
                large_activated,
                large_defense,
                state.rune_defense_fraction,
            ),
        )
    )
    state.rune_cooling_multiplier.copy_(
        torch.where(
            small_activated,
            1.0,
            torch.where(
                large_activated,
                large_cooling,
                state.rune_cooling_multiplier,
            ),
        )
    )

    roles = unit_roles(state.device)
    eligible_role = (
        (roles == Role.HERO)
        | (roles == Role.INFANTRY_3)
        | (roles == Role.INFANTRY_4)
        | (roles == Role.AERIAL)
    )
    alive_eligible = (state.alive & eligible_role[None, :]).view(
        state.num_envs, constants.TEAM_COUNT, constants.ROLES_PER_TEAM
    )
    eligible_count = alive_eligible.sum(dim=-1)
    shared_xp = constants.RUNE_LARGE_SHARED_XP / torch.clamp(
        eligible_count.to(state.dtype),
        min=1,
    )
    events.xp_gained.add_(
        (
            alive_eligible.to(state.dtype)
            * shared_xp[:, :, None]
            * large_activated[:, :, None].to(state.dtype)
        ).reshape_as(events.xp_gained)
    )
    events.rune_activated.copy_(activated)

    trigger_expired = activating & (state.rune_trigger_remaining_s <= 0) & ~activated
    finished = activated | trigger_expired
    state.rune_mode.copy_(
        torch.where(
            finished,
            torch.full_like(state.rune_mode, RuneMode.IDLE),
            state.rune_mode,
        )
    )
    state.rune_trigger_remaining_s.copy_(torch.where(finished, 0.0, state.rune_trigger_remaining_s))
    state.rune_group_remaining_s.copy_(torch.where(finished, 0.0, state.rune_group_remaining_s))
    state.rune_groups.copy_(torch.where(finished, 0, state.rune_groups))
    state.rune_arms.copy_(torch.where(finished, 0, state.rune_arms))
    state.rune_ring_sum.copy_(torch.where(finished, 0.0, state.rune_ring_sum))
    state.rune_ring_count.copy_(torch.where(finished, 0, state.rune_ring_count))
