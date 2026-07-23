"""Hero, engineer, aerial, and sentry special mechanisms."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.inputs import RuleInputs
from rm_referee.schema import (
    HeroProfile,
    Role,
    SentryStance,
    Team,
    slot,
)
from rm_referee.state import GameState


def apply_roles(state: GameState, inputs: RuleInputs, dt: float) -> None:
    active = (~state.done)[:, None]
    hero_slots = torch.tensor(
        [slot(Team.RED, Role.HERO), slot(Team.BLUE, Role.HERO)],
        device=state.device,
    )
    sentry_slots = torch.tensor(
        [slot(Team.RED, Role.SENTRY), slot(Team.BLUE, Role.SENTRY)],
        device=state.device,
    )
    aerial_slots = torch.tensor(
        [slot(Team.RED, Role.AERIAL), slot(Team.BLUE, Role.AERIAL)],
        device=state.device,
    )

    state.velocity_lock_s.sub_(dt * active[:, :, None]).clamp_(min=0)
    state.immediate_power_boost_s.sub_(dt * active).clamp_(min=0)
    state.sentry_stance_cooldown_s.sub_(dt * active).clamp_(min=0)
    state.aerial_counter_lock_s.sub_(dt * active).clamp_(min=0)

    hero_alive = state.alive[:, hero_slots]
    remote_profile = state.hero_profile[:, hero_slots] == HeroProfile.REMOTE
    deploy_condition = (
        active
        & inputs.hero_deploy_request
        & inputs.in_hero_deploy_zone[:, hero_slots]
        & hero_alive
        & remote_profile
    )
    state.hero_deploy_progress_s.copy_(
        torch.where(
            deploy_condition & ~state.hero_deployed,
            state.hero_deploy_progress_s + dt,
            torch.where(
                state.hero_deployed & deploy_condition,
                state.hero_deploy_progress_s,
                torch.zeros_like(state.hero_deploy_progress_s),
            ),
        )
    )
    state.hero_deployed.copy_(
        deploy_condition
        & (state.hero_deployed | (state.hero_deploy_progress_s >= constants.HERO_DEPLOY_CONFIRM_S))
    )

    requested_stance = inputs.sentry_stance_request.to(torch.long)
    valid_request = requested_stance >= 0
    stance_change = (
        active
        & valid_request
        & (requested_stance != state.sentry_stance)
        & (state.sentry_stance_cooldown_s <= 0)
        & state.alive[:, sentry_slots]
    )
    operator_cost = stance_change & inputs.sentry_operator_command & state.sentry_automatic
    affordable = ~operator_cost | (state.team_coin >= 50)
    stance_change &= affordable
    state.team_coin.sub_(operator_cost.to(torch.long) * stance_change.to(torch.long) * 50)
    state.sentry_stance.copy_(torch.where(stance_change, requested_stance, state.sentry_stance))
    state.sentry_stance_cooldown_s.copy_(
        torch.where(
            stance_change,
            torch.full_like(
                state.sentry_stance_cooldown_s,
                constants.SENTRY_STANCE_SWITCH_COOLDOWN_S,
            ),
            state.sentry_stance_cooldown_s,
        )
    )
    stance_one_hot = functional.one_hot(
        state.sentry_stance.to(torch.long),
        constants.SENTRY_STANCE_COUNT,
    ).to(state.dtype)
    state.sentry_stance_time_s.add_(
        stance_one_hot * (active & state.alive[:, sentry_slots])[:, :, None].to(state.dtype) * dt
    )
    current_stance_time = torch.gather(
        state.sentry_stance_time_s,
        2,
        state.sentry_stance.to(torch.long).unsqueeze(-1),
    ).squeeze(-1)
    degraded = current_stance_time > constants.SENTRY_STANCE_DEGRADE_S
    offensive = state.sentry_stance == SentryStance.OFFENSIVE
    defensive = state.sentry_stance == SentryStance.DEFENSIVE
    mobile = state.sentry_stance == SentryStance.MOBILE

    sentry_cooling = torch.where(
        offensive,
        torch.where(
            degraded,
            torch.full_like(current_stance_time, 2.0),
            torch.full_like(current_stance_time, 3.0),
        ),
        torch.full_like(current_stance_time, 1.0 / 3.0),
    )
    sentry_power = torch.where(
        offensive | defensive,
        torch.full_like(current_stance_time, 0.5),
        torch.where(
            degraded,
            torch.full_like(current_stance_time, 1.2),
            torch.full_like(current_stance_time, 1.5),
        ),
    )
    sentry_defense = torch.where(
        defensive,
        torch.where(
            degraded,
            torch.full_like(current_stance_time, 0.25),
            torch.full_like(current_stance_time, 0.50),
        ),
        torch.zeros_like(current_stance_time),
    )
    sentry_vulnerability = torch.where(
        offensive | mobile,
        torch.full_like(current_stance_time, 0.25),
        torch.zeros_like(current_stance_time),
    )

    state.role_defense_fraction.zero_()
    state.role_vulnerability_fraction.zero_()
    state.role_cooling_multiplier.fill_(1.0)
    state.role_power_multiplier.fill_(1.0)
    state.role_defense_fraction[:, hero_slots] = (
        state.hero_deployed.to(state.dtype) * constants.HERO_DEPLOY_DEFENSE
    )
    early_engineer_slots = torch.tensor(
        [slot(Team.RED, Role.ENGINEER), slot(Team.BLUE, Role.ENGINEER)],
        device=state.device,
    )
    early_engineer = (
        state.elapsed_s[:, None] < constants.ENGINEER_EARLY_DEFENSE_END_S
    ) & state.alive[:, early_engineer_slots]
    state.role_defense_fraction[:, early_engineer_slots] = (
        early_engineer.to(state.dtype) * constants.ENGINEER_EARLY_DEFENSE
    )
    state.role_defense_fraction[:, sentry_slots] = sentry_defense
    state.role_vulnerability_fraction[:, sentry_slots] = sentry_vulnerability
    state.role_cooling_multiplier[:, sentry_slots] = sentry_cooling
    state.role_power_multiplier[:, sentry_slots] = sentry_power

    thresholds = torch.arange(
        60.0,
        constants.MATCH_DURATION_S + 0.1,
        60.0,
        device=state.device,
        dtype=state.dtype,
    )
    crossed = (
        (state.elapsed_s[:, None] < thresholds)
        & (state.elapsed_s[:, None] + dt >= thresholds)
        & (~state.done)[:, None]
    ).sum(dim=1)
    state.aerial_support_bank_s.add_(
        crossed[:, None].to(state.dtype) * constants.AERIAL_SUPPORT_PER_MINUTE_S
    )

    wants_support = active & inputs.aerial_support_request
    bank_active = wants_support & (state.aerial_support_bank_s > 0)
    paid_active = wants_support & ~bank_active & (state.team_coin > 0)
    support_active = bank_active | paid_active
    state.aerial_support_bank_s.sub_(bank_active.to(state.dtype) * dt).clamp_(min=0)
    state.aerial_coin_accumulator_s.copy_(
        torch.where(
            paid_active,
            state.aerial_coin_accumulator_s + dt,
            torch.where(
                wants_support,
                state.aerial_coin_accumulator_s,
                torch.zeros_like(state.aerial_coin_accumulator_s),
            ),
        )
    )
    coins_due = torch.floor(state.aerial_coin_accumulator_s).to(torch.long)
    coins_paid = torch.minimum(coins_due, state.team_coin)
    state.team_coin.sub_(coins_paid)
    state.aerial_coin_accumulator_s.sub_(coins_paid.to(state.dtype))
    support_active &= ~((coins_due > 0) & (coins_paid < coins_due))
    state.aerial_support_active.copy_(support_active)
    state.aerial_on_pad.copy_(inputs.aerial_on_pad)
    state.alive[:, aerial_slots] = support_active

    target_illuminated = inputs.radar_illuminating.flip(dims=(1,))
    target_illuminated |= inputs.aerial_laser_module_offline
    target_illuminated &= support_active
    state.aerial_counter_streak.copy_(
        torch.where(
            target_illuminated,
            state.aerial_counter_streak + 1,
            torch.zeros_like(state.aerial_counter_streak),
        )
    )
    state.aerial_counter_p.copy_(
        torch.where(
            target_illuminated,
            state.aerial_counter_p + state.aerial_counter_streak.to(state.dtype),
            torch.clamp(state.aerial_counter_p - 0.5 * dt, min=0),
        )
    )
    counter_threshold = torch.where(
        state.aerial_counter_uses == 0,
        torch.full_like(state.aerial_counter_p, constants.AERIAL_COUNTER_FIRST_THRESHOLD),
        torch.full_like(state.aerial_counter_p, constants.AERIAL_COUNTER_LATER_THRESHOLD),
    )
    counter_trigger = (
        active
        & (state.aerial_counter_p >= counter_threshold)
        & (state.aerial_counter_uses < constants.AERIAL_COUNTER_MAX_USES)
    )
    state.aerial_counter_lock_s.copy_(
        torch.where(
            counter_trigger,
            torch.full_like(
                state.aerial_counter_lock_s,
                constants.AERIAL_COUNTER_LOCK_S,
            ),
            state.aerial_counter_lock_s,
        )
    )
    state.aerial_counter_p.copy_(torch.where(counter_trigger, 0.0, state.aerial_counter_p))
    state.aerial_counter_streak.copy_(torch.where(counter_trigger, 0, state.aerial_counter_streak))
    state.aerial_counter_uses.add_(counter_trigger.to(torch.long))

    state.hero_42_invalid_block |= (
        (state.dead_s[:, hero_slots] >= constants.HERO_INVALID_OFFLINE_DELAY_S)
        | (
            state.controller_offline[:, hero_slots]
            & (state.controller_offline_s[:, hero_slots] >= constants.HERO_INVALID_OFFLINE_DELAY_S)
        )
    ) & active

    deployed_velocity = torch.full(
        (state.num_envs, constants.TEAM_COUNT),
        constants.HERO_DEPLOY_MUZZLE_LIMIT_MPS,
        device=state.device,
        dtype=state.dtype,
    )
    normal_velocity = torch.full_like(deployed_velocity, 12.0)
    state.muzzle_velocity_limit_mps[:, hero_slots, 1] = torch.where(
        state.hero_deployed,
        deployed_velocity,
        normal_velocity,
    )
