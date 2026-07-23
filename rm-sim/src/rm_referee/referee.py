"""Fixed-order composition of the vectorized rule modules."""

from __future__ import annotations

from dataclasses import dataclass

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.inputs import RuleInputs
from rm_referee.modules.buffs import apply_field_buffs, compose_modifiers
from rm_referee.modules.combat import apply_combat
from rm_referee.modules.compliance import apply_compliance
from rm_referee.modules.dart import apply_dart
from rm_referee.modules.economy import apply_economy
from rm_referee.modules.heat import apply_heat
from rm_referee.modules.objective import apply_objective
from rm_referee.modules.power import apply_power
from rm_referee.modules.radar import apply_radar
from rm_referee.modules.roles import apply_roles
from rm_referee.modules.rune import apply_rune
from rm_referee.modules.survive import apply_survive
from rm_referee.modules.technology import apply_technology
from rm_referee.modules.upgrade import apply_upgrade
from rm_referee.state import GameState


@dataclass(frozen=True)
class Referee:
    """Advance authoritative state by one official 100 ms tick."""

    validate_inputs: bool = True
    validate_state: bool = True

    def step(
        self,
        state: GameState,
        inputs: RuleInputs,
        *,
        dt: float = constants.REFEREE_DT_S,
    ) -> tuple[GameState, RefereeEvents]:
        if abs(dt - constants.REFEREE_DT_S) > 1.0e-9:
            raise ValueError("Phase 1 referee dt is fixed at 0.1 seconds")
        if inputs.shots_fired.shape[0] != state.num_envs:
            raise ValueError("input and state batch sizes differ")
        if self.validate_inputs:
            inputs.validate(state, dt)
        if self.validate_state:
            state.validate()

        next_state = state.clone()
        events = RefereeEvents.empty(next_state)
        active_at_start = ~next_state.done

        apply_roles(next_state, inputs, dt)
        apply_field_buffs(next_state, inputs, events, dt)
        compose_modifiers(next_state)
        apply_combat(next_state, inputs, events, dt)
        apply_compliance(next_state, inputs, events, dt)
        apply_dart(next_state, inputs, events, dt)
        apply_heat(next_state, dt)
        apply_power(next_state, inputs, dt)
        apply_economy(next_state, inputs, events, dt)
        apply_technology(next_state, inputs, events)
        apply_rune(next_state, inputs, events, dt)
        apply_survive(next_state, inputs, events, dt)
        apply_upgrade(next_state, events)
        apply_radar(next_state, inputs, dt)

        next_state.elapsed_s.add_(dt * active_at_start)
        next_state.elapsed_s.clamp_(max=constants.MATCH_DURATION_S)
        apply_objective(next_state, inputs, events, dt)
        compose_modifiers(next_state)

        if self.validate_state:
            next_state.validate()
        return next_state, events
