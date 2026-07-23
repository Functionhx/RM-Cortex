"""Backend protocol shared by the Torch world and Isaac Lab adapter."""

from __future__ import annotations

from typing import Protocol, TypeVar

from rm_referee.inputs import RuleInputs
from rm_referee.random_tape import RandomTape
from rm_referee.state import GameState

ActionT = TypeVar("ActionT", contravariant=True)
WorldStateT = TypeVar("WorldStateT", contravariant=True)


class RuleBackend(Protocol[ActionT, WorldStateT]):
    """Convert backend state/actions into normalized referee inputs."""

    def rule_inputs(
        self,
        world: WorldStateT,
        state: GameState,
        actions: ActionT,
        random_tape: RandomTape,
        dt: float,
    ) -> RuleInputs: ...
