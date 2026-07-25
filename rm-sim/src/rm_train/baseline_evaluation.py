"""Paired-side evaluation for deterministic full-action world controllers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import math
from typing import Any, Protocol

import torch

from rm_referee import constants
from rm_referee.schema import Team, Winner
from rm_referee.state import GameState
from rm_train.actions import replace_team_actions
from rm_world import (
    ArenaGeometry,
    BehaviorTreeOpponent,
    KinematicState,
    TorchEnvConfig,
    TorchRMArena,
    WorldActions,
)


class WorldController(Protocol):
    """Minimal deterministic baseline interface."""

    def act(
        self,
        game: GameState,
        world: KinematicState,
        *,
        team: Team | int | None = None,
    ) -> WorldActions: ...


ControllerFactory = Callable[[ArenaGeometry], WorldController]


@dataclass(frozen=True)
class ControllerLegReport:
    seed: int
    controller_team: int
    opponent_team: int
    winner: int
    completed: bool
    policy_steps: int
    elapsed_s: float
    controller_return: float
    opponent_return: float
    controller_diagnostics: dict[str, Any]
    opponent_diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerEvaluationReport:
    controller_name: str
    opponent_name: str
    oracle_full_action: bool
    controller_wins: int
    opponent_wins: int
    draws: int
    unfinished: int
    mean_controller_return: float
    mean_opponent_return: float
    balanced_score: float | None
    legs: tuple[ControllerLegReport, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _controller_diagnostics(controller: WorldController) -> dict[str, Any]:
    if not isinstance(controller, BehaviorTreeOpponent):
        return {}
    return {
        "navigation": asdict(controller.navigation_diagnostics),
        "selected_leaf_hits": controller.leaf_hit_counts,
    }


def _run_leg(
    controller_factory: ControllerFactory,
    opponent_factory: ControllerFactory,
    *,
    controller_team: Team,
    seed: int,
    device: str,
    max_policy_steps: int | None,
) -> ControllerLegReport:
    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=1,
            device=device,
            seed=seed,
            validate_referee=False,
        )
    )
    full_match_steps = math.ceil(constants.MATCH_DURATION_S / environment.config.policy_dt_s)
    steps_limit = full_match_steps if max_policy_steps is None else max_policy_steps
    if steps_limit <= 0:
        raise ValueError("max_policy_steps must be positive")

    opponent_team = Team(1 - int(controller_team))
    controller = controller_factory(environment.arena)
    opponent = opponent_factory(environment.arena)
    environment.reset(seed=seed)
    returns = torch.zeros(
        constants.TEAM_COUNT,
        device=environment.game.device,
        dtype=environment.game.dtype,
    )
    completed = False
    steps_taken = 0
    for steps_taken in range(1, steps_limit + 1):
        actions = controller.act(
            environment.game,
            environment.world,
            team=controller_team,
        )
        opponent_actions = opponent.act(
            environment.game,
            environment.world,
            team=opponent_team,
        )
        replace_team_actions(actions, opponent_actions, opponent_team)
        actions.validate(environment.game)
        result = environment.step(actions)
        per_team = result.reward.view(
            constants.TEAM_COUNT,
            constants.ROLES_PER_TEAM,
        ).mean(dim=-1)
        returns += per_team
        if bool(result.terminated[0].item()):
            completed = True
            break

    winner = int(environment.game.winner[0].item()) if completed else int(Winner.UNDECIDED)
    return ControllerLegReport(
        seed=seed,
        controller_team=int(controller_team),
        opponent_team=int(opponent_team),
        winner=winner,
        completed=completed,
        policy_steps=steps_taken,
        elapsed_s=float(environment.game.elapsed_s[0].item()),
        controller_return=float(returns[controller_team].item()),
        opponent_return=float(returns[opponent_team].item()),
        controller_diagnostics=_controller_diagnostics(controller),
        opponent_diagnostics=_controller_diagnostics(opponent),
    )


@torch.no_grad()
def evaluate_world_controllers(
    controller_factory: ControllerFactory,
    opponent_factory: ControllerFactory,
    *,
    controller_name: str,
    opponent_name: str,
    seeds: Iterable[int] = (1007,),
    controller_teams: Iterable[Team | int] = (Team.RED, Team.BLUE),
    device: str = "cpu",
    max_policy_steps: int | None = None,
) -> ControllerEvaluationReport:
    """Evaluate deterministic controllers on paired RED/BLUE legs.

    Both controllers consume complete world state and the full ``WorldActions``
    schema. The resulting score is an oracle engineering comparison, not a
    belief-policy algorithm comparison.
    """

    seed_values = tuple(int(seed) for seed in seeds)
    team_values = tuple(Team(int(team)) for team in controller_teams)
    if not seed_values:
        raise ValueError("seeds must not be empty")
    if not team_values:
        raise ValueError("controller_teams must not be empty")
    if len(set(team_values)) != len(team_values):
        raise ValueError("controller_teams must be unique")

    legs = tuple(
        _run_leg(
            controller_factory,
            opponent_factory,
            controller_team=team,
            seed=seed,
            device=device,
            max_policy_steps=max_policy_steps,
        )
        for seed in seed_values
        for team in team_values
    )
    controller_wins = sum(leg.completed and leg.winner == leg.controller_team for leg in legs)
    opponent_wins = sum(leg.completed and leg.winner == leg.opponent_team for leg in legs)
    draws = sum(leg.completed and leg.winner == int(Winner.DRAW) for leg in legs)
    unfinished = sum(not leg.completed for leg in legs)
    completed_legs = len(legs) - unfinished
    return ControllerEvaluationReport(
        controller_name=controller_name,
        opponent_name=opponent_name,
        oracle_full_action=True,
        controller_wins=controller_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        unfinished=unfinished,
        mean_controller_return=sum(leg.controller_return for leg in legs) / len(legs),
        mean_opponent_return=sum(leg.opponent_return for leg in legs) / len(legs),
        balanced_score=(
            (controller_wins + 0.5 * draws) / completed_legs if completed_legs > 0 else None
        ),
        legs=legs,
    )


__all__ = [
    "ControllerEvaluationReport",
    "ControllerFactory",
    "ControllerLegReport",
    "WorldController",
    "evaluate_world_controllers",
]
