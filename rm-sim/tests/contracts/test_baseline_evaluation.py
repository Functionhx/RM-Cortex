from __future__ import annotations

import json

from rm_referee.schema import Team, Winner
import rm_train.baseline_evaluation as baseline_evaluation
from rm_train.baseline_evaluation import (
    ControllerLegReport,
    evaluate_world_controllers,
)
from rm_world import (
    BehaviorTreeOpponent,
    GridAStarPlanner,
    ScriptedOpponent,
)


def test_controller_evaluation_runs_paired_sides_and_marks_smoke_unfinished() -> None:
    report = evaluate_world_controllers(
        lambda _arena: ScriptedOpponent(),
        lambda _arena: ScriptedOpponent(),
        controller_name="first",
        opponent_name="second",
        seeds=(3,),
        max_policy_steps=1,
    )

    assert len(report.legs) == 2
    assert {leg.controller_team for leg in report.legs} == {
        int(Team.RED),
        int(Team.BLUE),
    }
    assert all(leg.winner == int(Winner.UNDECIDED) for leg in report.legs)
    assert report.unfinished == 2
    assert report.balanced_score is None
    assert json.loads(json.dumps(report.as_dict()))["balanced_score"] is None
    assert report.oracle_full_action


def test_balanced_score_excludes_unfinished_legs(monkeypatch) -> None:
    def fake_run_leg(
        *_args: object,
        controller_team: Team,
        seed: int,
        **_kwargs: object,
    ) -> ControllerLegReport:
        opponent_team = Team(1 - int(controller_team))
        completed = controller_team == Team.RED
        winner = int(controller_team) if completed else int(Winner.UNDECIDED)
        return ControllerLegReport(
            seed=seed,
            controller_team=int(controller_team),
            opponent_team=int(opponent_team),
            winner=winner,
            completed=completed,
            policy_steps=1,
            elapsed_s=0.2,
            controller_return=0.0,
            opponent_return=0.0,
            controller_diagnostics={},
            opponent_diagnostics={},
        )

    monkeypatch.setattr(baseline_evaluation, "_run_leg", fake_run_leg)

    report = evaluate_world_controllers(
        lambda _arena: ScriptedOpponent(),
        lambda _arena: ScriptedOpponent(),
        controller_name="first",
        opponent_name="second",
        seeds=(3,),
    )

    assert report.controller_wins == 1
    assert report.unfinished == 1
    assert report.balanced_score == 1.0


def test_behavior_tree_evaluation_exposes_navigation_and_leaf_diagnostics() -> None:
    def behavior_factory(arena):
        return BehaviorTreeOpponent(
            arena=arena,
            planner=GridAStarPlanner(
                arena,
                resolution_m=0.2,
                clearance_m=0.02,
            ),
        )

    report = evaluate_world_controllers(
        behavior_factory,
        lambda _arena: ScriptedOpponent(),
        controller_name="behavior-tree-astar",
        opponent_name="objective",
        seeds=(5,),
        controller_teams=(Team.RED,),
        max_policy_steps=2,
    )

    diagnostics = report.legs[0].controller_diagnostics
    assert diagnostics["navigation"]["plan_requests"] == 5
    assert diagnostics["navigation"]["plan_failures"] == 0
    assert diagnostics["selected_leaf_hits"]
