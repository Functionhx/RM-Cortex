#!/usr/bin/env python3
"""Evaluate the hierarchical behavior-tree/A* baseline with paired sides."""

from __future__ import annotations

import argparse
import json

from rm_train.baseline_evaluation import evaluate_world_controllers
from rm_world import (
    ArenaGeometry,
    BehaviorTreeOpponent,
    ScriptedOpponent,
    TacticalScriptedOpponent,
)


def _behavior_factory(arena: ArenaGeometry) -> BehaviorTreeOpponent:
    return BehaviorTreeOpponent(arena=arena)


def _objective_factory(_arena: ArenaGeometry) -> ScriptedOpponent:
    return ScriptedOpponent()


def _tactical_factory(arena: ArenaGeometry) -> TacticalScriptedOpponent:
    return TacticalScriptedOpponent(arena=arena)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponent",
        choices=("objective", "tactical"),
        default="tactical",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(1007,))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        help="explicit smoke-test truncation; omit for the complete 420-second match",
    )
    args = parser.parse_args()

    opponent_factory = _objective_factory if args.opponent == "objective" else _tactical_factory
    report = evaluate_world_controllers(
        _behavior_factory,
        opponent_factory,
        controller_name="behavior-tree-astar",
        opponent_name=args.opponent,
        seeds=args.seeds,
        device=args.device,
        max_policy_steps=args.max_policy_steps,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
