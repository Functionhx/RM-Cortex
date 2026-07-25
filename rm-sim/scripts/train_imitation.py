#!/usr/bin/env python3
"""Train five-second RMUC tactical intentions, not 0.2-second control actions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from rm_train.imitation_runner import ImitationConfig, ImitationTrainingRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Behavior-clone five-second navigation intent and future-window firing "
            "occurrence from the 1 Hz RMUC referee export. Labels are auxiliary "
            "tactical proxies, not 0.2-second robot actions or target choices."
        )
    )
    parser.add_argument("--config", default="configs/rmuc_bc.json")
    parser.add_argument("--database", dest="database_path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="number of team-perspective windows per batch (six robot examples each)",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        help="deterministic random cap; zero uses every training window",
    )
    parser.add_argument(
        "--max-validation-windows",
        type=int,
        help="deterministic random cap; zero uses every validation window",
    )
    parser.add_argument("--workers", dest="num_workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--goal-coefficient",
        type=float,
        help="navigation-intent loss coefficient; zero disables this auxiliary task",
    )
    parser.add_argument(
        "--fire-coefficient",
        type=float,
        help="future-window firing-occurrence loss coefficient; zero disables it",
    )
    parser.add_argument(
        "--fire-pos-weight",
        type=float,
        help="fixed positive-class weight fitted on the training split, in [1, 4]",
    )
    parser.add_argument(
        "--goal-scale",
        dest="goal_scale_m",
        type=float,
        help="component-wise displacement cap and normalization scale in meters",
    )
    parser.add_argument("--history", dest="history_steps", type=int)
    parser.add_argument("--horizon", dest="future_horizon_s", type=float)
    parser.add_argument("--stride", dest="stride_s", type=float)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    config = ImitationConfig.from_json(args.config)
    overrides = {
        name: value for name, value in vars(args).items() if name != "config" and value is not None
    }
    config = replace(config, **overrides)
    try:
        config.validate(require_database=True)
    except ValueError as error:
        parser.error(str(error))

    progress, checkpoint = ImitationTrainingRunner(config).train()
    result = {
        "checkpoint": str(checkpoint),
        "epochs": len(progress),
        "label_semantics": (
            f"{config.future_horizon_s:g}-second component-wise capped high-level "
            "tactical subgoal; not a 0.2-second low-level action"
        ),
        "last_epoch": progress[-1],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
