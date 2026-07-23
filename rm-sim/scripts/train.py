#!/usr/bin/env python3
"""Train the Phase 1 parameter-shared MAPPO baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from rm_train import MAPPOConfig, MAPPOTrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_mappo.json")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = MAPPOConfig.from_json(args.config)
    overrides = {
        "total_updates": args.updates,
        "num_envs": args.num_envs,
        "device": args.device,
        "output_dir": args.output_dir,
    }
    config = replace(
        config,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    config.validate()
    progress, checkpoint = MAPPOTrainingRunner(config).train()
    result = {
        "checkpoint": str(checkpoint),
        "last_update": progress[-1].as_dict(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
