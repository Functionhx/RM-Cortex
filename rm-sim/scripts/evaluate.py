#!/usr/bin/env python3
"""Evaluate a MAPPO checkpoint against the scripted blue opponent."""

from __future__ import annotations

import argparse
import json

import torch

from rm_train import MAPPOConfig, SharedMAPPOPolicy, evaluate_against_scripted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-policy-steps", type=int, default=2100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = MAPPOConfig(**payload["config"])
    device = config.resolved_device if args.device == "auto" else args.device
    schema_version = payload.get("policy_schema_version", 1)
    if schema_version != SharedMAPPOPolicy.CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "checkpoint policy schema is incompatible with the full-entity actor; "
            "retrain it or migrate the checkpoint"
        )
    policy_kwargs = payload.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict):
        raise RuntimeError("checkpoint is missing policy architecture metadata")
    policy = SharedMAPPOPolicy(**policy_kwargs)
    policy.load_state_dict(payload["policy"])
    report = evaluate_against_scripted(
        policy,
        num_envs=args.num_envs,
        max_policy_steps=args.max_policy_steps,
        device=device,
        seed=config.seed + 1000,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
