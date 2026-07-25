#!/usr/bin/env python3
"""Benchmark complete Torch-world policy steps."""

from __future__ import annotations

import argparse
import time

import torch

from rm_world import ScriptedOpponent, TorchEnvConfig, TorchRMArena


def benchmark(
    num_envs: int,
    iterations: int,
    device: str,
    observation_mode: str,
) -> tuple[float, float]:
    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=num_envs,
            device=device,
            validate_referee=False,
            observation_mode=observation_mode,
        )
    )
    opponent = ScriptedOpponent()
    for _ in range(2):
        environment.step(opponent.act(environment.game, environment.world))
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        environment.step(opponent.act(environment.game, environment.world))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000.0 / iterations
    env_steps_per_second = num_envs * iterations / elapsed
    return milliseconds, env_steps_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("belief", "oracle"),
        default="belief",
    )
    args = parser.parse_args()
    for num_envs in args.num_envs:
        milliseconds, throughput = benchmark(
            num_envs,
            args.iterations,
            args.device,
            args.observation_mode,
        )
        print(
            f"{args.device} envs={num_envs}: "
            f"{milliseconds:.3f} ms/policy-step, {throughput:,.0f} env-steps/s"
        )


if __name__ == "__main__":
    main()
