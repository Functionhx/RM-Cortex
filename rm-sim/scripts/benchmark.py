#!/usr/bin/env python3
"""Small referee throughput sanity benchmark."""

from __future__ import annotations

import argparse
import time

import torch

from rm_referee import GameState, Referee, RuleInputs


def benchmark(num_envs: int, iterations: int, device: str) -> tuple[float, float]:
    state = GameState.create(num_envs, device=device)
    referee = Referee(validate_inputs=False, validate_state=False)
    inputs = RuleInputs.empty(state)

    for _ in range(5):
        state, _ = referee.step(state, inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        state, _ = referee.step(state, inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000 / iterations
    env_steps_per_second = num_envs * iterations / elapsed
    return milliseconds, env_steps_per_second


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    for num_envs in args.num_envs:
        milliseconds, throughput = benchmark(
            num_envs,
            args.iterations,
            args.device,
        )
        print(
            f"{args.device} envs={num_envs}: "
            f"{milliseconds:.3f} ms/tick, {throughput:,.0f} env-ticks/s"
        )


if __name__ == "__main__":
    main()
