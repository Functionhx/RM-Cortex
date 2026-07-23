#!/usr/bin/env python3
"""Launch the minimal RM-Cortex DirectMARLEnv inside Isaac Lab."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch

        from rm_isaac import RMCortexDirectMARLEnv, RMCortexDirectMARLEnvCfg

        cfg = RMCortexDirectMARLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        env = RMCortexDirectMARLEnv(cfg)
        actions = {
            agent: torch.zeros(
                (args.num_envs, cfg.action_spaces[agent]),
                device=env.device,
            )
            for agent in cfg.possible_agents
        }
        env.reset()
        for _ in range(10):
            env.step(actions)
        print(f"Isaac smoke passed: envs={args.num_envs}, agents={len(cfg.possible_agents)}")
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
