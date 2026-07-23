#!/usr/bin/env python3
"""Open the Isaac Lab viewport with a moving RM-Cortex demonstration."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many policy steps; 0 runs until the window closes.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch

        from rm_isaac import RMCortexDirectMARLEnv, RMCortexDirectMARLEnvCfg

        cfg = RMCortexDirectMARLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = RMCortexDirectMARLEnv(cfg)
        actions = {
            agent: torch.zeros(
                (args.num_envs, cfg.action_spaces[agent]),
                device=env.device,
            )
            for agent in cfg.possible_agents
        }
        robot_agents = [
            agent
            for agent in cfg.possible_agents
            if not agent.endswith("_dart") and not agent.endswith("_radar")
        ]
        for agent in robot_agents:
            actions[agent][:, 0] = 0.65
            actions[agent][:, 4] = 0.50
            actions[agent][:, 5] = 1.0
        actions["red_aerial"][:, 6] = 1.0
        actions["blue_aerial"][:, 6] = 1.0
        actions["red_radar"][:, 3] = 1.0
        actions["blue_radar"][:, 3] = 1.0

        env.reset()
        step = 0
        while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
            turn = 0.18 * math.sin(step * 0.045)
            altitude = 0.30 * math.sin(step * 0.035)
            for agent in robot_agents:
                actions[agent][:, 2] = turn
            actions["red_aerial"][:, 3] = altitude
            actions["blue_aerial"][:, 3] = -altitude
            env.step(actions)
            step += 1
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
