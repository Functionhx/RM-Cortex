#!/usr/bin/env python3
"""Open the Isaac Lab viewport with a moving RM-Cortex demonstration."""

from __future__ import annotations

import argparse

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
        from rm_world import MOBILE_UNIT_SLOTS, terrain_demo_targets

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
        mobile_slots = torch.tensor(MOBILE_UNIT_SLOTS, device=env.device)
        step = 0
        while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
            target_xy = terrain_demo_targets(
                float(env.game.elapsed_s[0]),
                device=env.device,
                dtype=env.game.dtype,
            )
            position = env.world.position_xy[:, mobile_slots]
            delta = target_xy[None, ...] - position
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            desired_world = delta / torch.clamp(distance, min=1.0e-6) * 1.8
            yaw = env.world.yaw[:, mobile_slots]
            cosine = torch.cos(yaw)
            sine = torch.sin(yaw)
            body_velocity = torch.stack(
                (
                    cosine * desired_world[..., 0] + sine * desired_world[..., 1],
                    -sine * desired_world[..., 0] + cosine * desired_world[..., 1],
                ),
                dim=-1,
            )
            desired_yaw = torch.atan2(delta[..., 1], delta[..., 0])
            yaw_error = torch.atan2(
                torch.sin(desired_yaw - yaw),
                torch.cos(desired_yaw - yaw),
            )
            for index, agent in enumerate(robot_agents):
                actions[agent][:, :2] = body_velocity[:, index] / 3.0
                actions[agent][:, 2] = torch.clamp(
                    yaw_error[:, index] * 3.0 / (2.0 * torch.pi),
                    min=-1.0,
                    max=1.0,
                )
            for agent, unit_slot in (("red_aerial", 4), ("blue_aerial", 12)):
                altitude_error = 1.6 - env.world.position_z[:, unit_slot]
                actions[agent][:, 3] = torch.clamp(
                    altitude_error * 0.75,
                    min=-0.75,
                    max=0.75,
                )
            env.step(actions)
            step += 1
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
