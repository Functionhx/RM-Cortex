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
        from rm_isaac.direct_env_runtime import ROBOT_AGENT_SLOTS
        from rm_referee import constants
        from rm_referee.schema import PurchaseKind, Role, Team, Weapon
        from rm_world import TacticalScriptedOpponent

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
        controller = TacticalScriptedOpponent(arena=env.arena)
        step = 0
        while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
            world_actions = controller.act(env.game, env.world)
            for action in actions.values():
                action.zero_()

            for agent, unit_slot in ROBOT_AGENT_SLOTS.items():
                command = actions[agent]
                command[:, :2] = torch.clamp(
                    world_actions.body_velocity_xy[:, unit_slot] / 3.0,
                    min=-1.0,
                    max=1.0,
                )
                command[:, 2] = torch.clamp(
                    world_actions.yaw_rate[:, unit_slot] / (2.0 * math.pi),
                    min=-1.0,
                    max=1.0,
                )
                command[:, 3] = torch.clamp(
                    world_actions.vertical_velocity[:, unit_slot] / 2.0,
                    min=-1.0,
                    max=1.0,
                )
                command[:, 4] = (
                    world_actions.target[:, unit_slot].to(
                        command.dtype,
                    )
                    / 4.0
                    - 1.0
                )
                command[:, 5] = torch.where(
                    world_actions.fire[:, unit_slot],
                    torch.ones_like(command[:, 5]),
                    -torch.ones_like(command[:, 5]),
                )

                team = unit_slot // constants.ROLES_PER_TEAM
                role = unit_slot % constants.ROLES_PER_TEAM
                if role == Role.HERO:
                    command[:, 6] = torch.where(
                        world_actions.hero_deploy[:, team],
                        1.0,
                        -1.0,
                    )
                elif role == Role.ENGINEER:
                    command[:, 6] = torch.where(
                        world_actions.rebuild_outpost[:, unit_slot],
                        1.0,
                        -1.0,
                    )
                    command[:, 7] = (
                        world_actions.tech_complete_level[:, team].to(command.dtype) / 2.0 - 1.0
                    )
                    command[:, 13] = world_actions.rune_trigger[:, team].to(command.dtype) / 2.0
                elif role == Role.AERIAL:
                    command[:, 6] = torch.where(
                        world_actions.aerial_support[:, team],
                        1.0,
                        -1.0,
                    )
                elif role == Role.SENTRY:
                    command[:, 6] = world_actions.sentry_stance[:, team].to(command.dtype) - 1.0
                    command[:, 7] = torch.where(
                        world_actions.sentry_claim_ammo[:, team],
                        1.0,
                        -1.0,
                    )

                command[:, 8] = torch.where(
                    world_actions.remote_heal[:, unit_slot],
                    1.0,
                    -1.0,
                )
                command[:, 9] = torch.where(
                    world_actions.immediate_respawn[:, unit_slot],
                    1.0,
                    -1.0,
                )
                purchase = world_actions.purchase_unit[:, team] == unit_slot
                command[:, 10] = torch.where(purchase, 1.0, -1.0)
                command[:, 11] = torch.where(
                    purchase & (world_actions.purchase_kind[:, team] == int(PurchaseKind.REMOTE)),
                    1.0,
                    -1.0,
                )
                command[:, 12] = torch.where(
                    purchase & (world_actions.purchase_weapon[:, team] == int(Weapon.MM42)),
                    1.0,
                    -1.0,
                )

            for team, prefix in ((Team.RED, "red"), (Team.BLUE, "blue")):
                dart = actions[f"{prefix}_dart"]
                dart[:, 0] = torch.where(
                    world_actions.dart_open_gate[:, team],
                    1.0,
                    -1.0,
                )
                dart[:, 1] = torch.where(
                    world_actions.dart_close_gate[:, team],
                    1.0,
                    -1.0,
                )
                dart[:, 2] = torch.where(
                    world_actions.dart_target[:, team] >= 0,
                    world_actions.dart_target[:, team].to(dart.dtype) / 2.0 - 1.0,
                    -torch.ones_like(dart[:, 2]),
                )

                radar = actions[f"{prefix}_radar"]
                radar_target = torch.clamp(
                    world_actions.radar_target[:, team],
                    min=0,
                    max=constants.UNIT_COUNT - 1,
                )
                radar[:, 0] = radar_target.to(radar.dtype) / 7.5 - 1.0
                actual_xy = torch.gather(
                    env.world.position_xy,
                    1,
                    radar_target[:, None, None].expand(-1, 1, 2),
                ).squeeze(1)
                radar[:, 1:3] = torch.clamp(
                    (world_actions.radar_report_xy[:, team] - actual_xy) / 2.0,
                    min=-1.0,
                    max=1.0,
                )
                radar[:, 3] = torch.where(
                    world_actions.radar_illuminate[:, team],
                    1.0,
                    -1.0,
                )
                radar[:, 4] = torch.where(
                    world_actions.radar_double[:, team],
                    1.0,
                    -1.0,
                )
                radar[:, 5] = torch.where(
                    world_actions.radar_key_solved[:, team],
                    1.0,
                    -1.0,
                )
            env.step(actions)
            step += 1
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
