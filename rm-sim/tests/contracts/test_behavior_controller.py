from __future__ import annotations

import torch

from rm_referee import constants
from rm_referee.schema import DartGateState, DartTarget, Role, Team, slot
from rm_world import (
    BehaviorTreeOpponent,
    GridAStarPlanner,
    TacticalMission,
    TorchEnvConfig,
    TorchRMArena,
)
from rm_world.scripted import tactical_route_waypoints


def _controller(
    environment: TorchRMArena,
    *,
    resolution_m: float = 0.2,
) -> BehaviorTreeOpponent:
    return BehaviorTreeOpponent(
        arena=environment.arena,
        planner=GridAStarPlanner(
            environment.arena,
            resolution_m=resolution_m,
            clearance_m=0.02,
        ),
    )


def test_behavior_tree_controller_runs_global_and_all_role_trees_before_astar() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=1, seed=7))
    controller = _controller(environment)

    actions = controller.act(
        environment.game,
        environment.world,
        team=Team.RED,
    )

    actions.validate(environment.game)
    assert controller.last_trace is not None
    assert controller.last_blackboard is not None
    assert controller.last_trace.path(0, Team.RED)[0] == "global"
    for role in Role:
        assert controller.last_trace.path(0, Team.RED, role)[0] == role.name.lower()
    diagnostics = controller.navigation_diagnostics
    assert diagnostics.plan_requests == 5
    assert diagnostics.plan_successes == 5
    assert diagnostics.plan_failures == 0
    assert diagnostics.unique_cached_paths == 5
    assert controller.leaf_hit_counts["global/strategy_selector/publish_opening"] == (1, 0)
    assert not actions.body_velocity_xy[0, :6].eq(0).all()
    assert actions.body_velocity_xy[0, Role.BASE].eq(0).all()
    assert actions.body_velocity_xy[0, Role.OUTPOST].eq(0).all()


def test_unchanged_behavior_goals_reuse_paths_and_partial_reset_replans() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=2, seed=11))
    controller = _controller(environment)

    controller.act(environment.game, environment.world, team=Team.RED)
    initial = controller.navigation_diagnostics
    controller.act(environment.game, environment.world, team=Team.RED)

    assert controller.navigation_diagnostics.plan_requests == initial.plan_requests

    controller.reset(torch.tensor([0]))
    controller.act(environment.game, environment.world, team=Team.RED)

    assert controller.navigation_diagnostics.plan_requests > initial.plan_requests


def test_behavior_tree_astar_rollout_keeps_commands_finite_and_clear() -> None:
    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=1,
            seed=13,
            validate_referee=False,
        )
    )
    controller = _controller(environment, resolution_m=0.1)
    minimum_static_clearance = torch.tensor(torch.inf)

    for _ in range(20):
        actions = controller.act(environment.game, environment.world)
        actions.validate(environment.game)
        result = environment.step(actions)
        minimum_static_clearance = torch.minimum(
            minimum_static_clearance,
            environment.arena.signed_distance(environment.world.position_xy).min(),
        )
        assert torch.isfinite(environment.world.position_xy).all()
        if result.terminated.all():
            break

    assert minimum_static_clearance >= environment.arena.config.robot_radius_m - 1.0e-4
    assert controller.navigation_diagnostics.plan_failures == 0


def test_default_behavior_tree_navigation_uses_fifty_millimeter_grid() -> None:
    controller = BehaviorTreeOpponent()

    assert controller.planner.resolution_m == 0.05


def test_hero_deploy_request_is_held_across_ticks_until_deliberate_withdrawal() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=1, seed=17, validate_referee=False))
    controller = _controller(environment)
    hero = slot(Team.RED, Role.HERO)
    deploy_goal = tactical_route_waypoints(
        TacticalMission.HERO_DEPLOY,
        Team.RED,
        device=environment.game.device,
        dtype=environment.game.dtype,
    )[-1]
    environment.world.position_xy[0, hero] = deploy_goal

    for _ in range(11):
        actions = controller.act(environment.game, environment.world, team=Team.RED)
        assert actions.hero_deploy[0, Team.RED]
        environment.step(actions)

    assert environment.game.hero_deployed[0, Team.RED]
    held = controller.act(environment.game, environment.world, team=Team.RED)
    assert held.hero_deploy[0, Team.RED]
    environment.step(held)
    assert environment.game.hero_deployed[0, Team.RED]

    environment.game.alive[0, slot(Team.BLUE, Role.OUTPOST)] = False
    withdrawn = controller.act(environment.game, environment.world, team=Team.RED)
    assert not withdrawn.hero_deploy[0, Team.RED]
    environment.step(withdrawn)
    assert not environment.game.hero_deployed[0, Team.RED]


def test_structure_tree_commands_are_copied_to_world_actions() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=1, seed=19, validate_referee=False))
    controller = _controller(environment)
    environment.game.elapsed_s.fill_(200.0)
    environment.game.dart_gate_state[:, Team.RED] = int(DartGateState.OPEN)
    environment.game.dart_gate_timer_s[:, Team.RED] = 30.0
    environment.game.radar_double_charges[:, Team.RED] = 1
    environment.game.aerial_support_active[:, Team.BLUE] = True
    environment.game.alive[:, slot(Team.BLUE, Role.AERIAL)] = True

    actions = controller.act(environment.game, environment.world, team=Team.RED)

    assert actions.dart_target[0, Team.RED] == DartTarget.OUTPOST
    assert actions.radar_target[0, Team.RED] >= 0
    target = int(actions.radar_target[0, Team.RED].item())
    assert torch.allclose(
        actions.radar_report_xy[0, Team.RED],
        environment.world.position_xy[0, target],
    )
    assert actions.radar_double[0, Team.RED]
    assert actions.radar_illuminate[0, Team.RED]
    assert not actions.radar_double[0, Team.BLUE]
    assert not actions.radar_illuminate[0, Team.BLUE]

    immediate_repeat = controller.act(
        environment.game,
        environment.world,
        team=Team.RED,
    )
    assert immediate_repeat.dart_target[0, Team.RED] == -1

    environment.game.elapsed_s.add_(constants.DART_DETECTION_BLOCK_S)
    next_cadence = controller.act(
        environment.game,
        environment.world,
        team=Team.RED,
    )
    assert next_cadence.dart_target[0, Team.RED] == DartTarget.OUTPOST


def test_structure_commands_reach_referee_dart_and_radar_state() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=1, seed=23, validate_referee=False))
    controller = _controller(environment)
    environment.game.elapsed_s.fill_(200.0)
    environment.game.dart_gate_state[:, Team.RED] = int(DartGateState.OPEN)
    environment.game.dart_gate_timer_s[:, Team.RED] = 30.0
    environment.game.radar_double_charges[:, Team.RED] = 1
    rounds_before = environment.game.dart_rounds[0, Team.RED].clone()

    actions = controller.act(environment.game, environment.world, team=Team.RED)
    radar_target = int(actions.radar_target[0, Team.RED].item())
    environment.step(actions)

    assert environment.game.dart_rounds[0, Team.RED] == rounds_before - 1
    assert environment.game.radar_p[0, Team.RED, radar_target] > 0
    assert environment.game.radar_double_charges[0, Team.RED] == 0
    assert environment.game.radar_double_uses[0, Team.RED] == 1
    assert environment.game.radar_double_s[0, Team.RED] > 0
