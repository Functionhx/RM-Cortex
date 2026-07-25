from __future__ import annotations

import pytest
import torch

from rm_referee import GameState
from rm_referee.schema import Role, Team, slot
from rm_world.arena import ArenaConfig, ArenaGeometry
from rm_world.pathfinding import (
    GridAStarPlanner,
    PathStatus,
    team_symmetric_goal,
)
from rm_world.scripted import TACTICAL_RED_ROUTES, TacticalMission


def _assert_runtime_projection_keeps_path_unchanged(
    arena: ArenaGeometry,
    planner: GridAStarPlanner,
    path: torch.Tensor,
) -> None:
    fractions = torch.linspace(
        0.0,
        1.0,
        5,
        device=path.device,
        dtype=path.dtype,
    )
    segment_samples = (
        path[:-1, None, :] * (1.0 - fractions[None, :, None])
        + path[1:, None, :] * fractions[None, :, None]
    ).reshape(-1, 2)

    assert planner.points_are_traversable(path).all()
    assert planner.segments_are_traversable(path[:-1], path[1:]).all()
    torch.testing.assert_close(
        arena.project_out_of_obstacles(path, planner.safety_radius_m),
        path,
    )
    torch.testing.assert_close(
        arena.project_out_of_obstacles(
            segment_samples,
            planner.safety_radius_m,
        ),
        segment_samples,
    )


def test_n_env_one_path_avoids_inflated_static_obstacle_deterministically() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=6.0,
            field_width_m=6.0,
            robot_radius_m=0.4,
            terrain=(),
            obstacles=((-0.5, 0.5, -0.5, 0.5),),
        )
    )
    planner = GridAStarPlanner(arena, resolution_m=0.2, clearance_m=0.1)
    start = torch.tensor([[-2.0, 0.0]])
    goal = torch.tensor([[2.0, 0.0]])

    first = planner.plan(start, goal)
    second = planner.plan(start, goal)
    path = first.path()

    assert first.success.tolist() == [True]
    assert first.status.tolist() == [PathStatus.SUCCESS]
    assert first.waypoints_xy.ndim == 3
    assert first.lengths.shape == (1,)
    assert torch.equal(first.waypoints_xy, second.waypoints_xy)
    assert torch.equal(first.lengths, second.lengths)
    assert planner.points_are_traversable(path).all()
    assert torch.allclose(
        arena.project_out_of_obstacles(path, planner.safety_radius_m),
        path,
    )
    assert path[:, 1].abs().max() > 0.5


def test_grid_is_cached_and_shared_by_batched_queries() -> None:
    planner = GridAStarPlanner(resolution_m=0.5)

    cached = planner.grid
    result = planner.plan(
        torch.tensor(((-10.0, -3.0), (-10.0, 3.0))),
        torch.tensor(((10.0, -3.0), (10.0, 3.0))),
    )

    assert planner.grid is cached
    assert result.success.shape == (2,)
    assert result.success.all()

    repeated = planner.plan(
        torch.tensor(((-10.0, -3.0), (-10.0, -3.0))),
        torch.tensor(((10.0, -3.0), (10.0, -3.0))),
    )

    assert repeated.success.all()
    assert planner.cached_path_count == 2
    planner.clear_path_cache()
    assert planner.cached_path_count == 0


def test_default_grid_matches_fifty_millimeter_terrain_resolution() -> None:
    planner = GridAStarPlanner(
        ArenaGeometry(ArenaConfig(field_length_m=1.0, field_width_m=1.0, terrain=(), obstacles=())),
        robot_radius_m=0.0,
    )

    assert planner.resolution_m == 0.05
    torch.testing.assert_close(
        planner.grid.x[1] - planner.grid.x[0],
        torch.tensor(0.05),
    )


def test_safety_radius_rejects_start_near_field_boundary() -> None:
    planner = GridAStarPlanner(
        ArenaGeometry(ArenaConfig(terrain=(), obstacles=())),
        resolution_m=0.2,
    )

    result = planner.plan(
        torch.tensor((13.8, 0.0)),
        torch.tensor((0.0, 0.0)),
    )

    assert not result.success.item()
    assert result.status.item() == PathStatus.INVALID_START
    assert torch.allclose(
        planner.next_waypoint(result, torch.tensor((13.8, 0.0))),
        torch.tensor(((13.8, 0.0),)),
    )


def test_diagonal_move_cannot_cut_between_two_blocked_cardinal_cells() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=4.0,
            field_width_m=4.0,
            robot_radius_m=0.0,
            terrain=(),
            obstacles=(
                (-0.1, 0.1, -1.1, -0.9),
                (-1.1, -0.9, -0.1, 0.1),
            ),
        )
    )
    planner = GridAStarPlanner(
        arena,
        resolution_m=1.0,
        robot_radius_m=0.1,
    )

    result = planner.plan(
        torch.tensor((-1.0, -1.0)),
        torch.tensor((0.0, 0.0)),
    )

    assert not result.success.item()
    assert result.status.item() == PathStatus.NO_PATH


def test_search_rejects_an_edge_that_crosses_a_thin_obstacle_between_free_nodes() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=4.0,
            field_width_m=4.0,
            robot_radius_m=0.0,
            terrain=(),
            obstacles=((0.4, 0.6, 0.4, 0.6),),
        )
    )
    planner = GridAStarPlanner(
        arena,
        resolution_m=1.0,
        robot_radius_m=0.0,
    )

    result = planner.plan(
        torch.tensor((0.0, 0.0)),
        torch.tensor((1.0, 1.0)),
    )
    path = result.path()

    assert result.success.item()
    assert path.shape[0] > 2
    assert planner.segments_are_traversable(path[:-1], path[1:]).all()


def test_square_inflation_matches_runtime_projection_at_aabb_corner() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=4.0,
            field_width_m=4.0,
            robot_radius_m=0.4,
            terrain=(),
            obstacles=((-0.5, 0.5, -0.5, 0.5),),
        )
    )
    planner = GridAStarPlanner(arena, resolution_m=0.1)
    corner = torch.tensor(((0.8, 0.8),))
    outside = torch.tensor(((0.91, 0.91),))

    projected_corner = arena.project_out_of_obstacles(
        corner,
        planner.safety_radius_m,
    )

    assert not torch.allclose(projected_corner, corner)
    assert not planner.points_are_traversable(corner).item()
    x_index = torch.abs(planner.grid.x - corner[0, 0]).argmin()
    y_index = torch.abs(planner.grid.y - corner[0, 1]).argmin()
    assert planner.grid.blocked[y_index, x_index]
    assert planner.points_are_traversable(outside).item()
    assert planner.plan(torch.tensor((-1.5, -1.5)), corner).status.item() == (
        PathStatus.INVALID_GOAL
    )


def test_runtime_projection_fixed_point_is_a_valid_planning_start() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=6.0,
            field_width_m=6.0,
            robot_radius_m=0.4,
            terrain=(),
            obstacles=((-0.5, 0.5, -0.5, 0.5),),
        )
    )
    planner = GridAStarPlanner(arena, resolution_m=0.1)
    runtime_boundary = torch.tensor((0.9, 0.0))

    torch.testing.assert_close(
        arena.project_out_of_obstacles(runtime_boundary, planner.safety_radius_m),
        runtime_boundary,
    )
    assert planner.points_are_traversable(runtime_boundary).item()
    assert planner.plan(runtime_boundary, torch.tensor((2.0, 0.0))).success.item()


def test_nearly_parallel_segment_still_detects_an_obstacle_crossing() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=4.0,
            field_width_m=4.0,
            robot_radius_m=0.0,
            terrain=(),
            obstacles=((0.0, 1.0, -0.5, 0.5),),
        )
    )
    planner = GridAStarPlanner(arena, resolution_m=0.1, robot_radius_m=0.0)

    traversable = planner.segments_are_traversable(
        torch.tensor((-1.0e-6, -1.0)),
        torch.tensor((1.0e-6, 1.0)),
    )

    assert not traversable.item()


def test_traversable_arena_slope_is_not_added_to_static_occupancy() -> None:
    arena = ArenaGeometry()
    planner = GridAStarPlanner(arena, resolution_m=0.1)
    start = torch.tensor((-4.5, 6.9))
    goal = torch.tensor((-3.4, 6.9))

    result = planner.plan(start, goal)
    path = result.path()

    assert result.success.item()
    assert arena.terrain_height(path).max() - arena.terrain_height(path).min() > 0.1


def test_team_goal_symmetry_rotates_both_coordinates() -> None:
    red_goal = torch.tensor((-3.0, 2.0))

    red = team_symmetric_goal(red_goal, 0)
    blue = team_symmetric_goal(red_goal, 1)

    assert torch.equal(red, red_goal)
    assert torch.equal(blue, -red_goal)


def test_next_waypoint_and_velocity_use_local_lookahead_and_arrival_braking() -> None:
    planner = GridAStarPlanner(
        ArenaGeometry(ArenaConfig(terrain=(), obstacles=())),
        resolution_m=0.25,
    )
    result = planner.plan(
        torch.tensor((-2.0, 0.0)),
        torch.tensor((2.0, 0.0)),
    )

    waypoint = planner.next_waypoint(
        result,
        torch.tensor((-2.0, 0.0)),
        lookahead_m=0.6,
    )
    velocity = planner.velocity_to_waypoint(
        torch.tensor((-2.0, 0.0)),
        waypoint,
        max_speed_mps=2.0,
    )
    stopped = planner.velocity_to_waypoint(
        waypoint,
        waypoint,
        max_speed_mps=2.0,
    )

    assert waypoint[0, 0] >= -1.4
    assert waypoint[0, 1].abs() < 1.0e-6
    assert torch.allclose(velocity, torch.tensor(((2.0, 0.0),)))
    assert torch.equal(stopped, torch.zeros_like(stopped))


def test_next_waypoint_never_shortcuts_through_an_inflated_obstacle() -> None:
    arena = ArenaGeometry(
        ArenaConfig(
            field_length_m=6.0,
            field_width_m=6.0,
            robot_radius_m=0.4,
            terrain=(),
            obstacles=((-0.5, 0.5, -0.5, 0.5),),
        )
    )
    planner = GridAStarPlanner(arena, resolution_m=0.1)
    start = torch.tensor((-2.0, 0.0))
    goal = torch.tensor((2.0, 0.0))
    result = planner.plan(start, goal)

    waypoint = planner.next_waypoint(
        result,
        start,
        lookahead_m=10.0,
    )

    assert result.success.item()
    assert not torch.allclose(waypoint, result.resolved_goal_xy)
    assert planner.segments_are_traversable(start, waypoint).item()
    assert not planner.segments_are_traversable(start, result.resolved_goal_xy).item()


@pytest.mark.parametrize(
    ("role", "mission"),
    (
        (Role.ENGINEER, TacticalMission.ENGINEER_STAGE),
        (Role.SENTRY, TacticalMission.SENTRY_FORWARD),
    ),
)
def test_default_ground_paths_match_runtime_projection_for_nodes_and_segments(
    role: Role,
    mission: TacticalMission,
) -> None:
    game = GameState.create(1)
    arena = ArenaGeometry()
    planner = GridAStarPlanner(arena)
    spawn = arena.spawn_positions(game)[0]

    for team in (Team.RED, Team.BLUE):
        start = spawn[slot(team, role)]
        red_route = start.new_tensor(TACTICAL_RED_ROUTES[mission])
        route = red_route if team == Team.RED else -red_route
        default_route = torch.cat((start[None, :], route), dim=0)
        _assert_runtime_projection_keeps_path_unchanged(
            arena,
            planner,
            default_route,
        )

        # The behavior controller uses the route's final point as its mission
        # goal and replaces the intermediate scripted corridor with A*.
        goal = route[-1]
        result = planner.plan(start, goal)
        path = result.path()

        assert result.success.item()
        _assert_runtime_projection_keeps_path_unchanged(
            arena,
            planner,
            path,
        )
