"""Deterministic grid A* navigation for the Isaac-free Torch baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq
import math

import torch
from torch import Tensor

from rm_world.arena import ArenaGeometry


_GEOMETRY_TOLERANCE_M = 1.0e-5


class PathStatus(IntEnum):
    """Stable per-query planner outcomes."""

    SUCCESS = 0
    INVALID_START = 1
    INVALID_GOAL = 2
    NO_PATH = 3


@dataclass(frozen=True)
class NavigationGrid:
    """One cached static navigation grid.

    Rows index ``y`` and columns index ``x``. Terrain surfaces are deliberately
    absent from ``blocked``: the current kinematic backend treats its roads,
    platforms, and slope patches as traversable 2.5D surfaces.
    """

    x: Tensor
    y: Tensor
    blocked: Tensor
    clearance_m: Tensor
    edge_valid: Tensor


@dataclass(frozen=True)
class PathResult:
    """Padded batch of planned paths.

    ``waypoints_xy`` has shape ``[N, max_length, 2]``. Entries after each
    corresponding value in ``lengths`` repeat the final valid waypoint, so the
    tensor is safe to gather without consulting a ragged Python container.
    """

    waypoints_xy: Tensor
    lengths: Tensor
    success: Tensor
    status: Tensor
    path_length_m: Tensor
    requested_goal_xy: Tensor
    resolved_goal_xy: Tensor

    def path(self, index: int = 0) -> Tensor:
        """Return the unpadded path for one batch element."""

        if index < 0 or index >= self.waypoints_xy.shape[0]:
            raise IndexError("path batch index out of range")
        return self.waypoints_xy[index, : int(self.lengths[index].item())]


def team_symmetric_goal(red_goal_xy: Tensor, team: int | Tensor) -> Tensor:
    """Rotate a red-frame target by 180 degrees for the blue team."""

    goal = torch.as_tensor(red_goal_xy)
    if goal.shape[-1:] != (2,):
        raise ValueError("red_goal_xy must have shape [..., 2]")
    if not goal.is_floating_point():
        goal = goal.to(torch.float32)
    team_tensor = torch.as_tensor(team, device=goal.device)
    if ((team_tensor != 0) & (team_tensor != 1)).any():
        raise ValueError("team must contain only RED=0 or BLUE=1")
    sign = torch.where(
        team_tensor == 0,
        torch.ones_like(team_tensor, dtype=goal.dtype),
        -torch.ones_like(team_tensor, dtype=goal.dtype),
    )
    return goal * sign[..., None]


class GridAStarPlanner:
    """Deterministic 8-neighbor A* over cached static arena geometry."""

    _NEIGHBORS = (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    def __init__(
        self,
        arena: ArenaGeometry | None = None,
        *,
        resolution_m: float = 0.05,
        clearance_m: float = 0.0,
        robot_radius_m: float | None = None,
    ) -> None:
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if clearance_m < 0:
            raise ValueError("clearance_m must be non-negative")
        self.arena = arena or ArenaGeometry()
        radius = self.arena.config.robot_radius_m if robot_radius_m is None else robot_radius_m
        if radius < 0:
            raise ValueError("robot_radius_m must be non-negative")
        self.resolution_m = float(resolution_m)
        self.clearance_m = float(clearance_m)
        self.robot_radius_m = float(radius)
        self.safety_radius_m = self.robot_radius_m + self.clearance_m
        self._grid: NavigationGrid | None = None
        self._blocked_flat: tuple[bool, ...] | None = None
        self._edge_valid_flat: bytes | None = None
        self._path_cache: dict[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[tuple[int, int], ...], float] | None,
        ] = {}

    @property
    def grid(self) -> NavigationGrid:
        """Return the lazily built static grid shared by all batch queries."""

        if self._grid is None:
            if self.arena.config.obstacles:
                x, y, _ = self.arena.occupancy_grid(
                    self.resolution_m,
                    device="cpu",
                    dtype=torch.float32,
                )
            else:
                x = torch.arange(
                    -self.arena.config.field_length_m / 2,
                    self.arena.config.field_length_m / 2 + self.resolution_m / 2,
                    self.resolution_m,
                    dtype=torch.float32,
                )
                y = torch.arange(
                    -self.arena.config.field_width_m / 2,
                    self.arena.config.field_width_m / 2 + self.resolution_m / 2,
                    self.resolution_m,
                    dtype=torch.float32,
                )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack((xx, yy), dim=-1)
            clearance = self._runtime_clearance(points)
            blocked = clearance < self.safety_radius_m
            edge_valid = torch.zeros(
                (*blocked.shape, len(self._NEIGHBORS)),
                dtype=torch.bool,
            )
            rows, columns = blocked.shape
            for neighbor_index, (delta_y, delta_x) in enumerate(self._NEIGHBORS):
                source_y = slice(max(0, -delta_y), rows - max(0, delta_y))
                source_x = slice(max(0, -delta_x), columns - max(0, delta_x))
                target_y = slice(max(0, delta_y), rows - max(0, -delta_y))
                target_x = slice(max(0, delta_x), columns - max(0, -delta_x))
                source_points = points[source_y, source_x]
                target_points = points[target_y, target_x]
                edge_valid[source_y, source_x, neighbor_index] = self.segments_are_traversable(
                    source_points.reshape(-1, 2),
                    target_points.reshape(-1, 2),
                ).reshape(source_points.shape[:-1])
            self._grid = NavigationGrid(
                x=x,
                y=y,
                blocked=blocked,
                clearance_m=clearance,
                edge_valid=edge_valid,
            )
            self._blocked_flat = tuple(bool(value) for value in blocked.flatten().tolist())
            self._edge_valid_flat = bytes(edge_valid.flatten().tolist())
        return self._grid

    def _runtime_clearance(self, position_xy: Tensor) -> Tensor:
        """Return clearance under the runtime's square AABB expansion.

        ``ArenaGeometry.signed_distance`` rounds AABB corners because it uses
        Euclidean distance. Runtime collision projection instead expands every
        AABB independently along x and y. The L-infinity distance below has
        exactly the same square corners as ``project_out_of_obstacles``.
        """

        boundary = torch.minimum(
            self.arena.config.field_length_m / 2 - torch.abs(position_xy[..., 0]),
            self.arena.config.field_width_m / 2 - torch.abs(position_xy[..., 1]),
        )
        if not self.arena.config.obstacles:
            return boundary
        boxes = self.arena.obstacles(
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        center = 0.5 * (boxes[:, (0, 2)] + boxes[:, (1, 3)])
        half_extent = 0.5 * (boxes[:, (1, 3)] - boxes[:, (0, 2)])
        offset = torch.abs(position_xy[..., None, :] - center) - half_extent
        obstacle_clearance = offset.amax(dim=-1).amin(dim=-1)
        return torch.minimum(boundary, obstacle_clearance)

    def points_are_traversable(self, position_xy: Tensor) -> Tensor:
        """Return whether robot centers are valid under runtime collision geometry."""

        position = torch.as_tensor(position_xy)
        if position.shape[-1:] != (2,):
            raise ValueError("position_xy must have shape [..., 2]")
        if not position.is_floating_point():
            position = position.to(torch.float32)
        return torch.isfinite(position).all(dim=-1) & (
            self._runtime_clearance(position) >= self.safety_radius_m - _GEOMETRY_TOLERANCE_M
        )

    def segments_are_traversable(self, start_xy: Tensor, end_xy: Tensor) -> Tensor:
        """Return whether complete line segments avoid expanded AABB interiors."""

        start, end = self._broadcast_queries(start_xy, end_xy)
        endpoints_valid = self.points_are_traversable(start) & self.points_are_traversable(end)
        if not self.arena.config.obstacles:
            return endpoints_valid

        geometry_dtype = (
            torch.float32 if start.dtype in (torch.float16, torch.bfloat16) else start.dtype
        )
        geometry_start = start.to(geometry_dtype)
        geometry_end = end.to(geometry_dtype)
        boxes = self.arena.obstacles(
            device=start.device,
            dtype=geometry_dtype,
        )
        lower = boxes[:, (0, 2)] - self.safety_radius_m
        upper = boxes[:, (1, 3)] + self.safety_radius_m
        segment_start = geometry_start[:, None, :]
        direction = (geometry_end - geometry_start)[:, None, :]
        parallel = direction == 0
        parallel_inside = (segment_start > lower) & (segment_start < upper)
        safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
        first = (lower - segment_start) / safe_direction
        second = (upper - segment_start) / safe_direction
        entry = torch.minimum(first, second)
        exit = torch.maximum(first, second)
        entry = torch.where(
            parallel,
            torch.where(
                parallel_inside,
                torch.full_like(entry, -torch.inf),
                torch.full_like(entry, torch.inf),
            ),
            entry,
        )
        exit = torch.where(
            parallel,
            torch.where(
                parallel_inside,
                torch.full_like(exit, torch.inf),
                torch.full_like(exit, -torch.inf),
            ),
            exit,
        )
        entry_t = torch.maximum(
            entry.amax(dim=-1),
            torch.zeros((), device=start.device, dtype=geometry_dtype),
        )
        exit_t = torch.minimum(
            exit.amin(dim=-1),
            torch.ones((), device=start.device, dtype=geometry_dtype),
        )
        intersects_interior = entry_t < exit_t
        return endpoints_valid & ~intersects_interior.any(dim=-1)

    @staticmethod
    def _normalize_xy(value: Tensor, *, name: str) -> Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 1:
            if tensor.shape != (2,):
                raise ValueError(f"{name} must have shape [2] or [N, 2]")
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2 or tensor.shape[-1] != 2:
            raise ValueError(f"{name} must have shape [2] or [N, 2]")
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        return tensor

    @classmethod
    def _broadcast_queries(cls, start_xy: Tensor, goal_xy: Tensor) -> tuple[Tensor, Tensor]:
        start = cls._normalize_xy(start_xy, name="start_xy")
        goal = cls._normalize_xy(goal_xy, name="goal_xy").to(
            device=start.device,
            dtype=start.dtype,
        )
        if start.shape[0] == goal.shape[0]:
            return start, goal
        if start.shape[0] == 1:
            return start.expand(goal.shape[0], -1), goal
        if goal.shape[0] == 1:
            return start, goal.expand(start.shape[0], -1)
        raise ValueError("start_xy and goal_xy batch sizes are not broadcastable")

    def _nearest_free_index(self, point_xy: Tensor) -> tuple[int, int]:
        grid = self.grid
        x_index = int(torch.abs(grid.x - point_xy[0]).argmin().item())
        y_index = int(torch.abs(grid.y - point_xy[1]).argmin().item())
        if not bool(grid.blocked[y_index, x_index].item()):
            return y_index, x_index

        distance_sq = (grid.x[None, :] - point_xy[0]).square() + (
            grid.y[:, None] - point_xy[1]
        ).square()
        distance_sq = torch.where(
            grid.blocked,
            torch.full_like(distance_sq, torch.inf),
            distance_sq,
        )
        flat_index = int(distance_sq.argmin().item())
        return divmod(flat_index, grid.x.numel())

    def _search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[list[tuple[int, int]], float] | None:
        cache_key = (start, goal)
        if cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            if cached is None:
                return None
            indices, cost = cached
            return list(indices), cost

        grid = self.grid
        rows = grid.y.numel()
        columns = grid.x.numel()
        node_count = rows * columns
        blocked = self._blocked_flat
        edge_valid = self._edge_valid_flat
        assert blocked is not None
        assert edge_valid is not None

        start_flat = start[0] * columns + start[1]
        goal_flat = goal[0] * columns + goal[1]
        if start_flat == goal_flat:
            return [start], 0.0

        x_step = float((grid.x[1] - grid.x[0]).item()) if columns > 1 else 0.0
        y_step = float((grid.y[1] - grid.y[0]).item()) if rows > 1 else 0.0

        def heuristic(flat_index: int) -> float:
            row, column = divmod(flat_index, columns)
            return math.hypot(
                (column - goal[1]) * x_step,
                (row - goal[0]) * y_step,
            )

        costs = [math.inf] * node_count
        parent = [-1] * node_count
        closed = bytearray(node_count)
        costs[start_flat] = 0.0
        initial_h = heuristic(start_flat)
        open_heap: list[tuple[float, float, int]] = [(initial_h, initial_h, start_flat)]

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if closed[current]:
                continue
            closed[current] = 1
            if current == goal_flat:
                flat_path = [current]
                while flat_path[-1] != start_flat:
                    flat_path.append(parent[flat_path[-1]])
                flat_path.reverse()
                result = [divmod(index, columns) for index in flat_path]
                self._path_cache[cache_key] = (tuple(result), costs[current])
                return result, costs[current]

            row, column = divmod(current, columns)
            for neighbor_index, (delta_y, delta_x) in enumerate(self._NEIGHBORS):
                neighbor_y = row + delta_y
                neighbor_x = column + delta_x
                if not (0 <= neighbor_y < rows and 0 <= neighbor_x < columns):
                    continue
                neighbor = neighbor_y * columns + neighbor_x
                if (
                    blocked[neighbor]
                    or closed[neighbor]
                    or not edge_valid[current * len(self._NEIGHBORS) + neighbor_index]
                ):
                    continue
                if delta_x and delta_y:
                    horizontal = row * columns + neighbor_x
                    vertical = neighbor_y * columns + column
                    if blocked[horizontal] or blocked[vertical]:
                        continue
                edge_cost = math.hypot(delta_x * x_step, delta_y * y_step)
                tentative = costs[current] + edge_cost
                if tentative + 1.0e-12 >= costs[neighbor]:
                    continue
                costs[neighbor] = tentative
                parent[neighbor] = current
                neighbor_h = heuristic(neighbor)
                heapq.heappush(
                    open_heap,
                    (tentative + neighbor_h, neighbor_h, neighbor),
                )
        self._path_cache[cache_key] = None
        return None

    def clear_path_cache(self) -> None:
        """Drop cached query results without rebuilding the static grid."""

        self._path_cache.clear()

    @property
    def cached_path_count(self) -> int:
        """Return the number of unique static start/goal queries cached."""

        return len(self._path_cache)

    def plan(self, start_xy: Tensor, goal_xy: Tensor) -> PathResult:
        """Plan one or more static paths.

        Inputs accept ``[2]`` or ``[N, 2]`` tensors and broadcast a singleton
        batch. Results always retain an explicit leading batch dimension.
        """

        start, goal = self._broadcast_queries(start_xy, goal_xy)
        batch_size = start.shape[0]
        grid = self.grid
        valid_start = self.points_are_traversable(start)
        valid_goal = self.points_are_traversable(goal)
        start_cpu = start.detach().to(device="cpu", dtype=grid.x.dtype)
        goal_cpu = goal.detach().to(device="cpu", dtype=grid.x.dtype)

        path_indices: list[list[tuple[int, int]]] = []
        statuses: list[PathStatus] = []
        costs: list[float] = []
        resolved_goals: list[tuple[float, float]] = []
        for batch_index in range(batch_size):
            if not bool(valid_start[batch_index].item()):
                path_indices.append([])
                statuses.append(PathStatus.INVALID_START)
                costs.append(0.0)
                resolved_goals.append((math.nan, math.nan))
                continue

            start_index = self._nearest_free_index(start_cpu[batch_index])
            if not bool(valid_goal[batch_index].item()):
                path_indices.append([start_index])
                statuses.append(PathStatus.INVALID_GOAL)
                costs.append(0.0)
                resolved_goals.append((math.nan, math.nan))
                continue

            goal_index = self._nearest_free_index(goal_cpu[batch_index])
            resolved_goals.append(
                (
                    float(grid.x[goal_index[1]].item()),
                    float(grid.y[goal_index[0]].item()),
                )
            )
            search_result = self._search(start_index, goal_index)
            if search_result is None:
                path_indices.append([start_index])
                statuses.append(PathStatus.NO_PATH)
                costs.append(0.0)
            else:
                indices, path_cost = search_result
                path_indices.append(indices)
                statuses.append(PathStatus.SUCCESS)
                costs.append(path_cost)

        max_length = max((len(path) for path in path_indices), default=1)
        max_length = max(max_length, 1)
        waypoints = torch.empty(
            (batch_size, max_length, 2),
            device=start.device,
            dtype=start.dtype,
        )
        lengths = torch.empty(batch_size, device=start.device, dtype=torch.long)
        for batch_index, indices in enumerate(path_indices):
            if not indices:
                waypoint = start[batch_index]
                waypoints[batch_index] = waypoint
                lengths[batch_index] = 1
                continue
            coordinates = torch.tensor(
                [
                    (float(grid.x[x_index].item()), float(grid.y[y_index].item()))
                    for y_index, x_index in indices
                ],
                device=start.device,
                dtype=start.dtype,
            )
            length = coordinates.shape[0]
            waypoints[batch_index, :length] = coordinates
            waypoints[batch_index, length:] = coordinates[-1]
            lengths[batch_index] = length

        status = torch.tensor(
            [int(value) for value in statuses],
            device=start.device,
            dtype=torch.long,
        )
        resolved_goal = torch.tensor(
            resolved_goals,
            device=start.device,
            dtype=start.dtype,
        )
        return PathResult(
            waypoints_xy=waypoints,
            lengths=lengths,
            success=status == int(PathStatus.SUCCESS),
            status=status,
            path_length_m=torch.tensor(
                costs,
                device=start.device,
                dtype=start.dtype,
            ),
            requested_goal_xy=goal.clone(),
            resolved_goal_xy=resolved_goal,
        )

    def next_waypoint(
        self,
        result: PathResult,
        position_xy: Tensor,
        *,
        lookahead_m: float = 0.6,
    ) -> Tensor:
        """Return a local lookahead waypoint for each planned path.

        Failed queries return their current position, giving callers a
        deterministic hold behavior.
        """

        if lookahead_m < 0:
            raise ValueError("lookahead_m must be non-negative")
        position = self._normalize_xy(position_xy, name="position_xy").to(
            device=result.waypoints_xy.device,
            dtype=result.waypoints_xy.dtype,
        )
        if position.shape[0] == 1 and result.waypoints_xy.shape[0] != 1:
            position = position.expand(result.waypoints_xy.shape[0], -1)
        if position.shape[0] != result.waypoints_xy.shape[0]:
            raise ValueError("position_xy batch size differs from PathResult")

        waypoint = position.clone()
        for batch_index in range(position.shape[0]):
            if not bool(result.success[batch_index].item()):
                continue
            length = int(result.lengths[batch_index].item())
            path = result.waypoints_xy[batch_index, :length]
            nearest = int(
                torch.linalg.vector_norm(
                    path - position[batch_index],
                    dim=-1,
                )
                .argmin()
                .item()
            )
            target = nearest
            if not bool(
                self.segments_are_traversable(
                    position[batch_index],
                    path[target],
                ).item()
            ):
                continue
            traveled = 0.0
            while target + 1 < length and traveled < lookahead_m:
                next_target = target + 1
                next_traveled = traveled + float(
                    torch.linalg.vector_norm(path[next_target] - path[target]).item()
                )
                if not bool(
                    self.segments_are_traversable(
                        position[batch_index],
                        path[next_target],
                    ).item()
                ):
                    break
                target = next_target
                traveled = next_traveled
            waypoint[batch_index] = path[target]
        return waypoint

    @classmethod
    def velocity_to_waypoint(
        cls,
        position_xy: Tensor,
        waypoint_xy: Tensor,
        *,
        max_speed_mps: float = 2.0,
        slow_radius_m: float = 0.5,
        stop_tolerance_m: float = 0.05,
    ) -> Tensor:
        """Return a world-frame velocity command with linear arrival braking."""

        if max_speed_mps < 0:
            raise ValueError("max_speed_mps must be non-negative")
        if slow_radius_m <= 0:
            raise ValueError("slow_radius_m must be positive")
        if stop_tolerance_m < 0:
            raise ValueError("stop_tolerance_m must be non-negative")
        position, waypoint = cls._broadcast_queries(position_xy, waypoint_xy)
        delta = waypoint - position
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        speed = max_speed_mps * torch.clamp(distance / slow_radius_m, max=1.0)
        speed = torch.where(
            distance <= stop_tolerance_m,
            torch.zeros_like(speed),
            speed,
        )
        return delta / torch.clamp(distance, min=1.0e-9) * speed
