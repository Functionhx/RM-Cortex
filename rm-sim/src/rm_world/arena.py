"""Analytic Phase 1 arena primitives, zones, collision, and LOS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Zone, unit_roles, unit_teams
from rm_referee.state import GameState


# Manual V1.5.0 p. 34, figure 4-5. The blue outpost is 17,008mm
# from the red short edge and 3,643mm from the north long edge.
# With the field origin at center, the paired centers rotate by 180 degrees.
RED_OUTPOST_CENTER_XY = (-3.008, -3.643)
BLUE_OUTPOST_CENTER_XY = (3.008, 3.643)
# Figure 4-5 placement is diagram-derived here because the pad center is not
# separately dimensioned. Keep it centralized for spawn and contact checks.
RED_AERIAL_PAD_CENTER_XY = (-13.0, 5.0)
BLUE_AERIAL_PAD_CENTER_XY = (13.0, -5.0)


@dataclass(frozen=True)
class TerrainPrimitive:
    """Diagram-derived ``[SIM]`` terrain surface.

    Dimensions and slopes follow rule-manual figures 4-5 and 4-25--4-37.
    Placements remain approximate until official CAD/USD assets are available.
    Elevation changes along the primitive's local x-axis.
    """

    name: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    elevation_start_m: float
    elevation_end_m: float
    yaw_deg: float = 0.0
    category: str = "platform"
    team: int | None = None


DEFAULT_TERRAIN: tuple[TerrainPrimitive, ...] = (
    TerrainPrimitive(
        "central_highland",
        (0.0, 0.0),
        (5.8, 6.4),
        0.35,
        0.35,
        category="central",
    ),
    TerrainPrimitive(
        "central_north_ramp",
        (0.0, 4.25),
        (2.1, 5.8),
        0.35,
        0.0,
        yaw_deg=90.0,
        category="ramp",
    ),
    TerrainPrimitive(
        "central_south_ramp",
        (0.0, -4.25),
        (2.1, 5.8),
        0.0,
        0.35,
        yaw_deg=90.0,
        category="ramp",
    ),
    TerrainPrimitive(
        "red_assembly",
        (-3.2, 0.0),
        (1.3, 3.0),
        0.25,
        0.25,
        category="assembly",
        team=0,
    ),
    TerrainPrimitive(
        "blue_assembly",
        (3.2, 0.0),
        (1.3, 3.0),
        0.25,
        0.25,
        category="assembly",
        team=1,
    ),
    TerrainPrimitive(
        "red_trapezoid_highland",
        (-9.45, 5.55),
        (6.5, 2.4),
        0.30,
        0.30,
        category="trapezoid",
        team=0,
    ),
    TerrainPrimitive(
        "red_trapezoid_ramp",
        (-6.8, 3.9),
        (3.0, 1.6),
        0.0,
        0.30,
        yaw_deg=135.0,
        category="ramp",
        team=0,
    ),
    TerrainPrimitive(
        "blue_trapezoid_highland",
        (9.45, -5.55),
        (6.5, 2.4),
        0.30,
        0.30,
        yaw_deg=180.0,
        category="trapezoid",
        team=1,
    ),
    TerrainPrimitive(
        "blue_trapezoid_ramp",
        (6.8, -3.9),
        (3.0, 1.6),
        0.0,
        0.30,
        yaw_deg=315.0,
        category="ramp",
        team=1,
    ),
    TerrainPrimitive(
        "red_road",
        (-9.5, -5.4),
        (6.5, 2.0),
        0.25,
        0.25,
        category="road",
        team=0,
    ),
    TerrainPrimitive(
        "red_road_link",
        (-5.1, -4.4),
        (3.0, 1.5),
        0.25,
        0.35,
        yaw_deg=35.0,
        category="road",
        team=0,
    ),
    TerrainPrimitive(
        "blue_road",
        (9.5, 5.4),
        (6.5, 2.0),
        0.25,
        0.25,
        yaw_deg=180.0,
        category="road",
        team=1,
    ),
    TerrainPrimitive(
        "blue_road_link",
        (5.1, 4.4),
        (3.0, 1.5),
        0.25,
        0.35,
        yaw_deg=215.0,
        category="road",
        team=1,
    ),
    TerrainPrimitive(
        "red_fly_ramp",
        (-3.2, -6.2),
        (2.151, 1.145),
        0.203,
        0.553,
        category="fly_ramp",
        team=0,
    ),
    TerrainPrimitive(
        "blue_fly_ramp",
        (3.2, 6.2),
        (2.151, 1.145),
        0.203,
        0.553,
        yaw_deg=180.0,
        category="fly_ramp",
        team=1,
    ),
    TerrainPrimitive(
        "red_rough_road",
        (-10.4, -6.25),
        (2.0, 1.2),
        0.04,
        0.04,
        category="rough",
        team=0,
    ),
    TerrainPrimitive(
        "blue_rough_road",
        (10.4, 6.25),
        (2.0, 1.2),
        0.04,
        0.04,
        yaw_deg=180.0,
        category="rough",
        team=1,
    ),
    TerrainPrimitive(
        "red_fortress",
        (-3.8, -4.2),
        (1.12, 1.939),
        0.15,
        0.15,
        category="fortress",
        team=0,
    ),
    TerrainPrimitive(
        "blue_fortress",
        (3.8, 4.2),
        (1.12, 1.939),
        0.15,
        0.15,
        yaw_deg=180.0,
        category="fortress",
        team=1,
    ),
    TerrainPrimitive(
        "red_tunnel",
        (-4.1, -4.65),
        (1.6, 0.8),
        0.0,
        0.0,
        yaw_deg=35.0,
        category="tunnel",
        team=0,
    ),
    TerrainPrimitive(
        "blue_tunnel",
        (4.1, 4.65),
        (1.6, 0.8),
        0.0,
        0.0,
        yaw_deg=215.0,
        category="tunnel",
        team=1,
    ),
)


@dataclass(frozen=True)
class ArenaConfig:
    """Explicit ``[SIM]`` coordinate and diagram-derived geometry choices."""

    field_length_m: float = 28.0
    field_width_m: float = 15.0
    field_crown_slope_deg: float = 1.5
    terrain_resolution_m: float = 0.05
    robot_radius_m: float = 0.40
    terrain: tuple[TerrainPrimitive, ...] = DEFAULT_TERRAIN
    # xmin, xmax, ymin, ymax. Traversable platforms live in ``terrain``;
    # these boxes represent the central mechanism and retaining walls.
    obstacles: tuple[tuple[float, float, float, float], ...] = (
        (-0.65, 0.65, -0.80, 0.80),
        (-4.05, -3.72, -2.30, 2.30),
        (3.72, 4.05, -2.30, 2.30),
        (-8.15, -7.82, 3.25, 5.85),
        (7.82, 8.15, -5.85, -3.25),
    )


class ArenaGeometry:
    """Vectorized geometry shared by rollout, debugging, and the Isaac bridge."""

    def __init__(self, config: ArenaConfig | None = None) -> None:
        self.config = config or ArenaConfig()
        self._obstacle_cache: dict[tuple[str, torch.dtype], Tensor] = {}
        self._terrain_cache: dict[
            tuple[str, torch.dtype],
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        ] = {}
        self._height_grid_cache: dict[tuple[str, torch.dtype], Tensor] = {}

    def obstacles(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (str(device), dtype)
        if key not in self._obstacle_cache:
            self._obstacle_cache[key] = torch.tensor(
                self.config.obstacles,
                device=device,
                dtype=dtype,
            )
        return self._obstacle_cache[key]

    def outpost_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the figure 4-5 outpost centers in red/blue team order."""

        return torch.tensor(
            (RED_OUTPOST_CENTER_XY, BLUE_OUTPOST_CENTER_XY),
            device=device,
            dtype=dtype,
        )

    def aerial_pad_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the diagram-derived ``[SIM]`` aerial pad centers."""

        return torch.tensor(
            (RED_AERIAL_PAD_CENTER_XY, BLUE_AERIAL_PAD_CENTER_XY),
            device=device,
            dtype=dtype,
        )

    def _terrain_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        key = (str(device), dtype)
        if key not in self._terrain_cache:
            centers = torch.tensor(
                [primitive.center_xy for primitive in self.config.terrain],
                device=device,
                dtype=dtype,
            )
            sizes = torch.tensor(
                [primitive.size_xy for primitive in self.config.terrain],
                device=device,
                dtype=dtype,
            )
            yaw = torch.tensor(
                [primitive.yaw_deg for primitive in self.config.terrain],
                device=device,
                dtype=dtype,
            )
            angle = -yaw * torch.pi / 180.0
            start = torch.tensor(
                [primitive.elevation_start_m for primitive in self.config.terrain],
                device=device,
                dtype=dtype,
            )
            end = torch.tensor(
                [primitive.elevation_end_m for primitive in self.config.terrain],
                device=device,
                dtype=dtype,
            )
            self._terrain_cache[key] = (
                centers,
                sizes,
                torch.cos(angle),
                torch.sin(angle),
                start,
                end,
            )
        return self._terrain_cache[key]

    def spawn_positions(self, game: GameState) -> Tensor:
        red = torch.tensor(
            (
                (-10.8, 0.0),
                (-12.0, -2.0),
                (-10.0, -3.0),
                (-10.0, 3.0),
                RED_AERIAL_PAD_CENTER_XY,
                (-9.0, 0.0),
                (-12.5, 0.0),
                RED_OUTPOST_CENTER_XY,
            ),
            device=game.device,
            dtype=game.dtype,
        )
        # The manual describes a center-symmetric field, so team placements
        # rotate by 180 degrees rather than reflecting only the x coordinate.
        blue = -red
        positions = torch.cat((red, blue), dim=0)
        return positions.unsqueeze(0).expand(game.num_envs, -1, -1).clone()

    def terrain_corners(
        self,
        primitive: TerrainPrimitive,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        half_x = primitive.size_xy[0] / 2
        half_y = primitive.size_xy[1] / 2
        local = torch.tensor(
            ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)),
            device=device,
            dtype=dtype,
        )
        angle = torch.tensor(
            primitive.yaw_deg * torch.pi / 180.0,
            device=device,
            dtype=dtype,
        )
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine)),
                torch.stack((sine, cosine)),
            )
        )
        center = torch.tensor(primitive.center_xy, device=device, dtype=dtype)
        return local @ rotation.T + center

    def _terrain_height_analytic(self, position_xy: Tensor) -> Tensor:
        edge_distance = torch.clamp(
            self.config.field_width_m / 2 - torch.abs(position_xy[..., 1]),
            min=0.0,
        )
        crown = edge_distance * torch.tan(
            torch.tensor(
                self.config.field_crown_slope_deg * torch.pi / 180.0,
                device=position_xy.device,
                dtype=position_xy.dtype,
            )
        )
        height = crown
        if not self.config.terrain:
            return height
        centers, sizes, cosine, sine, start, end = self._terrain_tensors(
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        for index, primitive in enumerate(self.config.terrain):
            if primitive.category == "tunnel":
                continue
            offset = position_xy - centers[index]
            local_x = cosine[index] * offset[..., 0] - sine[index] * offset[..., 1]
            local_y = sine[index] * offset[..., 0] + cosine[index] * offset[..., 1]
            inside = (torch.abs(local_x) <= sizes[index, 0] / 2) & (
                torch.abs(local_y) <= sizes[index, 1] / 2
            )
            fraction = torch.clamp(
                local_x / sizes[index, 0] + 0.5,
                min=0.0,
                max=1.0,
            )
            elevation = start[index] + fraction * (end[index] - start[index])
            height = torch.where(
                inside,
                torch.maximum(height, crown + elevation),
                height,
            )
        return height

    def _terrain_height_grid(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (str(device), dtype)
        if key not in self._height_grid_cache:
            resolution = self.config.terrain_resolution_m
            x_count = round(self.config.field_length_m / resolution) + 1
            y_count = round(self.config.field_width_m / resolution) + 1
            x = torch.linspace(
                -self.config.field_length_m / 2,
                self.config.field_length_m / 2,
                x_count,
                device=device,
                dtype=dtype,
            )
            y = torch.linspace(
                -self.config.field_width_m / 2,
                self.config.field_width_m / 2,
                y_count,
                device=device,
                dtype=dtype,
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            self._height_grid_cache[key] = self._terrain_height_analytic(
                torch.stack((xx, yy), dim=-1)
            )
        return self._height_grid_cache[key]

    def terrain_height(self, position_xy: Tensor) -> Tensor:
        """Return the 2.5D surface height from a cached 50mm lookup grid."""

        resolution = self.config.terrain_resolution_m
        if resolution <= 0:
            raise ValueError("terrain_resolution_m must be positive")
        grid = self._terrain_height_grid(
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        x_index = torch.round(
            (position_xy[..., 0] / self.config.field_length_m + 0.5) * (grid.shape[1] - 1)
        ).to(torch.long)
        y_index = torch.round(
            (position_xy[..., 1] / self.config.field_width_m + 0.5) * (grid.shape[0] - 1)
        ).to(torch.long)
        x_index.clamp_(min=0, max=grid.shape[1] - 1)
        y_index.clamp_(min=0, max=grid.shape[0] - 1)
        return grid[y_index, x_index]

    def zone_occupancy(self, position_xy: Tensor) -> Tensor:
        if position_xy.shape[-2:] != (constants.UNIT_COUNT, 2):
            raise ValueError("position_xy must have shape [env, unit, 2]")
        device = position_xy.device
        dtype = position_xy.dtype
        # Per-team centers. Neutral central high ground is intentionally shared.
        red = torch.tensor(
            (
                (-11.5, -5.2),  # supply
                (-12.3, 0.0),  # base
                (0.0, 0.0),  # central high
                (-8.0, 4.8),  # trapezoid high
                RED_OUTPOST_CENTER_XY,  # outpost (figure 4-5)
                (-3.8, -4.2),  # own fortress
                (3.8, 4.2),  # enemy fortress
                (-0.9, -1.8),  # assembly
            ),
            device=device,
            dtype=dtype,
        )
        blue = -red
        blue[Zone.CENTRAL_HIGH] = red[Zone.CENTRAL_HIGH]
        centers = torch.stack((red, blue), dim=0)
        half_extent = torch.tensor(
            (
                (1.3, 1.2),
                (1.3, 1.3),
                (2.6, 2.0),
                (1.3, 1.2),
                (1.2, 1.2),
                (1.0, 1.0),
                (1.0, 1.0),
                (1.1, 1.0),
            ),
            device=device,
            dtype=dtype,
        )
        teams = unit_teams(device)
        unit_centers = centers[teams]
        delta = torch.abs(position_xy[:, :, None, :] - unit_centers[None, :, :, :])
        return (delta <= half_extent[None, None, :, :]).all(dim=-1)

    def terrain_contacts(self, position_xy: Tensor) -> Tensor:
        device = position_xy.device
        dtype = position_xy.dtype
        red_finish = torch.tensor(
            (
                (-3.0, -5.8),  # road
                (-0.5, 3.5),  # high ground
                (-5.0, 5.8),  # fly ramp
                (-1.5, -4.8),  # tunnel
            ),
            device=device,
            dtype=dtype,
        )
        blue_finish = -red_finish
        centers = torch.stack((red_finish, blue_finish), dim=0)
        teams = unit_teams(device)
        delta = torch.abs(position_xy[:, :, None, :] - centers[teams][None, :, :, :])
        half_extent = torch.tensor(
            ((0.5, 0.5), (0.7, 0.7), (0.6, 0.6), (0.5, 0.5)),
            device=device,
            dtype=dtype,
        )
        return (delta <= half_extent[None, None, :, :]).all(dim=-1)

    def hero_deploy_zone(self, position_xy: Tensor) -> Tensor:
        teams = unit_teams(position_xy.device)
        signed_x = torch.where(
            teams[None, :] == 0,
            position_xy[..., 0],
            -position_xy[..., 0],
        )
        return signed_x <= -8.5

    def line_of_sight(self, start_xy: Tensor, end_xy: Tensor) -> Tensor:
        """Return whether segments avoid every static obstacle."""

        if start_xy.shape != end_xy.shape or start_xy.shape[-1] != 2:
            raise ValueError("LOS endpoints must have matching [..., 2] shapes")
        boxes = self.obstacles(device=start_xy.device, dtype=start_xy.dtype)
        bounds_min = boxes[:, (0, 2)]
        bounds_max = boxes[:, (1, 3)]
        direction = end_xy - start_xy
        safe_direction = torch.where(
            torch.abs(direction) < 1.0e-9,
            torch.full_like(direction, 1.0e-9),
            direction,
        )
        inverse = 1.0 / safe_direction
        t1 = (bounds_min - start_xy[..., None, :]) * inverse[..., None, :]
        t2 = (bounds_max - start_xy[..., None, :]) * inverse[..., None, :]
        t_min = torch.minimum(t1, t2).amax(dim=-1)
        t_max = torch.maximum(t1, t2).amin(dim=-1)
        intersects = (
            (t_max >= torch.maximum(t_min, torch.zeros_like(t_min)))
            & (t_min <= 1.0)
            & (t_max >= 0.0)
        )
        return ~intersects.any(dim=-1)

    def project_out_of_obstacles(self, position_xy: Tensor, radius: float) -> Tensor:
        boxes = self.obstacles(device=position_xy.device, dtype=position_xy.dtype)
        lower = boxes[:, (0, 2)] - radius
        upper = boxes[:, (1, 3)] + radius
        point = position_xy[..., None, :]
        inside = ((point >= lower) & (point <= upper)).all(dim=-1)
        left = point[..., 0] - lower[:, 0]
        right = upper[:, 0] - point[..., 0]
        bottom = point[..., 1] - lower[:, 1]
        top = upper[:, 1] - point[..., 1]
        penetrations = torch.stack((left, right, bottom, top), dim=-1)
        penetrations = torch.where(
            inside[..., None],
            penetrations,
            torch.full_like(penetrations, torch.inf),
        )
        side = penetrations.reshape(*position_xy.shape[:-1], -1).argmin(dim=-1)
        obstacle = torch.div(side, 4, rounding_mode="floor")
        local_side = side % 4
        selected_lower = lower[obstacle]
        selected_upper = upper[obstacle]
        projected = position_xy.clone()
        projected[..., 0] = torch.where(
            local_side == 0,
            selected_lower[..., 0],
            torch.where(local_side == 1, selected_upper[..., 0], projected[..., 0]),
        )
        projected[..., 1] = torch.where(
            local_side == 2,
            selected_lower[..., 1],
            torch.where(local_side == 3, selected_upper[..., 1], projected[..., 1]),
        )
        any_inside = inside.any(dim=-1)
        return torch.where(any_inside[..., None], projected, position_xy)

    def signed_distance(self, position_xy: Tensor) -> Tensor:
        """Positive free-space clearance, negative inside static geometry."""

        boxes = self.obstacles(device=position_xy.device, dtype=position_xy.dtype)
        center = 0.5 * (boxes[:, (0, 2)] + boxes[:, (1, 3)])
        half = 0.5 * (boxes[:, (1, 3)] - boxes[:, (0, 2)])
        q = torch.abs(position_xy[..., None, :] - center) - half
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0), dim=-1)
        inside = torch.clamp(q.amax(dim=-1), max=0)
        obstacle_sdf = outside + inside
        boundary = torch.minimum(
            self.config.field_length_m / 2 - torch.abs(position_xy[..., 0]),
            self.config.field_width_m / 2 - torch.abs(position_xy[..., 1]),
        )
        return torch.minimum(boundary, obstacle_sdf.amin(dim=-1))

    def occupancy_grid(
        self,
        resolution_m: float = 0.1,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        x = torch.arange(
            -self.config.field_length_m / 2,
            self.config.field_length_m / 2 + resolution_m / 2,
            resolution_m,
            device=device,
            dtype=dtype,
        )
        y = torch.arange(
            -self.config.field_width_m / 2,
            self.config.field_width_m / 2 + resolution_m / 2,
            resolution_m,
            device=device,
            dtype=dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx, yy), dim=-1)
        return x, y, self.signed_distance(points) <= 0

    def movable_mask(self, game: GameState) -> Tensor:
        roles = unit_roles(game.device)
        return (
            (roles != Role.BASE) & (roles != Role.OUTPOST) & game.alive & ~game.controller_offline
        )
