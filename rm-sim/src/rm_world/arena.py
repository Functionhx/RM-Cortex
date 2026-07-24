"""Analytic Phase 1 arena primitives, zones, collision, and LOS."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Zone, unit_roles, unit_teams
from rm_referee.state import GameState


# Manual V1.5.0 p. 34, figure 4-5. Coordinates use the field center as the
# origin, +x from red to blue, and +y toward the red aerial pad. Paired
# features rotate by 180 degrees.
RED_BASE_CENTER_XY = (-11.593, 0.0)
BLUE_BASE_CENTER_XY = (11.593, 0.0)
RED_OUTPOST_CENTER_XY = (-3.008, -3.857)
BLUE_OUTPOST_CENTER_XY = (3.008, 3.857)
RED_FORTRESS_CENTER_XY = (-7.400, 0.0)
BLUE_FORTRESS_CENTER_XY = (7.400, 0.0)

# Figure 4-5 does not separately dimension these centers. They are digitized
# from the dimensioned plan and therefore remain explicit [SIM] placements.
RED_AERIAL_PAD_CENTER_XY = (-12.52, 5.68)
BLUE_AERIAL_PAD_CENTER_XY = (12.52, -5.68)
RED_SUPPLY_CENTER_XY = (-11.55, -5.65)
BLUE_SUPPLY_CENTER_XY = (11.55, 5.65)

# Manual V1.5.0 p. 63: the cable stop is approximately 14m from the team's
# short edge and the elastic safety tether is 2.4m long. The y limits are a
# Phase 1 [SIM] envelope for the pad/road airspace described by section 4.5
# because the manual does not publish a closed airspace boundary polygon.
AERIAL_FORWARD_LIMIT_M = 2.4
AERIAL_CORRIDOR_MIN_Y_M = 3.0
AERIAL_CORRIDOR_MAX_Y_M = 7.1


def _center_symmetric(
    footprint: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Rotate a polygon by 180 degrees while preserving winding."""

    return tuple((-x, -y) for x, y in reversed(footprint))


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
    footprint_xy: tuple[tuple[float, float], ...] = ()


_CENTRAL_HIGHLAND_FOOTPRINT = (
    (-3.85, -5.41),
    (1.50, -5.41),
    (3.85, -2.20),
    (3.85, 5.41),
    (-1.50, 5.41),
    (-3.85, 2.20),
)
_RED_ASSEMBLY_FOOTPRINT = (
    (-1.90, 0.20),
    (-1.65, 1.55),
    (-0.55, 1.30),
    (0.15, 0.55),
    (0.10, -0.35),
    (-0.85, -0.20),
)
_RED_TRAPEZOID_FOOTPRINT = (
    (-10.80, 7.40),
    (-0.30, 7.40),
    (-0.30, 6.40),
    (-4.09, 6.40),
    (-6.36, 3.02),
    (-10.80, 3.02),
)
_RED_ROAD_FOOTPRINT = (
    (-10.10, -7.40),
    (-1.20, -7.40),
    (-1.20, -6.15),
    (-2.60, -6.15),
    (-3.80, -4.15),
    (-4.85, -3.75),
    (-10.10, -3.75),
)
_RED_FORTRESS_FOOTPRINT = (
    (-8.52, 0.00),
    (-7.96, -0.9695),
    (-6.84, -0.9695),
    (-6.28, 0.00),
    (-6.84, 0.9695),
    (-7.96, 0.9695),
)


DEFAULT_TERRAIN: tuple[TerrainPrimitive, ...] = (
    # Public-road regions are at field height. Their plan footprint follows
    # figures 4-5 and 4-34; raised subfeatures are represented below.
    TerrainPrimitive(
        "red_road",
        (-5.65, -5.575),
        (8.901, 3.651),
        0.0,
        0.0,
        category="road",
        team=0,
        footprint_xy=_RED_ROAD_FOOTPRINT,
    ),
    TerrainPrimitive(
        "blue_road",
        (5.65, 5.575),
        (8.901, 3.651),
        0.0,
        0.0,
        yaw_deg=180.0,
        category="road",
        team=1,
        footprint_xy=_center_symmetric(_RED_ROAD_FOOTPRINT),
    ),
    # Figure 4-26 gives an irregular 10.505m x 4.380m footprint, 200--400mm
    # height, and 23/43 degree faces. Phase 1 uses the 300mm median surface.
    TerrainPrimitive(
        "red_trapezoid_highland",
        (-5.55, 5.21),
        (10.505, 4.380),
        0.30,
        0.30,
        category="trapezoid",
        team=0,
        footprint_xy=_RED_TRAPEZOID_FOOTPRINT,
    ),
    TerrainPrimitive(
        "blue_trapezoid_highland",
        (5.55, -5.21),
        (10.505, 4.380),
        0.30,
        0.30,
        yaw_deg=180.0,
        category="trapezoid",
        team=1,
        footprint_xy=_center_symmetric(_RED_TRAPEZOID_FOOTPRINT),
    ),
    # Figure 4-27 publishes the 7.700m x 10.820m envelope. Its skewed,
    # center-symmetric outline replaces the old rectangular approximation.
    TerrainPrimitive(
        "central_highland",
        (0.0, 0.0),
        (7.7, 10.82),
        0.35,
        0.35,
        category="central",
        footprint_xy=_CENTRAL_HIGHLAND_FOOTPRINT,
    ),
    # Figures 4-28 and 4-29 place the two assembly areas below the energy
    # mechanism at field center. They are overlays on the central highland.
    TerrainPrimitive(
        "red_assembly",
        (-0.875, 0.60),
        (2.05, 1.90),
        0.35,
        0.35,
        category="assembly",
        team=0,
        footprint_xy=_RED_ASSEMBLY_FOOTPRINT,
    ),
    TerrainPrimitive(
        "blue_assembly",
        (0.875, -0.60),
        (2.05, 1.90),
        0.35,
        0.35,
        yaw_deg=180.0,
        category="assembly",
        team=1,
        footprint_xy=_center_symmetric(_RED_ASSEMBLY_FOOTPRINT),
    ),
    # Figure 4-37: 17 degree face, 1.145m horizontal run, 0.860m width,
    # and 0.203m/0.553m edge elevations. The separate 0.650m value is the
    # flight gap and must not be added to the ramp footprint.
    TerrainPrimitive(
        "red_fly_ramp",
        (-0.79, -6.90),
        (1.145, 0.860),
        0.203,
        0.553,
        category="fly_ramp",
        team=0,
    ),
    TerrainPrimitive(
        "blue_fly_ramp",
        (0.79, 6.90),
        (1.145, 0.860),
        0.203,
        0.553,
        yaw_deg=180.0,
        category="fly_ramp",
        team=1,
    ),
    TerrainPrimitive(
        "red_rough_road",
        (-7.60, -6.25),
        (2.560, 1.450),
        0.0,
        0.0,
        category="rough",
        team=0,
    ),
    TerrainPrimitive(
        "blue_rough_road",
        (7.60, 6.25),
        (2.560, 1.450),
        0.0,
        0.0,
        yaw_deg=180.0,
        category="rough",
        team=1,
    ),
    # Figure 4-25: a regular hexagonal 20 degree platform with 1.120m side,
    # 1.939m height across flats, and a 150mm top elevation.
    TerrainPrimitive(
        "red_fortress",
        RED_FORTRESS_CENTER_XY,
        (2.240, 1.939),
        0.15,
        0.15,
        category="fortress",
        team=0,
        footprint_xy=_RED_FORTRESS_FOOTPRINT,
    ),
    TerrainPrimitive(
        "blue_fortress",
        BLUE_FORTRESS_CENTER_XY,
        (2.240, 1.939),
        0.15,
        0.15,
        yaw_deg=180.0,
        category="fortress",
        team=1,
        footprint_xy=_center_symmetric(_RED_FORTRESS_FOOTPRINT),
    ),
    # Figure 4-34 publishes the 0.700m opening and 0.800m roof width but not
    # a standalone global center dimension. The plan placement remains [SIM].
    TerrainPrimitive(
        "red_tunnel",
        (-4.05, -4.65),
        (1.6, 0.8),
        0.0,
        0.0,
        yaw_deg=35.0,
        category="tunnel",
        team=0,
    ),
    TerrainPrimitive(
        "blue_tunnel",
        (4.05, 4.65),
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
        (-4.05, -3.72, -2.30, -0.55),
        (-4.05, -3.72, 0.55, 2.30),
        (3.72, 4.05, -2.30, -0.55),
        (3.72, 4.05, 0.55, 2.30),
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

    def base_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the figure 4-5 base centers in red/blue team order."""

        return torch.tensor(
            (RED_BASE_CENTER_XY, BLUE_BASE_CENTER_XY),
            device=device,
            dtype=dtype,
        )

    def fortress_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the figure 4-5 fortress centers in red/blue team order."""

        return torch.tensor(
            (RED_FORTRESS_CENTER_XY, BLUE_FORTRESS_CENTER_XY),
            device=device,
            dtype=dtype,
        )

    def supply_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return diagram-derived ``[SIM]`` supply centers."""

        return torch.tensor(
            (RED_SUPPLY_CENTER_XY, BLUE_SUPPLY_CENTER_XY),
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

    def aerial_flight_area(
        self,
        position_xy: Tensor,
        team: int | Tensor,
    ) -> Tensor:
        """Return whether positions lie in the Phase 1 section 4.5 airspace.

        The manual defines the allowed surfaces and safety-tether distances but
        does not publish a closed flight polygon. This rectangular corridor is
        therefore marked ``[SIM]`` and deliberately stays on the team's
        pad/road side of the field.
        """

        if position_xy.shape[-1] != 2:
            raise ValueError("position_xy must have shape [..., 2]")
        team_tensor = torch.as_tensor(team, device=position_xy.device)
        sign = torch.where(
            team_tensor == 0,
            torch.ones_like(team_tensor, dtype=position_xy.dtype),
            -torch.ones_like(team_tensor, dtype=position_xy.dtype),
        )
        local = position_xy * sign[..., None]
        return (
            (local[..., 0] >= -self.config.field_length_m / 2)
            & (local[..., 0] <= AERIAL_FORWARD_LIMIT_M)
            & (local[..., 1] >= AERIAL_CORRIDOR_MIN_Y_M)
            & (local[..., 1] <= AERIAL_CORRIDOR_MAX_Y_M)
        )

    def project_to_aerial_flight_area(
        self,
        position_xy: Tensor,
        team: int | Tensor,
        *,
        margin_m: float = 0.0,
    ) -> Tensor:
        """Clamp aerial positions to the Phase 1 section 4.5 corridor."""

        if margin_m < 0:
            raise ValueError("margin_m must be non-negative")
        team_tensor = torch.as_tensor(team, device=position_xy.device)
        sign = torch.where(
            team_tensor == 0,
            torch.ones_like(team_tensor, dtype=position_xy.dtype),
            -torch.ones_like(team_tensor, dtype=position_xy.dtype),
        )
        local = position_xy * sign[..., None]
        local_x = torch.clamp(
            local[..., 0],
            min=-self.config.field_length_m / 2 + margin_m,
            max=AERIAL_FORWARD_LIMIT_M,
        )
        local_y = torch.clamp(
            local[..., 1],
            min=AERIAL_CORRIDOR_MIN_Y_M,
            max=AERIAL_CORRIDOR_MAX_Y_M,
        )
        return torch.stack((local_x, local_y), dim=-1) * sign[..., None]

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
                (-10.5, 1.3),
                (-12.0, -2.0),
                (-10.0, -3.0),
                (-10.0, 3.0),
                RED_AERIAL_PAD_CENTER_XY,
                (-9.0, 0.0),
                RED_BASE_CENTER_XY,
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
        if primitive.footprint_xy:
            return torch.tensor(
                primitive.footprint_xy,
                device=device,
                dtype=dtype,
            )
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

    @staticmethod
    def _points_in_polygon(position_xy: Tensor, polygon_xy: Tensor) -> Tensor:
        """Vectorized even-odd point-in-polygon test for an arbitrary batch."""

        inside = torch.zeros(
            position_xy.shape[:-1],
            device=position_xy.device,
            dtype=torch.bool,
        )
        x = position_xy[..., 0]
        y = position_xy[..., 1]
        previous = polygon_xy[-1]
        epsilon = torch.finfo(position_xy.dtype).eps
        for current in polygon_xy:
            crosses_y = (current[1] > y) != (previous[1] > y)
            denominator = previous[1] - current[1]
            denominator = torch.where(
                torch.abs(denominator) < epsilon,
                torch.full_like(denominator, epsilon),
                denominator,
            )
            edge_x = (previous[0] - current[0]) * (y - current[1]) / denominator + current[0]
            inside ^= crosses_y & (x < edge_x)
            previous = current
        return inside

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
            if primitive.footprint_xy:
                polygon = self.terrain_corners(
                    primitive,
                    device=position_xy.device,
                    dtype=position_xy.dtype,
                )
                inside = self._points_in_polygon(position_xy, polygon)
            else:
                inside = (torch.abs(local_x) <= sizes[index, 0] / 2) & (
                    torch.abs(local_y) <= sizes[index, 1] / 2
                )
            fraction = torch.clamp(
                local_x / sizes[index, 0] + 0.5,
                min=0.0,
                max=1.0,
            )
            elevation = start[index] + fraction * (end[index] - start[index])
            if primitive.category == "rough":
                # Figure 4-36 specifies 70mm bumps at 240mm pitch. A cosine
                # profile is the Phase 1 differentiable [SIM] approximation.
                elevation = elevation + 0.035 * (1.0 + torch.cos(2.0 * math.pi * local_x / 0.240))
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
                RED_SUPPLY_CENTER_XY,  # supply
                RED_BASE_CENTER_XY,  # base
                (0.0, 0.0),  # central high
                (-9.6, 4.2),  # trapezoid-high gain point [SIM]
                RED_OUTPOST_CENTER_XY,  # outpost (figure 4-5)
                RED_FORTRESS_CENTER_XY,  # own fortress
                BLUE_FORTRESS_CENTER_XY,  # enemy fortress
                (-0.875, 0.60),  # assembly (figures 4-28 and 4-29)
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
                (1.1, 1.0),
                (2.6, 2.0),
                (1.3, 1.2),
                (1.2, 1.2),
                (1.12, 0.97),
                (1.12, 0.97),
                (1.2, 1.1),
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
                (-2.0, -6.7),  # road [SIM]
                (2.8, 4.4),  # high ground [SIM]
                (-0.1, -6.9),  # fly ramp [SIM]
                (-3.6, -4.3),  # tunnel [SIM]
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
