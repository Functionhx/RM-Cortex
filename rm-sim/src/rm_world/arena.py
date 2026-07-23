"""Analytic Phase 1 arena primitives, zones, collision, and LOS."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role, Zone, unit_roles, unit_teams
from rm_referee.state import GameState


@dataclass(frozen=True)
class ArenaConfig:
    """Explicit ``[SIM]`` coordinate and simplified-geometry choices."""

    field_length_m: float = 28.0
    field_width_m: float = 15.0
    robot_radius_m: float = 0.35
    # xmin, xmax, ymin, ymax. These conservative blocks approximate elevated
    # structures whose exact local collision drawings are not available.
    obstacles: tuple[tuple[float, float, float, float], ...] = (
        (-2.0, 2.0, -0.8, 0.8),
        (-8.2, -6.2, 2.2, 4.4),
        (6.2, 8.2, -4.4, -2.2),
    )


class ArenaGeometry:
    """Vectorized geometry shared by rollout, debugging, and the Isaac bridge."""

    def __init__(self, config: ArenaConfig | None = None) -> None:
        self.config = config or ArenaConfig()

    def obstacles(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.tensor(self.config.obstacles, device=device, dtype=dtype)

    def spawn_positions(self, game: GameState) -> Tensor:
        red = torch.tensor(
            (
                (-11.0, 0.0),
                (-12.0, -2.0),
                (-10.0, -3.0),
                (-10.0, 3.0),
                (-13.0, 5.0),
                (-9.0, 0.0),
                (-12.5, 0.0),
                (-6.5, 0.0),
            ),
            device=game.device,
            dtype=game.dtype,
        )
        blue = red.clone()
        blue[:, 0].neg_()
        positions = torch.cat((red, blue), dim=0)
        return positions.unsqueeze(0).expand(game.num_envs, -1, -1).clone()

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
                (0.0, 2.2),  # central high
                (-8.0, 4.8),  # trapezoid high
                (-6.5, 0.0),  # outpost
                (-3.8, -4.2),  # own fortress
                (3.8, -4.2),  # enemy fortress
                (-0.9, -1.8),  # assembly
            ),
            device=device,
            dtype=dtype,
        )
        blue = red.clone()
        blue[:, 0].neg_()
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
        blue_finish = red_finish.clone()
        blue_finish[:, 0].neg_()
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
