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

# Published physical dimensions used by simulation diagnostics and
# visualization. These are not collision radii or referee-zone dimensions.
# Figure 4-9 gives the base core as 1609mm front-to-back by 1881mm laterally.
# Its axial dimension chain locates the rear edge and widest shoulders relative
# to the figure 4-5 rotation axis. The unlabelled front half-width is digitized,
# and peripheral side details are omitted from this core approximation.
BASE_PEDESTAL_SIZE_XY_M = (1.609, 1.881)
BASE_PEDESTAL_FRONT_HALF_WIDTH_M = 0.390  # [DIGITIZED], figure 4-9
RED_BASE_PEDESTAL_FOOTPRINT_XY_M = (
    (-12.345, 0.5445),
    (-11.680, 0.9405),
    (-10.736, BASE_PEDESTAL_FRONT_HALF_WIDTH_M),
    (-10.736, -BASE_PEDESTAL_FRONT_HALF_WIDTH_M),
    (-11.680, -0.9405),
    (-12.345, -0.5445),
)
# Figure 4-33 publishes a 550mm circular sweep envelope for the three rotating
# armor modules, not a solid-body diameter. It gives only one 650mm pedestal
# width in elevation, which must not be treated as a 650x650mm footprint.
OUTPOST_ARMOR_SWEEP_DIAMETER_M = 0.550
OUTPOST_PEDESTAL_WIDTH_M = 0.650
# Figure 4-16 publishes these frame reference spans. It does not establish a
# closed rectangular envelope, fully dimension the red platform octagon, or
# locate that platform inside the asymmetric outer structure. In particular,
# 1334mm labels the surrounding long frame members, not an octagon edge.
AERIAL_PAD_FRAME_REFERENCE_SPANS_XY_M = (2.200, 2.858)
AERIAL_PAD_INNER_FRAME_SPAN_XY_M = (2.149, 2.200)
AERIAL_PAD_FRAME_MEMBER_LENGTH_M = 1.334
# The platform outline below is intentionally visualization-only until an
# official closed coordinate chain is published.
AERIAL_PAD_PLATFORM_VISUAL_SIZE_XY_M = (1.850, 1.900)

# Figure 4-5 does not separately dimension these centers. They are digitized
# from the dimensioned plan and therefore remain explicit [SIM] placements.
RED_AERIAL_PAD_CENTER_XY = (-12.52, 5.68)
BLUE_AERIAL_PAD_CENTER_XY = (12.52, -5.68)
RED_SUPPLY_CENTER_XY = (-11.55, -5.65)
BLUE_SUPPLY_CENTER_XY = (11.55, 5.65)

# Figure 4-8 dimension chain, expressed in the figure 4-5 world frame. This
# is the pre-match starting area. Figure 5-24 only gives the base RFID region
# topology, so Phase 1 attaches that topology to this dimensioned footprint
# rather than inventing an unrelated rectangle.
RED_STARTING_ZONE_FOOTPRINT_XY_M = (
    (-11.792, 1.9530),
    (-9.815, 0.8115),
    (-9.815, -0.8115),
    (-11.792, -1.9530),
    (-13.170, -1.1925),
    (-13.170, 1.1925),
)

# Manual V1.5.0 p. 63: the cable stop is approximately 14m from the team's
# short edge and the elastic safety tether is 2.4m long. The y limits are a
# Phase 1 [SIM] envelope for the pad/road airspace described by section 4.5
# because the manual does not publish a closed airspace boundary polygon.
AERIAL_FORWARD_LIMIT_M = 2.4
AERIAL_CORRIDOR_MIN_Y_M = 3.0
AERIAL_CORRIDOR_MAX_Y_M = 7.1

# Phase 1 [SIM] axis-aligned rule-region envelopes, ordered by ``Zone``.
# Figure 5-24 gives topology but not numerical RFID boundaries. Keep these
# explicitly separate from the physical/reference footprints below.
ZONE_HALF_EXTENTS_XY_M: tuple[tuple[float, float], ...] = (
    (1.30, 1.20),  # supply
    (1.10, 1.00),  # base
    (2.60, 2.00),  # central high
    (1.30, 1.20),  # trapezoid-high gain point
    (1.20, 1.20),  # outpost
    (1.12, 0.97),  # own fortress
    (1.12, 0.97),  # enemy fortress
    (1.20, 1.10),  # assembly
)


def _center_symmetric(
    footprint: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Rotate a polygon by 180 degrees while preserving winding."""

    return tuple((-x, -y) for x, y in footprint)


@dataclass(frozen=True)
class TerrainPrimitive:
    """Rule-manual terrain surface with explicit geometry provenance.

    Values called out by rule-manual figures 4-5 and 4-25--4-37 use their
    published nominal dimensions. Plan-calibrated values are ``[DIGITIZED]``,
    constrained closures are ``[INFERRED]``, unresolved mappings are
    ``[AMB]``, and model choices are ``[SIM]``. Elevation normally changes
    along the primitive's local x-axis. Polygonal slope patches can instead
    provide one elevation per footprint vertex.
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
    vertex_elevations_m: tuple[float, ...] = ()


def _scale_polygon(
    footprint: tuple[tuple[float, float], ...],
    center_xy: tuple[float, float],
    scale: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            center_xy[0] + (x - center_xy[0]) * scale,
            center_xy[1] + (y - center_xy[1]) * scale,
        )
        for x, y in footprint
    )


def _inset_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    distance_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Move a counter-clockwise polygon edge toward its interior."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    normal = (-dy / length, dx / length)
    offset = (normal[0] * distance_m, normal[1] * distance_m)
    return (
        (start[0] + offset[0], start[1] + offset[1]),
        (end[0] + offset[0], end[1] + offset[1]),
    )


def _mirror_surface(
    primitive: TerrainPrimitive,
    *,
    name: str,
    team: int,
) -> TerrainPrimitive:
    return TerrainPrimitive(
        name=name,
        center_xy=(-primitive.center_xy[0], -primitive.center_xy[1]),
        size_xy=primitive.size_xy,
        elevation_start_m=primitive.elevation_start_m,
        elevation_end_m=primitive.elevation_end_m,
        yaw_deg=(primitive.yaw_deg + 180.0) % 360.0,
        category=primitive.category,
        team=team,
        footprint_xy=_center_symmetric(primitive.footprint_xy),
        vertex_elevations_m=primitive.vertex_elevations_m,
    )


def _paired_surfaces(
    red: TerrainPrimitive,
    *,
    blue_name: str,
) -> tuple[TerrainPrimitive, TerrainPrimitive]:
    return red, _mirror_surface(red, name=blue_name, team=1)


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Return the signed area of a simple 2D polygon."""

    return 0.5 * sum(
        x_0 * y_1 - x_1 * y_0
        for (x_0, y_0), (x_1, y_1) in zip(points, points[1:] + points[:1], strict=True)
    )


def _point_in_triangle(
    point: tuple[float, float],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    signs = []
    for start, end in zip(triangle, triangle[1:] + triangle[:1], strict=True):
        signs.append(
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )
    return min(signs) >= -1.0e-9


def triangulate_polygon(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Triangulate a simple polygon with deterministic ear clipping."""

    if len(points) < 3:
        raise ValueError("terrain footprint must have at least three points")
    remaining = list(range(len(points)))
    if polygon_area(points) < 0:
        remaining.reverse()
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        clipped = False
        for cursor, current in enumerate(remaining):
            previous = remaining[cursor - 1]
            following = remaining[(cursor + 1) % len(remaining)]
            a = points[previous]
            b = points[current]
            c = points[following]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross <= 1.0e-9:
                continue
            triangle = (a, b, c)
            if any(
                _point_in_triangle(points[index], triangle)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[cursor]
            clipped = True
            break
        if not clipped:
            raise ValueError("terrain footprint is not a simple polygon")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return tuple(triangles)


# Figure 4-27 directly publishes the 7.700m by 10.820m control bounds. The
# two unlabelled chamfer intercepts are calibrated from the plan and therefore
# remain [DIGITIZED], rather than being presented as official dimensions.
_CENTRAL_HIGHLAND_CHAMFER_X_M = 1.500
_CENTRAL_HIGHLAND_CHAMFER_Y_M = 2.200
_CENTRAL_HIGHLAND_FOOTPRINT = (
    (-3.85, -5.41),
    (_CENTRAL_HIGHLAND_CHAMFER_X_M, -5.41),
    (3.85, -_CENTRAL_HIGHLAND_CHAMFER_Y_M),
    (3.85, 5.41),
    (-_CENTRAL_HIGHLAND_CHAMFER_X_M, 5.41),
    (-3.85, _CENTRAL_HIGHLAND_CHAMFER_Y_M),
)
# Figures 4-28 and 4-29 publish the four red boundary segments and their
# angles. Their global axis and closure along the central structure are not
# numerically dimensioned. These vertices preserve the nominal chain, then
# use the plan-calibrated center/axis and one [INFERRED] closing segment.
RED_ASSEMBLY_REFERENCE_FOOTPRINT_XY_M = (
    (-1.158148, -0.735265),
    (-1.722633, 0.346288),
    (-1.029132, 1.575099),
    (-0.652619, 1.645296),
    (0.126735, 1.026723),
)
_ASSEMBLY_EDGE_CHAIN_M = (1.220, 1.411, 0.383, 0.995)
_ASSEMBLY_INTERNAL_ANGLES_DEG = (123.0, 130.0, 131.0)
_ASSEMBLY_SLOPE_ANGLES_DEG = (12.0, 14.0, 45.0, 15.0)
_ASSEMBLY_LOW_HEIGHT_M = 0.100
_ASSEMBLY_HIGH_HEIGHT_M = 0.200

# Manual V1.5.0 figure 4-26, printed p.53 (PDF p.54). The 10.805m by
# 4.380m values are control spans across guard strips, deck edges, and slope
# feet; they are not a ready-made closed terrain polygon. The points below
# follow the labelled axial stations while retaining a [DIGITIZED] global
# transform. The station chain and illustrated 125-degree corner are mildly
# overconstrained, so the intervening closure remains [INFERRED].
_TRAPEZOID_OUTER_STRIP_M = 0.300
_TRAPEZOID_SIZE_XY_M = (0.300 + 6.707 + 3.798, 4.380)
_TRAPEZOID_INNER_TOP_CHAIN_M = 6.707
_TRAPEZOID_DIAGONAL_FOOT_M = 4.440
_TRAPEZOID_STEP_M = 1.003
_RED_TRAPEZOID_DATUM_X_M = -10.805
_RED_TRAPEZOID_MIN_X_M = _RED_TRAPEZOID_DATUM_X_M - _TRAPEZOID_OUTER_STRIP_M
_RED_TRAPEZOID_MAX_Y_M = 7.400
_RED_TRAPEZOID_MAX_X_M = _RED_TRAPEZOID_DATUM_X_M + 6.707 + 3.798
_RED_TRAPEZOID_MIN_Y_M = _RED_TRAPEZOID_MAX_Y_M - _TRAPEZOID_SIZE_XY_M[1]
_RED_TRAPEZOID_FOOTPRINT = (
    (_RED_TRAPEZOID_DATUM_X_M, _RED_TRAPEZOID_MAX_Y_M),
    (_RED_TRAPEZOID_MAX_X_M, _RED_TRAPEZOID_MAX_Y_M),
    (_RED_TRAPEZOID_MAX_X_M, _RED_TRAPEZOID_MAX_Y_M - _TRAPEZOID_STEP_M),
    (
        _RED_TRAPEZOID_DATUM_X_M + _TRAPEZOID_INNER_TOP_CHAIN_M,
        _RED_TRAPEZOID_MAX_Y_M - _TRAPEZOID_STEP_M,
    ),
    (
        _RED_TRAPEZOID_DATUM_X_M + _TRAPEZOID_DIAGONAL_FOOT_M,
        _RED_TRAPEZOID_MIN_Y_M,
    ),
    (_RED_TRAPEZOID_DATUM_X_M + 2.361, _RED_TRAPEZOID_MIN_Y_M),
    (_RED_TRAPEZOID_DATUM_X_M + 2.626, _RED_TRAPEZOID_MAX_Y_M - 2.922),
    (_RED_TRAPEZOID_DATUM_X_M, _RED_TRAPEZOID_MAX_Y_M - 2.922),
)

# Manual V1.5.0 figure 4-34, printed p.60 (PDF p.61). The 8.901m and
# 1.561+2.090m dimensions define local control spans, not a complete road
# polygon. This [DIGITIZED] footprint now snaps every inferred corner to a
# published station instead of using rounded freehand coordinates.
_ROAD_SIZE_XY_M = (8.901, 1.561 + 2.090)
_RED_ROAD_MIN_X_M = -10.101
_RED_ROAD_MIN_Y_M = -7.400
_RED_ROAD_MAX_X_M = _RED_ROAD_MIN_X_M + _ROAD_SIZE_XY_M[0]
_RED_ROAD_MAX_Y_M = _RED_ROAD_MIN_Y_M + _ROAD_SIZE_XY_M[1]
RED_ROAD_CONTROL_FOOTPRINT_XY_M = (
    (_RED_ROAD_MIN_X_M, _RED_ROAD_MIN_Y_M),
    (_RED_ROAD_MAX_X_M, _RED_ROAD_MIN_Y_M),
    (_RED_ROAD_MAX_X_M, -5.780),
    (-2.709, -5.780),
    (-3.797, _RED_ROAD_MAX_Y_M),
    (-5.032, _RED_ROAD_MAX_Y_M),
    (_RED_ROAD_MIN_X_M, _RED_ROAD_MAX_Y_M),
)
_ROAD_X_STATIONS_M = (
    -10.101,
    -8.601,
    -8.243,
    -7.467,
    -6.976,
    -6.200,
    -5.032,
    -3.797,
    -2.709,
    -1.200,
)
_ROAD_Y_STATIONS_M = (-7.400, -5.780, -5.310, -3.749)
TUNNEL_REFERENCE_LENGTH_M = 1.600  # [SIM]
TUNNEL_OUTER_WIDTH_M = 0.800
_TUNNEL_OPENING_WIDTH_M = 0.700
_TUNNEL_SECTION_HEIGHTS_M = (0.250, 0.260)
RED_TUNNEL_REFERENCE_CENTER_XY = (-4.05, -4.65)  # [SIM]
RED_TUNNEL_REFERENCE_YAW_DEG = 35.0  # [SIM]

# Manual V1.5.0 figure 4-25, printed p.52 (PDF p.53).
_FORTRESS_OUTER_SIZE_XY_M = (2.240, 1.939)
_FORTRESS_INNER_SIZE_XY_M = (1.306, 1.131)
_FORTRESS_OUTER_EDGE_M = 1.120
_FORTRESS_INNER_EDGE_M = 0.653
_FORTRESS_TOP_HEIGHT_M = 0.150

# Manual V1.5.0 figures 4-35 and 4-36, printed pp.60--61 (PDF pp.61--62).
# Only the 0.070m bump height and 0.240m pitch are dimensioned. The 2.560m
# arrow in figure 4-5 belongs to the adjacent supply-side passage, not the
# rough-road strip. Its length, width, placement, and cosine profile remain
# explicit Phase 1 [SIM] choices.
_ROUGH_ROAD_LENGTH_M = 2.560
_ROUGH_ROAD_WIDTH_M = 1.450
_ROUGH_BUMP_HEIGHT_M = 0.070
_ROUGH_BUMP_PITCH_M = 0.240

# Manual V1.5.0 figure 4-37, printed p.62 (PDF p.63). The separate 0.650m
# dimension is the flight gap, not part of the ramp footprint.
_FLY_RAMP_SIZE_XY_M = (1.145, 0.860)
_FLY_RAMP_LOW_M = 0.203
_FLY_RAMP_HIGH_M = 0.553
_FLY_GAP_M = 0.650
_FLY_LANDING_EDGE_SPAN_M = 1.143
_FLY_LANDING_HEIGHT_M = 0.200
_RED_FLY_RAMP_CENTER_XY = (-0.790, -6.900)  # [DIGITIZED], figure 4-5
_RED_FLY_HIGH_EDGE_X_M = _RED_FLY_RAMP_CENTER_XY[0] + _FLY_RAMP_SIZE_XY_M[0] / 2
_RED_FLY_LANDING_EDGE_X_M = _RED_FLY_HIGH_EDGE_X_M + _FLY_GAP_M
# Figure 4-37's 1.143m value spans the landing edge transversely. The landing
# depth along the ramp axis is not dimensioned, so only this reference edge is
# encoded; no invented rectangular landing surface is added to the terrain.
RED_FLY_LANDING_REFERENCE_EDGE_XY_M = (
    (
        _RED_FLY_LANDING_EDGE_X_M,
        _RED_FLY_RAMP_CENTER_XY[1] - _FLY_LANDING_EDGE_SPAN_M / 2,
    ),
    (
        _RED_FLY_LANDING_EDGE_X_M,
        _RED_FLY_RAMP_CENTER_XY[1] + _FLY_LANDING_EDGE_SPAN_M / 2,
    ),
)
_RED_FORTRESS_FOOTPRINT = (
    (-8.52, 0.00),
    (-7.96, -0.9695),
    (-6.84, -0.9695),
    (-6.28, 0.00),
    (-6.84, 0.9695),
    (-7.96, 0.9695),
)

# Figure 4-27 (printed p.54/PDF p.55) directly dimensions a 1.000m plan band
# and separately publishes a 10.5 degree slope plus 200/300/350/400mm section
# levels. It does not uniquely map the band, angle, and levels to one surface.
# Phase 1 retains a 200-to-350mm band as [INFERRED]; its resulting 8.53 degree
# slope must not be labelled as the separate published 10.5 degree face.
_CENTRAL_CONNECTOR_RUN_M = 1.000
_CENTRAL_SECTION_LEVELS_M = (0.200, 0.300, 0.350, 0.400)
_CENTRAL_SIDE_PROFILE_LENGTHS_M = (2.972, 1.421)
_CENTRAL_LOW_RED = _CENTRAL_HIGHLAND_FOOTPRINT[1:3]
_CENTRAL_LOW_BLUE = _CENTRAL_HIGHLAND_FOOTPRINT[4:6]
_CENTRAL_HIGH_RED = _inset_edge(*_CENTRAL_LOW_RED, _CENTRAL_CONNECTOR_RUN_M)
_CENTRAL_HIGH_BLUE = _inset_edge(*_CENTRAL_LOW_BLUE, _CENTRAL_CONNECTOR_RUN_M)
_CENTRAL_PLATEAU_FOOTPRINT = (
    _CENTRAL_HIGHLAND_FOOTPRINT[0],
    _CENTRAL_HIGH_RED[0],
    _CENTRAL_HIGH_RED[1],
    _CENTRAL_HIGHLAND_FOOTPRINT[3],
    _CENTRAL_HIGH_BLUE[0],
    _CENTRAL_HIGH_BLUE[1],
)
_CENTRAL_RED_CONNECTOR_FOOTPRINT = (
    _CENTRAL_LOW_RED[0],
    _CENTRAL_LOW_RED[1],
    _CENTRAL_HIGH_RED[1],
    _CENTRAL_HIGH_RED[0],
)
_CENTRAL_BLUE_CONNECTOR_FOOTPRINT = (
    _CENTRAL_LOW_BLUE[0],
    _CENTRAL_LOW_BLUE[1],
    _CENTRAL_HIGH_BLUE[1],
    _CENTRAL_HIGH_BLUE[0],
)

# Figure 4-26 establishes a 200mm main deck, a 400mm small gain platform,
# and 23/43 degree edges down to the field. The drawing's 350/600mm arrows
# terminate on guard strips; they are not drive-surface elevations.
_TRAPEZOID_23_RUN_M = 0.200 / math.tan(math.radians(23.0))
_TRAPEZOID_43_RUN_M = 0.400 / math.tan(math.radians(43.0))


_RED_TRAPEZOID_23_RAMP = (
    (-3.706802, 6.134387),
    (-6.113831, 2.548830),
    (-8.444000, 2.548830),
    (-8.444000, 3.020000),
    (-6.365000, 3.020000),
    (-4.098000, 6.397000),
)
_RED_TRAPEZOID_400_TOP = (
    (-10.805, 4.328),
    (-8.444, 4.328),
    (-8.954, 3.128),
    (-10.805, 3.128),
)
_RED_TRAPEZOID_43_RAMP = (
    (-10.805, 3.128 - _TRAPEZOID_43_RUN_M),
    (-9.655, 3.128 - _TRAPEZOID_43_RUN_M),
    (-9.655, 3.128),
    (-10.805, 3.128),
)


def _fortress_surfaces(
    *,
    prefix: str,
    team: int,
    center_xy: tuple[float, float],
    outer: tuple[tuple[float, float], ...],
) -> tuple[TerrainPrimitive, ...]:
    inner = _scale_polygon(
        outer,
        center_xy,
        _FORTRESS_INNER_EDGE_M / _FORTRESS_OUTER_EDGE_M,
    )
    surfaces: list[TerrainPrimitive] = [
        TerrainPrimitive(
            f"{prefix}_fortress",
            center_xy,
            _FORTRESS_OUTER_SIZE_XY_M,
            0.0,
            0.0,
            category="fortress",
            team=team,
            footprint_xy=outer,
        )
    ]
    for index, (outer_start, outer_end, inner_start, inner_end) in enumerate(
        zip(
            outer,
            outer[1:] + outer[:1],
            inner,
            inner[1:] + inner[:1],
            strict=True,
        )
    ):
        surfaces.append(
            TerrainPrimitive(
                f"{prefix}_fortress_slope_{index}",
                center_xy,
                _FORTRESS_OUTER_SIZE_XY_M,
                0.0,
                _FORTRESS_TOP_HEIGHT_M,
                category="fortress_slope",
                team=team,
                footprint_xy=(outer_start, outer_end, inner_end, inner_start),
                vertex_elevations_m=(
                    0.0,
                    0.0,
                    _FORTRESS_TOP_HEIGHT_M,
                    _FORTRESS_TOP_HEIGHT_M,
                ),
            )
        )
    surfaces.append(
        TerrainPrimitive(
            f"{prefix}_fortress_top",
            center_xy,
            _FORTRESS_INNER_SIZE_XY_M,
            _FORTRESS_TOP_HEIGHT_M,
            _FORTRESS_TOP_HEIGHT_M,
            category="fortress_top",
            team=team,
            footprint_xy=inner,
        )
    )
    return tuple(surfaces)


DEFAULT_TERRAIN: tuple[TerrainPrimitive, ...] = (
    # Figure 4-34 fixes the road station grid, 11/15 degree faces, and section
    # heights, but does not uniquely map those faces into a closed world
    # surface. Its dimension-constrained plan is rendered as a dashed reference
    # and deliberately omitted here rather than simulated as a false flat deck.
    # Figure 4-26 gives a 10.805m control span containing a 10.505m main-deck
    # chain. The driving deck is 200mm; 350/600mm labels belong to guard
    # strips, not extra plateaus.
    TerrainPrimitive(
        "red_trapezoid_highland",
        (
            (_RED_TRAPEZOID_DATUM_X_M + _RED_TRAPEZOID_MAX_X_M) / 2,
            (_RED_TRAPEZOID_MIN_Y_M + _RED_TRAPEZOID_MAX_Y_M) / 2,
        ),
        (6.707 + 3.798, _TRAPEZOID_SIZE_XY_M[1]),
        0.20,
        0.20,
        category="trapezoid",
        team=0,
        footprint_xy=_RED_TRAPEZOID_FOOTPRINT,
    ),
    TerrainPrimitive(
        "blue_trapezoid_highland",
        (
            -(_RED_TRAPEZOID_DATUM_X_M + _RED_TRAPEZOID_MAX_X_M) / 2,
            -(_RED_TRAPEZOID_MIN_Y_M + _RED_TRAPEZOID_MAX_Y_M) / 2,
        ),
        (6.707 + 3.798, _TRAPEZOID_SIZE_XY_M[1]),
        0.20,
        0.20,
        yaw_deg=180.0,
        category="trapezoid",
        team=1,
        footprint_xy=_center_symmetric(_RED_TRAPEZOID_FOOTPRINT),
    ),
    *_paired_surfaces(
        TerrainPrimitive(
            "red_trapezoid_23_ramp",
            (-6.075, 4.473),
            (4.737, 3.848),
            0.0,
            0.20,
            category="trapezoid_slope",
            team=0,
            footprint_xy=_RED_TRAPEZOID_23_RAMP,
            vertex_elevations_m=(0.0, 0.0, 0.0, 0.20, 0.20, 0.20),
        ),
        blue_name="blue_trapezoid_23_ramp",
    ),
    *_paired_surfaces(
        TerrainPrimitive(
            "red_trapezoid_400_top",
            (-9.6245, 3.728),
            (2.361, 1.20),
            0.40,
            0.40,
            category="trapezoid_top",
            team=0,
            footprint_xy=_RED_TRAPEZOID_400_TOP,
        ),
        blue_name="blue_trapezoid_400_top",
    ),
    *_paired_surfaces(
        TerrainPrimitive(
            "red_trapezoid_43_ramp",
            (-10.230, 3.128 - _TRAPEZOID_43_RUN_M / 2),
            (1.150, _TRAPEZOID_43_RUN_M),
            0.0,
            0.40,
            category="trapezoid_slope",
            team=0,
            footprint_xy=_RED_TRAPEZOID_43_RAMP,
            vertex_elevations_m=(0.0, 0.0, 0.40, 0.40),
        ),
        blue_name="blue_trapezoid_43_ramp",
    ),
    # Figure 4-27 publishes 7.700m x 10.820m control bounds and a 1.000m
    # connector band. The two chamfer intercepts and assignment of section
    # levels remain explicitly [DIGITIZED]/[INFERRED].
    TerrainPrimitive(
        "central_highland",
        (0.0, 0.0),
        (7.7, 10.82),
        0.20,
        0.20,
        category="central",
        footprint_xy=_CENTRAL_HIGHLAND_FOOTPRINT,
    ),
    TerrainPrimitive(
        "central_plateau",
        (0.0, 0.0),
        (7.7, 10.82),
        0.35,
        0.35,
        category="central_top",
        footprint_xy=_CENTRAL_PLATEAU_FOOTPRINT,
    ),
    TerrainPrimitive(
        "central_red_connector",
        (2.35, -3.805),
        (4.0, _CENTRAL_CONNECTOR_RUN_M),
        0.20,
        0.35,
        category="central_slope",
        team=0,
        footprint_xy=_CENTRAL_RED_CONNECTOR_FOOTPRINT,
        vertex_elevations_m=(0.20, 0.20, 0.35, 0.35),
    ),
    TerrainPrimitive(
        "central_blue_connector",
        (-2.35, 3.805),
        (4.0, _CENTRAL_CONNECTOR_RUN_M),
        0.20,
        0.35,
        category="central_slope",
        team=1,
        footprint_xy=_CENTRAL_BLUE_CONNECTOR_FOOTPRINT,
        vertex_elevations_m=(0.20, 0.20, 0.35, 0.35),
    ),
    # Figures 4-28/4-29 publish an edge chain, 100/200mm levels, and several
    # face angles, but not enough plan-to-section mapping for one non-planar
    # polygon. The chain remains a dashed reference instead of generating
    # arbitrary triangulation or overlapping the central 350mm approximation.
    # Figure 4-37: 17 degree face, 1.145m horizontal run, 0.860m width,
    # and 0.203m/0.553m edge elevations. The separate 0.650m value is the
    # flight gap and must not be added to the ramp footprint.
    TerrainPrimitive(
        "red_fly_ramp",
        _RED_FLY_RAMP_CENTER_XY,
        _FLY_RAMP_SIZE_XY_M,
        _FLY_RAMP_LOW_M,
        _FLY_RAMP_HIGH_M,
        category="fly_ramp",
        team=0,
    ),
    TerrainPrimitive(
        "blue_fly_ramp",
        (-_RED_FLY_RAMP_CENTER_XY[0], -_RED_FLY_RAMP_CENTER_XY[1]),
        _FLY_RAMP_SIZE_XY_M,
        _FLY_RAMP_LOW_M,
        _FLY_RAMP_HIGH_M,
        yaw_deg=180.0,
        category="fly_ramp",
        team=1,
    ),
    TerrainPrimitive(
        "red_rough_road",
        (-7.60, -6.25),
        (_ROUGH_ROAD_LENGTH_M, _ROUGH_ROAD_WIDTH_M),
        0.0,
        0.0,
        category="rough",
        team=0,
    ),
    TerrainPrimitive(
        "blue_rough_road",
        (7.60, 6.25),
        (_ROUGH_ROAD_LENGTH_M, _ROUGH_ROAD_WIDTH_M),
        0.0,
        0.0,
        yaw_deg=180.0,
        category="rough",
        team=1,
    ),
    # Figure 4-25: a 150mm flat inner hex surrounded by six 20 degree faces.
    *_fortress_surfaces(
        prefix="red",
        team=0,
        center_xy=RED_FORTRESS_CENTER_XY,
        outer=_RED_FORTRESS_FOOTPRINT,
    ),
    *_fortress_surfaces(
        prefix="blue",
        team=1,
        center_xy=BLUE_FORTRESS_CENTER_XY,
        outer=_center_symmetric(_RED_FORTRESS_FOOTPRINT),
    ),
    # Figure 4-34 publishes a 0.700m opening, 0.800m outer width, and
    # 0.250/0.260m section dimensions, but not a unique axis length, center,
    # yaw, or height interpretation. A dashed reference is rendered instead
    # of maintaining contradictory Torch, Isaac, and 2D tunnel geometries.
)


def _regular_polygon_xy(
    center_xy: tuple[float, float],
    radius_m: float,
    sides: int,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            center_xy[0] + radius_m * math.cos(2.0 * math.pi * index / sides),
            center_xy[1] + radius_m * math.sin(2.0 * math.pi * index / sides),
        )
        for index in range(sides)
    )


@dataclass(frozen=True)
class ArenaConfig:
    """Official nominal constraints plus explicit geometry provenance."""

    field_length_m: float = 28.0
    field_width_m: float = 15.0
    # Manual V1.5.0 figure 4-6 (printed p.35/PDF p.36) specifies a 1--2
    # degree crown range. Phase 1 uses its 1.5 degree midpoint [SIM].
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
        for primitive in self.config.terrain:
            if primitive.vertex_elevations_m and (
                not primitive.footprint_xy
                or len(primitive.vertex_elevations_m) != len(primitive.footprint_xy)
            ):
                raise ValueError(
                    f"{primitive.name}: vertex elevations must match the polygon footprint"
                )
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
        epsilon = torch.finfo(position_xy.dtype).eps * 32
        on_edge = torch.zeros_like(inside)
        for current in polygon_xy:
            edge_x_delta = previous[0] - current[0]
            edge_y_delta = previous[1] - current[1]
            cross = edge_x_delta * (y - current[1]) - edge_y_delta * (x - current[0])
            within_segment = (x - current[0]) * (x - previous[0]) + (y - current[1]) * (
                y - previous[1]
            ) <= epsilon
            on_edge |= (
                torch.abs(cross)
                <= epsilon * (torch.abs(edge_x_delta) + torch.abs(edge_y_delta) + 1.0)
            ) & within_segment
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
        return inside | on_edge

    def field_height(self, position_xy: Tensor) -> Tensor:
        """Return the manual-specified 1--2 degree field crown."""

        edge_distance = torch.clamp(
            self.config.field_width_m / 2 - torch.abs(position_xy[..., 1]),
            min=0.0,
        )
        return edge_distance * torch.tan(
            torch.tensor(
                self.config.field_crown_slope_deg * torch.pi / 180.0,
                device=position_xy.device,
                dtype=position_xy.dtype,
            )
        )

    @staticmethod
    def _polygon_vertex_elevation(
        position_xy: Tensor,
        primitive: TerrainPrimitive,
        polygon_xy: Tensor,
    ) -> Tensor:
        """Interpolate a polygonal surface from its triangulated vertices."""

        vertex_height = torch.tensor(
            primitive.vertex_elevations_m,
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        elevation = torch.full(
            position_xy.shape[:-1],
            primitive.elevation_start_m,
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        epsilon = torch.finfo(position_xy.dtype).eps * 16
        for a_index, b_index, c_index in triangulate_polygon(primitive.footprint_xy):
            a = polygon_xy[a_index]
            b = polygon_xy[b_index]
            c = polygon_xy[c_index]
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            weight_a = (
                (b[1] - c[1]) * (position_xy[..., 0] - c[0])
                + (c[0] - b[0]) * (position_xy[..., 1] - c[1])
            ) / denominator
            weight_b = (
                (c[1] - a[1]) * (position_xy[..., 0] - c[0])
                + (a[0] - c[0]) * (position_xy[..., 1] - c[1])
            ) / denominator
            weight_c = 1.0 - weight_a - weight_b
            inside = (weight_a >= -epsilon) & (weight_b >= -epsilon) & (weight_c >= -epsilon)
            interpolated = (
                weight_a * vertex_height[a_index]
                + weight_b * vertex_height[b_index]
                + weight_c * vertex_height[c_index]
            )
            elevation = torch.where(inside, interpolated, elevation)
        return elevation

    def terrain_elevation(self, position_xy: Tensor) -> Tensor:
        """Return structural elevation above the local crowned field."""

        elevation_height = torch.zeros(
            position_xy.shape[:-1],
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        if not self.config.terrain:
            return elevation_height
        centers, sizes, cosine, sine, start, end = self._terrain_tensors(
            device=position_xy.device,
            dtype=position_xy.dtype,
        )
        for index, primitive in enumerate(self.config.terrain):
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
            if primitive.vertex_elevations_m:
                elevation = self._polygon_vertex_elevation(
                    position_xy,
                    primitive,
                    polygon,
                )
            else:
                fraction = torch.clamp(
                    local_x / sizes[index, 0] + 0.5,
                    min=0.0,
                    max=1.0,
                )
                elevation = start[index] + fraction * (end[index] - start[index])
            if primitive.category == "rough":
                # Figure 4-36 specifies 70mm bumps at 240mm pitch. A cosine
                # profile is the Phase 1 differentiable [SIM] approximation.
                elevation = elevation + (_ROUGH_BUMP_HEIGHT_M / 2) * (
                    1.0
                    + torch.cos(
                        2.0 * math.pi * local_x / _ROUGH_BUMP_PITCH_M,
                    )
                )
            elevation_height = torch.where(
                inside,
                torch.maximum(elevation_height, elevation),
                elevation_height,
            )
        return elevation_height

    def _terrain_height_analytic(self, position_xy: Tensor) -> Tensor:
        return self.field_height(position_xy) + self.terrain_elevation(position_xy)

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

    def zone_centers(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return Phase 1 ``[SIM]`` rule-region centers by team and ``Zone``."""

        red = torch.tensor(
            (
                RED_SUPPLY_CENTER_XY,
                RED_BASE_CENTER_XY,
                (0.0, 0.0),
                (-9.6, 4.2),
                RED_OUTPOST_CENTER_XY,
                RED_FORTRESS_CENTER_XY,
                BLUE_FORTRESS_CENTER_XY,
                (-0.875, 0.60),
            ),
            device=device,
            dtype=dtype,
        )
        blue = -red
        blue[Zone.CENTRAL_HIGH] = red[Zone.CENTRAL_HIGH]
        return torch.stack((red, blue), dim=0)

    def zone_half_extents(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return Phase 1 ``[SIM]`` rule-region half extents."""

        return torch.tensor(
            ZONE_HALF_EXTENTS_XY_M,
            device=device,
            dtype=dtype,
        )

    def zone_occupancy(self, position_xy: Tensor) -> Tensor:
        if position_xy.shape[-2:] != (constants.UNIT_COUNT, 2):
            raise ValueError("position_xy must have shape [env, unit, 2]")
        device = position_xy.device
        dtype = position_xy.dtype
        centers = self.zone_centers(device=device, dtype=dtype)
        half_extent = self.zone_half_extents(device=device, dtype=dtype)
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
