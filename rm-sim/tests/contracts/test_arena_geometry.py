from __future__ import annotations

import math

import pytest
import torch

from rm_referee import GameState, RandomTape, Referee
from rm_referee.schema import Role, Team, Weapon, Zone, slot
from rm_world import (
    BLUE_BASE_CENTER_XY,
    BLUE_FORTRESS_CENTER_XY,
    BLUE_OUTPOST_CENTER_XY,
    RED_BASE_CENTER_XY,
    RED_FORTRESS_CENTER_XY,
    RED_OUTPOST_CENTER_XY,
    ArenaGeometry,
    HitModel,
    HitModelConfig,
    KinematicState,
    TorchRuleBackend,
    WorldActions,
)
from rm_world.arena import (
    AERIAL_PAD_LANDING_SIZE_XY_M,
    AERIAL_PAD_LANDING_STRAIGHT_EDGE_M,
    AERIAL_PAD_OUTER_ENVELOPE_SIZE_XY_M,
    BASE_PEDESTAL_SIZE_XY_M,
    OUTPOST_BODY_DIAMETER_M,
    OUTPOST_PEDESTAL_WIDTH_M,
    ZONE_HALF_EXTENTS_XY_M,
)


def test_terrain_height_models_field_crown_highland_and_fly_ramp() -> None:
    arena = ArenaGeometry()
    samples = torch.tensor(
        (
            (13.0, 7.5),
            (13.0, 0.0),
            (0.0, 0.0),
            (-1.30, -6.9),
            (-0.28, -6.9),
        )
    )

    height = arena.terrain_height(samples)

    assert height[0] == pytest.approx(0.0, abs=1.0e-6)
    assert height[1] > height[0]
    assert height[2] > height[1]
    assert height[4] - height[3] == pytest.approx(0.312, abs=0.02)


def test_manual_figure_4_5_feature_centers_use_center_coordinates() -> None:
    arena = ArenaGeometry()

    assert RED_BASE_CENTER_XY == pytest.approx((-11.593, 0.0))
    assert BLUE_BASE_CENTER_XY == pytest.approx((11.593, 0.0))
    assert RED_OUTPOST_CENTER_XY == pytest.approx((-3.008, -3.857))
    assert BLUE_OUTPOST_CENTER_XY == pytest.approx((3.008, 3.857))
    assert RED_FORTRESS_CENTER_XY == pytest.approx((-7.4, 0.0))
    assert BLUE_FORTRESS_CENTER_XY == pytest.approx((7.4, 0.0))
    assert torch.allclose(
        arena.outpost_centers(device="cpu", dtype=torch.float32),
        torch.tensor((RED_OUTPOST_CENTER_XY, BLUE_OUTPOST_CENTER_XY)),
    )


def test_published_structure_envelopes_match_the_manual() -> None:
    assert BASE_PEDESTAL_SIZE_XY_M == pytest.approx((1.609, 1.881))
    assert OUTPOST_BODY_DIAMETER_M == pytest.approx(0.550)
    assert OUTPOST_PEDESTAL_WIDTH_M == pytest.approx(0.650)
    assert AERIAL_PAD_OUTER_ENVELOPE_SIZE_XY_M == pytest.approx((2.200, 2.858))
    assert AERIAL_PAD_LANDING_SIZE_XY_M == pytest.approx((2.149, 2.200))
    assert AERIAL_PAD_LANDING_STRAIGHT_EDGE_M == pytest.approx(1.334)


@pytest.mark.parametrize(
    ("name", "expected_size"),
    (
        ("red_trapezoid_highland", (10.805, 4.380)),
        ("red_road", (8.901, 3.651)),
    ),
)
def test_published_plan_envelopes_match_polygon_bounds(
    name: str,
    expected_size: tuple[float, float],
) -> None:
    arena = ArenaGeometry()
    primitive = next(item for item in arena.config.terrain if item.name == name)
    corners = arena.terrain_corners(
        primitive,
        device="cpu",
        dtype=torch.float64,
    )
    bounds = corners.amax(dim=0) - corners.amin(dim=0)

    assert primitive.size_xy == pytest.approx(expected_size)
    assert bounds.tolist() == pytest.approx(expected_size, abs=1.0e-9)


def test_trapezoid_figure_4_26_uses_the_published_dimension_chain() -> None:
    arena = ArenaGeometry()
    trapezoid = next(item for item in arena.config.terrain if item.name == "red_trapezoid_highland")
    points = trapezoid.footprint_xy

    inner_min_x = points[0][0] + 0.300
    assert points[1][0] - points[0][0] == pytest.approx(0.300 + 6.707 + 3.798)
    assert points[0][1] - points[5][1] == pytest.approx(4.380)
    assert points[0][1] - points[2][1] == pytest.approx(1.003)
    assert points[3][0] - inner_min_x == pytest.approx(6.707)
    assert points[1][0] - points[3][0] == pytest.approx(3.798)
    assert points[4][0] - inner_min_x == pytest.approx(4.440)


def test_zone_geometry_has_one_shared_source_for_centers_and_extents() -> None:
    arena = ArenaGeometry()
    centers = arena.zone_centers(device="cpu", dtype=torch.float64)
    extents = arena.zone_half_extents(device="cpu", dtype=torch.float64)

    assert centers.shape == (2, len(Zone), 2)
    assert torch.allclose(
        extents,
        torch.tensor(ZONE_HALF_EXTENTS_XY_M, dtype=torch.float64),
    )
    assert torch.allclose(
        centers[Team.BLUE, :],
        -centers[Team.RED, :],
    )
    assert torch.equal(
        centers[Team.BLUE, Zone.CENTRAL_HIGH],
        centers[Team.RED, Zone.CENTRAL_HIGH],
    )


def test_central_highland_uses_the_figure_4_27_polygon_not_a_rectangle() -> None:
    arena = ArenaGeometry()
    central = next(
        primitive for primitive in arena.config.terrain if primitive.name == "central_highland"
    )
    corners = arena.terrain_corners(
        central,
        device="cpu",
        dtype=torch.float32,
    )
    samples = torch.tensor(((0.0, -4.5), (3.7, -4.5)))
    heights = arena._terrain_height_analytic(samples)

    assert central.size_xy == pytest.approx((7.7, 10.82))
    assert corners.shape == (6, 2)
    assert heights[0] - heights[1] == pytest.approx(0.35, abs=1.0e-5)


@pytest.mark.parametrize(
    "feature",
    ("road", "trapezoid_highland", "assembly", "fly_ramp", "rough_road", "fortress", "tunnel"),
)
def test_paired_terrain_footprints_are_center_symmetric(feature: str) -> None:
    arena = ArenaGeometry()
    by_name = {primitive.name: primitive for primitive in arena.config.terrain}
    red = arena.terrain_corners(
        by_name[f"red_{feature}"],
        device="cpu",
        dtype=torch.float32,
    )
    blue = arena.terrain_corners(
        by_name[f"blue_{feature}"],
        device="cpu",
        dtype=torch.float32,
    )

    distance = torch.cdist(-red, blue)
    assert (distance.amin(dim=0) < 1.0e-5).all()
    assert (distance.amin(dim=1) < 1.0e-5).all()


def test_rough_road_models_figure_4_36_bump_height_and_pitch() -> None:
    arena = ArenaGeometry()
    rough = next(item for item in arena.config.terrain if item.name == "red_rough_road")
    peak_trough_peak = torch.tensor(((-7.60, -6.25), (-7.48, -6.25), (-7.36, -6.25)))

    heights = arena._terrain_height_analytic(peak_trough_peak)

    assert rough.size_xy[0] == pytest.approx(2.560)
    assert heights[0] - heights[1] == pytest.approx(0.070, abs=1.0e-5)
    assert heights[0] == pytest.approx(heights[2], abs=1.0e-5)


def test_fly_ramp_matches_figure_4_37_dimensions_and_slope() -> None:
    arena = ArenaGeometry()
    ramp = next(item for item in arena.config.terrain if item.name == "red_fly_ramp")
    slope_deg = math.degrees(
        math.atan2(
            ramp.elevation_end_m - ramp.elevation_start_m,
            ramp.size_xy[0],
        )
    )

    assert ramp.size_xy == pytest.approx((1.145, 0.860))
    assert (ramp.elevation_start_m, ramp.elevation_end_m) == pytest.approx((0.203, 0.553))
    assert slope_deg == pytest.approx(17.0, abs=0.05)


def test_polygonal_slope_vertices_match_their_terrain_elevations() -> None:
    arena = ArenaGeometry()
    slope = next(
        primitive for primitive in arena.config.terrain if primitive.name == "red_trapezoid_23_ramp"
    )
    samples = torch.tensor(slope.footprint_xy)

    elevation = arena.terrain_elevation(samples)

    assert elevation.tolist() == pytest.approx(slope.vertex_elevations_m, abs=1.0e-5)


def test_fortress_uses_six_twenty_degree_faces_and_a_150mm_top() -> None:
    arena = ArenaGeometry()
    by_name = {primitive.name: primitive for primitive in arena.config.terrain}
    slopes = [
        primitive
        for primitive in arena.config.terrain
        if primitive.name.startswith("red_fortress_slope_")
    ]
    top = by_name["red_fortress_top"]
    outer = by_name["red_fortress"]
    first_slope = slopes[0]
    outer_midpoint = torch.tensor(first_slope.footprint_xy[:2]).mean(dim=0)
    inner_midpoint = torch.tensor(first_slope.footprint_xy[2:]).mean(dim=0)
    samples = torch.stack(
        (
            outer_midpoint,
            (outer_midpoint + inner_midpoint) / 2,
            inner_midpoint,
            torch.tensor(top.center_xy),
        )
    )

    elevation = arena.terrain_elevation(samples)

    assert len(slopes) == 6
    assert outer.size_xy == pytest.approx((2.240, 1.939))
    assert outer.footprint_xy[4][0] - outer.footprint_xy[5][0] == pytest.approx(1.120)
    assert top.size_xy == pytest.approx((1.306, 1.131))
    assert elevation.tolist() == pytest.approx((0.0, 0.075, 0.15, 0.15), abs=1.0e-5)


def test_central_connectors_preserve_the_200_to_350mm_levels() -> None:
    arena = ArenaGeometry()
    connector = next(
        primitive for primitive in arena.config.terrain if primitive.name == "central_red_connector"
    )
    low_midpoint = torch.tensor(connector.footprint_xy[:2]).mean(dim=0)
    high_midpoint = torch.tensor(connector.footprint_xy[2:]).mean(dim=0)
    samples = torch.stack(
        (
            low_midpoint,
            (low_midpoint + high_midpoint) / 2,
            high_midpoint,
        )
    )

    elevation = arena.terrain_elevation(samples)

    assert elevation.tolist() == pytest.approx((0.20, 0.275, 0.35), abs=1.0e-5)


def test_every_vertex_elevation_matches_its_polygon_arity() -> None:
    for primitive in ArenaGeometry().config.terrain:
        if primitive.vertex_elevations_m:
            assert len(primitive.vertex_elevations_m) == len(primitive.footprint_xy)


def test_aerial_area_enforces_the_section_4_5_tether_envelope() -> None:
    arena = ArenaGeometry()
    red = torch.tensor(((-12.52, 5.68), (2.4, 4.0), (2.41, 4.0), (0.0, 2.9)))
    blue = -red

    assert arena.aerial_flight_area(red, Team.RED).tolist() == [True, True, False, False]
    assert arena.aerial_flight_area(blue, Team.BLUE).tolist() == [True, True, False, False]
    projected = arena.project_to_aerial_flight_area(
        torch.tensor(((9.0, 0.0),)),
        Team.RED,
    )
    assert torch.allclose(projected, torch.tensor(((2.4, 3.0),)))


def test_team_spawns_are_center_symmetric() -> None:
    game = GameState.create(1)
    positions = ArenaGeometry().spawn_positions(game)[0]

    assert torch.allclose(
        positions[:8],
        -positions[8:],
    )


def test_static_aabb_blocks_los_and_signed_distance_marks_occupancy() -> None:
    arena = ArenaGeometry()
    blocked = arena.line_of_sight(
        torch.tensor([[-3.0, 0.0]]),
        torch.tensor([[3.0, 0.0]]),
    )
    clear = arena.line_of_sight(
        torch.tensor([[-3.0, 6.0]]),
        torch.tensor([[3.0, 6.0]]),
    )

    assert not blocked.item()
    assert clear.item()
    assert arena.signed_distance(torch.tensor([[0.0, 0.0]])).item() < 0
    assert arena.signed_distance(torch.tensor([[0.0, 6.0]])).item() > 0


def test_torch_backend_generates_armor_hit_from_target_action() -> None:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    shooter = slot(Team.RED, Role.INFANTRY_3)
    target = slot(Team.BLUE, Role.INFANTRY_3)
    world.position_xy[0, shooter] = torch.tensor([-5.0, 6.0])
    world.position_xy[0, target] = torch.tensor([5.0, 6.0])
    game.ammo[0, shooter, Weapon.MM17] = 1
    actions = WorldActions.zeros(game)
    actions.target[0, shooter] = 2
    actions.fire[0, shooter] = True
    backend = TorchRuleBackend(
        hit_model=HitModel(
            HitModelConfig(
                base_accuracy=1.0,
                range_scale_m=1.0e6,
                motion_scale_mps=1.0e6,
                projection_floor=1.0,
                critical_probability=0.0,
            )
        )
    )
    tape = RandomTape(torch.zeros((1, backend_hit_draws())))

    inputs = backend.rule_inputs(world, game, actions, tape, 0.1)
    next_state, events = Referee().step(game, inputs)

    assert inputs.hits.source.eq(shooter).any()
    assert events.hit_accepted.any()
    assert next_state.hp[0, target] == 180


def test_torch_backend_scatters_team_radar_reports_to_selected_targets() -> None:
    game = GameState.create(2)
    world = KinematicState.spawn(game)
    actions = WorldActions.zeros(game)
    red_target = slot(Team.BLUE, Role.INFANTRY_3)
    blue_target = slot(Team.RED, Role.HERO)
    actions.radar_target[0, Team.RED] = red_target
    actions.radar_report_xy[0, Team.RED] = torch.tensor((3.5, -2.0))
    actions.radar_target[1, Team.BLUE] = blue_target
    actions.radar_report_xy[1, Team.BLUE] = torch.tensor((-4.0, 1.25))
    tape = RandomTape(torch.zeros((2, backend_hit_draws())))

    inputs = TorchRuleBackend().rule_inputs(world, game, actions, tape, 0.1)

    assert inputs.radar_update.sum() == 2
    assert inputs.radar_update[0, Team.RED, red_target]
    assert inputs.radar_update[1, Team.BLUE, blue_target]
    assert torch.equal(
        inputs.radar_report_xy[0, Team.RED, red_target],
        torch.tensor((3.5, -2.0)),
    )
    assert torch.equal(
        inputs.radar_report_xy[1, Team.BLUE, blue_target],
        torch.tensor((-4.0, 1.25)),
    )
    assert torch.count_nonzero(inputs.radar_report_xy * (~inputs.radar_update[..., None])) == 0


def backend_hit_draws() -> int:
    return 16 * 2 + 2
