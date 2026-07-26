from __future__ import annotations

import pytest

from rm_world import ArenaViewport


def test_default_viewport_is_isotropic_and_uses_the_exact_field_extent() -> None:
    viewport = ArenaViewport()

    assert viewport.scale_x_px_per_m == viewport.scale_y_px_per_m == 36
    assert viewport.field_width_px == 1008
    assert viewport.field_height_px == 540
    assert viewport.field_bounds == (56, 74, 1064, 614)
    assert viewport.field_bounds_px == viewport.field_bounds


def test_default_viewport_maps_field_corners_and_center() -> None:
    viewport = ArenaViewport()

    assert viewport.to_pixel(-14.0, 7.5) == (56, 74)
    assert viewport.to_pixel(14.0, 7.5) == (1064, 74)
    assert viewport.to_pixel(-14.0, -7.5) == (56, 614)
    assert viewport.to_pixel(14.0, -7.5) == (1064, 614)
    assert viewport.to_pixel(0.0, 0.0) == (560, 344)


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    (
        (1.25, 2.75),
        (-3.008, -3.857),
        (11.593, 0.0),
        (0.05, -0.05),
    ),
)
def test_center_symmetric_world_points_are_pixel_symmetric(
    x_m: float,
    y_m: float,
) -> None:
    viewport = ArenaViewport()
    center_x, center_y = viewport.to_pixel(0.0, 0.0)
    positive = viewport.to_pixel(x_m, y_m)
    negative = viewport.to_pixel(-x_m, -y_m)

    assert positive[0] + negative[0] == 2 * center_x
    assert positive[1] + negative[1] == 2 * center_y


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    (
        (-13.999, 7.499),
        (-3.008, -3.857),
        (0.013, -0.019),
        (11.593, 0.0),
        (13.999, -7.499),
    ),
)
def test_pixel_round_trip_stays_within_half_a_pixel(
    x_m: float,
    y_m: float,
) -> None:
    viewport = ArenaViewport()

    reconstructed = viewport.to_world(*viewport.to_pixel(x_m, y_m))
    half_pixel_m = 0.5 / viewport.pixels_per_meter

    assert reconstructed[0] == pytest.approx(x_m, abs=half_pixel_m)
    assert reconstructed[1] == pytest.approx(y_m, abs=half_pixel_m)


def test_custom_field_dimensions_use_the_same_metric_scale() -> None:
    viewport = ArenaViewport(
        canvas_size_px=(800, 500),
        field_size_m=(20.0, 10.0),
        margin_top_px=80,
        margin_bottom_px=120,
        pixels_per_meter=30,
    )

    assert viewport.field_bounds == (100, 80, 700, 380)
    assert viewport.to_pixel(-10.0, 5.0) == (100, 80)
    assert viewport.to_pixel(10.0, -5.0) == (700, 380)
    assert viewport.to_pixel(0.0, 0.0) == (400, 230)
    assert viewport.meters_to_pixels(1.5) == 45
    assert viewport.to_world(400, 230) == pytest.approx((0.0, 0.0))
