from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "visualize_torch.py"
_SPEC = importlib.util.spec_from_file_location("rm_cortex_visualize_torch", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
visualize_torch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(visualize_torch)


def _bounds(points: list[tuple[int, int]]) -> tuple[int, int]:
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    return max(x) - min(x), max(y) - min(y)


def test_video_uses_the_exact_isotropic_field_viewport() -> None:
    viewport = visualize_torch.VIEWPORT

    assert visualize_torch.CANVAS == (1120, 706)
    assert viewport.field_bounds == (56, 74, 1064, 614)
    assert visualize_torch._to_pixel(-14.0, 7.5) == (56, 74)
    assert visualize_torch._to_pixel(14.0, -7.5) == (1064, 614)
    assert visualize_torch._meters_to_pixels(1.0) == 36


def test_published_base_and_pad_dimensions_keep_metric_scale() -> None:
    viewport = visualize_torch.VIEWPORT
    base = visualize_torch._rectangle_points(
        (0.0, 0.0),
        visualize_torch.BASE_PEDESTAL_SIZE_XY_M,
        0.0,
    )
    pad_outer = visualize_torch._rectangle_points(
        (0.0, 0.0),
        visualize_torch.AERIAL_PAD_OUTER_ENVELOPE_SIZE_XY_M,
        0.0,
    )
    pad_landing = visualize_torch._dimensioned_octagon(
        (0.0, 0.0),
        visualize_torch.AERIAL_PAD_LANDING_SIZE_XY_M,
        horizontal_straight_edge_m=visualize_torch.AERIAL_PAD_LANDING_STRAIGHT_EDGE_M,
        vertical_straight_edge_m=visualize_torch.AERIAL_PAD_LANDING_STRAIGHT_EDGE_M,
    )

    base_width, base_height = _bounds(base)
    pad_outer_width, pad_outer_height = _bounds(pad_outer)
    pad_landing_width, pad_landing_height = _bounds(pad_landing)

    assert base_width == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.BASE_PEDESTAL_SIZE_XY_M[0],
        abs=1.0,
    )
    assert base_height == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.BASE_PEDESTAL_SIZE_XY_M[1],
        abs=1.0,
    )
    assert pad_outer_width == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_OUTER_ENVELOPE_SIZE_XY_M[0],
        abs=1.0,
    )
    assert pad_outer_height == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_OUTER_ENVELOPE_SIZE_XY_M[1],
        abs=1.0,
    )
    assert pad_landing_width == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_LANDING_SIZE_XY_M[0],
        abs=1.0,
    )
    assert pad_landing_height == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_LANDING_SIZE_XY_M[1],
        abs=1.0,
    )
    assert abs(pad_landing[1][0] - pad_landing[0][0]) == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_LANDING_STRAIGHT_EDGE_M,
        abs=1.0,
    )
    assert abs(pad_landing[3][1] - pad_landing[2][1]) == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_LANDING_STRAIGHT_EDGE_M,
        abs=1.0,
    )


def test_circle_boxes_use_the_same_scale_on_both_axes() -> None:
    left, top, right, bottom = visualize_torch._ellipse_box(
        (0.0, 0.0),
        (0.40, 0.40),
    )

    assert right - left == bottom - top
    assert right - left == pytest.approx(
        0.80 * visualize_torch.VIEWPORT.pixels_per_meter,
        abs=1.0,
    )
