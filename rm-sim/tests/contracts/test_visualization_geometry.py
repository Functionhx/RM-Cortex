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


def test_base_core_and_pad_reference_spans_keep_metric_scale() -> None:
    viewport = visualize_torch.VIEWPORT
    base = [
        visualize_torch._to_pixel(x, y) for x, y in visualize_torch.RED_BASE_PEDESTAL_FOOTPRINT_XY_M
    ]
    pad_reference = visualize_torch._rectangle_points(
        (0.0, 0.0),
        visualize_torch.AERIAL_PAD_FRAME_REFERENCE_SPANS_XY_M,
        0.0,
    )

    base_width, base_height = _bounds(base)
    pad_reference_width, pad_reference_height = _bounds(pad_reference)

    assert base_width == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.BASE_PEDESTAL_SIZE_XY_M[0],
        abs=1.0,
    )
    assert base_height == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.BASE_PEDESTAL_SIZE_XY_M[1],
        abs=1.0,
    )
    assert pad_reference_width == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_FRAME_REFERENCE_SPANS_XY_M[0],
        abs=1.0,
    )
    assert pad_reference_height == pytest.approx(
        viewport.pixels_per_meter * visualize_torch.AERIAL_PAD_FRAME_REFERENCE_SPANS_XY_M[1],
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
