"""Metric-to-pixel mapping for arena visualizations."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArenaViewport:
    """An isotropic, field-centered viewport with an integer pixel scale.

    ``canvas_size_px`` is ``(width, height)`` and ``field_size_m`` is
    ``(length_along_x, width_along_y)``. The field is centered horizontally;
    the vertical position is fixed by the top and bottom reservations.
    """

    canvas_size_px: tuple[int, int] = (1120, 706)
    field_size_m: tuple[float, float] = (28.0, 15.0)
    margin_top_px: int = 74
    margin_bottom_px: int = 92
    pixels_per_meter: int = 36

    def __post_init__(self) -> None:
        canvas_width, canvas_height = self.canvas_size_px
        if (
            isinstance(canvas_width, bool)
            or isinstance(canvas_height, bool)
            or not isinstance(canvas_width, int)
            or not isinstance(canvas_height, int)
            or canvas_width <= 0
            or canvas_height <= 0
        ):
            raise ValueError("canvas dimensions must be positive integers")
        if (
            isinstance(self.pixels_per_meter, bool)
            or not isinstance(self.pixels_per_meter, int)
            or self.pixels_per_meter <= 0
        ):
            raise ValueError("pixels_per_meter must be a positive integer")
        if (
            isinstance(self.margin_top_px, bool)
            or isinstance(self.margin_bottom_px, bool)
            or not isinstance(self.margin_top_px, int)
            or not isinstance(self.margin_bottom_px, int)
            or self.margin_top_px < 0
            or self.margin_bottom_px < 0
        ):
            raise ValueError("vertical margins must be non-negative integers")

        field_length, field_width = self.field_size_m
        if (
            not math.isfinite(field_length)
            or not math.isfinite(field_width)
            or field_length <= 0
            or field_width <= 0
        ):
            raise ValueError("field dimensions must be finite and positive")

        field_width_px = self._pixel_extent(field_length)
        field_height_px = self._pixel_extent(field_width)
        if field_width_px > canvas_width:
            raise ValueError("field width does not fit inside the canvas")
        if (canvas_width - field_width_px) % 2:
            raise ValueError("field must have an integer-pixel horizontal center")
        if self.margin_top_px + field_height_px + self.margin_bottom_px != canvas_height:
            raise ValueError("vertical margins and field height must exactly fill the canvas")

    def _pixel_extent(self, distance_m: float) -> int:
        extent = distance_m * self.pixels_per_meter
        rounded = round(extent)
        if not math.isclose(extent, rounded, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("field dimensions must map to an integer number of pixels")
        return rounded

    @property
    def canvas_width_px(self) -> int:
        """Canvas width in pixels."""

        return self.canvas_size_px[0]

    @property
    def canvas_height_px(self) -> int:
        """Canvas height in pixels."""

        return self.canvas_size_px[1]

    @property
    def field_length_m(self) -> float:
        """Field length along the world x-axis."""

        return self.field_size_m[0]

    @property
    def field_width_m(self) -> float:
        """Field width along the world y-axis."""

        return self.field_size_m[1]

    @property
    def field_width_px(self) -> int:
        """Rendered field extent along the canvas x-axis."""

        return self._pixel_extent(self.field_length_m)

    @property
    def field_height_px(self) -> int:
        """Rendered field extent along the canvas y-axis."""

        return self._pixel_extent(self.field_width_m)

    @property
    def scale_x_px_per_m(self) -> int:
        """Horizontal metric scale."""

        return self.pixels_per_meter

    @property
    def scale_y_px_per_m(self) -> int:
        """Vertical metric scale."""

        return self.pixels_per_meter

    @property
    def field_bounds(self) -> tuple[int, int, int, int]:
        """Return ``(left, top, right, bottom)`` field-edge coordinates."""

        left = (self.canvas_width_px - self.field_width_px) // 2
        top = self.margin_top_px
        return (
            left,
            top,
            left + self.field_width_px,
            top + self.field_height_px,
        )

    @property
    def field_bounds_px(self) -> tuple[int, int, int, int]:
        """Alias spelling that makes the pixel unit explicit."""

        return self.field_bounds

    def to_pixel(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Map a world coordinate to its nearest canvas pixel."""

        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("world coordinates must be finite")
        left, top, _, _ = self.field_bounds
        pixel_x = left + (x_m + self.field_length_m / 2) * self.pixels_per_meter
        pixel_y = top + (self.field_width_m / 2 - y_m) * self.pixels_per_meter
        return round(pixel_x), round(pixel_y)

    def to_world(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Map a canvas position back to world coordinates."""

        if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
            raise ValueError("pixel coordinates must be finite")
        left, top, _, _ = self.field_bounds
        x_m = (pixel_x - left) / self.pixels_per_meter - self.field_length_m / 2
        y_m = self.field_width_m / 2 - (pixel_y - top) / self.pixels_per_meter
        return x_m, y_m

    def meters_to_pixels(self, distance_m: float) -> int:
        """Convert a non-negative metric distance to its nearest pixel count."""

        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("distance_m must be finite and non-negative")
        return round(distance_m * self.pixels_per_meter)
