#!/usr/bin/env python3
"""Export a scripted RM-Cortex match as an H.264 MP4."""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import subprocess

import torch

from rm_referee import constants
from rm_referee.schema import Role, Team, Winner, unit_roles, unit_teams
from rm_world import (
    TacticalScriptedOpponent,
    TorchEnvConfig,
    TorchRMArena,
    tactical_phase_label,
)
from rm_world.geometry import resolve_target_slots


ROLE_LABELS = ("H", "E", "I3", "I4", "A", "S", "B", "O")
RED = (255, 58, 83)
BLUE = (20, 136, 255)
CANVAS = (1120, 700)
MARGIN_X = 44
MARGIN_TOP = 74
MARGIN_BOTTOM = 92
HEIGHT_STOPS = (
    (0.00, (14, 35, 46)),
    (0.15, (51, 78, 84)),
    (0.20, (67, 94, 99)),
    (0.30, (87, 116, 119)),
    (0.35, (108, 137, 139)),
    (0.40, (132, 158, 159)),
    (0.55, (180, 194, 192)),
)
FEATURE_LABELS = {
    "central_highland": "CENTRAL HIGH · 0.20–0.35m · 10.5°",
    "red_trapezoid_highland": "TRAPEZOID · 23°/43°",
    "blue_trapezoid_highland": "TRAPEZOID · 23°/43°",
    "red_road": "ROAD · 11°/15°",
    "blue_road": "ROAD · 11°/15°",
    "red_rough_road": "BUMPS · 70/240mm",
    "blue_rough_road": "BUMPS · 70/240mm",
    "red_fortress": "FORT · 20°",
    "blue_fortress": "FORT · 20°",
    "red_tunnel": "TUNNEL",
    "blue_tunnel": "TUNNEL",
}
SLOPE_LABELS = {
    "red_trapezoid_23_ramp": "23°",
    "blue_trapezoid_23_ramp": "23°",
    "red_trapezoid_43_ramp": "43°",
    "blue_trapezoid_43_ramp": "43°",
    "central_red_connector": "10.5°",
    "central_blue_connector": "10.5°",
    "red_fly_ramp": "FLY · 17°",
    "blue_fly_ramp": "FLY · 17°",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=constants.MATCH_DURATION_S,
        help="Simulated match duration; defaults to the complete 420s match.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional policy-step override for quick smoke exports.",
    )
    parser.add_argument(
        "--capture-every",
        type=int,
        default=5,
        help="Capture one frame per N policy steps (default: 1 simulated second).",
    )
    parser.add_argument("--fps", type=int, default=20, help="MP4 playback frame rate.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/torch_demo.mp4")
    return parser


def _font(size: int, *, bold: bool = False) -> object:
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _to_pixel(x: float, y: float) -> tuple[int, int]:
    width, height = CANVAS
    field_width = width - 2 * MARGIN_X
    field_height = height - MARGIN_TOP - MARGIN_BOTTOM
    px = MARGIN_X + (x + 14.0) / 28.0 * field_width
    py = MARGIN_TOP + (7.5 - y) / 15.0 * field_height
    return round(px), round(py)


def _meters_to_pixels(distance_m: float) -> int:
    field_width_pixels = CANVAS[0] - 2 * MARGIN_X
    return max(1, round(distance_m / 28.0 * field_width_pixels))


def _regular_polygon(
    center_xy: tuple[float, float],
    radius_xy: tuple[float, float],
    sides: int,
) -> list[tuple[int, int]]:
    return [
        _to_pixel(
            center_xy[0] + radius_xy[0] * math.cos(2 * math.pi * index / sides),
            center_xy[1] + radius_xy[1] * math.sin(2 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def _height_color(elevation_m: float) -> tuple[int, int, int]:
    elevation_m = max(0.0, min(elevation_m, HEIGHT_STOPS[-1][0]))
    for (low_height, low_color), (high_height, high_color) in zip(
        HEIGHT_STOPS,
        HEIGHT_STOPS[1:],
        strict=True,
    ):
        if elevation_m <= high_height:
            span = high_height - low_height
            blend = (elevation_m - low_height) / span
            return tuple(
                round(low + blend * (high - low))
                for low, high in zip(low_color, high_color, strict=True)
            )
    return HEIGHT_STOPS[-1][1]


def _rectangle_points(
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_deg: float,
) -> list[tuple[int, int]]:
    half_x = size_xy[0] / 2
    half_y = size_xy[1] / 2
    angle = math.radians(yaw_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [
        _to_pixel(
            center_xy[0] + cosine * x - sine * y,
            center_xy[1] + sine * x + cosine * y,
        )
        for x, y in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    ]


def _terrain_points(arena: object, primitive: object) -> list[tuple[int, int]]:
    corners = arena.terrain_corners(
        primitive,
        device="cpu",
        dtype=torch.float32,
    )
    return [_to_pixel(float(point[0]), float(point[1])) for point in corners]


def _render_elevation_layer(image: object, arena: object) -> None:
    from PIL import Image, ImageDraw

    left, top = _to_pixel(-14.0, 7.5)
    right, bottom = _to_pixel(14.0, -7.5)
    field_size = (right - left + 1, bottom - top + 1)
    sample_size = (field_size[0] // 2, field_size[1] // 2)
    x = torch.linspace(-14.0, 14.0, sample_size[0])
    y = torch.linspace(7.5, -7.5, sample_size[1])
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    position = torch.stack((xx, yy), dim=-1)
    elevation = arena.terrain_elevation(position)
    crown = arena.field_height(position)
    max_crown = 7.5 * math.tan(math.radians(arena.config.field_crown_slope_deg))
    pixels = []
    for height, crown_height in zip(
        elevation.reshape(-1).tolist(),
        crown.reshape(-1).tolist(),
        strict=True,
    ):
        if height < 0.012:
            crown_lift = round(8 * crown_height / max(max_crown, 1.0e-6))
            pixels.append(tuple(channel + crown_lift for channel in HEIGHT_STOPS[0][1]))
        else:
            band_height = round(height / 0.05) * 0.05
            pixels.append(_height_color(band_height))
    terrain_layer = Image.new("RGB", sample_size)
    terrain_layer.putdata(pixels)
    terrain_layer = terrain_layer.resize(field_size, Image.Resampling.NEAREST)
    mask = Image.new("L", field_size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, field_size[0] - 1, field_size[1] - 1),
        radius=8,
        fill=255,
    )
    image.paste(terrain_layer, (left, top), mask)


def _draw_slope_arrow(
    draw: object,
    primitive: object,
    label: str,
    label_font: object,
) -> None:
    if primitive.vertex_elevations_m:
        minimum = min(primitive.vertex_elevations_m)
        maximum = max(primitive.vertex_elevations_m)
        low_vertices = [
            point
            for point, height in zip(
                primitive.footprint_xy,
                primitive.vertex_elevations_m,
                strict=True,
            )
            if abs(height - minimum) < 1.0e-6
        ]
        high_vertices = [
            point
            for point, height in zip(
                primitive.footprint_xy,
                primitive.vertex_elevations_m,
                strict=True,
            )
            if abs(height - maximum) < 1.0e-6
        ]
        low_xy = (
            sum(point[0] for point in low_vertices) / len(low_vertices),
            sum(point[1] for point in low_vertices) / len(low_vertices),
        )
        high_xy = (
            sum(point[0] for point in high_vertices) / len(high_vertices),
            sum(point[1] for point in high_vertices) / len(high_vertices),
        )
    else:
        direction = (
            math.cos(math.radians(primitive.yaw_deg)),
            math.sin(math.radians(primitive.yaw_deg)),
        )
        half_arrow = primitive.size_xy[0] * 0.31
        low_xy = (
            primitive.center_xy[0] - direction[0] * half_arrow,
            primitive.center_xy[1] - direction[1] * half_arrow,
        )
        high_xy = (
            primitive.center_xy[0] + direction[0] * half_arrow,
            primitive.center_xy[1] + direction[1] * half_arrow,
        )
        if primitive.elevation_start_m > primitive.elevation_end_m:
            low_xy, high_xy = high_xy, low_xy
    low = _to_pixel(*low_xy)
    high = _to_pixel(*high_xy)
    draw.line((*low, *high), fill=(239, 245, 245), width=2)
    dx = high[0] - low[0]
    dy = high[1] - low[1]
    length = max(math.hypot(dx, dy), 1.0)
    unit = (dx / length, dy / length)
    normal = (-unit[1], unit[0])
    arrow = (
        high,
        (
            round(high[0] - unit[0] * 7 + normal[0] * 4),
            round(high[1] - unit[1] * 7 + normal[1] * 4),
        ),
        (
            round(high[0] - unit[0] * 7 - normal[0] * 4),
            round(high[1] - unit[1] * 7 - normal[1] * 4),
        ),
    )
    draw.polygon(arrow, fill=(239, 245, 245))
    midpoint = ((low[0] + high[0]) // 2, (low[1] + high[1]) // 2 - 7)
    draw.text(
        midpoint,
        label,
        fill=(239, 245, 245),
        font=label_font,
        anchor="mm",
        stroke_width=2,
        stroke_fill=(18, 28, 33),
    )


def _draw_height_legend(draw: object, legend_font: object) -> None:
    levels = (0.00, 0.15, 0.20, 0.30, 0.35, 0.40, 0.55)
    start_x = 338
    y = CANVAS[1] - 73
    draw.text(
        (start_x - 12, y + 6),
        "HEIGHT ABOVE LOCAL FIELD",
        fill=(139, 159, 168),
        font=legend_font,
        anchor="ra",
    )
    x = start_x
    for level in levels:
        draw.rectangle(
            (x, y, x + 22, y + 12),
            fill=_height_color(level),
            outline=(172, 188, 194),
        )
        draw.text(
            (x + 26, y + 6),
            f"{level:.2f}",
            fill=(172, 188, 194),
            font=legend_font,
            anchor="lm",
        )
        x += 93 if level < 0.10 else 99
    draw.text(
        (CANVAS[0] - MARGIN_X, y + 6),
        "m",
        fill=(172, 188, 194),
        font=legend_font,
        anchor="rm",
    )


def _draw_static(
    image: object,
    arena: object,
    label_font: object,
    legend_font: object,
) -> None:
    from PIL import ImageDraw

    _render_elevation_layer(image, arena)
    draw = ImageDraw.Draw(image)
    left, top = _to_pixel(-14.0, 7.5)
    right, bottom = _to_pixel(14.0, -7.5)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=8,
        outline=(205, 222, 232),
        width=2,
    )
    for y in (-5.0, -2.5, 2.5, 5.0):
        start = _to_pixel(-14.0, y)
        end = _to_pixel(14.0, y)
        draw.line((*start, *end), fill=(24, 50, 61), width=1)
    center_top = _to_pixel(0.0, 7.5)
    center_bottom = _to_pixel(0.0, -7.5)
    draw.line((*center_top, *center_bottom), fill=(142, 164, 176), width=2)

    for primitive in arena.config.terrain:
        points = _terrain_points(arena, primitive)
        team_color = RED if primitive.team == Team.RED else BLUE
        primary_surface = primitive.name in FEATURE_LABELS or primitive.category == "assembly"
        outline = (151, 168, 175) if primitive.team is None else team_color
        maximum_height = max(
            primitive.vertex_elevations_m
            or (primitive.elevation_start_m, primitive.elevation_end_m)
        )
        if maximum_height >= 0.12 and (
            primary_surface
            or primitive.category in {"central_top", "trapezoid_top", "fortress_top"}
        ):
            shadow = [(x + 2, y + 3) for x, y in points]
            draw.line((*shadow, shadow[0]), fill=(4, 12, 17), width=4)
        if primitive.category == "tunnel":
            draw.polygon(points, fill=_height_color(0.10))
            shoulder = _rectangle_points(
                primitive.center_xy,
                (primitive.size_xy[0] * 0.88, primitive.size_xy[1] * 0.78),
                primitive.yaw_deg,
            )
            roof = _rectangle_points(
                primitive.center_xy,
                (primitive.size_xy[0] * 0.78, primitive.size_xy[1] * 0.66),
                primitive.yaw_deg,
            )
            opening = _rectangle_points(
                primitive.center_xy,
                (primitive.size_xy[0] * 0.68, primitive.size_xy[1] * 0.34),
                primitive.yaw_deg,
            )
            draw.polygon(shoulder, fill=_height_color(0.20))
            draw.polygon(roof, fill=_height_color(0.25))
            draw.polygon(opening, fill=(5, 15, 21))
            draw.line((*opening, opening[0]), fill=(176, 190, 195), width=1)
        line_width = 2 if primary_surface else 1
        if primitive.category in {
            "central_slope",
            "trapezoid_slope",
            "fortress_slope",
            "central_top",
            "trapezoid_top",
            "fortress_top",
        }:
            outline = (188, 201, 204)
        draw.line((*points, points[0]), fill=outline, width=line_width)
        if primitive.category == "rough":
            angle = math.radians(primitive.yaw_deg)
            along = (math.cos(angle), math.sin(angle))
            across = (-math.sin(angle), math.cos(angle))
            half_x = primitive.size_xy[0] / 2
            half_y = primitive.size_xy[1] / 2
            ridge = -half_x + 0.12
            ridge_index = 0
            while ridge < half_x:
                center_x = primitive.center_xy[0] + along[0] * ridge
                center_y = primitive.center_xy[1] + along[1] * ridge
                draw.line(
                    (
                        *_to_pixel(
                            center_x - across[0] * half_y,
                            center_y - across[1] * half_y,
                        ),
                        *_to_pixel(
                            center_x + across[0] * half_y,
                            center_y + across[1] * half_y,
                        ),
                    ),
                    fill=_height_color(0.07 if ridge_index % 2 == 0 else 0.04),
                    width=2,
                )
                ridge += 0.24
                ridge_index += 1

    for primitive in arena.config.terrain:
        slope_label = SLOPE_LABELS.get(primitive.name)
        if slope_label is not None:
            _draw_slope_arrow(draw, primitive, slope_label, label_font)

    for primitive in arena.config.terrain:
        label = FEATURE_LABELS.get(primitive.name)
        if label is not None:
            label_xy = primitive.center_xy
            if primitive.name == "central_highland":
                label_xy = (0.0, 2.2)
            center = _to_pixel(*label_xy)
            draw.text(
                center,
                label,
                fill=(222, 231, 234),
                font=label_font,
                anchor="mm",
                stroke_width=2,
                stroke_fill=(22, 31, 36),
            )
    for team in (Team.RED, Team.BLUE):
        color = RED if team == Team.RED else BLUE
        base_center_tensor = arena.base_centers(
            device="cpu",
            dtype=torch.float32,
        )[team]
        base_center = (
            float(base_center_tensor[0]),
            float(base_center_tensor[1]),
        )
        base_zone = _regular_polygon(base_center, (1.05, 0.90), 6)
        draw.line((*base_zone, base_zone[0]), fill=color, width=2)
        outpost_center = arena.outpost_centers(
            device="cpu",
            dtype=torch.float32,
        )[team]
        outpost_zone = _regular_polygon(
            (float(outpost_center[0]), float(outpost_center[1])),
            (0.95, 1.05),
            6,
        )
        draw.line((*outpost_zone, outpost_zone[0]), fill=color, width=2)
        supply_center_tensor = arena.supply_centers(
            device="cpu",
            dtype=torch.float32,
        )[team]
        supply_center = (
            float(supply_center_tensor[0]),
            float(supply_center_tensor[1]),
        )
        supply_half = (1.25, 1.05)
        supply_box = (
            *_to_pixel(
                supply_center[0] - supply_half[0],
                supply_center[1] + supply_half[1],
            ),
            *_to_pixel(
                supply_center[0] + supply_half[0],
                supply_center[1] - supply_half[1],
            ),
        )
        draw.rounded_rectangle(supply_box, radius=4, outline=color, width=2)
        draw.text(
            _to_pixel(*supply_center),
            "SUPPLY",
            fill=(190, 205, 212),
            font=label_font,
            anchor="mm",
        )
        pad_center_tensor = arena.aerial_pad_centers(
            device="cpu",
            dtype=torch.float32,
        )[team]
        pad_center = (
            float(pad_center_tensor[0]),
            float(pad_center_tensor[1]),
        )
        pad_outline = _regular_polygon(pad_center, (0.95, 0.85), 8)
        draw.line((*pad_outline, pad_outline[0]), fill=color, width=2)
        draw.text(
            _to_pixel(*pad_center),
            "PAD",
            fill=(190, 205, 212),
            font=label_font,
            anchor="mm",
        )

    for x_min, x_max, y_min, y_max in arena.config.obstacles:
        upper_left = _to_pixel(x_min, y_max)
        lower_right = _to_pixel(x_max, y_min)
        draw.rectangle(
            (*upper_left, *lower_right),
            fill=(52, 67, 76),
            outline=(125, 149, 161),
            width=2,
        )
    draw.text(
        _to_pixel(0.0, 0.0),
        "ENERGY CORE",
        fill=(222, 231, 234),
        font=label_font,
        anchor="mm",
        stroke_width=2,
        stroke_fill=(22, 31, 36),
    )
    for label_xy in ((-1.45, 0.45), (1.45, -0.45)):
        draw.text(
            _to_pixel(*label_xy),
            "ASSEMBLY · 12°–45°",
            fill=(226, 235, 237),
            font=label_font,
            anchor="mm",
            stroke_width=2,
            stroke_fill=(22, 31, 36),
        )
    _draw_height_legend(draw, legend_font)


def main() -> None:
    args = _parser().parse_args()
    if args.duration_s <= 0 or args.duration_s > constants.MATCH_DURATION_S:
        raise ValueError(f"--duration-s must be in (0, {constants.MATCH_DURATION_S:g}]")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided")
    if args.capture_every <= 0 or args.fps <= 0:
        raise ValueError("--capture-every and --fps must be positive")
    if Path(args.output).suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            'Pillow is required: python -m pip install -e ".[visualization]"'
        ) from error

    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=1,
            device=args.device,
            seed=args.seed,
            validate_referee=False,
        )
    )
    opponent = TacticalScriptedOpponent(arena=environment.arena)
    environment.reset(seed=args.seed)
    total_steps = (
        args.steps
        if args.steps is not None
        else math.ceil(args.duration_s / environment.config.policy_dt_s)
    )
    replay_speed = environment.config.policy_dt_s * args.capture_every * args.fps
    roles = unit_roles(environment.game.device).cpu()
    teams = unit_teams(environment.game.device).cpu()
    title_font = _font(22, bold=True)
    score_font = _font(15, bold=True)
    unit_font = _font(12, bold=True)
    small_font = _font(11)
    terrain_font = _font(9, bold=True)
    static_image = Image.new("RGB", CANVAS, (6, 15, 23))
    _draw_static(static_image, environment.arena, terrain_font, small_font)
    trails: deque[torch.Tensor] = deque(maxlen=22)
    frames: list[object] = []
    radii_m = torch.where(
        roles == Role.BASE,
        torch.full_like(roles, 0.95, dtype=torch.float32),
        torch.where(
            roles == Role.OUTPOST,
            torch.full_like(roles, 0.40, dtype=torch.float32),
            torch.full_like(roles, 0.40, dtype=torch.float32),
        ),
    )
    ground_pair = (
        (roles[:, None] != Role.AERIAL)
        & (roles[None, :] != Role.AERIAL)
        & ~torch.eye(constants.UNIT_COUNT, dtype=torch.bool)
    )
    minimum_clearance_m = torch.inf

    simulated_steps = 0
    for policy_step in range(total_steps):
        actions = opponent.act(environment.game, environment.world)
        source_xy = environment.world.position_xy[0].detach().cpu()
        target_slots = resolve_target_slots(actions.target)[0]
        firing = actions.fire[0] & (target_slots >= 0)
        result = environment.step(actions)
        simulated_steps += 1

        xy = environment.world.position_xy[0].detach().cpu()
        distance_matrix = torch.cdist(xy, xy)
        clearance = distance_matrix - radii_m[:, None] - radii_m[None, :]
        minimum_clearance_m = min(
            minimum_clearance_m,
            float(clearance[ground_pair].min()),
        )
        capture_frame = (
            policy_step == 0
            or (policy_step + 1) % args.capture_every == 0
            or policy_step + 1 == total_steps
            or bool(result.terminated[0])
        )
        if not capture_frame:
            continue

        yaw = environment.world.yaw[0].detach().cpu()
        hp = environment.game.hp[0].detach().cpu()
        max_hp = environment.game.max_hp[0].detach().cpu()
        alive = environment.game.alive[0].detach().cpu()
        weak = environment.game.weak[0].detach().cpu()
        trails.append(xy.clone())

        image = static_image.copy()
        draw = ImageDraw.Draw(image)
        draw.text((MARGIN_X, 18), "RM-CORTEX // TORCH ARENA", fill=(236, 246, 250), font=title_font)
        draw.text((MARGIN_X, 48), "RED", fill=RED, font=score_font)
        draw.text((CANVAS[0] - MARGIN_X, 48), "BLUE", fill=BLUE, font=score_font, anchor="ra")

        elapsed = float(environment.game.elapsed_s[0])
        remaining_s = max(
            0,
            math.ceil(constants.MATCH_DURATION_S - elapsed - 1.0e-6),
        )
        clock = f"{remaining_s // 60:02d}:{remaining_s % 60:02d}"
        red_base = int(environment.game.hp[0, Role.BASE])
        blue_base = int(environment.game.hp[0, constants.ROLES_PER_TEAM + Role.BASE])
        red_outpost = int(environment.game.hp[0, Role.OUTPOST])
        blue_outpost = int(environment.game.hp[0, constants.ROLES_PER_TEAM + Role.OUTPOST])
        red_coin = int(environment.game.team_coin[0, Team.RED])
        blue_coin = int(environment.game.team_coin[0, Team.BLUE])
        air_parts = []
        for team, label in ((Team.RED, "R"), (Team.BLUE, "B")):
            if bool(environment.game.aerial_support_active[0, team]):
                air_state = "ON"
            elif bool(environment.game.aerial_on_pad[0, team]):
                air_state = "PAD"
            else:
                air_state = "PAUSE"
            air_bank = float(environment.game.aerial_support_bank_s[0, team])
            air_parts.append(f"{label} {air_state}/{air_bank:.0f}s")
        air_status = " · ".join(air_parts)
        score = (
            f"{clock}   "
            f"{red_base:4d} B · {red_outpost:4d} O · {red_coin:3d} C   "
            f"—   {blue_base:4d} B · {blue_outpost:4d} O · {blue_coin:3d} C"
        )
        draw.text((CANVAS[0] // 2, 49), score, fill=(194, 211, 220), font=score_font, anchor="ma")
        draw.text(
            (CANVAS[0] // 2, 67),
            (
                f"TACTICAL SCRIPTED BASELINE · {tactical_phase_label(elapsed)} · "
                f"AIR {air_status} · {replay_speed:.0f}×"
            ),
            fill=(114, 139, 151),
            font=small_font,
            anchor="mm",
        )

        if len(trails) > 1:
            history = torch.stack(tuple(trails))
            for unit in range(constants.UNIT_COUNT):
                if int(roles[unit]) in (Role.BASE, Role.OUTPOST):
                    continue
                color = RED if int(teams[unit]) == Team.RED else BLUE
                points = [
                    _to_pixel(float(position[0]), float(position[1]))
                    for position in history[:, unit]
                ]
                if len(points) > 1:
                    draw.line(points, fill=tuple(channel // 3 for channel in color), width=2)

        target_safe = torch.clamp(target_slots, min=0)
        target_xy = environment.world.position_xy[0, target_safe].detach().cpu()
        for unit in torch.nonzero(firing, as_tuple=False).squeeze(-1).tolist():
            draw.line(
                (
                    *_to_pixel(float(source_xy[unit, 0]), float(source_xy[unit, 1])),
                    *_to_pixel(float(target_xy[unit, 0]), float(target_xy[unit, 1])),
                ),
                fill=(255, 205, 86),
                width=2,
            )

        for unit in range(constants.UNIT_COUNT):
            x, y = _to_pixel(float(xy[unit, 0]), float(xy[unit, 1]))
            role = int(roles[unit])
            color = RED if int(teams[unit]) == Team.RED else BLUE
            if not bool(alive[unit]):
                color = tuple(channel // 4 for channel in color)
            radius = _meters_to_pixels(float(radii_m[unit]))
            if role == Role.AERIAL:
                radius = _meters_to_pixels(0.32)
            outline = (255, 190, 82) if bool(weak[unit]) else (235, 244, 248)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=outline,
                width=2,
            )
            heading_length = radius + 9
            heading_end = (
                round(x + torch.cos(yaw[unit]).item() * heading_length),
                round(y - torch.sin(yaw[unit]).item() * heading_length),
            )
            draw.line((x, y, *heading_end), fill=(255, 255, 255), width=2)
            draw.text(
                (x, y),
                ROLE_LABELS[role],
                fill=(255, 255, 255),
                font=unit_font,
                anchor="mm",
            )
            if max_hp[unit] > 0:
                fraction = max(0.0, min(1.0, float(hp[unit] / max_hp[unit])))
                bar_left = x - radius
                bar_right = x + radius
                bar_y = y + radius + 6
                draw.rectangle(
                    (bar_left, bar_y, bar_right, bar_y + 4),
                    fill=(39, 47, 52),
                )
                draw.rectangle(
                    (
                        bar_left,
                        bar_y,
                        round(bar_left + (bar_right - bar_left) * fraction),
                        bar_y + 4,
                    ),
                    fill=(84, 224, 145),
                )

        if bool(environment.game.done[0]):
            winner = int(environment.game.winner[0])
            winner_text = {
                int(Winner.RED): "RED WINS",
                int(Winner.BLUE): "BLUE WINS",
                int(Winner.DRAW): "DRAW",
            }.get(winner, "MATCH ENDED")
            winner_color = (
                RED if winner == Winner.RED else BLUE if winner == Winner.BLUE else (210, 221, 226)
            )
            banner = (CANVAS[0] // 2 - 92, MARGIN_TOP + 12, CANVAS[0] // 2 + 92, MARGIN_TOP + 47)
            draw.rounded_rectangle(
                banner,
                radius=8,
                fill=(6, 15, 23),
                outline=winner_color,
                width=2,
            )
            draw.text(
                (CANVAS[0] // 2, MARGIN_TOP + 29),
                winner_text,
                fill=winner_color,
                font=score_font,
                anchor="mm",
            )

        draw.text(
            (CANVAS[0] // 2, CANVAS[1] - 16),
            "H hero · E engineer · I infantry · A aerial · S sentry · B base · O outpost",
            fill=(144, 164, 175),
            font=small_font,
            anchor="ms",
        )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

        if bool(result.terminated[0]):
            break

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoder = subprocess.Popen(
            (
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{CANVAS[0]}x{CANVAS[1]}",
                "-framerate",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg with libx264 is required to export MP4") from error
    assert encoder.stdin is not None
    assert encoder.stderr is not None
    try:
        for frame in frames:
            encoder.stdin.write(frame.convert("RGB").tobytes())
    except BrokenPipeError as error:
        encoder.stdin.close()
        message = encoder.stderr.read().decode(errors="replace")
        encoder.wait()
        raise RuntimeError(f"ffmpeg failed while encoding MP4: {message}") from error
    encoder.stdin.close()
    return_code = encoder.wait()
    encoder_error = encoder.stderr.read().decode(errors="replace")
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while encoding MP4: {encoder_error}")
    print(f"Saved Torch visualization to {destination.resolve()}")
    simulated_s = simulated_steps * environment.config.policy_dt_s
    winner_name = {
        int(Winner.UNDECIDED): "undecided",
        int(Winner.RED): "red",
        int(Winner.BLUE): "blue",
        int(Winner.DRAW): "draw",
    }[int(environment.game.winner[0])]
    print(
        f"Simulated {simulated_s:.1f}s in {len(frames)} frames "
        f"({len(frames) / args.fps:.1f}s playback); winner={winner_name}"
    )
    print(f"Minimum ground-unit clearance: {minimum_clearance_m:.4f} m")
    if minimum_clearance_m < -1.0e-4:
        raise RuntimeError("visualized rollout contains overlapping ground units")


if __name__ == "__main__":
    main()
