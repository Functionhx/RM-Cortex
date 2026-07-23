#!/usr/bin/env python3
"""Export a scripted RM-Cortex match as a dependency-light GIF."""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path

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
CANVAS = (1120, 650)
MARGIN_X = 44
MARGIN_TOP = 74
MARGIN_BOTTOM = 42
TERRAIN_COLORS = {
    "central": (73, 87, 96),
    "assembly": (83, 94, 101),
    "trapezoid": (68, 82, 90),
    "ramp": (92, 103, 109),
    "road": (57, 72, 80),
    "fly_ramp": (101, 108, 108),
    "rough": (50, 64, 72),
    "fortress": (88, 96, 98),
    "tunnel": (8, 18, 24),
}
FEATURE_LABELS = {
    "central_highland": "CENTRAL HIGH · 0.25–0.40m",
    "red_trapezoid_highland": "TRAPEZOID HIGH",
    "blue_trapezoid_highland": "TRAPEZOID HIGH",
    "red_road": "ROAD · 11°/15°",
    "blue_road": "ROAD · 11°/15°",
    "red_fly_ramp": "FLY RAMP · 17°",
    "blue_fly_ramp": "FLY RAMP · 17°",
    "red_rough_road": "BUMPS",
    "blue_rough_road": "BUMPS",
    "red_fortress": "FORT",
    "blue_fortress": "FORT",
    "red_tunnel": "TUNNEL",
    "blue_tunnel": "TUNNEL",
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
    parser.add_argument("--fps", type=int, default=20, help="GIF playback frame rate.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/torch_demo.gif")
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


def _draw_static(draw: object, arena: object, label_font: object) -> None:
    left, top = _to_pixel(-14.0, 7.5)
    right, bottom = _to_pixel(14.0, -7.5)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=8,
        fill=(14, 35, 46),
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
        corners = arena.terrain_corners(
            primitive,
            device="cpu",
            dtype=torch.float32,
        )
        points = [_to_pixel(float(point[0]), float(point[1])) for point in corners]
        team_color = RED if primitive.team == Team.RED else BLUE
        outline = (145, 158, 165) if primitive.team is None else team_color
        draw.polygon(
            points,
            fill=TERRAIN_COLORS[primitive.category],
            outline=outline,
        )
        draw.line((*points, points[0]), fill=outline, width=2)
        if primitive.elevation_start_m != primitive.elevation_end_m:
            angle = torch.deg2rad(torch.tensor(primitive.yaw_deg))
            direction = torch.stack((torch.cos(angle), torch.sin(angle)))
            low = torch.tensor(primitive.center_xy) - direction * primitive.size_xy[0] * 0.32
            high = torch.tensor(primitive.center_xy) + direction * primitive.size_xy[0] * 0.32
            draw.line(
                (
                    *_to_pixel(float(low[0]), float(low[1])),
                    *_to_pixel(float(high[0]), float(high[1])),
                ),
                fill=(228, 235, 237),
                width=2,
            )
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

    for team, sign in ((Team.RED, -1.0), (Team.BLUE, 1.0)):
        color = RED if team == Team.RED else BLUE
        base_zone = _regular_polygon((12.2 * sign, 0.0), (1.65, 1.75), 6)
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
        supply_center = (11.9 * sign, 5.6 * sign)
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


def main() -> None:
    args = _parser().parse_args()
    if args.duration_s <= 0 or args.duration_s > constants.MATCH_DURATION_S:
        raise ValueError(f"--duration-s must be in (0, {constants.MATCH_DURATION_S:g}]")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided")
    if args.capture_every <= 0 or args.fps <= 0:
        raise ValueError("--capture-every and --fps must be positive")
    if Path(args.output).suffix.lower() != ".gif":
        raise ValueError("--output must end in .gif")

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

        image = Image.new("RGB", CANVAS, (6, 15, 23))
        draw = ImageDraw.Draw(image)
        _draw_static(draw, environment.arena, terrain_font)
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
    first, *remaining = frames
    first.save(
        destination,
        save_all=True,
        append_images=remaining,
        duration=round(1000 / args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
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
