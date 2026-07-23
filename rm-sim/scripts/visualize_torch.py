#!/usr/bin/env python3
"""Export a scripted RM-Cortex match as a dependency-light GIF."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import torch

from rm_referee import constants
from rm_referee.schema import Role, Team, unit_roles, unit_teams
from rm_world import ScriptedOpponent, TorchEnvConfig, TorchRMArena
from rm_world.geometry import resolve_target_slots


ROLE_LABELS = ("H", "E", "I3", "I4", "A", "S", "B", "O")
RED = (255, 58, 83)
BLUE = (20, 136, 255)
CANVAS = (1120, 650)
MARGIN_X = 44
MARGIN_TOP = 74
MARGIN_BOTTOM = 42


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Number of 0.2s policy steps.")
    parser.add_argument("--fps", type=int, default=10, help="GIF playback frame rate.")
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


def _draw_static(draw: object, arena: object) -> None:
    left, top = _to_pixel(-14.0, 7.5)
    right, bottom = _to_pixel(14.0, -7.5)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=8,
        fill=(14, 35, 46),
        outline=(205, 222, 232),
        width=2,
    )
    center_top = _to_pixel(0.0, 7.5)
    center_bottom = _to_pixel(0.0, -7.5)
    draw.line((*center_top, *center_bottom), fill=(142, 164, 176), width=1)
    for x_min, x_max, y_min, y_max in arena.config.obstacles:
        upper_left = _to_pixel(x_min, y_max)
        lower_right = _to_pixel(x_max, y_min)
        draw.rectangle(
            (*upper_left, *lower_right),
            fill=(52, 67, 76),
            outline=(125, 149, 161),
            width=2,
        )
        for offset in range(-80, 160, 12):
            x0 = max(upper_left[0], upper_left[0] + offset)
            y0 = max(upper_left[1], upper_left[1] - offset)
            x1 = min(lower_right[0], x0 + 75)
            y1 = min(lower_right[1], y0 + 75)
            if x0 <= x1 and y0 <= y1:
                draw.line((x0, y1, x1, y0), fill=(72, 88, 98), width=1)


def main() -> None:
    args = _parser().parse_args()
    if args.steps <= 0 or args.fps <= 0:
        raise ValueError("--steps and --fps must be positive")
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
    opponent = ScriptedOpponent()
    environment.reset(seed=args.seed)
    roles = unit_roles(environment.game.device)
    teams = unit_teams(environment.game.device)
    title_font = _font(22, bold=True)
    score_font = _font(15, bold=True)
    unit_font = _font(12, bold=True)
    small_font = _font(11)
    trails: deque[torch.Tensor] = deque(maxlen=22)
    frames: list[object] = []

    for _ in range(args.steps):
        actions = opponent.act(environment.game, environment.world)
        source_xy = environment.world.position_xy[0].detach().cpu()
        target_slots = resolve_target_slots(actions.target)[0]
        firing = actions.fire[0] & (target_slots >= 0)
        result = environment.step(actions)
        if bool(result.terminated[0]):
            environment.reset(torch.tensor([0], device=environment.game.device))

        xy = environment.world.position_xy[0].detach().cpu()
        yaw = environment.world.yaw[0].detach().cpu()
        hp = environment.game.hp[0].detach().cpu()
        max_hp = environment.game.max_hp[0].detach().cpu()
        alive = environment.game.alive[0].detach().cpu()
        trails.append(xy.clone())

        image = Image.new("RGB", CANVAS, (6, 15, 23))
        draw = ImageDraw.Draw(image)
        _draw_static(draw, environment.arena)
        draw.text((MARGIN_X, 18), "RM-CORTEX // TORCH ARENA", fill=(236, 246, 250), font=title_font)
        draw.text((MARGIN_X, 48), "RED", fill=RED, font=score_font)
        draw.text((CANVAS[0] - MARGIN_X, 48), "BLUE", fill=BLUE, font=score_font, anchor="ra")

        elapsed = float(environment.game.elapsed_s[0])
        red_base = int(environment.game.hp[0, Role.BASE])
        blue_base = int(environment.game.hp[0, constants.ROLES_PER_TEAM + Role.BASE])
        red_coin = int(environment.game.team_coin[0, Team.RED])
        blue_coin = int(environment.game.team_coin[0, Team.BLUE])
        score = (
            f"t={elapsed:05.1f}s   "
            f"{red_base:4d} HP / {red_coin:3d} C   "
            f"—   {blue_base:4d} HP / {blue_coin:3d} C"
        )
        draw.text((CANVAS[0] // 2, 49), score, fill=(194, 211, 220), font=score_font, anchor="ma")

        if len(trails) > 1:
            history = torch.stack(tuple(trails))
            for unit in range(constants.UNIT_COUNT):
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
            radius = 13
            if role == Role.BASE:
                radius = 24
            elif role == Role.OUTPOST:
                radius = 19
            elif role == Role.SENTRY:
                radius = 16
            elif role == Role.AERIAL:
                radius = 11
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(235, 244, 248),
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

        draw.text(
            (CANVAS[0] // 2, CANVAS[1] - 16),
            "H hero · E engineer · I infantry · A aerial · S sentry · B base · O outpost",
            fill=(144, 164, 175),
            font=small_font,
            anchor="ms",
        )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

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


if __name__ == "__main__":
    main()
