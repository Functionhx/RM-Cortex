from __future__ import annotations

import math

import pytest
import torch

from rm_referee import GameState
from rm_referee.schema import Role, Team, slot
from rm_world import KinematicCommands, KinematicState, KinematicWorld


def test_body_velocity_is_rotated_and_batched() -> None:
    game = GameState.create(2)
    world = KinematicState.zeros(game)
    commands = KinematicCommands.zeros(game)
    hero = slot(Team.RED, Role.HERO)
    world.yaw[:, hero] = math.pi / 2
    commands.body_velocity_xy[:, hero, 0] = 1

    next_world = KinematicWorld().step(world, game, commands, dt=0.1)

    assert torch.allclose(next_world.position_xy[:, hero, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        next_world.position_xy[:, hero, 1],
        torch.full((2,), 0.1),
        atol=1e-6,
    )


def test_buildings_do_not_move_and_field_bounds_are_enforced() -> None:
    game = GameState.create(1)
    world = KinematicState.zeros(game)
    commands = KinematicCommands.zeros(game)
    base = slot(Team.RED, Role.BASE)
    hero = slot(Team.RED, Role.HERO)
    commands.body_velocity_xy[0, base, 0] = 3
    commands.body_velocity_xy[0, hero, 0] = 3
    world.position_xy[0, hero, 0] = 13.9

    next_world = KinematicWorld().step(world, game, commands, dt=1)

    assert next_world.position_xy[0, base, 0] == 0
    assert next_world.position_xy[0, hero, 0] == pytest.approx(14)
