from __future__ import annotations

import pytest
import torch

from rm_referee import GameState, RandomTape, Referee
from rm_referee.schema import Role, Team, Weapon, slot
from rm_world import (
    ArenaGeometry,
    HitModel,
    HitModelConfig,
    KinematicState,
    TorchRuleBackend,
    WorldActions,
)


def test_terrain_height_models_field_crown_highland_and_fly_ramp() -> None:
    arena = ArenaGeometry()
    samples = torch.tensor(
        (
            (13.0, 7.5),
            (13.0, 0.0),
            (0.0, 0.0),
            (-4.20, -6.2),
            (-2.20, -6.2),
        )
    )

    height = arena.terrain_height(samples)

    assert height[0] == pytest.approx(0.0, abs=1.0e-6)
    assert height[1] > height[0]
    assert height[2] > height[1]
    assert height[4] - height[3] == pytest.approx(0.325, abs=0.01)


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


def backend_hit_draws() -> int:
    return 16 * 2 + 2
