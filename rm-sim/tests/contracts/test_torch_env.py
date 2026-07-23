from __future__ import annotations

from dataclasses import fields

import torch

from rm_referee.schema import Role, Team, slot
from rm_referee.state import GameState
from rm_world import ScriptedOpponent, TorchEnvConfig, TorchRMArena, WorldActions


def test_end_to_end_torch_environment_returns_masks_rewards_and_events() -> None:
    env = TorchRMArena(TorchEnvConfig(num_envs=2, seed=7))
    actions = ScriptedOpponent().act(env.game, env.world)

    result = env.step(actions)

    assert result.observation.agents.shape == (2, 16, 34)
    assert result.observation.entities.shape == (2, 16, 16, 15)
    assert result.observation.target_mask.shape == (2, 16, 9)
    assert result.reward.shape == (2, 16)
    assert torch.isfinite(result.observation.agents).all()
    assert torch.isfinite(result.reward).all()
    engineer = slot(Team.RED, Role.ENGINEER)
    assert not result.observation.fire_mask[0, engineer]


def test_seeded_torch_environments_roll_out_identically() -> None:
    left = TorchRMArena(TorchEnvConfig(num_envs=2, seed=11))
    right = TorchRMArena(TorchEnvConfig(num_envs=2, seed=11))
    left_actions = WorldActions.zeros(left.game)
    right_actions = WorldActions.zeros(right.game)
    shooter = slot(Team.RED, Role.INFANTRY_3)
    left.game.ammo[:, shooter, 0] = 10
    right.game.ammo[:, shooter, 0] = 10
    left_actions.target[:, shooter] = 2
    right_actions.target[:, shooter] = 2
    left_actions.fire[:, shooter] = True
    right_actions.fire[:, shooter] = True

    left.step(left_actions)
    right.step(right_actions)

    for field in fields(GameState):
        assert torch.equal(
            getattr(left.game, field.name),
            getattr(right.game, field.name),
        )
    for field in fields(type(left.world)):
        assert torch.equal(
            getattr(left.world, field.name),
            getattr(right.world, field.name),
        )


def test_partial_reset_only_replaces_selected_environment() -> None:
    env = TorchRMArena(TorchEnvConfig(num_envs=3))
    env.game.team_coin[0] = 1
    env.game.team_coin[1] = 2
    env.game.team_coin[2] = 3

    env.reset(torch.tensor([1]))

    assert env.game.team_coin[0, 0] == 1
    assert env.game.team_coin[1, 0] == 400
    assert env.game.team_coin[2, 0] == 3
