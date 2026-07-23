from __future__ import annotations

import math

import torch

from rm_referee.schema import Role, Team, slot
from rm_train import (
    MAPPOConfig,
    MAPPOTrainingRunner,
    SharedMAPPOPolicy,
    decode_policy_actions,
    evaluate_against_scripted,
)
from rm_world import TorchEnvConfig, TorchRMArena


def test_policy_sampling_respects_dynamic_masks_and_decodes_roles() -> None:
    torch.manual_seed(3)
    environment = TorchRMArena(TorchEnvConfig(num_envs=2, seed=3))
    observation = environment.observe()
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)

    step = policy.act(
        observation.agents,
        observation.central,
        observation.target_mask,
        observation.fire_mask,
    )
    selected_is_valid = torch.gather(
        observation.target_mask,
        2,
        step.action.target[..., None],
    ).squeeze(-1)
    assert torch.all(selected_is_valid)
    assert not torch.any(step.action.fire & ~observation.fire_mask)
    assert step.log_prob.shape == (2, 16)
    assert step.value.shape == (2, 16)

    actions = decode_policy_actions(
        environment.game,
        environment.world,
        step.action,
    )
    actions.validate(environment.game)
    aerial = slot(Team.RED, Role.AERIAL)
    base = slot(Team.RED, Role.BASE)
    assert torch.equal(actions.vertical_velocity[:, base], torch.zeros(2))
    assert torch.isfinite(actions.vertical_velocity[:, aerial]).all()
    assert actions.radar_target.shape == (2, 2)


def test_tiny_mappo_update_is_finite_and_changes_parameters() -> None:
    config = MAPPOConfig(
        num_envs=2,
        rollout_steps=2,
        total_updates=1,
        update_epochs=1,
        minibatch_size=64,
        hidden_dim=32,
        role_embedding_dim=8,
        device="cpu",
    )
    runner = MAPPOTrainingRunner(config)
    before = [parameter.detach().clone() for parameter in runner.policy.parameters()]

    progress = runner.train_update()

    assert progress.update == 1
    assert progress.environment_steps == 4
    assert all(math.isfinite(value) for value in progress.metrics.as_dict().values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, runner.policy.parameters(), strict=True)
    )


def test_evaluation_reports_unfinished_short_smoke_rollout() -> None:
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)

    report = evaluate_against_scripted(
        policy,
        num_envs=1,
        max_policy_steps=1,
        device="cpu",
    )

    assert report.policy_steps == 1
    assert report.completed_episodes + report.unfinished_environments == 1
    assert math.isfinite(report.mean_red_return)
