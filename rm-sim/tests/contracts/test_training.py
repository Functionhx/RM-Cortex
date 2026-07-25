from __future__ import annotations

import math
from pathlib import Path

import torch

from rm_referee.schema import Role, Team, slot
from rm_train import (
    MAPPOConfig,
    MAPPOTrainingRunner,
    PolicyAction,
    RolloutBatch,
    SharedMAPPOPolicy,
    decode_policy_actions,
    evaluate_against_scripted,
)
from rm_world import TorchEnvConfig, TorchRMArena


def test_rollout_flatten_preserves_entity_axes() -> None:
    time_steps, num_envs, num_agents, entity_dim = 2, 3, 16, 7
    rollout_shape = (time_steps, num_envs, num_agents)
    actions = PolicyAction(
        motion=torch.zeros(*rollout_shape, 4),
        target=torch.zeros(rollout_shape, dtype=torch.long),
        fire=torch.zeros(rollout_shape, dtype=torch.bool),
        common=torch.zeros(*rollout_shape, 2, dtype=torch.bool),
        special=torch.zeros(*rollout_shape, 3, dtype=torch.bool),
        mode=torch.zeros(rollout_shape, dtype=torch.long),
        radar_target=torch.zeros(rollout_shape, dtype=torch.long),
        radar_offset=torch.zeros(*rollout_shape, 2),
    )
    entities = torch.arange(
        time_steps * num_envs * num_agents * num_agents * entity_dim,
    ).reshape(*rollout_shape, num_agents, entity_dim)
    entity_mask = torch.ones(*rollout_shape, num_agents, dtype=torch.bool)
    rollout = RolloutBatch(
        observations=torch.zeros(*rollout_shape, 34),
        entities=entities,
        entity_mask=entity_mask,
        central_state=torch.zeros(time_steps, num_envs, 117),
        target_mask=torch.ones(*rollout_shape, 9, dtype=torch.bool),
        fire_mask=torch.ones(rollout_shape, dtype=torch.bool),
        actions=actions,
        old_log_prob=torch.zeros(rollout_shape),
        old_value=torch.zeros(rollout_shape),
        returns=torch.zeros(rollout_shape),
        advantages=torch.zeros(rollout_shape),
    )

    flat = rollout.flatten()

    assert flat.entities.shape == (time_steps * num_envs * num_agents, num_agents, entity_dim)
    assert flat.entity_mask.shape == (time_steps * num_envs * num_agents, num_agents)
    assert torch.equal(flat.entities[16], entities[0, 1, 0])
    assert torch.equal(flat.entity_mask, torch.ones_like(flat.entity_mask))


def test_policy_sampling_respects_dynamic_masks_and_decodes_roles() -> None:
    torch.manual_seed(3)
    environment = TorchRMArena(TorchEnvConfig(num_envs=2, seed=3))
    observation = environment.observe()
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)

    step = policy.act(
        observation.agents,
        observation.entities,
        observation.entity_mask,
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
    agent_ids = policy.default_agent_ids(
        observation.agents.shape[:-1],
        observation.agents.device,
    )
    evaluated = policy.evaluate_actions(
        observation.agents,
        observation.entities,
        observation.entity_mask,
        observation.central,
        observation.target_mask,
        observation.fire_mask,
        step.action,
        agent_ids,
    )
    assert torch.allclose(evaluated.log_prob, step.log_prob)

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


def test_tiny_mappo_update_is_finite_and_changes_parameters(tmp_path: Path) -> None:
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
    entity_before = {
        name: parameter.detach().clone()
        for name, parameter in runner.policy.named_parameters()
        if name.startswith("entity_")
    }

    progress = runner.train_update()

    assert progress.update == 1
    assert progress.environment_steps == 4
    assert all(math.isfinite(value) for value in progress.metrics.as_dict().values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, runner.policy.parameters(), strict=True)
    )
    assert any(
        not torch.equal(previous, dict(runner.policy.named_parameters())[name])
        for name, previous in entity_before.items()
    )
    checkpoint = runner.save_checkpoint(tmp_path / "entity_policy.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["policy_schema_version"] == runner.policy.CHECKPOINT_SCHEMA_VERSION
    assert payload["policy_kwargs"] == runner.policy.architecture_kwargs()


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
