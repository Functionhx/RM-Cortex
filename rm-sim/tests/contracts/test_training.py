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
        radar_report=torch.zeros(*rollout_shape, 2),
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


def test_radar_report_decode_is_absolute_and_independent_of_target_truth() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=2, seed=7))
    agent_shape = (environment.game.num_envs, 16)
    radar_slots = [
        slot(Team.RED, Role.OUTPOST),
        slot(Team.BLUE, Role.OUTPOST),
    ]
    radar_report = torch.zeros(*agent_shape, 2)
    radar_report[:, radar_slots] = torch.tensor(
        [
            [[0.25, -0.5], [-0.75, 1.25]],
            [[1.5, -2.0], [-0.1, 0.8]],
        ]
    )
    radar_target = torch.zeros(agent_shape, dtype=torch.long)
    radar_target[:, radar_slots[0]] = slot(Team.BLUE, Role.HERO)
    radar_target[:, radar_slots[1]] = slot(Team.RED, Role.SENTRY)
    policy_action = PolicyAction(
        motion=torch.zeros(*agent_shape, 4),
        target=torch.zeros(agent_shape, dtype=torch.long),
        fire=torch.zeros(agent_shape, dtype=torch.bool),
        common=torch.zeros(*agent_shape, 2, dtype=torch.bool),
        special=torch.zeros(*agent_shape, 3, dtype=torch.bool),
        mode=torch.zeros(agent_shape, dtype=torch.long),
        radar_target=radar_target,
        radar_report=radar_report,
    )

    decoded_before = decode_policy_actions(
        environment.game,
        environment.world,
        policy_action,
    )
    expected = torch.tanh(radar_report[:, radar_slots]) * torch.tensor([14.0, 7.5])
    assert torch.allclose(decoded_before.radar_report_xy, expected)
    assert torch.equal(
        decoded_before.radar_target,
        radar_target[:, radar_slots],
    )

    environment.world.position_xy.add_(1000.0)
    decoded_after = decode_policy_actions(
        environment.game,
        environment.world,
        policy_action,
    )
    assert torch.equal(decoded_after.radar_report_xy, decoded_before.radar_report_xy)


def test_radar_policy_can_choose_not_to_report() -> None:
    environment = TorchRMArena(TorchEnvConfig(num_envs=1))
    observation = environment.observe()
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    step = policy.act(
        observation.agents,
        observation.entities,
        observation.entity_mask,
        observation.central,
        observation.target_mask,
        observation.fire_mask,
        deterministic=True,
    )
    radar_slots = [
        slot(Team.RED, Role.OUTPOST),
        slot(Team.BLUE, Role.OUTPOST),
    ]
    step.action.radar_target[:, radar_slots] = 16

    actions = decode_policy_actions(
        environment.game,
        environment.world,
        step.action,
    )

    assert torch.equal(actions.radar_target, torch.full((1, 2), -1))
    actions.validate(environment.game)

    agent_ids = policy.default_agent_ids(
        observation.agents.shape[:-1],
        observation.agents.device,
    )
    without_coordinates = policy.evaluate_actions(
        observation.agents,
        observation.entities,
        observation.entity_mask,
        observation.central,
        observation.target_mask,
        observation.fire_mask,
        step.action,
        agent_ids,
    )
    changed_action = PolicyAction(
        **{name: getattr(step.action, name).clone() for name in step.action.__dataclass_fields__}
    )
    changed_action.radar_report[:, radar_slots] = 1000
    changed_coordinates = policy.evaluate_actions(
        observation.agents,
        observation.entities,
        observation.entity_mask,
        observation.central,
        observation.target_mask,
        observation.fire_mask,
        changed_action,
        agent_ids,
    )
    assert torch.equal(
        changed_coordinates.log_prob[:, radar_slots],
        without_coordinates.log_prob[:, radar_slots],
    )


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
    assert runner.environment.config.observation_mode == "belief"
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
