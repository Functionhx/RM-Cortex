from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from rm_train.imitation import TacticalBehaviorCloner
from rm_train.imitation_runner import ImitationConfig, ImitationTrainingRunner
from rm_world.observations import ENTITY_DIM


@dataclass(frozen=True)
class _SyntheticPolicyBatch:
    observations: torch.Tensor
    entities: torch.Tensor
    entity_mask: torch.Tensor
    agent_ids: torch.Tensor
    goal_xy: torch.Tensor
    goal_valid: torch.Tensor
    fire: torch.Tensor
    fire_valid: torch.Tensor


def _batch(size: int = 8) -> _SyntheticPolicyBatch:
    generator = torch.Generator().manual_seed(13)
    return _SyntheticPolicyBatch(
        observations=torch.randn(size, 34, generator=generator),
        entities=torch.randn(size, 16, ENTITY_DIM, generator=generator),
        entity_mask=torch.rand(size, 16, generator=generator) > 0.25,
        agent_ids=torch.arange(size).remainder(16),
        goal_xy=torch.rand(size, 2, generator=generator) * 1.5 - 0.75,
        goal_valid=torch.ones(size, dtype=torch.bool),
        fire=(torch.arange(size).remainder(3) == 0).to(torch.float32),
        fire_valid=torch.ones(size, dtype=torch.bool),
    )


def _slice(
    batch: _SyntheticPolicyBatch,
    start: int,
    end: int,
) -> _SyntheticPolicyBatch:
    return _SyntheticPolicyBatch(
        observations=batch.observations[start:end],
        entities=batch.entities[start:end],
        entity_mask=batch.entity_mask[start:end],
        agent_ids=batch.agent_ids[start:end],
        goal_xy=batch.goal_xy[start:end],
        goal_valid=batch.goal_valid[start:end],
        fire=batch.fire[start:end],
        fire_valid=batch.fire_valid[start:end],
    )


def _config(
    tmp_path: Path,
    *,
    goal_scale_m: float = 10.0,
    movement_threshold_m: float = 0.25,
) -> ImitationConfig:
    return ImitationConfig(
        database_path=str((tmp_path / "official-secret.sqlite").resolve()),
        device="cpu",
        epochs=1,
        batch_size=2,
        hidden_dim=16,
        role_embedding_dim=8,
        goal_scale_m=goal_scale_m,
        movement_threshold_m=movement_threshold_m,
        output_dir=str(tmp_path / "run"),
    )


def test_one_behavior_cloning_update_is_finite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = TacticalBehaviorCloner(
        hidden_dim=config.hidden_dim,
        role_embedding_dim=config.role_embedding_dim,
    )
    runner = ImitationTrainingRunner(config, model=model)
    before = {
        name: parameter.detach().clone() for name, parameter in runner.model.named_parameters()
    }

    metrics = runner.train_epoch([_batch()])

    assert runner.epoch_index == 1
    assert metrics["batch_count"] == 1
    assert metrics["goal_valid_count"] == 8
    assert metrics["fire_valid_count"] == 8
    assert all(
        math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float))
    )
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in runner.model.named_parameters()
    )


def test_validation_reports_masked_goal_and_fire_baselines(tmp_path: Path) -> None:
    config = _config(tmp_path, goal_scale_m=10.0, movement_threshold_m=0.25)
    model = TacticalBehaviorCloner(
        hidden_dim=config.hidden_dim,
        role_embedding_dim=config.role_embedding_dim,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fire_head.bias.fill_(-10.0)
    runner = ImitationTrainingRunner(config, model=model)
    source = _batch(size=4)
    batch = _SyntheticPolicyBatch(
        observations=torch.zeros_like(source.observations),
        entities=source.entities,
        entity_mask=source.entity_mask,
        agent_ids=source.agent_ids,
        goal_xy=torch.tensor(
            (
                (0.3, 0.4),
                (0.6, 0.8),
                (float("nan"), float("nan")),
                (0.0, 0.0),
            )
        ),
        goal_valid=torch.tensor((True, True, False, True)),
        fire=torch.tensor((1.0, 0.0, float("nan"), 1.0)),
        fire_valid=torch.tensor((True, True, False, False)),
    )

    metrics = runner.validate([batch])

    assert metrics["goal_valid_count"] == 3
    assert metrics["goal_moving_count"] == 2
    assert math.isclose(float(metrics["goal_mae_m"]), 5.0)
    assert math.isclose(
        float(metrics["goal_rmse_m"]),
        math.sqrt(125.0 / 3.0),
        rel_tol=1.0e-6,
    )
    assert metrics["goal_mae_m"] == metrics["goal_zero_baseline_mae_m"]
    assert metrics["goal_rmse_m"] == metrics["goal_zero_baseline_rmse_m"]
    assert metrics["goal_mae_m"] == metrics["goal_velocity_baseline_mae_m"]
    assert metrics["goal_rmse_m"] == metrics["goal_velocity_baseline_rmse_m"]
    assert metrics["goal_direction_cosine"] == 0.0
    assert metrics["fire_valid_count"] == 2
    assert metrics["fire_positive_rate"] == 0.5
    assert metrics["fire_precision"] == 0.0
    assert metrics["fire_recall"] == 0.0
    assert metrics["fire_f1"] == 0.0
    assert metrics["fire_no_fire_baseline_accuracy"] == 0.5
    assert all(
        math.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float))
    )


def test_validation_loss_is_independent_of_batch_partition(tmp_path: Path) -> None:
    runner = ImitationTrainingRunner(
        _config(tmp_path),
        model=TacticalBehaviorCloner(hidden_dim=16, role_embedding_dim=8),
    )
    batch = _batch(size=8)

    whole = runner.validate([batch])
    partitioned = runner.validate([_slice(batch, 0, 3), _slice(batch, 3, 8)])

    for name in ("loss_goal", "loss_fire", "loss_total"):
        assert math.isclose(
            float(whole[name]),
            float(partitioned[name]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )


def test_checkpoint_excludes_database_path_and_raw_data(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = ImitationTrainingRunner(
        config,
        model=TacticalBehaviorCloner(
            hidden_dim=config.hidden_dim,
            role_embedding_dim=config.role_embedding_dim,
        ),
    )
    destination = runner.save_checkpoint(
        tmp_path / "latest.pt",
        split={
            "train": {
                "component_hashes": ["0123456789abcdef"],
                "component_count": 1,
                "team_count": 12,
                "game_count": 40,
            }
        },
        metrics={"validation_goal_mae_m": 1.25},
    )

    payload = torch.load(destination, map_location="cpu", weights_only=False)

    assert set(payload) == {
        "checkpoint_schema_version",
        "model_state",
        "actor_backbone",
        "architecture",
        "feature_schema",
        "config",
        "split",
        "metrics",
    }
    assert "database_path" not in payload["config"]
    assert "output_dir" not in payload["config"]
    assert config.database_path not in repr(payload)
    assert config.output_dir not in repr(payload)
    assert payload["actor_backbone"]
    assert payload["feature_schema"]["coordinate_frame"] == "global_centered"
    assert not any("optimizer" in key for key in payload)
