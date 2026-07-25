"""Training and evaluation orchestration for RMUC tactical behavior cloning.

The referee export is sampled at one hertz. Its component-wise capped future
displacement label is a five-second tactical subgoal, not the 0.2-second motion
action consumed by the online MAPPO policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Protocol, cast

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset

from rm_train.data import (
    AGENT_FEATURE_NAMES,
    RMUCSQLiteAdapter,
    RMUCTacticalDataset,
    TacticalPolicyCollator,
    TacticalWindowConfig,
    TeamTimeSplitManifest,
)
from rm_train.imitation import (
    IMITATION_CHECKPOINT_SCHEMA_VERSION,
    IMITATION_FEATURE_SCHEMA_VERSION,
    TacticalBehaviorCloner,
    TacticalImitationLoss,
    TacticalPredictions,
    export_shared_actor_backbone,
    tactical_imitation_loss,
)

_VELOCITY_START = AGENT_FEATURE_NAMES.index("velocity_x")


class PolicyBatch(Protocol):
    """Tensor interface produced by the RMUC policy-feature collator."""

    @property
    def observations(self) -> Tensor: ...

    @property
    def entities(self) -> Tensor: ...

    @property
    def entity_mask(self) -> Tensor: ...

    @property
    def agent_ids(self) -> Tensor: ...

    @property
    def goal_xy(self) -> Tensor: ...

    @property
    def goal_valid(self) -> Tensor: ...

    @property
    def fire(self) -> Tensor: ...

    @property
    def fire_valid(self) -> Tensor: ...


@dataclass(frozen=True)
class ImitationConfig:
    """Reproducible settings for the one-hertz tactical BC pilot."""

    database_path: str = ""
    seed: int = 7
    device: str = "auto"
    epochs: int = 5
    batch_size: int = 64
    max_train_windows: int = 8_192  # Zero selects the complete strict split.
    max_validation_windows: int = 2_048  # Zero selects the complete strict split.
    num_workers: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    goal_scale_m: float = 10.0
    history_steps: int = 10
    future_horizon_s: float = 5.0
    stride_s: float = 1.0
    hidden_dim: int = 128
    role_embedding_dim: int = 16
    goal_coefficient: float = 1.0
    fire_coefficient: float = 0.0
    # Rounded negative/positive ratio measured on the strict southern-region
    # training pilot (positive rate ~= 0.304); fixed for comparable validation.
    fire_pos_weight: float = 2.25
    max_grad_norm: float = 1.0
    movement_threshold_m: float = 0.25
    local_sensor_range_m: float = 12.0
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    output_dir: str = "runs/rmuc_bc"

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def validate(self, *, require_database: bool = False) -> None:
        positive_integers = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "history_steps": self.history_steps,
            "hidden_dim": self.hidden_dim,
            "role_embedding_dim": self.role_embedding_dim,
        }
        for name, integer_value in positive_integers.items():
            if integer_value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_train_windows < 0 or self.max_validation_windows < 0:
            raise ValueError("maximum window counts cannot be negative")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        positive_floats = {
            "learning_rate": self.learning_rate,
            "goal_scale_m": self.goal_scale_m,
            "future_horizon_s": self.future_horizon_s,
            "stride_s": self.stride_s,
            "max_grad_norm": self.max_grad_norm,
            "local_sensor_range_m": self.local_sensor_range_m,
        }
        for name, positive_value in positive_floats.items():
            if not math.isfinite(positive_value) or positive_value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 1.0 <= self.fire_pos_weight <= 4.0:
            raise ValueError("fire_pos_weight must be in [1, 4]")
        nonnegative_floats = {
            "weight_decay": self.weight_decay,
            "goal_coefficient": self.goal_coefficient,
            "fire_coefficient": self.fire_coefficient,
            "movement_threshold_m": self.movement_threshold_m,
        }
        for name, nonnegative_value in nonnegative_floats.items():
            if not math.isfinite(nonnegative_value) or nonnegative_value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must sum to less than one")
        if require_database and not self.database_path:
            raise ValueError("database_path is required")

    def to_dict(self, *, include_database: bool = False) -> dict[str, Any]:
        """Serialize config without machine-local paths by default."""

        values = asdict(self)
        if not include_database:
            values.pop("database_path")
            values.pop("output_dir")
        return values

    @classmethod
    def from_json(cls, path: str | Path) -> ImitationConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("imitation config must be a JSON object")
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class _DeviceBatch:
    observations: Tensor
    entities: Tensor
    entity_mask: Tensor
    agent_ids: Tensor
    goal_xy: Tensor
    goal_valid: Tensor
    fire: Tensor
    fire_valid: Tensor


@dataclass
class _MetricAccumulator:
    goal_scale_m: float
    future_horizon_s: float
    movement_threshold_m: float
    goal_coefficient: float
    fire_coefficient: float
    batch_count: int = 0
    goal_loss_sum: float = 0.0
    goal_loss_weight: float = 0.0
    fire_loss_sum: float = 0.0
    fire_loss_weight: float = 0.0
    goal_error_sum_m: float = 0.0
    goal_error_square_sum_m2: float = 0.0
    zero_error_sum_m: float = 0.0
    zero_error_square_sum_m2: float = 0.0
    velocity_error_sum_m: float = 0.0
    velocity_error_square_sum_m2: float = 0.0
    goal_count: int = 0
    direction_cosine_sum: float = 0.0
    moving_count: int = 0
    fire_count: int = 0
    fire_positive_count: int = 0
    fire_true_positive: int = 0
    fire_false_positive: int = 0
    fire_false_negative: int = 0
    fire_correct: int = 0

    def update(
        self,
        predictions: TacticalPredictions,
        batch: _DeviceBatch,
        loss: TacticalImitationLoss,
    ) -> None:
        self.batch_count += 1
        goal_loss_weight = float(loss.goal_weight_sum.detach().item())
        fire_loss_weight = float(loss.fire_weight_sum.detach().item())
        self.goal_loss_sum += float(loss.goal.detach().item()) * goal_loss_weight
        self.goal_loss_weight += goal_loss_weight
        self.fire_loss_sum += float(loss.fire.detach().item()) * fire_loss_weight
        self.fire_loss_weight += fire_loss_weight

        goal_valid = batch.goal_valid.to(torch.bool) & torch.isfinite(batch.goal_xy).all(dim=-1)
        if torch.any(goal_valid):
            predicted_goal_m = predictions.goal_xy.detach()[goal_valid] * self.goal_scale_m
            target_goal_m = batch.goal_xy[goal_valid] * self.goal_scale_m
            errors_m = torch.linalg.vector_norm(predicted_goal_m - target_goal_m, dim=-1)
            zero_errors_m = torch.linalg.vector_norm(target_goal_m, dim=-1)
            velocity_goal_m = (
                batch.observations[goal_valid, _VELOCITY_START : _VELOCITY_START + 2]
                * 3.0
                * self.future_horizon_s
            ).clamp(min=-self.goal_scale_m, max=self.goal_scale_m)
            velocity_errors_m = torch.linalg.vector_norm(
                velocity_goal_m - target_goal_m,
                dim=-1,
            )
            self.goal_error_sum_m += float(errors_m.sum().item())
            self.goal_error_square_sum_m2 += float(errors_m.square().sum().item())
            self.zero_error_sum_m += float(zero_errors_m.sum().item())
            self.zero_error_square_sum_m2 += float(zero_errors_m.square().sum().item())
            self.velocity_error_sum_m += float(velocity_errors_m.sum().item())
            self.velocity_error_square_sum_m2 += float(velocity_errors_m.square().sum().item())
            self.goal_count += int(errors_m.numel())

            moving = zero_errors_m > self.movement_threshold_m
            if torch.any(moving):
                moved_prediction = predicted_goal_m[moving]
                moved_target = target_goal_m[moving]
                dot = (moved_prediction * moved_target).sum(dim=-1)
                denominator = (
                    torch.linalg.vector_norm(moved_prediction, dim=-1)
                    * torch.linalg.vector_norm(moved_target, dim=-1)
                ).clamp_min(1.0e-8)
                cosine = torch.where(
                    torch.linalg.vector_norm(moved_prediction, dim=-1) > 1.0e-8,
                    dot / denominator,
                    torch.zeros_like(dot),
                )
                self.direction_cosine_sum += float(cosine.sum().item())
                self.moving_count += int(cosine.numel())

        fire_valid = batch.fire_valid.to(torch.bool) & torch.isfinite(batch.fire)
        if torch.any(fire_valid):
            target_fire = batch.fire[fire_valid] >= 0.5
            predicted_fire = torch.sigmoid(predictions.fire_logits.detach()[fire_valid]) >= 0.5
            self.fire_count += int(target_fire.numel())
            self.fire_positive_count += int(target_fire.sum().item())
            self.fire_true_positive += int((predicted_fire & target_fire).sum().item())
            self.fire_false_positive += int((predicted_fire & ~target_fire).sum().item())
            self.fire_false_negative += int((~predicted_fire & target_fire).sum().item())
            self.fire_correct += int((predicted_fire == target_fire).sum().item())

    def as_dict(self) -> dict[str, float | int]:
        goal_loss = _safe_ratio(self.goal_loss_sum, self.goal_loss_weight)
        fire_loss = _safe_ratio(self.fire_loss_sum, self.fire_loss_weight)
        goal_mae_m = _safe_ratio(self.goal_error_sum_m, self.goal_count)
        goal_rmse_m = math.sqrt(_safe_ratio(self.goal_error_square_sum_m2, self.goal_count))
        zero_mae_m = _safe_ratio(self.zero_error_sum_m, self.goal_count)
        zero_rmse_m = math.sqrt(_safe_ratio(self.zero_error_square_sum_m2, self.goal_count))
        velocity_mae_m = _safe_ratio(self.velocity_error_sum_m, self.goal_count)
        velocity_rmse_m = math.sqrt(_safe_ratio(self.velocity_error_square_sum_m2, self.goal_count))
        precision = _safe_ratio(
            self.fire_true_positive,
            self.fire_true_positive + self.fire_false_positive,
        )
        recall = _safe_ratio(
            self.fire_true_positive,
            self.fire_true_positive + self.fire_false_negative,
        )
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        positive_rate = _safe_ratio(self.fire_positive_count, self.fire_count)
        return {
            "loss_total": self.goal_coefficient * goal_loss + self.fire_coefficient * fire_loss,
            "loss_goal": goal_loss,
            "loss_fire": fire_loss,
            "batch_count": self.batch_count,
            "goal_valid_count": self.goal_count,
            "goal_mae_m": goal_mae_m,
            "goal_rmse_m": goal_rmse_m,
            "goal_zero_baseline_mae_m": zero_mae_m,
            "goal_zero_baseline_rmse_m": zero_rmse_m,
            "goal_velocity_baseline_mae_m": velocity_mae_m,
            "goal_velocity_baseline_rmse_m": velocity_rmse_m,
            "goal_moving_count": self.moving_count,
            "goal_direction_cosine": _safe_ratio(
                self.direction_cosine_sum,
                self.moving_count,
            ),
            "fire_valid_count": self.fire_count,
            "fire_positive_rate": positive_rate,
            "fire_precision": precision,
            "fire_recall": recall,
            "fire_f1": f1,
            "fire_accuracy": _safe_ratio(self.fire_correct, self.fire_count),
            "fire_no_fire_baseline_accuracy": _safe_ratio(
                self.fire_count - self.fire_positive_count,
                self.fire_count,
            ),
        }


@dataclass
class _DataBundle:
    train_loader: Iterable[PolicyBatch]
    validation_loader: Iterable[PolicyBatch]
    manifest: TeamTimeSplitManifest
    train_dataset: RMUCTacticalDataset
    validation_dataset: RMUCTacticalDataset
    train_available_windows: int
    validation_available_windows: int
    train_selected_windows: int
    validation_selected_windows: int

    def close(self) -> None:
        self.train_dataset.close()
        self.validation_dataset.close()


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _move_batch(batch: PolicyBatch, device: torch.device) -> _DeviceBatch:
    return _DeviceBatch(
        observations=batch.observations.to(device, non_blocking=True),
        entities=batch.entities.to(device, non_blocking=True),
        entity_mask=batch.entity_mask.to(device, non_blocking=True),
        agent_ids=batch.agent_ids.to(device, non_blocking=True),
        goal_xy=batch.goal_xy.to(device, non_blocking=True),
        goal_valid=batch.goal_valid.to(device, non_blocking=True),
        fire=batch.fire.to(device, non_blocking=True),
        fire_valid=batch.fire_valid.to(device, non_blocking=True),
    )


def _deterministic_subset(
    dataset: Dataset[Any],
    maximum: int,
    *,
    seed: int,
) -> Dataset[Any]:
    dataset_length = _dataset_length(dataset)
    if maximum == 0 or maximum >= dataset_length:
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_length, generator=generator)[:maximum]
    # Sorted random indices retain deterministic sampling while reducing SQLite
    # seek churn during validation and before training shuffles the subset.
    return Subset(dataset, indices.sort().values.tolist())


def _dataset_length(dataset: Dataset[Any]) -> int:
    return len(cast(Sized, dataset))


def _split_summary(
    manifest: TeamTimeSplitManifest,
    *,
    train_available_windows: int,
    validation_available_windows: int,
    train_selected_windows: int,
    validation_selected_windows: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        components = tuple(
            component for component in manifest.components if component.split == split
        )
        result[split] = {
            "component_hashes": sorted(component.component_id for component in components),
            "component_count": len(components),
            "team_count": sum(len(component.teams) for component in components),
            "game_count": sum(len(component.game_ids) for component in components),
        }
    result["windows"] = {
        "train_available": train_available_windows,
        "validation_available": validation_available_windows,
        "train_selected": train_selected_windows,
        "validation_selected": validation_selected_windows,
    }
    return result


def build_rmuc_data_loaders(config: ImitationConfig) -> _DataBundle:
    """Create strict manifest-backed train and validation loaders."""

    config.validate(require_database=True)
    adapter = RMUCSQLiteAdapter(config.database_path)
    manifest = adapter.build_team_time_split(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    window = TacticalWindowConfig(
        history_steps=config.history_steps,
        future_horizon_s=config.future_horizon_s,
        stride_s=config.stride_s,
    )
    train_dataset = RMUCTacticalDataset(adapter, manifest, "train", window)
    validation_dataset = RMUCTacticalDataset(adapter, manifest, "validation", window)
    train_available = len(train_dataset)
    validation_available = len(validation_dataset)
    train_source = _deterministic_subset(
        train_dataset,
        config.max_train_windows,
        seed=config.seed + 101,
    )
    validation_source = _deterministic_subset(
        validation_dataset,
        config.max_validation_windows,
        seed=config.seed + 202,
    )
    train_selected = _dataset_length(train_source)
    validation_selected = _dataset_length(validation_source)
    if train_selected == 0 or validation_selected == 0:
        train_dataset.close()
        validation_dataset.close()
        raise ValueError("strict train and validation splits must both contain windows")

    collator = TacticalPolicyCollator(
        goal_scale_m=config.goal_scale_m,
        local_sensor_range_m=config.local_sensor_range_m,
    )
    pin_memory = config.resolved_device.startswith("cuda")
    train_generator = torch.Generator().manual_seed(config.seed + 303)
    train_loader = DataLoader(
        train_source,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_source,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )
    return _DataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        manifest=manifest,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_available_windows=train_available,
        validation_available_windows=validation_available,
        train_selected_windows=train_selected,
        validation_selected_windows=validation_selected,
    )


class ImitationTrainingRunner:
    """Optimize and validate a tactical behavior cloner."""

    CHECKPOINT_SCHEMA_VERSION = IMITATION_CHECKPOINT_SCHEMA_VERSION

    def __init__(
        self,
        config: ImitationConfig,
        *,
        model: TacticalBehaviorCloner | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.resolved_device)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.model = model or TacticalBehaviorCloner(
            hidden_dim=config.hidden_dim,
            role_embedding_dim=config.role_embedding_dim,
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.epoch_index = 0

    def _new_accumulator(self) -> _MetricAccumulator:
        return _MetricAccumulator(
            goal_scale_m=self.config.goal_scale_m,
            future_horizon_s=self.config.future_horizon_s,
            movement_threshold_m=self.config.movement_threshold_m,
            goal_coefficient=self.config.goal_coefficient,
            fire_coefficient=self.config.fire_coefficient,
        )

    def _loss(
        self,
        predictions: TacticalPredictions,
        batch: _DeviceBatch,
    ) -> TacticalImitationLoss:
        return tactical_imitation_loss(
            predictions,
            goal_xy=batch.goal_xy,
            fire=batch.fire,
            goal_valid=batch.goal_valid,
            fire_valid=batch.fire_valid,
            goal_coefficient=self.config.goal_coefficient,
            fire_coefficient=self.config.fire_coefficient,
            fire_pos_weight=self.config.fire_pos_weight,
        )

    def train_epoch(self, loader: Iterable[PolicyBatch]) -> dict[str, float | int]:
        self.model.train()
        accumulator = self._new_accumulator()
        for source_batch in loader:
            batch = _move_batch(source_batch, self.device)
            predictions = self.model(
                batch.observations,
                batch.entities,
                batch.entity_mask,
                batch.agent_ids,
            )
            loss = self._loss(predictions, batch)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite tactical imitation loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            accumulator.update(predictions, batch, loss)
        if accumulator.batch_count == 0:
            raise ValueError("training loader yielded no batches")
        self.epoch_index += 1
        return accumulator.as_dict()

    @torch.no_grad()
    def validate(self, loader: Iterable[PolicyBatch]) -> dict[str, float | int]:
        self.model.eval()
        accumulator = self._new_accumulator()
        for source_batch in loader:
            batch = _move_batch(source_batch, self.device)
            predictions = self.model(
                batch.observations,
                batch.entities,
                batch.entity_mask,
                batch.agent_ids,
            )
            loss = self._loss(predictions, batch)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite tactical validation loss")
            accumulator.update(predictions, batch, loss)
        if accumulator.batch_count == 0:
            raise ValueError("validation loader yielded no batches")
        return accumulator.as_dict()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        split: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> Path:
        """Save only reusable weights and provenance-safe training metadata."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model_state = {
            key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
        }
        actor_backbone = {
            key: value.detach().cpu().clone()
            for key, value in export_shared_actor_backbone(self.model.backbone).items()
        }
        torch.save(
            {
                "checkpoint_schema_version": self.CHECKPOINT_SCHEMA_VERSION,
                "model_state": model_state,
                "actor_backbone": actor_backbone,
                "architecture": {
                    "model": "TacticalBehaviorCloner",
                    "policy_kwargs": self.model.backbone.architecture_kwargs(),
                    "goal_head_dim": 2,
                    "goal_label": "componentwise_capped_future_displacement",
                    "goal_cap_m": self.config.goal_scale_m,
                    "fire_head_dim": 1,
                    "target_head": self.model.target_head is not None,
                },
                "feature_schema": {
                    "version": IMITATION_FEATURE_SCHEMA_VERSION,
                    "observation_dim": self.model.backbone.observation_dim,
                    "entity_dim": self.model.backbone.entity_dim,
                    "coordinate_frame": "global_centered",
                    "belief_history": "truncated_causal_window",
                },
                # Never persist the official database's absolute local path.
                "config": self.config.to_dict(include_database=False),
                "split": dict(split),
                "metrics": dict(metrics),
            },
            destination,
        )
        return destination

    def train(self) -> tuple[list[dict[str, Any]], Path]:
        """Build strict RMUC splits, train, validate, and save ``latest.pt``."""

        bundle = build_rmuc_data_loaders(self.config)
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        split = _split_summary(
            bundle.manifest,
            train_available_windows=bundle.train_available_windows,
            validation_available_windows=bundle.validation_available_windows,
            train_selected_windows=bundle.train_selected_windows,
            validation_selected_windows=bundle.validation_selected_windows,
        )
        progress: list[dict[str, Any]] = []
        best_score = math.inf
        best_state: dict[str, Tensor] | None = None
        best_metrics: dict[str, Any] | None = None
        metrics_path = output / "metrics.jsonl"
        try:
            with metrics_path.open("w", encoding="utf-8") as stream:
                for _ in range(self.config.epochs):
                    train_metrics = self.train_epoch(bundle.train_loader)
                    validation_metrics = self.validate(bundle.validation_loader)
                    current: dict[str, Any] = {
                        "epoch": self.epoch_index,
                        "intent_horizon_s": self.config.future_horizon_s,
                        "goal_semantics": "componentwise-capped tactical subgoal",
                        **{f"train_{name}": value for name, value in train_metrics.items()},
                        **{
                            f"validation_{name}": value
                            for name, value in validation_metrics.items()
                        },
                    }
                    progress.append(current)
                    stream.write(json.dumps(current, sort_keys=True) + "\n")
                    stream.flush()
                    validation_score = float(current["validation_loss_total"])
                    if validation_score < best_score:
                        best_score = validation_score
                        best_metrics = current
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
            self.save_checkpoint(
                output / "latest.pt",
                split=split,
                metrics=progress[-1],
            )
            if best_state is None or best_metrics is None:
                raise RuntimeError("training did not produce a best checkpoint")
            self.model.load_state_dict(best_state)
            checkpoint = self.save_checkpoint(
                output / "best.pt",
                split=split,
                metrics=best_metrics,
            )
        finally:
            bundle.close()
        return progress, checkpoint


__all__ = [
    "ImitationConfig",
    "ImitationTrainingRunner",
    "PolicyBatch",
    "build_rmuc_data_loaders",
]
