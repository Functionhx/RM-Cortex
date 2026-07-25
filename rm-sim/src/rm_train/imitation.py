"""Tactical behavior cloning on top of the MAPPO actor backbone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from rm_train.policy import SharedMAPPOPolicy
from rm_world.observations import ENTITY_DIM

SHARED_ACTOR_COMPONENTS = (
    "role_embedding",
    "team_embedding",
    "entity_encoder",
    "entity_query",
    "entity_key",
    "entity_value",
    "actor_body",
)
IMITATION_CHECKPOINT_SCHEMA_VERSION = 2
IMITATION_FEATURE_SCHEMA_VERSION = 1


@dataclass
class TacticalPredictions:
    """High-level labels available from one-hertz referee data."""

    goal_xy: Tensor
    fire_logits: Tensor
    target_logits: Tensor | None = None


@dataclass
class TacticalImitationLoss:
    """Individually normalized behavior-cloning losses."""

    total: Tensor
    goal: Tensor
    fire: Tensor
    fire_pos_weight: Tensor
    goal_weight_sum: Tensor
    fire_weight_sum: Tensor


def export_shared_actor_backbone(policy: SharedMAPPOPolicy) -> dict[str, Tensor]:
    """Copy the transferable actor parameters without action or value heads."""

    state: dict[str, Tensor] = {}
    for component_name in SHARED_ACTOR_COMPONENTS:
        component = getattr(policy, component_name)
        state.update(
            {
                f"{component_name}.{key}": value.detach().clone()
                for key, value in component.state_dict().items()
            }
        )
    return state


def load_shared_actor_backbone(
    policy: SharedMAPPOPolicy,
    state: Mapping[str, Tensor],
    *,
    preserve_role_rows: tuple[int, ...] = (),
) -> None:
    """Load only the shared actor components into ``policy``.

    Validation happens before any parameter is changed, so malformed or
    architecture-incompatible transfers cannot partially mutate a policy.
    """

    expected = export_shared_actor_backbone(policy)
    missing = expected.keys() - state.keys()
    unexpected = state.keys() - expected.keys()
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise ValueError("; ".join(details))

    for role in preserve_role_rows:
        if role < 0 or role >= policy.role_embedding.num_embeddings:
            raise ValueError(f"preserved role row {role} is out of range")
    if len(set(preserve_role_rows)) != len(preserve_role_rows):
        raise ValueError("preserved role rows must be unique")

    for key, expected_value in expected.items():
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"actor backbone value for {key!r} must be a tensor")
        if value.shape != expected_value.shape:
            raise ValueError(
                f"actor backbone shape mismatch for {key!r}: "
                f"expected {tuple(expected_value.shape)}, got {tuple(value.shape)}"
            )

    for component_name in SHARED_ACTOR_COMPONENTS:
        prefix = f"{component_name}."
        component_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if component_name == "role_embedding" and preserve_role_rows:
            role_weight = component_state["weight"].clone()
            preserved = expected["role_embedding.weight"][list(preserve_role_rows)].to(
                device=role_weight.device,
                dtype=role_weight.dtype,
            )
            role_weight[list(preserve_role_rows)] = preserved
            component_state["weight"] = role_weight
        getattr(policy, component_name).load_state_dict(component_state, strict=True)


def load_imitation_actor_checkpoint(
    policy: SharedMAPPOPolicy,
    path: str | Path,
) -> None:
    """Load a BC checkpoint's shared actor into MAPPO without replacing buildings."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("imitation checkpoint must contain a mapping")
    if payload.get("checkpoint_schema_version") != IMITATION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported imitation checkpoint schema")
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, Mapping):
        raise ValueError("imitation checkpoint is missing feature_schema")
    if feature_schema.get("version") != IMITATION_FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported imitation feature schema")
    expected_dimensions = (
        policy.observation_dim,
        policy.entity_dim,
    )
    dimensions = (
        feature_schema.get("observation_dim"),
        feature_schema.get("entity_dim"),
    )
    if dimensions != expected_dimensions:
        raise ValueError(
            f"imitation feature dimensions {dimensions} do not match policy {expected_dimensions}"
        )
    if feature_schema.get("coordinate_frame") != "global_centered":
        raise ValueError("imitation checkpoint uses an incompatible coordinate frame")
    actor_state = payload.get("actor_backbone")
    if not isinstance(actor_state, Mapping):
        raise ValueError("imitation checkpoint is missing actor_backbone")
    state: dict[str, Tensor] = {}
    for key, value in actor_state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError("actor_backbone must map string keys to tensors")
        state[key] = value
    load_shared_actor_backbone(
        policy,
        state,
        preserve_role_rows=(6, 7),
    )


class TacticalBehaviorCloner(nn.Module):
    """Predict coarse navigation intent and firing occurrence from actor features."""

    def __init__(
        self,
        backbone: SharedMAPPOPolicy | None = None,
        *,
        observation_dim: int = 34,
        entity_dim: int = ENTITY_DIM,
        state_dim: int = 117,
        hidden_dim: int = 128,
        role_embedding_dim: int = 16,
        enable_target_head: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone or SharedMAPPOPolicy(
            observation_dim=observation_dim,
            entity_dim=entity_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            role_embedding_dim=role_embedding_dim,
        )
        feature_dim = self.backbone.entity_query.out_features
        self.goal_xy_head = nn.Linear(feature_dim, 2)
        self.fire_head = nn.Linear(feature_dim, 1)
        self.target_head = nn.Linear(feature_dim, 9) if enable_target_head else None

    def forward(
        self,
        observations: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        agent_ids: Tensor,
    ) -> TacticalPredictions:
        features, _, _ = self.backbone.actor_features(
            observations,
            entities,
            entity_mask,
            agent_ids,
        )
        target_logits = self.target_head(features) if self.target_head is not None else None
        return TacticalPredictions(
            goal_xy=torch.tanh(self.goal_xy_head(features)),
            fire_logits=self.fire_head(features).squeeze(-1),
            target_logits=target_logits,
        )

    def import_mappo_actor(self, policy: SharedMAPPOPolicy) -> None:
        """Initialize the shared behavior-cloning backbone from MAPPO."""

        load_shared_actor_backbone(self.backbone, export_shared_actor_backbone(policy))

    def export_mappo_actor(self, policy: SharedMAPPOPolicy) -> None:
        """Transfer demonstrated actor features while preserving building roles."""

        load_shared_actor_backbone(
            policy,
            export_shared_actor_backbone(self.backbone),
            preserve_role_rows=(6, 7),
        )


def _valid_weights(
    valid: Tensor | None,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    if valid is None:
        return torch.ones_like(reference)
    if valid.shape != reference.shape:
        raise ValueError(f"{name} must match the prediction batch shape")
    weights = valid.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(weights).all():
        raise ValueError(f"{name} must contain finite weights")
    if torch.any(weights < 0):
        raise ValueError(f"{name} cannot contain negative weights")
    return weights


def tactical_imitation_loss(
    predictions: TacticalPredictions,
    *,
    goal_xy: Tensor,
    fire: Tensor,
    goal_valid: Tensor | None = None,
    fire_valid: Tensor | None = None,
    goal_coefficient: float = 1.0,
    fire_coefficient: float = 1.0,
    fire_pos_weight: float | Tensor | None = None,
) -> TacticalImitationLoss:
    """Compute masked goal and fire losses with independent denominators."""

    if predictions.goal_xy.shape != goal_xy.shape or goal_xy.shape[-1] != 2:
        raise ValueError("goal_xy must match the two-dimensional goal prediction")
    if predictions.fire_logits.shape != fire.shape:
        raise ValueError("fire must match the fire-logit batch shape")
    if goal_coefficient < 0 or fire_coefficient < 0:
        raise ValueError("loss coefficients cannot be negative")

    goal_weights = _valid_weights(
        goal_valid,
        predictions.goal_xy[..., 0],
        name="goal_valid",
    )
    fire_weights = _valid_weights(
        fire_valid,
        predictions.fire_logits,
        name="fire_valid",
    )
    goal_active = goal_weights > 0
    fire_active = fire_weights > 0
    safe_goal = torch.where(goal_active.unsqueeze(-1), goal_xy, predictions.goal_xy.detach())
    safe_fire = torch.where(fire_active, fire, torch.zeros_like(fire))
    if not torch.isfinite(safe_goal[goal_active]).all():
        raise ValueError("valid goal labels must be finite")
    if not torch.isfinite(safe_fire[fire_active]).all():
        raise ValueError("valid fire labels must be finite")
    if torch.any((safe_fire[fire_active] < 0) | (safe_fire[fire_active] > 1)):
        raise ValueError("valid fire labels must lie in [0, 1]")

    if fire_pos_weight is None:
        positive_mass = (safe_fire * fire_weights).sum()
        negative_mass = ((1.0 - safe_fire) * fire_weights).sum()
        positive_weight = (negative_mass / positive_mass.clamp_min(1.0)).clamp(1.0, 4.0)
    else:
        positive_weight = torch.as_tensor(
            fire_pos_weight,
            device=predictions.fire_logits.device,
            dtype=predictions.fire_logits.dtype,
        )
        if positive_weight.numel() != 1 or not torch.isfinite(positive_weight):
            raise ValueError("fire_pos_weight must be one finite scalar")
        positive_weight = positive_weight.reshape(()).clamp(1.0, 4.0)

    goal_elements = functional.smooth_l1_loss(
        predictions.goal_xy,
        safe_goal,
        reduction="none",
    ).mean(dim=-1)
    fire_elements = functional.binary_cross_entropy_with_logits(
        predictions.fire_logits,
        safe_fire.to(predictions.fire_logits.dtype),
        reduction="none",
        pos_weight=positive_weight,
    )
    goal_weight_sum = goal_weights.sum()
    fire_weight_sum = fire_weights.sum()
    goal_loss = (goal_elements * goal_weights).sum() / goal_weight_sum.clamp_min(1.0)
    fire_loss = (fire_elements * fire_weights).sum() / fire_weight_sum.clamp_min(1.0)
    total = goal_coefficient * goal_loss + fire_coefficient * fire_loss
    return TacticalImitationLoss(
        total=total,
        goal=goal_loss,
        fire=fire_loss,
        fire_pos_weight=positive_weight,
        goal_weight_sum=goal_weight_sum,
        fire_weight_sum=fire_weight_sum,
    )
