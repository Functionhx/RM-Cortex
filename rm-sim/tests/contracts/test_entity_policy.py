from __future__ import annotations

import pytest
import torch

from rm_train import SharedMAPPOPolicy
from rm_world.observations import ENTITY_DIM


def _policy_inputs() -> tuple[torch.Tensor, ...]:
    observations = torch.zeros(1, 16, 34)
    entities = torch.zeros(1, 16, 16, ENTITY_DIM)
    entity_mask = torch.zeros(1, 16, 16, dtype=torch.bool)
    central_state = torch.zeros(1, 117)
    target_mask = torch.zeros(1, 16, 9, dtype=torch.bool)
    target_mask[..., -1] = True
    fire_mask = torch.zeros(1, 16, dtype=torch.bool)
    return (
        observations,
        entities,
        entity_mask,
        central_state,
        target_mask,
        fire_mask,
    )


def test_unobserved_entity_belief_changes_continuous_actor_output() -> None:
    torch.manual_seed(17)
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    (
        observations,
        entities,
        entity_mask,
        central_state,
        target_mask,
        fire_mask,
    ) = _policy_inputs()

    baseline = policy.act(
        observations,
        entities,
        entity_mask,
        central_state,
        target_mask,
        fire_mask,
        deterministic=True,
    )
    changed_entities = entities.clone()
    changed_entities[0, 0, 8, :4] = torch.tensor((0.8, -0.6, 0.2, 1.0))
    changed_entities[0, 0, 8, 8] = 0.75
    changed = policy.act(
        observations,
        changed_entities,
        entity_mask,
        central_state,
        target_mask,
        fire_mask,
        deterministic=True,
    )

    assert not torch.allclose(
        baseline.action.motion[0, 0],
        changed.action.motion[0, 0],
    )


def test_visibility_is_an_entity_feature_without_masking_attention() -> None:
    torch.manual_seed(29)
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    observations, entities, entity_mask, *_ = _policy_inputs()
    entities[0, 0, 8, 0] = 0.5

    hidden_context, hidden_weights = policy.entity_attention(
        observations,
        entities,
        entity_mask,
    )
    visible_mask = entity_mask.clone()
    visible_mask[0, 0, 8] = True
    visible_context, visible_weights = policy.entity_attention(
        observations,
        entities,
        visible_mask,
    )

    assert torch.isfinite(hidden_context).all()
    assert torch.isfinite(hidden_weights).all()
    assert torch.allclose(hidden_weights.sum(dim=-1), torch.ones(1, 16))
    assert not torch.allclose(hidden_context[0, 0], visible_context[0, 0])
    assert not torch.allclose(hidden_weights[0, 0], visible_weights[0, 0])


@pytest.mark.parametrize(
    ("entities", "entity_mask", "message"),
    (
        (
            torch.zeros(1, 16, 15, ENTITY_DIM),
            torch.zeros(1, 16, 15, dtype=torch.bool),
            "16 unit slots",
        ),
        (
            torch.zeros(1, 16, 16, ENTITY_DIM - 1),
            torch.zeros(1, 16, 16, dtype=torch.bool),
            "feature dimension",
        ),
        (
            torch.zeros(1, 16, 16, ENTITY_DIM),
            torch.zeros(1, 16, 16),
            "bool dtype",
        ),
    ),
)
def test_entity_attention_rejects_invalid_shapes(
    entities: torch.Tensor,
    entity_mask: torch.Tensor,
    message: str,
) -> None:
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    observations = torch.zeros(1, 16, 34)

    with pytest.raises(ValueError, match=message):
        policy.entity_attention(observations, entities, entity_mask)
