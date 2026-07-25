from __future__ import annotations

from pathlib import Path

import torch

from rm_train.imitation import (
    IMITATION_CHECKPOINT_SCHEMA_VERSION,
    IMITATION_FEATURE_SCHEMA_VERSION,
    SHARED_ACTOR_COMPONENTS,
    TacticalBehaviorCloner,
    TacticalPredictions,
    export_shared_actor_backbone,
    load_imitation_actor_checkpoint,
    tactical_imitation_loss,
)
from rm_train.policy import SharedMAPPOPolicy
from rm_world.observations import ENTITY_DIM


def _actor_inputs(batch_size: int = 6) -> tuple[torch.Tensor, ...]:
    observations = torch.randn(batch_size, 34)
    entities = torch.randn(batch_size, 16, ENTITY_DIM)
    entity_mask = torch.rand(batch_size, 16) > 0.5
    agent_ids = torch.arange(batch_size).remainder(16)
    return observations, entities, entity_mask, agent_ids


def test_behavior_cloner_outputs_tactical_head_shapes() -> None:
    torch.manual_seed(3)
    backbone = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    model = TacticalBehaviorCloner(backbone, enable_target_head=True)
    observations, entities, entity_mask, agent_ids = _actor_inputs()

    features, roles, attention = backbone.actor_features(
        observations,
        entities,
        entity_mask,
        agent_ids,
    )
    predictions = model(observations, entities, entity_mask, agent_ids)

    assert features.shape == (6, 32)
    assert roles.shape == (6,)
    assert attention.shape == (6, 16)
    assert predictions.goal_xy.shape == (6, 2)
    assert predictions.fire_logits.shape == (6,)
    assert predictions.target_logits is not None
    assert predictions.target_logits.shape == (6, 9)
    assert torch.all(predictions.goal_xy.abs() <= 1.0)


def test_imitation_loss_masks_each_head_independently_and_handles_empty_masks() -> None:
    predictions = TacticalPredictions(
        goal_xy=torch.tensor(((1.0, 0.0), (20.0, -20.0)), requires_grad=True),
        fire_logits=torch.tensor((0.0, 100.0), requires_grad=True),
    )
    goal_xy = torch.tensor(((0.0, 0.0), (float("nan"), float("nan"))))
    fire = torch.tensor((1.0, float("nan")))
    loss = tactical_imitation_loss(
        predictions,
        goal_xy=goal_xy,
        fire=fire,
        goal_valid=torch.tensor((True, False)),
        fire_valid=torch.tensor((True, False)),
    )

    assert torch.allclose(loss.goal, torch.tensor(0.25))
    assert torch.allclose(loss.fire, torch.log(torch.tensor(2.0)))
    assert loss.fire_pos_weight.item() == 1.0
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert torch.isfinite(predictions.goal_xy.grad).all()
    assert torch.isfinite(predictions.fire_logits.grad).all()

    empty_predictions = TacticalPredictions(
        goal_xy=torch.zeros(2, 2, requires_grad=True),
        fire_logits=torch.zeros(2, requires_grad=True),
    )
    empty = tactical_imitation_loss(
        empty_predictions,
        goal_xy=torch.full((2, 2), float("nan")),
        fire=torch.full((2,), float("nan")),
        goal_valid=torch.zeros(2, dtype=torch.bool),
        fire_valid=torch.zeros(2, dtype=torch.bool),
        fire_pos_weight=100.0,
    )
    assert empty.total.item() == 0.0
    assert torch.isfinite(empty.total)
    assert empty.fire_pos_weight.item() == 4.0
    empty.total.backward()
    assert torch.isfinite(empty_predictions.goal_xy.grad).all()
    assert torch.isfinite(empty_predictions.fire_logits.grad).all()


def test_behavior_cloner_overfits_one_small_batch() -> None:
    torch.manual_seed(11)
    model = TacticalBehaviorCloner(hidden_dim=16, role_embedding_dim=8)
    observations = torch.zeros(8, 34)
    observations[:, 0] = torch.tensor((-1.0, 1.0)).repeat(4)
    entities = torch.zeros(8, 16, ENTITY_DIM)
    entity_mask = torch.zeros(8, 16, dtype=torch.bool)
    agent_ids = torch.full((8,), 2)
    goal_xy = torch.stack(
        (
            observations[:, 0] * 0.4,
            observations[:, 0] * -0.25,
        ),
        dim=-1,
    )
    fire = (observations[:, 0] > 0).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-2)

    with torch.no_grad():
        initial = tactical_imitation_loss(
            model(observations, entities, entity_mask, agent_ids),
            goal_xy=goal_xy,
            fire=fire,
        ).total.item()
    for _ in range(150):
        loss = tactical_imitation_loss(
            model(observations, entities, entity_mask, agent_ids),
            goal_xy=goal_xy,
            fire=fire,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()
    with torch.no_grad():
        final = tactical_imitation_loss(
            model(observations, entities, entity_mask, agent_ids),
            goal_xy=goal_xy,
            fire=fire,
        ).total.item()

    assert final < 0.02
    assert final < initial * 0.05


def test_backbone_transfer_preserves_all_mappo_specific_parameters() -> None:
    torch.manual_seed(19)
    source_policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    model = TacticalBehaviorCloner(
        SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8),
    )
    model.import_mappo_actor(source_policy)
    imported = export_shared_actor_backbone(model.backbone)
    source = export_shared_actor_backbone(source_policy)
    assert imported.keys() == source.keys()
    assert all(torch.equal(imported[key], source[key]) for key in source)

    with torch.no_grad():
        for component_name in SHARED_ACTOR_COMPONENTS:
            for parameter in getattr(model.backbone, component_name).parameters():
                parameter.add_(0.125)

    torch.manual_seed(23)
    destination = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    before = {key: value.detach().clone() for key, value in destination.state_dict().items()}
    model.export_mappo_actor(destination)
    after = destination.state_dict()
    transferred = export_shared_actor_backbone(model.backbone)

    role_key = "role_embedding.weight"
    assert torch.equal(after[role_key][:6], transferred[role_key][:6])
    assert torch.equal(after[role_key][6:], before[role_key][6:])
    assert all(
        torch.equal(after[key], value) for key, value in transferred.items() if key != role_key
    )
    untouched = before.keys() - transferred.keys()
    assert untouched
    assert all(torch.equal(after[key], before[key]) for key in untouched)


def test_cpu_checkpoint_load_preserves_building_rows_on_destination_device(
    tmp_path: Path,
) -> None:
    source = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    checkpoint = tmp_path / "imitation.pt"
    torch.save(
        {
            "checkpoint_schema_version": IMITATION_CHECKPOINT_SCHEMA_VERSION,
            "feature_schema": {
                "version": IMITATION_FEATURE_SCHEMA_VERSION,
                "observation_dim": 34,
                "entity_dim": ENTITY_DIM,
                "coordinate_frame": "global_centered",
            },
            "actor_backbone": export_shared_actor_backbone(source),
        },
        checkpoint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    destination = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8).to(device)
    building_before = destination.role_embedding.weight[6:].detach().clone()

    load_imitation_actor_checkpoint(destination, checkpoint)

    assert torch.equal(
        destination.role_embedding.weight[:6],
        source.role_embedding.weight[:6].to(device),
    )
    assert torch.equal(destination.role_embedding.weight[6:], building_before)
