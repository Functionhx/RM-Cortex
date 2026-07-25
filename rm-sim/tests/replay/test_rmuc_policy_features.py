from __future__ import annotations

from dataclasses import replace

import torch

from rm_train.data import (
    AGENT_FEATURE_NAMES,
    RMUC_FEATURE_NAMES,
    TacticalPolicyCollator,
    TacticalSample,
    TacticalSampleKey,
    featurize_tactical_sample,
)
from rm_world.arena import (
    ArenaGeometry,
    BLUE_BASE_CENTER_XY,
    BLUE_OUTPOST_CENTER_XY,
    RED_BASE_CENTER_XY,
    RED_OUTPOST_CENTER_XY,
)
from rm_world.observations import ENTITY_DIM, ENTITY_FEATURE_NAMES


_RED_IDS = (1, 2, 3, 4, 6, 7, 10, 11)
_BLUE_IDS = tuple(robot_id + 100 for robot_id in _RED_IDS)
_RAW = {name: index for index, name in enumerate(RMUC_FEATURE_NAMES)}
_AGENT = {name: index for index, name in enumerate(AGENT_FEATURE_NAMES)}
_ENTITY = {name: index for index, name in enumerate(ENTITY_FEATURE_NAMES)}


def _absolute_slot(robot_id: int) -> int:
    team = int(robot_id >= 100)
    return team * 8 + _RED_IDS.index(robot_id - 100 * team)


def _sample(team: int = 0) -> TacticalSample:
    robot_ids = _RED_IDS + _BLUE_IDS if team == 0 else _BLUE_IDS + _RED_IDS
    source_for_slot = {
        _absolute_slot(robot_id): source for source, robot_id in enumerate(robot_ids)
    }
    history = 3
    features = torch.zeros(history, 16, len(RMUC_FEATURE_NAMES))
    feature_valid = torch.ones_like(features, dtype=torch.bool)
    position_valid = torch.ones(history, 16, dtype=torch.bool)

    for source, robot_id in enumerate(robot_ids):
        slot = _absolute_slot(robot_id)
        role = slot % 8
        is_blue = slot >= 8
        features[:, source, _RAW["hp"]] = 100.0
        features[:, source, _RAW["max_hp"]] = 200.0
        features[:, source, _RAW["x"]] = 25.0 + 0.1 * role if is_blue else 1.0 + 0.1 * role
        features[:, source, _RAW["y"]] = 13.0 if is_blue else 1.0
        features[:, source, _RAW["turret_yaw"]] = 123.0
        features[:, source, _RAW["heat_limit_17mm"]] = 100.0
        features[:, source, _RAW["heat_limit_42mm"]] = 100.0
        features[:, source, _RAW["team_coin_remaining"]] = 300.0
        if role >= 6:
            position_valid[:, source] = False

    red_hero = source_for_slot[0]
    features[:, red_hero, _RAW["x"]] = torch.tensor((1.0, 2.0, 4.0))
    features[:, red_hero, _RAW["y"]] = 1.0

    own_entity = torch.zeros(16, dtype=torch.bool)
    own_entity[:8] = True
    controllable = torch.zeros(16, dtype=torch.bool)
    controllable[:6] = True
    future_displacement = torch.zeros(16, 2)
    future_valid = torch.zeros(16, dtype=torch.bool)
    fired = torch.zeros(16, dtype=torch.bool)
    fire_valid = torch.zeros(16, dtype=torch.bool)
    future_displacement[:6] = torch.tensor((5.0, -2.0))
    future_valid[:6] = True
    fired[0] = True
    fire_valid[:6] = True
    return TacticalSample(
        key=TacticalSampleKey(game_id=7, team=team, anchor_s=3.0),
        times_s=torch.tensor((1.0, 2.0, 3.0)),
        robot_ids=torch.tensor(robot_ids),
        features=features,
        feature_valid=feature_valid,
        entity_present=torch.ones(history, 16, dtype=torch.bool),
        row_unique=torch.ones(history, 16, dtype=torch.bool),
        position_valid=position_valid,
        own_entity=own_entity,
        controllable=controllable,
        future_displacement_xy=future_displacement,
        future_displacement_valid=future_valid,
        fired=fired,
        fire_valid=fire_valid,
    )


def test_policy_example_shapes_global_slots_and_blue_coordinates_are_not_mirrored() -> None:
    red = featurize_tactical_sample(_sample(0))
    blue = featurize_tactical_sample(_sample(1))
    batch = TacticalPolicyCollator()((_sample(0), _sample(1)))

    assert red.observations.shape == (6, 34)
    assert red.entities.shape == (6, 16, ENTITY_DIM)
    assert red.entity_mask.shape == (6, 16)
    assert red.agent_ids.tolist() == list(range(6))
    assert blue.agent_ids.tolist() == list(range(8, 14))
    assert blue.observations[0, _AGENT["position_x"]] > 0.0
    assert red.observations[0, _AGENT["position_x"]] < 0.0
    assert batch.observations.shape == (12, 34)
    assert batch.entities.shape == (12, 16, ENTITY_DIM)
    assert batch.agent_ids.tolist() == [*range(6), *range(8, 14)]
    torch.testing.assert_close(batch.goal_xy[0], torch.tensor((0.5, -0.2)))
    assert batch.fire.dtype == torch.float32
    assert batch.entity_mask.dtype == torch.bool
    assert all(
        torch.isfinite(value).all()
        for value in (batch.observations, batch.entities, batch.goal_xy, batch.fire)
    )
    if torch.cuda.is_available():
        pinned = batch.pin_memory()
        assert pinned.observations.is_pinned()
        assert pinned.entities.is_pinned()


def test_official_building_positions_are_injected_in_absolute_slots() -> None:
    arena = ArenaGeometry()
    for team, expected in (
        (0, (RED_BASE_CENTER_XY, RED_OUTPOST_CENTER_XY)),
        (1, (BLUE_BASE_CENTER_XY, BLUE_OUTPOST_CENTER_XY)),
    ):
        example = featurize_tactical_sample(_sample(team))
        observer_x = example.observations[0, _AGENT["position_x"]] * 14.0
        observer_y = example.observations[0, _AGENT["position_y"]] * 7.5
        base_slot = team * 8 + 6
        outpost_slot = team * 8 + 7
        recovered_base = (
            example.entities[0, base_slot, _ENTITY["relative_x"]] * 28.0 + observer_x,
            example.entities[0, base_slot, _ENTITY["relative_y"]] * 15.0 + observer_y,
        )
        recovered_outpost = (
            example.entities[0, outpost_slot, _ENTITY["relative_x"]] * 28.0 + observer_x,
            example.entities[0, outpost_slot, _ENTITY["relative_y"]] * 15.0 + observer_y,
        )
        torch.testing.assert_close(torch.stack(recovered_base), torch.tensor(expected[0]))
        torch.testing.assert_close(torch.stack(recovered_outpost), torch.tensor(expected[1]))
        observer_z = example.observations[0, _AGENT["position_z"]] * 5.0
        expected_z = arena.terrain_height(torch.tensor(expected))
        recovered_z = (
            example.entities[0, (base_slot, outpost_slot), _ENTITY["relative_z"]] * 5.0 + observer_z
        )
        torch.testing.assert_close(recovered_z, expected_z)
        expected_yaw_cos = 1.0 if team == 0 else -1.0
        assert (
            example.entities[:, (base_slot, outpost_slot), _ENTITY["target_yaw_sin"]].eq(0.0).all()
        )
        assert example.entities[:, base_slot, _ENTITY["target_yaw_cos"]].eq(expected_yaw_cos).all()
        assert example.entities[:, outpost_slot, _ENTITY["target_yaw_cos"]].eq(0.0).all()
        assert example.entity_mask[:, (6, 7, 14, 15)].all()


def test_velocity_is_causal_and_turret_yaw_is_not_used_as_chassis_yaw() -> None:
    example = featurize_tactical_sample(_sample(0))
    # Only t=2 -> t=3 is used: (4 - 2) / 1 = 2 m/s.
    torch.testing.assert_close(
        example.observations[0, _AGENT["velocity_x"]],
        torch.tensor(2.0 / 3.0),
    )
    assert example.observations[0, _AGENT["yaw_sin"]].item() == 0.0
    assert example.observations[0, _AGENT["yaw_cos"]].item() == 1.0
    # Engineer is stationary; the 123 degree turret value must not affect yaw.
    assert example.observations[1, _AGENT["yaw_sin"]].item() == 0.0
    assert example.observations[1, _AGENT["yaw_cos"]].item() == 1.0

    sample = _sample(0)
    features = sample.features.clone()
    hero_source = int(torch.nonzero(sample.robot_ids == 1)[0].item())
    features[:, hero_source, _RAW["x"]] = torch.tensor((1.0, 1.01, 20.0))
    bounded = featurize_tactical_sample(replace(sample, features=features))
    velocity = bounded.observations[0, _AGENT["velocity_x"] : _AGENT["velocity_y"] + 1]
    torch.testing.assert_close(torch.linalg.vector_norm(velocity), torch.tensor(1.0))


def test_missing_shared_mobile_uses_prior_and_is_not_directly_masked() -> None:
    sample = _sample(0)
    missing = sample.position_valid.clone()
    engineer_source = int(torch.nonzero(sample.robot_ids == 2)[0].item())
    missing[:, engineer_source] = False
    example = featurize_tactical_sample(replace(sample, position_valid=missing))

    assert not example.entity_mask[:, 1].any()
    assert example.entities[:, 1, _ENTITY["source_prior"]].eq(1.0).all()
    assert example.entity_mask[:, (6, 7, 14, 15)].all()


def test_hidden_anchor_truth_cannot_change_inputs_but_future_truth_changes_label() -> None:
    sample = _sample(0)
    enemy_hero_source = int(torch.nonzero(sample.robot_ids == 101)[0].item())
    features = sample.features.clone()
    # Visible at t=1/2, then outside 12 m at the anchor.
    features[:, enemy_hero_source, _RAW["x"]] = torch.tensor((10.0, 11.0, 25.0))
    features[:, enemy_hero_source, _RAW["y"]] = 1.0
    tracked_sample = replace(sample, features=features)
    tracked = featurize_tactical_sample(tracked_sample)

    assert not tracked.entity_mask[:, 8].any()
    assert tracked.entities[:, 8, _ENTITY["source_prediction"]].eq(1.0).all()
    assert tracked.entities[:, 8, _ENTITY["position_observation_age_fraction"]].eq(0.1).all()
    # Last visible x=11 (centered -3), causal vx=1, age=1 -> predicted x=-2.
    observer_x = tracked.observations[0, _AGENT["position_x"]] * 14.0
    predicted_x = tracked.entities[0, 8, _ENTITY["relative_x"]] * 28.0 + observer_x
    torch.testing.assert_close(predicted_x, torch.tensor(-2.0))

    changed_hidden = features.clone()
    changed_hidden[-1, enemy_hero_source, _RAW["x"]] = 26.0
    hidden_result = featurize_tactical_sample(replace(tracked_sample, features=changed_hidden))
    torch.testing.assert_close(hidden_result.observations, tracked.observations)
    torch.testing.assert_close(hidden_result.entities, tracked.entities)
    assert torch.equal(hidden_result.entity_mask, tracked.entity_mask)

    changed_future = tracked_sample.future_displacement_xy.clone()
    changed_future[0] = torch.tensor((8.0, 1.0))
    future_result = featurize_tactical_sample(
        replace(tracked_sample, future_displacement_xy=changed_future)
    )
    torch.testing.assert_close(future_result.observations, tracked.observations)
    torch.testing.assert_close(future_result.entities, tracked.entities)
    assert torch.equal(future_result.entity_mask, tracked.entity_mask)
    assert not torch.equal(future_result.goal_xy, tracked.goal_xy)
