from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from rm_isaac import (
    IsaacGeometryFrame,
    IsaacLabUnavailableError,
    IsaacRuleAdapter,
    RMCortexDirectMARLEnv,
    isaaclab_available,
)
from rm_referee import GameState, Referee, RuleInputs


def test_isaac_adapter_preserves_normalized_geometry_tensors() -> None:
    game = GameState.create(2)
    expected = RuleInputs.empty(game)
    expected.shots_fired[0, 2, 0] = 1
    expected.chassis_power_w[1, 1] = 42
    frame = IsaacGeometryFrame(
        shots_fired=expected.shots_fired.clone(),
        hit_source=expected.hits.source.clone(),
        hit_time_offset_s=expected.hits.time_offset_s.clone(),
        hit_critical=expected.hits.critical.clone(),
        chassis_power_w=expected.chassis_power_w.clone(),
    )

    actual = IsaacRuleAdapter().normalize(game, frame)

    assert torch.equal(actual.shots_fired, expected.shots_fired)
    assert torch.equal(actual.hits.source, expected.hits.source)
    assert torch.equal(actual.chassis_power_w, expected.chassis_power_w)


def test_isaac_adapter_preserves_complete_normalized_frame() -> None:
    game = GameState.create(2)
    expected = RuleInputs.empty(game)
    expected.in_supply_zone[0, 0] = True
    expected.remote_heal_request[1, 3] = True
    expected.radar_update[0, 0, 8] = True
    expected.radar_quality[0, 0, 8] = 1
    frame = IsaacGeometryFrame(
        shots_fired=expected.shots_fired,
        hit_source=expected.hits.source,
        hit_time_offset_s=expected.hits.time_offset_s,
        hit_critical=expected.hits.critical,
        chassis_power_w=expected.chassis_power_w,
    )

    actual = IsaacRuleAdapter().normalize(game, frame, base=expected)

    for field in fields(RuleInputs):
        if field.name == "hits":
            assert torch.equal(actual.hits.source, expected.hits.source)
            assert torch.equal(actual.hits.time_offset_s, expected.hits.time_offset_s)
            assert torch.equal(actual.hits.critical, expected.hits.critical)
        else:
            assert torch.equal(getattr(actual, field.name), getattr(expected, field.name))
    actual.in_supply_zone[0, 0] = False
    assert expected.in_supply_zone[0, 0]


def test_isaac_and_torch_normalized_inputs_produce_identical_rule_state() -> None:
    game = GameState.create(1)
    expected = RuleInputs.empty(game)
    expected.chassis_power_w[0, 0] = 62
    expected.supercap_input_w[0, 0] = 1
    expected.in_supply_zone[0, 2] = True
    frame = IsaacGeometryFrame(
        shots_fired=expected.shots_fired,
        hit_source=expected.hits.source,
        hit_time_offset_s=expected.hits.time_offset_s,
        hit_critical=expected.hits.critical,
        chassis_power_w=expected.chassis_power_w,
    )
    normalized = IsaacRuleAdapter().normalize(game, frame, base=expected)

    direct_state, _ = Referee().step(game, expected)
    isaac_state, _ = Referee().step(game, normalized)

    for field in fields(GameState):
        assert torch.equal(getattr(direct_state, field.name), getattr(isaac_state, field.name))


def test_isaac_entry_point_fails_loudly_without_optional_runtime() -> None:
    if isaaclab_available():
        pytest.skip("Isaac Lab is installed in this interpreter")
    with pytest.raises(IsaacLabUnavailableError, match="Isaac Lab"):
        RMCortexDirectMARLEnv()
