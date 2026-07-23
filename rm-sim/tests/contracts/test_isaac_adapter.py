from __future__ import annotations

import torch

from rm_isaac import IsaacGeometryFrame, IsaacRuleAdapter
from rm_referee import GameState, RuleInputs


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
