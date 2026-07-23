"""Isaac-facing normalization boundary with no duplicated game rules."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

from torch import Tensor

from rm_referee.inputs import HitCandidates, RuleInputs
from rm_referee.state import GameState


def isaaclab_available() -> bool:
    """Return whether the current Python environment exposes Isaac Lab."""

    return importlib.util.find_spec("isaaclab") is not None


@dataclass
class IsaacGeometryFrame:
    """Rule-relevant tensors extracted from an Isaac scene tick."""

    shots_fired: Tensor
    hit_source: Tensor
    hit_time_offset_s: Tensor
    hit_critical: Tensor
    chassis_power_w: Tensor


class IsaacRuleAdapter:
    """Normalize scene facts; damage and other rules remain in ``rm_referee``."""

    def normalize(
        self,
        game: GameState,
        frame: IsaacGeometryFrame,
        *,
        base: RuleInputs | None = None,
    ) -> RuleInputs:
        """Merge Isaac geometry into a complete normalized referee frame."""

        inputs = RuleInputs.empty(game) if base is None else base.clone()
        inputs.shots_fired.copy_(frame.shots_fired)
        inputs.hits = HitCandidates(
            source=frame.hit_source.clone(),
            time_offset_s=frame.hit_time_offset_s.clone(),
            critical=frame.hit_critical.clone(),
        )
        inputs.chassis_power_w.copy_(frame.chassis_power_w)
        return inputs
