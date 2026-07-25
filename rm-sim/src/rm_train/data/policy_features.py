"""Convert RMUC referee snapshots into deployable policy inputs.

The referee export contains global state, but the deployed actor receives
team-shared, range/LOS-limited observations.  This module reconstructs that
contract without exposing an enemy's hidden anchor state.  History is used
only causally for backward-difference velocity and a constant-velocity belief.
Belief history is intentionally truncated to the configured sample window;
targets last seen before that window fall back to a bounded neutral prior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import Role
from rm_train.data.rmuc_sqlite import RMUC_FEATURE_NAMES, TacticalSample
from rm_world.arena import ArenaGeometry
from rm_world.belief import BeliefConfig, EntitySource
from rm_world.observations import ENTITY_DIM, ENTITY_FEATURE_NAMES


AGENT_FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "yaw_sin",
    "yaw_cos",
    "velocity_x",
    "velocity_y",
    "hp_fraction",
    "alive",
    "level_fraction",
    "xp_fraction",
    "heat_17mm_fraction",
    "heat_42mm_fraction",
    "ammo_17mm_fraction",
    "ammo_42mm_fraction",
    "chassis_energy_fraction",
    "power_buffer_fraction",
    "weak",
    "invulnerable_fraction",
    "own_coin_fraction",
    "own_level_cap_fraction",
    "own_base_hp_fraction",
    "own_outpost_hp_fraction",
    "enemy_base_hp_fraction",
    "enemy_outpost_hp_fraction",
    "match_time_remaining_fraction",
    "role_hero",
    "role_engineer",
    "role_infantry_3",
    "role_infantry_4",
    "role_aerial",
    "role_sentry",
    "role_base",
    "role_outpost",
)
AGENT_DIM = len(AGENT_FEATURE_NAMES)

_ROLE_ROBOT_IDS = (1, 2, 3, 4, 6, 7, 10, 11)
_BUILDING_ROLES = (Role.BASE, Role.OUTPOST)
_FIELD_CENTER_XY = (14.0, 7.5)
_LOCAL_SENSOR_RANGE_M = 12.0
_GOAL_SCALE_M = 10.0
_MAX_LINEAR_SPEED_MPS = 3.0
_YAW_SPEED_THRESHOLD_MPS = 0.10
# The export's binary vulnerable flag has no published multiplier.  This
# conservative Phase 1 value is an explicit simulation choice [SIM].
_VULNERABLE_FRACTION_SIM = 0.15

_RAW = {name: index for index, name in enumerate(RMUC_FEATURE_NAMES)}
_AGENT = {name: index for index, name in enumerate(AGENT_FEATURE_NAMES)}
_ENTITY = {name: index for index, name in enumerate(ENTITY_FEATURE_NAMES)}


@dataclass(frozen=True)
class TacticalPolicyExample:
    """Six controllable actors derived from one team-perspective sample."""

    observations: Tensor
    entities: Tensor
    entity_mask: Tensor
    agent_ids: Tensor
    goal_xy: Tensor
    goal_valid: Tensor
    fire: Tensor
    fire_valid: Tensor


@dataclass(frozen=True)
class TacticalPolicyBatch:
    """Flattened policy inputs and imitation labels; ``N = 6 * samples``."""

    observations: Tensor
    entities: Tensor
    entity_mask: Tensor
    agent_ids: Tensor
    goal_xy: Tensor
    goal_valid: Tensor
    fire: Tensor
    fire_valid: Tensor

    def pin_memory(self) -> TacticalPolicyBatch:
        """Let ``DataLoader(pin_memory=True)`` recurse through this dataclass."""

        return TacticalPolicyBatch(
            observations=self.observations.pin_memory(),
            entities=self.entities.pin_memory(),
            entity_mask=self.entity_mask.pin_memory(),
            agent_ids=self.agent_ids.pin_memory(),
            goal_xy=self.goal_xy.pin_memory(),
            goal_valid=self.goal_valid.pin_memory(),
            fire=self.fire.pin_memory(),
            fire_valid=self.fire_valid.pin_memory(),
        )


@dataclass(frozen=True)
class TacticalPolicyCollator:
    """Pickle-safe configurable ``DataLoader.collate_fn``."""

    goal_scale_m: float = _GOAL_SCALE_M
    local_sensor_range_m: float = _LOCAL_SENSOR_RANGE_M

    def __post_init__(self) -> None:
        _validate_scales(self.goal_scale_m, self.local_sensor_range_m)

    def __call__(self, samples: Sequence[TacticalSample]) -> TacticalPolicyBatch:
        return collate_tactical_policy(
            samples,
            goal_scale_m=self.goal_scale_m,
            local_sensor_range_m=self.local_sensor_range_m,
        )


def _validate_scales(goal_scale_m: float, local_sensor_range_m: float) -> None:
    if goal_scale_m <= 0.0:
        raise ValueError("goal_scale_m must be positive")
    if local_sensor_range_m <= 0.0:
        raise ValueError("local_sensor_range_m must be positive")


def _absolute_slot(robot_id: int) -> int:
    team = int(robot_id >= 100)
    role_robot_id = robot_id - 100 * team
    try:
        role = _ROLE_ROBOT_IDS.index(role_robot_id)
    except ValueError as error:
        raise ValueError(f"unsupported RMUC robot_id {robot_id}") from error
    return team * constants.ROLES_PER_TEAM + role


def _absolute_order(sample: TacticalSample) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return history tensors reordered from perspective order to global slots."""

    source_by_slot = torch.full((constants.UNIT_COUNT,), -1, dtype=torch.long)
    for source, robot_id in enumerate(sample.robot_ids.tolist()):
        slot = _absolute_slot(int(robot_id))
        if source_by_slot[slot] >= 0:
            raise ValueError(f"duplicate RMUC robot slot {slot}")
        source_by_slot[slot] = source
    if (source_by_slot < 0).any():
        raise ValueError("tactical sample must contain all 16 absolute robot slots")
    return (
        sample.features[:, source_by_slot].to(dtype=torch.float32),
        sample.feature_valid[:, source_by_slot].to(dtype=torch.bool),
        sample.position_valid[:, source_by_slot].to(dtype=torch.bool),
        source_by_slot,
    )


def _center_positions(
    features: Tensor,
    valid: Tensor,
    position_valid: Tensor,
    arena: ArenaGeometry,
) -> Tensor:
    positions = torch.zeros(
        (*features.shape[:2], 2),
        device=features.device,
        dtype=features.dtype,
    )
    positions[..., 0] = features[..., _RAW["x"]] - _FIELD_CENTER_XY[0]
    positions[..., 1] = features[..., _RAW["y"]] - _FIELD_CENTER_XY[1]
    positions = torch.where(position_valid[..., None], positions, torch.zeros_like(positions))

    base = arena.base_centers(device=features.device, dtype=features.dtype)
    outpost = arena.outpost_centers(device=features.device, dtype=features.dtype)
    for team in range(constants.TEAM_COUNT):
        for role, center in ((Role.BASE, base[team]), (Role.OUTPOST, outpost[team])):
            slot = team * constants.ROLES_PER_TEAM + role
            positions[:, slot] = center
            position_valid[:, slot] = True
            features[:, slot, _RAW["z"]] = arena.terrain_height(center)
            valid[:, slot, _RAW["z"]] = True
    return positions


def _known_alive(features: Tensor, valid: Tensor) -> Tensor:
    hp_valid = valid[..., _RAW["hp"]]
    return hp_valid & (features[..., _RAW["hp"]] > 0.0)


def _team_visibility(
    positions: Tensor,
    position_valid: Tensor,
    features: Tensor,
    valid: Tensor,
    *,
    team: int,
    arena: ArenaGeometry,
    range_m: float,
) -> Tensor:
    """Return ``[history, 16]`` team-shared synthetic local visibility."""

    history = positions.shape[0]
    result = torch.zeros(
        (history, constants.UNIT_COUNT),
        device=positions.device,
        dtype=torch.bool,
    )
    own_mobile = torch.arange(team * 8, team * 8 + 6, device=positions.device)
    enemy_start = (1 - team) * 8
    enemy_mobile = torch.arange(enemy_start, enemy_start + 6, device=positions.device)

    observer_xy = positions[:, own_mobile, None, :].expand(-1, -1, 6, -1)
    target_xy = positions[:, None, enemy_mobile, :].expand(-1, 6, -1, -1)
    distance = torch.linalg.vector_norm(target_xy - observer_xy, dim=-1)
    line_of_sight = arena.line_of_sight(observer_xy, target_xy)
    observers_active = position_valid[:, own_mobile] & _known_alive(
        features[:, own_mobile], valid[:, own_mobile]
    )
    targets_tracked = position_valid[:, enemy_mobile]
    visible = (
        (distance <= range_m)
        & line_of_sight
        & observers_active[:, :, None]
        & targets_tracked[:, None, :]
    ).any(dim=1)
    result[:, enemy_mobile] = visible
    return result


def _last_index(mask: Tensor, end_index: int) -> int | None:
    indices = torch.nonzero(mask[: end_index + 1], as_tuple=False).flatten()
    return None if indices.numel() == 0 else int(indices[-1].item())


def _causal_velocity(
    positions: Tensor,
    times_s: Tensor,
    measurement_mask: Tensor,
    end_index: int,
) -> Tensor:
    """Estimate velocity using only the last two allowed samples at or before ``end``."""

    indices = torch.nonzero(measurement_mask[: end_index + 1], as_tuple=False).flatten()
    if indices.numel() < 2:
        return torch.zeros(2, device=positions.device, dtype=positions.dtype)
    current = int(indices[-1].item())
    previous = int(indices[-2].item())
    delta_t = float((times_s[current] - times_s[previous]).item())
    if delta_t <= 0.0:
        return torch.zeros(2, device=positions.device, dtype=positions.dtype)
    velocity = (positions[current] - positions[previous]) / delta_t
    speed = torch.linalg.vector_norm(velocity)
    scale = torch.clamp(
        velocity.new_tensor(_MAX_LINEAR_SPEED_MPS) / speed.clamp_min(1.0e-8),
        max=1.0,
    )
    return velocity * scale


def _yaw_features(velocity_xy: Tensor) -> tuple[Tensor, Tensor]:
    speed = torch.linalg.vector_norm(velocity_xy)
    if float(speed.item()) <= _YAW_SPEED_THRESHOLD_MPS:
        return velocity_xy.new_tensor(0.0), velocity_xy.new_tensor(1.0)
    # Chassis heading is inferred from causal motion.  The export's turret yaw
    # must not be substituted for chassis yaw.
    return velocity_xy[1] / speed, velocity_xy[0] / speed


def _fraction(
    features: Tensor,
    valid: Tensor,
    time_index: int,
    slot: int,
    numerator: str,
    denominator: str,
) -> Tensor:
    numerator_valid = bool(valid[time_index, slot, _RAW[numerator]].item())
    denominator_valid = bool(valid[time_index, slot, _RAW[denominator]].item())
    denominator_value = features[time_index, slot, _RAW[denominator]]
    if not numerator_valid or not denominator_valid or float(denominator_value.item()) <= 0.0:
        return features.new_tensor(0.0)
    return torch.clamp(
        features[time_index, slot, _RAW[numerator]] / denominator_value,
        min=0.0,
        max=1.0,
    )


def _raw_or_zero(
    features: Tensor,
    valid: Tensor,
    time_index: int,
    slot: int,
    feature: str,
) -> Tensor:
    if bool(valid[time_index, slot, _RAW[feature]].item()):
        return features[time_index, slot, _RAW[feature]]
    return features.new_tensor(0.0)


def _structure_hp_fraction(
    features: Tensor,
    valid: Tensor,
    time_index: int,
    slot: int,
) -> Tensor:
    return _fraction(features, valid, time_index, slot, "hp", "max_hp")


def _agent_observation(
    *,
    slot: int,
    team: int,
    anchor: int,
    positions: Tensor,
    position_valid: Tensor,
    features: Tensor,
    valid: Tensor,
    times_s: Tensor,
    anchor_s: float,
) -> Tensor:
    observation = torch.zeros(AGENT_DIM, device=features.device, dtype=features.dtype)
    current_position_valid = bool(position_valid[anchor, slot].item())
    if current_position_valid:
        observation[_AGENT["position_x"]] = positions[anchor, slot, 0] / 14.0
        observation[_AGENT["position_y"]] = positions[anchor, slot, 1] / 7.5
    observation[_AGENT["position_z"]] = _raw_or_zero(features, valid, anchor, slot, "z") / 5.0

    velocity = _causal_velocity(
        positions[:, slot],
        times_s,
        position_valid[:, slot],
        anchor,
    )
    yaw_sin, yaw_cos = _yaw_features(velocity)
    observation[_AGENT["yaw_sin"]] = yaw_sin
    observation[_AGENT["yaw_cos"]] = yaw_cos
    observation[_AGENT["velocity_x"] : _AGENT["velocity_y"] + 1] = velocity / 3.0

    observation[_AGENT["hp_fraction"]] = _fraction(features, valid, anchor, slot, "hp", "max_hp")
    hp_valid = bool(valid[anchor, slot, _RAW["hp"]].item())
    observation[_AGENT["alive"]] = float(
        hp_valid and float(features[anchor, slot, _RAW["hp"]].item()) > 0.0
    )
    observation[_AGENT["heat_17mm_fraction"]] = _fraction(
        features, valid, anchor, slot, "heat_17mm", "heat_limit_17mm"
    )
    observation[_AGENT["heat_42mm_fraction"]] = _fraction(
        features, valid, anchor, slot, "heat_42mm", "heat_limit_42mm"
    )
    observation[_AGENT["own_coin_fraction"]] = torch.clamp(
        _raw_or_zero(features, valid, anchor, slot, "team_coin_remaining") / 2000.0,
        min=0.0,
        max=1.0,
    )

    own_base = team * 8 + Role.BASE
    own_outpost = team * 8 + Role.OUTPOST
    enemy_base = (1 - team) * 8 + Role.BASE
    enemy_outpost = (1 - team) * 8 + Role.OUTPOST
    observation[_AGENT["own_base_hp_fraction"]] = _structure_hp_fraction(
        features, valid, anchor, own_base
    )
    observation[_AGENT["own_outpost_hp_fraction"]] = _structure_hp_fraction(
        features, valid, anchor, own_outpost
    )
    observation[_AGENT["enemy_base_hp_fraction"]] = _structure_hp_fraction(
        features, valid, anchor, enemy_base
    )
    observation[_AGENT["enemy_outpost_hp_fraction"]] = _structure_hp_fraction(
        features, valid, anchor, enemy_outpost
    )
    observation[_AGENT["match_time_remaining_fraction"]] = max(
        0.0,
        min(1.0, (constants.MATCH_DURATION_S - anchor_s) / constants.MATCH_DURATION_S),
    )
    role = slot % constants.ROLES_PER_TEAM
    observation[_AGENT["role_hero"] + role] = 1.0
    return observation


def _diagonal_covariance(
    reference: Tensor,
    position_std_m: float,
    velocity_std_mps: float,
) -> Tensor:
    diagonal = reference.new_tensor(
        (
            position_std_m**2,
            position_std_m**2,
            velocity_std_mps**2,
            velocity_std_mps**2,
        )
    )
    return torch.diag(diagonal)


def _predicted_covariance(
    reference: Tensor,
    age_s: float,
    config: BeliefConfig,
) -> Tensor:
    covariance = _diagonal_covariance(
        reference,
        config.direct_position_std_m,
        config.direct_velocity_std_mps,
    )
    if age_s <= 0.0:
        return covariance
    transition = torch.eye(4, device=reference.device, dtype=reference.dtype)
    transition[0, 2] = age_s
    transition[1, 3] = age_s
    acceleration_variance = config.process_acceleration_std_mps2**2
    dt2 = age_s**2
    dt3 = age_s**3
    dt4 = age_s**4
    process = torch.zeros_like(covariance)
    process[0, 0] = 0.25 * dt4 * acceleration_variance
    process[1, 1] = 0.25 * dt4 * acceleration_variance
    process[0, 2] = process[2, 0] = 0.5 * dt3 * acceleration_variance
    process[1, 3] = process[3, 1] = 0.5 * dt3 * acceleration_variance
    process[2, 2] = dt2 * acceleration_variance
    process[3, 3] = dt2 * acceleration_variance
    covariance = transition @ covariance @ transition.T + process

    max_std = reference.new_tensor(
        (
            config.max_position_std_x_m,
            config.max_position_std_y_m,
            config.max_velocity_std_mps,
            config.max_velocity_std_mps,
        )
    )
    std = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=1.0e-12))
    scale = torch.clamp(max_std / std, max=1.0)
    return covariance * scale[:, None] * scale[None, :]


def _pack_covariance(covariance: Tensor) -> Tensor:
    scale = covariance.new_tensor((14.0, 7.5, 3.0, 3.0))
    normalized = covariance / (scale[:, None] * scale[None, :])
    pairs = (
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 3),
    )
    return torch.stack(tuple(normalized[row, column] for row, column in pairs))


def _target_state(
    *,
    slot: int,
    time_index: int | None,
    velocity: Tensor,
    features: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    zero = features.new_tensor(0.0)
    yaw_sin, yaw_cos = _yaw_features(velocity)
    role = slot % constants.ROLES_PER_TEAM
    if role == Role.BASE:
        yaw_sin = zero
        yaw_cos = features.new_tensor(1.0 if slot < constants.ROLES_PER_TEAM else -1.0)
    elif role == Role.OUTPOST:
        # The referee export does not contain the rotating outpost angle.
        yaw_sin = zero
        yaw_cos = zero
    state = {
        "target_z": zero,
        "yaw_sin": yaw_sin,
        "yaw_cos": yaw_cos,
        "hp_fraction": zero,
        "alive": zero,
        "heat_17": zero,
        "heat_42": zero,
        "effective_vulnerability": zero,
    }
    if time_index is None:
        return state
    state["target_z"] = _raw_or_zero(features, valid, time_index, slot, "z")
    state["hp_fraction"] = _fraction(features, valid, time_index, slot, "hp", "max_hp")
    hp_valid = bool(valid[time_index, slot, _RAW["hp"]].item())
    state["alive"] = features.new_tensor(
        float(hp_valid and float(features[time_index, slot, _RAW["hp"]].item()) > 0.0)
    )
    state["heat_17"] = _fraction(features, valid, time_index, slot, "heat_17mm", "heat_limit_17mm")
    state["heat_42"] = _fraction(features, valid, time_index, slot, "heat_42mm", "heat_limit_42mm")
    vulnerable = _raw_or_zero(features, valid, time_index, slot, "vulnerable")
    state["effective_vulnerability"] = features.new_tensor(
        _VULNERABLE_FRACTION_SIM if float(vulnerable.item()) >= 0.5 else 0.0
    )
    return state


def _entity_belief(
    *,
    target_slot: int,
    team: int,
    anchor: int,
    positions: Tensor,
    position_valid: Tensor,
    visible: Tensor,
    features: Tensor,
    valid: Tensor,
    times_s: Tensor,
    belief_config: BeliefConfig,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, float, EntitySource, bool]:
    target_team = target_slot // constants.ROLES_PER_TEAM
    target_role = target_slot % constants.ROLES_PER_TEAM
    shared = target_team == team or target_role in _BUILDING_ROLES
    if shared:
        measurements = position_valid[:, target_slot]
    else:
        measurements = position_valid[:, target_slot] & visible[:, target_slot]

    measurement_index = _last_index(measurements, anchor)
    velocity = _causal_velocity(
        positions[:, target_slot],
        times_s,
        measurements,
        anchor,
    )
    current_direct = bool(measurements[anchor].item())
    if measurement_index is None:
        mean_xy = positions.new_zeros(2)
        covariance = _diagonal_covariance(
            positions,
            belief_config.prior_position_std_m,
            belief_config.prior_velocity_std_mps,
        )
        age_s = belief_config.age_feature_horizon_s
        source = EntitySource.PRIOR
        velocity.zero_()
    else:
        age_s = max(
            0.0,
            float((times_s[anchor] - times_s[measurement_index]).item()),
        )
        mean_xy = positions[measurement_index, target_slot] + velocity * age_s
        mean_xy = torch.stack(
            (
                mean_xy[0].clamp(-14.0, 14.0),
                mean_xy[1].clamp(-7.5, 7.5),
            )
        )
        covariance = _predicted_covariance(positions, age_s, belief_config)
        if shared:
            source = EntitySource.SHARED
        elif current_direct:
            source = EntitySource.LOCAL
        else:
            source = EntitySource.PREDICTION
    target_state = _target_state(
        slot=target_slot,
        time_index=measurement_index,
        velocity=velocity,
        features=features,
        valid=valid,
    )
    # A shared-state category does not make a missing current measurement
    # usable. Buildings remain direct because their official positions were
    # injected at every history step.
    entity_direct = current_direct
    return mean_xy, velocity, target_state, covariance, age_s, source, entity_direct


def _entity_features(
    *,
    observer_slot: int,
    target_slot: int,
    team: int,
    anchor: int,
    observer_xy: Tensor,
    observer_z: Tensor,
    positions: Tensor,
    position_valid: Tensor,
    visible: Tensor,
    features: Tensor,
    valid: Tensor,
    times_s: Tensor,
    belief_config: BeliefConfig,
) -> tuple[Tensor, bool]:
    (
        mean_xy,
        velocity,
        state,
        covariance,
        age_s,
        source,
        direct,
    ) = _entity_belief(
        target_slot=target_slot,
        team=team,
        anchor=anchor,
        positions=positions,
        position_valid=position_valid,
        visible=visible,
        features=features,
        valid=valid,
        times_s=times_s,
        belief_config=belief_config,
    )
    result = torch.zeros(ENTITY_DIM, device=features.device, dtype=features.dtype)
    relative = mean_xy - observer_xy
    distance = torch.linalg.vector_norm(relative)
    result[_ENTITY["relative_x"]] = relative[0] / 28.0
    result[_ENTITY["relative_y"]] = relative[1] / 15.0
    result[_ENTITY["relative_z"]] = (state["target_z"] - observer_z) / 5.0
    result[_ENTITY["distance_xy"]] = distance / 31.0
    result[_ENTITY["target_yaw_sin"]] = state["yaw_sin"]
    result[_ENTITY["target_yaw_cos"]] = state["yaw_cos"]
    result[_ENTITY["target_velocity_x"] : _ENTITY["target_velocity_y"] + 1] = velocity / 3.0
    result[_ENTITY["target_hp_fraction"]] = state["hp_fraction"]
    result[_ENTITY["target_alive"]] = state["alive"]
    result[_ENTITY["target_heat_17mm_fraction"]] = state["heat_17"]
    result[_ENTITY["target_heat_42mm_fraction"]] = state["heat_42"]
    result[_ENTITY["target_effective_vulnerability"]] = state["effective_vulnerability"]
    result[_ENTITY["same_team"]] = float(target_slot // constants.ROLES_PER_TEAM == team)
    role = target_slot % constants.ROLES_PER_TEAM
    result[_ENTITY["target_role_hero"] + role] = 1.0
    age_fraction = min(age_s / belief_config.age_feature_horizon_s, 1.0)
    result[_ENTITY["position_observation_age_fraction"]] = age_fraction
    result[_ENTITY["state_observation_age_fraction"]] = age_fraction
    covariance_start = _ENTITY["covariance_x_x"]
    result[covariance_start : covariance_start + 10] = _pack_covariance(covariance)
    result[_ENTITY["source_oracle"] + int(source)] = 1.0
    return result, direct


def featurize_tactical_sample(
    sample: TacticalSample,
    *,
    goal_scale_m: float = _GOAL_SCALE_M,
    local_sensor_range_m: float = _LOCAL_SENSOR_RANGE_M,
    arena: ArenaGeometry | None = None,
    belief_config: BeliefConfig | None = None,
) -> TacticalPolicyExample:
    """Build six actor inputs without mirroring blue-team world coordinates."""

    _validate_scales(goal_scale_m, local_sensor_range_m)
    arena = arena or ArenaGeometry()
    belief_config = belief_config or BeliefConfig()
    belief_config.validate()
    features, valid, position_valid, source_by_slot = _absolute_order(sample)
    position_valid = position_valid.clone()
    positions = _center_positions(features, valid, position_valid, arena)
    times_s = sample.times_s.to(device=features.device, dtype=features.dtype)
    anchor = features.shape[0] - 1
    team = sample.key.team
    visible = _team_visibility(
        positions,
        position_valid,
        features,
        valid,
        team=team,
        arena=arena,
        range_m=local_sensor_range_m,
    )
    own_slots = torch.arange(
        team * constants.ROLES_PER_TEAM,
        team * constants.ROLES_PER_TEAM + 6,
        device=features.device,
        dtype=torch.long,
    )

    observations = torch.stack(
        tuple(
            _agent_observation(
                slot=int(slot.item()),
                team=team,
                anchor=anchor,
                positions=positions,
                position_valid=position_valid,
                features=features,
                valid=valid,
                times_s=times_s,
                anchor_s=sample.key.anchor_s,
            )
            for slot in own_slots
        )
    )
    entity_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    for observer_slot_tensor in own_slots:
        observer_slot = int(observer_slot_tensor.item())
        observer_xy = (
            positions[anchor, observer_slot]
            if bool(position_valid[anchor, observer_slot].item())
            else positions.new_zeros(2)
        )
        observer_z = _raw_or_zero(features, valid, anchor, observer_slot, "z")
        target_rows: list[Tensor] = []
        direct_values: list[bool] = []
        for target_slot in range(constants.UNIT_COUNT):
            target, direct = _entity_features(
                observer_slot=observer_slot,
                target_slot=target_slot,
                team=team,
                anchor=anchor,
                observer_xy=observer_xy,
                observer_z=observer_z,
                positions=positions,
                position_valid=position_valid,
                visible=visible,
                features=features,
                valid=valid,
                times_s=times_s,
                belief_config=belief_config,
            )
            target_rows.append(target)
            direct_values.append(direct)
        entity_rows.append(torch.stack(tuple(target_rows)))
        mask_rows.append(torch.tensor(direct_values, device=features.device, dtype=torch.bool))
    entities = torch.stack(tuple(entity_rows))
    entity_mask = torch.stack(tuple(mask_rows))

    labels_by_slot = source_by_slot.to(device=sample.future_displacement_xy.device)
    future_displacement = sample.future_displacement_xy[labels_by_slot].to(
        device=features.device,
        dtype=features.dtype,
    )
    goal_valid = sample.future_displacement_valid[labels_by_slot].to(
        device=features.device,
        dtype=torch.bool,
    )
    fired = sample.fired[labels_by_slot].to(device=features.device, dtype=features.dtype)
    fire_valid = sample.fire_valid[labels_by_slot].to(
        device=features.device,
        dtype=torch.bool,
    )
    return TacticalPolicyExample(
        observations=observations,
        entities=entities,
        entity_mask=entity_mask,
        agent_ids=own_slots,
        goal_xy=torch.clamp(
            future_displacement[own_slots] / goal_scale_m,
            min=-1.0,
            max=1.0,
        ),
        goal_valid=goal_valid[own_slots],
        fire=fired[own_slots],
        fire_valid=fire_valid[own_slots],
    )


def collate_tactical_policy(
    samples: Sequence[TacticalSample],
    *,
    goal_scale_m: float = _GOAL_SCALE_M,
    local_sensor_range_m: float = _LOCAL_SENSOR_RANGE_M,
) -> TacticalPolicyBatch:
    """Featurize and concatenate samples for the imitation trainer."""

    if not samples:
        raise ValueError("cannot collate an empty tactical policy batch")
    arena = ArenaGeometry()
    belief_config = BeliefConfig()
    examples = tuple(
        featurize_tactical_sample(
            sample,
            goal_scale_m=goal_scale_m,
            local_sensor_range_m=local_sensor_range_m,
            arena=arena,
            belief_config=belief_config,
        )
        for sample in samples
    )
    return TacticalPolicyBatch(
        observations=torch.cat(tuple(item.observations for item in examples)),
        entities=torch.cat(tuple(item.entities for item in examples)),
        entity_mask=torch.cat(tuple(item.entity_mask for item in examples)),
        agent_ids=torch.cat(tuple(item.agent_ids for item in examples)),
        goal_xy=torch.cat(tuple(item.goal_xy for item in examples)),
        goal_valid=torch.cat(tuple(item.goal_valid for item in examples)),
        fire=torch.cat(tuple(item.fire for item in examples)),
        fire_valid=torch.cat(tuple(item.fire_valid for item in examples)),
    )
