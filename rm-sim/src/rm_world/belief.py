"""Team-shared enemy tracks for partially observed policy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import RadarQuality, Role, unit_roles, unit_teams
from rm_referee.state import GameState
from rm_world.kinematics import KinematicState


TARGET_STATE_FEATURE_NAMES = (
    "target_z_m",
    "target_yaw_sin",
    "target_yaw_cos",
    "target_hp_fraction",
    "target_alive",
    "target_level_fraction",
    "target_xp_fraction",
    "target_heat_17mm_fraction",
    "target_heat_42mm_fraction",
    "target_ammo_17mm_fraction",
    "target_ammo_42mm_fraction",
    "target_chassis_energy_fraction",
    "target_power_buffer_fraction",
    "target_weak",
    "target_invulnerable_fraction",
    "target_effective_vulnerability",
)
TARGET_STATE_DIM = len(TARGET_STATE_FEATURE_NAMES)
TARGET_ALIVE_INDEX = TARGET_STATE_FEATURE_NAMES.index("target_alive")


class EntitySource(IntEnum):
    """Information source attached to one team-target belief."""

    ORACLE = 0
    SHARED = 1
    LOCAL = 2
    RADAR_CONFIRMED = 3
    RADAR_ESTIMATE = 4
    PREDICTION = 5
    PRIOR = 6


ENTITY_SOURCE_COUNT = len(EntitySource)


@dataclass(frozen=True)
class BeliefConfig:
    """Calibratable ``[SIM]`` noise values for the Phase 1 tracker."""

    process_acceleration_std_mps2: float = 1.5
    prior_position_std_m: float = 1.5
    prior_velocity_std_mps: float = 1.0
    direct_position_std_m: float = 0.03
    direct_velocity_std_mps: float = 0.10
    radar_confirmed_position_std_m: float = 0.10
    radar_accurate_position_std_m: float = 0.40
    radar_half_accurate_position_std_m: float = 1.20
    radar_wrong_position_std_m: float = 2.50
    max_position_std_x_m: float = 14.0
    max_position_std_y_m: float = 7.5
    max_velocity_std_mps: float = 3.0
    age_feature_horizon_s: float = 10.0

    def validate(self) -> None:
        values = (
            self.process_acceleration_std_mps2,
            self.prior_position_std_m,
            self.prior_velocity_std_mps,
            self.direct_position_std_m,
            self.direct_velocity_std_mps,
            self.radar_confirmed_position_std_m,
            self.radar_accurate_position_std_m,
            self.radar_half_accurate_position_std_m,
            self.radar_wrong_position_std_m,
            self.max_position_std_x_m,
            self.max_position_std_y_m,
            self.max_velocity_std_mps,
            self.age_feature_horizon_s,
        )
        if any(value <= 0 for value in values):
            raise ValueError("belief noise scales and age horizon must be positive")


@dataclass(frozen=True)
class EntityBeliefView:
    """Read-only tensors indexed by ``[env, observer_team, target, ...]``."""

    mean: Tensor
    covariance: Tensor
    target_state: Tensor
    valid: Tensor
    position_age_s: Tensor
    state_age_s: Tensor
    source: Tensor


def build_target_state(game: GameState, world: KinematicState) -> Tensor:
    """Return target-only normalized state without observer-relative geometry."""

    hp_fraction = game.hp / torch.clamp(game.max_hp, min=1)
    heat_fraction = game.heat / torch.clamp(game.heat_limit, min=1)
    effective_vulnerability = torch.maximum(
        game.vulnerability_fraction,
        game.radar_vulnerability_fraction,
    )
    return torch.cat(
        (
            world.position_z[..., None],
            torch.sin(world.yaw)[..., None],
            torch.cos(world.yaw)[..., None],
            hp_fraction[..., None],
            game.alive.to(game.dtype)[..., None],
            game.level.to(game.dtype)[..., None] / 10.0,
            game.xp[..., None] / 5000.0,
            heat_fraction,
            torch.clamp(game.ammo.to(game.dtype) / 750.0, max=1.0),
            game.chassis_energy[..., None] / constants.CHASSIS_ENERGY_MAX,
            game.power_buffer_j[..., None] / constants.POWER_BUFFER_MAX_J,
            game.weak.to(game.dtype)[..., None],
            torch.clamp(game.invulnerable_s[..., None] / 30.0, max=1.0),
            effective_vulnerability[..., None],
        ),
        dim=-1,
    )


class EntityBeliefTracker:
    """Vectorized constant-velocity tracker updated once per policy transition."""

    def __init__(
        self,
        game: GameState,
        *,
        config: BeliefConfig | None = None,
    ) -> None:
        self.config = config or BeliefConfig()
        self.config.validate()
        shape = (game.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT)
        self.mean = torch.zeros((*shape, 4), device=game.device, dtype=game.dtype)
        self.covariance = torch.zeros((*shape, 4, 4), device=game.device, dtype=game.dtype)
        self.target_state = torch.zeros(
            (*shape, TARGET_STATE_DIM),
            device=game.device,
            dtype=game.dtype,
        )
        self.valid = torch.zeros(shape, device=game.device, dtype=torch.bool)
        self.position_age_s = torch.zeros(shape, device=game.device, dtype=game.dtype)
        self.state_age_s = torch.zeros(shape, device=game.device, dtype=game.dtype)
        self.source = torch.full(
            shape,
            EntitySource.PRIOR,
            device=game.device,
            dtype=torch.int8,
        )

    @staticmethod
    def shared_mask(device: torch.device) -> Tensor:
        """Return targets whose state is shared without a sensor measurement."""

        roles = unit_roles(device)
        target_teams = unit_teams(device)
        observer_teams = torch.arange(constants.TEAM_COUNT, device=device)
        own_target = observer_teams[:, None] == target_teams[None, :]
        building = (roles == Role.BASE) | (roles == Role.OUTPOST)
        return own_target | building[None, :]

    @staticmethod
    def _truth_mean(world: KinematicState) -> Tensor:
        target = torch.cat((world.position_xy, world.velocity_xy), dim=-1)
        return target[:, None, :, :].expand(-1, constants.TEAM_COUNT, -1, -1)

    def _diagonal_covariance(
        self,
        position_std: float,
        velocity_std: float,
    ) -> Tensor:
        diagonal = torch.tensor(
            (
                position_std**2,
                position_std**2,
                velocity_std**2,
                velocity_std**2,
            ),
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return torch.diag(diagonal).view(1, 1, 1, 4, 4)

    def reset(
        self,
        game: GameState,
        world: KinematicState,
        local_by_team: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Reset selected environments to known starts and broad enemy priors."""

        if env_ids is None:
            env_ids = torch.arange(game.num_envs, device=game.device)
        env_ids = env_ids.to(device=game.device, dtype=torch.long)
        expected = (game.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT)
        if local_by_team.shape != expected:
            raise ValueError("team-local visibility must have shape [env, team, target]")

        shared = self.shared_mask(game.device)[None, :, :].expand_as(local_by_team)
        direct = shared | local_by_team
        truth_mean = self._truth_mean(world)
        truth_state = build_target_state(game, world)[:, None, :, :].expand(
            -1,
            constants.TEAM_COUNT,
            -1,
            -1,
        )
        prior_covariance = self._diagonal_covariance(
            self.config.prior_position_std_m,
            self.config.prior_velocity_std_mps,
        ).expand_as(self.covariance)
        direct_covariance = self._diagonal_covariance(
            self.config.direct_position_std_m,
            self.config.direct_velocity_std_mps,
        ).expand_as(self.covariance)
        initial_source = torch.where(
            shared,
            torch.full_like(self.source, EntitySource.SHARED),
            torch.where(
                local_by_team,
                torch.full_like(self.source, EntitySource.LOCAL),
                torch.full_like(self.source, EntitySource.PRIOR),
            ),
        )

        self.mean[env_ids] = truth_mean[env_ids]
        self.target_state[env_ids] = truth_state[env_ids]
        self.covariance[env_ids] = torch.where(
            direct[env_ids, :, :, None, None],
            direct_covariance[env_ids],
            prior_covariance[env_ids],
        )
        self.valid[env_ids] = direct[env_ids]
        prior_age = torch.where(
            direct,
            torch.zeros_like(self.position_age_s),
            torch.full_like(
                self.position_age_s,
                self.config.age_feature_horizon_s,
            ),
        )
        self.position_age_s[env_ids] = prior_age[env_ids]
        self.state_age_s[env_ids] = prior_age[env_ids]
        self.source[env_ids] = initial_source[env_ids]

    def _bound_covariance(self, covariance: Tensor) -> Tensor:
        """Cap marginal uncertainty with a PSD-preserving congruence scale."""

        max_std = torch.tensor(
            (
                self.config.max_position_std_x_m,
                self.config.max_position_std_y_m,
                self.config.max_velocity_std_mps,
                self.config.max_velocity_std_mps,
            ),
            device=covariance.device,
            dtype=covariance.dtype,
        )
        standard_deviation = torch.sqrt(
            torch.clamp(
                torch.diagonal(covariance, dim1=-2, dim2=-1),
                min=1.0e-12,
            )
        )
        scale = torch.clamp(max_std / standard_deviation, max=1.0)
        return covariance * scale[..., :, None] * scale[..., None, :]

    def _predict(self, dt_s: Tensor) -> None:
        dt = dt_s[:, None, None]
        predicted_position = self.mean[..., :2] + self.mean[..., 2:] * dt[..., None]
        self.mean[..., :2] = torch.where(
            self.valid[..., None],
            predicted_position,
            self.mean[..., :2],
        )
        self.mean[..., 0].clamp_(-14.0, 14.0)
        self.mean[..., 1].clamp_(-7.5, 7.5)

        transition = torch.eye(
            4,
            device=self.mean.device,
            dtype=self.mean.dtype,
        ).view(1, 1, 1, 4, 4)
        transition = transition.expand(*self.mean.shape[:-1], 4, 4).clone()
        transition[..., 0, 2] = dt
        transition[..., 1, 3] = dt

        acceleration_variance = self.config.process_acceleration_std_mps2**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process = torch.zeros_like(self.covariance)
        process[..., 0, 0] = 0.25 * dt4 * acceleration_variance
        process[..., 1, 1] = 0.25 * dt4 * acceleration_variance
        process[..., 0, 2] = 0.5 * dt3 * acceleration_variance
        process[..., 2, 0] = process[..., 0, 2]
        process[..., 1, 3] = 0.5 * dt3 * acceleration_variance
        process[..., 3, 1] = process[..., 1, 3]
        process[..., 2, 2] = dt2 * acceleration_variance
        process[..., 3, 3] = dt2 * acceleration_variance
        predicted_covariance = transition @ self.covariance @ transition.transpose(-1, -2) + process
        self.covariance = torch.where(
            self.valid[..., None, None],
            predicted_covariance,
            self.covariance,
        )
        self.covariance = self._bound_covariance(
            0.5 * (self.covariance + self.covariance.transpose(-1, -2))
        )

    def _position_update(
        self,
        measurement_xy: Tensor,
        measurement_variance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        position_covariance = self.covariance[..., :2, :2]
        measurement_noise = torch.diag_embed(
            measurement_variance[..., None].expand(*measurement_variance.shape, 2)
        )
        innovation_covariance = position_covariance + measurement_noise
        cholesky = torch.linalg.cholesky(innovation_covariance)
        gain = torch.cholesky_solve(
            self.covariance[..., :, :2].transpose(-1, -2),
            cholesky,
        ).transpose(-1, -2)
        innovation = measurement_xy - self.mean[..., :2]
        mean = self.mean + (gain @ innovation[..., None]).squeeze(-1)

        identity = torch.eye(
            4,
            device=self.mean.device,
            dtype=self.mean.dtype,
        ).view(1, 1, 1, 4, 4)
        correction = identity.expand_as(self.covariance).clone()
        correction[..., :, :2].sub_(gain)
        covariance = correction @ self.covariance @ correction.transpose(
            -1, -2
        ) + gain @ measurement_noise @ gain.transpose(-1, -2)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return mean, covariance

    def advance(
        self,
        game: GameState,
        world: KinematicState,
        local_by_team: Tensor,
        *,
        dt_s: float,
    ) -> None:
        """Predict once, then fuse shared, local, and radar information."""

        if dt_s <= 0:
            raise ValueError("belief advance dt_s must be positive")
        expected = (game.num_envs, constants.TEAM_COUNT, constants.UNIT_COUNT)
        if local_by_team.shape != expected:
            raise ValueError("team-local visibility must have shape [env, team, target]")

        dt = torch.full(
            (game.num_envs,),
            dt_s,
            device=game.device,
            dtype=game.dtype,
        )
        self._predict(dt)
        age_increment = dt[:, None, None] * self.valid.to(game.dtype)
        self.position_age_s.add_(age_increment)
        self.state_age_s.add_(age_increment)
        self.source.copy_(
            torch.where(
                self.valid,
                torch.full_like(self.source, EntitySource.PREDICTION),
                torch.full_like(self.source, EntitySource.PRIOR),
            )
        )

        shared = self.shared_mask(game.device)[None, :, :].expand_as(local_by_team)
        direct = shared | local_by_team
        truth_mean = self._truth_mean(world)
        truth_state = build_target_state(game, world)[:, None, :, :].expand(
            -1,
            constants.TEAM_COUNT,
            -1,
            -1,
        )
        direct_covariance = self._diagonal_covariance(
            self.config.direct_position_std_m,
            self.config.direct_velocity_std_mps,
        ).expand_as(self.covariance)
        self.mean = torch.where(direct[..., None], truth_mean, self.mean)
        self.target_state = torch.where(
            direct[..., None],
            truth_state,
            self.target_state,
        )
        self.covariance = torch.where(
            direct[..., None, None],
            direct_covariance,
            self.covariance,
        )
        self.valid |= direct
        self.position_age_s.masked_fill_(direct, 0)
        self.state_age_s.masked_fill_(direct, 0)
        self.source.copy_(
            torch.where(
                shared,
                torch.full_like(self.source, EntitySource.SHARED),
                torch.where(
                    local_by_team,
                    torch.full_like(self.source, EntitySource.LOCAL),
                    self.source,
                ),
            )
        )

        target_mobile = unit_roles(game.device) <= Role.SENTRY
        radar_confirmed = game.radar_truth_visible & target_mobile[None, None, :] & ~direct
        confirmed_variance = self.config.radar_confirmed_position_std_m**2
        confirmed_mean, confirmed_covariance = self._position_update(
            truth_mean[..., :2],
            torch.full_like(game.radar_p, confirmed_variance),
        )
        self.mean = torch.where(
            radar_confirmed[..., None],
            confirmed_mean,
            self.mean,
        )
        self.covariance = torch.where(
            radar_confirmed[..., None, None],
            confirmed_covariance,
            self.covariance,
        )
        self.valid |= radar_confirmed
        self.position_age_s.masked_fill_(radar_confirmed, 0)
        self.source.masked_fill_(radar_confirmed, EntitySource.RADAR_CONFIRMED)

        fresh_report = (
            game.radar_report_valid
            & (game.radar_report_age_s <= 1.0e-6)
            & target_mobile[None, None, :]
        )
        radar_estimate = fresh_report & ~direct & ~radar_confirmed
        quality = game.radar_last_quality
        measurement_std = torch.where(
            quality == RadarQuality.ACCURATE,
            torch.full_like(game.radar_p, self.config.radar_accurate_position_std_m),
            torch.where(
                quality == RadarQuality.HALF_ACCURATE,
                torch.full_like(
                    game.radar_p,
                    self.config.radar_half_accurate_position_std_m,
                ),
                torch.full_like(game.radar_p, self.config.radar_wrong_position_std_m),
            ),
        )
        measured_mean, measured_covariance = self._position_update(
            game.radar_report_xy,
            torch.square(measurement_std),
        )
        self.mean = torch.where(radar_estimate[..., None], measured_mean, self.mean)
        self.covariance = torch.where(
            radar_estimate[..., None, None],
            measured_covariance,
            self.covariance,
        )
        self.valid |= radar_estimate
        self.position_age_s.masked_fill_(radar_estimate, 0)
        self.source.masked_fill_(radar_estimate, EntitySource.RADAR_ESTIMATE)
        self.mean[..., 0].clamp_(-14.0, 14.0)
        self.mean[..., 1].clamp_(-7.5, 7.5)
        self.mean[..., 2:].clamp_(-3.0, 3.0)
        self.covariance = self._bound_covariance(self.covariance)

    def view(self) -> EntityBeliefView:
        return EntityBeliefView(
            mean=self.mean,
            covariance=self.covariance,
            target_state=self.target_state,
            valid=self.valid,
            position_age_s=self.position_age_s,
            state_age_s=self.state_age_s,
            source=self.source,
        )
