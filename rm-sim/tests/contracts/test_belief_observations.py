from __future__ import annotations

from dataclasses import fields

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import RadarQuality, Role, Team, slot
from rm_referee.state import GameState
from rm_train import SharedMAPPOPolicy
from rm_world import EntityBeliefTracker, EntitySource, TorchEnvConfig, TorchRMArena
from rm_world.kinematics import KinematicState
from rm_world.observations import WorldObservation


TRACKER_FIELDS = (
    "mean",
    "covariance",
    "target_state",
    "valid",
    "position_age_s",
    "state_age_s",
    "source",
)


def _acquired_enemy_tracker() -> tuple[
    GameState,
    KinematicState,
    EntityBeliefTracker,
    Tensor,
    int,
]:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    tracker = EntityBeliefTracker(game)
    target = slot(Team.BLUE, Role.HERO)
    world.position_xy[0, target] = torch.tensor([0.0, 0.0])
    world.velocity_xy[0, target] = torch.tensor([1.0, -0.5])
    local = torch.zeros(
        (1, constants.TEAM_COUNT, constants.UNIT_COUNT),
        dtype=torch.bool,
    )
    local[0, Team.RED, target] = True
    tracker.reset(game, world, local)
    return game, world, tracker, local, target


def _assert_observation_equal(left: WorldObservation, right: WorldObservation) -> None:
    for field in fields(WorldObservation):
        torch.testing.assert_close(
            getattr(left, field.name),
            getattr(right, field.name),
            rtol=0,
            atol=0,
        )


def test_repeated_observe_is_idempotent_and_does_not_advance_belief() -> None:
    env = TorchRMArena(TorchEnvConfig(num_envs=2, seed=17, observation_mode="belief"))
    tracker = env.belief_tracker
    assert tracker is not None
    tracker_before = {name: getattr(tracker, name).clone() for name in TRACKER_FIELDS}

    first = env.observe()
    second = env.observe()

    _assert_observation_equal(first, second)
    for name, expected in tracker_before.items():
        torch.testing.assert_close(
            getattr(tracker, name),
            expected,
            rtol=0,
            atol=0,
        )


def test_hidden_truth_does_not_change_entity_input_or_target_mask_after_loss() -> None:
    env = TorchRMArena(TorchEnvConfig(num_envs=1, seed=19, observation_mode="belief"))
    tracker = env.belief_tracker
    assert tracker is not None
    env.observation_builder.local_sensor_range_m = 0.25
    observer = slot(Team.RED, Role.HERO)
    target = slot(Team.BLUE, Role.HERO)

    env.game.controller_offline[0, : constants.ROLES_PER_TEAM] = True
    env.game.controller_offline[0, observer] = False
    env.world.position_xy[0, target] = env.world.position_xy[0, observer]
    env.world.velocity_xy[0, target] = torch.tensor([0.6, -0.2])
    acquired = env.observation_builder.visibility(env.game, env.world)
    assert acquired.local_by_team[0, Team.RED, target]
    tracker.advance(
        env.game,
        env.world,
        acquired.local_by_team,
        dt_s=env.config.policy_dt_s,
    )

    env.game.controller_offline[0, observer] = True
    env.world.position_xy[0, target] = torch.tensor([12.0, 6.0])
    lost = env.observation_builder.visibility(env.game, env.world)
    assert not lost.local_by_team[0, Team.RED, target]
    tracker.advance(
        env.game,
        env.world,
        lost.local_by_team,
        dt_s=env.config.policy_dt_s,
    )
    before = env.observe()
    assert not before.entity_mask[0, observer, target]
    assert not before.target_mask[0, observer, Role.HERO]
    torch.manual_seed(29)
    policy = SharedMAPPOPolicy(hidden_dim=32, role_embedding_dim=8)
    before_policy = policy.act(
        before.agents,
        before.entities,
        before.entity_mask,
        before.central,
        before.target_mask,
        before.fire_mask,
        deterministic=True,
    )

    env.world.position_xy[0, target] = torch.tensor([-12.0, -6.0])
    env.world.position_z[0, target] = 4.0
    env.world.velocity_xy[0, target] = torch.tensor([-2.5, 2.0])
    env.world.yaw[0, target] = -1.25
    env.game.hp[0, target] = 0
    env.game.alive[0, target] = False
    env.game.level[0, target] = 9
    env.game.xp[0, target] = 4500
    env.game.heat[0, target] = torch.tensor([90.0, 180.0])
    env.game.ammo[0, target] = 0
    env.game.chassis_energy[0, target] = 0
    env.game.power_buffer_j[0, target] = 0
    env.game.weak[0, target] = True
    env.game.invulnerable_s[0, target] = 20
    env.game.vulnerability_fraction[0, target] = 1
    after = env.observe()

    assert not after.entity_mask[0, observer, target]
    torch.testing.assert_close(
        after.entities[0, observer, target],
        before.entities[0, observer, target],
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        after.target_mask[0, observer],
        before.target_mask[0, observer],
    )
    after_policy = policy.act(
        after.agents,
        after.entities,
        after.entity_mask,
        after.central,
        after.target_mask,
        after.fire_mask,
        deterministic=True,
    )
    for name in before_policy.action.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(after_policy.action, name)[:, : constants.ROLES_PER_TEAM],
            getattr(before_policy.action, name)[:, : constants.ROLES_PER_TEAM],
            rtol=0,
            atol=0,
        )


def test_hidden_track_uses_constant_velocity_and_covariance_remains_psd() -> None:
    game, world, tracker, local, target = _acquired_enemy_tracker()
    initial_mean = tracker.mean[0, Team.RED, target].clone()
    initial_covariance = tracker.covariance[0, Team.RED, target].clone()
    local.zero_()

    tracker.advance(game, world, local, dt_s=0.4)

    estimate = tracker.mean[0, Team.RED, target]
    expected_position = initial_mean[:2] + initial_mean[2:] * 0.4
    torch.testing.assert_close(estimate[:2], expected_position)
    torch.testing.assert_close(estimate[2:], initial_mean[2:])
    covariance = tracker.covariance[0, Team.RED, target]
    torch.testing.assert_close(covariance, covariance.T)
    assert torch.linalg.eigvalsh(covariance).min() >= -1.0e-7
    assert torch.trace(covariance) > torch.trace(initial_covariance)
    assert tracker.position_age_s[0, Team.RED, target] == 0.4
    assert tracker.state_age_s[0, Team.RED, target] == 0.4
    assert tracker.source[0, Team.RED, target] == EntitySource.PREDICTION


def test_prior_does_not_diffuse_and_lost_track_covariance_is_physically_bounded() -> None:
    game, world, tracker, local, target = _acquired_enemy_tracker()
    prior_target = slot(Team.BLUE, Role.ENGINEER)
    prior_covariance = tracker.covariance[0, Team.RED, prior_target].clone()
    local.zero_()

    for _ in range(500):
        tracker.advance(game, world, local, dt_s=0.2)

    torch.testing.assert_close(
        tracker.covariance[0, Team.RED, prior_target],
        prior_covariance,
    )
    covariance = tracker.covariance[0, Team.RED, target]
    diagonal = torch.diagonal(covariance)
    upper_bound = torch.tensor([14.0**2, 7.5**2, 3.0**2, 3.0**2])
    assert torch.all(diagonal <= upper_bound + 1.0e-5)
    assert torch.linalg.eigvalsh(covariance).min() >= -1.0e-5


def test_local_reacquisition_replaces_prediction_and_reduces_uncertainty() -> None:
    game, world, tracker, local, target = _acquired_enemy_tracker()
    local.zero_()
    tracker.advance(game, world, local, dt_s=1.0)
    predicted_covariance = tracker.covariance[0, Team.RED, target].clone()

    world.position_xy[0, target] = torch.tensor([-4.0, 2.0])
    world.velocity_xy[0, target] = torch.tensor([-0.25, 0.75])
    game.hp[0, target] = game.max_hp[0, target] / 2
    local[0, Team.RED, target] = True
    tracker.advance(game, world, local, dt_s=0.2)

    expected = torch.cat((world.position_xy[0, target], world.velocity_xy[0, target]))
    torch.testing.assert_close(tracker.mean[0, Team.RED, target], expected)
    assert tracker.target_state[0, Team.RED, target, 3] == 0.5
    assert tracker.position_age_s[0, Team.RED, target] == 0
    assert tracker.state_age_s[0, Team.RED, target] == 0
    assert tracker.source[0, Team.RED, target] == EntitySource.LOCAL
    covariance = tracker.covariance[0, Team.RED, target]
    assert torch.trace(covariance) < torch.trace(predicted_covariance)
    torch.testing.assert_close(
        torch.diagonal(covariance),
        torch.tensor([0.03**2, 0.03**2, 0.10**2, 0.10**2]),
    )


def test_radar_report_estimates_then_threshold_confirms_position() -> None:
    game, world, tracker, local, target = _acquired_enemy_tracker()
    local.zero_()
    report_xy = torch.tensor([3.0, -1.5])
    game.radar_p[0, Team.RED, target] = 99
    game.radar_report_xy[0, Team.RED, target] = report_xy
    game.radar_report_valid[0, Team.RED, target] = True
    game.radar_report_age_s[0, Team.RED, target] = 0
    game.radar_last_quality[0, Team.RED, target] = RadarQuality.HALF_ACCURATE
    before_distance = torch.linalg.vector_norm(tracker.mean[0, Team.RED, target, :2] - report_xy)

    tracker.advance(game, world, local, dt_s=0.2)

    estimated_distance = torch.linalg.vector_norm(tracker.mean[0, Team.RED, target, :2] - report_xy)
    assert estimated_distance < before_distance
    assert tracker.source[0, Team.RED, target] == EntitySource.RADAR_ESTIMATE
    assert tracker.position_age_s[0, Team.RED, target] == 0
    assert tracker.state_age_s[0, Team.RED, target] == 0.2

    estimated = tracker.mean[0, Team.RED, target].clone()
    game.radar_report_age_s[0, Team.RED, target] = 0.2
    tracker.advance(game, world, local, dt_s=0.2)
    torch.testing.assert_close(
        tracker.mean[0, Team.RED, target, :2],
        estimated[:2] + estimated[2:] * 0.2,
    )
    assert tracker.source[0, Team.RED, target] == EntitySource.PREDICTION

    world.position_xy[0, target] = torch.tensor([7.0, 4.0])
    tracker.mean[0, Team.RED, target, 2:] = 0
    game.radar_p[0, Team.RED, target] = 100
    game.radar_truth_visible[0, Team.RED, target] = True
    before_confirm_distance = torch.linalg.vector_norm(
        tracker.mean[0, Team.RED, target, :2] - world.position_xy[0, target]
    )
    tracker.advance(game, world, local, dt_s=0.2)

    assert (
        torch.linalg.vector_norm(
            tracker.mean[0, Team.RED, target, :2] - world.position_xy[0, target]
        )
        < before_confirm_distance
    )
    assert tracker.source[0, Team.RED, target] == EntitySource.RADAR_CONFIRMED
    assert tracker.position_age_s[0, Team.RED, target] == 0
    assert tracker.state_age_s[0, Team.RED, target] == 0.6
    assert torch.all(torch.diagonal(tracker.covariance[0, Team.RED, target])[:2] <= 0.10**2)

    for _ in range(12):
        world.position_xy[0, target] += torch.tensor([0.2, -0.1])
        tracker.advance(game, world, local, dt_s=0.2)

    torch.testing.assert_close(
        tracker.mean[0, Team.RED, target, 2:],
        torch.tensor([1.0, -0.5]),
        atol=0.15,
        rtol=0,
    )


def test_partial_reset_only_resets_selected_belief_environment() -> None:
    env = TorchRMArena(TorchEnvConfig(num_envs=2, seed=23, observation_mode="belief"))
    tracker = env.belief_tracker
    assert tracker is not None
    tracker.mean[0].add_(1.25)
    tracker.mean[1].sub_(2.5)
    tracker.covariance[0].add_(torch.eye(4) * 0.5)
    tracker.covariance[1].add_(torch.eye(4) * 0.75)
    tracker.target_state[0].add_(0.1)
    tracker.target_state[1].sub_(0.2)
    tracker.valid.fill_(True)
    tracker.position_age_s[0].fill_(3.0)
    tracker.position_age_s[1].fill_(5.0)
    tracker.state_age_s[0].fill_(4.0)
    tracker.state_age_s[1].fill_(6.0)
    tracker.source.fill_(EntitySource.PREDICTION)
    untouched = {name: getattr(tracker, name)[1].clone() for name in TRACKER_FIELDS}
    fresh = TorchRMArena(TorchEnvConfig(num_envs=1, seed=23, observation_mode="belief"))
    fresh_tracker = fresh.belief_tracker
    assert fresh_tracker is not None

    env.reset(torch.tensor([0]))

    for name in TRACKER_FIELDS:
        torch.testing.assert_close(
            getattr(tracker, name)[0],
            getattr(fresh_tracker, name)[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            getattr(tracker, name)[1],
            untouched[name],
            rtol=0,
            atol=0,
        )
