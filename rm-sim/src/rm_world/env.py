"""End-to-end vectorized Torch environment."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.random_tape import RandomTape
from rm_referee.referee import Referee
from rm_referee.schema import Role, unit_roles
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.arena import ArenaGeometry
from rm_world.backend import TorchRuleBackend
from rm_world.kinematics import KinematicConfig, KinematicState, KinematicWorld
from rm_world.observations import ObservationBuilder, WorldObservation
from rm_world.rewards import RewardBuilder


@dataclass(frozen=True)
class TorchEnvConfig:
    num_envs: int = 1
    device: str = "cpu"
    seed: int = 0
    referee_dt_s: float = constants.REFEREE_DT_S
    physics_dt_s: float = 1.0 / 60.0
    policy_dt_s: float = 0.2
    validate_referee: bool = True

    def validate(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        referee_ticks = self.policy_dt_s / self.referee_dt_s
        physics_steps = self.referee_dt_s / self.physics_dt_s
        if abs(referee_ticks - round(referee_ticks)) > 1.0e-9:
            raise ValueError("policy_dt_s must be an integer multiple of referee_dt_s")
        if abs(physics_steps - round(physics_steps)) > 1.0e-9:
            raise ValueError("referee_dt_s must be an integer multiple of physics_dt_s")


@dataclass
class TorchEnvStep:
    observation: WorldObservation
    reward: Tensor
    terminated: Tensor
    events: RefereeEvents


def _merge_events(total: RefereeEvents, current: RefereeEvents) -> None:
    sum_fields = (
        "shots_fired",
        "overfire",
        "damage_taken",
        "damage_dealt",
        "damage_by_source_target",
        "purchased_rounds",
        "sentry_claimed_rounds",
        "scheduled_coin",
        "xp_gained",
    )
    bool_fields = (
        "hit_accepted",
        "deaths",
        "respawns",
        "immediate_respawns",
        "remote_heals",
        "rune_activated",
        "dart_fired",
        "dart_hits",
        "match_ended",
    )
    for name in sum_fields:
        getattr(total, name).add_(getattr(current, name))
    for name in bool_fields:
        getattr(total, name).__ior__(getattr(current, name))
    total.kill_source.copy_(
        torch.where(current.kill_source >= 0, current.kill_source, total.kill_source)
    )
    total.tech_completed.copy_(torch.maximum(total.tech_completed, current.tech_completed))


class TorchRMArena:
    """Pure-Torch multi-agent environment with no Gym or Isaac dependency."""

    random_draws_per_referee_tick = constants.UNIT_COUNT * 2 + constants.TEAM_COUNT

    def __init__(
        self,
        config: TorchEnvConfig | None = None,
        *,
        referee: Referee | None = None,
        arena: ArenaGeometry | None = None,
    ) -> None:
        self.config = config or TorchEnvConfig()
        self.config.validate()
        self.arena = arena or ArenaGeometry()
        self.referee = referee or Referee(
            validate_inputs=self.config.validate_referee,
            validate_state=self.config.validate_referee,
        )
        self.world_model = KinematicWorld(
            config=KinematicConfig(
                boundary_margin_m=0.40,
                enable_static_collisions=True,
                enable_unit_collisions=True,
            ),
            arena=self.arena,
        )
        self.backend = TorchRuleBackend(arena=self.arena)
        self.observation_builder = ObservationBuilder(arena=self.arena)
        self.reward_builder = RewardBuilder()
        self.generator = torch.Generator(device=self.config.device)
        self.generator.manual_seed(self.config.seed)
        self.game = GameState.create(
            self.config.num_envs,
            device=self.config.device,
        )
        self.world = KinematicState.spawn(self.game, self.arena)

    @property
    def controllable_mask(self) -> Tensor:
        roles = unit_roles(self.game.device)
        return (roles != Role.BASE) & (roles != Role.OUTPOST)

    def observe(self) -> WorldObservation:
        return self.observation_builder.build(self.game, self.world)

    def reset(
        self,
        env_ids: Tensor | None = None,
        *,
        seed: int | None = None,
    ) -> WorldObservation:
        if seed is not None:
            self.generator.manual_seed(seed)
        if env_ids is None:
            env_ids = torch.arange(self.config.num_envs, device=self.game.device)
        env_ids = env_ids.to(device=self.game.device, dtype=torch.long)
        fresh_game = GameState.create(
            self.config.num_envs,
            device=self.game.device,
            dtype=self.game.dtype,
        )
        fresh_world = KinematicState.spawn(fresh_game, self.arena)
        for field in fields(GameState):
            current = getattr(self.game, field.name)
            current[env_ids] = getattr(fresh_game, field.name)[env_ids]
        for field in fields(KinematicState):
            current = getattr(self.world, field.name)
            current[env_ids] = getattr(fresh_world, field.name)[env_ids]
        return self.observe()

    def step(self, actions: WorldActions) -> TorchEnvStep:
        actions.validate(self.game)
        referee_ticks = round(self.config.policy_dt_s / self.config.referee_dt_s)
        physics_steps = round(self.config.referee_dt_s / self.config.physics_dt_s)
        total_events = RefereeEvents.empty(self.game)
        total_reward = torch.zeros_like(self.game.hp)
        commands = self.backend.kinematic_commands(self.game, actions)

        for _ in range(referee_ticks):
            terrain_crossed = torch.zeros_like(self.world.terrain_crossed)
            for _ in range(physics_steps):
                self.world = self.world_model.step(
                    self.world,
                    self.game,
                    commands,
                    dt=self.config.physics_dt_s,
                )
                terrain_crossed |= self.world.terrain_crossed
            self.world.terrain_crossed.copy_(terrain_crossed)

            values = torch.rand(
                (
                    self.config.num_envs,
                    self.random_draws_per_referee_tick,
                ),
                generator=self.generator,
                device=self.game.device,
                dtype=self.game.dtype,
            )
            inputs = self.backend.rule_inputs(
                self.world,
                self.game,
                actions,
                RandomTape(values),
                self.config.referee_dt_s,
            )
            self.game, events = self.referee.step(
                self.game,
                inputs,
                dt=self.config.referee_dt_s,
            )
            total_reward.add_(self.reward_builder.build(self.game, events))
            _merge_events(total_events, events)

        return TorchEnvStep(
            observation=self.observe(),
            reward=total_reward,
            terminated=self.game.done.clone(),
            events=total_events,
        )
