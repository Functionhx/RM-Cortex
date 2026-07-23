"""Isaac Lab runtime implementation.

This module is imported only when ``isaaclab`` is available. The direct
environment follows Isaac Lab's current DirectMARLEnv hooks while delegating
all game semantics to ``rm_referee``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
import math

import torch

import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore[import-not-found]
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg  # type: ignore[import-not-found]
from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import-not-found]
from isaaclab.sim import SimulationCfg  # type: ignore[import-not-found]
from isaaclab.sim.spawners.from_files import (  # type: ignore[import-not-found]
    GroundPlaneCfg,
    spawn_ground_plane,
)
from isaaclab.utils import configclass  # type: ignore[import-not-found]

from rm_isaac.adapter import IsaacGeometryFrame, IsaacRuleAdapter
from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.random_tape import RandomTape
from rm_referee.referee import Referee
from rm_referee.schema import Role, Team, slot
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.arena import ArenaGeometry
from rm_world.backend import TorchRuleBackend
from rm_world.kinematics import KinematicConfig, KinematicState, KinematicWorld
from rm_world.observations import ObservationBuilder
from rm_world.rewards import RewardBuilder


UNIT_NAMES = (
    "red_hero",
    "red_engineer",
    "red_infantry_3",
    "red_infantry_4",
    "red_aerial",
    "red_sentry",
    "red_base",
    "red_outpost",
    "blue_hero",
    "blue_engineer",
    "blue_infantry_3",
    "blue_infantry_4",
    "blue_aerial",
    "blue_sentry",
    "blue_base",
    "blue_outpost",
)
ROBOT_AGENT_SLOTS = {
    name: index
    for index, name in enumerate(UNIT_NAMES)
    if index % constants.ROLES_PER_TEAM < Role.BASE
}
POSSIBLE_AGENTS = (
    "red_hero",
    "red_engineer",
    "red_infantry_3",
    "red_infantry_4",
    "red_aerial",
    "red_sentry",
    "red_dart",
    "red_radar",
    "blue_hero",
    "blue_engineer",
    "blue_infantry_3",
    "blue_infantry_4",
    "blue_aerial",
    "blue_sentry",
    "blue_dart",
    "blue_radar",
)
AGENT_OBSERVATION_SLOT = {
    **ROBOT_AGENT_SLOTS,
    "red_dart": slot(Team.RED, Role.BASE),
    "red_radar": slot(Team.RED, Role.OUTPOST),
    "blue_dart": slot(Team.BLUE, Role.BASE),
    "blue_radar": slot(Team.BLUE, Role.OUTPOST),
}
ACTION_DIM = 14


def _unit_cfg(name: str, role: int, team: int) -> RigidObjectCfg:
    if role == Role.BASE:
        size = (1.88, 1.61, 1.18)
    elif role == Role.OUTPOST:
        size = (0.75, 0.75, 1.88)
    elif role == Role.AERIAL:
        size = (0.55, 0.55, 0.20)
    else:
        size = (0.60, 0.50, 0.40)
    color = (0.12, 0.25, 0.90) if team == Team.BLUE else (0.90, 0.12, 0.12)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, size[2] / 2)),
    )


@configclass
class RMCortexDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Isaac Lab scene and PettingZoo-style multi-agent spaces."""

    decimation = 12
    episode_length_s = constants.MATCH_DURATION_S
    possible_agents = list(POSSIBLE_AGENTS)
    action_spaces = {agent: ACTION_DIM for agent in POSSIBLE_AGENTS}
    observation_spaces = {agent: 34 for agent in POSSIBLE_AGENTS}
    state_space = 117
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=32.0,
        replicate_physics=True,
    )
    unit_cfgs: dict[str, RigidObjectCfg] = {
        name: _unit_cfg(
            name,
            index % constants.ROLES_PER_TEAM,
            index // constants.ROLES_PER_TEAM,
        )
        for index, name in enumerate(UNIT_NAMES)
    }


class RMCortexDirectMARLEnv(DirectMARLEnv):
    """Kinematic Isaac scene backed by the authoritative Torch referee."""

    cfg: RMCortexDirectMARLEnvCfg

    def __init__(
        self,
        cfg: RMCortexDirectMARLEnvCfg,
        render_mode: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self.arena = ArenaGeometry()
        self.referee = Referee(validate_inputs=False, validate_state=False)
        self.backend = TorchRuleBackend(arena=self.arena)
        self.rule_adapter = IsaacRuleAdapter()
        self.world_model = KinematicWorld(
            config=KinematicConfig(
                boundary_margin_m=0.35,
                enable_static_collisions=True,
                enable_unit_collisions=True,
            ),
            arena=self.arena,
        )
        self.observation_builder = ObservationBuilder(arena=self.arena)
        self.reward_builder = RewardBuilder()
        self.game = GameState.create(self.num_envs, device=self.device)
        self.world = KinematicState.spawn(self.game, self.arena)
        self.world_actions = WorldActions.zeros(self.game)
        self._reward_buffer = torch.zeros_like(self.game.hp)
        self._last_events = RefereeEvents.empty(self.game)
        self._physics_substep = 0
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(cfg.seed)
        self._sync_scene()

    def _setup_scene(self) -> None:
        self.units: dict[str, RigidObject] = {
            name: RigidObject(self.cfg.unit_cfgs[name]) for name in UNIT_NAMES
        }
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for name, unit in self.units.items():
            self.scene.rigid_objects[name] = unit
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.world_actions = WorldActions.zeros(self.game)
        self._reward_buffer.zero_()
        self._last_events = RefereeEvents.empty(self.game)
        self._decode_actions(actions)

    def _decode_actions(self, actions: dict[str, torch.Tensor]) -> None:
        for agent, unit_slot in ROBOT_AGENT_SLOTS.items():
            command = torch.clamp(actions[agent], min=-1.0, max=1.0)
            self.world_actions.body_velocity_xy[:, unit_slot] = command[:, :2] * 3.0
            self.world_actions.yaw_rate[:, unit_slot] = command[:, 2] * (2.0 * math.pi)
            self.world_actions.vertical_velocity[:, unit_slot] = command[:, 3] * 2.0
            self.world_actions.target[:, unit_slot] = torch.round((command[:, 4] + 1.0) * 4.0).to(
                torch.long
            )
            self.world_actions.fire[:, unit_slot] = command[:, 5] > 0
            team = unit_slot // constants.ROLES_PER_TEAM
            role = unit_slot % constants.ROLES_PER_TEAM
            if role == Role.HERO:
                self.world_actions.hero_deploy[:, team] = command[:, 6] > 0
            elif role == Role.ENGINEER:
                self.world_actions.rebuild_outpost[:, unit_slot] = command[:, 6] > 0
                self.world_actions.tech_complete_level[:, team] = torch.round(
                    (command[:, 7] + 1.0) * 2.0
                ).to(torch.int8)
                self.world_actions.rune_trigger[:, team] = torch.round(
                    torch.clamp(command[:, 13], min=0) * 2.0
                ).to(torch.int8)
            elif role == Role.AERIAL:
                self.world_actions.aerial_support[:, team] = command[:, 6] > 0
            elif role == Role.SENTRY:
                self.world_actions.sentry_stance[:, team] = torch.round((command[:, 6] + 1.0)).to(
                    torch.int8
                )
                self.world_actions.sentry_claim_ammo[:, team] = command[:, 7] > 0

            self.world_actions.remote_heal[:, unit_slot] = command[:, 8] > 0
            self.world_actions.immediate_respawn[:, unit_slot] = command[:, 9] > 0
            purchase_request = command[:, 10] > 0
            empty_team_request = self.world_actions.purchase_unit[:, team] < 0
            choose_purchase = purchase_request & empty_team_request
            self.world_actions.purchase_unit[:, team] = torch.where(
                choose_purchase,
                torch.full_like(
                    self.world_actions.purchase_unit[:, team],
                    unit_slot,
                ),
                self.world_actions.purchase_unit[:, team],
            )
            self.world_actions.purchase_kind[:, team] = torch.where(
                choose_purchase,
                torch.where(
                    command[:, 11] > 0,
                    torch.full_like(self.world_actions.purchase_kind[:, team], 2),
                    torch.full_like(self.world_actions.purchase_kind[:, team], 1),
                ),
                self.world_actions.purchase_kind[:, team],
            )
            self.world_actions.purchase_weapon[:, team] = torch.where(
                choose_purchase,
                (command[:, 12] > 0).to(torch.long),
                self.world_actions.purchase_weapon[:, team],
            )

        for team, prefix in ((Team.RED, "red"), (Team.BLUE, "blue")):
            dart = torch.clamp(actions[f"{prefix}_dart"], min=-1.0, max=1.0)
            self.world_actions.dart_open_gate[:, team] = dart[:, 0] > 0
            self.world_actions.dart_close_gate[:, team] = dart[:, 1] > 0
            self.world_actions.dart_target[:, team] = torch.where(
                dart[:, 2] < -0.8,
                torch.full(
                    (self.num_envs,),
                    -1,
                    device=self.device,
                    dtype=torch.int8,
                ),
                torch.round((dart[:, 2] + 1.0) * 2.0).to(torch.int8),
            )

            radar = torch.clamp(actions[f"{prefix}_radar"], min=-1.0, max=1.0)
            self.world_actions.radar_target[:, team] = torch.round((radar[:, 0] + 1.0) * 7.5).to(
                torch.long
            )
            target = self.world_actions.radar_target[:, team]
            actual = torch.gather(
                self.world.position_xy,
                1,
                target[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            self.world_actions.radar_report_xy[:, team] = actual + radar[:, 1:3] * 2.0
            self.world_actions.radar_illuminate[:, team] = radar[:, 3] > 0
            self.world_actions.radar_double[:, team] = radar[:, 4] > 0
            self.world_actions.radar_key_solved[:, team] = radar[:, 5] > 0

    def _apply_action(self) -> None:
        commands = self.backend.kinematic_commands(self.game, self.world_actions)
        self.world = self.world_model.step(
            self.world,
            self.game,
            commands,
            dt=self.physics_dt,
        )
        self._sync_scene()
        self._physics_substep += 1
        referee_stride = round(constants.REFEREE_DT_S / self.physics_dt)
        if self._physics_substep % referee_stride != 0:
            return

        values = torch.rand(
            (
                self.num_envs,
                constants.UNIT_COUNT * 2 + constants.TEAM_COUNT,
            ),
            generator=self._generator,
            device=self.device,
            dtype=self.game.dtype,
        )
        normalized = self.backend.rule_inputs(
            self.world,
            self.game,
            self.world_actions,
            RandomTape(values),
            constants.REFEREE_DT_S,
        )
        frame = IsaacGeometryFrame(
            shots_fired=normalized.shots_fired,
            hit_source=normalized.hits.source,
            hit_time_offset_s=normalized.hits.time_offset_s,
            hit_critical=normalized.hits.critical,
            chassis_power_w=normalized.chassis_power_w,
        )
        inputs = self.rule_adapter.normalize(self.game, frame, base=normalized)
        self.game, self._last_events = self.referee.step(self.game, inputs)
        self._reward_buffer.add_(self.reward_builder.build(self.game, self._last_events))

    def _sync_scene(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        half_height = torch.tensor(
            [self.cfg.unit_cfgs[name].spawn.size[2] / 2 for name in UNIT_NAMES],
            device=self.device,
            dtype=self.game.dtype,
        )
        for unit_slot, name in enumerate(UNIT_NAMES):
            yaw = self.world.yaw[env_ids, unit_slot]
            root_pose = torch.zeros(
                (env_ids.shape[0], 7),
                device=self.device,
                dtype=self.game.dtype,
            )
            root_pose[:, :2] = (
                self.world.position_xy[env_ids, unit_slot] + self.scene.env_origins[env_ids, :2]
            )
            root_pose[:, 2] = self.world.position_z[env_ids, unit_slot] + half_height[unit_slot]
            root_pose[:, 3] = torch.cos(yaw / 2)
            root_pose[:, 6] = torch.sin(yaw / 2)
            self.units[name].write_root_pose_to_sim(root_pose, env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observation = self.observation_builder.build(self.game, self.world)
        return {
            agent: observation.agents[:, AGENT_OBSERVATION_SLOT[agent]]
            for agent in self.cfg.possible_agents
        }

    def _get_states(self) -> torch.Tensor:
        return self.observation_builder.build(self.game, self.world).central

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        team_reward = self._reward_buffer.view(
            self.num_envs,
            constants.TEAM_COUNT,
            constants.ROLES_PER_TEAM,
        ).mean(dim=-1)
        rewards: dict[str, torch.Tensor] = {}
        for agent in self.cfg.possible_agents:
            if agent.endswith("_dart") or agent.endswith("_radar"):
                team = Team.RED if agent.startswith("red_") else Team.BLUE
                rewards[agent] = team_reward[:, team]
            else:
                rewards[agent] = self._reward_buffer[:, AGENT_OBSERVATION_SLOT[agent]]
        return rewards

    def _get_dones(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated = {agent: self.game.done.clone() for agent in self.cfg.possible_agents}
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        truncated = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "game"):
            return
        if env_ids is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        fresh_game = GameState.create(
            self.num_envs,
            device=self.device,
            dtype=self.game.dtype,
        )
        fresh_world = KinematicState.spawn(fresh_game, self.arena)
        for field in fields(GameState):
            getattr(self.game, field.name)[indices] = getattr(
                fresh_game,
                field.name,
            )[indices]
        for field in fields(KinematicState):
            getattr(self.world, field.name)[indices] = getattr(
                fresh_world,
                field.name,
            )[indices]
        self._reward_buffer[indices] = 0
        self._sync_scene(indices)
