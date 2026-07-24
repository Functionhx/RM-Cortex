"""Isaac Lab runtime implementation.

This module is imported only when ``isaaclab`` is available. The direct
environment follows Isaac Lab's current DirectMARLEnv hooks while delegating
all game semantics to ``rm_referee``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
import math

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
from isaaclab.envs import (  # type: ignore[import-not-found]
    DirectMARLEnv,
    DirectMARLEnvCfg,
    ViewerCfg,
)
from isaaclab.markers import (  # type: ignore[import-not-found]
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import-not-found]
from isaaclab.sim import SimulationCfg  # type: ignore[import-not-found]
from isaaclab.sim.spawners.from_files import (  # type: ignore[import-not-found]
    GroundPlaneCfg,
    spawn_ground_plane,
)
from isaaclab.terrains.utils import create_prim_from_mesh  # type: ignore[import-not-found]
from isaaclab.utils import configclass  # type: ignore[import-not-found]

from rm_isaac.adapter import IsaacGeometryFrame, IsaacRuleAdapter
from rm_referee import constants
from rm_referee.events import RefereeEvents
from rm_referee.random_tape import RandomTape
from rm_referee.referee import Referee
from rm_referee.schema import Role, Team, slot
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.arena import ArenaGeometry, TerrainPrimitive
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
UNIT_HALF_HEIGHTS = (
    0.20,
    0.21,
    0.19,
    0.19,
    0.10,
    0.29,
    0.59,
    0.94,
) * constants.TEAM_COUNT
TERRAIN_COLORS = {
    "central": (0.24, 0.29, 0.32),
    "assembly": (0.30, 0.34, 0.36),
    "trapezoid": (0.22, 0.27, 0.30),
    "ramp": (0.34, 0.38, 0.40),
    "road": (0.19, 0.24, 0.27),
    "fly_ramp": (0.38, 0.40, 0.40),
    "rough": (0.16, 0.21, 0.24),
    "fortress": (0.32, 0.35, 0.36),
    "tunnel": (0.04, 0.07, 0.09),
}


def _quaternion_from_euler(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return a scalar-first quaternion from XYZ Euler angles."""

    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _terrain_color(primitive: TerrainPrimitive) -> tuple[float, float, float]:
    color = TERRAIN_COLORS[primitive.category]
    if primitive.team == Team.RED:
        return (min(color[0] + 0.12, 1.0), color[1], color[2])
    if primitive.team == Team.BLUE:
        return (color[0], color[1], min(color[2] + 0.16, 1.0))
    return color


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    """Return the signed area of a simple 2D polygon."""

    return 0.5 * sum(
        x_0 * y_1 - x_1 * y_0
        for (x_0, y_0), (x_1, y_1) in zip(points, points[1:] + points[:1], strict=True)
    )


def _point_in_triangle(
    point: tuple[float, float],
    triangle: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    """Return whether a point lies in or on a counter-clockwise triangle."""

    signs = []
    for start, end in zip(triangle, triangle[1:] + triangle[:1], strict=True):
        signs.append(
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )
    return min(signs) >= -1.0e-9


def _triangulate_polygon(
    points: tuple[tuple[float, float], ...],
) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon with deterministic ear clipping."""

    if len(points) < 3:
        raise ValueError("terrain footprint must have at least three points")
    remaining = list(range(len(points)))
    if _polygon_area(points) < 0:
        remaining.reverse()
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        clipped = False
        for cursor, current in enumerate(remaining):
            previous = remaining[cursor - 1]
            following = remaining[(cursor + 1) % len(remaining)]
            a = points[previous]
            b = points[current]
            c = points[following]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross <= 1.0e-9:
                continue
            triangle = (a, b, c)
            if any(
                _point_in_triangle(points[index], triangle)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[cursor]
            clipped = True
            break
        if not clipped:
            raise ValueError("terrain footprint is not a simple polygon")
    triangles.append(tuple(remaining))
    return triangles


def _polygon_prism_mesh(
    primitive: TerrainPrimitive,
    height_m: float,
) -> trimesh.Trimesh:
    """Build an extruded mesh from the same footprint used by Torch geometry."""

    points = primitive.footprint_xy
    triangles = _triangulate_polygon(points)
    if _polygon_area(points) < 0:
        ordered = list(reversed(range(len(points))))
    else:
        ordered = list(range(len(points)))
    center_x, center_y = primitive.center_xy
    bottom = [(x - center_x, y - center_y, 0.0) for x, y in points]
    top = [(x - center_x, y - center_y, height_m) for x, y in points]
    vertex_count = len(points)
    faces: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        faces.append((c, b, a))
        faces.append((vertex_count + a, vertex_count + b, vertex_count + c))
    for cursor, current in enumerate(ordered):
        following = ordered[(cursor + 1) % len(ordered)]
        faces.append((current, following, vertex_count + following))
        faces.append((current, vertex_count + following, vertex_count + current))
    return trimesh.Trimesh(
        vertices=np.asarray(bottom + top, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _unit_marker(role: int, team: int) -> object:
    if role == Role.BASE:
        size = (1.88, 1.61, 1.18)
    elif role == Role.OUTPOST:
        color = (0.10, 0.24, 0.58) if team == Team.BLUE else (0.58, 0.10, 0.12)
        return sim_utils.CylinderCfg(
            radius=0.375,
            height=1.88,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.35,
            ),
        )
    elif role == Role.AERIAL:
        color = (0.12, 0.65, 1.0) if team == Team.BLUE else (1.0, 0.24, 0.16)
        return sim_utils.CylinderCfg(
            radius=0.32,
            height=0.20,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=color,
            ),
        )
    elif role == Role.SENTRY:
        size = (0.72, 0.62, 0.58)
    else:
        size = (0.60, 0.50, 0.40)
    color = (0.12, 0.25, 0.90) if team == Team.BLUE else (0.90, 0.12, 0.12)
    return sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            metallic=0.25,
            roughness=0.45,
        ),
    )


def _unit_marker_cfg() -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path="/Visuals/RMCortexUnits",
        markers={
            name: _unit_marker(
                index % constants.ROLES_PER_TEAM,
                index // constants.ROLES_PER_TEAM,
            )
            for index, name in enumerate(UNIT_NAMES)
        },
    )


@configclass
class RMCortexDirectMARLEnvCfg(DirectMARLEnvCfg):
    """Isaac Lab scene and PettingZoo-style multi-agent spaces."""

    seed: int = 0
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
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, -18.0, 22.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
    )
    visualize_units: bool = True
    unit_marker_cfg: VisualizationMarkersCfg = _unit_marker_cfg()


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
                boundary_margin_m=0.40,
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
        self._generator.manual_seed(0 if cfg.seed is None else cfg.seed)
        self.unit_markers = (
            VisualizationMarkers(cfg.unit_marker_cfg) if cfg.visualize_units else None
        )
        self._sync_scene()

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        arena = ArenaGeometry()
        crown_angle = math.radians(arena.config.field_crown_slope_deg)
        crown_height = arena.config.field_width_m / 2 * math.tan(crown_angle)
        half_width = arena.config.field_width_m / 2
        floor_length = math.hypot(half_width, crown_height)
        for side, y_center, roll in (
            ("North", half_width / 2, -crown_angle),
            ("South", -half_width / 2, crown_angle),
        ):
            floor_cfg = sim_utils.CuboidCfg(
                size=(arena.config.field_length_m, floor_length, 0.04),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.055, 0.075, 0.085),
                    roughness=0.85,
                ),
            )
            floor_cfg.func(
                f"/World/envs/env_0/ArenaFloor{side}",
                floor_cfg,
                translation=(0.0, y_center, crown_height / 2 - 0.02),
                orientation=_quaternion_from_euler(roll=roll),
            )
            stripe_cfg = sim_utils.CuboidCfg(
                size=(0.06, floor_length, 0.012),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.75, 0.78, 0.82),
                ),
            )
            stripe_cfg.func(
                f"/World/envs/env_0/CenterLine{side}",
                stripe_cfg,
                translation=(0.0, y_center, crown_height / 2 + 0.012),
                orientation=_quaternion_from_euler(roll=roll),
            )

        for primitive in arena.config.terrain:
            yaw = math.radians(primitive.yaw_deg)
            crown = max(
                arena.config.field_width_m / 2 - abs(primitive.center_xy[1]),
                0.0,
            ) * math.tan(
                crown_angle,
            )
            if primitive.category == "tunnel":
                roof_cfg = sim_utils.CuboidCfg(
                    size=(*primitive.size_xy, 0.08),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_terrain_color(primitive),
                        metallic=0.25,
                    ),
                )
                roof_cfg.func(
                    f"/World/envs/env_0/Terrain_{primitive.name}",
                    roof_cfg,
                    translation=(*primitive.center_xy, crown + 0.48),
                    orientation=_quaternion_from_euler(yaw=yaw),
                )
                continue

            elevation_delta = primitive.elevation_end_m - primitive.elevation_start_m
            if primitive.footprint_xy and abs(elevation_delta) < 1.0e-6:
                if primitive.category == "assembly":
                    height = 0.012
                    base_z = crown + primitive.elevation_start_m
                else:
                    height = max(primitive.elevation_start_m, 0.025)
                    base_z = crown
                create_prim_from_mesh(
                    f"/World/envs/env_0/Terrain_{primitive.name}",
                    _polygon_prism_mesh(primitive, height),
                    translation=(*primitive.center_xy, base_z),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_terrain_color(primitive),
                        roughness=0.72,
                    ),
                )
                continue

            if abs(elevation_delta) < 1.0e-6:
                height = max(primitive.elevation_start_m, 0.025)
                size = (*primitive.size_xy, height)
                translation_z = crown + height / 2
                pitch = 0.0
            else:
                height = 0.055
                slope_length = math.hypot(primitive.size_xy[0], elevation_delta)
                size = (slope_length, primitive.size_xy[1], height)
                translation_z = (
                    crown
                    + (primitive.elevation_start_m + primitive.elevation_end_m) / 2
                    - height / 2
                )
                pitch = -math.atan2(elevation_delta, primitive.size_xy[0])
            terrain_cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=_terrain_color(primitive),
                    roughness=0.72,
                ),
            )
            terrain_cfg.func(
                f"/World/envs/env_0/Terrain_{primitive.name}",
                terrain_cfg,
                translation=(*primitive.center_xy, translation_z),
                orientation=_quaternion_from_euler(pitch=pitch, yaw=yaw),
            )

        for index, (x_min, x_max, y_min, y_max) in enumerate(arena.config.obstacles):
            center_xy = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            surface_z = float(
                arena.terrain_height(torch.tensor(center_xy, dtype=torch.float32)).item()
            )
            block_cfg = sim_utils.CuboidCfg(
                size=(x_max - x_min, y_max - y_min, 0.55),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.24, 0.28, 0.31),
                    metallic=0.15,
                ),
            )
            block_cfg.func(
                f"/World/envs/env_0/Obstacle_{index}",
                block_cfg,
                translation=(
                    *center_xy,
                    surface_z + 0.275,
                ),
            )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2600.0,
            color=(0.82, 0.86, 0.92),
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
        del env_ids
        if self.unit_markers is None:
            return
        half_height = torch.tensor(
            UNIT_HALF_HEIGHTS,
            device=self.device,
            dtype=self.game.dtype,
        ).view(1, constants.UNIT_COUNT)
        translations = torch.zeros(
            (self.num_envs, constants.UNIT_COUNT, 3),
            device=self.device,
            dtype=self.game.dtype,
        )
        vertical_scale = torch.where(
            self.game.alive,
            torch.ones_like(self.world.position_z),
            torch.full_like(self.world.position_z, 0.25),
        )
        translations[..., :2] = self.world.position_xy + self.scene.env_origins[:, None, :2]
        translations[..., 2] = self.world.position_z + half_height * vertical_scale
        orientations = torch.zeros(
            (self.num_envs, constants.UNIT_COUNT, 4),
            device=self.device,
            dtype=self.game.dtype,
        )
        orientations[..., 0] = torch.cos(self.world.yaw / 2)
        orientations[..., 3] = torch.sin(self.world.yaw / 2)
        scales = torch.ones_like(translations)
        scales[..., 2] = vertical_scale
        marker_indices = (
            torch.arange(constants.UNIT_COUNT, device=self.device)
            .view(1, -1)
            .expand(self.num_envs, -1)
        )
        self.unit_markers.visualize(
            translations=translations.reshape(-1, 3),
            orientations=orientations.reshape(-1, 4),
            scales=scales.reshape(-1, 3),
            marker_indices=marker_indices.reshape(-1),
        )

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
