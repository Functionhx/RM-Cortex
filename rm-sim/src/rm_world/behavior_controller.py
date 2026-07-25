"""Hierarchical behavior-tree baseline executed through cached A* paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as functional

from rm_referee import constants
from rm_referee.schema import Role, Team, slot
from rm_referee.state import GameState
from rm_world.actions import WorldActions
from rm_world.arena import ArenaGeometry
from rm_world.behavior_tree import (
    NO_MISSION,
    BehaviorTreeTrace,
    HierarchicalBehaviorTrees,
    TeamBlackboard,
)
from rm_world.kinematics import KinematicState
from rm_world.pathfinding import GridAStarPlanner, PathResult
from rm_world.scripted import (
    TACTICAL_RED_ROUTES,
    TacticalMission,
    TacticalScriptedOpponent,
)


_GROUND_MOBILE_ROLES = (
    Role.HERO,
    Role.ENGINEER,
    Role.INFANTRY_3,
    Role.INFANTRY_4,
    Role.SENTRY,
)

# The old infantry-3 route ended only 0.35 m from a retaining wall, inside the
# 0.40 m robot footprint. This interaction point keeps the same tactical side
# while remaining planable by the footprint-aware grid.
_MISSION_GOAL_OVERRIDES = {
    TacticalMission.INFANTRY_3_OUTPOST: (3.20, 2.65),
    TacticalMission.INFANTRY_3_HOLD: (3.20, 2.65),
}

# Policy-side [SIM] cadence. It prevents a missed dart from immediately
# emptying the magazine while the referee's detection block remains inactive.
_DART_COMMAND_INTERVAL_S = constants.DART_DETECTION_BLOCK_S


@dataclass(frozen=True)
class BehaviorNavigationDiagnostics:
    """Cumulative A* query outcomes for one controller lifecycle."""

    plan_requests: int
    plan_successes: int
    plan_failures: int
    unique_cached_paths: int


class BehaviorTreeOpponent(TacticalScriptedOpponent):
    """Oracle/full-action behavior-tree engineering baseline.

    The global tree publishes team intent, eight independent role trees choose
    missions, and ground missions are executed through cached 50 mm A* paths.
    Combat, local separation, and role-specific rule actions reuse the existing
    deterministic executor so referee semantics remain centralized.

    This controller reads complete ``GameState``/``KinematicState`` and can use
    actions absent from the current MAPPO schema. It is therefore an oracle
    engineering baseline, not a fair belief-policy comparison.
    """

    def __init__(
        self,
        *,
        arena: ArenaGeometry | None = None,
        planner: GridAStarPlanner | None = None,
        path_clearance_m: float = 0.0,
    ) -> None:
        if planner is not None:
            if arena is not None and planner.arena is not arena:
                raise ValueError("planner and controller must share one ArenaGeometry")
            shared_arena = planner.arena
        else:
            shared_arena = arena or ArenaGeometry()
        super().__init__(arena=shared_arena)
        self.forest = HierarchicalBehaviorTrees()
        self.planner = planner or GridAStarPlanner(
            shared_arena,
            resolution_m=0.05,
            clearance_m=path_clearance_m,
        )
        self._decision_team: Team | int | None = None
        self._decision_world: KinematicState | None = None
        self._last_blackboard: TeamBlackboard | None = None
        self._last_trace: BehaviorTreeTrace | None = None
        self._planned_missions: Tensor | None = None
        self._planned_goals: Tensor | None = None
        self._planned_paths: dict[int, PathResult] = {}
        self._path_elapsed_s: Tensor | None = None
        self._plan_requests = 0
        self._plan_successes = 0
        self._plan_failures = 0
        self._leaf_hits: Tensor | None = None
        self._next_dart_command_s: Tensor | None = None
        self._dart_elapsed_s: Tensor | None = None

    @property
    def last_blackboard(self) -> TeamBlackboard | None:
        return self._last_blackboard

    @property
    def last_trace(self) -> BehaviorTreeTrace | None:
        return self._last_trace

    @property
    def navigation_diagnostics(self) -> BehaviorNavigationDiagnostics:
        return BehaviorNavigationDiagnostics(
            plan_requests=self._plan_requests,
            plan_successes=self._plan_successes,
            plan_failures=self._plan_failures,
            unique_cached_paths=self.planner.cached_path_count,
        )

    @property
    def leaf_hit_counts(self) -> dict[str, tuple[int, int]]:
        """Return cumulative selected-leaf hits in red/blue team order."""

        if self._leaf_hits is None:
            return {}
        counts = self._leaf_hits.detach().cpu().tolist()
        return {
            "/".join(path): (int(counts[Team.RED][leaf]), int(counts[Team.BLUE][leaf]))
            for leaf, path in enumerate(self.forest.leaf_paths)
            if counts[Team.RED][leaf] or counts[Team.BLUE][leaf]
        }

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset all controller state, or invalidate selected environment paths."""

        if env_ids is None:
            super().reset()
            self._last_blackboard = None
            self._last_trace = None
            self._decision_world = None
            self._planned_missions = None
            self._planned_goals = None
            self._planned_paths.clear()
            self._path_elapsed_s = None
            self._plan_requests = 0
            self._plan_successes = 0
            self._plan_failures = 0
            self._leaf_hits = None
            self._next_dart_command_s = None
            self._dart_elapsed_s = None
            return
        if env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        if self._planned_missions is not None:
            ids = env_ids.to(self._planned_missions.device)
            self._planned_missions[ids] = NO_MISSION - 1
            assert self._planned_goals is not None
            self._planned_goals[ids] = torch.nan
            if self._path_elapsed_s is not None:
                self._path_elapsed_s[ids] = -1.0
        if self._next_dart_command_s is not None:
            ids = env_ids.to(self._next_dart_command_s.device)
            self._next_dart_command_s[ids] = 0.0
            assert self._dart_elapsed_s is not None
            self._dart_elapsed_s[ids] = -1.0

    def _dart_fire_ready(self, game: GameState) -> Tensor:
        expected = (game.num_envs, constants.TEAM_COUNT)
        if (
            self._next_dart_command_s is None
            or self._next_dart_command_s.shape != expected
            or self._next_dart_command_s.device != game.device
        ):
            self._next_dart_command_s = torch.zeros(
                expected,
                device=game.device,
                dtype=game.dtype,
            )
            self._dart_elapsed_s = torch.full_like(game.elapsed_s, -1.0)
        assert self._dart_elapsed_s is not None
        reset = game.elapsed_s + 1.0e-6 < self._dart_elapsed_s
        self._next_dart_command_s[reset] = 0.0
        return game.elapsed_s[:, None] >= self._next_dart_command_s

    def _missions(self, game: GameState) -> Tensor:
        dart_fire_ready = self._dart_fire_ready(game)
        blackboard, trace = self.forest.decide(
            game,
            team=self._decision_team,
            position_xy=(
                None if self._decision_world is None else self._decision_world.position_xy
            ),
            dart_fire_ready=dart_fire_ready,
        )
        assert self._next_dart_command_s is not None
        assert self._dart_elapsed_s is not None
        dart_commanded = blackboard.dart_target >= 0
        self._next_dart_command_s.copy_(
            torch.where(
                dart_commanded,
                game.elapsed_s[:, None] + _DART_COMMAND_INTERVAL_S,
                self._next_dart_command_s,
            )
        )
        self._dart_elapsed_s.copy_(game.elapsed_s)
        self._last_blackboard = blackboard
        self._last_trace = trace
        selected = trace.selected_leaf.to(torch.long)
        valid = selected >= 0
        leaf_hits = functional.one_hot(
            torch.clamp(selected, min=0),
            num_classes=len(self.forest.leaf_paths),
        )
        leaf_hits *= valid[..., None]
        tick_hits = leaf_hits.sum(dim=(0, 2))
        if self._leaf_hits is None or self._leaf_hits.device != game.device:
            self._leaf_hits = torch.zeros_like(tick_hits)
        self._leaf_hits += tick_hits
        return blackboard.mission

    def _mission_goals(
        self,
        world: KinematicState,
        missions: Tensor,
    ) -> Tensor:
        goals = world.position_xy.clone()
        for mission, route in TACTICAL_RED_ROUTES.items():
            red_goal = _MISSION_GOAL_OVERRIDES.get(mission, route[-1])
            goal = world.position_xy.new_tensor(red_goal)
            for team in (Team.RED, Team.BLUE):
                team_goal = goal if team == Team.RED else -goal
                team_slots = slice(
                    int(team) * constants.ROLES_PER_TEAM,
                    (int(team) + 1) * constants.ROLES_PER_TEAM,
                )
                active = missions[:, team_slots] == int(mission)
                goals[:, team_slots].copy_(
                    torch.where(
                        active[:, :, None],
                        team_goal[None, None, :],
                        goals[:, team_slots],
                    )
                )
        for team in (Team.RED, Team.BLUE):
            aerial = slot(team, Role.AERIAL)
            goals[:, aerial] = self.arena.project_to_aerial_flight_area(
                goals[:, aerial],
                int(team),
                margin_m=0.05,
            )
        return torch.where(
            (missions == NO_MISSION)[:, :, None],
            world.position_xy,
            goals,
        )

    def _ensure_path_state(self, game: GameState, goals: Tensor) -> Tensor:
        shape = (game.num_envs, constants.UNIT_COUNT)
        reset = torch.ones(game.num_envs, device=game.device, dtype=torch.bool)
        if (
            self._planned_missions is None
            or self._planned_missions.shape != shape
            or self._planned_missions.device != game.device
        ):
            self._planned_missions = torch.full(
                shape,
                NO_MISSION - 1,
                device=game.device,
                dtype=torch.long,
            )
            self._planned_goals = torch.full_like(goals, torch.nan)
            self._path_elapsed_s = torch.full_like(game.elapsed_s, -1.0)
            self._planned_paths.clear()
        else:
            assert self._path_elapsed_s is not None
            reset = game.elapsed_s + 1.0e-6 < self._path_elapsed_s
        return reset

    def _route_targets(
        self,
        game: GameState,
        world: KinematicState,
        missions: Tensor,
    ) -> Tensor:
        goals = self._mission_goals(world, missions)
        reset_env = self._ensure_path_state(game, goals)
        assert self._planned_missions is not None
        assert self._planned_goals is not None
        assert self._path_elapsed_s is not None

        targets = goals.clone()
        for team in (Team.RED, Team.BLUE):
            for role in _GROUND_MOBILE_ROLES:
                unit = slot(team, role)
                active = missions[:, unit] != NO_MISSION
                mission_changed = missions[:, unit] != self._planned_missions[:, unit]
                goal_changed = ~torch.isclose(
                    goals[:, unit],
                    self._planned_goals[:, unit],
                    atol=self.planner.resolution_m / 2,
                    rtol=0.0,
                ).all(dim=-1)
                if not bool(active.any().item()):
                    targets[:, unit] = world.position_xy[:, unit]
                    self._planned_missions[:, unit].copy_(missions[:, unit])
                    self._planned_goals[:, unit].copy_(goals[:, unit])
                    continue
                needs_plan = unit not in self._planned_paths or bool(
                    (active & (mission_changed | goal_changed | reset_env)).any().item()
                )
                if needs_plan:
                    planned = self.planner.plan(
                        world.position_xy[:, unit],
                        goals[:, unit],
                    )
                    self._planned_paths[unit] = planned
                    self._plan_requests += game.num_envs
                    successes = int(planned.success.sum().item())
                    self._plan_successes += successes
                    self._plan_failures += game.num_envs - successes
                cached_result = self._planned_paths.get(unit)
                if cached_result is not None:
                    waypoint = self.planner.next_waypoint(
                        cached_result,
                        world.position_xy[:, unit],
                        lookahead_m=0.60,
                    )
                    targets[:, unit] = torch.where(
                        active[:, None],
                        waypoint,
                        world.position_xy[:, unit],
                    )
                self._planned_missions[:, unit].copy_(missions[:, unit])
                self._planned_goals[:, unit].copy_(goals[:, unit])

        self._path_elapsed_s.copy_(game.elapsed_s)
        self._current_missions = missions.clone()
        return targets

    def act(
        self,
        game: GameState,
        world: KinematicState,
        *,
        team: Team | int | None = None,
    ) -> WorldActions:
        """Run the global tree, role trees, A*, and deterministic executor."""

        self._decision_team = team
        self._decision_world = world
        try:
            actions = super().act(game, world, team=team)
            self._apply_structure_commands(actions, game, team)
            return actions
        finally:
            self._decision_team = None
            self._decision_world = None

    def _apply_structure_commands(
        self,
        actions: WorldActions,
        game: GameState,
        team: Team | int | None,
    ) -> None:
        """Copy base/outpost tree commands into their owned action fields."""

        if self._last_blackboard is None:
            raise RuntimeError("behavior-tree blackboard is unavailable")
        controlled = torch.ones(
            (game.num_envs, constants.TEAM_COUNT),
            device=game.device,
            dtype=torch.bool,
        )
        if team is not None:
            controlled[:] = False
            controlled[:, int(team)] = True
        blackboard = self._last_blackboard
        actions.dart_open_gate.copy_(
            torch.where(controlled, blackboard.dart_open_gate, actions.dart_open_gate)
        )
        actions.dart_close_gate.copy_(
            torch.where(controlled, blackboard.dart_close_gate, actions.dart_close_gate)
        )
        actions.dart_target.copy_(
            torch.where(controlled, blackboard.dart_target, actions.dart_target)
        )
        actions.radar_target.copy_(
            torch.where(controlled, blackboard.radar_target, actions.radar_target)
        )
        actions.radar_report_xy.copy_(
            torch.where(
                controlled[:, :, None],
                blackboard.radar_report_xy,
                actions.radar_report_xy,
            )
        )
        actions.radar_illuminate.copy_(
            torch.where(controlled, blackboard.radar_illuminate, actions.radar_illuminate)
        )
        actions.radar_double.copy_(
            torch.where(controlled, blackboard.radar_double, actions.radar_double)
        )
        actions.radar_key_solved.copy_(
            torch.where(controlled, blackboard.radar_key_solved, actions.radar_key_solved)
        )

    def _apply_motion(
        self,
        actions: WorldActions,
        game: GameState,
        world: KinematicState,
        missions: Tensor,
        controlled: Tensor,
        controlled_teams: Tensor,
    ) -> None:
        """Execute A* targets and preserve final-goal action semantics."""

        super()._apply_motion(
            actions,
            game,
            world,
            missions,
            controlled,
            controlled_teams,
        )
        deploy_zone = self.arena.hero_deploy_zone(world.position_xy)
        for team in (Team.RED, Team.BLUE):
            hero = slot(team, Role.HERO)
            actions.hero_deploy[:, team] = (
                controlled_teams[team]
                & (missions[:, hero] == int(TacticalMission.HERO_DEPLOY))
                & deploy_zone[:, hero]
            )


__all__ = [
    "BehaviorNavigationDiagnostics",
    "BehaviorTreeOpponent",
]
