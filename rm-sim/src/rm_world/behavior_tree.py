"""Vectorized, auditable behavior-tree decisions for the oracle baseline.

This module intentionally stops at the decision blackboard.  Navigation and
``WorldActions`` execution consume the selected missions elsewhere, keeping
the tree topology independently testable and easy to inspect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Mapping

import torch
from torch import Tensor

from rm_referee import constants
from rm_referee.schema import DartGateState, DartTarget, HeroProfile, Role, Team, slot
from rm_referee.state import GameState
from rm_world.scripted import TacticalMission, TacticalPhase, tactical_phase


NO_MISSION = -1


class NodeStatus(IntEnum):
    """Behavior-tree result values stored in trace tensors."""

    FAILURE = 0
    SUCCESS = 1
    RUNNING = 2


class TeamIntent(IntEnum):
    """High-level intent published by the global team tree."""

    STOPPED = 0
    OPENING = 1
    TERRAIN_CONTROL = 2
    OUTPOST_PRESSURE = 3
    BASE_ASSAULT = 4
    DEFEND = 5


@dataclass
class TeamBlackboard:
    """Batched decisions written by the global and role trees."""

    intent: Tensor
    mission: Tensor
    dart_open_gate: Tensor
    dart_close_gate: Tensor
    dart_target: Tensor
    radar_target: Tensor
    radar_report_xy: Tensor
    radar_illuminate: Tensor
    radar_double: Tensor
    radar_key_solved: Tensor

    @classmethod
    def create(cls, game: GameState) -> "TeamBlackboard":
        return cls(
            intent=torch.full(
                (game.num_envs, constants.TEAM_COUNT),
                int(TeamIntent.STOPPED),
                device=game.device,
                dtype=torch.int8,
            ),
            mission=torch.full(
                (game.num_envs, constants.UNIT_COUNT),
                NO_MISSION,
                device=game.device,
                dtype=torch.long,
            ),
            dart_open_gate=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
            dart_close_gate=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
            dart_target=torch.full(
                (game.num_envs, constants.TEAM_COUNT),
                -1,
                device=game.device,
                dtype=torch.int8,
            ),
            radar_target=torch.full(
                (game.num_envs, constants.TEAM_COUNT),
                -1,
                device=game.device,
                dtype=torch.long,
            ),
            radar_report_xy=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT, 2),
                device=game.device,
                dtype=game.dtype,
            ),
            radar_illuminate=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
            radar_double=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
            radar_key_solved=torch.zeros(
                (game.num_envs, constants.TEAM_COUNT),
                device=game.device,
                dtype=torch.bool,
            ),
        )


@dataclass(frozen=True)
class BehaviorTreeTrace:
    """One decision tick with stable node IDs and selected leaf paths."""

    tree_names: tuple[str, ...]
    node_paths: tuple[tuple[str, ...], ...]
    leaf_paths: tuple[tuple[str, ...], ...]
    visited: Tensor
    selected_leaf: Tensor
    status: Tensor

    def path(
        self,
        env_index: int,
        team: Team | int,
        role: Role | int | None = None,
    ) -> tuple[str, ...]:
        """Return the selected root-to-action path for one tree invocation."""

        tree_index = 0 if role is None else int(role) + 1
        leaf_id = int(self.selected_leaf[env_index, int(team), tree_index].item())
        return () if leaf_id < 0 else self.leaf_paths[leaf_id]

    @property
    def selected_leaf_hits(self) -> Mapping[str, Tensor]:
        """Return selected-leaf masks shaped ``[env, team]``.

        Unlike visited-node diagnostics, these masks count exactly one terminal
        decision per active tree and do not inflate metrics with selector probes.
        """

        hits: dict[str, Tensor] = {}
        for leaf_id, path in enumerate(self.leaf_paths):
            hits["/".join(path)] = (self.selected_leaf == leaf_id).any(dim=-1)
        return hits


Predicate = Callable[["TreeContext"], Tensor]
Effect = Callable[["TreeContext", Tensor], Tensor | None]


@dataclass
class TreeContext:
    """Mutable invocation context shared by nodes in one tree tick."""

    game: GameState
    blackboard: TeamBlackboard
    team: int
    role: Role | None
    tree_index: int
    runtime: "_TraceRuntime"
    position_xy: Tensor | None = None
    dart_fire_ready: Tensor | None = None

    @property
    def unit_slot(self) -> int:
        if self.role is None:
            raise RuntimeError("the global team tree has no unit slot")
        return slot(self.team, self.role)


class BehaviorNode(ABC):
    """Base class for fixed-topology vectorized behavior-tree nodes."""

    def __init__(self, name: str) -> None:
        if not name or "/" in name:
            raise ValueError("node names must be non-empty and cannot contain '/'")
        self.name = name

    @property
    def children(self) -> tuple["BehaviorNode", ...]:
        return ()

    @abstractmethod
    def tick(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> Tensor:
        """Return an int8 ``NodeStatus`` tensor shaped ``[env]``."""

    def _begin(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> tuple[str, ...]:
        node_path = (*path, self.name)
        context.runtime.visit(node_path, active, context.team)
        return node_path


class Condition(BehaviorNode):
    """Leaf predicate that succeeds where its batched condition is true."""

    def __init__(self, name: str, predicate: Predicate) -> None:
        super().__init__(name)
        self.predicate = predicate

    def tick(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> Tensor:
        self._begin(context, active, path)
        result = self.predicate(context)
        if result.shape != active.shape or result.dtype != torch.bool:
            raise ValueError("condition predicates must return bool [env] tensors")
        return torch.where(
            active & result,
            torch.full_like(active, int(NodeStatus.SUCCESS), dtype=torch.int8),
            torch.full_like(active, int(NodeStatus.FAILURE), dtype=torch.int8),
        )


class Action(BehaviorNode):
    """Terminal leaf that mutates only the decision blackboard."""

    def __init__(self, name: str, effect: Effect) -> None:
        super().__init__(name)
        self.effect = effect

    def tick(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> Tensor:
        node_path = self._begin(context, active, path)
        succeeded = self.effect(context, active)
        if succeeded is None:
            succeeded = active
        if succeeded.shape != active.shape or succeeded.dtype != torch.bool:
            raise ValueError("action effects must return None or bool [env] tensors")
        succeeded &= active
        context.runtime.select_leaf(
            node_path,
            succeeded,
            context.team,
            context.tree_index,
        )
        return torch.where(
            succeeded,
            torch.full_like(active, int(NodeStatus.SUCCESS), dtype=torch.int8),
            torch.full_like(active, int(NodeStatus.FAILURE), dtype=torch.int8),
        )


class Composite(BehaviorNode):
    def __init__(self, name: str, children: Iterable[BehaviorNode]) -> None:
        super().__init__(name)
        self._children = tuple(children)
        if not self._children:
            raise ValueError("composite nodes need at least one child")

    @property
    def children(self) -> tuple[BehaviorNode, ...]:
        return self._children


class Sequence(Composite):
    """Tick children in order until one fails or remains running."""

    def tick(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> Tensor:
        node_path = self._begin(context, active, path)
        pending = active.clone()
        running = torch.zeros_like(active)
        for child in self.children:
            child_status = child.tick(context, pending, node_path)
            running |= pending & (child_status == NodeStatus.RUNNING)
            pending &= child_status == NodeStatus.SUCCESS
        return torch.where(
            pending,
            torch.full_like(active, int(NodeStatus.SUCCESS), dtype=torch.int8),
            torch.where(
                running,
                torch.full_like(active, int(NodeStatus.RUNNING), dtype=torch.int8),
                torch.full_like(active, int(NodeStatus.FAILURE), dtype=torch.int8),
            ),
        )


class Selector(Composite):
    """Tick children in priority order until one succeeds or runs."""

    def tick(
        self,
        context: TreeContext,
        active: Tensor,
        path: tuple[str, ...],
    ) -> Tensor:
        node_path = self._begin(context, active, path)
        pending = active.clone()
        succeeded = torch.zeros_like(active)
        running = torch.zeros_like(active)
        for child in self.children:
            child_status = child.tick(context, pending, node_path)
            succeeded |= pending & (child_status == NodeStatus.SUCCESS)
            running |= pending & (child_status == NodeStatus.RUNNING)
            pending &= child_status == NodeStatus.FAILURE
        return torch.where(
            succeeded,
            torch.full_like(active, int(NodeStatus.SUCCESS), dtype=torch.int8),
            torch.where(
                running,
                torch.full_like(active, int(NodeStatus.RUNNING), dtype=torch.int8),
                torch.full_like(active, int(NodeStatus.FAILURE), dtype=torch.int8),
            ),
        )


@dataclass(frozen=True)
class BehaviorTree:
    """Named root used by the fixed global/role forest."""

    name: str
    root: BehaviorNode


class _TraceRuntime:
    def __init__(
        self,
        game: GameState,
        tree_names: tuple[str, ...],
        node_paths: tuple[tuple[str, ...], ...],
        leaf_paths: tuple[tuple[str, ...], ...],
    ) -> None:
        self.tree_names = tree_names
        self.node_paths = node_paths
        self.leaf_paths = leaf_paths
        self._node_ids = {path: index for index, path in enumerate(node_paths)}
        self._leaf_ids = {path: index for index, path in enumerate(leaf_paths)}
        self.visited = torch.zeros(
            (game.num_envs, constants.TEAM_COUNT, len(node_paths)),
            device=game.device,
            dtype=torch.bool,
        )
        self.selected_leaf = torch.full(
            (game.num_envs, constants.TEAM_COUNT, len(tree_names)),
            -1,
            device=game.device,
            dtype=torch.int16,
        )
        self.status = torch.full_like(
            self.selected_leaf,
            int(NodeStatus.FAILURE),
            dtype=torch.int8,
        )

    def visit(self, path: tuple[str, ...], active: Tensor, team: int) -> None:
        self.visited[:, team, self._node_ids[path]] |= active

    def select_leaf(
        self,
        path: tuple[str, ...],
        selected: Tensor,
        team: int,
        tree_index: int,
    ) -> None:
        leaf_id = self._leaf_ids[path]
        self.selected_leaf[:, team, tree_index].copy_(
            torch.where(
                selected,
                torch.full_like(self.selected_leaf[:, team, tree_index], leaf_id),
                self.selected_leaf[:, team, tree_index],
            )
        )

    def freeze(self) -> BehaviorTreeTrace:
        return BehaviorTreeTrace(
            tree_names=self.tree_names,
            node_paths=self.node_paths,
            leaf_paths=self.leaf_paths,
            visited=self.visited,
            selected_leaf=self.selected_leaf,
            status=self.status,
        )


def _catalog(
    trees: tuple[BehaviorTree, ...],
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    node_paths: list[tuple[str, ...]] = []
    leaf_paths: list[tuple[str, ...]] = []

    def walk(node: BehaviorNode, parent: tuple[str, ...]) -> None:
        path = (*parent, node.name)
        if path in node_paths:
            raise ValueError(f"duplicate behavior-tree path: {'/'.join(path)}")
        node_paths.append(path)
        if isinstance(node, Action):
            leaf_paths.append(path)
        for child in node.children:
            walk(child, path)

    for tree in trees:
        walk(tree.root, (tree.name,))
    return tuple(node_paths), tuple(leaf_paths)


def _set_intent(intent: TeamIntent) -> Effect:
    def effect(context: TreeContext, active: Tensor) -> None:
        context.blackboard.intent[:, context.team].copy_(
            torch.where(
                active,
                torch.full_like(
                    context.blackboard.intent[:, context.team],
                    int(intent),
                ),
                context.blackboard.intent[:, context.team],
            )
        )

    return effect


def _set_mission(mission: TacticalMission | int) -> Effect:
    mission_id = int(mission)

    def effect(context: TreeContext, active: Tensor) -> None:
        unit = context.unit_slot
        context.blackboard.mission[:, unit].copy_(
            torch.where(
                active,
                torch.full_like(context.blackboard.mission[:, unit], mission_id),
                context.blackboard.mission[:, unit],
            )
        )

    return effect


def _intent_is(intent: TeamIntent) -> Predicate:
    return lambda context: context.blackboard.intent[:, context.team] == int(intent)


def _dead(context: TreeContext) -> Tensor:
    return ~context.game.alive[:, context.unit_slot]


def _weak(context: TreeContext) -> Tensor:
    return context.game.weak[:, context.unit_slot]


def _branch(
    name: str,
    condition: Predicate,
    action_name: str,
    effect: Effect,
) -> Sequence:
    return Sequence(
        name,
        (
            Condition(f"{name}_condition", condition),
            Action(action_name, effect),
        ),
    )


def _global_tree() -> BehaviorTree:
    def emergency(context: TreeContext) -> Tensor:
        base = slot(context.team, Role.BASE)
        outpost = slot(context.team, Role.OUTPOST)
        base_low = context.game.hp[:, base] < context.game.max_hp[:, base] * 0.45
        outpost_low = context.game.alive[:, outpost] & (
            context.game.hp[:, outpost] < context.game.max_hp[:, outpost] * 0.50
        )
        return base_low | outpost_low

    def enemy_outpost_down(context: TreeContext) -> Tensor:
        return ~context.game.alive[:, slot(1 - context.team, Role.OUTPOST)]

    def at_least(phase: TacticalPhase) -> Predicate:
        return lambda context: tactical_phase(context.game.elapsed_s) >= int(phase)

    root = Selector(
        "strategy_selector",
        (
            _branch(
                "emergency_defense", emergency, "publish_defend", _set_intent(TeamIntent.DEFEND)
            ),
            _branch(
                "base_assault",
                enemy_outpost_down,
                "publish_base_assault",
                _set_intent(TeamIntent.BASE_ASSAULT),
            ),
            _branch(
                "outpost_pressure",
                at_least(TacticalPhase.OUTPOST_PRESSURE),
                "publish_outpost_pressure",
                _set_intent(TeamIntent.OUTPOST_PRESSURE),
            ),
            _branch(
                "terrain_control",
                at_least(TacticalPhase.TERRAIN_CONTROL),
                "publish_terrain_control",
                _set_intent(TeamIntent.TERRAIN_CONTROL),
            ),
            Action("publish_opening", _set_intent(TeamIntent.OPENING)),
        ),
    )
    return BehaviorTree("global", root)


def _hero_tree() -> BehaviorTree:
    def wants_remote_deployment(context: TreeContext) -> Tensor:
        """Hold deployment only during the strategy's remote-hero phases."""

        remote_profile = context.game.hero_profile[:, context.unit_slot] == int(HeroProfile.REMOTE)
        remote_phase = (context.blackboard.intent[:, context.team] == int(TeamIntent.OPENING)) | (
            context.blackboard.intent[:, context.team] == int(TeamIntent.TERRAIN_CONTROL)
        )
        return remote_profile & remote_phase

    root = Selector(
        "hero_selector",
        (
            _branch("dead", _dead, "hold_dead", _set_mission(NO_MISSION)),
            _branch(
                "recover", _weak, "recover_supply", _set_mission(TacticalMission.ENGINEER_RESUPPLY)
            ),
            _branch(
                "deploy",
                wants_remote_deployment,
                "deploy_at_platform",
                _set_mission(TacticalMission.HERO_DEPLOY),
            ),
            _branch(
                "defend",
                _intent_is(TeamIntent.DEFEND),
                "defend_outpost",
                _set_mission(TacticalMission.SENTRY_DEFEND),
            ),
            _branch(
                "assault_base",
                _intent_is(TeamIntent.BASE_ASSAULT),
                "attack_base",
                _set_mission(TacticalMission.INFANTRY_3_BASE),
            ),
            _branch(
                "pressure_outpost",
                _intent_is(TeamIntent.OUTPOST_PRESSURE),
                "attack_outpost",
                _set_mission(TacticalMission.INFANTRY_3_OUTPOST),
            ),
            Action("support_center", _set_mission(TacticalMission.INFANTRY_3_CONTEST)),
        ),
    )
    return BehaviorTree("hero", root)


def _engineer_tree() -> BehaviorTree:
    def can_attempt_rebuild(context: TreeContext) -> Tensor:
        own_outpost = slot(context.team, Role.OUTPOST)
        return (
            ~context.game.alive[:, own_outpost]
            & (context.game.outpost_rebuild_charges[:, context.team] > 0)
            & (context.game.elapsed_s < constants.OUTPOST_REBUILD_DEADLINE_S)
        )

    root = Selector(
        "engineer_selector",
        (
            _branch("dead", _dead, "hold_dead", _set_mission(NO_MISSION)),
            _branch(
                "rebuild",
                can_attempt_rebuild,
                "stage_at_outpost",
                _set_mission(TacticalMission.ENGINEER_REBUILD),
            ),
            _branch(
                "recover", _weak, "recover_supply", _set_mission(TacticalMission.ENGINEER_RESUPPLY)
            ),
            Action("manage_resources", _set_mission(TacticalMission.ENGINEER_STAGE)),
        ),
    )
    return BehaviorTree("engineer", root)


def _infantry_tree(role: Role) -> BehaviorTree:
    is_three = role == Role.INFANTRY_3
    prefix = "infantry_3" if is_three else "infantry_4"
    opening = TacticalMission.INFANTRY_3_OPENING if is_three else TacticalMission.INFANTRY_4_OPENING
    contest = TacticalMission.INFANTRY_3_CONTEST if is_three else TacticalMission.INFANTRY_4_CONTEST
    outpost = TacticalMission.INFANTRY_3_OUTPOST if is_three else TacticalMission.INFANTRY_4_OUTPOST
    base = TacticalMission.INFANTRY_3_BASE if is_three else TacticalMission.INFANTRY_4_BASE
    recover = TacticalMission.INFANTRY_3_RECOVER if is_three else TacticalMission.INFANTRY_4_RECOVER
    root = Selector(
        f"{prefix}_selector",
        (
            _branch("dead", _dead, "hold_dead", _set_mission(NO_MISSION)),
            _branch("recover", _weak, "recover_supply", _set_mission(recover)),
            _branch(
                "defend",
                _intent_is(TeamIntent.DEFEND),
                "screen_outpost",
                _set_mission(recover),
            ),
            _branch(
                "assault_base",
                _intent_is(TeamIntent.BASE_ASSAULT),
                "attack_base",
                _set_mission(base),
            ),
            _branch(
                "pressure_outpost",
                _intent_is(TeamIntent.OUTPOST_PRESSURE),
                "attack_outpost",
                _set_mission(outpost),
            ),
            _branch(
                "control_center",
                _intent_is(TeamIntent.TERRAIN_CONTROL),
                "contest_center",
                _set_mission(contest),
            ),
            Action("opening_route", _set_mission(opening)),
        ),
    )
    return BehaviorTree(prefix, root)


def _aerial_tree() -> BehaviorTree:
    def unavailable(context: TreeContext) -> Tensor:
        return (
            context.game.aerial_support_bank_s[:, context.team] <= 0
        ) & ~context.game.aerial_support_active[:, context.team]

    root = Selector(
        "aerial_selector",
        (
            _branch(
                "unavailable", unavailable, "hold_pad", _set_mission(TacticalMission.AERIAL_PAD)
            ),
            _branch("recover", _weak, "return_to_pad", _set_mission(TacticalMission.AERIAL_RETURN)),
            _branch(
                "assault_base",
                _intent_is(TeamIntent.BASE_ASSAULT),
                "assault_base",
                _set_mission(TacticalMission.AERIAL_ASSAULT),
            ),
            _branch(
                "pressure_outpost",
                _intent_is(TeamIntent.OUTPOST_PRESSURE),
                "pressure_outpost",
                _set_mission(TacticalMission.AERIAL_PRESSURE),
            ),
            _branch(
                "control",
                _intent_is(TeamIntent.TERRAIN_CONTROL),
                "control_high_ground",
                _set_mission(TacticalMission.AERIAL_CONTROL),
            ),
            Action("opening_sortie", _set_mission(TacticalMission.AERIAL_OPENING)),
        ),
    )
    return BehaviorTree("aerial", root)


def _sentry_tree() -> BehaviorTree:
    root = Selector(
        "sentry_selector",
        (
            _branch("dead", _dead, "hold_dead", _set_mission(NO_MISSION)),
            _branch(
                "recover",
                _weak,
                "recover_supply",
                _set_mission(TacticalMission.SENTRY_RECOVER),
            ),
            _branch(
                "defend",
                _intent_is(TeamIntent.DEFEND),
                "defensive_stance",
                _set_mission(TacticalMission.SENTRY_DEFEND),
            ),
            _branch(
                "assault_base",
                _intent_is(TeamIntent.BASE_ASSAULT),
                "offensive_stance",
                _set_mission(TacticalMission.SENTRY_BASE),
            ),
            Action("mobile_forward", _set_mission(TacticalMission.SENTRY_FORWARD)),
        ),
    )
    return BehaviorTree("sentry", root)


def _set_dart_command(
    *,
    open_gate: bool = False,
    close_gate: bool = False,
    target: DartTarget | int = -1,
) -> Effect:
    def effect(context: TreeContext, active: Tensor) -> None:
        team = context.team
        context.blackboard.dart_open_gate[:, team] |= active & open_gate
        context.blackboard.dart_close_gate[:, team] |= active & close_gate
        context.blackboard.dart_target[:, team].copy_(
            torch.where(
                active,
                torch.full_like(context.blackboard.dart_target[:, team], int(target)),
                context.blackboard.dart_target[:, team],
            )
        )

    return effect


def _base_tree() -> BehaviorTree:
    def gate_is(state: DartGateState) -> Predicate:
        return lambda context: context.game.dart_gate_state[:, context.team] == int(state)

    def can_open(context: TreeContext) -> Tensor:
        return (
            gate_is(DartGateState.CLOSED)(context)
            & (context.game.dart_gate_opportunities[:, context.team] > 0)
            & (context.game.dart_rounds[:, context.team] > 0)
        )

    def open_and_empty(context: TreeContext) -> Tensor:
        return gate_is(DartGateState.OPEN)(context) & (
            context.game.dart_rounds[:, context.team] <= 0
        )

    def ready_to_fire(context: TreeContext) -> Tensor:
        policy_ready = (
            torch.ones(
                context.game.num_envs,
                device=context.game.device,
                dtype=torch.bool,
            )
            if context.dart_fire_ready is None
            else context.dart_fire_ready[:, context.team]
        )
        return (
            gate_is(DartGateState.OPEN)(context)
            & (context.game.dart_rounds[:, context.team] > 0)
            & (context.game.dart_detection_block_s[:, context.team] <= 0)
            & policy_ready
        )

    def enemy_outpost_alive(context: TreeContext) -> Tensor:
        return context.game.alive[:, slot(1 - context.team, Role.OUTPOST)]

    root = Selector(
        "base_selector",
        (
            _branch("destroyed", _dead, "remain_inactive", _set_mission(NO_MISSION)),
            _branch(
                "close_empty_gate",
                open_and_empty,
                "close_gate",
                _set_dart_command(close_gate=True),
            ),
            _branch(
                "fire_outpost",
                lambda context: ready_to_fire(context) & enemy_outpost_alive(context),
                "target_outpost",
                _set_dart_command(target=DartTarget.OUTPOST),
            ),
            _branch(
                "fire_base",
                ready_to_fire,
                "target_base",
                _set_dart_command(target=DartTarget.BASE_TERMINAL_MOVING),
            ),
            _branch(
                "open_available_gate",
                can_open,
                "open_gate",
                _set_dart_command(open_gate=True),
            ),
            Action("monitor_gate", _set_dart_command()),
        ),
    )
    return BehaviorTree("base", root)


def _set_radar_command() -> Effect:
    candidate_roles = (
        Role.HERO,
        Role.ENGINEER,
        Role.INFANTRY_3,
        Role.INFANTRY_4,
        Role.AERIAL,
        Role.SENTRY,
    )

    def effect(context: TreeContext, active: Tensor) -> None:
        team = context.team
        enemy = 1 - team
        candidates = torch.tensor(
            tuple(slot(enemy, role) for role in candidate_roles),
            device=context.game.device,
            dtype=torch.long,
        )
        candidate_alive = context.game.alive[:, candidates]
        if context.position_xy is None:
            candidate_score = torch.arange(
                len(candidate_roles),
                device=context.game.device,
                dtype=context.game.dtype,
            )[None, :].expand(context.game.num_envs, -1)
        else:
            observer_xy = context.position_xy[:, slot(team, Role.OUTPOST)]
            candidate_score = torch.linalg.vector_norm(
                context.position_xy[:, candidates] - observer_xy[:, None, :],
                dim=-1,
            )
        candidate_score = torch.where(
            candidate_alive,
            candidate_score,
            torch.full_like(candidate_score, torch.inf),
        )
        nearest_index = candidate_score.argmin(dim=-1)
        target = candidates[nearest_index]
        target_valid = candidate_alive.any(dim=-1) & active
        if context.position_xy is None:
            target_valid &= False
        target = torch.where(target_valid, target, torch.full_like(target, -1))
        context.blackboard.radar_target[:, team].copy_(target)
        if context.position_xy is not None:
            target_safe = torch.clamp(target, min=0)
            report = torch.gather(
                context.position_xy,
                1,
                target_safe[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            context.blackboard.radar_report_xy[:, team].copy_(
                torch.where(target_valid[:, None], report, torch.zeros_like(report))
            )
        context.blackboard.radar_illuminate[:, team].copy_(
            active & context.game.aerial_support_active[:, enemy]
        )
        aggressive = (context.blackboard.intent[:, team] == int(TeamIntent.OUTPOST_PRESSURE)) | (
            context.blackboard.intent[:, team] == int(TeamIntent.BASE_ASSAULT)
        )
        context.blackboard.radar_double[:, team].copy_(
            target_valid
            & aggressive
            & (context.game.radar_double_charges[:, team] > 0)
            & (context.game.radar_double_uses[:, team] == 0)
            & (context.game.radar_double_s[:, team] <= 0)
        )

    return effect


def _outpost_tree() -> BehaviorTree:
    def has_live_enemy(context: TreeContext) -> Tensor:
        enemy_start = (1 - context.team) * constants.ROLES_PER_TEAM
        enemy_mobile = slice(enemy_start, enemy_start + int(Role.BASE))
        result = context.game.alive[:, enemy_mobile].any(dim=-1)
        return torch.zeros_like(result) if context.position_xy is None else result

    root = Selector(
        "outpost_selector",
        (
            _branch(
                "track_enemy",
                has_live_enemy,
                "report_priority_target",
                _set_radar_command(),
            ),
            Action("monitor_radar", _set_mission(NO_MISSION)),
        ),
    )
    return BehaviorTree("outpost", root)


class HierarchicalBehaviorTrees:
    """Global team tree followed by eight independent role trees."""

    def __init__(self) -> None:
        self.trees = (
            _global_tree(),
            _hero_tree(),
            _engineer_tree(),
            _infantry_tree(Role.INFANTRY_3),
            _infantry_tree(Role.INFANTRY_4),
            _aerial_tree(),
            _sentry_tree(),
            _base_tree(),
            _outpost_tree(),
        )
        self.tree_names = tuple(tree.name for tree in self.trees)
        expected = ("global",) + tuple(role.name.lower() for role in Role)
        if self.tree_names != expected:
            raise RuntimeError("behavior-tree order must match the stable role schema")
        self.node_paths, self.leaf_paths = _catalog(self.trees)

    def decide(
        self,
        game: GameState,
        *,
        team: Team | int | None = None,
        position_xy: Tensor | None = None,
        dart_fire_ready: Tensor | None = None,
    ) -> tuple[TeamBlackboard, BehaviorTreeTrace]:
        """Run global decisions before role decisions for each controlled team."""

        if position_xy is not None and position_xy.shape != (
            game.num_envs,
            constants.UNIT_COUNT,
            2,
        ):
            raise ValueError("position_xy must have shape [env, unit, 2]")
        if dart_fire_ready is not None and (
            dart_fire_ready.shape != (game.num_envs, constants.TEAM_COUNT)
            or dart_fire_ready.dtype != torch.bool
        ):
            raise ValueError("dart_fire_ready must be a bool [env, team] tensor")
        blackboard = TeamBlackboard.create(game)
        runtime = _TraceRuntime(
            game,
            self.tree_names,
            self.node_paths,
            self.leaf_paths,
        )
        teams = (int(team),) if team is not None else tuple(range(constants.TEAM_COUNT))
        active = ~game.done
        for team_index in teams:
            if team_index < 0 or team_index >= constants.TEAM_COUNT:
                raise ValueError("team must be RED, BLUE, 0, 1, or None")
            for tree_index, tree in enumerate(self.trees):
                role = None if tree_index == 0 else Role(tree_index - 1)
                context = TreeContext(
                    game=game,
                    blackboard=blackboard,
                    team=team_index,
                    role=role,
                    tree_index=tree_index,
                    runtime=runtime,
                    position_xy=position_xy,
                    dart_fire_ready=dart_fire_ready,
                )
                status = tree.root.tick(context, active, (tree.name,))
                runtime.status[:, team_index, tree_index].copy_(
                    torch.where(
                        active,
                        status,
                        runtime.status[:, team_index, tree_index],
                    )
                )
        return blackboard, runtime.freeze()


__all__ = [
    "NO_MISSION",
    "Action",
    "BehaviorNode",
    "BehaviorTree",
    "BehaviorTreeTrace",
    "Condition",
    "HierarchicalBehaviorTrees",
    "NodeStatus",
    "Selector",
    "Sequence",
    "TeamBlackboard",
    "TeamIntent",
    "TreeContext",
]
