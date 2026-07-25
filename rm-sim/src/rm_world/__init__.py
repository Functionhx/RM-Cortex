"""Batched Torch world, geometry, observations, and training environment."""

from rm_world.actions import WorldActions
from rm_world.arena import (
    BLUE_AERIAL_PAD_CENTER_XY,
    BLUE_BASE_CENTER_XY,
    BLUE_FORTRESS_CENTER_XY,
    BLUE_OUTPOST_CENTER_XY,
    BLUE_SUPPLY_CENTER_XY,
    RED_AERIAL_PAD_CENTER_XY,
    RED_BASE_CENTER_XY,
    RED_FORTRESS_CENTER_XY,
    RED_OUTPOST_CENTER_XY,
    RED_SUPPLY_CENTER_XY,
    ArenaConfig,
    ArenaGeometry,
    TerrainPrimitive,
)
from rm_world.backend import TorchRuleBackend
from rm_world.behavior_controller import (
    BehaviorNavigationDiagnostics,
    BehaviorTreeOpponent,
)
from rm_world.behavior_tree import (
    BehaviorTreeTrace,
    HierarchicalBehaviorTrees,
    NodeStatus,
    TeamBlackboard,
    TeamIntent,
)
from rm_world.belief import (
    BeliefConfig,
    EntityBeliefTracker,
    EntityBeliefView,
    EntitySource,
)
from rm_world.env import TorchEnvConfig, TorchEnvStep, TorchRMArena
from rm_world.geometry import ArmorSolution, HitModel, HitModelConfig
from rm_world.kinematics import (
    KinematicCommands,
    KinematicConfig,
    KinematicState,
    KinematicWorld,
)
from rm_world.observations import ObservationBuilder, WorldObservation
from rm_world.pathfinding import (
    GridAStarPlanner,
    NavigationGrid,
    PathResult,
    PathStatus,
    team_symmetric_goal,
)
from rm_world.rewards import RewardBuilder, RewardConfig
from rm_world.scripted import (
    AERIAL_SORTIE_WINDOWS_S,
    MOBILE_UNIT_SLOTS,
    TACTICAL_RED_ROUTES,
    ScriptedOpponent,
    TacticalMission,
    TacticalPhase,
    TacticalScriptedOpponent,
    aerial_sortie_state,
    tactical_phase,
    tactical_phase_label,
    tactical_route_waypoints,
)

__all__ = [
    "ArenaConfig",
    "ArenaGeometry",
    "ArmorSolution",
    "AERIAL_SORTIE_WINDOWS_S",
    "BLUE_AERIAL_PAD_CENTER_XY",
    "BLUE_BASE_CENTER_XY",
    "BLUE_FORTRESS_CENTER_XY",
    "BLUE_OUTPOST_CENTER_XY",
    "BLUE_SUPPLY_CENTER_XY",
    "BeliefConfig",
    "BehaviorNavigationDiagnostics",
    "BehaviorTreeOpponent",
    "BehaviorTreeTrace",
    "EntityBeliefTracker",
    "EntityBeliefView",
    "EntitySource",
    "GridAStarPlanner",
    "HitModel",
    "HitModelConfig",
    "HierarchicalBehaviorTrees",
    "KinematicCommands",
    "KinematicConfig",
    "KinematicState",
    "KinematicWorld",
    "MOBILE_UNIT_SLOTS",
    "NavigationGrid",
    "NodeStatus",
    "ObservationBuilder",
    "PathResult",
    "PathStatus",
    "RewardBuilder",
    "RewardConfig",
    "RED_AERIAL_PAD_CENTER_XY",
    "RED_BASE_CENTER_XY",
    "RED_FORTRESS_CENTER_XY",
    "RED_OUTPOST_CENTER_XY",
    "RED_SUPPLY_CENTER_XY",
    "ScriptedOpponent",
    "TACTICAL_RED_ROUTES",
    "TeamBlackboard",
    "TeamIntent",
    "TacticalMission",
    "TacticalPhase",
    "TacticalScriptedOpponent",
    "TorchEnvConfig",
    "TorchEnvStep",
    "TorchRMArena",
    "TorchRuleBackend",
    "TerrainPrimitive",
    "tactical_phase",
    "tactical_phase_label",
    "tactical_route_waypoints",
    "team_symmetric_goal",
    "WorldActions",
    "WorldObservation",
    "aerial_sortie_state",
]
