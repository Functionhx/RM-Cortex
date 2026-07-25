from __future__ import annotations

import torch

from rm_referee import GameState
from rm_referee import constants
from rm_referee.schema import DartGateState, DartTarget, HeroProfile, Role, Team, slot
from rm_world.behavior_tree import (
    NO_MISSION,
    HierarchicalBehaviorTrees,
    NodeStatus,
    TeamIntent,
)
from rm_world.kinematics import KinematicState
from rm_world.scripted import TacticalMission


def test_global_tree_uses_vectorized_priority_and_stable_paths() -> None:
    game = GameState.create(4)
    game.elapsed_s[:] = torch.tensor((0.0, 200.0, 350.0, 350.0))
    blue_outpost = slot(Team.BLUE, Role.OUTPOST)
    red_base = slot(Team.RED, Role.BASE)
    game.alive[2, blue_outpost] = False
    game.hp[3, red_base] = game.max_hp[3, red_base] * 0.20

    forest = HierarchicalBehaviorTrees()
    blackboard, trace = forest.decide(game, team=Team.RED)

    assert blackboard.intent[:, Team.RED].tolist() == [
        TeamIntent.OPENING,
        TeamIntent.OUTPOST_PRESSURE,
        TeamIntent.BASE_ASSAULT,
        TeamIntent.DEFEND,
    ]
    assert trace.path(0, Team.RED) == (
        "global",
        "strategy_selector",
        "publish_opening",
    )
    assert trace.path(2, Team.RED)[-1] == "publish_base_assault"
    assert trace.path(3, Team.RED)[-1] == "publish_defend"
    assert torch.all(trace.status[:, Team.RED] == NodeStatus.SUCCESS)
    assert torch.all(trace.selected_leaf[:, Team.BLUE] == -1)


def test_all_eight_role_trees_select_a_leaf_and_publish_missions() -> None:
    game = GameState.create(1)
    forest = HierarchicalBehaviorTrees()

    blackboard, trace = forest.decide(game, team=Team.RED)

    assert forest.tree_names == (
        "global",
        "hero",
        "engineer",
        "infantry_3",
        "infantry_4",
        "aerial",
        "sentry",
        "base",
        "outpost",
    )
    for role in Role:
        path = trace.path(0, Team.RED, role)
        assert path
        assert path[0] == role.name.lower()
        assert trace.status[0, Team.RED, int(role) + 1] == NodeStatus.SUCCESS

    assert blackboard.mission[0, slot(Team.RED, Role.HERO)] == TacticalMission.HERO_DEPLOY
    assert (
        blackboard.mission[0, slot(Team.RED, Role.INFANTRY_3)] == TacticalMission.INFANTRY_3_OPENING
    )
    assert (
        blackboard.mission[0, slot(Team.RED, Role.INFANTRY_4)] == TacticalMission.INFANTRY_4_OPENING
    )
    assert blackboard.mission[0, slot(Team.RED, Role.BASE)] == NO_MISSION
    assert blackboard.mission[0, slot(Team.RED, Role.OUTPOST)] == NO_MISSION


def test_engineer_rebuild_preempts_weak_recovery() -> None:
    game = GameState.create(1)
    engineer = slot(Team.RED, Role.ENGINEER)
    outpost = slot(Team.RED, Role.OUTPOST)
    game.alive[0, outpost] = False
    game.outpost_rebuild_charges[0, Team.RED] = 1
    game.weak[0, engineer] = True

    blackboard, trace = HierarchicalBehaviorTrees().decide(game, team=Team.RED)

    assert blackboard.mission[0, engineer] == TacticalMission.ENGINEER_REBUILD
    assert trace.path(0, Team.RED, Role.ENGINEER)[-1] == "stage_at_outpost"


def test_engineer_rebuild_stops_at_the_official_deadline() -> None:
    game = GameState.create(2)
    engineer = slot(Team.RED, Role.ENGINEER)
    outpost = slot(Team.RED, Role.OUTPOST)
    game.alive[:, outpost] = False
    game.outpost_rebuild_charges[:, Team.RED] = 1
    game.elapsed_s[:] = torch.tensor(
        (constants.OUTPOST_REBUILD_DEADLINE_S - 0.1, constants.OUTPOST_REBUILD_DEADLINE_S)
    )

    blackboard, trace = HierarchicalBehaviorTrees().decide(game, team=Team.RED)

    assert blackboard.mission[0, engineer] == TacticalMission.ENGINEER_REBUILD
    assert blackboard.mission[1, engineer] == TacticalMission.ENGINEER_STAGE
    assert trace.path(0, Team.RED, Role.ENGINEER)[-1] == "stage_at_outpost"
    assert trace.path(1, Team.RED, Role.ENGINEER)[-1] == "manage_resources"


def test_hero_deployment_is_held_only_for_remote_control_phases() -> None:
    game = GameState.create(4)
    hero = slot(Team.RED, Role.HERO)
    game.elapsed_s[:] = torch.tensor((0.0, 100.0, 200.0, 0.0))
    game.hero_profile[3, hero] = int(HeroProfile.CLOSE)

    blackboard, trace = HierarchicalBehaviorTrees().decide(game, team=Team.RED)

    assert blackboard.mission[0, hero] == TacticalMission.HERO_DEPLOY
    assert blackboard.mission[1, hero] == TacticalMission.HERO_DEPLOY
    assert blackboard.mission[2, hero] == TacticalMission.INFANTRY_3_OUTPOST
    assert blackboard.mission[3, hero] == TacticalMission.INFANTRY_3_CONTEST
    assert trace.path(0, Team.RED, Role.HERO)[-1] == "deploy_at_platform"
    assert trace.path(1, Team.RED, Role.HERO)[-1] == "deploy_at_platform"
    assert trace.path(2, Team.RED, Role.HERO)[-1] == "attack_outpost"
    assert trace.path(3, Team.RED, Role.HERO)[-1] == "support_center"


def test_base_and_outpost_trees_publish_structure_commands() -> None:
    game = GameState.create(5)
    world = KinematicState.spawn(game)
    game.elapsed_s.fill_(200.0)
    game.dart_gate_opportunities[0, Team.RED] = 1
    game.dart_gate_state[1:, Team.RED] = int(DartGateState.OPEN)
    game.dart_rounds[2, Team.RED] = 0
    game.alive[3, slot(Team.BLUE, Role.OUTPOST)] = False
    game.dart_detection_block_s[4, Team.RED] = 1.0
    game.radar_double_charges[:, Team.RED] = 1
    game.aerial_support_active[:, Team.BLUE] = True

    blackboard, trace = HierarchicalBehaviorTrees().decide(
        game,
        team=Team.RED,
        position_xy=world.position_xy,
    )

    assert blackboard.dart_open_gate[0, Team.RED]
    assert blackboard.dart_target[1, Team.RED] == DartTarget.OUTPOST
    assert blackboard.dart_close_gate[2, Team.RED]
    assert blackboard.dart_target[3, Team.RED] == DartTarget.BASE_TERMINAL_MOVING
    assert blackboard.dart_target[4, Team.RED] == -1
    assert trace.path(0, Team.RED, Role.BASE)[-1] == "open_gate"
    assert trace.path(1, Team.RED, Role.BASE)[-1] == "target_outpost"
    assert trace.path(2, Team.RED, Role.BASE)[-1] == "close_gate"
    assert trace.path(3, Team.RED, Role.BASE)[-1] == "target_base"
    assert trace.path(4, Team.RED, Role.BASE)[-1] == "monitor_gate"

    radar_target = blackboard.radar_target[:, Team.RED]
    assert torch.all((radar_target >= slot(Team.BLUE, Role.HERO)) & (radar_target < 14))
    expected_report = torch.gather(
        world.position_xy,
        1,
        radar_target[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    assert torch.allclose(blackboard.radar_report_xy[:, Team.RED], expected_report)
    assert blackboard.radar_double[:, Team.RED].all()
    assert blackboard.radar_illuminate[:, Team.RED].all()
    assert trace.path(0, Team.RED, Role.OUTPOST)[-1] == "report_priority_target"


def test_radar_tree_remains_available_after_physical_outpost_destruction() -> None:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    game.alive[:, slot(Team.RED, Role.OUTPOST)] = False
    enemy_start = slot(Team.BLUE, Role.HERO)
    game.alive[:, enemy_start : enemy_start + int(Role.BASE)] = False
    game.alive[:, slot(Team.BLUE, Role.AERIAL)] = True

    blackboard, trace = HierarchicalBehaviorTrees().decide(
        game,
        team=Team.RED,
        position_xy=world.position_xy,
    )

    assert blackboard.radar_target[0, Team.RED] == slot(Team.BLUE, Role.AERIAL)
    assert trace.path(0, Team.RED, Role.OUTPOST)[-1] == "report_priority_target"


def test_role_trees_are_distinct_and_center_symmetric_by_schema() -> None:
    game = GameState.create(1)
    game.elapsed_s.fill_(200.0)

    blackboard, trace = HierarchicalBehaviorTrees().decide(game)

    for team in (Team.RED, Team.BLUE):
        infantry_3 = slot(team, Role.INFANTRY_3)
        infantry_4 = slot(team, Role.INFANTRY_4)
        hero = slot(team, Role.HERO)
        assert blackboard.mission[0, infantry_3] == TacticalMission.INFANTRY_3_OUTPOST
        assert blackboard.mission[0, infantry_4] == TacticalMission.INFANTRY_4_OUTPOST
        assert blackboard.mission[0, hero] == TacticalMission.INFANTRY_3_OUTPOST
        assert trace.path(0, team, Role.INFANTRY_3)[0] == "infantry_3"
        assert trace.path(0, team, Role.INFANTRY_4)[0] == "infantry_4"
        assert trace.path(0, team, Role.HERO)[-1] == "attack_outpost"


def test_trace_is_deterministic_and_done_environments_do_not_hit_leaves() -> None:
    game = GameState.create(2)
    game.done[1] = True
    forest = HierarchicalBehaviorTrees()

    first_blackboard, first = forest.decide(game)
    second_blackboard, second = forest.decide(game)

    assert torch.equal(first_blackboard.intent, second_blackboard.intent)
    assert torch.equal(first_blackboard.mission, second_blackboard.mission)
    assert torch.equal(first.visited, second.visited)
    assert torch.equal(first.selected_leaf, second.selected_leaf)
    assert torch.all(first.selected_leaf[1] == -1)
    assert not first.visited[1].any()
    for mask in first.selected_leaf_hits.values():
        assert not mask[1].any()
