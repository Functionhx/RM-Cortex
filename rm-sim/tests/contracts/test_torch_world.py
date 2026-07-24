from __future__ import annotations

import math

import pytest
import torch

from rm_referee import GameState
from rm_referee import constants
from rm_referee.schema import PurchaseKind, Role, Team, Weapon, Zone, slot, unit_roles
from rm_world import (
    BLUE_OUTPOST_CENTER_XY,
    RED_OUTPOST_CENTER_XY,
    ArenaGeometry,
    KinematicCommands,
    KinematicConfig,
    KinematicState,
    KinematicWorld,
    ScriptedOpponent,
    TacticalMission,
    TacticalScriptedOpponent,
    TorchEnvConfig,
    TorchRMArena,
    WorldActions,
    aerial_sortie_state,
    tactical_route_waypoints,
)


def _minimum_ground_clearance(position_xy: torch.Tensor) -> torch.Tensor:
    roles = unit_roles(position_xy.device)
    radii = torch.where(
        roles == Role.BASE,
        torch.full_like(roles, 0.95, dtype=position_xy.dtype),
        torch.where(
            roles == Role.OUTPOST,
            torch.full_like(roles, 0.40, dtype=position_xy.dtype),
            torch.full_like(roles, 0.40, dtype=position_xy.dtype),
        ),
    )
    distance = torch.cdist(position_xy, position_xy)
    clearance = distance - radii[:, None] - radii[None, :]
    valid = (
        (roles[:, None] != Role.AERIAL)
        & (roles[None, :] != Role.AERIAL)
        & ~torch.eye(constants.UNIT_COUNT, device=position_xy.device, dtype=torch.bool)
    )
    return clearance[:, valid].amin(dim=-1)


@pytest.mark.parametrize("mission", list(TacticalMission))
def test_tactical_routes_remain_center_symmetric(mission: TacticalMission) -> None:
    red = tactical_route_waypoints(
        mission,
        Team.RED,
        device="cpu",
        dtype=torch.float32,
    )
    blue = tactical_route_waypoints(
        mission,
        Team.BLUE,
        device="cpu",
        dtype=torch.float32,
    )

    assert red.ndim == 2
    assert red.shape[-1] == 2
    assert torch.allclose(red, -blue)


def test_outposts_use_figure_4_5_diagonal_centers_and_zones() -> None:
    game = GameState.create(1)
    arena = ArenaGeometry()
    world = KinematicState.spawn(game, arena)
    red = slot(Team.RED, Role.OUTPOST)
    blue = slot(Team.BLUE, Role.OUTPOST)

    assert torch.allclose(
        world.position_xy[0, red],
        torch.tensor(RED_OUTPOST_CENTER_XY),
    )
    assert torch.allclose(
        world.position_xy[0, blue],
        torch.tensor(BLUE_OUTPOST_CENTER_XY),
    )
    occupancy = arena.zone_occupancy(world.position_xy)
    assert occupancy[0, red, Zone.OUTPOST]
    assert occupancy[0, blue, Zone.OUTPOST]


def test_all_aerial_tactical_routes_stay_inside_the_section_4_5_airspace() -> None:
    arena = ArenaGeometry()
    aerial_missions = (
        TacticalMission.AERIAL_OPENING,
        TacticalMission.AERIAL_CONTROL,
        TacticalMission.AERIAL_PRESSURE,
        TacticalMission.AERIAL_ASSAULT,
        TacticalMission.AERIAL_RETURN,
        TacticalMission.AERIAL_PAD,
    )

    for mission in aerial_missions:
        for team in (Team.RED, Team.BLUE):
            waypoints = tactical_route_waypoints(
                mission,
                team,
                device="cpu",
                dtype=torch.float32,
            )
            assert arena.aerial_flight_area(waypoints, team).all(), (mission, team)


def test_aerial_sortie_schedule_has_explicit_flight_and_return_windows() -> None:
    elapsed = torch.tensor(
        (0.0, 5.0, 16.9, 17.0, 24.9, 25.0, 74.9, 75.0, 84.9, 85.0),
    )

    requested, returning = aerial_sortie_state(elapsed)

    assert requested.tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
    assert returning.tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_aerial_pad_contact_requires_altitude_and_horizontal_footprint() -> None:
    environment = TorchRMArena(
        TorchEnvConfig(num_envs=1, validate_referee=False),
    )
    aerial = slot(Team.RED, Role.AERIAL)
    environment.world.position_xy[0, aerial] = torch.tensor((0.0, 0.0))
    environment.world.position_z[0, aerial] = 0.0

    environment.step(WorldActions.zeros(environment.game))

    assert not environment.game.aerial_on_pad[0, Team.RED]


def test_tactical_controller_assigns_distinct_roles_and_symmetric_teams() -> None:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    controller = TacticalScriptedOpponent()

    actions = controller.act(game, world)

    assert not torch.allclose(
        actions.body_velocity_xy[:, Role.ENGINEER],
        actions.body_velocity_xy[:, Role.INFANTRY_3],
    )
    assert torch.allclose(
        actions.body_velocity_xy[:, : Role.BASE],
        actions.body_velocity_xy[
            :,
            constants.ROLES_PER_TEAM : constants.ROLES_PER_TEAM + Role.BASE,
        ],
        atol=1.0e-5,
    )
    assert not actions.aerial_support.any()


def test_tactical_controller_uses_remote_flow_for_infantry_ammo() -> None:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    infantry_3 = slot(Team.RED, Role.INFANTRY_3)
    infantry_4 = slot(Team.RED, Role.INFANTRY_4)
    game.out_of_combat_s[0, (infantry_3, infantry_4)] = constants.OUT_OF_COMBAT_S

    actions = TacticalScriptedOpponent().act(game, world)

    assert actions.purchase_unit[0, Team.RED] == infantry_3
    assert actions.purchase_weapon[0, Team.RED] == Weapon.MM17
    assert actions.purchase_kind[0, Team.RED] == PurchaseKind.REMOTE


def test_body_velocity_is_rotated_and_batched() -> None:
    game = GameState.create(2)
    world = KinematicState.zeros(game)
    commands = KinematicCommands.zeros(game)
    hero = slot(Team.RED, Role.HERO)
    world.yaw[:, hero] = math.pi / 2
    commands.body_velocity_xy[:, hero, 0] = 1

    next_world = KinematicWorld().step(world, game, commands, dt=0.1)

    assert torch.allclose(next_world.position_xy[:, hero, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        next_world.position_xy[:, hero, 1],
        torch.full((2,), 0.1),
        atol=1e-6,
    )


def test_buildings_do_not_move_and_field_bounds_are_enforced() -> None:
    game = GameState.create(1)
    world = KinematicState.zeros(game)
    commands = KinematicCommands.zeros(game)
    base = slot(Team.RED, Role.BASE)
    hero = slot(Team.RED, Role.HERO)
    commands.body_velocity_xy[0, base, 0] = 3
    commands.body_velocity_xy[0, hero, 0] = 3
    world.position_xy[0, hero, 0] = 13.9

    next_world = KinematicWorld().step(world, game, commands, dt=1)

    assert next_world.position_xy[0, base, 0] == 0
    assert next_world.position_xy[0, hero, 0] == pytest.approx(14)


def test_exactly_coincident_ground_units_are_separated() -> None:
    game = GameState.create(1)
    world = KinematicState.spawn(game)
    red = slot(Team.RED, Role.INFANTRY_3)
    blue = slot(Team.BLUE, Role.INFANTRY_3)
    world.position_xy[0, red] = torch.tensor([0.0, 6.5])
    world.position_xy[0, blue] = torch.tensor([0.0, 6.5])

    next_world = KinematicWorld(
        KinematicConfig(
            enable_unit_collisions=True,
        )
    ).step(world, game, KinematicCommands.zeros(game))

    separation = torch.linalg.vector_norm(
        next_world.position_xy[0, red] - next_world.position_xy[0, blue]
    )
    assert separation.item() >= 0.80 - 1.0e-4


@pytest.mark.parametrize(
    "policy",
    (ScriptedOpponent(), TacticalScriptedOpponent()),
    ids=("objective", "tactical"),
)
def test_scripted_rollout_keeps_every_ground_footprint_disjoint(
    policy: ScriptedOpponent | TacticalScriptedOpponent,
) -> None:
    environment = TorchRMArena(
        TorchEnvConfig(
            num_envs=1,
            validate_referee=False,
            seed=7,
        )
    )
    minimum_clearance = torch.tensor(torch.inf)
    minimum_static_clearance = torch.tensor(torch.inf)
    roles = unit_roles(environment.game.device)
    moving_ground = (roles != Role.AERIAL) & (roles != Role.BASE) & (roles != Role.OUTPOST)

    for _ in range(60):
        environment.step(policy.act(environment.game, environment.world))
        minimum_clearance = torch.minimum(
            minimum_clearance,
            _minimum_ground_clearance(environment.world.position_xy),
        )
        minimum_static_clearance = torch.minimum(
            minimum_static_clearance,
            environment.arena.signed_distance(environment.world.position_xy)[
                :, moving_ground
            ].amin()
            - environment.world_model.config.robot_radius_m,
        )

    assert minimum_clearance.item() >= -1.0e-4
    assert minimum_static_clearance.item() >= -1.0e-4


def test_random_batched_motion_preserves_contact_constraints() -> None:
    game = GameState.create(32)
    model = KinematicWorld(
        KinematicConfig(
            boundary_margin_m=0.40,
            enable_static_collisions=True,
            enable_unit_collisions=True,
        )
    )
    world = KinematicState.spawn(game, model.arena)
    generator = torch.Generator().manual_seed(23)
    minimum_clearance = torch.tensor(torch.inf)

    for _ in range(80):
        commands = KinematicCommands.zeros(game)
        commands.body_velocity_xy.uniform_(-3.0, 3.0, generator=generator)
        commands.yaw_rate.uniform_(-math.pi, math.pi, generator=generator)
        world = model.step(world, game, commands)
        minimum_clearance = torch.minimum(
            minimum_clearance,
            _minimum_ground_clearance(world.position_xy).amin(),
        )

    assert minimum_clearance.item() >= -1.0e-4
