from __future__ import annotations

from collections import defaultdict
import math

import pytest
import torch

from rm_isaac.terrain_mesh import build_terrain_prism_mesh_data
from rm_world import ArenaGeometry
from rm_world.arena import TerrainPrimitive, polygon_area


def _primitive_surface_height(
    arena: ArenaGeometry,
    primitive: TerrainPrimitive,
    world_xy: torch.Tensor,
) -> torch.Tensor:
    crown = arena.field_height(world_xy)
    if primitive.vertex_elevations_m:
        polygon = arena.terrain_corners(
            primitive,
            device=world_xy.device,
            dtype=world_xy.dtype,
        )
        elevation = arena._polygon_vertex_elevation(
            world_xy,
            primitive,
            polygon,
        )
    else:
        offset = world_xy - torch.tensor(
            primitive.center_xy,
            device=world_xy.device,
            dtype=world_xy.dtype,
        )
        yaw = math.radians(primitive.yaw_deg)
        local_x = math.cos(yaw) * offset[:, 0] + math.sin(yaw) * offset[:, 1]
        fraction = torch.clamp(
            local_x / primitive.size_xy[0] + 0.5,
            min=0.0,
            max=1.0,
        )
        elevation = primitive.elevation_start_m + fraction * (
            primitive.elevation_end_m - primitive.elevation_start_m
        )
    return crown + elevation


@pytest.mark.parametrize(
    "primitive_name",
    (
        "central_highland",
        "central_plateau",
        "red_trapezoid_highland",
        "red_fortress",
        "red_fortress_top",
        "central_red_connector",
        "red_trapezoid_23_ramp",
        "red_trapezoid_43_ramp",
    ),
)
def test_isaac_mesh_surface_vertices_match_crowned_torch_terrain(
    primitive_name: str,
) -> None:
    arena = ArenaGeometry()
    primitive = next(item for item in arena.config.terrain if item.name == primitive_name)
    mesh = build_terrain_prism_mesh_data(
        primitive,
        field_width_m=arena.config.field_width_m,
        field_crown_slope_deg=arena.config.field_crown_slope_deg,
    )
    local_surface = torch.tensor(
        mesh.vertices[mesh.surface_vertex_offset :],
        dtype=torch.float64,
    )
    world_xy = local_surface[:, :2] + torch.tensor(
        primitive.center_xy,
        dtype=torch.float64,
    )
    crown = arena.field_height(world_xy)

    assert local_surface[:, 2].tolist() == pytest.approx(
        _primitive_surface_height(arena, primitive, world_xy).tolist(),
        abs=1.0e-12,
    )
    center_crown = arena.field_height(torch.tensor((primitive.center_xy,), dtype=torch.float64))
    assert torch.any(torch.abs(crown - center_crown) > 1.0e-4)


@pytest.mark.parametrize(
    "primitive_name",
    (
        "central_highland",
        "central_plateau",
        "red_fortress",
        "red_fortress_top",
        "central_red_connector",
        "red_trapezoid_23_ramp",
        "red_trapezoid_43_ramp",
    ),
)
def test_isaac_mesh_face_interiors_match_torch_and_form_a_closed_prism(
    primitive_name: str,
) -> None:
    arena = ArenaGeometry()
    primitive = next(item for item in arena.config.terrain if item.name == primitive_name)
    mesh = build_terrain_prism_mesh_data(
        primitive,
        field_width_m=arena.config.field_width_m,
        field_crown_slope_deg=arena.config.field_crown_slope_deg,
    )
    vertices = torch.tensor(mesh.vertices, dtype=torch.float64)
    top_faces = [face for face in mesh.faces if min(face) >= mesh.surface_vertex_offset]
    assert top_faces

    projected_area = 0.0
    center_xy = torch.tensor(primitive.center_xy, dtype=torch.float64)
    for face in top_faces:
        triangle = vertices[list(face)]
        world_xy = triangle[:, :2] + center_xy
        centroid_xy = world_xy.mean(dim=0, keepdim=True)
        interpolated_z = triangle[:, 2].mean()
        expected_z = _primitive_surface_height(
            arena,
            primitive,
            centroid_xy,
        )[0]

        assert interpolated_z.item() == pytest.approx(
            expected_z.item(),
            abs=1.0e-12,
        )
        assert not (world_xy[:, 1].amin() < -1.0e-12 and world_xy[:, 1].amax() > 1.0e-12)
        edge_a = triangle[1, :2] - triangle[0, :2]
        edge_b = triangle[2, :2] - triangle[0, :2]
        projected_area += abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])) / 2

    assert projected_area == pytest.approx(
        abs(polygon_area(primitive.footprint_xy)),
        abs=1.0e-10,
    )
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face in mesh.faces:
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_key = (min(start, end), max(start, end))
            edge_uses[edge_key].append((start, end))
    assert {len(uses) for uses in edge_uses.values()} == {2}
    assert all(uses[0] == (uses[1][1], uses[1][0]) for uses in edge_uses.values())


def test_isaac_mesh_bottom_follows_the_field_crown_per_vertex() -> None:
    arena = ArenaGeometry()
    primitive = next(item for item in arena.config.terrain if item.name == "central_highland")
    mesh = build_terrain_prism_mesh_data(
        primitive,
        field_width_m=arena.config.field_width_m,
        field_crown_slope_deg=arena.config.field_crown_slope_deg,
    )
    local_bottom = torch.tensor(
        mesh.vertices[: mesh.surface_vertex_offset],
        dtype=torch.float64,
    )
    world_xy = local_bottom[:, :2] + torch.tensor(
        primitive.center_xy,
        dtype=torch.float64,
    )

    assert local_bottom[:, 2].tolist() == pytest.approx(
        (arena.field_height(world_xy) - 0.025).tolist(),
        abs=1.0e-12,
    )
