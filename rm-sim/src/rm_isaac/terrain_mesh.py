"""Import-safe terrain mesh geometry shared with the Isaac runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math

from rm_world.arena import TerrainPrimitive, triangulate_polygon


_SurfaceVertex = tuple[float, float, float]
_GEOMETRY_EPSILON = 1.0e-12


@dataclass(frozen=True)
class TerrainPrismMeshData:
    """Plain numeric mesh data that does not require Isaac Lab or ``trimesh``."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    surface_vertex_offset: int


def _field_crown_height(
    world_y_m: float,
    *,
    field_width_m: float,
    field_crown_slope_deg: float,
) -> float:
    edge_distance_m = max(field_width_m / 2 - abs(world_y_m), 0.0)
    return edge_distance_m * math.tan(math.radians(field_crown_slope_deg))


def _surface_elevations(primitive: TerrainPrimitive) -> tuple[float, ...]:
    points = primitive.footprint_xy
    if primitive.vertex_elevations_m:
        if len(primitive.vertex_elevations_m) != len(points):
            raise ValueError("vertex elevations must match the polygon footprint")
        return primitive.vertex_elevations_m

    if primitive.size_xy[0] <= 0:
        raise ValueError("terrain primitive length must be positive")
    yaw = math.radians(primitive.yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    elevation_delta = primitive.elevation_end_m - primitive.elevation_start_m
    elevations = []
    for world_x, world_y in points:
        offset_x = world_x - primitive.center_xy[0]
        offset_y = world_y - primitive.center_xy[1]
        local_x = cosine * offset_x + sine * offset_y
        fraction = min(max(local_x / primitive.size_xy[0] + 0.5, 0.0), 1.0)
        elevations.append(primitive.elevation_start_m + fraction * elevation_delta)
    return tuple(elevations)


def _ridge_intersection(start: _SurfaceVertex, end: _SurfaceVertex) -> _SurfaceVertex:
    """Interpolate a triangle edge where it crosses the world ``y=0`` ridge."""

    denominator = end[1] - start[1]
    if abs(denominator) <= _GEOMETRY_EPSILON:
        raise ValueError("cannot intersect an edge parallel to the crown ridge")
    fraction = -start[1] / denominator
    return (
        start[0] + fraction * (end[0] - start[0]),
        0.0,
        start[2] + fraction * (end[2] - start[2]),
    )


def _deduplicate_polygon(vertices: list[_SurfaceVertex]) -> tuple[_SurfaceVertex, ...]:
    result: list[_SurfaceVertex] = []
    for vertex in vertices:
        if not result or any(
            abs(value - previous) > _GEOMETRY_EPSILON
            for value, previous in zip(vertex, result[-1], strict=True)
        ):
            result.append(vertex)
    if len(result) > 1 and all(
        abs(value - first) <= _GEOMETRY_EPSILON
        for value, first in zip(result[-1], result[0], strict=True)
    ):
        result.pop()
    return tuple(result)


def _clip_to_crown_side(
    triangle: tuple[_SurfaceVertex, _SurfaceVertex, _SurfaceVertex],
    *,
    north: bool,
) -> tuple[_SurfaceVertex, ...]:
    """Clip one source triangle to a half-field without changing its surface."""

    clipped: list[_SurfaceVertex] = []
    previous = triangle[-1]
    previous_distance = previous[1] if north else -previous[1]
    previous_inside = previous_distance >= -_GEOMETRY_EPSILON
    for current in triangle:
        current_distance = current[1] if north else -current[1]
        current_inside = current_distance >= -_GEOMETRY_EPSILON
        if current_inside != previous_inside:
            clipped.append(_ridge_intersection(previous, current))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return _deduplicate_polygon(clipped)


def _signed_triangle_area_twice(
    a: _SurfaceVertex,
    b: _SurfaceVertex,
    c: _SurfaceVertex,
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _crown_split_surface(
    primitive: TerrainPrimitive,
) -> tuple[tuple[_SurfaceVertex, ...], tuple[tuple[int, int, int], ...]]:
    """Triangulate a primitive without letting a face cross the crown ridge."""

    elevations = _surface_elevations(primitive)
    source_vertices = tuple(
        (world_x, world_y, elevation)
        for (world_x, world_y), elevation in zip(
            primitive.footprint_xy,
            elevations,
            strict=True,
        )
    )
    vertices: list[_SurfaceVertex] = []
    vertex_lookup: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []

    def vertex_index(vertex: _SurfaceVertex) -> int:
        key = (
            round(vertex[0], 12),
            round(vertex[1], 12),
            round(vertex[2], 12),
        )
        existing = vertex_lookup.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        vertices.append(vertex)
        vertex_lookup[key] = index
        return index

    for a_index, b_index, c_index in triangulate_polygon(primitive.footprint_xy):
        source_triangle = (
            source_vertices[a_index],
            source_vertices[b_index],
            source_vertices[c_index],
        )
        for north in (True, False):
            clipped = _clip_to_crown_side(source_triangle, north=north)
            if len(clipped) < 3:
                continue
            for cursor in range(1, len(clipped) - 1):
                triangle = (clipped[0], clipped[cursor], clipped[cursor + 1])
                area_twice = _signed_triangle_area_twice(*triangle)
                if abs(area_twice) <= _GEOMETRY_EPSILON:
                    continue
                if area_twice < 0:
                    triangle = (triangle[0], triangle[2], triangle[1])
                face = (
                    vertex_index(triangle[0]),
                    vertex_index(triangle[1]),
                    vertex_index(triangle[2]),
                )
                if len(set(face)) == 3:
                    faces.append(face)

    if not faces:
        raise ValueError("terrain footprint produced no surface triangles")
    return tuple(vertices), tuple(faces)


def build_terrain_prism_mesh_data(
    primitive: TerrainPrimitive,
    *,
    field_width_m: float,
    field_crown_slope_deg: float,
    minimum_thickness_m: float = 0.025,
) -> TerrainPrismMeshData:
    """Build a closed crowned prism whose full surface matches Torch terrain."""

    points = primitive.footprint_xy
    if len(points) < 3:
        raise ValueError("terrain mesh requires a polygon footprint")
    if field_width_m <= 0:
        raise ValueError("field_width_m must be positive")
    if minimum_thickness_m <= 0:
        raise ValueError("minimum_thickness_m must be positive")

    surface_vertices, surface_faces = _crown_split_surface(primitive)
    crowns = tuple(
        _field_crown_height(
            world_y,
            field_width_m=field_width_m,
            field_crown_slope_deg=field_crown_slope_deg,
        )
        for _, world_y, _ in surface_vertices
    )
    center_x, center_y = primitive.center_xy
    bottom = tuple(
        (
            world_x - center_x,
            world_y - center_y,
            crown + min(elevation, 0.0) - minimum_thickness_m,
        )
        for (world_x, world_y, elevation), crown in zip(
            surface_vertices,
            crowns,
            strict=True,
        )
    )
    surface = tuple(
        (
            world_x - center_x,
            world_y - center_y,
            crown + elevation,
        )
        for (world_x, world_y, elevation), crown in zip(
            surface_vertices,
            crowns,
            strict=True,
        )
    )

    vertex_count = len(surface_vertices)
    faces: list[tuple[int, int, int]] = []
    boundary_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b, c in surface_faces:
        faces.append((c, b, a))
        faces.append((vertex_count + a, vertex_count + b, vertex_count + c))
        for current, following in ((a, b), (b, c), (c, a)):
            key = (min(current, following), max(current, following))
            boundary_edges.setdefault(key, []).append((current, following))
    for directed_edges in boundary_edges.values():
        if len(directed_edges) == 2:
            continue
        if len(directed_edges) != 1:
            raise ValueError("terrain surface triangulation is non-manifold")
        current, following = directed_edges[0]
        faces.append((current, following, vertex_count + following))
        faces.append((current, vertex_count + following, vertex_count + current))

    return TerrainPrismMeshData(
        vertices=bottom + surface,
        faces=tuple(faces),
        surface_vertex_offset=vertex_count,
    )
