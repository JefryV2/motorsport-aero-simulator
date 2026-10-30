"""Body-fitted O-grid generation for 2D incompressible RANS airfoil cases.

The mesh is written directly as an OpenFOAM ``polyMesh``: a single-block
O-grid extruded one cell in span.  The wall is represented exactly, the first
cell height is set from the target y+, and the three mesh levels are
geometrically similar (constant 1.5 refinement ratio), so the family can be
used for a grid-convergence study.

This replaced a ``blockMesh`` + ``snappyHexMesh`` pipeline.  That pipeline
produced 80-degree non-orthogonal faces around the section and wall layers
attached to only a few percent of the surface, so neither wall resolution nor
a systematic refinement family could be demonstrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel, MeshStats

#: Extrusion thickness for the 2D section.  One metre makes the force
#: coefficients directly comparable with a one-metre span.
SPAN_M = 1.0


@dataclass(frozen=True)
class MeshResolution:
    """Tangential and wall-normal resolution of one mesh level."""

    n_surface: int
    n_radial: int


#: Points per side (upper/lower) and radial cells for each level.  Both scale
#: by 1.5 between levels, giving a constant refinement ratio for GCI.
MESH_LEVELS: dict[MeshLevel, MeshResolution] = {
    MeshLevel.COARSE: MeshResolution(n_surface=60, n_radial=64),
    MeshLevel.MEDIUM: MeshResolution(n_surface=90, n_radial=96),
    MeshLevel.FINE: MeshResolution(n_surface=135, n_radial=144),
}

#: Target refinement ratio between consecutive levels.
MESH_REFINEMENT_RATIO = 1.5

#: Wall treatment.  k-omega SST is run with wall functions, so the first cell
#: is placed in the log layer rather than resolving the viscous sublayer.
TARGET_Y_PLUS = 30.0

#: The flat-plate correlation estimates the wall shear from the chord Reynolds
#: number, but an aerofoil is dominated by the leading-edge suction peak, so the
#: first cell lands at roughly half the intended y+.  This factor was calibrated
#: on the measured medium-mesh pilot (target 30 -> measured 13.6) and must be
#: re-checked whenever the mesh family or the wall treatment changes.
WALL_FUNCTION_HEIGHT_FACTOR = 2.2


@dataclass(frozen=True)
class ProfileGeometry:
    """A closed, rotated and translated airfoil profile plus its domain."""

    designation: str
    loop: list[tuple[float, float]]
    x_le: float
    x_te: float
    y_min: float
    y_max: float
    y_mean: float
    chord_m: float
    is_front_wing: bool
    ground_height_m: float | None
    x_in: float
    x_out: float
    y_bottom: float
    y_top: float
    z0: float
    z1: float

    @property
    def domain_area(self) -> float:
        return (self.x_out - self.x_in) * (self.y_top - self.y_bottom)


def generate_naca4_coordinates(
    designation: str = "0012",
    chord: float = 1.0,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate upper and lower surface coordinates for a NACA 4-digit airfoil.

    Returns (x_upper, y_upper, x_lower, y_lower), ordered from leading edge to
    trailing edge.
    """
    m = int(designation[0]) / 100.0  # Maximum camber
    p = int(designation[1]) / 10.0  # Location of maximum camber
    t = int(designation[2:4]) / 100.0  # Maximum thickness as fraction of chord

    # Half-cosine spacing along chord for leading-edge resolution
    beta = np.linspace(0.0, np.pi, num_points)
    x = chord * 0.5 * (1.0 - np.cos(beta))

    # Thickness distribution
    yt = (
        5.0
        * t
        * chord
        * (
            0.2969 * np.sqrt(x / chord)
            - 0.1260 * (x / chord)
            - 0.3516 * (x / chord) ** 2
            + 0.2843 * (x / chord) ** 3
            - 0.1015 * (x / chord) ** 4
        )
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0.0 and p > 0.0:
        for i, xi in enumerate(x):
            xc = xi / chord
            if xc < p:
                yc[i] = chord * (m / (p**2)) * (2.0 * p * xc - xc**2)
                dyc_dx[i] = (2.0 * m / (p**2)) * (p - xc)
            else:
                yc[i] = chord * (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2)
                dyc_dx[i] = (2.0 * m / ((1.0 - p) ** 2)) * (p - xc)

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl


def estimate_first_cell_height(
    speed_ms: float,
    chord_m: float,
    target_y_plus: float = 1.0,
    nu: float = 1.5e-5,
    rho: float = 1.225,
) -> float:
    """Estimate required first cell height y1 using flat plate boundary layer theory."""
    re = speed_ms * chord_m / nu
    if re < 1e3:
        re = 1e3
    cf = (2.0 * math.log10(re) - 0.65) ** -2.3
    tau_w = 0.5 * rho * (speed_ms**2) * cf
    u_tau = math.sqrt(tau_w / rho)
    return float(WALL_FUNCTION_HEIGHT_FACTOR * (target_y_plus * nu) / u_tau)


def naca_designation(airfoil: AirfoilGeometry) -> str:
    """Return a four-digit NACA designation for the requested section."""
    name = airfoil.name.strip().upper()
    if name.startswith("NACA") and len(name) >= 8:
        digits = name[4:8]
        if digits.isdigit():
            return digits
    if name.startswith("FRONTWING"):
        return "4412"
    return "0012"


def is_front_wing(airfoil: AirfoilGeometry) -> bool:
    """Front-wing sections are inverted so positive lift is downforce."""
    return airfoil.name.strip().upper().startswith("FRONTWING")


def build_profile(airfoil: AirfoilGeometry, num_points: int = 90) -> ProfileGeometry:
    """Build the closed airfoil loop, rotate it, and place it in the domain."""
    chord = float(airfoil.chord_m)
    if chord <= 0.0:
        raise ValueError("chord_m must be positive")

    designation = naca_designation(airfoil)
    front_wing = is_front_wing(airfoil)
    xu, yu, xl, yl = generate_naca4_coordinates(designation, chord, num_points)

    # A front wing works upside down: invert camber so positive incidence
    # produces downforce rather than lift.
    if front_wing:
        yu = -yu
        yl = -yl

    # Front-wing sections are inverted and mounted nose-down: a positive UI
    # element angle means a more aggressive downforce setting.  Free-air
    # benchmark sections use the conventional nose-up positive angle of attack.
    if front_wing:
        angle = math.radians(airfoil.angle_of_attack_deg)
    else:
        angle = -math.radians(airfoil.angle_of_attack_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    pivot_x, pivot_y = 0.25 * chord, 0.0

    def rotate(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - pivot_x, y - pivot_y
        return (pivot_x + dx * ca - dy * sa, pivot_y + dx * sa + dy * ca)

    upper = [rotate(float(x), float(y)) for x, y in zip(xu, yu, strict=True)]
    lower = [rotate(float(x), float(y)) for x, y in zip(xl, yl, strict=True)]

    ground = airfoil.ground_height_m
    if ground is not None:
        # Place the closest point of the section exactly at the requested ride height.
        dy = float(ground) - min(y for _, y in upper + lower)
        upper = [(x, y + dy) for x, y in upper]
        lower = [(x, y + dy) for x, y in lower]

    loop = upper + list(reversed(lower[1:]))

    x_values = [p[0] for p in loop]
    y_values = [p[1] for p in loop]
    x_le, x_te = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    y_mean = sum(y_values) / len(y_values)

    if ground is not None:
        x_in, x_out = -4.0 * chord, 10.0 * chord
        y_bottom, y_top = 0.0, 6.0 * chord
    else:
        x_in, x_out = -5.0 * chord, 10.0 * chord
        y_bottom, y_top = -5.0 * chord, 5.0 * chord

    return ProfileGeometry(
        designation=designation,
        loop=loop,
        x_le=x_le,
        x_te=x_te,
        y_min=y_min,
        y_max=y_max,
        y_mean=y_mean,
        chord_m=chord,
        is_front_wing=front_wing,
        ground_height_m=ground,
        x_in=x_in,
        x_out=x_out,
        y_bottom=y_bottom,
        y_top=y_top,
        z0=0.0,
        z1=SPAN_M,
    )


@dataclass(frozen=True)
class OGrid:
    """A single-block body-fitted O-grid in the xy plane."""

    points: np.ndarray  # (n_surface, n_radial + 1, 2)
    wall_points: np.ndarray
    outer_points: np.ndarray
    outward_normals: np.ndarray
    first_cell_height: float

    @property
    def n_surface(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_radial(self) -> int:
        return int(self.points.shape[1] - 1)


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _surface_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit tangents and outward normals for a closed profile loop."""
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("airfoil profile contains coincident points")
    tangent = tangent / length[:, None]
    normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    centroid = points.mean(axis=0)
    if float(np.mean(np.einsum("ij,ij->i", normals, points - centroid))) < 0.0:
        normals = -normals
    return tangent, normals


def _ray_rectangle_exit(
    point: np.ndarray,
    direction: np.ndarray,
    profile: ProfileGeometry,
) -> np.ndarray:
    """Intersect an outward ray with the rectangular far-field boundary."""
    limits = (
        (0, profile.x_in),
        (0, profile.x_out),
        (1, profile.y_bottom),
        (1, profile.y_top),
    )
    distances = []
    for axis, limit in limits:
        if abs(float(direction[axis])) < 1e-14:
            continue
        distance = (limit - float(point[axis])) / float(direction[axis])
        if distance > 1e-9:
            distances.append(distance)
    if not distances:
        raise ValueError("could not project a wall point onto the far field")
    exit_point = point + min(distances) * direction
    exit_point[0] = min(max(exit_point[0], profile.x_in), profile.x_out)
    exit_point[1] = min(max(exit_point[1], profile.y_bottom), profile.y_top)
    return exit_point


def _solve_growth_ratio(first_cell: float, distance: float, n_cells: int) -> float:
    """Return the geometric growth ratio that spans ``distance`` in n cells.

    When the requested first cell does not fit, the spacing is uniform instead
    (which resolves the wall more finely than the target y+ rather than folding
    the grid).
    """
    if first_cell * n_cells >= distance:
        return 1.0

    def residual(growth: float) -> float:
        return first_cell * (growth**n_cells - 1.0) / (growth - 1.0) - distance

    low, high = 1.0 + 1e-12, 2.0
    while residual(high) < 0.0:
        high *= 1.5
    for _ in range(100):
        middle = 0.5 * (low + high)
        if residual(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _minimum_cell_area(points: np.ndarray) -> float:
    """Signed area of every O-grid cell; a valid grid has all areas positive."""
    corner = points
    next_i = np.roll(points, -1, axis=0)
    next_ij = np.roll(next_i, -1, axis=1)
    next_j = np.roll(points, -1, axis=1)
    diagonal_1 = next_ij - corner
    diagonal_2 = next_j - next_i
    area = 0.5 * (
        diagonal_1[..., 0] * diagonal_2[..., 1] - diagonal_1[..., 1] * diagonal_2[..., 0]
    )
    return float(area[:, :-1].min())


def _march_steps(first_cell: float, path_length: float, n_cells: int) -> np.ndarray:
    """Geometric step lengths whose sum is the path length.

    When the target first cell does not fit, the spacing becomes uniform and
    slightly finer than the target (better y+ rather than a folded grid).
    """
    if first_cell * n_cells >= path_length:
        return np.full(n_cells, path_length / n_cells)
    growth = _solve_growth_ratio(first_cell, path_length, n_cells)
    return first_cell * growth ** np.arange(n_cells, dtype=float)


def build_ogrid(
    profile: ProfileGeometry,
    resolution: MeshResolution,
    first_cell_height: float,
) -> OGrid:
    """Generate the body-fitted O-grid points for a profile.

    Every radial line is the outward wall normal, so the first cell height
    matches the y+ target exactly and the wall-adjacent rows stay orthogonal to
    the surface.  The far field is where that normal meets the rectangle: a
    projection from the section centroid instead leaves the downstream side
    uncovered, because no radial direction of a thin section points downstream.
    """
    points = np.asarray(profile.loop, dtype=float)
    if _polygon_signed_area(points) > 0.0:
        points = points[::-1].copy()

    _, normals = _surface_frames(points)
    n_surface = len(points)
    n_radial = resolution.n_radial
    grid = np.empty((n_surface, n_radial + 1, 2), dtype=float)
    outer = np.empty((n_surface, 2), dtype=float)
    for i in range(n_surface):
        wall = points[i]
        normal = normals[i]
        far = _ray_rectangle_exit(wall, normal, profile)
        length = float(np.linalg.norm(far - wall))
        if length < first_cell_height:
            raise ValueError("far field is closer than one cell to the airfoil")

        steps = _march_steps(first_cell_height, length, n_radial)
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        grid[i] = wall + cumulative[:, None] * normal
        grid[i, -1] = far
        outer[i] = far

    if not np.all(np.isfinite(grid)):
        raise ValueError("O-grid generation produced non-finite coordinates")
    if _minimum_cell_area(grid) <= 0.0:
        raise ValueError("O-grid generation folded cells; check the profile geometry")

    first_cell = float(np.min(np.linalg.norm(grid[:, 1] - grid[:, 0], axis=1)))
    return OGrid(
        points=grid,
        wall_points=points,
        outer_points=outer,
        outward_normals=normals,
        first_cell_height=first_cell,
    )


def estimate_mesh_stats(
    profile: ProfileGeometry,
    mesh_level: MeshLevel,
    resolution: MeshResolution,
    first_cell_height: float,
) -> MeshStats:
    """Return the nominal size of an O-grid before ``checkMesh`` measures it."""
    n_surface = 2 * resolution.n_surface - 1
    cells = n_surface * resolution.n_radial
    return MeshStats(
        level=mesh_level,
        num_cells=cells,
        num_points=2 * n_surface * (resolution.n_radial + 1),
        grid_spacing_h=math.sqrt(profile.domain_area / cells),
        estimated_y_plus=TARGET_Y_PLUS,
        first_cell_height_m=first_cell_height,
    )


def _poly_mesh_header(cls: str, obj: str) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v1912                                 |
|   \\\\  /    A nd           | Web:      www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _write_foam_list(path: Path, cls: str, obj: str, entries: list[str]) -> None:
    path.write_text(
        _poly_mesh_header(cls, obj) + f"{len(entries)}\n(\n" + "".join(entries) + ")\n",
        encoding="utf-8",
        newline="\n",
    )


def _side_of_point(
    point: np.ndarray, profile: ProfileGeometry, tolerance: float
) -> list[str]:
    sides = []
    if abs(float(point[0]) - profile.x_in) <= tolerance:
        sides.append("inlet")
    if abs(float(point[0]) - profile.x_out) <= tolerance:
        sides.append("outlet")
    if abs(float(point[1]) - profile.y_top) <= tolerance:
        sides.append("top")
    if abs(float(point[1]) - profile.y_bottom) <= tolerance:
        sides.append("ground")
    return sides


def _classify_outer_face(
    start: np.ndarray,
    end: np.ndarray,
    outward: np.ndarray,
    profile: ProfileGeometry,
) -> str:
    """Assign an outer-boundary face to one far-field patch.

    Faces along a rectangle side share that side with both endpoints.  A face
    spanning a far-field corner shares none, so it is assigned by the outward
    direction of the wall normal that generated it.  Without that fallback a
    side hit by a single point can end up with no faces at all.
    """
    tolerance = 1e-7 * max(profile.chord_m, 1.0)
    shared = set(_side_of_point(start, profile, tolerance)) & set(
        _side_of_point(end, profile, tolerance)
    )
    if shared:
        for name in ("inlet", "outlet", "top", "ground"):
            if name in shared:
                return name

    if abs(float(outward[0])) >= abs(float(outward[1])):
        return "outlet" if float(outward[0]) > 0.0 else "inlet"
    return "top" if float(outward[1]) > 0.0 else "ground"


def write_ogrid_mesh(
    case_dir: Path,
    profile: ProfileGeometry,
    mesh_level: MeshLevel,
    speed_ms: float,
    nu: float = 1.5e-5,
    rho: float = 1.225,
) -> MeshStats:
    """Write a complete OpenFOAM ``polyMesh`` for the extruded section."""
    resolution = MESH_LEVELS[mesh_level]
    first_cell = estimate_first_cell_height(
        speed_ms,
        profile.chord_m,
        target_y_plus=TARGET_Y_PLUS,
        nu=nu,
        rho=rho,
    )
    grid = build_ogrid(profile, resolution, first_cell)
    n_surface = grid.n_surface
    n_radial = grid.n_radial
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    def point_id(i: int, j: int, plane: int) -> int:
        return plane * n_surface * (n_radial + 1) + (i % n_surface) * (n_radial + 1) + j

    point_lines = []
    for z in (profile.z0, profile.z1):
        for i in range(n_surface):
            for j in range(n_radial + 1):
                x, y = grid.points[i, j]
                point_lines.append(f"({x:.12g} {y:.12g} {z:.12g})\n")
    _write_foam_list(mesh_dir / "points", "vectorField", "points", point_lines)

    faces: list[tuple[int, int, int, int]] = []
    owners: list[int] = []
    neighbours: list[int] = []

    # Tangential faces.  The wrap face uses cell 0 as owner so every internal
    # face has owner < neighbour, and its normal points from cell 0 to the
    # last cell.
    for i in range(n_surface):
        for j in range(n_radial):
            if i == 0:
                faces.append(
                    (
                        point_id(0, j, 0),
                        point_id(0, j, 1),
                        point_id(0, j + 1, 1),
                        point_id(0, j + 1, 0),
                    )
                )
                owners.append(j)
                neighbours.append((n_surface - 1) * n_radial + j)
            if i < n_surface - 1:
                faces.append(
                    (
                        point_id(i + 1, j, 0),
                        point_id(i + 1, j + 1, 0),
                        point_id(i + 1, j + 1, 1),
                        point_id(i + 1, j, 1),
                    )
                )
                owners.append(i * n_radial + j)
                neighbours.append((i + 1) * n_radial + j)

    # Radial faces.
    for i in range(n_surface):
        for j in range(n_radial - 1):
            faces.append(
                (
                    point_id(i, j + 1, 0),
                    point_id(i, j + 1, 1),
                    point_id(i + 1, j + 1, 1),
                    point_id(i + 1, j + 1, 0),
                )
            )
            owners.append(i * n_radial + j)
            neighbours.append(i * n_radial + j + 1)

    n_internal = len(faces)

    # OpenFOAM requires internal faces in upper-triangular order (owner, then
    # neighbour, both ascending); otherwise checkMesh reports unordered faces
    # and the lduMatrix assembly is not guaranteed to work.
    order = sorted(range(n_internal), key=lambda k: (owners[k], neighbours[k]))
    faces = [faces[k] for k in order]
    owners = [owners[k] for k in order]
    neighbours = [neighbours[k] for k in order]
    if any(owner >= neighbour for owner, neighbour in zip(owners, neighbours, strict=True)):
        raise ValueError("internal face owner/neighbour ordering is invalid")

    airfoil_faces = []
    airfoil_owners = []
    for i in range(n_surface):
        airfoil_faces.append(
            (
                point_id(i, 0, 0),
                point_id(i + 1, 0, 0),
                point_id(i + 1, 0, 1),
                point_id(i, 0, 1),
            )
        )
        airfoil_owners.append(i * n_radial)

    outer_faces: dict[str, list[tuple[int, int, int, int]]] = {
        "inlet": [],
        "outlet": [],
        "top": [],
        "ground": [],
    }
    outer_owners: dict[str, list[int]] = {name: [] for name in outer_faces}
    for i in range(n_surface):
        face = (
            point_id(i, n_radial, 0),
            point_id(i, n_radial, 1),
            point_id(i + 1, n_radial, 1),
            point_id(i + 1, n_radial, 0),
        )
        name = _classify_outer_face(
            grid.outer_points[i],
            grid.outer_points[(i + 1) % n_surface],
            0.5 * (grid.outward_normals[i] + grid.outward_normals[(i + 1) % n_surface]),
            profile,
        )
        outer_faces[name].append(face)
        outer_owners[name].append(i * n_radial + n_radial - 1)

    span_faces = []
    span_owners: list[int] = []
    for i in range(n_surface):
        for j in range(n_radial):
            span_faces.append(
                (
                    point_id(i, j, 0),
                    point_id(i, j + 1, 0),
                    point_id(i + 1, j + 1, 0),
                    point_id(i + 1, j, 0),
                )
            )
            span_faces.append(
                (
                    point_id(i, j, 1),
                    point_id(i + 1, j, 1),
                    point_id(i + 1, j + 1, 1),
                    point_id(i, j + 1, 1),
                )
            )
            span_owners.extend((i * n_radial + j, i * n_radial + j))

    ground_type = "wall" if profile.ground_height_m is not None else "patch"
    patch_definitions: list[tuple[str, str, list[tuple[int, int, int, int]], list[int]]] = [
        ("inlet", "patch", outer_faces["inlet"], outer_owners["inlet"]),
        ("outlet", "patch", outer_faces["outlet"], outer_owners["outlet"]),
        ("top", "patch", outer_faces["top"], outer_owners["top"]),
        ("ground", ground_type, outer_faces["ground"], outer_owners["ground"]),
        ("frontAndBack", "empty", span_faces, span_owners),
        ("airfoil", "wall", airfoil_faces, airfoil_owners),
    ]

    for name, _, patch_faces, _patch_owners in patch_definitions:
        if not patch_faces:
            raise ValueError(f"far-field patch '{name}' has no faces")

    boundary_faces: list[tuple[int, int, int, int]] = []
    boundary_owners = []
    patch_counts: list[tuple[str, str, int, int]] = []
    for name, patch_type, patch_faces, patch_owners in patch_definitions:
        start = n_internal + len(boundary_faces)
        patch_counts.append((name, patch_type, len(patch_faces), start))
        boundary_faces.extend(patch_faces)
        boundary_owners.extend(patch_owners)

    face_lines = [f"4({a} {b} {c} {d})\n" for a, b, c, d in faces + boundary_faces]
    _write_foam_list(mesh_dir / "faces", "faceList", "faces", face_lines)
    _write_foam_list(
        mesh_dir / "owner", "labelList", "owner", [f"{value}\n" for value in owners + boundary_owners]
    )
    _write_foam_list(
        mesh_dir / "neighbour", "labelList", "neighbour", [f"{value}\n" for value in neighbours]
    )

    boundary_text = _poly_mesh_header("polyBoundaryMesh", "boundary") + (
        f"{len(patch_counts)}\n(\n"
    )
    for name, patch_type, count, start in patch_counts:
        boundary_text += (
            f"    {name}\n"
            "    {\n"
            f"        type            {patch_type};\n"
            f"        nFaces          {count};\n"
            f"        startFace       {start};\n"
            "    }\n"
        )
    boundary_text += ")\n"
    (mesh_dir / "boundary").write_text(boundary_text, encoding="utf-8", newline="\n")

    total_faces = n_internal + len(boundary_faces)
    if total_faces != len(face_lines) or len(owners + boundary_owners) != total_faces:
        raise ValueError("inconsistent polyMesh face accounting")

    return estimate_mesh_stats(profile, mesh_level, resolution, first_cell)


