"""Imported-surface measurement for the 3D CFD tier.

An uploaded mesh is measured, not fitted: this module reads the triangles the
user actually supplied, optionally re- oriens them into the solver frame, and
reports the numbers the OpenFOAM case needs.

Dataclasses here are frozen and derived only from the file; nothing is inferred
from the built-in parametric car.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.visualization.mesh_io import parse_obj, parse_stl

# Runtime sanity bounds after scaling, in metres. A mesh the solver cannot see
# (a millimetre-scale part) or cannot fit in a laptop-size domain (30 m) is
# refused with its measured extents rather than silently reinterpreted.
MIN_LENGTH_M = 0.3
MAX_LENGTH_M = 12.0
MAX_HEIGHT_M = 4.0
MAX_LATERAL_M = 6.0

# Above this many triangles per case the run is labelled in provenance, but
# still allowed: snappyHexMesh tolerates large surfaces, it just costs time.
HEAVY_TRIANGLE_COUNT = 400_000


@dataclass(frozen=True)
class ImportedSurface:
    """A measured, solver-oriented imported mesh.

    ``vertices`` are triangles in the solver frame: +x streamwise, +y up,
    +z lateral (the frame the 2D O-grid cases already use). ``area_front`` is
    the projected silhouette area on the y–z plane, the reference area the
    force coefficients are divided by.

    ``floor_y_m`` is where the tunnel's ground plane sits. It defaults to the
    mesh's own lowest point, which is what a car resting on its tyres wants; a
    body authored with a genuine gap under it needs ``ground_clearance_m`` or
    the gap is measured as part of the body and the belt lands on the belly.
    """

    vertices: np.ndarray
    faces: np.ndarray
    scale_m_per_unit: float
    axes: str
    source_triangle_count: int
    frontal_area_m2: float
    watertight_edge_fraction: float
    extent_x_m: float
    extent_y_m: float
    extent_z_m: float
    height_waterline_m: float  # the mesh's own minimum y after ground alignment
    floor_y_m: float = 0.0  # solver-frame y of the tunnel ground plane
    ground_clearance_m: float = 0.0  # floor_y_m below the mesh's lowest point
    rasters_used: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _frontal_area_raster(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, faces: np.ndarray, bins: int = 200
) -> tuple[float, int]:
    """Projected area on the y–z plane by silhouette rasterization.

    Frontal area is the area of the *union* of projected triangles, so the
    projected triangles are filled into a binary grid instead of summed: an
    overlapping soup of degenerate triangles cannot double-count pixels.
    """
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max - y_min < 1e-9:
        return 0.0, 0
    dy = (y_max - y_min) / bins
    grid = np.zeros((bins, bins), dtype=bool)
    v0 = np.stack([y[faces[:, 0]], z[faces[:, 0]]], axis=1)
    v1 = np.stack([y[faces[:, 1]], z[faces[:, 1]]], axis=1)
    v2 = np.stack([y[faces[:, 2]], z[faces[:, 2]]], axis=1)

    y_lo = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    y_hi = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    i0 = np.clip(((y_lo - y_min) / dy).astype(int), 0, bins - 1)
    i1 = np.clip(((y_hi - y_min) / dy).astype(int), 0, bins - 1)
    z_min = float(z.min())
    z_max = float(z.max())
    dz = max(z_max - z_min, 1e-9)
    j0 = np.clip((((np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])) - z_min) / dz * bins).astype(int), 0, bins - 1)
    j1 = np.clip((((np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])) - z_min) / dz * bins).astype(int), 0, bins - 1)
    # Per-triangle scanline fill on the projection grid.
    for idx in range(len(faces)):
        a, b, c = v0[idx], v1[idx], v2[idx]
        for i in range(int(i0[idx]), int(i1[idx]) + 1):
            yc = y_min + (i + 0.5) * dy
            # y-crossing z-range of triangle edges at this raster line
            zcross: list[float] = []
            for p, q in ((a, b), (b, c), (c, a)):
                if (p[0] - yc) * (q[0] - yc) < 0.0 or (p[0] == yc and q[0] == yc):
                    if abs(q[0] - p[0]) < 1e-12:
                        zcross.append(min(p[1], q[1]))
                    else:
                        t = (yc - p[0]) / (q[0] - p[0])
                        zcross.append(p[1] + t * (q[1] - p[1]))
            if not zcross:
                continue
            j_lo = int(np.clip((min(zcross) - z_min) / dz * bins, 0, bins)) - 1
            j_hi = int(np.clip((max(zcross) - z_min) / dz * bins, 0, bins)) + 1
            j_lo = max(j_lo, int(j0[idx]) - 1)
            j_hi = min(j_hi, int(j1[idx]) + 1)
            if j_hi < j_lo:
                j_lo, j_hi = j_hi, j_lo
            j_lo = max(j_lo, 0)
            j_hi = min(j_hi, bins - 1)
            grid[i, j_lo : j_hi + 1] = True
    # Conservative: cells marked by their centres get half-cell padding built in
    # by the scan bounds above; un-marked fully-outside cells stay 0.
    used = int(grid.sum())
    return used * dy * (dz / bins), used


def _watertight_fraction(faces: np.ndarray) -> float:
    """Fraction of mesh edges shared by exactly two triangles.

    A fully watertight closed surface scores 1.0. Snappy tolerates some
    openness, but a report below ~0.5 usually means seams or missing panels -
    reported, never silently repaired.
    """
    if len(faces) == 0:
        return 0.0
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    paired = counts[counts >= 2].sum()
    total = len(edges)
    if total == 0:
        return 0.0
    return float(paired / total)


def axes_to_solver(
    vertices: np.ndarray, axes: str
) -> np.ndarray:
    """Rotate an uploaded mesh into the solver frame.

    ``solver`` means the file is already x-streamwise / y-up / z-lateral.
    ``three`` is the reader-assumed Three.js authoring frame (+x lateral,
    +y up, +z rearward), converted per ``docs/COORDINATES.md`` so the streamwise
    axis becomes -z.
    """
    if axes == "solver":
        return np.array(vertices, copy=True)
    if axes == "three":
        # solver.x = -z_three, solver.y = y_three, solver.z = x_three
        out = np.empty_like(vertices)
        out[:, 0] = -vertices[:, 2]
        out[:, 1] = vertices[:, 1]
        out[:, 2] = vertices[:, 0]
        return out
    raise ValueError("axes must be 'three' (default) or 'solver'")


def measure_imported_surface(
    data: bytes,
    extension: str,
    *,
    axes: str = "three",
    scale_m_per_unit: float | None = None,
    align_ground: bool = True,
    ground_clearance_m: float = 0.0,
) -> ImportedSurface:
    """Measure an uploaded mesh and return the solver-oriented surface.

    ``ground_clearance_m`` lowers the tunnel floor that many metres below the
    mesh's own lowest point; 0.0 (the default) rests the floor on it, which is
    what a car standing on its tyres means.

    Raises ``ValueError`` (mapped to HTTP 422 by the API) when the scaled mesh
    is outside the sanity envelope or the format is unreadable.
    """
    ext = extension.lower().lstrip(".")
    if ext == "stl":
        raw_vertices, faces, _ = parse_stl(data)
    elif ext == "obj":
        raw_vertices, faces, _ = parse_obj(data)
    else:
        raise ValueError(f"Unsupported CFD format '{ext}': upload an .stl or .obj")

    if len(faces) == 0 or len(raw_vertices) == 0:
        raise ValueError("Mesh contains no triangles to solve.")

    scale = 1.0 if scale_m_per_unit is None else float(scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_m_per_unit must be a positive number")

    warnings: list[str] = []
    vertices = axes_to_solver(np.asarray(raw_vertices, dtype=float), axes) * scale

    if align_ground:
        shift = float(np.min(vertices[:, 1]))
        vertices[:, 1] -= shift
        if shift != 0.0 and abs(shift) > 0.01:
            warnings.append(
                f"mesh bottom sat {shift:+.3f} m off the floor plane; moved to y = 0"
            )

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    ext_x, ext_y, ext_z = (float(v) for v in extent)
    if not (MIN_LENGTH_M <= ext_x <= MAX_LENGTH_M):
        raise ValueError(
            f"Scaled mesh is {ext_x:.3f} m long (streamwise). The solver accepts "
            f"{MIN_LENGTH_M}-{MAX_LENGTH_M} m; pass scale_m_per_unit explicitly "
            "if the file is not authored in metres."
        )
    if ext_y <= 0.0 or ext_y > MAX_HEIGHT_M:
        raise ValueError(f"Scaled mesh is {ext_y:.3f} m tall; limit is {MAX_HEIGHT_M} m.")
    if ext_z <= 0.0 or ext_z > MAX_LATERAL_M:
        raise ValueError(f"Scaled mesh is {ext_z:.3f} m wide; limit is {MAX_LATERAL_M} m.")

    clearance = float(ground_clearance_m)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("ground_clearance_m must be a finite number, 0.0 or larger")
    mesh_floor_y = float(np.min(vertices[:, 1]))
    floor_y = mesh_floor_y - clearance
    tunnel_height = float(np.max(vertices[:, 1])) - floor_y
    if tunnel_height > MAX_HEIGHT_M:
        raise ValueError(
            f"Mesh top would sit {tunnel_height:.3f} m above the ground plane "
            f"(limit {MAX_HEIGHT_M} m) with ground_clearance_m={clearance:.3f}; "
            "reduce the clearance or scale the mesh down."
        )

    frontal, rasters = _frontal_area_raster(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
    if frontal <= 0.0:
        raise ValueError(
            "Projected frontal area measured 0.0 - the mesh has no area normal to "
            "the flow axis. Check the axes/scale settings."
        )

    watertight = _watertight_fraction(faces)
    if watertight < 0.9:
        warnings.append(
            f"only {watertight:.0%} of detected edges are shared by two triangles - "
            "the surface may be open; snappyHexMesh can fail to snap it"
        )
    if len(faces) > HEAVY_TRIANGLE_COUNT:
        warnings.append(f"{len(faces)} triangles in the upload; meshing will be slow")

    return ImportedSurface(
        vertices=vertices,
        faces=faces,
        scale_m_per_unit=scale,
        axes=axes,
        source_triangle_count=int(len(faces)),
        frontal_area_m2=float(frontal),
        watertight_edge_fraction=watertight,
        extent_x_m=ext_x,
        extent_y_m=ext_y,
        extent_z_m=ext_z,
        height_waterline_m=mesh_floor_y,
        floor_y_m=floor_y,
        ground_clearance_m=clearance,
        rasters_used=rasters,
        warnings=tuple(warnings),
    )


def write_solver_stl(surface: ImportedSurface, dest: Path) -> Path:
    """Write the mesh as an OpenFOAM-readable ASCII STL (``triSurface/car.stl``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    v = surface.vertices
    f = surface.faces
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    with dest.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("solid car\n")
        for ai, bi, ci in zip(a, b, c, strict=True):
            fh.write(" facet normal 0.0 0.0 0.0\n")
            fh.write("  outer loop\n")
            fh.write(f"   vertex {ai[0]:.6f} {ai[1]:.6f} {ai[2]:.6f}\n")
            fh.write(f"   vertex {bi[0]:.6f} {bi[1]:.6f} {bi[2]:.6f}\n")
            fh.write(f"   vertex {ci[0]:.6f} {ci[1]:.6f} {ci[2]:.6f}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write("endsolid car\n")
    return dest
