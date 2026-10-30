"""Unit tests for the body-fitted O-grid mesh family and y+ instrumentation."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from backend.simulation.cfd.case_builder import build_openfoam_case
from backend.simulation.cfd.mesh_builder import (
    MESH_LEVELS,
    MESH_REFINEMENT_RATIO,
    TARGET_Y_PLUS,
    build_ogrid,
    build_profile,
    estimate_first_cell_height,
)
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel
from backend.simulation.cfd.parser import parse_y_plus


def _foam_list_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\n(\d+)\s*\n\(", text)
    assert match is not None
    return int(match.group(1))


def _patch_counts(boundary: Path) -> dict[str, tuple[str, int, int]]:
    text = boundary.read_text(encoding="utf-8")
    patches: dict[str, tuple[str, int, int]] = {}
    for match in re.finditer(
        r"(\w+)\s*\{\s*type\s+(\w+);\s*nFaces\s+(\d+);\s*startFace\s+(\d+);",
        text,
    ):
        patches[match.group(1)] = (match.group(2), int(match.group(3)), int(match.group(4)))
    return patches


def test_mesh_levels_refine_by_a_constant_ratio() -> None:
    coarse = MESH_LEVELS[MeshLevel.COARSE]
    medium = MESH_LEVELS[MeshLevel.MEDIUM]
    fine = MESH_LEVELS[MeshLevel.FINE]

    assert math.isclose(medium.n_surface / coarse.n_surface, MESH_REFINEMENT_RATIO)
    assert math.isclose(fine.n_surface / medium.n_surface, MESH_REFINEMENT_RATIO)
    assert math.isclose(medium.n_radial / coarse.n_radial, MESH_REFINEMENT_RATIO)
    assert math.isclose(fine.n_radial / medium.n_radial, MESH_REFINEMENT_RATIO)


def test_ogrid_has_positive_cells_and_exact_wall_spacing() -> None:
    airfoil = AirfoilGeometry(name="NACA0012", chord_m=1.0, angle_of_attack_deg=4.0)
    resolution = MESH_LEVELS[MeshLevel.COARSE]
    profile = build_profile(airfoil, num_points=resolution.n_surface)
    first_cell = estimate_first_cell_height(
        45.0, 1.0, target_y_plus=TARGET_Y_PLUS
    )

    grid = build_ogrid(profile, resolution, first_cell)

    assert grid.points.shape == (2 * resolution.n_surface - 1, resolution.n_radial + 1, 2)
    thickness = np.linalg.norm(grid.points[:, 1] - grid.points[:, 0], axis=1)
    assert np.allclose(thickness, first_cell, rtol=1e-9, atol=1e-12)

    corner = grid.points
    next_i = np.roll(corner, -1, axis=0)
    next_ij = np.roll(next_i, -1, axis=1)
    next_j = np.roll(corner, -1, axis=1)
    diagonal_1 = next_ij - corner
    diagonal_2 = next_j - next_i
    area = 0.5 * (
        diagonal_1[..., 0] * diagonal_2[..., 1] - diagonal_1[..., 1] * diagonal_2[..., 0]
    )
    assert float(area[:, :-1].min()) > 0.0


def test_case_builder_writes_a_consistent_body_fitted_mesh(tmp_path) -> None:
    airfoil = AirfoilGeometry(name="NACA0012", chord_m=1.0, angle_of_attack_deg=4.0)
    case_dir, stats = build_openfoam_case(
        airfoil=airfoil,
        mesh_level=MeshLevel.COARSE,
        speed_ms=45.0,
        base_dir=tmp_path,
    )

    mesh_dir = case_dir / "constant" / "polyMesh"
    for name in ("points", "faces", "owner", "neighbour", "boundary"):
        assert (mesh_dir / name).exists()
    assert not (case_dir / "system" / "snappyHexMeshDict").exists()
    assert not (case_dir / "constant" / "triSurface").exists()

    n_surface = 2 * MESH_LEVELS[MeshLevel.COARSE].n_surface - 1
    n_radial = MESH_LEVELS[MeshLevel.COARSE].n_radial
    internal_faces = n_surface * (2 * n_radial - 1)
    boundary_faces = 2 * n_surface * (n_radial + 1)
    total_faces = internal_faces + boundary_faces

    assert _foam_list_count(mesh_dir / "points") == 2 * n_surface * (n_radial + 1)
    assert _foam_list_count(mesh_dir / "faces") == total_faces
    assert _foam_list_count(mesh_dir / "owner") == total_faces
    assert _foam_list_count(mesh_dir / "neighbour") == internal_faces
    assert stats.num_cells == n_surface * n_radial

    patches = _patch_counts(mesh_dir / "boundary")
    assert set(patches) == {"inlet", "outlet", "top", "ground", "frontAndBack", "airfoil"}
    assert patches["airfoil"][0] == "wall"
    assert patches["frontAndBack"][0] == "empty"
    assert patches["ground"][0] == "patch"
    assert patches["airfoil"][1] == n_surface
    assert patches["frontAndBack"][1] == 2 * n_surface * n_radial
    assert sum(count for _, count, _ in patches.values()) == total_faces - internal_faces

    control = (case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    params = (case_dir / "case_params.json").read_text(encoding="utf-8")
    assert "endTime         2000" in control
    assert "yPlus" in control
    assert '"max_iterations": 2000' in params
    assert '"n_surface"' in params
    assert '"first_cell_height_m"' in params


def test_ground_case_uses_a_wall_ground_patch(tmp_path) -> None:
    airfoil = AirfoilGeometry(
        name="FRONTWING-4412",
        chord_m=1.0,
        angle_of_attack_deg=8.0,
        ground_height_m=0.15,
    )
    case_dir, _ = build_openfoam_case(
        airfoil=airfoil,
        mesh_level=MeshLevel.COARSE,
        speed_ms=45.0,
        base_dir=tmp_path,
    )

    patches = _patch_counts(case_dir / "constant" / "polyMesh" / "boundary")
    assert patches["ground"][0] == "wall"
    assert patches["ground"][1] > 0


def test_first_cell_height_scales_with_target_y_plus() -> None:
    y1 = estimate_first_cell_height(45.0, 1.0, target_y_plus=1.0)
    y30 = estimate_first_cell_height(45.0, 1.0, target_y_plus=30.0)
    assert y1 > 0.0
    assert math.isclose(y30 / y1, 30.0, rel_tol=1e-12)


def test_parse_y_plus_reads_last_numeric_row(tmp_path) -> None:
    history = tmp_path / "postProcessing" / "yPlus" / "1" / "yPlus.dat"
    history.parent.mkdir(parents=True)
    history.write_text(
        "# Time\tpatch\tmin\tmax\taverage\n"
        "100\tairfoil\t0.8\t12.5\t3.2\n"
        "200\tairfoil\t0.9\t14.1\t3.8\n",
        encoding="utf-8",
    )

    result = parse_y_plus(tmp_path)

    assert result == {"y_plus_min": 0.9, "y_plus_max": 14.1, "y_plus_avg": 3.8}
