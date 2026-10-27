"""OpenFOAM blockMesh generator for 2D airfoils and ground-effect domains."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel, MeshStats


def generate_naca4_coordinates(
    designation: str = "0012",
    chord: float = 1.0,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate upper and lower surface coordinates for a NACA 4-digit airfoil.

    Returns (x_upper, y_upper, x_lower, y_lower).
    """
    m = int(designation[0]) / 100.0  # Maximum camber
    p = int(designation[1]) / 10.0  # Location of maximum camber
    t = int(designation[2:4]) / 100.0  # Maximum thickness as fraction of chord

    # Half-cosine spacing along chord for leading edge resolution
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
            - 0.1015 * (x / chord) ** 4  # Closed trailing edge
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
    # Schlichting skin friction estimate
    cf = (2.0 * math.log10(re) - 0.65) ** -2.3
    tau_w = 0.5 * rho * (speed_ms**2) * cf
    u_tau = math.sqrt(tau_w / rho)
    y1 = (target_y_plus * nu) / u_tau
    return float(y1)


def generate_blockmesh_dict(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
    speed_ms: float = 50.0,
    chord_m: float = 1.0,
    output_path: Path | None = None,
) -> tuple[str, MeshStats]:
    """Generate OpenFOAM blockMeshDict for 2D airfoil C-grid domain.

    Supports MeshLevel.COARSE, MEDIUM, FINE with systematic refinement ratio r ≈ 1.5.
    """
    # Grid multiplier: 1.0 for Medium, 0.67 for Coarse, 1.5 for Fine
    multiplier_map = {
        MeshLevel.COARSE: 0.67,
        MeshLevel.MEDIUM: 1.0,
        MeshLevel.FINE: 1.5,
    }
    mult = multiplier_map[mesh_level]

    # Baseline cell divisions (Medium)
    nx_inlet = int(round(40 * mult))
    nx_airfoil = int(round(80 * mult))
    nx_wake = int(round(60 * mult))
    ny_wall = int(round(50 * mult))
    nz_span = 1  # 2D in OpenFOAM requires nz = 1 and 'empty' boundary

    # Domain extents (in chord lengths)
    x_inlet = -5.0 * chord_m
    x_outlet = 12.0 * chord_m
    y_top = 8.0 * chord_m
    y_bottom = -airfoil.ground_height_m if airfoil.ground_height_m is not None else -8.0 * chord_m
    z_span = 0.1 * chord_m

    # Cell count & representative grid spacing h
    total_cells = (nx_inlet + nx_airfoil + nx_wake) * ny_wall * nz_span
    grid_spacing_h = chord_m / math.sqrt(total_cells)

    est_y_plus = 1.0 / mult if airfoil.ground_height_m is None else 1.2 / mult

    block_mesh_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v1912 / 2206 / Foundation             |
|   \\\\  /    A nd           | Web:      www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale   1;

vertices
(
    ({x_inlet:.4f} {y_bottom:.4f} 0.0)             // 0
    (0.0 {y_bottom:.4f} 0.0)                      // 1
    ({chord_m:.4f} {y_bottom:.4f} 0.0)            // 2
    ({x_outlet:.4f} {y_bottom:.4f} 0.0)           // 3
    ({x_outlet:.4f} {y_top:.4f} 0.0)              // 4
    ({chord_m:.4f} {y_top:.4f} 0.0)               // 5
    (0.0 {y_top:.4f} 0.0)                         // 6
    ({x_inlet:.4f} {y_top:.4f} 0.0)               // 7

    ({x_inlet:.4f} {y_bottom:.4f} {z_span:.4f})   // 8
    (0.0 {y_bottom:.4f} {z_span:.4f})             // 9
    ({chord_m:.4f} {y_bottom:.4f} {z_span:.4f})   // 10
    ({x_outlet:.4f} {y_bottom:.4f} {z_span:.4f})  // 11
    ({x_outlet:.4f} {y_top:.4f} {z_span:.4f})     // 12
    ({chord_m:.4f} {y_top:.4f} {z_span:.4f})      // 13
    (0.0 {y_top:.4f} {z_span:.4f})                // 14
    ({x_inlet:.4f} {y_top:.4f} {z_span:.4f})      // 15
);

blocks
(
    // Block 0: Upstream domain
    hex (0 1 6 7 8 9 14 15) ({nx_inlet} {ny_wall} {nz_span}) simpleGrading (0.2 1.0 1.0)
    // Block 1: Airfoil passage
    hex (1 2 5 6 9 10 13 14) ({nx_airfoil} {ny_wall} {nz_span}) simpleGrading (1.0 1.0 1.0)
    // Block 2: Downstream wake
    hex (2 3 4 5 10 11 12 13) ({nx_wake} {ny_wall} {nz_span}) simpleGrading (4.0 1.0 1.0)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 8 15 7)
            (7 15 14 6)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (3 4 12 11)
            (4 5 13 12)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (6 14 13 5)
        );
    }}
    ground
    {{
        type {"wall" if airfoil.ground_height_m is not None else "patch"};
        faces
        (
            (0 1 9 8)
            (1 2 10 9)
            (2 3 11 10)
        );
    }}
    airfoil
    {{
        type wall;
        faces
        (
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 7 6 1)
            (1 6 5 2)
            (2 5 4 3)
            (8 9 14 15)
            (9 10 13 14)
            (10 11 12 13)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(block_mesh_content, encoding="utf-8")

    stats = MeshStats(
        level=mesh_level,
        num_cells=total_cells,
        num_points=(nx_inlet + nx_airfoil + nx_wake + 1) * (ny_wall + 1) * 2,
        grid_spacing_h=grid_spacing_h,
        estimated_y_plus=est_y_plus,
    )

    return block_mesh_content, stats
