"""Deterministic OpenFOAM case builder for 2D incompressible RANS airfoil cases.

The case is a real, runnable OpenFOAM case:

* ``blockMesh`` builds a box background mesh.
* ``surfaceFeatureExtract`` extracts the airfoil feature edges.
* ``snappyHexMesh`` inserts the extruded NACA section from an STL.
* ``simpleFoam`` solves k-omega SST with a forceCoeffs function object.

The builder never writes a fake solver log.  Solver metadata is written by the
runner only after a real solve.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from backend.simulation.cfd.mesh_builder import (
    MESH_LEVELS,
    SPAN_M,
    build_profile,
    generate_blockmesh_dict,
    generate_snappy_dict,
    generate_surface_feature_dict,
    write_airfoil_stl,
)
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel, MeshStats

DEFAULT_WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cfd_cases"
CASE_FORMAT_VERSION = "real-airfoil-rans-v4"


def foam_header(cls: str, obj: str) -> str:
    """Return a compact OpenFOAM file header."""
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
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def compute_case_hash(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel,
    speed_ms: float,
    nu: float = 1.5e-5,
    turbulence_intensity: float = 0.01,
    rho: float = 1.225,
    max_iterations: int = 400,
) -> str:
    """Compute a deterministic SHA-256 hash for the given simulation parameters."""
    payload = {
        "format": CASE_FORMAT_VERSION,
        "airfoil_name": airfoil.name,
        "chord_m": round(airfoil.chord_m, 6),
        "aoa_deg": round(airfoil.angle_of_attack_deg, 4),
        "ground_height_m": (
            round(airfoil.ground_height_m, 6) if airfoil.ground_height_m is not None else None
        ),
        "mesh_level": mesh_level.value,
        "speed_ms": round(speed_ms, 4),
        "nu": nu,
        "turbulence_intensity": turbulence_intensity,
        "rho": rho,
        "max_iterations": max_iterations,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _field(
    name: str,
    cls: str,
    dimensions: str,
    internal: str,
    boundary_field: str,
) -> str:
    return (
        foam_header(cls, name)
        + f"dimensions      {dimensions};\n"
        + f"internalField   {internal};\n"
        + "boundaryField\n{\n"
        + boundary_field
        + "}\n"
    )


def build_openfoam_case(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
    speed_ms: float = 50.0,
    base_dir: Path | None = None,
    nu: float = 1.5e-5,
    rho: float = 1.225,
    turbulence_intensity: float = 0.01,
    max_iterations: int = 400,
) -> tuple[Path, MeshStats]:
    """Generate a complete, deterministic, runnable OpenFOAM case directory.

    Returns ``(case_dir, estimated_mesh_stats)``.  The actual cell count and
    residuals are filled in by the runner after ``checkMesh`` and ``simpleFoam``.
    """
    case_hash = compute_case_hash(
        airfoil,
        mesh_level,
        speed_ms,
        nu=nu,
        turbulence_intensity=turbulence_intensity,
        rho=rho,
        max_iterations=max_iterations,
    )
    target_dir = (base_dir or DEFAULT_WORK_DIR) / f"case_{case_hash}"
    target_dir.mkdir(parents=True, exist_ok=True)

    profile = build_profile(airfoil)
    _, _, surf_level, _ = MESH_LEVELS[mesh_level]

    u_mag = float(speed_ms)
    k_inlet = 1.5 * (u_mag * turbulence_intensity) ** 2
    length_scale = 0.07 * airfoil.chord_m
    c_mu = 0.09
    omega_inlet = math.sqrt(k_inlet) / ((c_mu**0.25) * length_scale)
    nut_inlet = k_inlet / omega_inlet

    dir_0 = target_dir / "0"
    dir_constant = target_dir / "constant"
    dir_system = target_dir / "system"
    dir_tri = dir_constant / "triSurface"
    for d in (dir_0, dir_constant, dir_system, dir_tri):
        d.mkdir(exist_ok=True)

    has_ground = profile.ground_height_m is not None

    # 0/U -----------------------------------------------------------------
    ground_u = (
        "    ground\n    {\n        type movingWallVelocity;\n"
        "        value uniform (0 0 0);\n    }\n"
        if has_ground
        else "    ground\n    {\n        type slip;\n    }\n"
    )
    (dir_0 / "U").write_text(
        _field(
            "U",
            "volVectorField",
            "[0 1 -1 0 0 0 0]",
            "uniform (0 0 0)",
            f"""    inlet
    {{
        type fixedValue;
        value uniform ({u_mag:.6f} 0 0);
    }}
    outlet
    {{
        type zeroGradient;
    }}
    top
    {{
        type slip;
    }}
{ground_u}    airfoil
    {{
        type noSlip;
    }}
    frontAndBack
    {{
        type empty;
    }}
""",
        ),
        encoding="utf-8",
        newline="\n",
    )

    # 0/p -----------------------------------------------------------------
    (dir_0 / "p").write_text(
        _field(
            "p",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            "uniform 0",
            """    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    top         { type zeroGradient; }
    ground      { type zeroGradient; }
    airfoil     { type zeroGradient; }
    frontAndBack { type empty; }
""",
        ),
        encoding="utf-8",
        newline="\n",
    )

    # 0/k -----------------------------------------------------------------
    ground_k = (
        f"    ground      {{ type kqRWallFunction; value uniform {k_inlet:.6e}; }}\n"
        if has_ground
        else "    ground      { type zeroGradient; }\n"
    )
    (dir_0 / "k").write_text(
        _field(
            "k",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            f"uniform {k_inlet:.6e}",
            f"""    inlet       {{ type fixedValue; value uniform {k_inlet:.6e}; }}
    outlet      {{ type zeroGradient; }}
    top         {{ type zeroGradient; }}
{ground_k}    airfoil     {{ type kqRWallFunction; value uniform {k_inlet:.6e}; }}
    frontAndBack {{ type empty; }}
""",
        ),
        encoding="utf-8",
        newline="\n",
    )

    # 0/omega -------------------------------------------------------------
    ground_omega = (
        f"    ground      {{ type omegaWallFunction; value uniform {omega_inlet:.6e}; }}\n"
        if has_ground
        else "    ground      { type zeroGradient; }\n"
    )
    (dir_0 / "omega").write_text(
        _field(
            "omega",
            "volScalarField",
            "[0 0 -1 0 0 0 0]",
            f"uniform {omega_inlet:.6e}",
            f"""    inlet       {{ type fixedValue; value uniform {omega_inlet:.6e}; }}
    outlet      {{ type zeroGradient; }}
    top         {{ type zeroGradient; }}
{ground_omega}    airfoil     {{ type omegaWallFunction; value uniform {omega_inlet:.6e}; }}
    frontAndBack {{ type empty; }}
""",
        ),
        encoding="utf-8",
        newline="\n",
    )

    # 0/nut ---------------------------------------------------------------
    ground_nut = (
        "    ground      { type nutkWallFunction; value uniform 0; }\n"
        if has_ground
        else "    ground      { type calculated; value uniform 0; }\n"
    )
    (dir_0 / "nut").write_text(
        _field(
            "nut",
            "volScalarField",
            "[0 2 -1 0 0 0 0]",
            f"uniform {nut_inlet:.6e}",
            f"""    inlet       {{ type calculated; value uniform {nut_inlet:.6e}; }}
    outlet      {{ type calculated; value uniform {nut_inlet:.6e}; }}
    top         {{ type calculated; value uniform 0; }}
{ground_nut}    airfoil     {{ type nutkWallFunction; value uniform 0; }}
    frontAndBack {{ type empty; }}
""",
        ),
        encoding="utf-8",
        newline="\n",
    )

    # constant ------------------------------------------------------------
    (dir_constant / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties")
        + f"transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {nu:.6e};\n",
        encoding="utf-8",
        newline="\n",
    )
    (dir_constant / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties")
        + "simulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n",
        encoding="utf-8",
        newline="\n",
    )

    write_airfoil_stl(dir_tri / "airfoil.stl", profile)

    # system --------------------------------------------------------------
    (dir_system / "surfaceFeatureExtractDict").write_text(
        generate_surface_feature_dict(), encoding="utf-8", newline="\n"
    )
    (dir_system / "snappyHexMeshDict").write_text(
        generate_snappy_dict(airfoil, mesh_level), encoding="utf-8", newline="\n"
    )

    _, stats = generate_blockmesh_dict(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=speed_ms,
        chord_m=airfoil.chord_m,
        output_path=dir_system / "blockMeshDict",
        nu=nu,
    )

    # Force coefficients: positive Cl means downforce for the front wing and
    # positive lift for a free-air benchmark.
    lift_dir = "(0 -1 0)" if profile.is_front_wing else "(0 1 0)"
    cof_x = profile.x_le + 0.25 * (profile.x_te - profile.x_le)
    cof_y = profile.y_mean
    cof_z = 0.5 * (profile.z0 + profile.z1)

    (dir_system / "controlDict").write_text(
        foam_header("dictionary", "controlDict")
        + f"""application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {max_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {max_iterations};
purgeWrite      2;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (airfoil);
        rho             rhoInf;
        rhoInf          {rho};
        liftDir         {lift_dir};
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {u_mag:.6f};
        lRef            {airfoil.chord_m:.6f};
        Aref            {airfoil.chord_m * SPAN_M:.6f};
        CofR            ({cof_x:.6f} {cof_y:.6f} {cof_z:.6f});
    }}
}}
""",
        encoding="utf-8",
        newline="\n",
    )

    (dir_system / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes")
        + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U)      bounded Gauss limitedLinearV 1;
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""",
        encoding="utf-8",
        newline="\n",
    )

    (dir_system / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution")
        + """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }
    "(k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 4;
    consistent      yes;
    pRefCell        0;
    pRefPoint       (0 0 0);
    pRefValue       0;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        k               1e-4;
        omega           1e-4;
    }
}
relaxationFactors
{
    fields { p 0.2; }
    equations { U 0.5; k 0.6; omega 0.6; }
}
""",
        encoding="utf-8",
        newline="\n",
    )

    (target_dir / "run_case.sh").write_text(
        """#!/usr/bin/env bash
set -e
if [ -f /usr/share/openfoam/etc/bashrc ]; then
    export WM_PROJECT_DIR=/usr/share/openfoam
    export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
elif [ -d /opt/openfoam/etc ]; then
    . /opt/openfoam/etc/bashrc
else
    for f in /opt/openfoam*/etc/bashrc /usr/lib/openfoam/openfoam*/etc/bashrc; do
        if [ -f "$f" ]; then . "$f"; break; fi
    done
fi
blockMesh > log.blockMesh 2>&1
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
checkMesh > log.checkMesh 2>&1 || true
simpleFoam > log.simpleFoam 2>&1
""",
        encoding="utf-8",
        newline="\n",
    )

    params = {
        "case_hash": case_hash,
        "format": CASE_FORMAT_VERSION,
        "airfoil": airfoil.name,
        "designation": profile.designation,
        "inverted": profile.is_front_wing,
        "chord_m": airfoil.chord_m,
        "aoa_deg": airfoil.angle_of_attack_deg,
        "ground_height_m": airfoil.ground_height_m,
        "speed_ms": speed_ms,
        "reynolds": speed_ms * airfoil.chord_m / nu,
        "mesh_level": mesh_level.value,
        "surface_level": surf_level,
        "cells_estimated": stats.num_cells,
        "y_plus_estimated": stats.estimated_y_plus,
        "rho": rho,
        "nu": nu,
        "max_iterations": max_iterations,
        "domain": {
            "x_in": profile.x_in,
            "x_out": profile.x_out,
            "y_bottom": profile.y_bottom,
            "y_top": profile.y_top,
            "z_span": profile.z1 - profile.z0,
        },
    }
    (target_dir / "case_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8", newline="\n"
    )

    return target_dir, stats
