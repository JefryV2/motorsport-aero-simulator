"""Deterministic OpenFOAM case builder for simpleFoam incompressible RANS simulations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from backend.simulation.cfd.mesh_builder import generate_blockmesh_dict
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel, MeshStats

DEFAULT_WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cfd_cases"


def compute_case_hash(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel,
    speed_ms: float,
    nu: float = 1.5e-5,
    turbulence_intensity: float = 0.01,
) -> str:
    """Compute a deterministic SHA-256 hash for the given simulation parameters."""
    payload = {
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
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_openfoam_case(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
    speed_ms: float = 50.0,
    base_dir: Path | None = None,
    nu: float = 1.5e-5,
    rho: float = 1.225,
    turbulence_intensity: float = 0.01,
    max_iterations: int = 500,
) -> tuple[Path, MeshStats]:
    """Generate a complete, deterministic, runnable OpenFOAM case directory.

    Returns (case_dir, mesh_stats).
    """
    case_hash = compute_case_hash(airfoil, mesh_level, speed_ms, nu, turbulence_intensity)
    target_dir = (base_dir or DEFAULT_WORK_DIR) / f"case_{case_hash}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Angles and velocities
    aoa_rad = math.radians(airfoil.angle_of_attack_deg)
    ux = speed_ms * math.cos(aoa_rad)
    uy = speed_ms * math.sin(aoa_rad)
    u_mag = speed_ms

    # Turbulence quantities for k-omega SST
    # k = 1.5 * (u * I)^2
    k_inlet = 1.5 * (u_mag * turbulence_intensity) ** 2
    # Turbulent length scale l ~ 0.07 * chord
    length_scale = 0.07 * airfoil.chord_m
    c_mu = 0.09
    omega_inlet = math.sqrt(k_inlet) / ((c_mu**0.25) * length_scale)
    nut_inlet = k_inlet / omega_inlet

    # Subdirectories
    dir_0 = target_dir / "0"
    dir_constant = target_dir / "constant"
    dir_system = target_dir / "system"
    for d in (dir_0, dir_constant, dir_system):
        d.mkdir(exist_ok=True)

    # 2. 0/U
    (dir_0 / "U").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({ux:.4f} {uy:.4f} 0.0);
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({ux:.4f} {uy:.4f} 0.0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            slip;
    }}
    ground
    {{
        type            {"noSlip" if airfoil.ground_height_m is not None else "slip"};
    }}
    airfoil
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 3. 0/p
    (dir_0 / "p").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class volScalarField; object p; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    top
    {{
        type            zeroGradient;
    }}
    ground
    {{
        type            zeroGradient;
    }}
    airfoil
    {{
        type            zeroGradient;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 4. 0/k
    (dir_0 / "k").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inlet:.6e};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_inlet:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            zeroGradient;
    }}
    ground
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6e};
    }}
    airfoil
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6e};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 5. 0/omega
    (dir_0 / "omega").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inlet:.6e};
boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.6e};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            zeroGradient;
    }}
    ground
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.6e};
    }}
    airfoil
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.6e};
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 6. 0/nut
    (dir_0 / "nut").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut_inlet:.6e};
boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform {nut_inlet:.6e};
    }}
    outlet
    {{
        type            calculated;
        value           uniform {nut_inlet:.6e};
    }}
    top
    {{
        type            calculated;
        value           uniform 0;
    }}
    ground
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    airfoil
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
""",
        encoding="utf-8",
    )

    # 7. constant/transportProperties & constant/turbulenceProperties
    (dir_constant / "transportProperties").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {nu:.6e};
rho             [1 -3 0 0 0 0 0] {rho:.4f};
""",
        encoding="utf-8",
    )

    (dir_constant / "turbulenceProperties").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""",
        encoding="utf-8",
    )

    # 8. system/controlDict
    (dir_system / "controlDict").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
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
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {u_mag:.4f};
        lRef            {airfoil.chord_m:.4f};
        Aref            {airfoil.chord_m * 0.1:.4f};
        CofR            (0.25 0 0);
    }}
}}
""",
        encoding="utf-8",
    )

    # 9. system/fvSchemes
    (dir_system / "fvSchemes").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         corrected;
}
""",
        encoding="utf-8",
    )

    # 10. system/fvSolution
    (dir_system / "fvSolution").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    "(k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
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
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
    }
    fields
    {
        p               0.3;
    }
}
""",
        encoding="utf-8",
    )

    # 11. system/blockMeshDict
    blockmesh_file = dir_system / "blockMeshDict"
    _, stats = generate_blockmesh_dict(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=speed_ms,
        chord_m=airfoil.chord_m,
        output_path=blockmesh_file,
    )

    # Save case provenance metadata
    meta = {
        "case_hash": case_hash,
        "airfoil": airfoil.name,
        "aoa_deg": airfoil.angle_of_attack_deg,
        "ground_height_m": airfoil.ground_height_m,
        "speed_ms": speed_ms,
        "mesh_level": mesh_level.value,
        "cells": stats.num_cells,
        "y_plus_estimated": stats.estimated_y_plus,
    }
    (target_dir / "case_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return target_dir, stats
