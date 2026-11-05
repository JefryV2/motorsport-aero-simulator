"""Deterministic OpenFOAM case builder for imported 3D meshes (snappyHexMesh).

The case is a real, runnable wind-tunnel case around the mesh the user
uploaded, nothing else:

* a background block-mesh box defines the tunnel (patches inlet / outlet /
  ground / top / sides);
* the imported triangles are written to ``constant/triSurface/car.stl`` and
  are what ``snappyHexMesh`` snaps a hex-dominant mesh onto;
* ``simpleFoam`` solves steady k-omega SST with a ``forceCoeffs`` function
  object on the ``car`` walls only, like the 2D tier;
* ``checkMesh`` records the measured cell count and mesh quality.

The builder never writes a fake solver log. Solver metadata is written by the
runner only after a real solve.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.simulation.cfd.case_builder import foam_header
from backend.simulation.cfd3d.models import (
    INLET_MARGIN_LENGTHS,
    LATERAL_CELL_SHRINK,
    LEVELS,
    OUTLET_MARGIN_LENGTHS,
    SIDE_MARGIN_LENGTHS,
    TOP_MARGIN_LENGTHS,
    ImportedCFDLevel,
)
from backend.simulation.cfd3d.surface import ImportedSurface, write_solver_stl

DEFAULT_WORK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cfd_cases_3d"
CASE3D_FORMAT_VERSION = "imported-mesh-rans-v1-snappy"

RHO = 1.225
NU = 1.5e-5


@dataclass(frozen=True)
class Case3DBuild:
    """What the builder produced, before any solver ran."""

    case_dir: Path
    case_hash: str
    background_cells: tuple[int, int, int]
    estimated_background_cells: int
    frontal_area_m2: float
    speed_ms: float
    max_iterations: int


def background_box(
    surface: ImportedSurface,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Tunnel box corners ``(min, max)`` sized from the mesh's own extents.

    Length margins are measured in car-lengths ahead/behind, so a brick and a
    limousine both get proportional run-up and wake space.
    """
    length = surface.extent_x_m
    zmin, zmax = float(surface.vertices[:, 2].min()), float(surface.vertices[:, 2].max())
    pad = (zmax - zmin) * (SIDE_MARGIN_LENGTHS - 1.0) * 0.5
    # Measured from the ground plane, not from the belly: a body with a real
    # clearance under it keeps that gap, and still gets the same headroom.
    floor_y = surface.floor_y_m
    roof_span = float(surface.vertices[:, 1].max()) - floor_y

    box_min = (
        float(surface.vertices[:, 0].min()) - INLET_MARGIN_LENGTHS * length,
        floor_y,
        zmin - pad,
    )
    box_max = (
        float(surface.vertices[:, 0].max()) + OUTLET_MARGIN_LENGTHS * length,
        floor_y + TOP_MARGIN_LENGTHS * roof_span,
        zmax + pad,
    )
    return box_min, box_max


def background_cells(
    surface: ImportedSurface,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    cell_share: float,
) -> tuple[int, int, int]:
    """Cell counts per axis so the background cell is a fixed share of the length."""
    cell_m = cell_share * surface.extent_x_m
    counts = []
    for lo, hi, axis in zip(box_min, box_max, range(3), strict=True):
        span = hi - lo
        # keep refinement from exploding on thin axes: cap the lateral/vertical
        # counts at 40 so a thin wing alone cannot make a million-cell box.
        counts.append(
            int(np.clip(round(span / cell_m / (LATERAL_CELL_SHRINK if axis == 2 else 1.0)), 12, 40))
        )
    return (counts[0], counts[1], counts[2])


def _block_mesh_dict(
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    n: tuple[int, int, int],
) -> str:
    """The background hex box, vertices 0-7 with outward-facing patch faces."""
    return (
        foam_header("dictionary", "blockMeshDict")
        + "convertToMeters 1;\n\nvertices\n(\n"
        + "\n".join(
            f"    ({x:.6f} {y:.6f} {z:.6f})"
            for x, y, z in (
                (box_min[0], box_min[1], box_min[2]),
                (box_max[0], box_min[1], box_min[2]),
                (box_max[0], box_max[1], box_min[2]),
                (box_min[0], box_max[1], box_min[2]),
                (box_min[0], box_min[1], box_max[2]),
                (box_max[0], box_min[1], box_max[2]),
                (box_max[0], box_max[1], box_max[2]),
                (box_min[0], box_max[1], box_max[2]),
            )
        )
        + "\n);\n\n"
        + "blocks\n(\n    hex (0 1 2 3 4 5 6 7) "
        + f"({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)\n);\n\n"
        + """boundary
(
    inlet
    {
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (1 5 6 2)
        );
    }
    ground
    {
        type wall;
        faces
        (
            (0 1 5 4)
        );
    }
    top
    {
        type patch;
        faces
        (
            (2 6 7 3)
        );
    }
    sides
    {
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
"""
    )


def snappy_dict(
    location_in_mesh: tuple[float, float, float],
    layers: int,
    surface_levels: int,
) -> str:
    """Background-to-snapped hex mesh instructions around the uploaded surface."""
    return foam_header("dictionary", "snappyHexMeshDict") + f"""castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    car
    {{
        type            triSurfaceMesh;
        file            "car.stl";
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      4000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 2;

    features            ();

    refinementSurfaces
    {{
        car
        {{
            level ({surface_levels} {surface_levels});
            patchInfo {{ type wall; }}
        }}
    }}

    refinementRegions
    {{
    }}

    resolveFeatureAngle 60;

    locationInMesh      ({location_in_mesh[0]:.6f} {location_in_mesh[1]:.6f} {location_in_mesh[2]:.6f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 0;
}}

addLayersControls
{{
    relativeSizes   true;
    featureAngle    60;
    slipFeatureAngle 30;

    layers
    {{
        car
        {{
            nSurfaceLayers {layers};
        }}
    }}
    expansionRatio   1.3;
    finalLayerThickness 0.35;
    minThickness     0.08;

    nLayerIter                50;
    nRelaxedIter              20;
    nRelaxIter                5;

    nGrow                     0;

    nSmoothSurfaceNormals     1;
    nSmoothNormals            3;
    nSmoothThickness          10;
    maxFaceThicknessRatio     0.5;
    maxThicknessToMedialRatio 0.3;

    minMedialAxisAngle        90;
    nMedialAxisIter           10;

    nBufferCellsNoExtrude     0;
    additionalReporting       false;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;

    relaxed
    {{
        maxNonOrtho 75;
    }}

    nSmoothScale        4;
    errorReduction      0.75;
}}

mergeTolerance  1e-6;
"""


def _field(
    name: str,
    cls: str,
    dimensions: str,
    internal: str,
    boundary_field: str,
) -> str:
    return foam_header(cls, name) + (
        f"dimensions      {dimensions};\n"
        + f"internalField   {internal};\n"
        + "boundaryField\n{\n"
        + boundary_field
        + "}\n"
    )


def _fields(
    case_dir: Path,
    u_mag: float,
    k_inlet: float,
    omega_inlet: float,
    car_patch: str = "car",
) -> None:
    dir_0 = case_dir / "0"
    dir_0.mkdir(parents=True, exist_ok=True)
    outer_u = (
        f"""    inlet
    {{
        type            freestream;
        freestreamValue uniform ({u_mag:.6f} 0 0);
    }}
    outlet
    {{
        type            freestream;
        freestreamValue uniform ({u_mag:.6f} 0 0);
    }}
    top
    {{
        type            freestream;
        freestreamValue uniform ({u_mag:.6f} 0 0);
    }}
    sides
    {{
        type            freestream;
        freestreamValue uniform ({u_mag:.6f} 0 0);
    }}
"""
    )
    (dir_0 / "U").write_text(
        _field(
            "U",
            "volVectorField",
            "[0 1 -1 0 0 0 0]",
            "uniform (0 0 0)",
            outer_u
            + f"""    ground
    {{
        type            movingWallVelocity;
        value           uniform ({u_mag:.6f} 0 0);
    }}
    {car_patch}
    {{
        type            noSlip;
    }}
""",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (dir_0 / "p").write_text(
        _field(
            "p",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            "uniform 0",
            """    inlet    { type freestreamPressure; freestreamValue uniform 0; }
    outlet   { type freestreamPressure; freestreamValue uniform 0; }
    top      { type freestreamPressure; freestreamValue uniform 0; }
    sides    { type freestreamPressure; freestreamValue uniform 0; }
    ground   { type fixedFluxPressure; value uniform 0; }
"""
            + f"    {car_patch}   {{ type zeroGradient; }}\n",
        ),
        encoding="utf-8",
        newline="\n",
    )
    turb_wall_k = f"    {car_patch}   {{ type kqRWallFunction; value uniform {k_inlet:.6e}; }}\n"
    (dir_0 / "k").write_text(
        _field(
            "k",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            f"uniform {k_inlet:.6e}",
            f"""    inlet    {{ type inletOutlet; inletValue uniform {k_inlet:.6e}; value uniform {k_inlet:.6e}; }}
    outlet   {{ type inletOutlet; inletValue uniform {k_inlet:.6e}; value uniform {k_inlet:.6e}; }}
    top      {{ type inletOutlet; inletValue uniform {k_inlet:.6e}; value uniform {k_inlet:.6e}; }}
    sides    {{ type inletOutlet; inletValue uniform {k_inlet:.6e}; value uniform {k_inlet:.6e}; }}
    ground   {{ type kqRWallFunction; value uniform {k_inlet:.6e}; }}
"""
            + turb_wall_k,
        ),
        encoding="utf-8",
        newline="\n",
    )
    turb_wall_omega = (
        f"    {car_patch}   {{ type omegaWallFunction; value uniform {omega_inlet:.6e}; }}\n"
    )
    (dir_0 / "omega").write_text(
        _field(
            "omega",
            "volScalarField",
            "[0 0 -1 0 0 0 0]",
            f"uniform {omega_inlet:.6e}",
            f"""    inlet    {{ type inletOutlet; inletValue uniform {omega_inlet:.6e}; value uniform {omega_inlet:.6e}; }}
    outlet   {{ type inletOutlet; inletValue uniform {omega_inlet:.6e}; value uniform {omega_inlet:.6e}; }}
    top      {{ type inletOutlet; inletValue uniform {omega_inlet:.6e}; value uniform {omega_inlet:.6e}; }}
    sides    {{ type inletOutlet; inletValue uniform {omega_inlet:.6e}; value uniform {omega_inlet:.6e}; }}
    ground   {{ type omegaWallFunction; value uniform {omega_inlet:.6e}; }}
"""
            + turb_wall_omega,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (dir_0 / "nut").write_text(
        _field(
            "nut",
            "volScalarField",
            "[0 2 -1 0 0 0 0]",
            "uniform 0",
            """    inlet    { type calculated; value uniform 0; }
    outlet   { type calculated; value uniform 0; }
    top      { type calculated; value uniform 0; }
    sides    { type calculated; value uniform 0; }
    ground   { type nutkWallFunction; value uniform 0; }
"""
            + f"    {car_patch}   {{ type nutkWallFunction; value uniform 0; }}\n",
        ),
        encoding="utf-8",
        newline="\n",
    )


def _fv_schemes() -> str:
    return foam_header("dictionary", "fvSchemes") + """ddtSchemes { default steadyState; }
gradSchemes { default cellLimited Gauss linear 1; }
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
"""


def _fv_solution(max_iterations: int) -> str:
    return foam_header("dictionary", "fvSolution") + """solvers
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
        smoother        symGaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }
    "(k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl
    {
        p               1e-5;
        U               1e-5;
        k               1e-5;
        omega           1e-5;
    }
}
relaxationFactors
{
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
        "(k|omega)"     0.7;
    }
    fields
    {
        p               0.3;
    }
}
cache
{
    grad(U);
}
"""


RUN_CASE_SH = """#!/usr/bin/env bash
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
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
checkMesh > log.checkMesh 2>&1
simpleFoam > log.simpleFoam 2>&1
"""


def build_imported_case(
    surface: ImportedSurface,
    *,
    speed_ms: float,
    mesh_level: ImportedCFDLevel = ImportedCFDLevel.MEDIUM,
    max_iterations: int = 1500,
    base_dir: Path | None = None,
) -> Case3DBuild:
    """Write a complete, deterministic case and return the build record.

    The case directory is keyed by hash of the triangles plus run parameters,
    so re-running the same import is free.
    """
    surface_hash = hashlib.sha256(
        np.ascontiguousarray(surface.vertices).tobytes()
        + np.ascontiguousarray(surface.faces).tobytes()
    ).digest()
    # A lowered floor is a different case, but the default floor (resting on the
    # mesh's lowest point) keeps the historical payload byte for byte, so cases
    # already solved on disk stay reusable.
    params_tag = f"{speed_ms:.4f}|{max_iterations}|{mesh_level.value}"
    if surface.ground_clearance_m != 0.0:
        params_tag += f"|fy{surface.floor_y_m:.6f}"
    payload = (
        CASE3D_FORMAT_VERSION.encode()
        + b"|"
        + params_tag.encode()
        + b"|"
        + surface_hash
    )
    case_hash = hashlib.sha256(payload).hexdigest()[:16]

    target_dir = (base_dir or DEFAULT_WORK_DIR) / f"case3d_{case_hash}"
    target_dir.mkdir(parents=True, exist_ok=True)

    box_min, box_max = background_box(surface)
    level_params = LEVELS[mesh_level]
    n = background_cells(surface, box_min, box_max, level_params.cell_share)
    estimated = int(np.prod(n))
    u_mag = float(speed_ms)
    k_inlet = 1.5 * (u_mag * 0.01) ** 2
    length_scale = 0.07 * max(surface.extent_x_m, 0.2)
    omega_inlet = math.sqrt(k_inlet) / ((0.09**0.25) * length_scale)

    (target_dir / "system").mkdir(parents=True, exist_ok=True)
    (target_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    write_solver_stl(surface, target_dir / "constant" / "triSurface" / "car.stl")

    (target_dir / "system" / "blockMeshDict").write_text(
        _block_mesh_dict(box_min, box_max, n), encoding="utf-8", newline="\n"
    )

    # A point inside the box, outside the car: top of the box at the inlet end.
    loc = (box_min[0] + 0.05 * (box_max[0] - box_min[0]), box_max[1] * 0.95, box_max[2] * 0.95)
    (target_dir / "system" / "snappyHexMeshDict").write_text(
        snappy_dict(
            loc,
            layers=level_params.n_layers,
            surface_levels=level_params.surface_levels,
        ),
        encoding="utf-8",
        newline="\n",
    )

    (target_dir / "constant" / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties")
        + f"transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {NU:.6e};\n",
        encoding="utf-8",
        newline="\n",
    )
    (target_dir / "constant" / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties")
        + "simulationType  RAS;\nRAS\n{\n    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n",
        encoding="utf-8",
        newline="\n",
    )

    _fields(target_dir, u_mag, k_inlet, omega_inlet)

    frontal = float(surface.frontal_area_m2)
    (target_dir / "system" / "controlDict").write_text(
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
        patches         (car);
        rho             rhoInf;
        rhoInf          {RHO};
        liftDir         (0 -1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 -1);
        magUInf         {u_mag:.6f};
        lRef            {surface.extent_x_m:.6f};
        Aref            {frontal:.6f};
        CofR            ({0.5 * (box_min[0] + box_max[0]):.6f} {0.5 * (box_min[1] + box_max[1]):.6f} {0.5 * (box_min[2] + box_max[2]):.6f});
    }}
    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   100;
        patches         (car);
    }}
}}
""",
        encoding="utf-8",
        newline="\n",
    )
    (target_dir / "system" / "fvSchemes").write_text(
        _fv_schemes(), encoding="utf-8", newline="\n"
    )
    (target_dir / "system" / "fvSolution").write_text(
        _fv_solution(max_iterations), encoding="utf-8", newline="\n"
    )
    (target_dir / "run_case.sh").write_text(
        RUN_CASE_SH, encoding="utf-8", newline="\n"
    )

    params = {
        "case_hash": case_hash,
        "format": CASE3D_FORMAT_VERSION,
        "source_triangles": surface.source_triangle_count,
        "scale_m_per_unit": surface.scale_m_per_unit,
        "axes": surface.axes,
        "frontal_area_m2": frontal,
        "watertight_edge_fraction": surface.watertight_edge_fraction,
        "extent_m": {
            "x": surface.extent_x_m,
            "y": surface.extent_y_m,
            "z": surface.extent_z_m,
        },
        "floor_y_m": surface.floor_y_m,
        "ground_clearance_m": surface.ground_clearance_m,
        "speed_ms": speed_ms,
        "reynolds_length": surface.extent_x_m,
        "reynolds": speed_ms * surface.extent_x_m / NU,
        "mesh_level": mesh_level.value,
        "surface_levels": level_params.surface_levels,
        "layers": level_params.n_layers,
        "background_cells": list(n),
        "background_cells_estimated": estimated,
        "cl_sign_convention": "positive Cl is downforce (liftDir (0 -1 0))",
        "box_m": {"min": list(box_min), "max": list(box_max)},
        "warnings": list(surface.warnings),
        "max_iterations": max_iterations,
    }
    (target_dir / "case_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8", newline="\n"
    )

    return Case3DBuild(
        case_dir=target_dir,
        case_hash=case_hash,
        background_cells=n,
        estimated_background_cells=estimated,
        frontal_area_m2=frontal,
        speed_ms=speed_ms,
        max_iterations=max_iterations,
    )
