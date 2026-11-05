"""Pydantic models for the imported-mesh 3D CFD tier.

The HTTP contract mirrors the honesty rules of the rest of the API:

* the mesh is the file the user uploaded — the built-in parametric car is
  never involved;
* a solve without OpenFOAM is refused (HTTP 503), never simulated;
* every result carries a provenance block describing what really ran.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AxesConvention = Literal["three", "solver"]
Cfd3dLevel = Literal["coarse", "medium", "fine"]
JobState = Literal["queued", "running", "done", "error"]


class ImportedCFDMeasureRequest(BaseModel):
    """Ask what the backend measured in an uploaded mesh (no solver)."""

    model_id: str
    axes: AxesConvention = "three"
    scale_m_per_unit: float | None = Field(
        default=None,
        gt=0.0,
        description="Metres per file unit. None = the file is authored in metres.",
    )
    align_ground: bool = True
    ground_clearance_m: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Metres of tunnel floor below the mesh's lowest point. 0.0 rests the "
            "floor on it, which is right for a car standing on its tyres and wrong "
            "for a benchmark body that floats (the Ahmed body sits 0.1 m off the "
            "ground plane)."
        ),
    )


class ImportedCFDRequest(ImportedCFDMeasureRequest):
    """Start a background 3D RANS solve around an uploaded mesh."""

    speed_kmh: float = Field(gt=0.0, le=500.0)
    level: Cfd3dLevel = "medium"
    max_iterations: int = Field(default=1500, ge=100, le=20_000)


class ImportedCFDProvenanceOut(BaseModel):
    """What actually produced a 3D result, exposed over HTTP."""

    backend: str
    solver_used: str
    solver_version: str
    case_hash: str
    case_format: str
    case_dir: str
    frontal_area_m2: float
    mesh_triangles: int
    watertight_edge_fraction: float
    scale_m_per_unit: float
    axes: str
    mesh_cells: int
    max_iterations: int
    iterations: int
    convergence_status: str
    is_converged: bool
    cd: float
    cl: float
    cm: float
    cd_std: float
    cl_std: float
    residual_p: float
    residual_u: float
    residual_k: float
    residual_omega: float
    y_plus_avg: float | None = None
    execution_time_s: float
    floor_y_m: float = 0.0
    ground_clearance_m: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ImportedCFDResultOut(BaseModel):
    """Forces on the uploaded mesh. Newtons, from a real solver only."""

    drag_N: float
    downforce_N: float
    lift_N: float
    cd: float
    cl: float
    q_dynamic_pa: float
    speed_kmh: float
    speed_ms: float
    frontal_area_m2: float
    reynolds: float
    provenance: ImportedCFDProvenanceOut


class ImportedMeshMeasureOut(BaseModel):
    """Measurement-only preview of an uploaded mesh."""

    model_id: str
    triangle_count: int
    frontal_area_m2: float
    watertight_edge_fraction: float
    extent_m: dict[str, float]
    floor_y_m: float = 0.0
    ground_clearance_m: float = 0.0
    axes: str
    scale_m_per_unit: float
    rasters_used: int
    warnings: list[str] = Field(default_factory=list)


class JobStatusOut(BaseModel):
    """Polling response for a 3D CFD job."""

    job_id: str
    status: JobState
    created_at: float
    finished_at: float | None = None
    detail: str | None = None
    result: ImportedCFDResultOut | None = None
