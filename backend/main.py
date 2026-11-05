"""FastAPI backend for the motorsport aero simulator."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ai.advisor import get_advice
from backend.models.aero_config import (
    AdvisorRequest,
    FormulaAeroConfig,
    OptimizationRequest,
    SimulationMethod,
    SimulationResult,
)
from backend.models.cfd3d import (
    ImportedCFDMeasureRequest,
    ImportedCFDProvenanceOut,
    ImportedCFDRequest,
    ImportedCFDResultOut,
    ImportedMeshMeasureOut,
    JobStatusOut,
)
from backend.models.lap import LapRequest, LapResponse
from backend.optimization.optimizer import optimize
from backend.simulation.analytical import simulate
from backend.simulation.cfd.openfoam_runner import is_openfoam_available, run_cfd
from backend.simulation.cfd.runner import CFDRunError, detect_cfd_backend, get_solver_version
from backend.simulation.cfd3d.models import ImportedCFDLevel
from backend.simulation.cfd3d.runner import ImportedCFDResult, run_imported_cfd, solver_available
from backend.simulation.cfd3d.surface import ImportedSurface, measure_imported_surface
from backend.simulation.lap.aero_map import AeroMap
from backend.simulation.lap.solver import LapSettings, solve_lap
from backend.simulation.lap.track import TrackError, build_track
from backend.simulation.lap.track_io import parse_centerline_csv
from backend.simulation.panel_method import simulate_panel
from backend.visualization.aero_field import compute_cp_field, force_vectors, generate_streamlines
from backend.visualization.mesh_io import get_model_path, save_upload

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aero_backend")

app = FastAPI(
    title="Motorsport Aero Simulator",
    description="Formula open-wheel aerodynamic simulation with AI optimization",
    version="0.1.0",
)

cors_origins_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()] or [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/api/health")
def health():
    backend = detect_cfd_backend()
    solver_version = get_solver_version()
    return {
        "status": "ok",
        "cfd_backend": backend,
        "openfoam_available": is_openfoam_available(),
        "solver_version": solver_version,
        "cfd_ready": backend in ("native", "wsl"),
        "methods": [
            SimulationMethod.ANALYTICAL.value,
            SimulationMethod.ANALYTICAL_ENHANCED.value,
            SimulationMethod.PANEL_VLM.value,
            SimulationMethod.CFD_STUB.value,
            SimulationMethod.CFD.value,
        ],
        "provenance": {
            "cfd_solver_installed": backend in ("native", "wsl"),
            "cfd_backend": backend,
            "solver_version": solver_version,
            "cfd_fallback": "calibrated_synthetic" if backend == "synthetic" else None,
            "note": (
                "No OpenFOAM solver detected. CFD-family results are a calibrated "
                "synthetic model, not a mesh solution."
                if backend == "synthetic"
                else (
                    "OpenFOAM detected. The front-wing section is solved with 2D "
                    "RANS; floor and rear-wing terms remain empirical, so the total "
                    "is a hybrid estimate, not a full-car CFD result."
                )
            ),
        },
    }


@app.post("/api/simulate", response_model=SimulationResult)
def api_simulate(
    config: FormulaAeroConfig,
    method: SimulationMethod = SimulationMethod.ANALYTICAL_ENHANCED,
):
    if method == SimulationMethod.ANALYTICAL:
        return simulate(config)
    if method in (SimulationMethod.ANALYTICAL_ENHANCED, SimulationMethod.PANEL_VLM):
        label = "panel_vlm" if method == SimulationMethod.PANEL_VLM else "analytical_enhanced"
        return simulate_panel(config, method=label)
    if method in (SimulationMethod.CFD_STUB, SimulationMethod.CFD):
        try:
            return run_cfd(config)
        except CFDRunError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "OpenFOAM is installed but the case failed. No synthetic "
                    f"fallback was substituted.\n{exc}"
                ),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive API boundary
            raise HTTPException(
                status_code=503,
                detail=(
                    "The CFD pipeline raised an unexpected error. No synthetic "
                    f"fallback was substituted.\n{type(exc).__name__}: {exc}"
                ),
            ) from exc
    raise HTTPException(
        400,
        f"Unknown method: {method}. Use analytical, analytical_enhanced, panel_vlm, cfd_stub, or cfd.",
    )


@app.post("/api/optimize")
def api_optimize(
    request: OptimizationRequest,
    method: SimulationMethod = SimulationMethod.ANALYTICAL_ENHANCED,
):
    use_panel = method != SimulationMethod.ANALYTICAL
    return optimize(request, use_panel=use_panel)


@app.post("/api/lap/simulate", response_model=LapResponse)
def api_lap_simulate(body: LapRequest):
    """Run the quasi-steady lap simulation on a user-supplied centreline.

    The track is never invented: callers must send coordinates in metres or a
    CSV centreline. CFD is rejected here because a single RANS point is not an
    aero map for a lap sweep.
    """
    try:
        if body.track.csv:
            points = parse_centerline_csv(body.track.csv)
        elif body.track.points:
            points = body.track.points
        else:
            raise TrackError("Provide track.points or track.csv")
        track = build_track(
            points, closed=body.track.closed, ds_m=body.track.ds_m
        )
    except TrackError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        aero = AeroMap(body.config, method=body.method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        vehicle = body.vehicle.to_vehicle()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = solve_lap(
        track,
        vehicle,
        aero,
        LapSettings(initial_speed_ms=body.initial_speed_ms),
    )
    return LapResponse(
        lap_time_s=result.lap_time_s,
        sector_times_s=result.sector_times_s,
        length_m=result.length_m,
        track_points=track.count,
        closed=track.closed,
        ds_m=result.ds_m,
        top_speed_kmh=result.top_speed_kmh,
        min_speed_kmh=result.min_speed_kmh,
        mean_speed_kmh=result.mean_speed_kmh,
        iterations=result.iterations,
        converged=result.converged,
        entry_speed_limited=result.entry_speed_limited,
        max_utilisation=result.max_utilisation,
        s_m=result.s_m.tolist(),
        speed_kmh=result.speed_kmh.tolist(),
        ay_ms2=result.ay_ms2.tolist(),
        ax_ms2=result.ax_ms2.tolist(),
        curvature_1pm=result.curvature_1pm.tolist(),
        downforce_N=result.downforce_N.tolist(),
        drag_N=result.drag_N.tolist(),
        aero_source=result.aero_source,
        aero_fidelity=result.aero_fidelity,
        track_source=result.track_source,
        provenance=result.provenance,
    )


@app.post("/api/advise")
async def api_advise(request: AdvisorRequest):
    return await get_advice(request)


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload STL/OBJ 3D car model for aero visualization."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50 MB)")
    try:
        meta = save_upload(file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    # Don't send full vertex arrays in upload response — client already has the file
    return {
        "model_id": meta["model_id"],
        "filename": meta["filename"],
        "format": meta["format"],
        "vertex_count": meta["vertex_count"],
        "face_count": meta["face_count"],
        "bounds": meta["bounds"],
    }


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    path = get_model_path(model_id)
    if not path:
        raise HTTPException(404, "Model not found")
    return FileResponse(path)


class AeroMapRequest(BaseModel):
    config: FormulaAeroConfig
    min_frh: float = 10.0
    max_frh: float = 50.0
    min_rrh: float = 20.0
    max_rrh: float = 80.0
    steps: int = 5


@app.post("/api/aero-map")
def api_aero_map(
    body: AeroMapRequest,
    method: SimulationMethod = SimulationMethod.ANALYTICAL_ENHANCED,
):
    import copy
    import math

    # A ride-height map is a 25-point sweep.  Never launch 25 real CFD cases
    # from one map click; use the fast VLM/empirical path for the map and keep
    # CFD as an explicit single-point verification.
    map_method = (
        SimulationMethod.ANALYTICAL_ENHANCED
        if method in (SimulationMethod.CFD, SimulationMethod.CFD_STUB)
        else method
    )
    sim_fn = simulate_panel if map_method != SimulationMethod.ANALYTICAL else simulate
    grid = []

    frh_vals = [
        body.min_frh + i * (body.max_frh - body.min_frh) / max(1, body.steps - 1)
        for i in range(body.steps)
    ]
    rrh_vals = [
        body.min_rrh + j * (body.max_rrh - body.min_rrh) / max(1, body.steps - 1)
        for j in range(body.steps)
    ]

    for frh in frh_vals:
        row = []
        for rrh in rrh_vals:
            cfg = copy.deepcopy(body.config)
            cfg.ride_height_front_mm = frh
            cfg.ride_height_rear_mm = rrh

            # Recalculate rake based on ride height change
            wheelbase_mm = 3.35 * 1000.0
            cfg.rake_deg = math.degrees(math.atan((rrh - frh) / wheelbase_mm))

            res = sim_fn(cfg)
            row.append(
                {
                    "frh": round(frh, 1),
                    "rrh": round(rrh, 1),
                    "downforce": round(res.forces.downforce_total_N, 1),
                    "drag": round(res.forces.drag_N, 1),
                    "balance": round(res.forces.balance_front_pct, 1),
                    "efficiency": round(res.forces.efficiency, 2),
                }
            )
        grid.append(row)

    return {
        "frh_vals": [round(v, 1) for v in frh_vals],
        "rrh_vals": [round(v, 1) for v in rrh_vals],
        "grid": grid,
    }


class AeroFieldRequest(BaseModel):
    config: FormulaAeroConfig
    vertices: list[list[float]]
    normals: list[list[float]] | None = None
    bounds: dict[str, list[float]] | None = None


class VisualizationRequest(BaseModel):
    config: FormulaAeroConfig
    bounds: dict[str, list[float]] | None = None


@app.post("/api/aero-field")
def api_aero_field(body: AeroFieldRequest):
    """Compute pressure coefficient field on mesh vertices."""
    return compute_cp_field(body.vertices, body.normals, body.config, body.bounds)


@app.post("/api/visualization")
def api_visualization(body: VisualizationRequest):
    """Full visualization payload: forces, streamlines, simulation."""
    return {
        "simulation": simulate_panel(body.config),
        "forces": force_vectors(body.config),
        "streamlines": generate_streamlines(body.config, body.bounds),
    }


@app.post("/api/simulate-and-optimize")
async def simulate_and_optimize(
    request: OptimizationRequest,
    method: SimulationMethod = SimulationMethod.ANALYTICAL_ENHANCED,
):
    """One-shot: simulate baseline, optimize, then get AI advice."""
    use_panel = method != SimulationMethod.ANALYTICAL
    sim_fn = simulate_panel if use_panel else simulate
    baseline = await asyncio.to_thread(sim_fn, request.config)

    opt_result = await asyncio.to_thread(optimize, request, use_panel=use_panel)
    optimized_sim = await asyncio.to_thread(sim_fn, opt_result.best_config)

    advisor_request = AdvisorRequest(
        config=request.config,
        simulation=baseline,
        optimization=opt_result,
    )
    advice = await get_advice(advisor_request)

    return {
        "baseline": baseline,
        "optimization": opt_result,
        "optimized_simulation": optimized_sim,
        "advice": advice,
    }


# --------------------------------------------------------------------------
# 3D CFD on uploaded meshes (snappyHexMesh tier)
# --------------------------------------------------------------------------

_JOB_TTL_S = 6 * 3600


@dataclass
class Cfd3DJob:
    """Registry record for one background imported-mesh solve."""

    job_id: str
    created_at: float
    status: Literal["queued", "running", "done", "error"] = "queued"
    finished_at: float | None = None
    detail: str | None = None
    result: ImportedCFDResult | None = None


_cfd3d_jobs: dict[str, Cfd3DJob] = {}


def _measure_imported(request: ImportedCFDMeasureRequest) -> ImportedSurface:
    """Measure an uploaded mesh, mapping failures to honest HTTP codes."""
    path = get_model_path(request.model_id)
    if not path:
        raise HTTPException(
            404, f"Unknown model_id '{request.model_id}' - upload the mesh first"
        )
    try:
        return measure_imported_surface(
            path.read_bytes(),
            path.suffix,
            axes=request.axes,
            scale_m_per_unit=request.scale_m_per_unit,
            align_ground=request.align_ground,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _cfd3d_result_out(result: ImportedCFDResult, speed_kmh: float) -> ImportedCFDResultOut:
    prov = result.provenance
    return ImportedCFDResultOut(
        drag_N=result.drag_N,
        downforce_N=result.downforce_N,
        lift_N=result.lift_N,
        cd=result.cd,
        cl=result.cl,
        q_dynamic_pa=result.q_dynamic_pa,
        speed_kmh=speed_kmh,
        speed_ms=result.speed_ms,
        frontal_area_m2=result.frontal_area_m2,
        reynolds=prov.reynolds,
        provenance=ImportedCFDProvenanceOut(
            backend=prov.backend,
            solver_used=prov.solver_used,
            solver_version=prov.solver_version,
            case_hash=prov.case_hash,
            case_format=prov.case_format,
            case_dir=prov.case_dir,
            frontal_area_m2=prov.frontal_area_m2,
            mesh_triangles=prov.mesh_triangles,
            watertight_edge_fraction=prov.watertight_edge_fraction,
            scale_m_per_unit=prov.scale_m_per_unit,
            axes=prov.axes,
            mesh_cells=prov.mesh_cells,
            max_iterations=prov.max_iterations,
            iterations=prov.iterations,
            convergence_status=prov.convergence_status,
            is_converged=prov.is_converged,
            cd=prov.cd_3d,
            cl=prov.cl_3d,
            cm=prov.cm_3d,
            cd_std=prov.cd_std,
            cl_std=prov.cl_std,
            residual_p=prov.residual_p,
            residual_u=prov.residual_u,
            residual_k=prov.residual_k,
            residual_omega=prov.residual_omega,
            y_plus_avg=prov.y_plus_avg,
            execution_time_s=prov.execution_time_s,
            warnings=list(prov.warnings),
        ),
    )


def _cfd3d_worker(job_id: str, surface: ImportedSurface, request: ImportedCFDRequest) -> None:
    job = _cfd3d_jobs[job_id]
    job.status = "running"
    try:
        result = run_imported_cfd(
            surface,
            request.speed_kmh / 3.6,
            mesh_level=ImportedCFDLevel(request.level),
            max_iterations=request.max_iterations,
        )
        job.result = result
        job.status = "done"
        job.finished_at = time.time()
    except CFDRunError as exc:
        job.status = "error"
        job.detail = str(exc)
        job.finished_at = time.time()
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        job.status = "error"
        job.detail = f"{type(exc).__name__}: {exc}"
        job.finished_at = time.time()


@app.post("/api/cfd/3d/measure", response_model=ImportedMeshMeasureOut)
def api_cfd3d_measure(request: ImportedCFDMeasureRequest):
    """Measure an uploaded mesh without solving anything."""
    surface = _measure_imported(request)
    return ImportedMeshMeasureOut(
        model_id=request.model_id,
        triangle_count=surface.source_triangle_count,
        frontal_area_m2=surface.frontal_area_m2,
        watertight_edge_fraction=surface.watertight_edge_fraction,
        extent_m={
            "x": surface.extent_x_m,
            "y": surface.extent_y_m,
            "z": surface.extent_z_m,
        },
        axes=surface.axes,
        scale_m_per_unit=surface.scale_m_per_unit,
        rasters_used=surface.rasters_used,
        warnings=list(surface.warnings),
    )


@app.post("/api/cfd/3d", response_model=JobStatusOut)
def api_cfd3d_start(request: ImportedCFDRequest):
    """Queue a background 3D RANS solve around the uploaded mesh.

    Refuses to run at all without a real solver: this tier has no synthetic
    fallback, so results are either a measured OpenFOAM solution or an error.
    """
    if not solver_available():
        raise HTTPException(
            503,
            "No OpenFOAM solver detected. The imported-mesh tier refuses to "
            "invent results - install OpenFOAM (README) or use the analytic/VLM "
            "tiers for the parametric car.",
        )
    surface = _measure_imported(request)

    now = time.time()
    for stale_id in [
        j for j, rec in _cfd3d_jobs.items() if now - rec.created_at > _JOB_TTL_S
    ]:
        _cfd3d_jobs.pop(stale_id, None)

    job_id = uuid.uuid4().hex[:12]
    _cfd3d_jobs[job_id] = Cfd3DJob(job_id=job_id, created_at=now)
    threading.Thread(target=_cfd3d_worker, args=(job_id, surface, request), daemon=True).start()
    return JobStatusOut(
        job_id=job_id, status="queued", created_at=now, detail=None, result=None
    )


@app.get("/api/cfd/3d/{job_id}", response_model=JobStatusOut)
def api_cfd3d_status(job_id: str):
    job = _cfd3d_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job '{job_id}'")
    result = job.result
    speed_kmh = 0.0 if result is None else result.speed_ms * 3.6
    return JobStatusOut(
        job_id=job_id,
        status=job.status,
        created_at=job.created_at,
        finished_at=job.finished_at,
        detail=job.detail,
        result=None if result is None else _cfd3d_result_out(result, speed_kmh),
    )


@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Motorsport Aero Simulator API. Open /docs for API explorer."}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
