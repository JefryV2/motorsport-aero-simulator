"""FastAPI backend for the motorsport aero simulator."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

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
from backend.optimization.optimizer import optimize
from backend.simulation.analytical import simulate
from backend.simulation.cfd.openfoam_runner import is_openfoam_available, run_cfd
from backend.simulation.cfd.runner import CFDRunError, detect_cfd_backend, get_solver_version
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


@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Motorsport Aero Simulator API. Open /docs for API explorer."}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
