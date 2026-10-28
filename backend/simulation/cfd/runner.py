"""OpenFOAM CFD runner supporting native execution, WSL execution, and calibrated fallback."""

from __future__ import annotations

import functools
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from backend.simulation.cfd.case_builder import build_openfoam_case
from backend.simulation.cfd.models import (
    AirfoilGeometry,
    CFDResult,
    ConvergenceStatus,
    ConvergenceVerdict,
    MeshLevel,
)
from backend.simulation.cfd.parser import evaluate_convergence

SYNTHETIC_SOLVER_VERSION = "none (calibrated-synthetic-v1)"


@functools.lru_cache(maxsize=1)
def detect_cfd_backend() -> str:
    """Detect available CFD solver backend: 'native', 'wsl', or 'synthetic'."""
    if shutil.which("simpleFoam") and shutil.which("blockMesh"):
        return "native"
    if shutil.which("wsl"):
        try:
            # Check if simpleFoam exists inside default WSL distribution
            proc = subprocess.run(
                ["wsl", "bash", "-c", "which simpleFoam"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and "simpleFoam" in proc.stdout:
                return "wsl"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    return "synthetic"


def evaluate_calibrated_synthetic_cfd(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel,
    speed_ms: float,
    nu: float = 1.5e-5,
    rho: float = 1.225,
) -> tuple[float, float, float, ConvergenceVerdict]:
    """Deterministic calibrated surrogate for RANS-scale coefficients.

    This is NOT a CFD solve. It is a closed-form fit used only when no
    OpenFOAM binary is available, so that the surrogate pipeline and tests can
    run on a laptop. Every result carrying this source is tagged
    ``calibrated_synthetic`` and must not be reported as CFD.
    """
    alpha_deg = airfoil.angle_of_attack_deg
    re = speed_ms * airfoil.chord_m / nu

    # Discretization error delta per mesh level (Celik / Roache GCI model)
    mesh_error_map = {
        MeshLevel.COARSE: 0.015,
        MeshLevel.MEDIUM: 0.005,
        MeshLevel.FINE: 0.000,
    }
    grid_bias = mesh_error_map[mesh_level]

    # 1. Lift calculation with boundary layer decambering and stall
    # Linear slope a0 ≈ 0.1081 per degree (6.19 / rad for NACA 0012)
    a0 = 0.1081
    cl_linear = a0 * alpha_deg

    # Mild non-linear stall roll-off starting above 12 degrees
    if abs(alpha_deg) > 12.0:
        excess = abs(alpha_deg) - 12.0
        stall_factor = math.exp(-0.03 * excess**2)
        cl_base = cl_linear * stall_factor
    else:
        cl_base = cl_linear

    # 2. Ground effect augmentation (Zerihan & Zhang 2000)
    # Suction increases as h/c decreases relative to isolated airfoil (h/c = 1.0)
    ground_factor = 1.0
    if airfoil.ground_height_m is not None:
        hc = max(0.05, airfoil.ground_height_m / airfoil.chord_m)
        if hc <= 1.0:
            ground_factor = 1.0 + 0.185 * ((1.0 / hc) ** 0.58 - 1.0)
        else:
            ground_factor = 1.0

    cl = float((cl_base * ground_factor) + grid_bias)

    # 3. 2D Drag calculation (skin friction + form pressure drag)
    # Zero-lift profile drag matches NASA TM-4074 (Ladson 1988) at Re = 3.0M: Cd0 = 0.0082
    cd0_benchmark = 0.0082 * ((3e6 / max(re, 1e4)) ** 0.11)
    cd_pressure = 0.0055 * (cl_base**2)
    cd_stall = 0.0015 * (max(0.0, abs(alpha_deg) - 11.0) ** 2) if abs(alpha_deg) > 11.0 else 0.0
    cd = float(cd0_benchmark + cd_pressure + cd_stall + (grid_bias * 0.1))

    # 4. Pitching moment about quarter-chord
    cm = float(-0.005 - 0.002 * alpha_deg)

    verdict = ConvergenceVerdict(
        status=ConvergenceStatus.MOCK_EVALUATED,
        iterations_run=0,
        cl_mean=cl,
        cl_std=0.0,
        cd_mean=cd,
        cd_std=0.0,
        final_p_residual=0.0,
        final_u_residual=0.0,
        final_k_residual=0.0,
        final_omega_residual=0.0,
        is_converged=False,
    )

    return cl, cd, cm, verdict


def run_cfd_case(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
    speed_ms: float = 50.0,
    base_dir: Path | None = None,
    timeout_s: float = 300.0,
) -> CFDResult:
    """Execute complete CFD pipeline: build case, mesh, run solver, and parse results."""
    t0 = time.time()
    case_dir, mesh_stats = build_openfoam_case(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=speed_ms,
        base_dir=base_dir,
    )

    backend = detect_cfd_backend()

    if backend == "native":
        # Native OpenFOAM execution
        try:
            with open(case_dir / "log.blockMesh", "w") as f_out:
                subprocess.run(
                    ["blockMesh"], cwd=case_dir, stdout=f_out, stderr=subprocess.STDOUT, check=True
                )
            with open(case_dir / "log.simpleFoam", "w") as f_out:
                subprocess.run(
                    ["simpleFoam"],
                    cwd=case_dir,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    check=True,
                )
            verdict = evaluate_convergence(case_dir)
            cl = verdict.cl_mean
            cd = verdict.cd_mean
            cm = 0.0
            solver_used = "openfoam_native"
        except (subprocess.SubprocessError, FileNotFoundError):
            backend = "synthetic"

    elif backend == "wsl":
        # WSL OpenFOAM execution
        try:
            # Convert Windows path to WSL path (e.g. C:\... -> /mnt/c/...)
            wsl_path = (
                str(case_dir).replace("\\", "/").replace("C:", "/mnt/c").replace("c:", "/mnt/c")
            )
            wsl_cmd = f"cd '{wsl_path}' && blockMesh > log.blockMesh && simpleFoam > log.simpleFoam"
            subprocess.run(
                ["wsl", "bash", "-c", wsl_cmd],
                timeout=timeout_s,
                check=True,
                capture_output=True,
            )
            verdict = evaluate_convergence(case_dir)
            cl = verdict.cl_mean
            cd = verdict.cd_mean
            cm = 0.0
            solver_used = "openfoam_wsl"
        except (subprocess.SubprocessError, FileNotFoundError):
            backend = "synthetic"

    if backend == "synthetic":
        cl, cd, cm, verdict = evaluate_calibrated_synthetic_cfd(
            airfoil=airfoil,
            mesh_level=mesh_level,
            speed_ms=speed_ms,
        )

        # Do not impersonate OpenFOAM output. Write a clearly-labelled provenance
        # file instead of a fake log.simpleFoam / coefficient.dat.
        provenance = {
            "solver_used": "calibrated_synthetic",
            "solver_version": SYNTHETIC_SOLVER_VERSION,
            "is_synthetic": True,
            "is_converged": False,
            "convergence_status": verdict.status.value,
            "note": (
                "No CFD solver ran. These coefficients come from a closed-form "
                "calibration model. Do not cite as CFD."
            ),
        }
        (case_dir / "SYNTHETIC_MODEL.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        solver_used = "calibrated_synthetic"
    elif backend == "native":
        solver_used = "openfoam_native"
    else:
        solver_used = "openfoam_wsl"

    is_synthetic = solver_used == "calibrated_synthetic"
    solver_version = SYNTHETIC_SOLVER_VERSION if is_synthetic else "OpenFOAM (version not recorded)"

    # Compute dimensional forces (span b = 1.0 m, reference chord)
    q = 0.5 * 1.225 * (speed_ms**2)
    s_ref = airfoil.chord_m * 1.0  # 1m unit span
    downforce_N = cl * q * s_ref
    drag_N = cd * q * s_ref
    exec_time = time.time() - t0

    return CFDResult(
        case_id=case_dir.name,
        case_dir=str(case_dir),
        mesh_stats=mesh_stats,
        convergence=verdict,
        cl=cl,
        cd=cd,
        cm=cm,
        downforce_N=downforce_N,
        drag_N=drag_N,
        execution_time_s=exec_time,
        solver_used=solver_used,
        solver_version=solver_version,
        is_synthetic=is_synthetic,
    )
