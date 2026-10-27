"""Parser for OpenFOAM force coefficients, residuals, and convergence verdicts."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from backend.simulation.cfd.models import ConvergenceStatus, ConvergenceVerdict


def parse_force_coeffs(
    case_dir: Path,
    last_n_iterations: int = 50,
) -> tuple[float, float, float, float, float, float, int]:
    """Parse postProcessing/forceCoeffs file for Cl, Cd, Cm histories.

    Returns (cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, total_iterations).
    """
    post_dir = case_dir / "postProcessing" / "forceCoeffs"
    if not post_dir.exists():
        # Check subdirectories
        candidates = list(case_dir.glob("postProcessing/**/coefficient*.dat")) + list(
            case_dir.glob("postProcessing/**/forceCoeffs*.dat")
        )
    else:
        candidates = list(post_dir.glob("*/coefficient*.dat")) + list(
            post_dir.glob("*/forceCoeffs*.dat")
        )

    if not candidates:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    target_file = candidates[0]
    lines = target_file.read_text(encoding="utf-8").splitlines()

    times: list[float] = []
    cms: list[float] = []
    cds: list[float] = []
    cls: list[float] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                t = float(parts[0])
                cm = float(parts[1])
                cd = float(parts[2])
                cl = float(parts[3])
                times.append(t)
                cms.append(cm)
                cds.append(cd)
                cls.append(cl)
            except ValueError:
                continue

    total_iters = len(times)
    if total_iters == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    sample_size = min(total_iters, last_n_iterations)
    cl_window = cls[-sample_size:]
    cd_window = cds[-sample_size:]
    cm_window = cms[-sample_size:]

    cl_mean = float(np.mean(cl_window))
    cl_std = float(np.std(cl_window))
    cd_mean = float(np.mean(cd_window))
    cd_std = float(np.std(cd_window))
    cm_mean = float(np.mean(cm_window))
    cm_std = float(np.std(cm_window))

    return cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, total_iters


def parse_residuals(log_file: Path) -> dict[str, float]:
    """Parse log.simpleFoam for the final initial residuals of p, Ux, Uy, k, omega."""
    if not log_file.exists():
        return {"p": 1.0, "Ux": 1.0, "Uy": 1.0, "k": 1.0, "omega": 1.0}

    content = log_file.read_text(encoding="utf-8", errors="ignore")
    residuals: dict[str, float] = {}

    for var in ("p", "Ux", "Uy", "k", "omega"):
        matches = re.findall(rf"Solving for {var}, Initial residual = ([\d\.\+eE\-]+)", content)
        if matches:
            try:
                residuals[var] = float(matches[-1])
            except ValueError:
                residuals[var] = 1.0
        else:
            residuals[var] = 1.0

    return residuals


def evaluate_convergence(
    case_dir: Path,
    std_threshold: float = 0.005,
    residual_threshold: float = 1e-4,
    last_n_iterations: int = 50,
) -> ConvergenceVerdict:
    """Evaluate convergence from force coefficient history and solver log."""
    log_file = case_dir / "log.simpleFoam"
    cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, total_iters = parse_force_coeffs(
        case_dir, last_n_iterations=last_n_iterations
    )
    residuals = parse_residuals(log_file)

    p_res = residuals.get("p", 1.0)
    u_res = max(residuals.get("Ux", 1.0), residuals.get("Uy", 1.0))
    k_res = residuals.get("k", 1.0)
    omega_res = residuals.get("omega", 1.0)

    # Check for divergence (NaNs or exploding values)
    if (
        math.isnan(cl_mean)
        or math.isnan(cd_mean)
        or abs(cl_mean) > 10.0
        or cd_mean > 5.0
        or p_res > 1e2
    ):
        status = ConvergenceStatus.DIVERGED
        is_converged = False
    elif total_iters == 0:
        status = ConvergenceStatus.FAILED
        is_converged = False
    elif (
        cl_std <= std_threshold
        and cd_std <= std_threshold
        and (p_res <= residual_threshold or total_iters >= 200)
    ):
        status = ConvergenceStatus.CONVERGED
        is_converged = True
    else:
        status = ConvergenceStatus.UNCONVERGED
        is_converged = False

    return ConvergenceVerdict(
        status=status,
        iterations_run=total_iters,
        cl_mean=cl_mean,
        cl_std=cl_std,
        cd_mean=cd_mean,
        cd_std=cd_std,
        final_p_residual=p_res,
        final_u_residual=u_res,
        final_k_residual=k_res,
        final_omega_residual=omega_res,
        is_converged=is_converged,
    )
