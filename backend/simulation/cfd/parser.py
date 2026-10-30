"""Parser for OpenFOAM force coefficients, residuals, mesh quality, and verdicts."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from backend.simulation.cfd.models import ConvergenceStatus, ConvergenceVerdict


def _coefficient_columns(lines: list[str]) -> list[str] | None:
    """Return the column names from the ``forceCoeffs`` header, if present."""
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9()]*", line)
        if "Time" in tokens and ("Cd" in tokens or "Cl" in tokens):
            return tokens
    return None


def parse_force_coeffs(
    case_dir: Path,
    last_n_iterations: int = 50,
) -> tuple[float, float, float, float, float, float, int]:
    """Parse ``postProcessing/forceCoeffs`` for Cl, Cd and Cm histories.

    Returns ``(cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, n_rows)``.
    The mean and standard deviation are taken over the last ``last_n_iterations``
    rows, which is what the convergence check uses.
    """
    post_dir = case_dir / "postProcessing" / "forceCoeffs"
    if post_dir.exists():
        candidates = list(post_dir.glob("*/coefficient*.dat")) + list(
            post_dir.glob("*/forceCoeffs*.dat")
        )
    else:
        candidates = list(case_dir.glob("postProcessing/**/coefficient*.dat")) + list(
            case_dir.glob("postProcessing/**/forceCoeffs*.dat")
        )

    if not candidates:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    target_file = sorted(candidates)[-1]
    lines = target_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    columns = _coefficient_columns(lines)

    times: list[float] = []
    cms: list[float] = []
    cds: list[float] = []
    cls: list[float] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            t = float(parts[0])
            if columns is not None:
                lookup = {name: i for i, name in enumerate(columns)}
                cd = float(parts[lookup["Cd"]])
                cl = float(parts[lookup["Cl"]])
                cm_index = lookup.get("CmPitch", lookup.get("Cm"))
                cm = float(parts[cm_index]) if cm_index is not None else 0.0
            elif len(parts) >= 6:
                # Standard OpenFOAM order: Time Cd Cs Cl CmRoll CmPitch ...
                cd = float(parts[1])
                cl = float(parts[3])
                cm = float(parts[5])
            elif len(parts) == 4:
                # Legacy custom order: Time Cm Cd Cl
                cm = float(parts[1])
                cd = float(parts[2])
                cl = float(parts[3])
            else:
                continue
        except (ValueError, IndexError, KeyError):
            continue

        if not (math.isfinite(t) and math.isfinite(cd) and math.isfinite(cl)):
            continue
        times.append(t)
        cms.append(cm if math.isfinite(cm) else 0.0)
        cds.append(cd)
        cls.append(cl)

    total_iters = len(times)
    if total_iters == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    sample_size = min(total_iters, max(1, last_n_iterations))
    cl_mean = float(np.mean(cls[-sample_size:]))
    cl_std = float(np.std(cls[-sample_size:]))
    cd_mean = float(np.mean(cds[-sample_size:]))
    cd_std = float(np.std(cds[-sample_size:]))
    cm_mean = float(np.mean(cms[-sample_size:]))
    cm_std = float(np.std(cms[-sample_size:]))

    return cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, total_iters


def parse_residuals(log_file: Path) -> dict[str, float]:
    """Parse ``log.simpleFoam`` for the final initial residuals of p, U, k, omega."""
    if not log_file.exists():
        return {"p": 1.0, "Ux": 1.0, "Uy": 1.0, "k": 1.0, "omega": 1.0}

    content = log_file.read_text(encoding="utf-8", errors="ignore")
    residuals: dict[str, float] = {}

    for var in ("p", "Ux", "Uy", "k", "omega"):
        matches = re.findall(rf"Solving for {var}, Initial residual = ([\d.+\-eE]+)", content)
        if matches:
            try:
                residuals[var] = float(matches[-1])
            except ValueError:
                residuals[var] = 1.0
        else:
            residuals[var] = 1.0

    return residuals


def parse_mesh_stats(case_dir: Path) -> dict[str, float | int]:
    """Read cell/point counts and mesh-quality flags from the real solver logs."""
    cells = 0
    points = 0
    max_non_ortho = 0.0
    severe_non_ortho = 0
    failed_checks = 0

    for name in ("log.checkMesh",):
        path = case_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not cells:
            match = re.search(r"cells:\s*(\d+)", text)
            if match:
                cells = int(match.group(1))
        if not points:
            match = re.search(r"points:\s*(\d+)", text)
            if match:
                points = int(match.group(1))
        match = re.search(r"Mesh non-orthogonality Max:\s*([\d.]+)", text)
        if match:
            max_non_ortho = max(max_non_ortho, float(match.group(1)))
        match = re.search(r"Number of severely non-orthogonal.*?faces:\s*(\d+)", text)
        if match:
            severe_non_ortho = max(severe_non_ortho, int(match.group(1)))
        match = re.search(r"Failed\s+(\d+)\s+mesh checks", text)
        if match:
            failed_checks = int(match.group(1))

    return {
        "cells": cells,
        "points": points,
        "max_non_orthogonality_deg": max_non_ortho,
        "severe_non_orthogonal_faces": severe_non_ortho,
        "failed_mesh_checks": failed_checks,
    }


def parse_y_plus(case_dir: Path) -> dict[str, float | None]:
    """Read the wall y+ history written by the ``yPlus`` function object.

    OpenFOAM writes ``postProcessing/yPlus/<time>/yPlus.dat`` with columns
    ``Time patch min max average``.  The patch name is not numeric, so numeric
    token filtering leaves ``[time, min, max, average]``.
    """
    post_dir = case_dir / "postProcessing"
    paths = sorted(
        list(post_dir.glob("yPlus*/*/yPlus.dat"))
        + list(post_dir.glob("yPlus*/*/yPlus_*.dat"))
    )
    result: dict[str, float | None] = {
        "y_plus_min": None,
        "y_plus_max": None,
        "y_plus_avg": None,
    }
    for path in reversed(paths):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            numbers: list[float] = []
            for token in line.split():
                try:
                    numbers.append(float(token))
                except ValueError:
                    continue
            if len(numbers) >= 4:
                result["y_plus_min"] = numbers[-3]
                result["y_plus_max"] = numbers[-2]
                result["y_plus_avg"] = numbers[-1]
                return result
    return result


def evaluate_convergence(
    case_dir: Path,
    std_threshold: float = 0.005,
    residual_threshold: float = 1e-4,
    last_n_iterations: int = 50,
) -> ConvergenceVerdict:
    """Evaluate convergence from the force history and the actual solver log.

    A stable force history alone is not enough: the pressure and velocity
    residuals must also be below the solve targets.  This prevents a stalled or
    partially converged run from being presented as a converged CFD result.
    """
    log_file = case_dir / "log.simpleFoam"
    cl_mean, cl_std, cd_mean, cd_std, cm_mean, cm_std, total_iters = parse_force_coeffs(
        case_dir, last_n_iterations=last_n_iterations
    )
    residuals = parse_residuals(log_file)

    p_res = residuals.get("p", 1.0)
    u_res = max(residuals.get("Ux", 1.0), residuals.get("Uy", 1.0))
    k_res = residuals.get("k", 1.0)
    omega_res = residuals.get("omega", 1.0)

    finite = all(math.isfinite(v) for v in (cl_mean, cd_mean, cm_mean, p_res, u_res))

    if (
        not finite
        or abs(cl_mean) > 10.0
        or cd_mean > 5.0
        or cd_mean < 0.0
        or p_res > 1e2
        or u_res > 1e2
    ):
        status = ConvergenceStatus.DIVERGED
        is_converged = False
    elif total_iters == 0:
        status = ConvergenceStatus.FAILED
        is_converged = False
    elif (
        total_iters >= 20
        and cl_std <= std_threshold
        and cd_std <= std_threshold
        and p_res <= residual_threshold
        and u_res <= residual_threshold * 50.0
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
        cm_mean=cm_mean,
        cm_std=cm_std,
    )
