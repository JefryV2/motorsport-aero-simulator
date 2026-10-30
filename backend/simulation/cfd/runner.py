"""Real OpenFOAM execution for 2D airfoil RANS cases.

Execution is supported three ways:

* ``native`` – OpenFOAM binaries are on the Linux/Windows PATH.
* ``wsl`` – a WSL distribution exposes ``simpleFoam``/``blockMesh``.  Cases are
  copied to the WSL home directory because OpenFOAM rejects the space in this
  Windows user path.
* ``synthetic`` – no solver exists.  This is an explicitly labelled closed-form
  fallback, never presented as CFD.

A real solver failure is raised as :class:`CFDRunError`; it is not silently
replaced with the synthetic model.
"""

from __future__ import annotations

import functools
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

from backend.simulation.cfd.case_builder import build_openfoam_case
from backend.simulation.cfd.models import (
    AirfoilGeometry,
    CFDResult,
    ConvergenceStatus,
    ConvergenceVerdict,
    MeshLevel,
    MeshStats,
)
from backend.simulation.cfd.parser import evaluate_convergence, parse_mesh_stats, parse_y_plus

SYNTHETIC_SOLVER_VERSION = "none (calibrated-synthetic-v1)"
DEFAULT_TIMEOUT_S = 900.0


class CFDRunError(RuntimeError):
    """A real OpenFOAM case was attempted and failed."""


@functools.lru_cache(maxsize=1)
def _wsl_distro() -> str:
    """Return the configured WSL distribution name (empty means the default)."""
    return os.getenv("AERO_CFD_WSL_DISTRO", "").strip()


def _wsl_command(*args: str) -> list[str]:
    distro = _wsl_distro()
    prefix = ["wsl"]
    if distro:
        prefix += ["-d", distro]
    return prefix + list(args)


def _run(
    command: list[str],
    *,
    timeout: float,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CFDRunError(f"Command timed out after {timeout:.0f} s: {' '.join(command)}") from exc
    except FileNotFoundError as exc:
        raise CFDRunError(f"Command not found: {command[0]}") from exc


@functools.lru_cache(maxsize=1)
def detect_cfd_backend() -> str:
    """Detect the available solver backend: ``native``, ``wsl`` or ``synthetic``."""
    if shutil.which("simpleFoam") and shutil.which("blockMesh"):
        return "native"

    if shutil.which("wsl"):
        try:
            proc = _run(
                _wsl_command(
                    "bash",
                    "-lc",
                    "command -v simpleFoam >/dev/null 2>&1 && command -v blockMesh >/dev/null 2>&1 && echo WSL_OK",
                ),
                timeout=15.0,
            )
            if proc.returncode == 0 and "WSL_OK" in proc.stdout:
                return "wsl"
        except CFDRunError:
            pass

    return "synthetic"


def clear_cfd_backend_cache() -> None:
    """Clear detection caches (useful after installing a solver in-process)."""
    detect_cfd_backend.cache_clear()
    _wsl_home.cache_clear()
    get_solver_version.cache_clear()


@functools.lru_cache(maxsize=1)
def get_solver_version() -> str:
    """Return a human-readable solver version for the health endpoint."""
    backend = detect_cfd_backend()
    if backend == "native":
        return "OpenFOAM (native)"
    if backend == "wsl":
        try:
            proc = _run(
                _wsl_command(
                    "bash",
                    "-lc",
                    "dpkg-query -W openfoam 2>/dev/null || true",
                ),
                timeout=15.0,
            )
            fields = proc.stdout.strip().split()
            if len(fields) >= 2:
                return f"OpenFOAM {fields[1].split('.')[0]}"
        except CFDRunError:
            pass
        return "OpenFOAM (WSL)"
    return SYNTHETIC_SOLVER_VERSION


@functools.lru_cache(maxsize=1)
def _wsl_home() -> str:
    proc = _run(_wsl_command("bash", "-lc", 'printf "%s" "$HOME"'), timeout=15.0)
    home = proc.stdout.strip()
    if proc.returncode != 0 or not home.startswith("/"):
        raise CFDRunError(f"Could not determine the WSL home directory: {proc.stderr.strip()}")
    return home


def _wsl_case_path(case_dir: Path) -> PurePosixPath:
    return PurePosixPath(_wsl_home()) / "aero-cfd" / "cases" / case_dir.name


def _copy_to_wsl(case_dir: Path, wsl_case: PurePosixPath) -> None:
    """Copy only the input files of a case into a WSL-native scratch directory."""
    create = _run(
        _wsl_command("bash", "-lc", f"rm -rf '{wsl_case}' && mkdir -p '{wsl_case}'"),
        timeout=30.0,
    )
    if create.returncode != 0:
        raise CFDRunError(f"Could not create WSL case directory: {create.stderr.strip()}")

    copy = _run(
        _wsl_command(
            "--cd",
            str(case_dir),
            "--",
            "bash",
            "-lc",
            f"cp -r 0 constant system run_case.sh case_params.json '{wsl_case}'/",
        ),
        timeout=120.0,
    )
    if copy.returncode != 0:
        raise CFDRunError(
            f"Could not copy the case into WSL: {copy.stderr.strip() or copy.stdout.strip()}"
        )


def _copy_from_wsl(case_dir: Path, wsl_case: PurePosixPath) -> None:
    """Copy solver products back to the Windows case directory for parsing."""
    copy = _run(
        _wsl_command(
            "--cd",
            str(case_dir),
            "--",
            "bash",
            "-lc",
            (
                f"(cp -r '{wsl_case}/postProcessing' . 2>/dev/null || true); "
                f"(cp '{wsl_case}'/log.* . 2>/dev/null || true); "
                f"(cp '{wsl_case}/case_meta.json' . 2>/dev/null || true)"
            ),
        ),
        timeout=120.0,
    )
    if copy.returncode != 0:
        raise CFDRunError(f"Could not copy results back from WSL: {copy.stderr.strip()}")


def _solver_version(case_dir: Path) -> str:
    for name in ("log.checkMesh", "log.simpleFoam"):
        path = case_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        import re

        match = re.search(r"OPENFOAM=(\d{4})", text)
        if match:
            return f"OpenFOAM {match.group(1)}"
    return "OpenFOAM (version not recorded)"


def _failure_message(case_dir: Path) -> str:
    chunks: list[str] = []
    for name in (
        "log.checkMesh",
        "log.simpleFoam",
    ):
        path = case_dir / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = "\n".join(lines[-10:])
        chunks.append(f"--- {name} ---\n{tail}")
    return "OpenFOAM run failed in " + case_dir.name + "\n" + "\n".join(chunks)


def _domain_area(case_dir: Path) -> float | None:
    params_path = case_dir / "case_params.json"
    if not params_path.exists():
        return None
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        domain = params.get("domain", {})
        return float(domain["x_out"] - domain["x_in"]) * float(domain["y_top"] - domain["y_bottom"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _actual_mesh_stats(
    case_dir: Path, mesh_stats: MeshStats
) -> tuple[MeshStats, dict[str, float | int]]:
    info = parse_mesh_stats(case_dir)
    cells = int(info.get("cells") or 0)
    points = int(info.get("points") or 0)
    if cells > 0:
        mesh_stats.num_cells = cells
    if points > 0:
        mesh_stats.num_points = points
    area = _domain_area(case_dir)
    if cells > 0 and area is not None:
        mesh_stats.grid_spacing_h = math.sqrt(area / cells)
    y_plus = parse_y_plus(case_dir)
    mesh_stats.y_plus_min = y_plus["y_plus_min"]
    mesh_stats.y_plus_max = y_plus["y_plus_max"]
    mesh_stats.y_plus_avg = y_plus["y_plus_avg"]
    for key, value in y_plus.items():
        if value is not None:
            info[key] = value
    return mesh_stats, info


def _read_meta(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "case_meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_meta(case_dir: Path, meta: dict[str, Any]) -> None:
    (case_dir / "case_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n"
    )


def _cached_real_result(
    case_dir: Path,
    mesh_stats: MeshStats,
    airfoil: AirfoilGeometry,
    speed_ms: float,
) -> CFDResult | None:
    meta = _read_meta(case_dir)
    solver_used = str(meta.get("solver_used", ""))
    if solver_used not in {"openfoam_native", "openfoam_wsl"}:
        return None
    if meta.get("convergence_status") == ConvergenceStatus.FAILED.value:
        return None
    coefficient_files = list((case_dir / "postProcessing").glob("**/coefficient*.dat"))
    if not coefficient_files or not (case_dir / "log.simpleFoam").exists():
        return None

    return _result_from_logs(
        case_dir=case_dir,
        mesh_stats=mesh_stats,
        airfoil=airfoil,
        speed_ms=speed_ms,
        solver_used=solver_used,
        execution_time_s=float(meta.get("execution_time_s", 0.0)),
        write_meta=False,
    )


def _result_from_logs(
    case_dir: Path,
    mesh_stats: MeshStats,
    airfoil: AirfoilGeometry,
    speed_ms: float,
    solver_used: str,
    execution_time_s: float,
    *,
    write_meta: bool,
) -> CFDResult:
    verdict = evaluate_convergence(case_dir)
    if verdict.status == ConvergenceStatus.FAILED:
        raise CFDRunError(_failure_message(case_dir))

    mesh_stats, mesh_info = _actual_mesh_stats(case_dir, mesh_stats)
    cl = verdict.cl_mean
    cd = verdict.cd_mean
    cm = verdict.cm_mean
    q = 0.5 * 1.225 * speed_ms**2
    s_ref = airfoil.chord_m * 1.0  # coefficients are per one-metre span
    solver_version = _solver_version(case_dir)

    if write_meta:
        meta = _read_meta(case_dir)
        meta.update(
            {
                "solver_used": solver_used,
                "solver_version": solver_version,
                "is_synthetic": False,
                "convergence_status": verdict.status.value,
                "is_converged": verdict.is_converged,
                "iterations": verdict.iterations_run,
                "cl_section": cl,
                "cd_section": cd,
                "cm_section": cm,
                "cells": mesh_stats.num_cells,
                "points": mesh_stats.num_points,
                "grid_spacing_h": mesh_stats.grid_spacing_h,
                "estimated_y_plus": mesh_stats.estimated_y_plus,
                "first_cell_height_m": mesh_stats.first_cell_height_m,
                "y_plus_min": mesh_stats.y_plus_min,
                "y_plus_max": mesh_stats.y_plus_max,
                "y_plus_avg": mesh_stats.y_plus_avg,
                "max_non_orthogonality_deg": float(mesh_info["max_non_orthogonality_deg"]),
                "severe_non_orthogonal_faces": int(mesh_info["severe_non_orthogonal_faces"]),
                "failed_mesh_checks": int(mesh_info["failed_mesh_checks"]),
                "final_p_residual": verdict.final_p_residual,
                "final_u_residual": verdict.final_u_residual,
                "final_k_residual": verdict.final_k_residual,
                "final_omega_residual": verdict.final_omega_residual,
                "execution_time_s": execution_time_s,
                "completed_at_epoch": time.time(),
            }
        )
        _write_meta(case_dir, meta)

    return CFDResult(
        case_id=case_dir.name,
        case_dir=str(case_dir),
        mesh_stats=mesh_stats,
        convergence=verdict,
        cl=cl,
        cd=cd,
        cm=cm,
        downforce_N=cl * q * s_ref,
        drag_N=cd * q * s_ref,
        execution_time_s=execution_time_s,
        solver_used=solver_used,
        solver_version=solver_version,
        is_synthetic=False,
    )


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

    mesh_error_map = {
        MeshLevel.COARSE: 0.015,
        MeshLevel.MEDIUM: 0.005,
        MeshLevel.FINE: 0.000,
    }
    grid_bias = mesh_error_map[mesh_level]

    a0 = 0.1081
    cl_linear = a0 * alpha_deg
    if abs(alpha_deg) > 12.0:
        excess = abs(alpha_deg) - 12.0
        stall_factor = math.exp(-0.03 * excess**2)
        cl_base = cl_linear * stall_factor
    else:
        cl_base = cl_linear

    ground_factor = 1.0
    if airfoil.ground_height_m is not None:
        hc = max(0.05, airfoil.ground_height_m / airfoil.chord_m)
        if hc <= 1.0:
            ground_factor = 1.0 + 0.185 * ((1.0 / hc) ** 0.58 - 1.0)
        else:
            ground_factor = 1.0

    cl = float((cl_base * ground_factor) + grid_bias)
    cd0_benchmark = 0.0082 * ((3e6 / max(re, 1e4)) ** 0.11)
    cd_pressure = 0.0055 * (cl_base**2)
    cd_stall = 0.0015 * (max(0.0, abs(alpha_deg) - 11.0) ** 2) if abs(alpha_deg) > 11.0 else 0.0
    cd = float(cd0_benchmark + cd_pressure + cd_stall + (grid_bias * 0.1))
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
        cm_mean=cm,
        cm_std=0.0,
    )
    return cl, cd, cm, verdict


def run_cfd_case(
    airfoil: AirfoilGeometry,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
    speed_ms: float = 50.0,
    base_dir: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> CFDResult:
    """Build, mesh, solve, and parse a real OpenFOAM case.

    When no solver is installed the explicitly labelled synthetic fallback is
    returned.  When a solver is installed, a failure raises
    :class:`CFDRunError` rather than pretending the synthetic model is CFD.
    """
    t0 = time.time()
    case_dir, mesh_stats = build_openfoam_case(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=speed_ms,
        base_dir=base_dir,
    )

    backend = detect_cfd_backend()

    if backend == "synthetic":
        cl, cd, cm, verdict = evaluate_calibrated_synthetic_cfd(
            airfoil=airfoil, mesh_level=mesh_level, speed_ms=speed_ms
        )
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
            json.dumps(provenance, indent=2), encoding="utf-8", newline="\n"
        )
        q = 0.5 * 1.225 * speed_ms**2
        s_ref = airfoil.chord_m * 1.0
        return CFDResult(
            case_id=case_dir.name,
            case_dir=str(case_dir),
            mesh_stats=mesh_stats,
            convergence=verdict,
            cl=cl,
            cd=cd,
            cm=cm,
            downforce_N=cl * q * s_ref,
            drag_N=cd * q * s_ref,
            execution_time_s=time.time() - t0,
            solver_used="calibrated_synthetic",
            solver_version=SYNTHETIC_SOLVER_VERSION,
            is_synthetic=True,
        )

    cached = _cached_real_result(case_dir, mesh_stats, airfoil, speed_ms)
    if cached is not None:
        return cached

    if backend == "native":
        solver_used = "openfoam_native"
        proc = _run(
            ["bash", "run_case.sh"],
            timeout=timeout_s,
            cwd=case_dir,
        )
    else:
        solver_used = "openfoam_wsl"
        wsl_case = _wsl_case_path(case_dir)
        _copy_to_wsl(case_dir, wsl_case)
        try:
            proc = _run(
                _wsl_command("bash", "-lc", f"cd '{wsl_case}' && bash run_case.sh"),
                timeout=timeout_s,
            )
        finally:
            try:
                _copy_from_wsl(case_dir, wsl_case)
            except CFDRunError:
                # Preserve the original solver error; the missing copy is
                # reported through the failure message below.
                pass

    if proc.returncode != 0:
        failed = {
            "solver_used": solver_used,
            "solver_version": _solver_version(case_dir),
            "is_synthetic": False,
            "convergence_status": ConvergenceStatus.FAILED.value,
            "is_converged": False,
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "completed_at_epoch": time.time(),
        }
        _write_meta(case_dir, failed)
        raise CFDRunError(_failure_message(case_dir))

    return _result_from_logs(
        case_dir=case_dir,
        mesh_stats=mesh_stats,
        airfoil=airfoil,
        speed_ms=speed_ms,
        solver_used=solver_used,
        execution_time_s=time.time() - t0,
        write_meta=True,
    )
