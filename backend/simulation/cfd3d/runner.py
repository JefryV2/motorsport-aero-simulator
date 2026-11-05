"""Real OpenFOAM execution for the imported-mesh 3D tier.

Reuses the 2D tier's WSL orchestration (``cfd/runner.py``) and its parsers:
blockMesh → snappyHexMesh → checkMesh → simpleFoam, in the WSL home the same
way. Unlike the 2D tier there is NO synthetic fallback here: an imported
geometry with no solver is a 503, never simulated numbers.

Caching: the case directory is keyed by hash of the triangles plus run
parameters (``build_imported_case``), so a repeated request re-parses the
existing logs instead of re-solving.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.simulation.cfd.models import ConvergenceStatus, ConvergenceVerdict
from backend.simulation.cfd.parser import parse_mesh_stats, parse_y_plus
from backend.simulation.cfd.runner import (
    CFDRunError,
    _copy_from_wsl,
    _copy_to_wsl,
    _run,
    _wsl_case_path,
    _wsl_command,
    detect_cfd_backend,
)
from backend.simulation.cfd3d.case_builder import (
    CASE3D_FORMAT_VERSION,
    NU,
    RHO,
    Case3DBuild,
    build_imported_case,
)
from backend.simulation.cfd3d.models import ImportedCFDLevel
from backend.simulation.cfd3d.surface import ImportedSurface

MAX_TIMEOUT_S = 7200.0


@dataclass
class ImportedCFDProvenance:
    """What actually produced a 3D imported-mesh result."""

    backend: str
    solver_used: str
    case_hash: str
    case_format: str = CASE3D_FORMAT_VERSION
    frontal_area_m2: float = 0.0
    mesh_triangles: int = 0
    watertight_edge_fraction: float = 0.0
    scale_m_per_unit: float = 1.0
    axes: str = "three"
    mesh_cells: int = 0
    max_non_orthogonality_deg: float = 0.0
    severe_non_orthogonal_faces: int = 0
    failed_mesh_checks: int = 0
    iterations: int = 0
    convergence_status: str = ConvergenceStatus.FAILED.value
    is_converged: bool = False
    cd_3d: float = 0.0
    cl_3d: float = 0.0
    cm_3d: float = 0.0
    cd_std: float = 0.0
    cl_std: float = 0.0
    residual_p: float = 1.0
    residual_u: float = 1.0
    residual_k: float = 1.0
    residual_omega: float = 1.0
    y_plus_avg: float | None = None
    execution_time_s: float = 0.0
    solver_version: str = "OpenFOAM (version not recorded)"
    case_dir: str = ""
    max_iterations: int = 0
    reynolds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ImportedCFDResult:
    """Force result for an imported geometry, forces in newtons."""

    drag_N: float
    downforce_N: float  # positive Cl is downforce for car meshes (liftDir -y)
    lift_N: float  # the same number with the physics sign (positive = up)
    sideforce_N: float
    cd: float
    cl: float
    cm: float
    column_load_N: float  # Fz reference - what 'downforce' multiplies
    q_dynamic_pa: float
    speed_ms: float
    frontal_area_m2: float
    provenance: ImportedCFDProvenance


def solver_available() -> bool:
    return detect_cfd_backend() != "synthetic"


def _residuals_3d(case_dir: Path) -> dict[str, float]:
    """Final initial residuals from ``log.simpleFoam``, including Uz.

    The 2D parser only knows Ux/Uy; the 3D solver logs a streamwise ``Uz`` term
    that must not be silently ignored.
    """
    import re

    log_file = case_dir / "log.simpleFoam"
    if not log_file.exists():
        return {}
    content = log_file.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float] = {}
    for var in ("p", "Ux", "Uy", "Uz", "k", "omega"):
        matches = re.findall(rf"Solving for {var}, Initial residual = ([\d.+\-eE]+)", content)
        if matches:
            out[var] = float(matches[-1])
    return out


def _evaluate_convergence_3d(case_dir: Path) -> ConvergenceVerdict:
    """The shared verdict with the 3D ``Uz`` residual folded in.

    Only folds when a real ``Uz`` residual exists in the log; a missing log is
    the shared verdict's problem, not an excuse to fabricate a residual here.
    """

    from backend.simulation.cfd.parser import evaluate_convergence as shared

    verdict = shared(case_dir)
    uz = _residuals_3d(case_dir).get("Uz")
    if uz is None:
        return verdict
    if uz > 1e2:
        verdict.status = ConvergenceStatus.DIVERGED
        verdict.is_converged = False
        verdict.final_u_residual = max(verdict.final_u_residual, uz)
    elif uz > verdict.final_u_residual:
        verdict.final_u_residual = uz
        if verdict.status == ConvergenceStatus.CONVERGED and uz <= 5e-3:
            verdict.is_converged = True
        else:
            if verdict.status is ConvergenceStatus.CONVERGED:
                verdict.status = ConvergenceStatus.UNCONVERGED
            verdict.is_converged = False
    return verdict


def run_imported_cfd(
    surface: ImportedSurface,
    speed_ms: float,
    *,
    mesh_level: ImportedCFDLevel | str = ImportedCFDLevel.MEDIUM,
    max_iterations: int = 1500,
    timeout_s: float = MAX_TIMEOUT_S,
    base_dir: Path | None = None,
) -> ImportedCFDResult:
    """Build or reuse the 3D case for this mesh, then solve and parse it."""
    t0 = time.time()
    level = (
        mesh_level
        if isinstance(mesh_level, ImportedCFDLevel)
        else ImportedCFDLevel(mesh_level)
    )
    build: Case3DBuild = build_imported_case(
        surface,
        speed_ms=speed_ms,
        mesh_level=level,
        max_iterations=max_iterations,
        base_dir=base_dir,
    )
    case_dir = build.case_dir

    cached = _case_meta(case_dir)
    if cached.solver_used and cached.convergence_status != ConvergenceStatus.FAILED.value:
        # Re-parse logs rather than re-solve: same numbers, ~0 cost.
        return _result_from_logs(
            case_dir=case_dir,
            surface=surface,
            speed_ms=speed_ms,
            execution_time_s=float(cached.execution_time_s or 0.0),
            reparse=True,
            max_iterations=max_iterations,
        )

    backend = detect_cfd_backend()
    if backend == "synthetic":
        raise CFDRunError(
            "No OpenFOAM solver is installed, and the imported-mesh tier refuses "
            "to invent numbers. Install OpenFOAM (see README) or use the "
            "analytic/VLM tiers, which remain parameter-based and do not pretend "
            "to know this geometry."
        )

    solver_used = "openfoam_wsl" if backend == "wsl" else "openfoam_native"
    if backend == "native":
        proc = _run(["bash", "run_case.sh"], timeout=timeout_s, cwd=case_dir)
    else:
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
                pass

    if proc.returncode != 0:
        _write_failed_meta(case_dir, proc)
        raise CFDRunError(_failure_message(case_dir))

    return _result_from_logs(
        case_dir=case_dir,
        surface=surface,
        speed_ms=speed_ms,
        execution_time_s=time.time() - t0,
        reparse=False,
        max_iterations=max_iterations,
        solver_used=solver_used,
    )


def _case_meta(case_dir: Path) -> ImportedCFDProvenance:
    """Read back the provenance a previous real run wrote, if any."""
    path = case_dir / "case_meta.json"
    if not path.exists():
        return ImportedCFDProvenance(
            backend="none", solver_used="", case_hash=case_dir.name.removeprefix("case3d_")
        )
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return ImportedCFDProvenance(
        backend=str(data.get("backend", "")),
        solver_used=str(data.get("solver_used", "")),
        case_hash=str(data.get("case_hash", "")),
        execution_time_s=float(data.get("execution_time_s", 0.0)),
        convergence_status=str(data.get("convergence_status", "")),
    )


def _write_failed_meta(case_dir: Path, proc: subprocess.CompletedProcess[str]) -> None:
    import json

    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "solver_used": "openfoam",
                "convergence_status": ConvergenceStatus.FAILED.value,
                "is_converged": False,
                "error": (proc.stderr.strip() or proc.stdout.strip())[-4000:],
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _failure_message(case_dir: Path) -> str:
    chunks: list[str] = []
    for name in ("log.checkMesh", "log.simpleFoam", "log.snappyHexMesh", "log.blockMesh"):
        path = case_dir / name
        if not path.exists():
            continue
        tail = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-12:])
        chunks.append(f"--- {name} ---\n{tail}")
    return "OpenFOAM 3D run failed in " + case_dir.name + "\n" + "\n".join(chunks)


def _result_from_logs(
    case_dir: Path,
    surface: ImportedSurface,
    speed_ms: float,
    execution_time_s: float,
    *,
    reparse: bool,
    max_iterations: int,
    solver_used: str = "openfoam_wsl",
) -> ImportedCFDResult:
    """Parse the real logs into a result, then persist the provenance meta.

    ``reparse=True`` skips re-writing ``case_meta.json``: the cached log set
    already produced that file, and its ``execution_time_s`` is the original
    solve's, not this request's.
    """
    verdict = _evaluate_convergence_3d(case_dir)
    if verdict.status == ConvergenceStatus.FAILED:
        raise CFDRunError(_failure_message(case_dir))

    info: dict[str, float | int] = parse_mesh_stats(case_dir)
    cells = int(info.get("cells") or 0)

    y_plus = parse_y_plus(case_dir)

    q = 0.5 * RHO * speed_ms**2
    frontal = float(surface.frontal_area_m2)
    cd = verdict.cd_mean
    cl = verdict.cl_mean
    cm = verdict.cm_mean
    f_ref = q * frontal

    solver_version = _solver_version(case_dir)
    if not reparse:
        _write_success_meta(
            case_dir,
            solver_used=solver_used,
            solver_version=solver_version,
            verdict=verdict,
            cells=cells,
            info=info,
            y_plus=y_plus,
            execution_time_s=execution_time_s,
        )

    backend_now = detect_cfd_backend()
    provenance = ImportedCFDProvenance(
        backend=backend_now,
        solver_used=solver_used,
        case_hash=case_dir.name.removeprefix("case3d_"),
        frontal_area_m2=frontal,
        mesh_triangles=surface.source_triangle_count,
        watertight_edge_fraction=surface.watertight_edge_fraction,
        scale_m_per_unit=surface.scale_m_per_unit,
        axes=surface.axes,
        mesh_cells=cells,
        max_non_orthogonality_deg=float(info.get("max_non_orthogonality_deg", 0.0)),
        severe_non_orthogonal_faces=int(info.get("severe_non_orthogonal_faces", 0)),
        failed_mesh_checks=int(info.get("failed_mesh_checks", 0)),
        iterations=verdict.iterations_run,
        convergence_status=verdict.status.value,
        is_converged=verdict.is_converged,
        cd_3d=cd,
        cl_3d=cl,
        cm_3d=cm,
        cd_std=verdict.cd_std,
        cl_std=verdict.cl_std,
        residual_p=verdict.final_p_residual,
        residual_u=verdict.final_u_residual,
        residual_k=verdict.final_k_residual,
        residual_omega=verdict.final_omega_residual,
        y_plus_avg=y_plus["y_plus_avg"],
        execution_time_s=execution_time_s,
        solver_version=solver_version,
        case_dir=case_dir.as_posix(),
        max_iterations=max_iterations,
        reynolds=speed_ms * surface.extent_x_m / NU,
        warnings=list(surface.warnings),
    )

    return ImportedCFDResult(
        drag_N=cd * f_ref,
        downforce_N=cl * f_ref,
        lift_N=-cl * f_ref,
        sideforce_N=0.0,
        cd=cd,
        cl=cl,
        cm=cm,
        column_load_N=f_ref,
        q_dynamic_pa=q,
        speed_ms=speed_ms,
        frontal_area_m2=frontal,
        provenance=provenance,
    )


def _write_success_meta(
    case_dir: Path,
    *,
    solver_used: str,
    solver_version: str,
    verdict: ConvergenceVerdict,
    cells: int,
    info: dict[str, float | int],
    y_plus: dict[str, float | None],
    execution_time_s: float,
) -> None:
    import json

    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "solver_used": solver_used,
                "solver_version": solver_version,
                "is_synthetic": False,
                "backend": detect_cfd_backend(),
                "is_converged": verdict.is_converged,
                "convergence_status": verdict.status.value,
                "iterations": verdict.iterations_run,
                "cl": verdict.cl_mean,
                "cd": verdict.cd_mean,
                "cells": cells,
                "y_plus_avg": y_plus["y_plus_avg"],
                "y_plus_min": y_plus["y_plus_min"],
                "y_plus_max": y_plus["y_plus_max"],
                "max_non_orthogonality_deg": float(info.get("max_non_orthogonality_deg", 0.0)),
                "failed_mesh_checks": int(info.get("failed_mesh_checks", 0)),
                "execution_time_s": execution_time_s,
                "format": CASE3D_FORMAT_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )


def _solver_version(case_dir: Path) -> str:
    import re

    for name in ("log.checkMesh", "log.simpleFoam", "log.snappyHexMesh"):
        path = case_dir / name
        if not path.exists():
            continue
        match = re.search(r"OPENFOAM=(\d{4})", path.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return f"OpenFOAM {match.group(1)}"
    return "OpenFOAM (version not recorded)"

