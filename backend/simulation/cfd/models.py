"""Data models for OpenFOAM CFD pipeline and mesh hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MeshLevel(str, Enum):
    """Systematically refined grid levels for Richardson extrapolation."""

    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


class ConvergenceStatus(str, Enum):
    """CFD solve convergence classification."""

    CONVERGED = "converged"
    UNCONVERGED = "unconverged"
    DIVERGED = "diverged"
    FAILED = "failed"
    MOCK_EVALUATED = "mock_evaluated"


@dataclass
class AirfoilGeometry:
    """Parametric 2D airfoil definition."""

    name: str = "NACA0012"
    chord_m: float = 1.0
    angle_of_attack_deg: float = 0.0
    ground_height_m: float | None = None  # None for free-stream isolated airfoil
    x_coords: list[float] = field(default_factory=list)
    y_coords: list[float] = field(default_factory=list)


@dataclass
class MeshStats:
    """Statistics for generated CFD mesh."""

    level: MeshLevel
    num_cells: int
    num_points: int
    grid_spacing_h: float
    estimated_y_plus: float
    first_cell_height_m: float | None = None
    y_plus_min: float | None = None
    y_plus_max: float | None = None
    y_plus_avg: float | None = None


@dataclass
class ConvergenceVerdict:
    """Convergence metrics evaluated across final iterations."""

    status: ConvergenceStatus
    iterations_run: int
    cl_mean: float
    cl_std: float
    cd_mean: float
    cd_std: float
    final_p_residual: float
    final_u_residual: float
    final_k_residual: float
    final_omega_residual: float
    is_converged: bool
    cm_mean: float = 0.0
    cm_std: float = 0.0


@dataclass
class CFDResult:
    """Complete output of a CFD simulation run."""

    case_id: str
    case_dir: str
    mesh_stats: MeshStats
    convergence: ConvergenceVerdict
    cl: float
    cd: float
    cm: float
    downforce_N: float
    drag_N: float
    execution_time_s: float
    solver_used: str  # "openfoam_native", "openfoam_wsl", or "calibrated_synthetic"
    solver_version: str = "unknown"
    is_synthetic: bool = False
