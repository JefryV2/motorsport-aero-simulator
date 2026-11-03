"""OpenFOAM CFD simulation interface connecting API endpoints to the CFD pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from backend.models.aero_config import AeroForces, FormulaAeroConfig, SimulationResult
from backend.simulation.analytical import (
    REF_AREA,
    RHO,
    _body_drag,
    _underbody_forces,
    _wing_forces,
    effective_rear_flap,
    lap_time_delta_s,
)
from backend.simulation.cfd.case_builder import build_openfoam_case
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel
from backend.simulation.cfd.runner import detect_cfd_backend, run_cfd_case


@dataclass(frozen=True)
class CFDProvenance:
    """What actually produced a CFD-family result."""

    backend: str
    solver_used: str
    solver_version: str
    convergence_status: str
    is_converged: bool
    is_synthetic: bool
    fidelity: str
    note: str = ""
    case_id: str = ""
    mesh_cells: int = 0
    iterations: int = 0
    cl_section: float = 0.0
    cd_section: float = 0.0
    residual_p: float = 0.0
    residual_u: float = 0.0
    execution_time_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def is_openfoam_available() -> bool:
    """Check if OpenFOAM solver is available (native or WSL)."""
    return detect_cfd_backend() in ("native", "wsl")


def generate_case(
    config: FormulaAeroConfig,
    case_id: str,
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
) -> Path:
    """Generate OpenFOAM case directory from FormulaAeroConfig parameters."""
    total_angle = config.front_wing_main.angle_deg + config.front_wing_flap.angle_deg
    ground_h = config.ride_height_front_mm / 1000.0

    airfoil = AirfoilGeometry(
        name="FrontWingSection",
        chord_m=config.front_wing_main.chord_m + config.front_wing_flap.chord_m,
        angle_of_attack_deg=total_angle,
        ground_height_m=ground_h,
    )

    case_dir, _ = build_openfoam_case(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=config.speed_kmh / 3.6,
    )
    return case_dir


def run_cfd_with_provenance(
    config: FormulaAeroConfig,
    case_id: str = "latest",
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
) -> tuple[SimulationResult, CFDProvenance]:
    """Run the CFD-family pipeline and return the result with its real provenance.

    The returned provenance is derived from the run that actually executed, not
    from solver availability. This is what prevents synthetic fallbacks from
    being labelled as OpenFOAM.
    """
    total_angle = config.front_wing_main.angle_deg + config.front_wing_flap.angle_deg
    ground_h = config.ride_height_front_mm / 1000.0
    speed_ms = config.speed_kmh / 3.6

    airfoil = AirfoilGeometry(
        name="FrontWingSection",
        chord_m=config.front_wing_main.chord_m + config.front_wing_flap.chord_m,
        angle_of_attack_deg=total_angle,
        ground_height_m=ground_h,
    )

    result = run_cfd_case(
        airfoil=airfoil,
        mesh_level=mesh_level,
        speed_ms=speed_ms,
    )

    # Dynamic pressure
    q = 0.5 * RHO * (speed_ms**2)

    # Front wing 3D estimate from 2D section (span b = 1.8m)
    span_m = config.front_wing_main.span_m
    df_front_wing_N = result.cl * q * airfoil.chord_m * span_m
    drag_front_wing_N = result.cd * q * airfoil.chord_m * span_m

    # Underbody floor forces
    df_floor, drag_floor, floor_front_share = _underbody_forces(config, speed_ms)

    # Rear wing: the same two empirical elements the analytical tier charges, so
    # one setup does not come out with a different rear load depending on which
    # fidelity was asked for. The old restatement used 0.08 Cl per summed
    # main+flap degree over a fixed 1.0 m^2 planform, which ignored chord, span,
    # gurney and DRS and priced this wing about four times the model's own.
    df_rw_main, drag_rw_main = _wing_forces(config.rear_wing_main, speed_ms)
    df_rw_flap, drag_rw_flap = _wing_forces(effective_rear_flap(config), speed_ms)
    df_rear_wing_N = df_rw_main + df_rw_flap
    drag_rear_wing_N = drag_rw_main + drag_rw_flap

    df_front_total = df_front_wing_N + df_floor * floor_front_share
    df_rear_total = df_rear_wing_N + df_floor * (1.0 - floor_front_share)
    df_total_N = df_front_total + df_rear_total
    drag_total_N = drag_front_wing_N + drag_rear_wing_N + drag_floor + _body_drag(speed_ms)

    balance_front_pct = (df_front_total / df_total_N * 100.0) if df_total_N > 1.0 else 50.0
    efficiency = df_total_N / drag_total_N if drag_total_N > 1.0 else 1.0

    cl_total = df_total_N / (q * REF_AREA) if q > 0 else 0.0
    cd_total = drag_total_N / (q * REF_AREA) if q > 0 else 0.0

    forces = AeroForces(
        drag_N=round(drag_total_N, 1),
        downforce_total_N=round(df_total_N, 1),
        downforce_front_N=round(df_front_total, 1),
        downforce_rear_N=round(df_rear_total, 1),
        balance_front_pct=round(balance_front_pct, 1),
        cop_x_m=round(1.6 * (balance_front_pct / 100.0), 3),
        cl_total=round(cl_total, 3),
        cd_total=round(cd_total, 3),
        efficiency=round(efficiency, 2),
    )

    provenance = CFDProvenance(
        backend=detect_cfd_backend(),
        solver_used=result.solver_used,
        solver_version=result.solver_version,
        convergence_status=result.convergence.status.value,
        is_converged=result.convergence.is_converged,
        is_synthetic=result.is_synthetic,
        fidelity="synthetic_model" if result.is_synthetic else "rans_hybrid",
        note=(
            "No CFD solver ran. Coefficients come from a calibrated closed-form model."
            if result.is_synthetic
            else (
                "2D k-omega SST RANS front-wing section (OpenFOAM) combined with "
                "empirical floor and rear-wing terms. Not a full-car CFD solution."
            )
        ),
        case_id=result.case_id,
        mesh_cells=result.mesh_stats.num_cells,
        iterations=result.convergence.iterations_run,
        cl_section=result.cl,
        cd_section=result.cd,
        residual_p=result.convergence.final_p_residual,
        residual_u=result.convergence.final_u_residual,
        execution_time_s=result.execution_time_s,
    )

    method_tag = "cfd_synthetic_model" if result.is_synthetic else "openfoam_rans_hybrid"

    sim = SimulationResult(
        config=config,
        forces=forces,
        lap_time_delta_s=round(lap_time_delta_s(drag_total_N, df_total_N), 4),
        method=method_tag,
        fidelity=provenance.fidelity,
        provenance=provenance.as_dict(),
    )
    return sim, provenance


def run_cfd(
    config: FormulaAeroConfig,
    case_id: str = "latest",
    mesh_level: MeshLevel = MeshLevel.MEDIUM,
) -> SimulationResult:
    """Run the front-wing CFD-family path and return the simulation result."""
    sim, _ = run_cfd_with_provenance(config, case_id=case_id, mesh_level=mesh_level)
    return sim
