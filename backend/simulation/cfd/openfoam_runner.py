"""OpenFOAM CFD simulation interface connecting API endpoints to the CFD pipeline."""

from __future__ import annotations

from pathlib import Path

from backend.models.aero_config import AeroForces, FormulaAeroConfig, SimulationResult
from backend.simulation.cfd.case_builder import build_openfoam_case
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel
from backend.simulation.cfd.runner import detect_cfd_backend, run_cfd_case


def is_openfoam_available() -> bool:
    """Check if OpenFOAM solver is available (native or WSL)."""
    return detect_cfd_backend() in ("native", "wsl")


def generate_case(config: FormulaAeroConfig, case_id: str) -> Path:
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
        mesh_level=MeshLevel.MEDIUM,
        speed_ms=config.speed_kmh / 3.6,
    )
    return case_dir


def run_cfd(config: FormulaAeroConfig, case_id: str = "latest") -> SimulationResult:
    """Run CFD simulation for the front wing section in ground effect.

    Uses native OpenFOAM if present, WSL OpenFOAM if present, or calibrated
    RANS-physics evaluator if OpenFOAM binary is unavailable.
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
        mesh_level=MeshLevel.MEDIUM,
        speed_ms=speed_ms,
    )

    # Dynamic pressure
    q = 0.5 * 1.225 * (speed_ms**2)

    # Front wing 3D estimate from 2D section (span b = 1.8m)
    span_m = config.front_wing_main.span_m
    df_front_N = result.cl * q * airfoil.chord_m * span_m
    drag_front_N = result.cd * q * airfoil.chord_m * span_m

    # Rear wing and underbody empirical balance
    rear_angle = config.rear_wing_main.angle_deg + config.rear_wing_flap.angle_deg
    df_rear_N = 0.08 * rear_angle * q * 1.0
    drag_rear_N = (0.012 + 0.003 * rear_angle) * q * 1.0

    df_total_N = df_front_N + df_rear_N
    drag_total_N = drag_front_N + drag_rear_N

    balance_front_pct = (df_front_N / df_total_N * 100.0) if df_total_N > 1.0 else 50.0
    efficiency = df_total_N / drag_total_N if drag_total_N > 1.0 else 1.0

    area_ref = 1.6  # m^2 reference frontal area
    cl_total = df_total_N / (q * area_ref) if q > 0 else 0.0
    cd_total = drag_total_N / (q * area_ref) if q > 0 else 0.0

    forces = AeroForces(
        drag_N=round(drag_total_N, 1),
        downforce_total_N=round(df_total_N, 1),
        downforce_front_N=round(df_front_N, 1),
        downforce_rear_N=round(df_rear_N, 1),
        balance_front_pct=round(balance_front_pct, 1),
        cop_x_m=round(1.6 * (balance_front_pct / 100.0), 3),
        cl_total=round(cl_total, 3),
        cd_total=round(cd_total, 3),
        efficiency=round(efficiency, 2),
    )

    method_tag = "openfoam_cfd" if is_openfoam_available() else "cfd_synthetic_evaluated"

    return SimulationResult(
        config=config,
        forces=forces,
        lap_time_delta_s=round((drag_total_N - df_total_N * 0.3) / 10000.0, 4),
        method=method_tag,
    )
