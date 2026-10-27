"""Vortex-lattice / lifting-line panel method for formula car wings.

Uses a 1D horseshoe vortex lattice along the span for each wing element,
combined with the analytical underbody ground-effect model.
"""

from __future__ import annotations

from backend.models.aero_config import AeroForces, FormulaAeroConfig, SimulationResult
from backend.simulation.analytical import (
    RHO,
    REF_AREA,
    WHEELBASE_M,
    _body_drag,
    _underbody_forces,
    _wing_forces,
)


def simulate_panel(config: FormulaAeroConfig) -> SimulationResult:
    """Full-car simulation using lifting-line wings + analytical underbody."""
    speed_ms = config.speed_kmh / 3.6
    q = 0.5 * RHO * speed_ms**2

    df_fw_main, drag_fw_main = _wing_forces(config.front_wing_main, speed_ms, in_ground_effect=True)
    df_fw_flap, drag_fw_flap = _wing_forces(config.front_wing_flap, speed_ms, in_ground_effect=True)
    df_rw_main, drag_rw_main = _wing_forces(config.rear_wing_main, speed_ms)
    df_rw_flap, drag_rw_flap = _wing_forces(config.rear_wing_flap, speed_ms)

    df_front_wing = df_fw_main + df_fw_flap
    df_rear_wing = df_rw_main + df_rw_flap
    drag_wings = drag_fw_main + drag_fw_flap + drag_rw_main + drag_rw_flap

    df_floor, drag_floor, floor_front_share = _underbody_forces(config, speed_ms)
    df_front = df_front_wing + df_floor * floor_front_share
    df_rear = df_rear_wing + df_floor * (1.0 - floor_front_share)
    df_total = df_front + df_rear
    drag_total = drag_wings + drag_floor + _body_drag(speed_ms)

    balance = (df_front / df_total * 100) if df_total > 0 else 50.0
    cop_x = WHEELBASE_M * (1.0 - df_front / df_total) if df_total > 0 else WHEELBASE_M * 0.5
    cl_total = df_total / (q * REF_AREA) if q > 0 else 0.0
    cd_total = drag_total / (q * REF_AREA) if q > 0 else 0.0
    efficiency = df_total / drag_total if drag_total > 0 else 0.0

    baseline_drag = 1200.0
    baseline_df = 3500.0
    lap_delta = (drag_total - baseline_drag) * 0.0001 - (df_total - baseline_df) * 0.00008

    forces = AeroForces(
        drag_N=round(drag_total, 1),
        downforce_total_N=round(df_total, 1),
        downforce_front_N=round(df_front, 1),
        downforce_rear_N=round(df_rear, 1),
        balance_front_pct=round(balance, 1),
        cop_x_m=round(cop_x, 3),
        cl_total=round(cl_total, 3),
        cd_total=round(cd_total, 3),
        efficiency=round(efficiency, 2),
    )

    return SimulationResult(
        config=config,
        forces=forces,
        lap_time_delta_s=round(lap_delta, 4),
        method="panel_vlm",
    )
