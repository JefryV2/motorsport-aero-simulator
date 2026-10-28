"""Vortex-lattice panel method for multi-element formula car wings in ground effect.

Couples true 3D vortex lattice physics (horseshoe vortices, Biot-Savart influence,
image ground plane, Trefftz drag) with the underbody ground-effect floor model.
"""

from __future__ import annotations

import math

from backend.models.aero_config import AeroForces, FormulaAeroConfig, SimulationResult
from backend.simulation.analytical import (
    REF_AREA,
    RHO,
    WHEELBASE_M,
    _body_drag,
    _underbody_forces,
)
from backend.simulation.vlm.core import VLMSurface, solve_vlm


def simulate_panel(
    config: FormulaAeroConfig,
    method: str = "analytical_enhanced",
    n_span: int = 10,
    n_chord: int = 2,
) -> SimulationResult:
    """Full-car simulation using 3D Vortex Lattice wings + ground effect underbody."""
    speed_ms = config.speed_kmh / 3.6
    q = 0.5 * RHO * speed_ms**2

    h_front = config.ride_height_front_mm / 1000.0
    config.ride_height_rear_mm / 1000.0

    # Ensure front wing elements maintain positive ground clearance
    alpha_main_rad = math.radians(config.front_wing_main.angle_deg)
    alpha_flap_rad = math.radians(config.front_wing_flap.angle_deg)
    z_fw_main = h_front + config.front_wing_main.chord_m * max(0.0, math.sin(alpha_main_rad)) + 0.005
    z_fw_flap = z_fw_main + 0.035 + config.front_wing_flap.chord_m * max(0.0, math.sin(alpha_flap_rad))

    # Build multi-element wing surfaces
    fw_main = VLMSurface(
        name="front_wing_main",
        origin=(0.20, 0.0, z_fw_main),
        span=config.front_wing_main.span_m,
        chord=config.front_wing_main.chord_m,
        angle_deg=config.front_wing_main.angle_deg,
        camber=0.03,
        n_span=n_span,
        n_chord=n_chord,
    )
    fw_flap = VLMSurface(
        name="front_wing_flap",
        origin=(0.20 + config.front_wing_main.chord_m * 0.75, 0.0, z_fw_flap),
        span=config.front_wing_flap.span_m,
        chord=config.front_wing_flap.chord_m,
        angle_deg=config.front_wing_flap.angle_deg,
        camber=0.05,
        n_span=n_span,
        n_chord=n_chord,
    )
    rw_main = VLMSurface(
        name="rear_wing_main",
        origin=(WHEELBASE_M + 0.20, 0.0, 0.85),
        span=config.rear_wing_main.span_m,
        chord=config.rear_wing_main.chord_m,
        angle_deg=config.rear_wing_main.angle_deg,
        camber=0.04,
        n_span=n_span,
        n_chord=n_chord,
    )
    rw_flap = VLMSurface(
        name="rear_wing_flap",
        origin=(WHEELBASE_M + 0.20 + config.rear_wing_main.chord_m * 0.75, 0.0, 0.85 + 0.05),
        span=config.rear_wing_flap.span_m,
        chord=config.rear_wing_flap.chord_m,
        angle_deg=config.rear_wing_flap.angle_deg,
        camber=0.06,
        n_span=n_span,
        n_chord=n_chord,
    )

    # Solve coupled 3D VLM with ground plane at z=0.0
    vlm_sol = solve_vlm(
        [fw_main, fw_flap, rw_main, rw_flap],
        speed_kmh=config.speed_kmh,
        ground_height=0.0,
        use_symmetry=True,
    )

    df_front_wing = (
        vlm_sol.sections["front_wing_main"].downforce_N
        + vlm_sol.sections["front_wing_flap"].downforce_N
    )
    df_rear_wing = (
        vlm_sol.sections["rear_wing_main"].downforce_N
        + vlm_sol.sections["rear_wing_flap"].downforce_N
    )
    drag_wings = vlm_sol.total_drag_N

    # Underbody forces
    df_floor, drag_floor, floor_front_share = _underbody_forces(config, speed_ms)
    df_front = df_front_wing + df_floor * floor_front_share
    df_rear = df_rear_wing + df_floor * (1.0 - floor_front_share)
    df_total = df_front + df_rear
    drag_total = drag_wings + drag_floor + _body_drag(speed_ms)

    balance = (df_front / df_total * 100.0) if df_total > 0 else 50.0
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
        method=method,
        fidelity="low_fidelity_vlm",
        provenance={
            "source": "3D vortex lattice + empirical underbody",
            "solver": "vlm_horseshoe_biot_savart",
            "ground_effect": "image system",
            "validated_externally": False,
            "notes": "Inviscid VLM with empirical floor/body terms. Not a CFD solution.",
        },
    )
