"""Vortex-lattice panel method for multi-element formula car wings in ground effect.

Couples true 3D vortex lattice physics (horseshoe vortices, Biot-Savart influence,
image ground plane, Trefftz drag) with the underbody ground-effect floor model.
"""

from __future__ import annotations

from backend.models.aero_config import AeroForces, FormulaAeroConfig, SimulationResult
from backend.simulation.analytical import (
    REF_AREA,
    RHO,
    WHEELBASE_M,
    _body_drag,
    _underbody_forces,
    lap_time_delta_s,
    wing_stations,
)
from backend.simulation.vlm.core import VLMSurface, solve_vlm

# Mean camber height ratio per element (keyed by analytical.wing_stations names).
_VLM_CAMBER = {
    "front_wing_main": 0.03,
    "front_wing_flap": 0.05,
    "rear_wing_main": 0.04,
    "rear_wing_flap": 0.06,
}


def simulate_panel(
    config: FormulaAeroConfig,
    method: str = "analytical_enhanced",
    n_span: int = 10,
    n_chord: int = 2,
) -> SimulationResult:
    """Full-car simulation using 3D Vortex Lattice wings + ground effect underbody."""
    speed_ms = config.speed_kmh / 3.6
    q = 0.5 * RHO * speed_ms**2

    # Build multi-element wing surfaces at the shared mounting stations
    wings = [
        VLMSurface(
            name=station.name,
            origin=(station.x_le_m, 0.0, station.height_m),
            span=station.span_m,
            chord=station.chord_m,
            angle_deg=station.angle_deg,
            camber=_VLM_CAMBER[station.name],
            n_span=n_span,
            n_chord=n_chord,
        )
        for station in wing_stations(config)
    ]

    # Solve coupled 3D VLM with ground plane at z=0.0
    vlm_sol = solve_vlm(
        wings,
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

    lap_delta = lap_time_delta_s(drag_total, df_total)

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
