"""Semi-empirical aerodynamic model for formula open-wheel cars.

Uses lifting-line approximations for wings, ground-effect models for
underbody/diffuser, and ride-height sensitivity curves calibrated
against published formula car aero data (order-of-magnitude accurate).
"""

from __future__ import annotations

import math

from backend.models.aero_config import (
    AeroForces,
    FormulaAeroConfig,
    SimulationResult,
    WingElement,
)

RHO = 1.225  # air density kg/m³
REF_AREA = 1.5  # reference frontal area m²
WHEELBASE_M = 3.35  # typical formula car wheelbase
DRS_FLAP_ANGLE_DEG = 2.0  # rear flap angle once DRS is open


def geometric_rake_deg(config: FormulaAeroConfig) -> float:
    """Body attitude implied by the two ride heights.

    Not `config.rake_deg`: that is a suction multiplier applied to the underbody
    terms and the car can be drawn at a different angle than it reports.
    """
    rise_mm = config.ride_height_rear_mm - config.ride_height_front_mm
    return math.degrees(math.atan(rise_mm / (WHEELBASE_M * 1000.0)))


def effective_rear_flap(config: FormulaAeroConfig) -> WingElement:
    """The rear flap as the model sees it: DRS folds it toward the main plane.

    Opening the DRS gap drops the flap's effective incidence, so the model caps
    it at DRS_FLAP_ANGLE_DEG rather than inventing a separate drag term. A flap
    already set below that angle cannot be reduced further, so DRS is a no-op.
    """
    flap = config.rear_wing_flap
    if not config.drs_active or flap.angle_deg <= DRS_FLAP_ANGLE_DEG:
        return flap
    return flap.model_copy(update={"angle_deg": DRS_FLAP_ANGLE_DEG})


def _wing_cl(element: WingElement, in_ground_effect: bool = False) -> tuple[float, float]:
    """Estimate Cl and induced drag factor for a wing element."""
    alpha_rad = math.radians(element.angle_deg)
    # Thin-airfoil slope with stall softening
    cl_2d = 2 * math.pi * alpha_rad
    cl_2d = max(-0.5, min(cl_2d, 1.8))

    # Aspect ratio correction (Prandtl)
    ar = element.span_m / element.chord_m
    e = 0.85  # Oswald efficiency
    cl_3d = cl_2d / (1 + cl_2d / (math.pi * ar * e))

    # Gurney flap boost (~0.08 Cl per 10mm at racing Re)
    cl_3d += (element.gurney_mm / 10.0) * 0.08

    if in_ground_effect:
        cl_3d *= 1.12  # ground effect augmentation on front wing

    cd_i_factor = cl_3d**2 / (math.pi * ar * e)
    return cl_3d, cd_i_factor


def _wing_forces(
    element: WingElement, speed_ms: float, in_ground_effect: bool = False
) -> tuple[float, float]:
    """Return (downforce_N, drag_N) for a wing element."""
    cl, cd_i = _wing_cl(element, in_ground_effect)
    q = 0.5 * RHO * speed_ms**2
    area = element.chord_m * element.span_m
    lift = q * area * cl  # downforce-positive convention for the empirical wing model
    drag = q * area * (0.008 + cd_i)  # profile + induced drag
    return lift, drag


def _underbody_forces(config: FormulaAeroConfig, speed_ms: float) -> tuple[float, float, float]:
    """Ground-effect floor and diffuser contribution."""
    q = 0.5 * RHO * speed_ms**2

    # Ride height sensitivity (exponential ground effect)
    h_front = config.ride_height_front_mm / 1000.0
    h_rear = config.ride_height_rear_mm / 1000.0
    h_diff = config.diffuser_ride_height_mm / 1000.0

    ge_front = math.exp(-h_front / 0.015)
    ge_rear = math.exp(-h_rear / 0.020)
    rake_boost = 1.0 + config.rake_deg * 0.08

    floor_area = 2.2  # m² effective underbody
    cl_floor = 0.35 * ge_rear * rake_boost
    cd_floor = 0.015 + 0.02 * (1.0 - ge_rear)

    # Diffuser
    diff_angle_rad = math.radians(config.diffuser_angle_deg)
    cl_diff = 0.25 * math.sin(diff_angle_rad) * math.exp(-h_diff / 0.025) * rake_boost
    cd_diff = 0.008

    cl_total = cl_floor + cl_diff
    cd_total = cd_floor + cd_diff

    downforce = q * floor_area * cl_total
    drag = q * floor_area * cd_total

    # Split front/rear based on ride height distribution
    front_share = 0.35 * ge_front / (ge_front + ge_rear)
    return downforce, drag, front_share


def _body_drag(speed_ms: float) -> float:
    """Baseline body/wheels/drag from open-wheel formula car."""
    q = 0.5 * RHO * speed_ms**2
    cd_body = 0.75  # wheels, suspension, cockpit
    return q * REF_AREA * cd_body


def simulate(config: FormulaAeroConfig, method: str = "analytical") -> SimulationResult:
    """Run full-car aero simulation."""
    speed_ms = config.speed_kmh / 3.6

    # Front wing (both elements, in ground effect)
    df_fw_main, drag_fw_main = _wing_forces(config.front_wing_main, speed_ms, in_ground_effect=True)
    df_fw_flap, drag_fw_flap = _wing_forces(config.front_wing_flap, speed_ms, in_ground_effect=True)
    df_front_wing = df_fw_main + df_fw_flap
    drag_front_wing = drag_fw_main + drag_fw_flap

    # Rear wing (both elements, freestream)
    df_rw_main, drag_rw_main = _wing_forces(config.rear_wing_main, speed_ms)
    df_rw_flap, drag_rw_flap = _wing_forces(effective_rear_flap(config), speed_ms)
    df_rear_wing = df_rw_main + df_rw_flap
    drag_rear_wing = drag_rw_main + drag_rw_flap

    # Underbody
    df_floor, drag_floor, floor_front_share = _underbody_forces(config, speed_ms)
    df_floor_front = df_floor * floor_front_share
    df_floor_rear = df_floor * (1.0 - floor_front_share)

    # Totals
    df_front = df_front_wing + df_floor_front
    df_rear = df_rear_wing + df_floor_rear
    df_total = df_front + df_rear
    drag_total = drag_front_wing + drag_rear_wing + drag_floor + _body_drag(speed_ms)

    balance = (df_front / df_total * 100) if df_total > 0 else 50.0
    cop_x = WHEELBASE_M * (1.0 - df_front / df_total) if df_total > 0 else WHEELBASE_M * 0.5

    q = 0.5 * RHO * speed_ms**2
    cl_total = df_total / (q * REF_AREA) if q > 0 else 0.0
    cd_total = drag_total / (q * REF_AREA) if q > 0 else 0.0
    efficiency = df_total / drag_total if drag_total > 0 else 0.0

    # Lap time proxy: aero drag penalty + downforce cornering gain
    # Simplified: each 10N drag ≈ 0.001s/lap, each 100N DF ≈ -0.008s/lap at typical circuit
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
        fidelity="empirical",
        provenance={
            "source": "semi-empirical correlation",
            "solver": "none",
            "validated_externally": False,
            "notes": "Analytical baseline. Useful for trends, not a mesh solution.",
        },
    )
