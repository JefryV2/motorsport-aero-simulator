"""Semi-empirical aerodynamic model for formula open-wheel cars.

Uses lifting-line approximations for wings, ground-effect models for
underbody/diffuser, and ride-height sensitivity curves calibrated
against published formula car aero data (order-of-magnitude accurate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

# Suction decay lengths (m) and the rake coefficient below are calibrated
# lengths, not tuning knobs: `underbody_cl` is the only reader in this file and
# backend/visualization/aero_field.py goes through it, so the pressure map cannot
# drift away from the forces the solver reports.
GE_DECAY_FRONT_M = 0.015
GE_DECAY_REAR_M = 0.020
GE_DECAY_DIFFUSER_M = 0.025
RAKE_COEF_PER_DEG = 0.08

# The two underbody ceilings: the Cl each surface carries at zero gap, before its
# own ground-effect decay. `underbody_cl` is the only reader, and the pressure map
# goes through it so the picture shades the load the forces actually charge.
CL_FLOOR_MAX = 0.35
CL_DIFFUSER_MAX = 0.25

# Ground-effect lift gain for a lifting element near the plane, priced from the
# clearance of that element's own section (see `ground_effect_lift_factor`). The
# two numbers are anchored: across the default front wing the area-weighted gain
# is 1.1214, so the whole model keeps the flat 1.12 augmentation it used to
# apply - it is now distributed by each element's gap instead of asserted, and
# the front ride-height knob moves the wings as well as the floor.
GE_LIFT_GAIN_MAX = 0.162
GE_LIFT_DECAY_M = 0.14

# Floor and diffuser stations in car-local metres from the nose. These mirror
# REFERENCE in frontend/js/carSpec.js, which draws the same lines; the tests in
# tests/test_physics_sanity.py assert the two do not drift apart.
FRONT_AXLE_M = 0.55
REAR_AXLE_M = 3.9
FLOOR_X_START_M = 0.62
FLOOR_X_END_M = 3.62
DIFFUSER_LENGTH_M = 0.6
# The drawn exit plane is capped here (carSpec.js `diffuserGeometry`). The force
# model has no exit-plane term, so this only bounds the display geometry.
DIFFUSER_EXIT_MAX_M = 0.32

# Wing mounting stations. `panel_method.simulate_panel` places its four VLM
# surfaces here and `visualization/aero_field.py` shades the same windows, so the
# solved wing, the drawn wing and the pressure map are one geometry. The
# car-height/endplate and tyre numbers below are the drawn envelope: the aero
# model has no chassis, so they exist to put a region edge on a real surface.
WING_LE_X_M = 0.20  # front wing main plane leading edge
REAR_WING_LE_X_M = WHEELBASE_M + WING_LE_X_M
WING_FLAP_STATION_FRAC = 0.75  # flap leading edge = main chord * this
FW_FLAP_STEP_UP_M = 0.035  # flap plane above the main plane
FW_GROUND_CLEARANCE_M = 0.005  # keep the pitched plate off the floor
REAR_WING_MAIN_HEIGHT_M = 0.85
RW_FLAP_STEP_UP_M = 0.05
RW_ENDPLATE_HEIGHT_M = 0.36
RW_ENDPLATE_DROP_M = 0.12  # plate centre below the main plane
REAR_WING_TOP_M = REAR_WING_MAIN_HEIGHT_M - RW_ENDPLATE_DROP_M + RW_ENDPLATE_HEIGHT_M / 2
REAR_WING_BOTTOM_M = REAR_WING_MAIN_HEIGHT_M - RW_ENDPLATE_DROP_M - RW_ENDPLATE_HEIGHT_M / 2

WHEEL_RADIUS_FRONT_M = 0.33
WHEEL_RADIUS_REAR_M = 0.35
WHEEL_WIDTH_FRONT_M = 0.30
WHEEL_WIDTH_REAR_M = 0.40
TRACK_M = 1.60
FLOOR_HALF_WIDTH_M = 0.70
# Nose to tail and ground to endplate crown: the box the reference car fills.
CAR_LENGTH_M = REAR_AXLE_M + WHEEL_RADIUS_REAR_M
CAR_HEIGHT_M = REAR_WING_TOP_M
CAR_HALF_WIDTH_M = 0.90  # max(front wing span, track) / 2 at the default setup

# Lap-time proxy sensitivities. `lap_time_delta_s` below is the only
# implementation the three solver tiers use, and frontend/js/carSpec.js mirrors
# the two coefficients so the UI's track-weighted delta is the same arithmetic.
LAP_BASELINE_DRAG_N = 1200.0
LAP_BASELINE_DF_N = 3500.0
LAP_SEC_PER_DRAG_N = 0.0001
LAP_SEC_PER_DF_N = 0.00008


def geometric_rake_deg(config: FormulaAeroConfig) -> float:
    """Body attitude implied by the two ride heights.

    Not `config.rake_deg`: that is a suction multiplier applied to the underbody
    terms and the car can be drawn at a different angle than it reports.
    """
    rise_mm = config.ride_height_rear_mm - config.ride_height_front_mm
    return math.degrees(math.atan(rise_mm / (WHEELBASE_M * 1000.0)))


def centre_of_pressure_m(df_front_N: float, df_total_N: float) -> float:
    """Aerodynamic centre as a station behind the front axle.

    The rear axle carries the share of downforce the centre of pressure has
    travelled back, so `x = wheelbase * (1 - front share)`. Every tier reports
    this: the CFD hybrid used to print `1.6 * front share`, a different lever on a
    different origin, which sent the reported CoP rearward as the car became more
    front-loaded.
    """
    if df_total_N <= 0.0:
        return WHEELBASE_M * 0.5
    return WHEELBASE_M * (1.0 - df_front_N / df_total_N)


def floor_height_at_m(config: FormulaAeroConfig, x_m: float) -> float:
    """Floor line height in metres above the ground plane at station `x_m`.

    The two ride heights at the two axles define the ramp, the same line the
    viewer draws, so a setup's rake shows up in the geometry and the forces.
    """
    h_front = config.ride_height_front_mm / 1000.0
    h_rear = config.ride_height_rear_mm / 1000.0
    slope = (h_rear - h_front) / (REAR_AXLE_M - FRONT_AXLE_M)
    return h_front + slope * (x_m - FRONT_AXLE_M)


def diffuser_throat_height_m(config: FormulaAeroConfig) -> float:
    """The throat the car can actually build, never below the floor line.

    The diffuser starts where the floor ends and runs rearward, so a throat
    setting under the floor height at that station is not a shape the car has.
    `carSpec.js` already raises the drawn throat to the floor line; the forces
    have to price the same one.
    """
    x_start = FLOOR_X_END_M - DIFFUSER_LENGTH_M
    return max(config.diffuser_ride_height_mm / 1000.0, floor_height_at_m(config, x_start))


def diffuser_exit_height_m(config: FormulaAeroConfig) -> float:
    """Top of the drawn diffuser ramp at its exit station.

    The angle only enters the forces through sin(angle), so the shape the model
    rewards is this ramp over `DIFFUSER_LENGTH_M`: the throat in front, the exit
    plane behind. `visualization/aero_field.py` ends the diffuser region here.
    """
    throat = diffuser_throat_height_m(config)
    rise = DIFFUSER_LENGTH_M * math.tan(math.radians(config.diffuser_angle_deg))
    return min(throat + rise, DIFFUSER_EXIT_MAX_M)


def lap_time_delta_s(
    drag_N: float,
    downforce_N: float,
    baseline_drag_N: float = LAP_BASELINE_DRAG_N,
    baseline_df_N: float = LAP_BASELINE_DF_N,
    drag_scale: float = 1.0,
    downforce_scale: float = 1.0,
) -> float:
    """Seconds per lap against a reference forces pair: a sensitivity, not a simulation.

    Linear in both terms, so it only ever compares two setups of one car, and the
    coefficients are fixed sensitivities rather than anything integrated over a
    track. The tiers pass no baseline of their own - one number, one rule - while
    the UI passes the setup it loaded as the baseline and the track's weighting.
    A negative result means the comparison setup is faster on this proxy. Rounding
    stays with the caller, which knows how many digits it displays.
    """
    return (drag_N - baseline_drag_N) * LAP_SEC_PER_DRAG_N * drag_scale - (
        downforce_N - baseline_df_N
    ) * LAP_SEC_PER_DF_N * downforce_scale


@dataclass(frozen=True)
class WingStation:
    """One lifting surface: where it sits and how big it is, in car metres."""

    name: str
    x_le_m: float  # leading edge station, nose at x = 0
    x_te_m: float  # trailing edge, measured along the chord line
    height_m: float  # leading edge height above the ground plane
    span_m: float
    chord_m: float
    angle_deg: float

    @property
    def min_clearance_m(self) -> float:
        """Height of the lowest point of this section over the ground plane.

        Positive incidence drops the trailing edge, and `wing_stations` lifts the
        leading edge by exactly that drop, so the trailing edge is what the plane
        sees. That is the gap ground effect is priced at - pricing the leading
        edge instead would take lift away from a wing for pitching its trailing
        edge down toward the floor.
        """
        drop = self.chord_m * max(0.0, math.sin(math.radians(self.angle_deg)))
        return self.height_m - drop


def wing_stations(config: FormulaAeroConfig) -> tuple[WingStation, ...]:
    """The four lifting surfaces as the solver places them.

    `panel_method.simulate_panel` builds its VLM surfaces straight from these, so
    the wing that is solved, the wing the viewer draws and the wing the API
    pressure map shades cannot be given three different geometries. The front
    wing rides on the front ride height and pitches its trailing edge down with
    incidence; the rear wing is hung at a fixed height above the reference plane.
    """
    fw_main = config.front_wing_main
    fw_flap = config.front_wing_flap
    rw_main = config.rear_wing_main
    rw_flap = effective_rear_flap(config)

    # simulate_panel: clear the plate by its own trailing-edge drop so a highly
    # incident front wing cannot sink through the ground plane.
    z_fw_main = (
        config.ride_height_front_mm / 1000.0
        + fw_main.chord_m * max(0.0, math.sin(math.radians(fw_main.angle_deg)))
        + FW_GROUND_CLEARANCE_M
    )
    z_fw_flap = (
        z_fw_main
        + FW_FLAP_STEP_UP_M
        + fw_flap.chord_m * max(0.0, math.sin(math.radians(fw_flap.angle_deg)))
    )
    x_fw_flap = WING_LE_X_M + fw_main.chord_m * WING_FLAP_STATION_FRAC
    x_rw_flap = REAR_WING_LE_X_M + rw_main.chord_m * WING_FLAP_STATION_FRAC

    return (
        WingStation(
            "front_wing_main",
            WING_LE_X_M,
            WING_LE_X_M + fw_main.chord_m,
            z_fw_main,
            fw_main.span_m,
            fw_main.chord_m,
            fw_main.angle_deg,
        ),
        WingStation(
            "front_wing_flap",
            x_fw_flap,
            x_fw_flap + fw_flap.chord_m,
            z_fw_flap,
            fw_flap.span_m,
            fw_flap.chord_m,
            fw_flap.angle_deg,
        ),
        WingStation(
            "rear_wing_main",
            REAR_WING_LE_X_M,
            REAR_WING_LE_X_M + rw_main.chord_m,
            REAR_WING_MAIN_HEIGHT_M,
            rw_main.span_m,
            rw_main.chord_m,
            rw_main.angle_deg,
        ),
        WingStation(
            "rear_wing_flap",
            x_rw_flap,
            x_rw_flap + rw_flap.chord_m,
            REAR_WING_MAIN_HEIGHT_M + RW_FLAP_STEP_UP_M,
            rw_flap.span_m,
            rw_flap.chord_m,
            rw_flap.angle_deg,
        ),
    )


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


def ground_effect_lift_factor(clearance_m: float | None) -> float:
    """Lift augmentation of an element whose lowest point is `clearance_m` up.

    None means the model does not place that element in ground effect (the rear
    wing). A negative or zero clearance - a setup that would run the plane through
    the floor - is clipped to the in-contact gain rather than extrapolated. The
    constants are anchored so the default front wing as a whole still reads the
    1.12 this model used to assert flat.
    """
    if clearance_m is None:
        return 1.0
    return 1.0 + GE_LIFT_GAIN_MAX * math.exp(-max(0.0, clearance_m) / GE_LIFT_DECAY_M)


# Post-stall branch. The thin-airfoil line grows with angle forever, so once it
# reaches its own ceiling the section is handed over to the flat-plate polar: a
# stalled element has lost its suction, and whatever lift it still makes comes
# from the normal force of a bluff plate - along with the pressure drag the
# attached model cannot see at all. `STALL_BLEND_MAX_DEG` is where the handover
# finishes, at the plate's own 45 deg lift peak. Below `STALL_ONSET_DEG` the
# blend weight is zero, so every angle the model priced before prices the same.
SECTION_CL_MAX = 1.8
STALL_BLEND_MAX_DEG = 45.0
FLAT_PLATE_CN = 2.0
STALL_ONSET_DEG = math.degrees(SECTION_CL_MAX / (2 * math.pi))


def _flat_plate_forces(alpha_deg: float) -> tuple[float, float]:
    """Lift and drag of the fully separated section, from its normal force."""
    a = math.radians(abs(alpha_deg))
    cn = FLAT_PLATE_CN * math.sin(a)
    return cn * math.cos(a), cn * math.sin(a)


def _stall_blend(alpha_deg: float) -> float:
    """How far past the knee the section is: 0 while attached, 1 as a plate."""
    excess = abs(alpha_deg) - STALL_ONSET_DEG
    if excess <= 0.0:
        return 0.0
    return min(1.0, excess / (STALL_BLEND_MAX_DEG - STALL_ONSET_DEG))


def _wing_cl(
    element: WingElement, clearance_m: float | None = None
) -> tuple[float, float]:
    """Estimate Cl and the drag factor (induced + separation) for a wing element.

    `clearance_m` is the element's trailing-edge gap to the ground plane, from
    `WingStation.min_clearance_m`; the front wing elements get their gain from
    `ground_effect_lift_factor`, the rear wing passes None.
    """
    alpha_rad = math.radians(element.angle_deg)
    blend = _stall_blend(element.angle_deg)
    plate_cl, plate_cd = (
        _flat_plate_forces(element.angle_deg) if blend > 0.0 else (0.0, 0.0)
    )
    if blend > 0.0:
        sign = 1.0 if element.angle_deg >= 0.0 else -1.0
        cl_2d = sign * ((1.0 - blend) * SECTION_CL_MAX + blend * plate_cl)
    else:
        # Thin-airfoil slope. The cap below is exactly where the blend starts,
        # so it never bites on this branch.
        cl_2d = max(-0.5, min(2 * math.pi * alpha_rad, SECTION_CL_MAX))

    # Aspect ratio correction (Prandtl)
    ar = element.span_m / element.chord_m
    e = 0.85  # Oswald efficiency
    cl_3d = cl_2d / (1 + cl_2d / (math.pi * ar * e))

    # Gurney flap boost (~0.08 Cl per 10mm at racing Re)
    cl_3d += (element.gurney_mm / 10.0) * 0.08

    cl_3d *= ground_effect_lift_factor(clearance_m)

    cd_factor = cl_3d**2 / (math.pi * ar * e) + blend * plate_cd
    return cl_3d, cd_factor


def wing_section_cl(element: WingElement, clearance_m: float | None = None) -> float:
    """The Cl `simulate` charges this one element, tip loss, Gurney and ground effect in.

    Picture-making consumers (the pressure map) need the same number the forces
    used, and clearance follows the rule `_wing_forces` passes it: the element's
    trailing-edge gap for a wing in ground effect, None for one in freestream.
    """
    return _wing_cl(element, clearance_m)[0]


def _wing_forces(
    element: WingElement, speed_ms: float, clearance_m: float | None = None
) -> tuple[float, float]:
    """Return (downforce_N, drag_N) for a wing element."""
    cl, cd_extra = _wing_cl(element, clearance_m)
    q = 0.5 * RHO * speed_ms**2
    area = element.chord_m * element.span_m
    lift = q * area * cl  # downforce-positive convention for the empirical wing model
    drag = q * area * (0.008 + cd_extra)  # profile + induced + separated-flow drag
    return lift, drag


def underbody_cl(config: FormulaAeroConfig) -> tuple[float, float]:
    """The (floor, diffuser) Cl the underbody is charged with.

    `_underbody_forces` prices the downforce from these two numbers and
    `backend/visualization/aero_field.py` shades the same two surfaces from them,
    so a ride-height or rake move cannot load the floor in the forces and leave the
    picture on the old number.
    """
    ge_rear = math.exp(-config.ride_height_rear_mm / 1000.0 / GE_DECAY_REAR_M)
    rake_boost = 1.0 + config.rake_deg * RAKE_COEF_PER_DEG
    cl_floor = CL_FLOOR_MAX * ge_rear * rake_boost
    cl_diff = (
        CL_DIFFUSER_MAX
        * math.sin(math.radians(config.diffuser_angle_deg))
        * math.exp(-diffuser_throat_height_m(config) / GE_DECAY_DIFFUSER_M)
        * rake_boost
    )
    return cl_floor, cl_diff


def _underbody_forces(config: FormulaAeroConfig, speed_ms: float) -> tuple[float, float, float]:
    """Ground-effect floor and diffuser contribution."""
    q = 0.5 * RHO * speed_ms**2

    # Ride height sensitivity (exponential ground effect)
    h_front = config.ride_height_front_mm / 1000.0
    h_rear = config.ride_height_rear_mm / 1000.0

    ge_front = math.exp(-h_front / GE_DECAY_FRONT_M)
    ge_rear = math.exp(-h_rear / GE_DECAY_REAR_M)

    floor_area = 2.2  # m² effective underbody
    cl_floor, cl_diff = underbody_cl(config)
    cd_floor = 0.015 + 0.02 * (1.0 - ge_rear)
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

    # Front wing (both elements, in ground effect at their own trailing-edge gap)
    stations = {s.name: s for s in wing_stations(config)}
    df_fw_main, drag_fw_main = _wing_forces(
        config.front_wing_main, speed_ms, stations["front_wing_main"].min_clearance_m
    )
    df_fw_flap, drag_fw_flap = _wing_forces(
        config.front_wing_flap, speed_ms, stations["front_wing_flap"].min_clearance_m
    )
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
    cop_x = centre_of_pressure_m(df_front, df_total)

    q = 0.5 * RHO * speed_ms**2
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
        fidelity="empirical",
        provenance={
            "source": "semi-empirical correlation",
            "solver": "none",
            "validated_externally": False,
            "notes": "Analytical baseline. Useful for trends, not a mesh solution.",
        },
    )
