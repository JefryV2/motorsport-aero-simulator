"""Validation Case V10: lap simulation against analytic limits.

Checks the Phase 5 lap core against closed-form results where they exist:

* curvature recovery on a known circle,
* constant-radius skidpad with and without downforce,
* top speed from the power/drag balance,
* constant-power acceleration (v^3 grows linearly with distance),
* braking distance into a known corner,
* the g-g envelope is respected everywhere on a tightening spiral,
* aero balance (CoP split) changes axle grip through load sensitivity,
* malformed centrelines are rejected rather than repaired.

The analytic geometries are generated inside this script. They are not circuit
data and are never used by the application. No numbers are hard-coded from a
real track, and a failed check is reported as a failure.

Usage:
    .venv\\Scripts\\python.exe validation\\v10_lap_sim.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import brentq

from backend.simulation.lap.aero_map import ConstantAeroMap
from backend.simulation.lap.solver import LapSettings, solve_lap
from backend.simulation.lap.track import TrackError, build_track
from backend.simulation.lap.track_io import parse_centerline_csv
from backend.simulation.lap.vehicle import Vehicle

G = 9.80665
RHO = 1.225
OUT = Path("data/datasets/v10_lap_sim.json")


@dataclass
class Check:
    name: str
    measured: float
    expected: float
    unit: str
    tolerance_pct: float
    error_pct: float
    passed: bool


def _check(
    name: str,
    measured: float,
    expected: float,
    unit: str,
    tolerance_pct: float,
) -> Check:
    error = abs(measured - expected) / abs(expected) * 100.0
    return Check(
        name=name,
        measured=float(measured),
        expected=float(expected),
        unit=unit,
        tolerance_pct=tolerance_pct,
        error_pct=float(error),
        passed=bool(error <= tolerance_pct),
    )


def _at_most(
    name: str, measured: float, limit: float, unit: str, tolerance_pct: float = 0.0
) -> Check:
    """Pass when ``measured <= limit`` (excess is the only error)."""
    error = max(0.0, measured - limit) / abs(limit) * 100.0
    return Check(
        name=name,
        measured=float(measured),
        expected=float(limit),
        unit=unit,
        tolerance_pct=tolerance_pct,
        error_pct=float(error),
        passed=bool(error <= tolerance_pct),
    )


def _at_least(name: str, measured: float, floor: float, unit: str) -> Check:
    """Pass when ``measured >= floor`` (shortfall is the only error)."""
    error = max(0.0, floor - measured) / abs(floor) * 100.0
    return Check(
        name=name,
        measured=float(measured),
        expected=float(floor),
        unit=unit,
        tolerance_pct=0.0,
        error_pct=float(error),
        passed=bool(error <= 0.0),
    )


def circle_points(radius_m: float, n: int = 720) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [
        [radius_m * math.cos(t), radius_m * math.sin(t)] for t in theta
    ]


def straight_then_arc(
    straight_m: float, radius_m: float, ds_m: float = 0.5, arc_deg: float = 90.0
) -> list[list[float]]:
    """Straight along +x, then a left-hand arc of ``radius_m``."""
    n_straight = max(2, int(round(straight_m / ds_m)) + 1)
    points = [[float(x), 0.0] for x in np.linspace(0.0, straight_m, n_straight)]
    arc_length = radius_m * math.radians(arc_deg)
    n_arc = max(2, int(round(arc_length / ds_m)) + 1)
    angles = np.linspace(-math.pi / 2.0, -math.pi / 2.0 + math.radians(arc_deg), n_arc)
    for angle in angles:
        points.append(
            [
                straight_m + radius_m * math.cos(angle),
                radius_m + radius_m * math.sin(angle),
            ]
        )
    return points


def spiral_points(
    r_start_m: float = 70.0,
    r_end_m: float = 18.0,
    turns: float = 1.25,
    n: int = 900,
) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * math.pi * turns, n)
    radius = r_start_m + (r_end_m - r_start_m) * theta / theta[-1]
    return [
        [float(r * math.cos(t)), float(r * math.sin(t))]
        for r, t in zip(radius, theta, strict=False)
    ]


def run_v10_validation() -> dict:
    checks: list[Check] = []

    print("=== Case V10: Lap Simulation Analytic Cases ===")
    print("Quasi-steady point mass, load-sensitive tyres, g-g envelope.")
    print("Analytic geometries in this script are test fixtures, not circuit data.\n")

    # 1. Curvature recovery -------------------------------------------------
    radius = 25.0
    track = build_track(circle_points(radius), closed=True)
    measured_radius = 1.0 / float(np.mean(np.abs(track.curvature_1pm)))
    checks.append(
        _check("curvature radius (circle)", measured_radius, radius, "m", 0.5)
    )

    # 2. Skidpad, no aero ---------------------------------------------------
    mu = 1.5
    radius = 30.0
    vehicle = Vehicle(
        mass_kg=800.0,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    aero = ConstantAeroMap()
    result = solve_lap(
        build_track(circle_points(radius), closed=True), vehicle, aero
    )
    v_expected = math.sqrt(mu * G * radius)
    t_expected = 2.0 * math.pi * radius / v_expected
    checks.append(_check("skidpad speed (no aero)", result.speed_ms.mean(), v_expected, "m/s", 0.5))
    checks.append(_check("skidpad lap time (no aero)", result.lap_time_s, t_expected, "s", 0.5))

    # 3. Skidpad with downforce --------------------------------------------
    cla = 3.0
    vehicle_df = Vehicle(
        mass_kg=800.0,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    aero_df = ConstantAeroMap(downforce_area_m2=cla, front_share=0.5)
    result = solve_lap(
        build_track(circle_points(radius), closed=True), vehicle_df, aero_df
    )
    v_df_expected = math.sqrt(
        mu * G * radius / (1.0 - mu * radius * (0.5 * RHO * cla) / 800.0)
    )
    t_df_expected = 2.0 * math.pi * radius / v_df_expected
    checks.append(
        _check("skidpad speed (downforce)", result.speed_ms.mean(), v_df_expected, "m/s", 0.5)
    )
    checks.append(
        _check("skidpad lap time (downforce)", result.lap_time_s, t_df_expected, "s", 0.5)
    )

    # 4. Top speed ----------------------------------------------------------
    power_kw = 200.0
    efficiency = 0.9
    cda = 1.2
    crr = 0.01
    mass = 800.0
    top_vehicle = Vehicle(
        mass_kg=mass,
        power_kw=power_kw,
        drivetrain_efficiency=efficiency,
        tyre_mu=1.6,
        tyre_load_sensitivity=0.0,
        rolling_resistance=crr,
        max_speed_ms=100.0,
    )
    top_aero = ConstantAeroMap(drag_area_m2=cda)
    p_eff = power_kw * 1000.0 * efficiency
    v_top_expected = brentq(
        lambda v: p_eff - (0.5 * RHO * cda * v * v + crr * mass * G) * v,
        1.0,
        120.0,
    )
    straight = [[float(x), 0.0] for x in np.arange(0.0, 5000.0, 2.0)]
    result = solve_lap(
        build_track(straight, closed=False, ds_m=2.0),
        top_vehicle,
        top_aero,
        LapSettings(initial_speed_ms=40.0),
    )
    checks.append(
        _check("straight-line top speed", result.speed_ms.max(), v_top_expected, "m/s", 0.5)
    )
    power_balance = (
        0.5 * RHO * cda * top_vehicle.top_speed(top_aero) ** 3
        + crr * mass * G * top_vehicle.top_speed(top_aero)
    )
    checks.append(
        _check("power balance at top speed", power_balance, p_eff, "W", 0.5)
    )

    # 5. Constant-power acceleration ---------------------------------------
    power_vehicle = Vehicle(
        mass_kg=800.0,
        power_kw=power_kw,
        drivetrain_efficiency=efficiency,
        tyre_mu=2.5,
        tyre_load_sensitivity=0.0,
        rolling_resistance=0.0,
        max_speed_ms=100.0,
    )
    accel_track = [[float(x), 0.0] for x in np.linspace(0.0, 400.0, 801)]
    v0 = 25.0
    result = solve_lap(
        build_track(accel_track, closed=False, ds_m=0.5),
        power_vehicle,
        ConstantAeroMap(),
        LapSettings(initial_speed_ms=v0),
    )
    v_power_expected = (v0**3 + 3.0 * p_eff * 400.0 / 800.0) ** (1.0 / 3.0)
    checks.append(
        _check(
            "constant-power speed at 400 m",
            result.speed_ms[-1],
            v_power_expected,
            "m/s",
            1.0,
        )
    )

    # 6. Braking distance into a known corner ------------------------------
    brake_vehicle = Vehicle(
        mass_kg=800.0,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    corner_radius = 40.0
    v_entry = 60.0
    v_corner = math.sqrt(mu * G * corner_radius)
    braking_distance = (v_entry**2 - v_corner**2) / (2.0 * mu * G)
    track = build_track(
        straight_then_arc(300.0, corner_radius, ds_m=0.5), closed=False, ds_m=0.5
    )
    result = solve_lap(
        track, brake_vehicle, ConstantAeroMap(), LapSettings(initial_speed_ms=v_entry)
    )
    brake_start = float(result.s_m[np.argmax(result.speed_ms < v_entry - 0.01)])
    expected_start = 300.0 - braking_distance
    checks.append(
        _check(
            "braking start before corner",
            brake_start,
            expected_start,
            "m",
            1.5,
        )
    )
    checks.append(
        _check("minimum speed in corner", result.speed_ms.min(), v_corner, "m/s", 0.5)
    )

    # 7. g-g envelope on a tightening spiral -------------------------------
    spiral_vehicle = Vehicle(
        mass_kg=800.0,
        tyre_mu=1.6,
        tyre_load_sensitivity=0.1,
        power_kw=450.0,
        rolling_resistance=0.012,
    )
    spiral_aero = ConstantAeroMap(downforce_area_m2=3.0, drag_area_m2=1.1)
    result = solve_lap(
        build_track(spiral_points(), closed=False, ds_m=0.5), spiral_vehicle, spiral_aero
    )
    checks.append(
        _at_most(
            "max g-g utilisation on spiral",
            result.max_utilisation,
            1.0,
            "-",
            2.0,
        )
    )
    checks.append(
        _at_most(
            "speed cap respected",
            float(result.speed_ms.max()),
            min(spiral_vehicle.max_speed_ms, 120.0),
            "m/s",
            0.0,
        )
    )

    # 8. CoP balance changes axle grip -------------------------------------
    # With load-sensitive tyres the total grip is maximised when all four
    # tyres carry the same load. For a given total downforce that happens at
    #   s* = 0.5 + (static_rear - static_front) / (2 * downforce)
    # which is a closed-form target for the CoP split.
    balance_vehicle = Vehicle(tyre_load_sensitivity=0.3, tyre_mu=1.6)
    speed = 60.0
    share_sweep = np.round(np.arange(0.20, 0.91, 0.01), 2)
    grip = []
    for share in share_sweep:
        aero = ConstantAeroMap(downforce_area_m2=6.0, front_share=float(share))
        grip.append(balance_vehicle.lateral_limit(speed, aero.at(speed)))
    grip = np.asarray(grip)
    best_share = float(share_sweep[int(np.argmax(grip))])
    downforce = ConstantAeroMap(downforce_area_m2=6.0).at(speed).downforce_N
    expected_share = 0.5 + (
        balance_vehicle.static_rear_N - balance_vehicle.static_front_N
    ) / (2.0 * downforce)
    ay_at_85 = balance_vehicle.lateral_limit(
        speed, ConstantAeroMap(downforce_area_m2=6.0, front_share=0.85).at(speed)
    )
    checks.append(
        _check("CoP share that maximises grip", best_share, expected_share, "-", 3.0)
    )
    checks.append(
        _at_least(
            "grip at load-matched CoP vs 0.85",
            float(grip.max() / ay_at_85),
            1.005,
            "ratio",
        )
    )
    print(
        f"  CoP sweep: best share {best_share:.2f} (analytic {expected_share:.3f}), "
        f"grip {grip.max():.3f} vs {ay_at_85:.3f} m/s2 at share 0.85"
    )

    # 9. Inputs are validated, never repaired ------------------------------
    validation_failures = 0
    try:
        build_track([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    except TrackError:
        validation_failures += 1
    try:
        build_track([[0.0, 0.0]] * 20)
    except TrackError:
        validation_failures += 1
    try:
        build_track([[0.0, 0.0], [float("nan"), 1.0]] * 10)
    except TrackError:
        validation_failures += 1
    csv_text = "x,y\n" + "\n".join(
        f"{30 * math.cos(t):.6f},{30 * math.sin(t):.6f}"
        for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)
    )
    csv_points = parse_centerline_csv(csv_text)
    parsed_ok = len(csv_points) == 40
    checks.append(
        _check(
            "invalid centreline rejection",
            validation_failures,
            3.0,
            "cases",
            0.0,
        )
    )
    checks.append(_check("CSV parsing rows", len(csv_points), 40.0, "rows", 0.0))
    print(f"  CSV parser kept {len(csv_points)} rows (expected 40): {parsed_ok}")

    # Report ---------------------------------------------------------------
    print("\n--- Analytic checks ---")
    print(
        f"{'Check':<36} {'Measured':>12} {'Expected':>12} "
        f"{'Error %':>9} {'Tol %':>7} {'Result':>7}"
    )
    print("-" * 92)
    for check in checks:
        print(
            f"{check.name:<36} {check.measured:>12.4f} {check.expected:>12.4f} "
            f"{check.error_pct:>9.3f} {check.tolerance_pct:>7.2f} "
            f"{'PASS' if check.passed else 'FAIL':>7}"
        )

    targets_met = all(check.passed for check in checks)
    if targets_met:
        print("\n>>> Case V10: all analytic checks passed.")
    else:
        print("\n>>> Case V10: analytic checks FAILED. Reported as a failure.")
    print(
        "\nScope: this validates the lap-simulation mathematics, not the aero "
        "model. Absolute lap times depend on the aero map and the supplied "
        "centreline; no real-circuit result is claimed here.\n"
    )

    scorecard = {
        "targets_met": bool(targets_met),
        "checks": [asdict(check) for check in checks],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    print(f"Scorecard written to {OUT}\n")
    return scorecard


if __name__ == "__main__":
    run_v10_validation()
