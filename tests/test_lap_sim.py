"""Fast tests for the Phase 5 lap simulation core.

These cover the mathematics, not a real circuit: analytic circles, straights
and arcs only. Slow/full analytic validation lives in
``validation/v10_lap_sim.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.optimize import brentq

from backend.main import app
from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.lap.aero_map import AeroMap, ConstantAeroMap
from backend.simulation.lap.solver import LapSettings, solve_lap
from backend.simulation.lap.track import TrackError, build_track
from backend.simulation.lap.track_io import parse_centerline_csv
from backend.simulation.lap.vehicle import Vehicle

G = 9.80665
RHO = 1.225
SPEED_KMH = 200.0


class TestTheLapEndpointOverHttp:
    """`POST /api/lap/simulate` keeps the same honesty rules as the other tiers."""

    def _post(self, **extra):
        client = TestClient(app)
        extra.setdefault("config", FormulaAeroConfig(speed_kmh=SPEED_KMH).model_dump())
        body = {"track": TRACK, "method": "analytical_enhanced", **extra}
        return client.post("/api/lap/simulate", json=body)

    def test_a_circle_lap_solves_and_labels_its_provenance(self) -> None:
        response = self._post()
        assert response.status_code == 200
        payload = response.json()
        assert payload["lap_time_s"] > 0.0
        assert payload["aero_fidelity"] == "low_fidelity_vlm"
        assert payload["track_points"] == pytest.approx(
            payload["length_m"] / payload["ds_m"], rel=1e-6
        )
        n = len(payload["s_m"])
        assert n > 0
        for key in ("speed_kmh", "ay_ms2", "ax_ms2", "curvature_1pm", "downforce_N", "drag_N"):
            assert len(payload[key]) == n, f"{key} left the profile grid"
        assert payload["provenance"].get("source") or payload["track_source"]

    def test_the_aero_map_carries_a_real_config(self) -> None:
        # Wing elements have no pydantic defaults, so override a full dump.
        cfg = FormulaAeroConfig(speed_kmh=SPEED_KMH).model_dump()
        cfg["front_wing_main"]["angle_deg"] = 0.0
        zeroed = self._post(config=cfg).json()
        default = self._post().json()
        # On the 50 m-radius circle the lap is grip-limited, so unloading the
        # front wing must slow the car down, not speed it up.
        assert zeroed["lap_time_s"] > default["lap_time_s"], (
            f"zeroing the front wing did not slow the lap "
            f"({zeroed['lap_time_s']:.4f} vs {default['lap_time_s']:.4f})"
        )

    def test_cfd_is_rejected_with_its_reason(self) -> None:
        response = self._post(method="cfd")
        assert response.status_code == 400
        assert "aero map" in response.json()["detail"]

    def test_an_empty_track_is_a_422_not_a_solve(self) -> None:
        response = self._post(track={"closed": True})
        assert response.status_code == 422

    def test_a_bad_centreline_is_a_422(self) -> None:
        # A 0.5 m-radius closed loop is 3.14 m long: under the 10 m minimum,
        # so build_track must reject it instead of solving.
        short = [[0.5 * math.cos(t), 0.5 * math.sin(t)] for t in np.linspace(0.0, 2 * math.pi, 12)]
        response = self._post(track={"points": short})
        assert response.status_code == 422
        assert "at least 10" in response.json()["detail"]

    def test_csv_track_text_solves_the_same_track(self) -> None:
        csv = "\n".join(f"{p[0]:.6f},{p[1]:.6f}" for p in TRACK["points"])
        response = self._post(track={"csv": csv, "closed": True, "ds_m": 5.0})
        assert response.status_code == 200
        assert response.json()["lap_time_s"] == pytest.approx(
            self._post().json()["lap_time_s"],
            rel=1e-6,
        )


def circle_points(radius_m: float, n: int = 360) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return [[radius_m * math.cos(t), radius_m * math.sin(t)] for t in theta]


TRACK: dict = {"points": circle_points(50.0, n=48), "closed": True, "ds_m": 5.0}


TRACK: dict = {"points": circle_points(50.0, n=48), "closed": True, "ds_m": 5.0}


def test_circle_curvature_recovers_radius() -> None:
    radius = 25.0
    track = build_track(circle_points(radius), closed=True, ds_m=1.0)
    measured = 1.0 / float(np.mean(np.abs(track.curvature_1pm)))
    assert measured == pytest.approx(radius, rel=5e-3)
    assert track.length_m == pytest.approx(2.0 * math.pi * radius, rel=2e-3)


def test_skidpad_speed_and_lap_time_are_closed_form() -> None:
    mu, radius = 1.5, 30.0
    vehicle = Vehicle(
        mass_kg=800.0,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    result = solve_lap(
        build_track(circle_points(radius), closed=True, ds_m=1.0),
        vehicle,
        ConstantAeroMap(),
    )
    v_expected = math.sqrt(mu * G * radius)
    assert result.speed_ms.mean() == pytest.approx(v_expected, rel=5e-3)
    assert result.lap_time_s == pytest.approx(2.0 * math.pi * radius / v_expected, rel=5e-3)
    assert result.converged
    assert result.max_utilisation <= 1.01


def test_downforce_raises_cornering_speed_to_closed_form() -> None:
    mu, radius, cla, mass = 1.5, 30.0, 3.0, 800.0
    vehicle = Vehicle(
        mass_kg=mass,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    aero = ConstantAeroMap(downforce_area_m2=cla, front_share=0.5)
    result = solve_lap(
        build_track(circle_points(radius), closed=True, ds_m=1.0), vehicle, aero
    )
    v_no_aero = math.sqrt(mu * G * radius)
    v_expected = math.sqrt(
        mu * G * radius / (1.0 - mu * radius * (0.5 * RHO * cla) / mass)
    )
    assert result.speed_ms.mean() > v_no_aero
    assert result.speed_ms.mean() == pytest.approx(v_expected, rel=5e-3)


def test_top_speed_matches_power_balance() -> None:
    power_kw, efficiency, cda, crr, mass = 200.0, 0.9, 1.2, 0.01, 800.0
    vehicle = Vehicle(
        mass_kg=mass,
        power_kw=power_kw,
        drivetrain_efficiency=efficiency,
        tyre_mu=1.6,
        tyre_load_sensitivity=0.0,
        rolling_resistance=crr,
        max_speed_ms=100.0,
    )
    aero = ConstantAeroMap(drag_area_m2=cda)
    p_eff = power_kw * 1000.0 * efficiency
    expected = brentq(
        lambda v: p_eff - (0.5 * RHO * cda * v * v + crr * mass * G) * v, 1.0, 120.0
    )
    straight = [[float(x), 0.0] for x in np.arange(0.0, 3000.0, 5.0)]
    result = solve_lap(
        build_track(straight, closed=False, ds_m=5.0),
        vehicle,
        aero,
        LapSettings(initial_speed_ms=40.0),
    )
    assert result.speed_ms.max() == pytest.approx(expected, rel=2e-3)
    assert vehicle.top_speed(aero) == pytest.approx(expected, rel=1e-6)


def test_braking_distance_into_a_corner() -> None:
    mu, radius = 1.5, 40.0
    vehicle = Vehicle(
        mass_kg=800.0,
        front_static_pct=0.5,
        tyre_mu=mu,
        tyre_load_sensitivity=0.0,
        power_kw=0.0,
        rolling_resistance=0.0,
    )
    v_entry = 60.0
    v_corner = math.sqrt(mu * G * radius)
    expected_start = 300.0 - (v_entry**2 - v_corner**2) / (2.0 * mu * G)

    points = [[float(x), 0.0] for x in np.linspace(0.0, 300.0, 301)]
    angles = np.linspace(-math.pi / 2.0, 0.0, 64)
    points += [
        [300.0 + radius * math.cos(a), radius + radius * math.sin(a)] for a in angles
    ]
    track = build_track(points, closed=False, ds_m=1.0)
    result = solve_lap(
        track, vehicle, ConstantAeroMap(), LapSettings(initial_speed_ms=v_entry)
    )
    brake_start = float(result.s_m[np.argmax(result.speed_ms < v_entry - 0.01)])
    assert brake_start == pytest.approx(expected_start, rel=2e-2)
    assert result.speed_ms.min() == pytest.approx(v_corner, rel=5e-3)


def test_g_g_envelope_is_respected_on_a_tightening_spiral() -> None:
    theta = np.linspace(0.0, 2.0 * math.pi * 1.1, 500)
    radius = 70.0 + (18.0 - 70.0) * theta / theta[-1]
    points = [
        [float(r * math.cos(t)), float(r * math.sin(t))] for r, t in zip(radius, theta, strict=False)
    ]
    vehicle = Vehicle(power_kw=450.0, tyre_mu=1.6, tyre_load_sensitivity=0.1)
    aero = ConstantAeroMap(downforce_area_m2=3.0, drag_area_m2=1.1)
    result = solve_lap(build_track(points, closed=False, ds_m=1.0), vehicle, aero)
    assert result.converged
    assert result.max_utilisation <= 1.02
    for i in range(result.speed_ms.size):
        point_aero = aero.at(float(result.speed_ms[i]))
        limit = vehicle.lateral_limit(float(result.speed_ms[i]), point_aero)
        assert abs(result.ay_ms2[i]) <= limit * 1.01 + 1e-9


def test_aero_map_scales_with_dynamic_pressure() -> None:
    config = FormulaAeroConfig()
    aero = AeroMap(config, method="analytical_enhanced")
    v_ref = aero.reference_speed_ms
    base = aero.at(v_ref)
    faster = aero.at(2.0 * v_ref)
    assert faster.downforce_N == pytest.approx(4.0 * base.downforce_N, rel=1e-9)
    assert faster.drag_N == pytest.approx(4.0 * base.drag_N, rel=1e-9)
    assert faster.front_share == base.front_share


def test_tyre_load_sensitivity_and_cop_shift_reduce_grip() -> None:
    vehicle = Vehicle(tyre_mu=1.6, tyre_load_sensitivity=0.3)
    ref = float(vehicle.tyre_reference_load_N)
    assert vehicle.tyre_mu_at(2.0 * ref) < vehicle.tyre_mu_at(ref)

    speed = 60.0
    balanced = ConstantAeroMap(downforce_area_m2=6.0, front_share=0.52).at(speed)
    nose_heavy = ConstantAeroMap(downforce_area_m2=6.0, front_share=0.85).at(speed)
    assert vehicle.lateral_limit(speed, balanced) > vehicle.lateral_limit(
        speed, nose_heavy
    )


def test_invalid_centrelines_are_rejected() -> None:
    with pytest.raises(TrackError):
        build_track([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    with pytest.raises(TrackError):
        build_track([[0.0, 0.0]] * 20)
    with pytest.raises(TrackError):
        build_track([[0.0, 0.0], [float("nan"), 1.0]] * 10)
    with pytest.raises(TrackError):
        build_track([])


def test_csv_parser_accepts_common_layouts() -> None:
    rows = [
        f"{30 * math.cos(t):.6f},{30 * math.sin(t):.6f}"
        for t in np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    ]
    assert len(parse_centerline_csv("x,y\n" + "\n".join(rows))) == 20
    assert len(parse_centerline_csv("; ".join(["# metres"]) + "\n" + "\n".join(rows))) == 20
    with pytest.raises(TrackError):
        parse_centerline_csv("x,y\n1,2\n3,4")
    with pytest.raises(TrackError):
        parse_centerline_csv("")
