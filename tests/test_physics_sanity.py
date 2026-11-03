"""Physics-based sanity test suite for motorsport-aero-simulator.

Verifies essential aerodynamic principles on the analytical baseline model:
1. Quadratic force scaling with freestream velocity (F ~ V^2)
2. Bounded aerodynamic balance and Center of Pressure within physical wheelbase
3. Monotonicity of wing incidence angles vs downforce and drag
4. Ground effect ride-height and diffuser angle sensitivity
5. Dynamic pressure and non-dimensional coefficient consistency
6. Schema boundary validation on speed
7. The API pressure map and streamlines follow the same config state as the forces
8. The diffuser throat the model prices is one the car can actually draw
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation import analytical
from backend.simulation.analytical import (
    WHEELBASE_M,
    diffuser_throat_height_m,
    floor_height_at_m,
    simulate,
)
from backend.visualization import aero_field
from backend.visualization.aero_field import compute_cp_field, generate_streamlines


class TestPhysicsSanity:
    """Aerodynamic sanity verification."""

    def test_quadratic_velocity_scaling(self):
        """Aero forces must scale exactly with V^2 at fixed air density."""
        v1 = 100.0  # km/h
        v2 = 200.0  # km/h -> 2x speed -> 4x force

        cfg1 = FormulaAeroConfig(speed_kmh=v1)
        cfg2 = FormulaAeroConfig(speed_kmh=v2)

        res1 = simulate(cfg1)
        res2 = simulate(cfg2)

        ratio_drag = res2.forces.drag_N / res1.forces.drag_N
        ratio_df = res2.forces.downforce_total_N / res1.forces.downforce_total_N

        assert math.isclose(ratio_drag, 4.0, rel_tol=1e-2), (
            f"Expected 4.0 drag ratio, got {ratio_drag}"
        )
        assert math.isclose(ratio_df, 4.0, rel_tol=1e-2), (
            f"Expected 4.0 downforce ratio, got {ratio_df}"
        )

        # Dimensionless coefficients must remain constant with velocity
        assert math.isclose(res1.forces.cl_total, res2.forces.cl_total, rel_tol=1e-3)
        assert math.isclose(res1.forces.cd_total, res2.forces.cd_total, rel_tol=1e-3)

    def test_balance_and_cop_bounds(self):
        """Aero balance and Center of Pressure must remain physically bounded."""
        for speed in [100.0, 200.0, 300.0]:
            cfg = FormulaAeroConfig(speed_kmh=speed)
            res = simulate(cfg)

            assert 0.0 < res.forces.balance_front_pct < 100.0, (
                f"Balance out of range: {res.forces.balance_front_pct}%"
            )
            assert 0.0 < res.forces.cop_x_m < WHEELBASE_M, (
                f"CoP {res.forces.cop_x_m}m outside wheelbase {WHEELBASE_M}m"
            )

    def test_rear_wing_incidence_monotonicity(self):
        """Increasing rear wing incidence must increase rear downforce and total drag."""
        cfg_low = FormulaAeroConfig(
            rear_wing_main={"angle_deg": 2.0, "span_m": 1.0, "chord_m": 0.40},
            rear_wing_flap={"angle_deg": 10.0, "span_m": 1.0, "chord_m": 0.20},
        )
        cfg_high = FormulaAeroConfig(
            rear_wing_main={"angle_deg": 10.0, "span_m": 1.0, "chord_m": 0.40},
            rear_wing_flap={"angle_deg": 25.0, "span_m": 1.0, "chord_m": 0.20},
        )

        res_low = simulate(cfg_low)
        res_high = simulate(cfg_high)

        assert res_high.forces.downforce_rear_N > res_low.forces.downforce_rear_N, (
            "Higher rear wing AoA did not increase rear downforce"
        )
        assert res_high.forces.drag_N > res_low.forces.drag_N, (
            "Higher rear wing AoA did not increase drag"
        )

    def test_front_wing_incidence_monotonicity(self):
        """Increasing front wing flap angle must increase front downforce."""
        cfg_low = FormulaAeroConfig(
            front_wing_flap={"angle_deg": 5.0, "span_m": 1.8, "chord_m": 0.15}
        )
        cfg_high = FormulaAeroConfig(
            front_wing_flap={"angle_deg": 18.0, "span_m": 1.8, "chord_m": 0.15}
        )

        res_low = simulate(cfg_low)
        res_high = simulate(cfg_high)

        assert res_high.forces.downforce_front_N > res_low.forces.downforce_front_N
        assert res_high.forces.balance_front_pct > res_low.forces.balance_front_pct

    def test_diffuser_angle_sensitivity(self):
        """Increasing diffuser expansion angle increases underbody downforce."""
        cfg_shallow = FormulaAeroConfig(diffuser_angle_deg=8.0)
        cfg_steep = FormulaAeroConfig(diffuser_angle_deg=18.0)

        res_shallow = simulate(cfg_shallow)
        res_steep = simulate(cfg_steep)

        assert res_steep.forces.downforce_total_N > res_shallow.forces.downforce_total_N, (
            "Steeper diffuser angle did not increase downforce"
        )

    def test_underbody_ride_height_sensitivity(self):
        """Lowering rear ride height increases underbody suction; lowering front ride height shifts balance front."""
        cfg_high_rear = FormulaAeroConfig(ride_height_rear_mm=60.0)
        cfg_low_rear = FormulaAeroConfig(ride_height_rear_mm=30.0)

        res_high_rear = simulate(cfg_high_rear)
        res_low_rear = simulate(cfg_low_rear)

        assert res_low_rear.forces.downforce_total_N > res_high_rear.forces.downforce_total_N, (
            "Lower rear ride height did not increase total downforce"
        )

        cfg_high_front = FormulaAeroConfig(ride_height_front_mm=45.0)
        cfg_low_front = FormulaAeroConfig(ride_height_front_mm=15.0)

        res_high_front = simulate(cfg_high_front)
        res_low_front = simulate(cfg_low_front)

        assert res_low_front.forces.downforce_front_N > res_high_front.forces.downforce_front_N, (
            "Lower front ride height did not shift downforce forward"
        )
        assert res_low_front.forces.balance_front_pct > res_high_front.forces.balance_front_pct

    def test_speed_bounds_validation(self):
        """FormulaAeroConfig strictly enforces physical speed limits (50 to 350 km/h)."""
        cfg_min = FormulaAeroConfig(speed_kmh=50.0)
        res_min = simulate(cfg_min)
        assert res_min.forces.drag_N > 0.0
        assert res_min.forces.downforce_total_N > 0.0

        with pytest.raises(ValidationError):
            FormulaAeroConfig(speed_kmh=49.0)

        with pytest.raises(ValidationError):
            FormulaAeroConfig(speed_kmh=351.0)


BOUNDS = {"min": [0.0, -0.9, 0.0], "max": [4.2, 0.9, 0.9]}
# Single points inside the map's own front-wing and diffuser masks
PROBE_FRONT_WING = [[0.0, 0.0, 0.0]]
PROBE_DIFFUSER = [[4.2, 0.0, 0.0]]


def _rear_flap(angle_deg: float) -> dict:
    return {"angle_deg": angle_deg, "span_m": 0.85, "chord_m": 0.05}


def _pinned_cp(monkeypatch, overrides: dict, probe: list[list[float]]) -> float:
    """One probe point with the map's force scale pinned to exactly 1.0.

    cl_total = 3.0 and cd_total = 1.2 are the divisors `compute_cp_field` uses,
    so every term but the one under test becomes a constant across the configs.
    """

    class _Forces:
        cl_total = 3.0
        cd_total = 1.2

    class _PinnedSim:
        forces = _Forces()

        @staticmethod
        def model_dump() -> dict:
            return {}

    monkeypatch.setattr(aero_field, "simulate_panel", lambda _cfg: _PinnedSim())
    config = FormulaAeroConfig(**overrides)
    return float(np.mean(aero_field.compute_cp_field(probe, None, config, BOUNDS)["cp"]))


def _spread(values: tuple[float, ...], factors: dict[float, float]) -> float:
    """Ratio of two differences taken against a common far station.

    Subtracting the same station from both removes every additive term, and
    dividing the two differences removes the region weight and the angle
    prefactor, which are common to both. Only the decay length survives.
    """
    far = values[-1]
    return (factors[values[0]] - factors[far]) / (factors[values[1]] - factors[far])


class TestFieldConsumersShareSolverState:
    """The pressure map and streamlines must follow the same state as the forces.

    `compute_cp_field` and `generate_streamlines` are the API's only picture of
    the car, and they read the sealed rear flap, the rear ride height in place of
    the diffuser throat, and their own rake coefficient, so the screen could show
    loads the solver was not charging.
    """

    REGIONS = {
        # station window, height window in metres
        "rear_wing": ((3.4, 4.2), (0.58, 0.88)),
        "front_wing": ((0.05, 0.60), (0.05, 0.28)),
    }

    def _cloud(self, region: str) -> list[list[float]]:
        stations = np.linspace(*self.REGIONS[region][0], 6)
        heights = np.linspace(*self.REGIONS[region][1], 4)
        return [
            [float(x), y, float(z)]
            for x in stations
            for y in (-0.5, 0.0, 0.5)
            for z in heights
        ]

    def _mean_cp(self, config: FormulaAeroConfig, region: str) -> float:
        return float(
            np.mean(compute_cp_field(self._cloud(region), None, config, BOUNDS)["cp"])
        )

    def test_rear_wing_pressure_responds_to_drs(self):
        """Opening DRS must relieve the rear wing in the map, not only in the forces."""
        sealed = FormulaAeroConfig(rear_wing_flap=_rear_flap(20.0))
        opened = sealed.model_copy(update={"drs_active": True})

        shift = self._mean_cp(opened, "rear_wing") - self._mean_cp(sealed, "rear_wing")

        # Measured shift is ~0.061 Cp; ignoring the flap angle leaves ~0.002,
        # which comes only from the force scale, so 0.04 needs the real flap.
        assert shift > 0.04, f"DRS barely moved the rear wing pressure ({shift:+.4f})"

    def test_wake_strength_responds_to_drs(self):
        """The drawn wake must weaken with the effective flap, not the sealed one."""
        sealed = FormulaAeroConfig(rear_wing_flap=_rear_flap(20.0))
        opened = sealed.model_copy(update={"drs_active": True})

        def mean_height(cfg: FormulaAeroConfig) -> float:
            lines = generate_streamlines(cfg, BOUNDS, n_lines=6, n_points=20)
            return float(np.mean([p[2] for line in lines for p in line]))

        assert mean_height(opened) > mean_height(sealed) + 0.001

    def test_front_wing_suction_follows_ride_height(self):
        """Lower front ride height must read as more suction on the map's front wing."""
        low = self._mean_cp(FormulaAeroConfig(ride_height_front_mm=10.0), "front_wing")
        high = self._mean_cp(FormulaAeroConfig(ride_height_front_mm=60.0), "front_wing")

        assert low < high - 0.02, f"front wing ignored ride height ({low:+.4f} vs {high:+.4f})"

    def test_front_wing_suction_uses_the_solver_decay_length(self, monkeypatch):
        """With the force scale pinned, the front wing must decay as the solver says.

        A 0.030 m decay length instead of the solver's 0.015 m would give 1.65
        here rather than the measured 2.72.
        """
        stations = (10.0, 25.0, 200.0)
        cp = {
            v: _pinned_cp(monkeypatch, {"ride_height_front_mm": v}, PROBE_FRONT_WING)
            for v in stations
        }
        ge = {v: math.exp(-v / 1000.0 / analytical.GE_DECAY_FRONT_M) for v in stations}

        assert _spread(stations, cp) == pytest.approx(_spread(stations, ge), rel=1e-3)

    def test_diffuser_suction_follows_throat_height(self, monkeypatch):
        """The map used to price the diffuser with the rear ride height only.

        `diffuser_ride_height_mm` moved the solver's diffuser Cl and nothing on
        the screen; the throat height now drives the same decay the solver uses.
        Both ride heights are pinned at 10 mm so the floor line stays under the
        whole sweep - a throat below it is raised to it, as it is on the car.
        """
        stations = (10.0, 30.0, 60.0)
        cp = {
            v: _pinned_cp(
                monkeypatch,
                {
                    "ride_height_front_mm": 10.0,
                    "ride_height_rear_mm": 10.0,
                    "diffuser_ride_height_mm": v,
                },
                PROBE_DIFFUSER,
            )
            for v in stations
        }
        ge = {v: math.exp(-v / 1000.0 / analytical.GE_DECAY_DIFFUSER_M) for v in stations}

        assert cp[stations[2]] - cp[stations[0]] > 0.02, (
            "diffuser throat height does not reach the pressure map"
        )
        assert _spread(stations, cp) == pytest.approx(_spread(stations, ge), rel=1e-3)


class TestDiffuserThroatMustBeBuildable:
    """The solver prices the throat the car can draw, not the number on the slider.

    A diffuser starts where the flat floor ends, so a throat below the floor line
    at that station is not a shape the model has. `carSpec.js` already raised the
    drawn throat to the line; the forces now apply the same rule.
    """

    THROAT_STATION_M = analytical.FLOOR_X_END_M - analytical.DIFFUSER_LENGTH_M

    def test_floor_line_passes_through_both_ride_heights(self):
        cfg = FormulaAeroConfig(ride_height_front_mm=25.0, ride_height_rear_mm=45.0)
        assert floor_height_at_m(cfg, analytical.FRONT_AXLE_M) == pytest.approx(
            0.025, abs=1e-9
        )
        assert floor_height_at_m(cfg, analytical.REAR_AXLE_M) == pytest.approx(
            0.045, abs=1e-9
        )

    def test_throat_below_the_floor_line_is_raised_to_it(self):
        cfg = FormulaAeroConfig(diffuser_ride_height_mm=10.0)
        line = floor_height_at_m(cfg, self.THROAT_STATION_M)
        assert diffuser_throat_height_m(cfg) == pytest.approx(line, abs=1e-12)
        assert line * 1000.0 > cfg.diffuser_ride_height_mm

    def test_throats_under_the_floor_line_price_identically(self):
        """Below the line the slider cannot move the car, so it must not move the forces."""
        base = FormulaAeroConfig()
        ten = simulate(base.model_copy(update={"diffuser_ride_height_mm": 10.0})).forces
        thirty = simulate(base.model_copy(update={"diffuser_ride_height_mm": 30.0})).forces
        assert ten.downforce_total_N == thirty.downforce_total_N

        # A throat above the line is still a live parameter.
        sixty = simulate(base.model_copy(update={"diffuser_ride_height_mm": 60.0})).forces
        assert sixty.downforce_total_N < thirty.downforce_total_N

    def test_pressure_map_prices_the_same_throat(self, monkeypatch):
        """The map must not reward a shape the forces refuse to charge."""
        low = _pinned_cp(monkeypatch, {"diffuser_ride_height_mm": 10.0}, PROBE_DIFFUSER)
        mid = _pinned_cp(monkeypatch, {"diffuser_ride_height_mm": 30.0}, PROBE_DIFFUSER)
        high = _pinned_cp(monkeypatch, {"diffuser_ride_height_mm": 60.0}, PROBE_DIFFUSER)

        assert low == pytest.approx(mid, abs=1e-9)
        assert high - mid > 0.02

    def test_solver_and_viewer_agree_on_the_stations(self):
        """carSpec.js draws from these same numbers; a one-sided edit must fail here."""
        js_path = (
            Path(__file__).resolve().parents[1] / "frontend" / "js" / "carSpec.js"
        )
        js = js_path.read_text(encoding="utf-8")
        shared = {
            "wheelbaseM": analytical.WHEELBASE_M,
            "refAreaM2": analytical.REF_AREA,
            "frontAxleM": analytical.FRONT_AXLE_M,
            "rearAxleM": analytical.REAR_AXLE_M,
            "floorXStartM": analytical.FLOOR_X_START_M,
            "floorXEndM": analytical.FLOOR_X_END_M,
            "diffuserLengthM": analytical.DIFFUSER_LENGTH_M,
        }
        for name, value in shared.items():
            match = re.search(rf"^\s*{name}:\s*([0-9.]+),", js, re.MULTILINE)
            assert match, f"carSpec.js no longer declares {name}"
            assert float(match.group(1)) == pytest.approx(value), (
                f"{name}: viewer draws {match.group(1)}, solver prices {value}"
            )
