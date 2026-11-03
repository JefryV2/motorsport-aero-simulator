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
9. The pressure map's surface regions sit on the solver's stations, not on
   fractions of the bounding box that arrived with the mesh
10. One lap-time proxy prices all three tiers and the frontend mirror
11. The wing ground-effect gain is priced at each element's own gap to the plane,
    so the front ride height moves the wing and not only the floor
12. The Reynolds readout is taken on the length the viewer actually draws, and
    names that length wherever it is shown
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation import analytical, panel_method
from backend.simulation.analytical import (
    WHEELBASE_M,
    diffuser_throat_height_m,
    floor_height_at_m,
    simulate,
)
from backend.simulation.cfd.openfoam_runner import run_cfd_with_provenance
from backend.simulation.cfd.runner import evaluate_calibrated_synthetic_cfd
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


# The map reads a mesh box as a formula car of these extents (aero_field
# `SurfaceRegions`), so the probes and clouds below are stated in the same
# metres the solver uses instead of fractions of an arbitrary box.
BOUNDS = {
    "min": [0.0, -analytical.CAR_HALF_WIDTH_M, 0.0],
    "max": [analytical.CAR_LENGTH_M, analytical.CAR_HALF_WIDTH_M, analytical.CAR_HEIGHT_M],
}
WING_STATIONS = {s.name: s for s in analytical.wing_stations(FormulaAeroConfig())}
DIFFUSER_X0_M = analytical.FLOOR_X_END_M - analytical.DIFFUSER_LENGTH_M
# One point on each ground-effect surface. The front-wing probe sits a quarter of
# the way along the wing band: the floor region fades in 0.24 m ahead of
# FLOOR_X_START_M and its height moves with the front ride height this file
# sweeps, so a probe further aft would not isolate a single term.
PROBE_FRONT_WING_X_M = WING_STATIONS["front_wing_main"].x_le_m + 0.25 * (
    WING_STATIONS["front_wing_flap"].x_te_m - WING_STATIONS["front_wing_main"].x_le_m
)
PROBE_FRONT_WING = [[PROBE_FRONT_WING_X_M, 0.0, 0.03]]
PROBE_DIFFUSER = [[0.5 * (DIFFUSER_X0_M + analytical.FLOOR_X_END_M), 0.0, 0.05]]


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
        # station window, height window in metres: the solved wing planform and
        # the drawn endplate box the map builds its regions from
        "rear_wing": (
            (WING_STATIONS["rear_wing_main"].x_le_m, WING_STATIONS["rear_wing_flap"].x_te_m),
            (analytical.REAR_WING_BOTTOM_M, analytical.REAR_WING_TOP_M),
        ),
        "front_wing": (
            (WING_STATIONS["front_wing_main"].x_le_m, WING_STATIONS["front_wing_flap"].x_te_m),
            (0.0, WING_STATIONS["front_wing_flap"].height_m),
        ),
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

        # Measured over the solved wing box: +0.143 Cp. Ignoring the flap angle
        # leaves +0.0052, which comes only from the force scale, so 0.10 needs
        # the real effective incidence.
        assert shift > 0.10, f"DRS barely moved the rear wing pressure ({shift:+.4f})"

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

        # Measured gap over the solved wing box is 0.108 Cp, but 0.072 of it still
        # survives with the map's decay length frozen, because the force scale
        # moves with the ride height too. This only proves the surface responds;
        # the decay *law* is pinned by the ratio test below.
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
            "rho": analytical.RHO,
            "carLengthM": analytical.CAR_LENGTH_M,
            "frontAxleM": analytical.FRONT_AXLE_M,
            "rearAxleM": analytical.REAR_AXLE_M,
            "floorXStartM": analytical.FLOOR_X_START_M,
            "floorXEndM": analytical.FLOOR_X_END_M,
            "diffuserLengthM": analytical.DIFFUSER_LENGTH_M,
            "diffuserExitMaxM": analytical.DIFFUSER_EXIT_MAX_M,
            "fwLeadingEdgeM": analytical.WING_LE_X_M,
            "fwFlapStationFrac": analytical.WING_FLAP_STATION_FRAC,
            "rwFlapStationFrac": analytical.WING_FLAP_STATION_FRAC,
            "fwFlapGapM": analytical.FW_FLAP_STEP_UP_M,
            "rwFlapRiseM": analytical.RW_FLAP_STEP_UP_M,
            "rwMainHeightM": analytical.REAR_WING_MAIN_HEIGHT_M,
            "rwEndplateHeightM": analytical.RW_ENDPLATE_HEIGHT_M,
            "rwEndplateDropM": analytical.RW_ENDPLATE_DROP_M,
            "trackM": analytical.TRACK_M,
            "floorHalfWidthM": analytical.FLOOR_HALF_WIDTH_M,
            "wheelRadiusFrontM": analytical.WHEEL_RADIUS_FRONT_M,
            "wheelRadiusRearM": analytical.WHEEL_RADIUS_REAR_M,
            "wheelWidthFrontM": analytical.WHEEL_WIDTH_FRONT_M,
            "wheelWidthRearM": analytical.WHEEL_WIDTH_REAR_M,
            "geLiftGainMax": analytical.GE_LIFT_GAIN_MAX,
            "geLiftDecayM": analytical.GE_LIFT_DECAY_M,
            "lapBaselineDragN": analytical.LAP_BASELINE_DRAG_N,
            "lapBaselineDfN": analytical.LAP_BASELINE_DF_N,
            "lapSecPerDragN": analytical.LAP_SEC_PER_DRAG_N,
            "lapSecPerDfN": analytical.LAP_SEC_PER_DF_N,
        }
        for name, value in shared.items():
            match = re.search(rf"^\s*{name}:\s*([0-9.]+),", js, re.MULTILINE)
            assert match, f"carSpec.js no longer declares {name}"
            assert float(match.group(1)) == pytest.approx(value), (
                f"{name}: viewer draws {match.group(1)}, solver prices {value}"
            )


class TestPressureMapRegionsAreStations:
    """The map must shade the surfaces the model has, not fractions of a box.

    `aero_field` used to write its region edges as literals of the normalised
    coordinate (`1 - x / 0.18`, `(x - 0.62) / 0.38`), so it called the first 18 %
    of any mesh a front wing and the last 38 % a diffuser whatever the setup said:
    on the reference car that shaded 2.6-4.2 m as a diffuser whose throat starts at
    3.02 m, and gave the whole nose cone front-wing suction.
    """

    def _regions(self, config: FormulaAeroConfig) -> aero_field.SurfaceRegions:
        return aero_field.SurfaceRegions.from_config(config)

    def test_front_wing_region_is_the_solved_planform(self):
        main = WING_STATIONS["front_wing_main"]
        flap = WING_STATIONS["front_wing_flap"]
        mid_x = 0.5 * (main.x_le_m + flap.x_te_m)

        on_wing = self._regions(FormulaAeroConfig()).weights(mid_x, 0.0, main.height_m)
        assert on_wing["front_wing"] == pytest.approx(1.0)

        # Ahead of the leading edge is the nose, aft of the trailing edge is the
        # floor; neither may carry the front-wing pressure.
        ahead = self._regions(FormulaAeroConfig()).weights(main.x_le_m - 0.12, 0.0, main.height_m)
        behind = self._regions(FormulaAeroConfig()).weights(flap.x_te_m + 0.12, 0.0, main.height_m)
        assert ahead["front_wing"] == 0.0
        assert behind["front_wing"] == 0.0

    def test_rear_wing_region_covers_the_flap_at_the_endplate_crown(self):
        main = WING_STATIONS["rear_wing_main"]
        flap = WING_STATIONS["rear_wing_flap"]
        cfg = FormulaAeroConfig()

        # The flap plane sits 1 cm under the crown, which is the top of the box:
        # a mask that faded across its own window used to lose the loaded element.
        crown = self._regions(cfg).weights(0.5 * (main.x_le_m + flap.x_te_m), 0.0, flap.height_m)
        assert crown["rear_wing"] == pytest.approx(1.0)

        # Ahead of the main plane is beam and bodywork, not a lifting surface.
        forward = self._regions(cfg).weights(main.x_le_m - 0.20, 0.0, main.height_m)
        assert forward["rear_wing"] == 0.0

    def test_floor_and_diffuser_regions_meet_at_the_throat_station(self):
        cfg = FormulaAeroConfig()
        regions = self._regions(cfg)
        floor_mid = 0.5 * (analytical.FLOOR_X_START_M + DIFFUSER_X0_M)

        on_floor = regions.weights(floor_mid, 0.0, floor_height_at_m(cfg, floor_mid))
        assert on_floor["underbody"] == pytest.approx(1.0)
        assert on_floor["diffuser"] == 0.0

        on_ramp = regions.weights(0.5 * (DIFFUSER_X0_M + analytical.FLOOR_X_END_M), 0.0, 0.05)
        assert on_ramp["diffuser"] == pytest.approx(1.0)
        assert on_ramp["underbody"] == 0.0

        # Past the floor's real end the tail is nothing the model charges.
        behind = regions.weights(analytical.FLOOR_X_END_M + 0.25, 0.0, 0.05)
        assert behind["diffuser"] == 0.0
        assert behind["underbody"] == 0.0

    def test_wing_region_edge_moves_with_the_chord_not_the_mesh(self):
        """A longer main plane means a longer region; a bigger box does not."""
        cfg = FormulaAeroConfig()
        short = self._regions(cfg)
        longer = cfg.model_copy(
            update={
                "rear_wing_main": cfg.rear_wing_main.model_copy(update={"chord_m": 0.45})
            }
        )
        long_regions = self._regions(longer)

        # The flap hangs 0.75 of the main chord aft, so only that share of the
        # extra chord reaches the assembly's trailing edge.
        shift = 0.05 * analytical.WING_FLAP_STATION_FRAC
        assert long_regions.rear_wing_x[1] - short.rear_wing_x[1] == pytest.approx(shift)
        edge = short.rear_wing_x[1] + 0.06
        assert short.weights(edge, 0.0, 0.85)["rear_wing"] == 0.0
        assert long_regions.weights(edge, 0.0, 0.85)["rear_wing"] > 0.0

    def test_the_map_does_not_reposition_itself_when_the_mesh_is_rescaled(self):
        """The same car in metres or twice-metres must shade the same surfaces."""
        point = PROBE_DIFFUSER[0]
        scaled_bounds = {
            "min": [2.0 * v for v in BOUNDS["min"]],
            "max": [2.0 * v for v in BOUNDS["max"]],
        }
        cfg = FormulaAeroConfig()

        base = compute_cp_field([point], None, cfg, BOUNDS)["cp"]
        scaled = compute_cp_field([[2.0 * v for v in point]], None, cfg, scaled_bounds)["cp"]

        assert base == scaled

    def test_the_solved_surfaces_and_the_map_share_one_wing_geometry(self, monkeypatch):
        """simulate_panel must not keep its own copy of where the wings are."""
        captured: list = []

        class _Section:
            downforce_N = 0.0

        class _Solution:
            sections = {}
            total_drag_N = 0.0

        def _capture(surfaces, **_kwargs):
            captured.extend(surfaces)
            solution = _Solution()
            solution.sections = {s.name: _Section() for s in surfaces}
            return solution

        monkeypatch.setattr(panel_method, "solve_vlm", _capture)
        cfg = FormulaAeroConfig()
        panel_method.simulate_panel(cfg)

        origins = {s.name: s for s in captured}
        assert len(origins) == 4, "simulate_panel no longer builds four named surfaces"
        for station in analytical.wing_stations(cfg):
            solved = origins[station.name]
            assert solved.origin == pytest.approx(
                (station.x_le_m, 0.0, station.height_m), abs=1e-12
            )
            assert (solved.chord, solved.span, solved.angle_deg) == (
                station.chord_m,
                station.span_m,
                station.angle_deg,
            )


class TestOneLapProxyAcrossTiers:
    """All three tiers must price the lap proxy with the same one function.

    `lap_time_delta_s` was restated in each tier and the CFD copy was not a delta
    at all - it reported (drag - 0.3 x downforce) / 10000, so a setup with no
    reference whatsoever still came back as +0.0300 s, and its sign flipped with
    the downforce term the other two tiers subtract.
    """

    def test_every_tier_reports_the_shared_proxy(self):
        cfg = FormulaAeroConfig()
        tiers = {
            "analytical": simulate(cfg),
            "panel_vlm": panel_method.simulate_panel(cfg),
            "cfd": run_cfd_with_provenance(cfg)[0],
        }
        for method, sim in tiers.items():
            want = analytical.lap_time_delta_s(
                sim.forces.drag_N, sim.forces.downforce_total_N
            )
            # The tiers round their forces before reporting, so the proxy the
            # result carries is within the rounding of the reported forces.
            assert sim.lap_time_delta_s == pytest.approx(want, abs=1e-4), (
                f"{method} prices its own lap proxy: {sim.lap_time_delta_s} vs {want}"
            )

    def test_the_proxy_is_zero_against_its_own_baseline(self):
        """A delta has to vanish when nothing changed; an absolute term cannot."""
        assert (
            analytical.lap_time_delta_s(
                analytical.LAP_BASELINE_DRAG_N, analytical.LAP_BASELINE_DF_N
            )
            == 0.0
        )

    def test_the_two_terms_move_in_opposite_directions(self):
        """More drag is slower, more downforce is faster, at the pinned rates."""
        base = analytical.lap_time_delta_s(1500.0, 3500.0)
        draggier = analytical.lap_time_delta_s(1600.0, 3500.0)
        loaded = analytical.lap_time_delta_s(1500.0, 4500.0)

        assert draggier - base == pytest.approx(100.0 * analytical.LAP_SEC_PER_DRAG_N)
        assert base - loaded == pytest.approx(1000.0 * analytical.LAP_SEC_PER_DF_N)


class TestWingGroundEffectIsPricedAtItsOwnGap:
    """The front wing used to get a flat x1.12 lift gain whatever its height was.

    `ride_height_front_mm` therefore moved the floor and the front/rear share but
    left the wing alone: at a 150 mm nose the model still claimed the same 12 %
    augmentation it used at 25 mm. The gain is now priced at each element's
    trailing-edge clearance, anchored so the default front wing as a whole keeps
    the 1.12 it used to assert.
    """

    def _stations(self, config: FormulaAeroConfig):
        return {s.name: s for s in analytical.wing_stations(config)}

    def _wing_only_front_df(self, monkeypatch, front_mm: float) -> float:
        """Front downforce with the underbody term switched off, so only the wing moves."""
        monkeypatch.setattr(
            analytical, "_underbody_forces", lambda config, speed_ms: (0.0, 0.0, 0.35)
        )
        return simulate(FormulaAeroConfig(ride_height_front_mm=front_mm)).forces.downforce_front_N

    def test_the_gain_decays_with_the_gap_and_vanishes_far_above_the_plane(self):
        near = analytical.ground_effect_lift_factor(0.030)
        mid = analytical.ground_effect_lift_factor(0.077)
        far = analytical.ground_effect_lift_factor(0.150)

        assert near > mid > far
        assert analytical.ground_effect_lift_factor(2.0) == pytest.approx(1.0, abs=1e-5)
        assert analytical.ground_effect_lift_factor(None) == 1.0

    def test_the_gap_is_the_trailing_edge_not_the_station_height(self):
        """A pitched wing's lowest point is its trailing edge, and that is the gap."""
        cfg = FormulaAeroConfig(ride_height_front_mm=25.0)
        main = self._stations(cfg)["front_wing_main"]
        assert main.min_clearance_m == pytest.approx(
            0.025 + analytical.FW_GROUND_CLEARANCE_M
        )

        pitched_cfg = cfg.model_copy(
            update={
                "front_wing_main": cfg.front_wing_main.model_copy(
                    update={"angle_deg": 12.0}
                )
            }
        )
        pitched = self._stations(pitched_cfg)["front_wing_main"]
        assert pitched.height_m > main.height_m + 0.02, "the nose did not lift with incidence"
        assert pitched.min_clearance_m == pytest.approx(main.min_clearance_m)

    def test_the_default_front_wing_keeps_the_flat_augmentation(self):
        """Anchor for the two constants: the wing as a whole still reads ~1.12."""
        cfg = FormulaAeroConfig()
        stations = self._stations(cfg)
        areas, weighted = 0.0, 0.0
        for name in ("front_wing_main", "front_wing_flap"):
            element = getattr(cfg, name)
            area = element.chord_m * element.span_m
            gain = analytical.ground_effect_lift_factor(stations[name].min_clearance_m)
            areas += area
            weighted += gain * area

        assert weighted / areas == pytest.approx(1.12, abs=0.005)

    def test_the_stacked_flap_gets_less_gain_than_the_main_plane(self):
        cfg = FormulaAeroConfig()
        stations = self._stations(cfg)
        main = analytical.ground_effect_lift_factor(
            stations["front_wing_main"].min_clearance_m
        )
        flap = analytical.ground_effect_lift_factor(
            stations["front_wing_flap"].min_clearance_m
        )

        assert main > 1.125 > flap > 1.05, f"main {main:.4f}, flap {flap:.4f}"

    def test_front_wing_lift_falls_as_the_nose_rises(self, monkeypatch):
        """The behaviour the old flat factor could not show at all."""
        low = self._wing_only_front_df(monkeypatch, 10.0)
        mid = self._wing_only_front_df(monkeypatch, 60.0)
        high = self._wing_only_front_df(monkeypatch, 150.0)

        assert low > mid > high, f"wing ignored ride height: {low} / {mid} / {high}"
        assert low - high > 30.0, f"only {low - high:.1f} N lost over 140 mm of nose lift"


class TestRequestPayloadTakesThePlanformFromTheModel:
    """The UI has no chord or span control, so it must not own those numbers.

    `getConfig()` used to write all eight planform literals itself - a third copy
    beside the `FormulaAeroConfig` factories and `carSpec.js`. Editing a model
    default would keep drawing and solving one car while the request sent another,
    because the solver re-derives its defaults from the factories, the viewer from
    `DEFAULT_ELEMENT`, and the payload from whichever copy nobody thought to edit.
    """

    WING_NAMES = (
        "front_wing_main",
        "front_wing_flap",
        "rear_wing_main",
        "rear_wing_flap",
    )

    def _js(self, name: str) -> str:
        path = Path(__file__).resolve().parents[1] / "frontend" / "js" / name
        return path.read_text(encoding="utf-8")

    def _config_body(self) -> str:
        js = self._js("app.js")
        start = js.index("function getConfig()")
        return js[start : js.index("function getFlowSettings()", start)]

    def test_car_spec_defaults_match_the_config_factories(self):
        block = re.search(
            r"export const DEFAULT_ELEMENT = \{(.*?)\n\};",
            self._js("carSpec.js"),
            re.S,
        )
        assert block, "carSpec.js no longer exports DEFAULT_ELEMENT"
        cfg = FormulaAeroConfig()
        for name in self.WING_NAMES:
            entry = re.search(name + r": \{([^}]*)\}", block.group(1))
            assert entry, f"DEFAULT_ELEMENT dropped {name}"
            fields = {
                key: float(value)
                for key, value in re.findall(r"(\w+): ([0-9.]+)", entry.group(1))
            }
            element = getattr(cfg, name)
            assert fields["chord_m"] == pytest.approx(element.chord_m), (
                f"{name}: viewer has chord {fields['chord_m']}, model has "
                f"{element.chord_m}"
            )
            assert fields["span_m"] == pytest.approx(element.span_m), (
                f"{name}: viewer has span {fields['span_m']}, model has "
                f"{element.span_m}"
            )

    def test_get_config_restates_no_planform_number(self):
        cfg = FormulaAeroConfig()
        banned = set()
        for name in self.WING_NAMES:
            element = getattr(cfg, name)
            banned.update({element.chord_m, element.span_m})

        body = self._config_body()
        assert "DEFAULT_ELEMENT" in body, (
            "getConfig() no longer reads chord and span from the model defaults"
        )
        leaked = sorted(
            float(token)
            for token in re.findall(r"(?<![\w.'])(\d+\.\d+)\b", body)
            if float(token) in banned
        )
        assert not leaked, (
            f"getConfig() hardcodes planform {leaked} again - the payload must "
            "take chord and span from DEFAULT_ELEMENT"
        )


class TestReynoldsReadoutSitsOnTheDrawnCar:
    """The viewer used to price Re on a length it neither drew nor named.

    `reynoldsNumber()` defaulted to a bare 3.5 m while `carGeometry.js` draws the
    4.25 m envelope `analytical.CAR_LENGTH_M` normalises uploaded meshes with, and
    it restated rho and mu next to that. The HUD printed a tunnel condition for a
    car 0.75 m shorter than the one on screen: 13.16 M where 200 km/h over the
    drawn length is 15.98 M.
    """

    def _js(self, name: str) -> str:
        path = Path(__file__).resolve().parents[1] / "frontend" / "js" / name
        return path.read_text(encoding="utf-8")

    def _ref(self, key: str) -> float:
        match = re.search(
            rf"^\s*{key}:\s*([0-9.eE+-]+),", self._js("carSpec.js"), re.MULTILINE
        )
        assert match, f"carSpec.js no longer declares {key}"
        return float(match.group(1))

    def _fn(self, name: str) -> str:
        body = re.search(
            rf"export function {name}\(.*?\n\}}", self._js("flowEngine.js"), re.DOTALL
        )
        assert body, f"flowEngine.js no longer defines {name}"
        return body.group(0)

    def test_the_declared_length_is_the_envelope_the_viewer_draws(self):
        drawn = self._ref("rearAxleM") + self._ref("wheelRadiusRearM")
        assert self._ref("carLengthM") == pytest.approx(drawn)
        assert self._ref("carLengthM") == pytest.approx(analytical.CAR_LENGTH_M)

    def test_reynolds_is_computed_from_the_declared_constants(self):
        src = self._fn("reynoldsNumber")
        for shared in ("REFERENCE.carLengthM", "REFERENCE.rho", "REFERENCE.muAirPaS"):
            assert shared in src, f"reynoldsNumber no longer reads {shared}"
        magic = [
            token
            for token in ("3.5", "1.225", "1.81e-5")
            if re.search(rf"(?<![\w.]){re.escape(token)}\b", src)
        ]
        assert not magic, f"reynoldsNumber hardcodes {magic} again"

    def test_the_readout_names_the_length_it_was_taken_on(self):
        label = self._fn("reynoldsLabel")
        assert "REFERENCE.carLengthM" in label, "the label dropped the length"
        for consumer in ("app.js", "viewer3d.js"):
            assert re.search(r"textContent[^\n]*reynoldsLabel\(", self._js(consumer)), (
                f"{consumer} prints a bare Reynolds number again"
            )

    def test_the_two_reynolds_conventions_describe_the_same_air(self):
        """The HUD's car-length Re and the CFD tier's chord Re are the same fluid.

        `evaluate_calibrated_synthetic_cfd` scales its skin-friction term with
        `speed * chord / nu` at nu = 1.5e-5 m²/s; the viewer reaches the same
        kinematic value through mu = 1.81e-5 Pa·s over rho = 1.225 kg/m³.
        """
        signature = inspect.signature(evaluate_calibrated_synthetic_cfd)
        nu_cfd = signature.parameters["nu"].default
        kinematic = self._ref("muAirPaS") / self._ref("rho")
        assert kinematic == pytest.approx(nu_cfd, rel=0.03)


class TestInflowAnglesAreViewerOnly:
    """Yaw and pitch tilt the smoke; nothing in the model knows they exist.

    Measured on the shipped field: `|wind|` is 55.5556 m/s at yaw 0 / pitch 0 and
    55.5556 m/s at yaw 35 / pitch 15, so every singularity strength built from it
    is identical whatever the sliders say - the tilt moves the tracers and the
    near-body cap, nothing else. At +15 deg the inflow carries +14.38 m/s of
    vertical velocity: air arriving from below, the flow equivalent of a nose-down
    car, while the drawn body stays level. The control used to read "Pitch angle",
    directly above a "Body rake" readout that *is* the car's angle.
    """

    def _text(self, *parts: str) -> str:
        return (Path(__file__).resolve().parents[1].joinpath(*parts)).read_text(
            encoding="utf-8"
        )

    def _payload_keys(self) -> set[str]:
        js = self._text("frontend", "js", "app.js")
        start = js.index("function getConfig()")
        body = js[start : js.index("function getFlowSettings()", start)]
        block = body[body.index("return {") :]
        return set(re.findall(r"^\s{4}(\w+):", block, re.MULTILINE))

    def test_every_payload_key_is_a_field_the_solver_reads(self):
        """An unknown key would be dropped without a word.

        `FormulaAeroConfig` does not set `extra`, so pydantic ignores unrecognised
        fields - the way `drs_active` was sent and silently discarded from the
        golden test for months.
        """
        fields = set(FormulaAeroConfig.model_fields)
        keys = self._payload_keys()
        assert keys, "getConfig() no longer returns an object literal"
        assert keys <= fields, (
            f"payload carries {sorted(keys - fields)}, which no solver field reads - "
            "pydantic would drop it and the UI would still believe in it"
        )
        assert "speed_kmh" in keys, "the air speed is the one flow number the solver owns"

    def test_the_inflow_angles_are_not_in_the_payload(self):
        keys = self._payload_keys()
        for name in ("yaw_deg", "pitch_deg", "yawDeg", "pitchDeg", "turbulence"):
            assert name not in keys, f"{name} reached the request: the loads cannot answer to it"

    def test_the_pitch_control_names_the_air_and_not_the_car(self):
        html = self._text("frontend", "index.html")
        label = re.search(r'<label for="wind_pitch">([^<]*)', html)
        assert label, "index.html no longer labels the wind_pitch control"
        assert "Airflow pitch" in label.group(1), label.group(1)
        for control in ("wind_pitch", "wind_yaw"):
            title = re.search(rf'id="{control}"[^>]*title="([^"]+)"', html)
            assert title, f"{control} lost its explanation"
            assert "not the car" in title.group(1), title.group(1)

    def test_the_engineer_panel_calls_them_inflow_angles(self):
        js = self._text("frontend", "js", "engineerTools.js")
        assert "V∞" in js, "the panel's freestream line is gone"
        assert re.search(
            r"inflow yaw[^`]*inflow pitch[^`]*solver gets the speed", js
        ), "the panel quotes the inflow angles as if the loads answered to them"


class TestTheDrawnWakeIsTheComputedWake:
    """The wake toggle used to draw a cone sized off the model's bounding box.

    Measured against the shipped field before this was fixed. On the built-in
    car (length 4.23 m, height 0.91 m, half-width 0.90 m) the cone's radius was
    half the box span, its axis sat at mid-box height, its opacity was fixed at
    0.08 whatever the momentum balance reported, and it tapered to a point
    downstream - while the field's deficit region starts 0.22 m tall (0.24 x
    height) and *grows* at 0.42 x height per car length, centred 0.38 m above
    the floor, with its depth set by drag. Worse, the field had no lateral
    structure at all: the deficit depended on height only, so a probe 9 m
    outboard of a 1.8 m car still read 0.605 U against 0.568 U on the axis.
    """

    def _text(self, *parts: str) -> str:
        return (Path(__file__).resolve().parents[1].joinpath(*parts)).read_text(
            encoding="utf-8"
        )

    def test_the_wake_deficit_has_edges_in_both_directions(self):
        js = self._text("frontend", "js", "flowEngine.js")
        assert re.search(r"z0: fr\.halfZ,\s*\n\s*zGrowth: fr\.halfZ \* 0\.42", js), (
            "the wake lost its lateral semi-axis"
        )
        deficit = re.search(r"if \(s > wake\.sStart\) \{(.*?)\n    \}", js, re.DOTALL)
        assert deficit, "the mean-deficit term behind the car is gone"
        assert "axes[1]" in deficit.group(1) and "zz" in deficit.group(1), (
            "the deficit ignores the lateral axis again: it is a sheet spanning "
            "the whole tunnel, so every tracer in it reads as wake"
        )
        turb = re.search(r"const inWake = (.*?);", js, re.DOTALL)
        assert turb and "axes[1]" in turb.group(1), (
            "the turbulence amplification no longer follows the same ellipse as "
            "the deficit it is supposed to mark"
        )

    def test_the_envelope_is_built_from_the_field(self):
        viewer = self._text("frontend", "js", "viewer3d.js")
        assert re.search(r"wakeShape\(this\.flow, this\.config, this\.bounds\)", viewer), (
            "the wake toggle no longer asks the field what its wake looks like"
        )
        assert "ConeGeometry" not in viewer, (
            "a cone is back in the viewer: nothing in the model makes a wake that shape"
        )

    def test_the_viewer_restates_no_wake_number(self):
        viewer = self._text("frontend", "js", "viewer3d.js")
        for name in ("sigma0", "hCenter", "decayLen", "zGrowth", "sStart"):
            assert name not in viewer, (
                f"{name} is duplicated in the viewer - the drawn wake would drift "
                "from the advected one"
            )

    def test_the_toggle_says_which_surface_it_draws(self):
        engine = self._text("frontend", "js", "flowEngine.js")
        defaults = re.search(
            r"export function wakeShape\([^)]*?edge = ([\d.]+), recover = ([\d.]+)", engine
        )
        assert defaults, "wakeShape no longer publishes the fractions it draws at"
        edge, recover = (float(x) for x in defaults.groups())
        html = self._text("frontend", "index.html")
        toggle = re.search(
            r'id="show_wake"[^>]*title="([^"]+)"\s*/>\s*([^<]+)</label>', html
        )
        assert toggle, "the wake toggle lost its label or its explanation"
        title, label = toggle.groups()
        assert "Wake hint" not in html, "the toggle still promises a hint, not a surface"
        assert f"{edge * 100:.0f}%" in label, (label, edge)
        assert f"{recover * 100:.0f}%" in title, (title, recover)
        assert "bounding box" in title and "momentum balance" in title, title

    def test_the_settings_round_trip_cannot_unpublish_the_geometry(self):
        """`flow.geo` is what makes the wake the car's own, not the box's.

        Only `_syncFlowGeo` publishes it, and `setFlowSettings` merges the page
        payload over the live object, so a `geo` key there - even `null`, which
        is what DEFAULT_FLOW carries - would silently drop the solved stations
        on every slider move and send the field back to bbox fractions.
        """
        app = self._text("frontend", "js", "app.js")
        body = re.search(r"function getFlowSettings\(\) \{\s*return \{(.*?)\};", app, re.DOTALL)
        assert body, "getFlowSettings no longer returns a literal payload"
        assert not re.search(r"(^|[{,\s])geo\s*:", body.group(1)), (
            "the page payload carries a geo key: it would overwrite the geometry "
            "the viewer publishes from the built-in car's solved surfaces"
        )
        assert re.search(r"this\.flow = \{ \.\.\.this\.flow, \.\.\.flow \};", self._text("frontend", "js", "viewer3d.js")), (
            "setFlowSettings no longer merges over the live flow object"
        )
