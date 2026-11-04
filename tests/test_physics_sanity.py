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
13. Past the stall knee the section is handed to the flat-plate polar instead of
    pinned at the top of the thin-airfoil line, and every angle below the knee
    still prices exactly as the attached line alone
14. The pressure map's legend reports the extremes of the field it returned and
    names its own paint limits, instead of a range no vertex ever reached
15. The pressure map's shading gains belong to each surface alone, so moving one
    surface cannot repaint another
16. The floor and diffuser are shaded from the Cl the forces charge them, so the
    map holds no second copy of the ground-effect, rake and throat math
17. The body and wheels shade the solver's one lumped drag coefficient between
    them instead of the map inventing two amplitudes for it
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
# One point on each ground-effect surface. The diffuser probe sits midway along the
# throat band, where the throat's own decay is the only term the sweep below finds
# moving. The wing terms are swept over the solved boxes instead.
PROBE_DIFFUSER = [[0.5 * (DIFFUSER_X0_M + analytical.FLOOR_X_END_M), 0.0, 0.05]]


def _rear_flap(angle_deg: float) -> dict:
    return {"angle_deg": angle_deg, "span_m": 0.85, "chord_m": 0.05}


def _rear_main(angle_deg: float) -> dict:
    return {"angle_deg": angle_deg, "span_m": 0.95, "chord_m": 0.4}


def _front_main(gurney_mm: float) -> dict:
    return {"angle_deg": 2.0, "span_m": 1.8, "chord_m": 0.35, "gurney_mm": gurney_mm}


def _floor_cloud() -> list[list[float]]:
    """A millimetre above the tunnel floor plane, across the whole track.

    On the default car these points average underbody 0.5899, diffuser 0.1595, body
    0.7744 and wheels 0.0674 weight, and 0.0008 of front wing - which is what makes
    this the cloud on which a wing move has to show nothing.
    """
    return [
        [float(x), y, 0.001]
        for x in np.linspace(analytical.FLOOR_X_START_M, analytical.FLOOR_X_END_M, 25)
        for y in (-0.6, -0.3, 0.0, 0.3, 0.6)
    ]


def _cloud_cp(overrides: dict, probe: list[list[float]]) -> float:
    """Mean Cp over a probe cloud for one setup.

    The picture's shading scale is a fixed paint gain now, so nothing has to be
    pinned to compare two setups: every term carries only its own surface's Cl, and
    a setup that does not move a surface cannot move that surface's shading.
    """
    config = FormulaAeroConfig(**overrides)
    return float(np.mean(aero_field.compute_cp_field(probe, None, config, BOUNDS)["cp"]))


def _mean_weights(config: FormulaAeroConfig, probe: list[list[float]]) -> dict[str, float]:
    """How much of each surface region the map assigns to a probe cloud.

    `compute_cp_field` normalises its vertices onto `BOUNDS` and maps them back to
    metres on the same envelope, which over these bounds is the identity, so the
    masks can be read straight off the probe coordinates.
    """
    frame = aero_field.SurfaceRegions.from_config(config)
    taken = [frame.weights(x, y, z) for x, y, z in probe]
    return {k: sum(w.get(k, 0.0) for w in taken) / len(taken) for k in taken[0]}


def _spread(values: tuple[float, ...], factors: dict[float, float]) -> float:
    """Ratio of two differences taken against a common far station.

    Subtracting the same station from both removes every additive term, and
    dividing the two differences removes the region weight and the angle
    prefactor, which are common to both. Only the decay length survives.
    """
    far = values[-1]
    return (factors[values[0]] - factors[far]) / (factors[values[1]] - factors[far])


def _wake_tail(config: FormulaAeroConfig) -> float:
    """Where the drawn air leaves the box, averaged over the streamline lines.

    `generate_streamlines` bends the air only aft of its own wake onset, so the
    end of a line is where that bend is largest and the points ahead of the car
    are a constant that dilutes it: the same DRS move reads 0.00392 of height at
    the tail and 0.00055 over every point.
    """
    lines = generate_streamlines(config, BOUNDS, n_lines=6, n_points=20)
    return float(np.mean([line[-1][2] for line in lines]))


def _envelope_cloud() -> list[list[float]]:
    """3,600 points through the reference envelope, in the bounds frame the map takes.

    `compute_cp_field` maps its normalised coordinates onto `CAR_LENGTH_M` x
    `CAR_HEIGHT_M` x 2`CAR_HALF_WIDTH_M`, so the metres below are converted the same
    way a mesh would be: this is the car the region masks describe, not a box.
    """
    length, height, half_width = (
        aero_field.CAR_LENGTH_M,
        aero_field.CAR_HEIGHT_M,
        aero_field.CAR_HALF_WIDTH_M,
    )
    vmin = np.array(BOUNDS["min"])
    span = np.array(BOUNDS["max"]) - vmin
    return [
        [
            float(vmin[0] + x / length * span[0]),
            float(vmin[1] + (0.5 + y / (2.0 * half_width)) * span[1]),
            float(vmin[2] + z / height * span[2]),
        ]
        for x in np.linspace(0.0, length, 60)
        for y in (-0.4, 0.0, 0.4)
        for z in np.linspace(0.01, height, 20)
    ]


class TestFieldConsumersShareSolverState:
    """The pressure map and streamlines must follow the same state as the forces.

    `compute_cp_field` and `generate_streamlines` are the API's only picture of
    the car, and they read the sealed rear flap, the rear ride height in place of
    the diffuser throat, and their own rake coefficient, so the screen could show
    loads the solver was not charging. They also priced a wing's suction and the
    air it bends from raw degrees over ceilings of the picture's own - `(deg / 15)
    * 1.8`, `deg / 30` - which kept growing through the stall knee the forces turn
    back at, so a stalled wing was the most heavily loaded thing on screen.
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
        cloud = self._cloud("rear_wing")
        unpinned_shift = _cloud_cp(
            {"rear_wing_flap": _rear_flap(20.0), "drs_active": True}, cloud
        ) - _cloud_cp({"rear_wing_flap": _rear_flap(20.0)}, cloud)

        # Measured over the solved wing box: +0.055683 sealed -> +0.080819 open, worth
        # +0.025136 Cp. It used to read +0.143 because the map multiplied every term by
        # the whole car's Cl, and opening DRS shrinks that too - the flap's own relief
        # is the smaller number, and the two paths now agree exactly because the scale
        # is a constant.
        assert shift > 0.02, f"DRS barely moved the rear wing pressure ({shift:+.4f})"
        assert unpinned_shift == pytest.approx(shift, abs=1e-9), (
            f"the picture still answers to the car's total Cl: {unpinned_shift} vs {shift}"
        )

    def test_wake_strength_responds_to_drs(self):
        """The drawn wake must weaken with the effective flap, not the sealed one."""
        sealed = FormulaAeroConfig(rear_wing_flap=_rear_flap(20.0))
        opened = sealed.model_copy(update={"drs_active": True})

        # Measured: the priced rear planform Cl drops 0.614520 -> 0.475869 and the
        # air leaves the box 0.46261154 -> 0.46653482 higher.
        assert _wake_tail(opened) > _wake_tail(sealed) + 0.001

    def test_the_rear_main_plane_reaches_the_drawn_wake(self):
        """`wake_strength` read the sealed flap's degrees alone: the main plane was invisible.

        Swept at the tail with the flap sealed at 20 deg: 0.46261154 at 5 deg,
        0.45166109 at 10, the lowest point 0.43977330 at the solver's own knee of
        16.4140 deg, then back up to 0.44205550 at 20 and 0.44904737 at 35 - the same
        turn the forces make. The old term drew all five identically, and every degree
        it was given deepened the wake further.
        """
        onset = math.degrees(analytical.SECTION_CL_MAX / (2 * math.pi))
        angles = (5.0, 10.0, onset, 20.0, 35.0)
        tail = [
            _wake_tail(
                FormulaAeroConfig(
                    rear_wing_flap=_rear_flap(20.0),
                    rear_wing_main={"angle_deg": a, "chord_m": 0.4, "span_m": 0.95},
                )
            )
            for a in angles
        ]

        assert tail[0] > tail[1] > tail[2], f"loading the main plane lifted the wake: {tail[:3]}"
        assert tail.index(min(tail)) == 2, f"the wake does not bottom out at the knee: {tail}"
        assert tail[-1] - min(tail) > 0.005, f"the stall only lifted the wake {tail[-1] - min(tail):.5f}"

    def test_a_surface_is_only_isolable_at_fixed_geometry(self):
        """The wing box is not a fixed set of surface points, so ride height cannot read off it.

        The old version of this test compared a 10 mm nose with a 60 mm one over the
        solved wing box and took the drop for front-wing suction. With the paint scale
        fixed, the same comparison shows what was really moving: those probes change
        mask between the two setups - front-wing weight 0.9525 -> 1.0000, underbody
        0.0469 -> 0.1591, body 0.0475 -> 0.0 - so the mean reads +0.146678 -> +0.133303
        (more relief) while the priced planform Cl goes the other way, 0.422245 ->
        0.408477. The ride-height physics is owned by `wing_planform_cl` itself, which
        `test_front_wing_lift_falls_as_the_nose_rises` pins; what the picture owes is
        that it shades that Cl linearly, and that is testable only where the geometry -
        and therefore the masks - do not move.
        """
        cloud = self._cloud("front_wing")
        default_w = _mean_weights(FormulaAeroConfig(), cloud)
        w_low = _mean_weights(FormulaAeroConfig(ride_height_front_mm=10.0), cloud)
        w_high = _mean_weights(FormulaAeroConfig(ride_height_front_mm=60.0), cloud)
        assert w_high["front_wing"] > w_low["front_wing"] + 0.02, (
            f"the wing box mask no longer moves with the nose: {w_low} -> {w_high}"
        )
        assert w_low["underbody"] < w_high["underbody"], (
            f"the floor mask did not grow with the nose: {w_low} -> {w_high}"
        )
        assert default_w["front_wing"] > 0.99, f"the setups below no longer share a mask: {default_w}"

        setups = {
            "clean": {"front_wing_main": _front_main(0.0)},
            "tab": {"front_wing_main": _front_main(15.0)},
            "angle": {"front_wing_main": {**_front_main(0.0), "angle_deg": 6.0}},
        }
        cl = {k: aero_field.wing_planform_cl(FormulaAeroConfig(**v))[0] for k, v in setups.items()}
        cp = {k: _cloud_cp(v, cloud) for k, v in setups.items()}
        tab_rate = (cp["tab"] - cp["clean"]) / (cl["tab"] - cl["clean"])
        angle_rate = (cp["angle"] - cp["clean"]) / (cl["angle"] - cl["clean"])

        # Measured: -0.23054940 per Cl from the tab, -0.23066415 from the incidence,
        # against -CL_PAINT_GAIN x the cloud's own mean wing weight = -0.23055.
        assert tab_rate == pytest.approx(angle_rate, rel=2e-3), (
            f"the surface shades two rates for one Cl: tab {tab_rate} vs angle {angle_rate}"
        )
        assert tab_rate == pytest.approx(
            -aero_field.CL_PAINT_GAIN * default_w["front_wing"], rel=0.02
        ), f"the wing term carries a gain that is not the picture's own: {tab_rate}"

    def test_the_map_shades_the_priced_section_not_its_own_ramp(self):
        """The map's two wing lines were `(degrees / 15) * 1.8` and `(degrees / 20) * 1.6`.

        Four literals the solver does not have, a 0.6 weight on the flap degrees no
        solver applies, and the *floor's* ground-effect decay (`GE_DECAY_FRONT_M`,
        0.015 m) bolted onto a wing the solver gains 15 % on at its own 0.14 m decay
        length. All four now come through `analytical.wing_section_cl`, so the only
        way to shade a wing is to price it.
        """
        src = inspect.getsource(aero_field.compute_cp_field)
        for restated in ("1.8", "1.6", "/ 15.0", "/ 20.0", "* 0.6", "GE_DECAY_FRONT_M"):
            assert restated not in src, f"the map still restates {restated} instead of pricing it"
        assert "wing_planform_cl" in src
        # The map shades a planform average, so the element-level call sits in the
        # helper that takes it rather than in `compute_cp_field` itself.
        assert "wing_section_cl" in inspect.getsource(aero_field._planform_cl)

    def test_the_surface_cl_weights_the_area_it_covers(self):
        """Each element carries the drawn suction in proportion to the area it covers.

        That is how `simulate` sums a planform, and the picture has to follow it: +2 deg
        on the 0.63 m² main plane moves the drawn front Cl by +0.17574, +2 deg on the
        0.21 m² flap by +0.05702, the 3.08 ratio of their areas. A plain mean of the two
        sections instead gives +0.11827 against +0.11404 - near-equal, so the steep
        little flap would own a surface it barely covers.
        """
        default = FormulaAeroConfig()
        base = aero_field.wing_planform_cl(default)[0]
        deeper_main = default.model_copy(
            update={"front_wing_main": default.front_wing_main.model_copy(update={"angle_deg": 4.0})}
        )
        steeper_flap = default.model_copy(
            update={"front_wing_flap": default.front_wing_flap.model_copy(update={"angle_deg": 10.0})}
        )
        main_delta = aero_field.wing_planform_cl(deeper_main)[0] - base
        flap_delta = aero_field.wing_planform_cl(steeper_flap)[0] - base

        assert main_delta > 2.0 * flap_delta, (
            f"the planform is not area-weighted: main {main_delta:+.5f} vs flap {flap_delta:+.5f}"
        )

    def test_rear_wing_suction_peaks_at_the_solver_knee_and_falls_past_it(self):
        """A stalled wing was the most heavily loaded surface on the screen.

        The old ramp drew -0.2232 Cp at 5 deg and -1.2752 at 35 deg, growing without
        a ceiling while the forces were already unloading that wing. The map now
        peaks at the solver's own knee and gives the suction back: +0.074933 at 5 deg
        down to +0.053206 at the knee (16.4140 deg) and back up to +0.062756 at 35.
        The numbers sit positive because the envelope cloud this test probes is mostly
        body and floor, not wing - the shape of the polar is what is under test here.
        """
        onset = math.degrees(analytical.SECTION_CL_MAX / (2 * math.pi))
        angles = (5.0, 10.0, 14.0, onset, 18.0, 20.0, 25.0, 35.0)
        cloud = self._cloud("rear_wing")
        cp = [_cloud_cp({"rear_wing_flap": _rear_flap(a)}, cloud) for a in angles]

        assert all(b < a for a, b in zip(cp[:4], cp[1:4], strict=False)), (
            f"the front of the polar is not attached-linear: {cp[:4]}"
        )
        assert cp.index(min(cp)) == 3, f"the peak suction is not at the knee: {cp}"
        assert all(b > cp[3] for b in cp[4:]), f"past the knee the wing keeps loading: {cp[4:]}"
        assert cp[-1] - cp[3] > 0.005, f"the stall only cost {cp[-1] - cp[3]:.4f} Cp"

    def test_a_gurney_tab_reaches_the_pressure_map(self):
        """15 mm of tab on the front wing main plane moved the picture by 0.00000.

        The ramp had no term for it while `_wing_cl` charges 0.08 Cl per 10 mm: the
        screen showed a tabbed wing and a clean one identically. Over the solved
        front wing box the tab is now worth +0.133725 -> +0.110262 Cp, -0.023462,
        for the 0.417587 -> 0.519355 planform Cl it buys.
        """
        cloud = self._cloud("front_wing")
        clean = _cloud_cp({"front_wing_main": _front_main(0.0)}, cloud)
        tabbed = _cloud_cp({"front_wing_main": _front_main(15.0)}, cloud)

        assert tabbed < clean - 0.015, f"a Gurney tab barely shows on the map ({tabbed - clean:+.5f})"

    def test_the_drawn_air_bends_with_the_priced_cl(self):
        """The streamlines took the raw flap degrees over an invented ceiling twice.

        `wake_strength` was `degrees / 30` and the front-wing downwash `-0.12 * (deg /
        15)`, so a 35 deg rear flap deflected the wake 1.16667 of span on a car whose
        rear wing the solver had already unloaded past its knee at 16.41 deg. Both now
        scale the priced planform Cl, calibrated so the shipped default still draws the
        numbers the ramps drew: rear 0.76343 Cl x 0.655 = 0.50005 against the old
        15/30 = 0.5, front 0.41759 x 0.153 = 0.06389 against 0.12 x 8/15 = 0.064.
        """
        default = FormulaAeroConfig()
        fw_cl, rw_cl = aero_field.wing_planform_cl(default)
        assert rw_cl * aero_field.WAKE_DEFLECT_PER_CL == pytest.approx(15.0 / 30.0, rel=0.01)
        assert fw_cl * aero_field.DOWNWASH_PER_CL == pytest.approx(0.12 * 8.0 / 15.0, rel=0.01)

        onset = math.degrees(analytical.SECTION_CL_MAX / (2 * math.pi))
        knee = FormulaAeroConfig(rear_wing_flap={"angle_deg": onset, "span_m": 0.9, "chord_m": 0.15})
        stalled = FormulaAeroConfig(rear_wing_flap={"angle_deg": 35.0, "span_m": 0.9, "chord_m": 0.15})
        _, knee_cl = aero_field.wing_planform_cl(knee)
        _, stalled_cl = aero_field.wing_planform_cl(stalled)
        assert stalled_cl < knee_cl, "the priced rear wing did not unload past the knee"

        def lowest(lines: list[list[list[float]]]) -> float:
            return min(p[2] for line in lines for p in line)

        knee_z = lowest(generate_streamlines(knee, BOUNDS, n_lines=6, n_points=40))
        stalled_z = lowest(generate_streamlines(stalled, BOUNDS, n_lines=6, n_points=40))
        assert stalled_z > knee_z, (
            f"the drawn wake still deepens past the knee ({knee_z:.5f} -> {stalled_z:.5f})"
        )

    def test_diffuser_suction_follows_throat_height(self):
        """The map used to price the diffuser with the rear ride height only.

        `diffuser_ride_height_mm` moved the solver's diffuser Cl and nothing on
        the screen; the throat height now drives the same decay the solver uses.
        Both ride heights are pinned at 10 mm so the floor line stays under the
        whole sweep - a throat below it is raised to it, as it is on the car.

        Measured at the frozen paint gain, throat 10 -> 30 -> 60 mm gives +0.127600
        -> +0.143400 -> +0.152500 Cp, and the ratio of those two legs against the
        60 mm station is 2.7362637 where the solver's own exp(-h / 0.025) decay gives
        2.7537647 - a 0.64 % deviation. It is not the law drifting: the map rounds
        every vertex to four decimals, and the differenced leg spans only 0.0091 Cp,
        so the rounding on its own is worth 0.55 % of it. The old scale was 4.3x
        larger, which made the same rounding invisible and the same lock look tighter
        than the picture can be.
        """
        stations = (10.0, 30.0, 60.0)
        cp = {
            v: _cloud_cp(
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
        assert _spread(stations, cp) == pytest.approx(_spread(stations, ge), rel=1e-2)


class TestEachSurfaceCarriesOnlyItsOwnLoad:
    """One surface's shading must not answer to another surface's Cl.

    `compute_cp_field` took its scale from the car it was handed - `cl_scale =
    sim.forces.cl_total / 3.0`, `cd_scale = cd_total / 1.2` - and multiplied *every*
    term by it, so the picture of the floor was a function of the wings. Over the
    cloud below, whose priced floor force is byte-identical at 221.213973 N in every
    one of these setups, the committed picture moved +17.59 % on a rear-wing stall,
    +0.85 % on unloading it, +0.90 % on DRS and -4.23 % on a 20 deg front wing. Both
    gains are now fixed constants frozen at the shipped default car, which keeps that
    car byte-identical (0 of 3,600 envelope vertices) and leaves each surface answering
    only to the load it was charged with.
    """

    def test_a_wing_move_leaves_the_floor_alone(self):
        """The floor cloud is now the same number whatever the wings do."""
        floor = _floor_cloud()
        base = _cloud_cp({}, floor)
        flap20 = {"angle_deg": 20.0, "span_m": 0.85, "chord_m": 0.05}
        wing_only = {
            "rear wing stalled": {"rear_wing_main": _rear_main(35.0)},
            "rear wing unloaded": {"rear_wing_main": _rear_main(0.0)},
            "rear flap sealed": {"rear_wing_flap": flap20},
            "DRS open": {"rear_wing_flap": flap20, "drs_active": True},
            "speed 350": {"speed_kmh": 350.0},
        }
        for name, overrides in wing_only.items():
            moved = _cloud_cp(overrides, floor)
            assert moved == pytest.approx(base, abs=1e-12), (
                f"{name} repainted the floor: {base:+.6f} -> {moved:+.6f}"
            )

    def test_the_map_takes_its_gain_from_nothing_but_its_own_constants(self):
        """No term may read the summed coefficients again."""
        src = inspect.getsource(aero_field.compute_cp_field)
        for summed in ("cl_total", "cd_total", "/ 3.0", "/ 1.2"):
            assert summed not in src, f"the map still scales itself by the car's {summed}"
        for name in ("CL_PAINT_GAIN", "CD_PAINT_GAIN"):
            assert name in src, f"the map no longer states its own {name}"

    def test_the_floor_still_answers_to_the_floor(self):
        """Freezing the gain must not have deadened the surface it was meant to free.

        Measured on the same cloud, against the default floor's +0.171992: rake 0 -> 3
        deg gives +1.29 % / -1.29 %, rear ride height 20 mm and 90 mm give -28.31 % and
        +10.61 %, and an 18 deg diffuser -0.40 %. The rear wing and DRS moves above are
        the ones that went silent on this cloud, and they should have.
        """
        floor = _floor_cloud()
        base = _cloud_cp({}, floor)

        assert _cloud_cp({"rake_deg": 3.0}, floor) < base - 0.002
        assert _cloud_cp({"rake_deg": 0.0}, floor) > base + 0.002
        assert _cloud_cp({"ride_height_rear_mm": 20.0}, floor) < base - 0.03
        assert _cloud_cp({"ride_height_rear_mm": 90.0}, floor) > base + 0.01
        assert _cloud_cp({"diffuser_angle_deg": 18.0}, floor) < base - 0.0005


class TestTheUnderbodyShadesTheClTheSolverChargesIt:
    """The floor and diffuser used to carry the solver's formula, written out twice.

    `compute_cp_field` shaded the underbody with `exp(-h_rear / GE_DECAY_REAR_M) *
    rake * 1.2` and the diffuser with `sin(angle) * 0.8 * exp(-throat /
    GE_DECAY_DIFFUSER_M) * rake` - the decay lengths, the rake coefficient and the
    throat geometry each lived in two files, and the ceilings 1.2 and 0.8 hid the
    priced 0.35 and 0.25 they were divided by. Both surfaces now take
    `analytical.underbody_cl`, the pair `_underbody_forces` prices the downforce
    from, times the ratio the picture had already been drawing: the whole field is
    byte-identical, 0 of 67,050 vertices across 18 setups and two clouds, with the
    two terms agreeing with the old formula to 1.1e-16 and 2.8e-17.
    """

    @staticmethod
    def _painting(config: FormulaAeroConfig) -> dict[str, float]:
        """Every Cl the map spends on a surface, already scaled by its own amplitude."""
        fw_cl, rw_cl = aero_field.wing_planform_cl(config)
        cl_floor, cl_diff = analytical.underbody_cl(config)
        return {
            "front_wing": fw_cl,
            "rear_wing": rw_cl,
            "underbody": cl_floor * aero_field.FLOOR_PAINT_PER_CL,
            "diffuser": cl_diff * aero_field.DIFFUSER_PAINT_PER_CL,
        }

    def test_the_map_holds_no_second_copy_of_the_ground_effect_math(self):
        src = inspect.getsource(aero_field.compute_cp_field)
        for restated in (
            "GE_DECAY",
            "RAKE_COEF",
            "math.exp",
            "math.sin",
            "ride_height_rear_mm",
            "diffuser_angle_deg",
            "diffuser_throat_height_m",
            "* 1.2",
            "* 0.8",
        ):
            assert restated not in src, f"the map still computes the solver's {restated}"
        assert "underbody_cl(" in src, "the map no longer asks the solver what it drew"
        for name in ("FLOOR_PAINT_PER_CL", "DIFFUSER_PAINT_PER_CL"):
            assert name in src, f"the map no longer states its own {name}"

    def test_the_amplitudes_are_the_ratio_they_replaced(self):
        """The picture is unchanged because 1.2 / 0.35 and 0.8 / 0.25 are the new numbers.

        Stated this way, a solver-side change to either ceiling moves the drawing
        with the forces instead of leaving it on the old amplitude.
        """
        assert (
            aero_field.FLOOR_PAINT_PER_CL * analytical.CL_FLOOR_MAX
            == pytest.approx(1.2, abs=1e-12)
        )
        assert (
            aero_field.DIFFUSER_PAINT_PER_CL * analytical.CL_DIFFUSER_MAX
            == pytest.approx(0.8, abs=1e-12)
        )

    def test_the_pair_the_map_shades_is_the_pair_the_forces_charge(self):
        """Both ends of the extraction: the formula the pair comes from, and who spends it.

        `underbody_cl` must still be the solver's own arithmetic over its named
        constants, and `_underbody_forces` must still build its downforce from that
        pair - if either end drifts, the picture is shading a load the car is not
        charged with while the numbers on the force cards say something else.
        """
        config = FormulaAeroConfig(
            ride_height_rear_mm=20.0, rake_deg=3.0, diffuser_angle_deg=18.0
        )
        ge_rear = math.exp(
            -config.ride_height_rear_mm / 1000.0 / analytical.GE_DECAY_REAR_M)
        rake_boost = 1.0 + config.rake_deg * analytical.RAKE_COEF_PER_DEG
        ge_throat = math.exp(
            -analytical.diffuser_throat_height_m(config) / analytical.GE_DECAY_DIFFUSER_M)
        expected = (
            analytical.CL_FLOOR_MAX * ge_rear * rake_boost,
            analytical.CL_DIFFUSER_MAX
            * math.sin(math.radians(config.diffuser_angle_deg))
            * ge_throat
            * rake_boost,
        )
        cl_floor, cl_diff = analytical.underbody_cl(config)
        assert (cl_floor, cl_diff) == pytest.approx(expected, abs=1e-12), (
            "the pair the picture spends is no longer the solver's own ground effect"
        )

        speed_ms = 200.0 / 3.6
        q = 0.5 * analytical.RHO * speed_ms**2
        downforce = analytical._underbody_forces(config, speed_ms)[0]
        assert downforce == pytest.approx(q * 2.2 * (cl_floor + cl_diff), abs=1e-9), (
            "the forces no longer spend the pair the picture shades"
        )

    def test_the_floor_paints_the_cl_the_floor_is_charged(self):
        """On a cloud over the floor, the masks do not move, so the difference is exact.

        Every one of these setups leaves the floor cloud's region weights identical
        (underbody 0.589881, diffuser 0.159524), which makes the change in the drawn
        mean fully attributable to the priced Cl. The helper reads the amplitude
        constants themselves, so this is not a test of their value - that is the
        ratio lock above - it is a test that the map spends exactly those constants
        on the pair the solver returns, which a multiplier restated inside the term
        would break. The tolerance is the map's own 4-decimal rounding band.
        """
        floor = _floor_cloud()
        cases = {
            "rear ride height 20 -> 90 mm": (
                {"ride_height_rear_mm": 20.0},
                {"ride_height_rear_mm": 90.0},
            ),
            "rake 0 -> 3 deg": ({"rake_deg": 0.0}, {"rake_deg": 3.0}),
            "diffuser 5 -> 18 deg": ({"diffuser_angle_deg": 5.0}, {"diffuser_angle_deg": 18.0}),
        }
        for name, (a_over, b_over) in cases.items():
            config_a = FormulaAeroConfig(**a_over)
            config_b = FormulaAeroConfig(**b_over)
            weights = _mean_weights(config_a, floor)
            assert _mean_weights(config_b, floor) == pytest.approx(weights, abs=1e-12), (
                f"{name} moved the region masks, so this cloud no longer isolates the Cl"
            )
            paint_a = self._painting(config_a)
            paint_b = self._painting(config_b)
            predicted = -sum(
                weights[key] * (paint_b[key] - paint_a[key]) for key in paint_a
            ) * aero_field.CL_PAINT_GAIN
            measured = _cloud_cp(b_over, floor) - _cloud_cp(a_over, floor)
            assert measured == pytest.approx(predicted, abs=1e-4), (
                f"{name}: priced {predicted:+.6f} vs painted {measured:+.6f}"
            )


class TestTheBodyAndWheelsShadeTheDragTheyAreCharged:
    """The map divided a drag the solver never divides, and lost 5 % of it.

    `_body_drag` charges wheels, suspension and cockpit as one lump,
    `q * REF_AREA * 0.75`, with no split of it anywhere in the model - while
    `compute_cp_field` shaded a body at 0.25 and wheels at 0.45 of that same drag
    gain. Those two amplitudes had no source: read as shares of the lump they are a
    third and 60 %, so the picture drew 0.70 of the 0.75 the car is charged with and
    5 % of the drag it was drawing from had no surface at all. Both terms now come
    from `CD_BODY` and add up to it, which keeps the body's drawn amplitude exactly
    0.25 (a third of 0.75) and lifts only the wheels. Measured on the shipped
    default: 330 of 3,600 envelope vertices moved, none of them anywhere the wheel
    region has no weight, each by its own wheel weight times 0.05 Cd, so the mean
    went +0.208306 -> +0.210661 Cp and the 64-point ring on the tyre stations
    +0.429894 -> +0.454509 (+5.73 %). Every force came back byte-identical.
    """

    def test_the_two_surfaces_partition_the_priced_lump(self):
        """The shares add up to the drag, and the body still draws what it drew."""
        assert aero_field.BODY_CD_SHARE + aero_field.WHEELS_CD_SHARE == pytest.approx(1.0)
        assert (
            analytical.CD_BODY * aero_field.BODY_CD_SHARE == pytest.approx(0.25, abs=1e-12)
        ), "the body's drawn amplitude moved, so this was not a wheels-only fix"
        assert (
            analytical.CD_BODY * aero_field.WHEELS_CD_SHARE
            == pytest.approx(0.45 + 0.05, abs=1e-12)
        )

    def test_the_map_writes_no_amplitudes_of_its_own(self):
        src = inspect.getsource(aero_field.compute_cp_field)
        for invented in ("0.25", "0.45"):
            assert invented not in src, f"the map still shades a {invented} it made up"
        for shared in ("CD_BODY", "BODY_CD_SHARE", "WHEELS_CD_SHARE"):
            assert shared in src, f"the map no longer spends the solver's {shared}"

    def test_the_lump_is_what_the_forces_charge(self):
        """Naming the constant must not have changed the bill."""
        speed_ms = 200.0 / 3.6
        q = 0.5 * analytical.RHO * speed_ms**2
        assert analytical._body_drag(speed_ms) == pytest.approx(
            q * analytical.REF_AREA * analytical.CD_BODY, abs=1e-12
        )

    def test_the_drawn_floor_is_the_sum_of_the_priced_terms(self):
        """Rebuild the whole cloud mean from the solver's numbers, nothing else.

        Normals are absent and no vertex reaches the picture's clamp on this cloud,
        so the mean is linear in the region weights and every amplitude in the map
        shows up in it - the two drag shares, both underbody ratios, both wing Cls.
        """
        floor = _floor_cloud()
        config = FormulaAeroConfig()
        weights = _mean_weights(config, floor)
        fw_cl, rw_cl = aero_field.wing_planform_cl(config)
        cl_floor, cl_diff = analytical.underbody_cl(config)
        suction = (
            weights["front_wing"] * fw_cl
            + weights["rear_wing"] * rw_cl
            + weights["underbody"] * cl_floor * aero_field.FLOOR_PAINT_PER_CL
            + weights["diffuser"] * cl_diff * aero_field.DIFFUSER_PAINT_PER_CL
        ) * aero_field.CL_PAINT_GAIN
        pressure = (
            0.05
            + weights["body"] * analytical.CD_BODY * aero_field.BODY_CD_SHARE
            + weights["wheels"] * analytical.CD_BODY * aero_field.WHEELS_CD_SHARE
        ) * aero_field.CD_PAINT_GAIN
        assert _cloud_cp({}, floor) == pytest.approx(pressure - suction, abs=1e-4), (
            "the drawn floor is no longer the sum of what the solver charges"
        )


class TestTheLegendReportsTheFieldItReturned:
    """The map declared a range its own numbers never entered.

    `compute_cp_field` returned `"legend": {"min": -2.5, "max": 1.0}` whatever it had
    just computed. Measured over the 3,600-point envelope cloud below, the default
    car's field spans -0.0883 to +0.5560 Cp, a fully stalled one -0.1452 to +0.5560, and
    no vertex reached either end of the declared range - or the picture's own clamp,
    which the same sweep never touched either. A consumer painting against that legend
    saw the default field occupy 18.4 % of its ramp (20.0 % fully stalled), so the wing
    suction the solver had just been priced at came back as near-flat grey.
    """

    CONFIGS = {
        "default": {},
        "low_rear": {"ride_height_rear_mm": 20.0},
        "high_df": {
            "ride_height_rear_mm": 20.0,
            "ride_height_front_mm": 10.0,
            "diffuser_angle_deg": 18.0,
            "diffuser_ride_height_mm": 15.0,
        },
        "stalled": {"rear_wing_main": {"angle_deg": 35.0, "chord_m": 0.4, "span_m": 0.95}},
    }

    def _field(self, cloud: list[list[float]], overrides: dict) -> dict:
        return compute_cp_field(cloud, None, FormulaAeroConfig(**overrides), BOUNDS)

    def test_the_legend_is_sampled_from_the_values_it_returns(self):
        cloud = _envelope_cloud()
        for name, overrides in self.CONFIGS.items():
            out = self._field(cloud, overrides)
            cp = out["cp"]
            legend = out["legend"]
            assert legend["min"] == min(cp), f"{name}: {legend['min']} vs the field's {min(cp)}"
            assert legend["max"] == max(cp), f"{name}: {legend['max']} vs the field's {max(cp)}"

    def test_every_vertex_sits_inside_the_painted_range(self):
        cloud = _envelope_cloud()
        for name, overrides in self.CONFIGS.items():
            out = self._field(cloud, overrides)
            assert out["legend"]["painted_min"] == aero_field.CP_PAINT_MIN
            assert out["legend"]["painted_max"] == aero_field.CP_PAINT_MAX
            assert all(
                aero_field.CP_PAINT_MIN <= cp <= aero_field.CP_PAINT_MAX for cp in out["cp"]
            ), f"{name}: a vertex escaped the picture's own clamp"
            span = max(out["cp"]) - min(out["cp"])
            painted = aero_field.CP_PAINT_MAX - aero_field.CP_PAINT_MIN
            assert span < 0.25 * painted, (
                f"{name}: the field now covers {span / painted:.1%} of the paint ramp, "
                "so the comment above the constants needs re-measuring"
            )

    def test_the_map_restates_no_legend_range(self):
        """The extremes are read off the field, and the clamp is named, not retyped."""
        src = inspect.getsource(aero_field.compute_cp_field)
        for literal in ("-2.5", "2.5", "-3.0", "1.5"):
            assert literal not in src, f"the map still hardcodes the range {literal}"
        for name in ("CP_PAINT_MIN", "CP_PAINT_MAX", "painted_min", "painted_max"):
            assert name in src, f"the legend no longer names its own {name}"


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

    def test_pressure_map_prices_the_same_throat(self):
        """The map must not reward a shape the forces refuse to charge."""
        low = _cloud_cp({"diffuser_ride_height_mm": 10.0}, PROBE_DIFFUSER)
        mid = _cloud_cp({"diffuser_ride_height_mm": 30.0}, PROBE_DIFFUSER)
        high = _cloud_cp({"diffuser_ride_height_mm": 60.0}, PROBE_DIFFUSER)

        # Both requests sit under the floor line at the diffuser start, so the car
        # draws 39.7463 mm for each: byte-identical shading is the point of the test.
        assert low == pytest.approx(mid, abs=1e-9)
        # A throat that clears the line is worth +0.147600 -> +0.152500 Cp at the
        # picture's fixed gain - 49 of its own four-decimal rounding steps.
        assert high - mid > 0.002, f"a buildable throat is invisible on the map ({high - mid:+.5f})"

    def test_solver_and_viewer_agree_on_the_stations(self):
        """carSpec.js draws from these same numbers; a one-sided edit must fail here."""
        js_path = (
            Path(__file__).resolve().parents[1] / "frontend" / "js" / "carSpec.js"
        )
        js = js_path.read_text(encoding="utf-8")
        shared = {
            "wheelbaseM": analytical.WHEELBASE_M,
            "refAreaM2": analytical.REF_AREA,
            "bodyCd": analytical.CD_BODY,
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
            "sectionClMax": analytical.SECTION_CL_MAX,
            "stallBlendMaxDeg": analytical.STALL_BLEND_MAX_DEG,
            "flatPlateCn": analytical.FLAT_PLATE_CN,
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


class TestTheSliceLegendNamesWhatTheSliceIsPaintedWith:
    """The slice plane's colour map switches quantity; the legend never did.

    `#cpRange` was written from `viewMode` - how the *car skin* is shaded - and
    `sliceMode` only ever reached the vertex colours, so a slice on "Velocity
    magnitude" was still reported as Cp on the -2 to +1 scale. The picture and
    the sentence about it have to come from the same branch.
    """

    def _text(self, *parts: str) -> str:
        return (Path(__file__).resolve().parents[1].joinpath(*parts)).read_text(
            encoding="utf-8"
        )

    def test_the_slice_reports_the_quantity_it_coloured(self):
        viewer = self._text("frontend", "js", "viewer3d.js")
        body = re.search(r"_updateSlicePlane\(\) \{(.*?)\n  \}", viewer, re.DOTALL)
        assert body, "the slice builder is gone"
        code = body.group(1)
        assert "this._sliceRangeText" in code, "the slice no longer reports a range of its own"
        # The last assignment is the legend line; the first is the hidden-slice reset.
        reported = code.rsplit("this._sliceRangeText", 1)[1]
        assert "this.sliceMode === 'pressure'" in reported, (
            "the slice line no longer depends on the colour map it used"
        )
        assert "Cp" in reported and "km/h" in reported, (
            "one of the two slice units is missing from the line"
        )
        assert "CP_MIN" in reported and "CP_MAX" in reported, (
            "the Cp line no longer says what scale the picture was clipped to"
        )
        assert re.search(r"this\.slicePos\.toFixed\(2\)\} m", code), (
            "the station is no longer printed in metres"
        )
        assert "${station}" in reported, (
            "the line no longer says where the slice is cut"
        )
        loop = code.split("for (let i = 0; i < count; i++)", 1)
        assert len(loop) == 2, "the slice no longer colours a vertex grid"
        grid = loop[1].split("geo.setAttribute", 1)[0]
        assert "lo = painted" in grid and "hi = painted" in grid, (
            "the extremes are no longer sampled from the vertices, so the legend "
            "would quote a range the picture was not built from"
        )
        assert "cpToColor(Math.max(CP_MIN, Math.min(CP_MAX, painted)))" in grid, (
            "the Cp colour and the Cp number have split into two quantities"
        )
        assert "speedToColor(painted," in grid, (
            "the speed colour and the speed number have split into two quantities"
        )

    def test_a_hidden_slice_stops_claiming_a_range(self):
        viewer = self._text("frontend", "js", "viewer3d.js")
        head = re.search(r"_updateSlicePlane\(\) \{(.*?)\n    \}", viewer, re.DOTALL)
        assert head and "this._sliceRangeText = null;" in head.group(1), (
            "hiding the slice leaves its numbers in the legend"
        )

    def test_the_slice_station_is_a_position_the_slider_can_reach(self):
        app = self._text("frontend", "js", "app.js")
        helper = re.search(r"function setSliderRange\(.*?\n\}", app, re.DOTALL)
        assert helper, "the position sliders are no longer aligned to their own step"
        body = helper.group(0)
        assert "Math.floor(min / step)" in body and "Math.ceil(max / step)" in body, (
            "the grid is not anchored on a multiple of the step, so 0.00 m and "
            "the other printed values stay unreachable"
        )
        assert app.count("setSliderRange(") == 4, "a position slider bypasses the helper"
        for raw in ("slicePosSlider.min = ", "wandYSlider.min = ", "wandZSlider.min = "):
            assert raw not in app, f"{raw}restates the bounds without the grid"

    def test_the_colour_map_says_which_field_it_reads(self):
        html = self._text("frontend", "index.html")
        tip = re.search(r'id="slice_mode"[^>]*title="([^"]+)"', html)
        assert tip, "the colour-map selector stopped explaining itself"
        title = tip.group(1)
        assert "inviscid" in title and "advected" in title, (
            "the two options read different fields and the tooltip no longer says so"
        )
        assert "saturat" in title, "the tooltip hides that both colour scales are fixed"

    def test_the_legend_is_written_in_exactly_one_place(self):
        viewer = self._text("frontend", "js", "viewer3d.js")
        assert viewer.count("getElementById('cpRange')") == 1, (
            "the legend has a second writer, so skin and slice lines can disagree"
        )
        composer = re.search(r"_writeRangeReadout\(\) \{(.*?)\n  \}", viewer, re.DOTALL)
        assert composer, "the legend no longer composes its lines in one place"
        for name in ("_skinRangeText", "_sliceRangeText"):
            assert name in composer.group(1), f"{name} is not part of the legend"


class TestTheInteractionLayerAnswersToTheFieldBudget:
    """The air/car deflection was added on top of a clamped field, unclamped.

    `potentialVelocityAt()` holds the singularity perturbation to
    `PERTURBATION_BUDGET` x U because the doublets and vortices grow as 1/d² and
    1/d³ - inside a core they reach hundreds of m/s and tracers skip clean
    through the car. `velocityAtMesh()` then piles a slide-along-the-skin term on
    that already-capped result with no ceiling of its own, which let the tracer
    field, the velocity map and the probe report an overshoot the analytic model
    deliberately refuses: a flat plate over the skin put 40 of 2304 samples past
    twice the freestream, and the live 200 km/h slice peaked at 421 km/h.
    """

    def _text(self, *parts: str) -> str:
        return (Path(__file__).resolve().parents[1].joinpath(*parts)).read_text(
            encoding="utf-8"
        )

    def _budget(self) -> str:
        engine = self._text("frontend", "js", "flowEngine.js")
        found = re.search(r"export const PERTURBATION_BUDGET = ([0-9.]+);", engine)
        assert found, "the field no longer states its own perturbation budget"
        return found.group(1)

    def test_the_budget_is_stated_once_and_imported(self):
        engine = self._text("frontend", "js", "flowEngine.js")
        assert engine.count("export const PERTURBATION_BUDGET") == 1
        cap = re.search(r"const pCap = ([A-Za-z_]+) \* uRef;", engine)
        assert cap, "the analytic cap no longer multiplies a named budget"
        assert cap.group(1) == "PERTURBATION_BUDGET", (
            "the analytic cap reads a different constant than the one exported"
        )
        mesh = self._text("frontend", "js", "meshFlow.js")
        imports = re.search(r"import \{([^}]*)\} from '\./flowEngine\.js';", mesh)
        assert imports, (
            "meshFlow no longer imports the field it modifies, so the two "
            "ceilings can drift apart"
        )
        assert "PERTURBATION_BUDGET" in imports.group(1)

    def test_the_deflection_is_capped_against_the_wind_it_started_from(self):
        mesh = self._text("frontend", "js", "meshFlow.js")
        body = re.search(
            r"export function velocityAtMesh\(.*?\n\}", mesh, re.DOTALL
        )
        assert body, "the tracer field no longer has a near-skin velocity fn of its own"
        code = body.group(0)
        assert "const [ux, uy, uz] = windVector(flow);" in code, (
            "the deflection is measured against something other than the "
            "freestream, so a yawed or pitched car gets a different ceiling"
        )
        block = code.split("const [ux, uy, uz] = windVector(flow);", 1)[1]
        assert "const cap = PERTURBATION_BUDGET * len([ux, uy, uz]);" in block, (
            "the ceiling is no longer the budget scaled by the actual |U|"
        )
        assert re.search(r"return \[ux \+ dx \* k, uy \+ dy \* k, uz \+ dz \* k\];", block), (
            "the over-budget vector is thrown away instead of pulled back onto "
            "the cap, which would delete the deflection rather than bound it"
        )
        guard = block.find("if (dMag > cap)")
        rescale = block.find("return [ux + dx * k")
        assert guard != -1 and rescale != -1, (
            "the cap no longer tests the deflection and pulls it back inside "
            "the budget, which is the whole point of the block"
        )
        assert guard < rescale, "the rescale runs without the test that triggers it"
        # Everything the deflection adds has to be inside the function before the
        # cap runs, or the ceiling would bound the base field and not the result.
        plain = block.rfind("return [vx, vy, vz];")
        assert plain == -1 or rescale < plain, (
            "the uncapped vector is returned after the ceiling, so the cap "
            "cannot be the last word on the result"
        )
        for term in ("slide[0] * speed * push", "vx += slide[0] * speed * push"):
            if term in code:
                assert code.index(term) < code.index("const [ux, uy, uz] = windVector")
        for literal in ("0.85", "1.85"):
            assert literal not in block, (
                f"the cap restates {literal} instead of using the exported budget"
            )

    def test_the_tooltip_shares_the_number_it_is_bound_to(self):
        html = self._text("frontend", "index.html")
        tip = re.search(r'id="show_interaction"[^>]*title="([^"]+)"', html)
        assert tip, "the interaction checkbox stopped explaining itself"
        title = tip.group(1)
        assert "tracer" in title, "the tooltip no longer says which field it touches"
        assert "inviscid" in title and "forces never see it" in title, (
            "the tooltip hides that the pressure map and the forces ignore it"
        )
        assert f"within {self._budget()} U" in title, (
            "the tooltip quotes an overshoot the code no longer enforces"
        )


class TestTheChangelogStillRendersAsATable:
    """GitHub splits a table row on every unescaped `|`, code spans included, and
    then truncates the row back to the header's column count - so the prose after
    the first pipe is dropped, not shifted.

    Measured against `gh api /markdown` on the shipped file: the `Airflow pitch`
    row put 326 of its 3228 characters on the page and showed the word `wind`
    where its commit hash belongs, and `Wing ground effect` lost 564 characters
    to the same split. Both still render as four cells, which is what makes it
    quiet - a column-count check alone passes, and only reading the cells back
    shows the tail is gone. Escaping the prose pipes brings both rows back whole
    (`\\|U\\|` renders as `|U|`, with no visible backslash), and the two checks
    below keep a new row from reopening it.
    """

    SEPARATOR = re.compile(r"(?<!\\)\|")
    ROW = re.compile(
        r"^\| \d{4}-\d{2}-\d{2} \| [^|]* \| "
        r".* \| (?:`[0-9a-f]{7}`|[0-9a-f]{7}) \|$",
        re.DOTALL,
    )

    def test_every_row_has_exactly_four_columns(self):
        text = (Path(__file__).resolve().parents[1] / "docs" / "STATUS.md").read_text(
            encoding="utf-8"
        )
        rows = [line for line in text.split("\n") if line.startswith("| 2026-")]
        assert len(rows) > 30, f"the change-log table lost rows ({len(rows)})"
        broken = [
            f"line {n}: {self.SEPARATOR.findall(row)} columns, "
            f"{row[:60].rstrip()}"
            for n, row in enumerate(rows, 1)
            if len(self.SEPARATOR.findall(row)) != 5
        ]
        assert not broken, "rows whose prose escaped the table:\n" + "\n".join(broken)

    def test_every_row_names_the_commit_it_describes(self):
        text = (Path(__file__).resolve().parents[1] / "docs" / "STATUS.md").read_text(
            encoding="utf-8"
        )
        rows = [line for line in text.split("\n") if line.startswith("| 2026-")]
        unnamed = [row[:70] for row in rows if not self.ROW.match(row)]
        assert not unnamed, "rows without a date/title/body/hash shape:\n" + "\n".join(
            unnamed
        )


class TestTheStalledSectionIsHandedToThePlate:
    """`_wing_cl` used to clamp the 2D section at Cl 1.8 and stop thinking there.

    Measured on the shipped model: every rear-flap incidence from 17 deg to the
    config's own 35 deg bound returned the same Cl 1.618203 and the same 0.163435 drag
    factor, so a car run deep post-stall reported forces indistinguishable from a
    setup six degrees leaner, and the advisor called one of those no-ops the fastest
    move. The thin-airfoil line cannot fall over on its own, so past the knee the
    section is blended onto the flat-plate polar (CN = 2 sin a), which loses lift the
    way a separated wing does and carries the pressure drag the attached line has no
    term for. Every angle at or below the knee keeps the blend weight at zero, which
    is what the frozen oracle below proves.
    """

    def _attached_only(self, element, clearance_m=None):
        """The model as it shipped before the branch, frozen as the attached oracle."""
        cl_2d = max(-0.5, min(2 * math.pi * math.radians(element.angle_deg), 1.8))
        ar = element.span_m / element.chord_m
        cl_3d = cl_2d / (1 + cl_2d / (math.pi * ar * 0.85))
        cl_3d += (element.gurney_mm / 10.0) * 0.08
        cl_3d *= analytical.ground_effect_lift_factor(clearance_m)
        return cl_3d, cl_3d**2 / (math.pi * ar * 0.85)

    def _flap(self, angle_deg: float):
        """The default rear flap (0.90 x 0.15 m, AR 6) at one incidence."""
        return FormulaAeroConfig().rear_wing_flap.model_copy(
            update={"angle_deg": angle_deg}
        )

    def test_every_angle_below_the_knee_prices_exactly_as_before(self):
        cfg = FormulaAeroConfig()
        elements = (
            cfg.front_wing_main,
            cfg.front_wing_flap,
            cfg.rear_wing_main,
            cfg.rear_wing_flap,
        )
        worst = 0.0
        count = 0
        steps = int(analytical.STALL_ONSET_DEG * 4)
        for step in range(-20, steps + 1):
            angle = step / 4.0
            for proto in elements:
                element = proto.model_copy(update={"angle_deg": angle})
                for clearance in (None, 0.03, 0.15):
                    got = analytical._wing_cl(element, clearance)
                    want = self._attached_only(element, clearance)
                    worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
                    count += 1
        assert worst == 0.0, (
            f"{count} attached-region samples priced, worst drift {worst:.3e}: the "
            "post-stall blend reached below its own onset"
        )

    def test_the_blend_weight_spans_the_knee_and_no_further(self):
        onset = analytical.STALL_ONSET_DEG
        # SECTION_CL_MAX / 2 pi in degrees, which is 162 / pi**2.
        assert onset == pytest.approx(16.414032, abs=1e-5)
        for angle in (-5.0, 0.0, 8.0, 15.0, 16.0, onset):
            assert analytical._stall_blend(angle) == 0.0, f"{angle} deg is attached"
        halfway = onset + (analytical.STALL_BLEND_MAX_DEG - onset) / 2
        assert analytical._stall_blend(halfway) == pytest.approx(0.5, rel=1e-12)
        assert analytical._stall_blend(35.0) == pytest.approx(
            (35.0 - onset) / (45.0 - onset), rel=1e-12
        )
        assert analytical._stall_blend(45.0) == 1.0
        assert analytical._stall_blend(60.0) == 1.0, "the handover stops at the plate"

    def test_past_the_knee_downforce_falls_and_drag_rises(self):
        angles = (18.0, 20.0, 25.0, 30.0, 35.0)
        cl = [analytical._wing_cl(self._flap(a))[0] for a in angles]
        cd = [analytical._wing_cl(self._flap(a))[1] for a in angles]
        cl_steps = list(zip(cl, cl[1:], strict=False))
        cd_steps = list(zip(cd, cd[1:], strict=False))
        assert all(b < a for a, b in cl_steps), f"Cl does not fall: {cl}"
        assert all(b > a for a, b in cd_steps), f"drag does not rise: {cd}"
        assert cl[0] - cl[-1] > 0.4, f"only {cl[0] - cl[-1]:.4f} of lift is lost"
        assert cd[-1] > 3 * cd[0], f"separation drag did not appear: {cd[0]:.4f} -> {cd[-1]:.4f}"

    def test_the_model_now_tells_two_post_stall_setups_apart(self):
        """The defect in one line: 18 through 35 deg used to return one number."""
        pairs = [
            analytical._wing_cl(self._flap(a)) for a in (18.0, 20.0, 25.0, 30.0, 35.0)
        ]
        for index, name in ((0, "Cl"), (1, "drag factor")):
            distinct = {round(pair[index], 6) for pair in pairs}
            assert len(distinct) == len(pairs), (
                f"{name} reports the same value for {len(pairs)} incidences: "
                f"{sorted(distinct)}"
            )

    def _steepest_slope(self, angle_from: float, angle_to: float, step: float) -> float:
        """Largest |dCl/ddeg| on the rear flap between two incidences."""
        previous = analytical._wing_cl(self._flap(angle_from))[0]
        worst = 0.0
        for index in range(1, int(round((angle_to - angle_from) / step)) + 1):
            value = analytical._wing_cl(self._flap(angle_from + index * step))[0]
            worst = max(worst, abs(value - previous) / step)
            previous = value
        return worst

    def test_the_knee_is_a_bend_and_not_a_step(self):
        onset = analytical.STALL_ONSET_DEG
        below = analytical._wing_cl(self._flap(onset - 1e-9))[0]
        above = analytical._wing_cl(self._flap(onset + 1e-9))[0]
        assert abs(above - below) < 1e-7, f"{below:.9f} -> {above:.9f} across the knee"
        step = 0.01
        attached = self._steepest_slope(onset - 2.0, onset, step)
        handover = self._steepest_slope(onset, onset + 2.0, step)
        assert handover < attached, (
            f"the handover moves Cl {handover:.4f} per degree against the attached "
            f"line's {attached:.4f}: the blend is a cliff, not a bend"
        )
        assert handover < 0.05, f"Cl drops {handover:.4f} per degree entering the plate"

    def test_at_the_handover_the_section_is_the_plate(self):
        element = self._flap(45.0)
        cl_3d, cd_factor = analytical._wing_cl(element, None)
        ar = element.span_m / element.chord_m
        constant = math.pi * ar * 0.85
        # Undo the Prandtl term to read the section the blend actually handed over.
        # The inversion amplifies rounding, hence 1e-9 rather than exact.
        cl_2d = cl_3d / (1 - cl_3d / constant)
        assert cl_2d == pytest.approx(analytical._flat_plate_forces(45.0)[0], rel=1e-9)
        assert cd_factor - cl_3d**2 / constant == pytest.approx(
            analytical._flat_plate_forces(45.0)[1], rel=1e-9
        )

    def test_the_default_car_never_reaches_the_branch(self):
        cfg = FormulaAeroConfig()
        for name in ("front_wing_main", "front_wing_flap", "rear_wing_main", "rear_wing_flap"):
            angle = getattr(cfg, name).angle_deg
            assert analytical._stall_blend(angle) == 0.0, f"{name} at {angle} deg is stalled"

    def test_the_viewer_mirror_carries_the_same_branch(self):
        engine = (
            Path(__file__).resolve().parents[1] / "frontend" / "js" / "flowEngine.js"
        ).read_text(encoding="utf-8")
        assert (
            "const STALL_ONSET_DEG = (REFERENCE.sectionClMax / (2 * Math.PI)) / DEG;"
            in engine
        ), "the mirror's onset is no longer the solver's own SECTION_CL_MAX / 2 pi"
        assert "REFERENCE.stallBlendMaxDeg - STALL_ONSET_DEG" in engine, (
            "the mirror blends to something other than the solver's handover angle"
        )
        assert "REFERENCE.flatPlateCn * Math.sin(a)" in engine, (
            "the mirror's separated section is not the solver's CN = 2 sin a plate"
        )
        assert re.search(
            r"return \[cn \* Math\.cos\(a\), cn \* Math\.sin\(a\)\];", engine
        ), "the plate's lift/drag split is not the solver's normal-force resolution"
        assert "blend) * REFERENCE.sectionClMax + blend * flatPlateForces" in engine, (
            "the mirror no longer blends the knee into the plate the way the solver does"
        )
        assert re.search(r"export function wingCdFactor\(w\) \{", engine), (
            "the mirror has no second return for _wing_cl's drag factor, so the "
            "field cannot see the separation term"
        )
        body = re.search(r"function dragAreaM2\(aero\) \{.*?\n\}", engine, re.DOTALL)
        assert body, "the mirror's Cd*A sum is gone"
        assert "wingCdFactor(w)" in body.group(0), (
            "dragAreaM2 rebuilds the induced term by hand and loses the separation drag"
        )
        assert "OSWALD_E)" not in body.group(0), (
            "dragAreaM2 still inlines an induced-drag term beside the shared one"
        )
        mirror = engine.split("const STALL_ONSET_DEG")[1].split(
            "export function underbodyTerms"
        )[0]
        for literal in ("1.8", "45.0", "2.0 *"):
            assert literal not in mirror, (
                f"the mirror states {literal} where it should read REFERENCE"
            )
