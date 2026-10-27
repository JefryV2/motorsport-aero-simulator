"""Physics-based sanity test suite for motorsport-aero-simulator.

Verifies essential aerodynamic principles on the analytical baseline model:
1. Quadratic force scaling with freestream velocity (F ~ V^2)
2. Bounded aerodynamic balance and Center of Pressure within physical wheelbase
3. Monotonicity of wing incidence angles vs downforce and drag
4. Ground effect ride-height and diffuser angle sensitivity
5. Dynamic pressure and non-dimensional coefficient consistency
6. Schema boundary validation on speed
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import WHEELBASE_M, simulate


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
