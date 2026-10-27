"""Golden baseline tests for motorsport-aero-simulator v0.1.0 analytical model.

Locks in exact numeric outputs across four reference setups:
1. default
2. low_speed_100kmh
3. monaco_high_df
4. monza_low_drag

Protects baseline physics against unintended regression during refactoring.
"""

from __future__ import annotations

import math

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import simulate

GOLDEN_TARGETS = {
    "default": {
        "config": FormulaAeroConfig(),
        "expected": {
            "drag_N": 2396.1,
            "downforce_total_N": 1657.4,
            "downforce_front_N": 724.4,
            "downforce_rear_N": 933.0,
            "balance_front_pct": 43.7,
            "cop_x_m": 1.886,
            "cl_total": 0.585,
            "cd_total": 0.845,
            "efficiency": 0.69,
            "lap_time_delta_s": 0.267,
        },
    },
    "low_speed_100kmh": {
        "config": FormulaAeroConfig(speed_kmh=100.0),
        "expected": {
            "drag_N": 599.0,
            "downforce_total_N": 414.4,
            "downforce_front_N": 181.1,
            "downforce_rear_N": 233.3,
            "balance_front_pct": 43.7,
            "cop_x_m": 1.886,
            "cl_total": 0.585,
            "cd_total": 0.845,
            "efficiency": 0.69,
            "lap_time_delta_s": 0.1868,
        },
    },
    "monaco_high_df": {
        "config": FormulaAeroConfig(
            front_wing_main={"angle_deg": 6.0, "span_m": 1.8, "chord_m": 0.35, "gurney_mm": 15.0},
            front_wing_flap={"angle_deg": 18.0, "span_m": 1.8, "chord_m": 0.25, "gurney_mm": 10.0},
            rear_wing_main={"angle_deg": 10.0, "span_m": 1.0, "chord_m": 0.40, "gurney_mm": 12.0},
            rear_wing_flap={"angle_deg": 25.0, "span_m": 1.0, "chord_m": 0.30, "gurney_mm": 15.0},
            ride_height_front_mm=25.0,
            ride_height_rear_mm=45.0,
            diffuser_angle_deg=16.0,
        ),
        "expected": {
            "drag_N": 2838.6,
            "downforce_total_N": 4612.5,
            "downforce_front_N": 2702.3,
            "downforce_rear_N": 1910.2,
            "balance_front_pct": 58.6,
            "cop_x_m": 1.387,
            "cl_total": 1.627,
            "cd_total": 1.001,
            "efficiency": 1.62,
            "lap_time_delta_s": 0.0749,
        },
    },
    "monza_low_drag": {
        "config": FormulaAeroConfig(
            front_wing_main={"angle_deg": 0.0, "span_m": 1.8, "chord_m": 0.35, "gurney_mm": 0.0},
            front_wing_flap={"angle_deg": 4.0, "span_m": 1.8, "chord_m": 0.25, "gurney_mm": 0.0},
            rear_wing_main={"angle_deg": 2.0, "span_m": 1.0, "chord_m": 0.40, "gurney_mm": 0.0},
            rear_wing_flap={"angle_deg": 6.0, "span_m": 1.0, "chord_m": 0.30, "gurney_mm": 0.0},
            drs_active=True,
            ride_height_front_mm=30.0,
            ride_height_rear_mm=50.0,
        ),
        "expected": {
            "drag_N": 2364.9,
            "downforce_total_N": 1123.4,
            "downforce_front_N": 453.7,
            "downforce_rear_N": 669.7,
            "balance_front_pct": 40.4,
            "cop_x_m": 1.997,
            "cl_total": 0.396,
            "cd_total": 0.834,
            "efficiency": 0.48,
            "lap_time_delta_s": 0.3066,
        },
    },
}


def test_golden_baseline():
    """Verify that simulate() matches all golden targets within floating point tolerance."""
    for case_name, data in GOLDEN_TARGETS.items():
        res = simulate(data["config"])
        exp = data["expected"]

        for k, v in exp.items():
            if k == "lap_time_delta_s":
                actual = res.lap_time_delta_s
            else:
                actual = getattr(res.forces, k)
            assert math.isclose(actual, v, rel_tol=1e-3, abs_tol=1e-3), (
                f"[{case_name}] Mismatch on {k}: actual={actual}, expected={v}"
            )
        print(f"PASS: {case_name}")


if __name__ == "__main__":
    test_golden_baseline()
    print("\nAll golden baseline tests PASSED.")
