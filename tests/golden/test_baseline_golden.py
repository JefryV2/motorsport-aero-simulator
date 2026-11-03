"""Golden baseline tests for motorsport-aero-simulator v0.1.0 analytical model.

Locks in exact numeric outputs across four reference setups:
1. default
2. low_speed_100kmh
3. monaco_high_df
4. monza_low_drag

Protects baseline physics against unintended regression during refactoring.

Regenerated when `analytical._wing_cl` stopped applying a flat 1.12 ground-effect
gain to the front wing and started pricing that gain at each element's own
trailing-edge clearance (`WingStation.min_clearance_m`): the main plane, at 30 mm,
keeps almost exactly the augmentation it had, while the flap, stacked 77 mm up,
loses part of it. Total rear-wing downforce is untouched because the model leaves
that wing in the freestream.
"""

from __future__ import annotations

import math

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import DRS_FLAP_ANGLE_DEG, effective_rear_flap, simulate
from backend.simulation.panel_method import simulate_panel

GOLDEN_TARGETS = {
    "default": {
        "config": FormulaAeroConfig(),
        "expected": {
            "drag_N": 2395.7,
            "downforce_total_N": 1627.6,
            "downforce_front_N": 712.8,
            "downforce_rear_N": 914.8,
            "balance_front_pct": 43.8,
            "cop_x_m": 1.883,
            "cl_total": 0.574,
            "cd_total": 0.845,
            "efficiency": 0.68,
            "lap_time_delta_s": 0.2694,
        },
    },
    "low_speed_100kmh": {
        "config": FormulaAeroConfig(speed_kmh=100.0),
        "expected": {
            "drag_N": 598.9,
            "downforce_total_N": 406.9,
            "downforce_front_N": 178.2,
            "downforce_rear_N": 228.7,
            "balance_front_pct": 43.8,
            "cop_x_m": 1.883,
            "cl_total": 0.574,
            "cd_total": 0.845,
            "efficiency": 0.68,
            "lap_time_delta_s": 0.1873,
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
            "drag_N": 2827.7,
            "downforce_total_N": 4529.8,
            "downforce_front_N": 2643.8,
            "downforce_rear_N": 1886.0,
            "balance_front_pct": 58.4,
            "cop_x_m": 1.395,
            "cl_total": 1.597,
            "cd_total": 0.997,
            "efficiency": 1.60,
            "lap_time_delta_s": 0.0804,
        },
    },
    "monza_low_drag": {
        "config": FormulaAeroConfig(
            front_wing_main={"angle_deg": 0.0, "span_m": 1.8, "chord_m": 0.35, "gurney_mm": 0.0},
            front_wing_flap={"angle_deg": 4.0, "span_m": 1.8, "chord_m": 0.25, "gurney_mm": 0.0},
            rear_wing_main={"angle_deg": 2.0, "span_m": 1.0, "chord_m": 0.40, "gurney_mm": 0.0},
            rear_wing_flap={"angle_deg": 6.0, "span_m": 1.0, "chord_m": 0.30, "gurney_mm": 0.0},
            # drs_active folds this 6 degree flap to analytical.DRS_FLAP_ANGLE_DEG
            drs_active=True,
            ride_height_front_mm=30.0,
            ride_height_rear_mm=50.0,
        ),
        "expected": {
            "drag_N": 2343.5,
            "downforce_total_N": 856.9,
            "downforce_front_N": 438.6,
            "downforce_rear_N": 418.3,
            "balance_front_pct": 51.2,
            "cop_x_m": 1.635,
            "cl_total": 0.302,
            "cd_total": 0.826,
            "efficiency": 0.37,
            "lap_time_delta_s": 0.3258,
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


def test_drs_flag_reaches_the_model():
    """`drs_active` must survive validation and actually fold the rear flap.

    Pydantic drops unknown kwargs without complaining, so a flag the model does
    not define looks wired up in the UI while changing nothing at all.

    Total drag is asserted for the analytical tier only: the VLM's coupled
    Trefftz integral is non-monotone in rear flap angle (2383 N at 0 deg, 2368 N
    at 10 deg, 2507 N at 25 deg), because unloading the rear wing increases the
    front wing's induced drag. That is a property of the model, not of DRS.
    """
    cfg = FormulaAeroConfig(
        rear_wing_flap={"angle_deg": 6.0, "span_m": 1.0, "chord_m": 0.30},
        drs_active=True,
    )
    assert cfg.drs_active is True
    assert cfg.rear_wing_flap.angle_deg == 6.0
    assert effective_rear_flap(cfg).angle_deg == DRS_FLAP_ANGLE_DEG

    for name, run in (("analytical", simulate), ("panel", simulate_panel)):
        closed = run(cfg.model_copy(update={"drs_active": False}))
        opened = run(cfg)
        assert opened.forces.downforce_rear_N < closed.forces.downforce_rear_N, name
        assert opened.forces.downforce_total_N < closed.forces.downforce_total_N, name

    closed = simulate(cfg.model_copy(update={"drs_active": False}))
    assert simulate(cfg).forces.drag_N < closed.forces.drag_N


if __name__ == "__main__":
    test_golden_baseline()
    print("\nAll golden baseline tests PASSED.")
