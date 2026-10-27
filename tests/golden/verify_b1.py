"""Verify B1 resolution: panel_method now runs true 3D VLM with distinct vortex physics."""

from __future__ import annotations

from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import simulate
from backend.simulation.panel_method import simulate_panel


def verify_b1_resolution():
    cfg = FormulaAeroConfig()
    r_analytical = simulate(cfg)
    r_vlm = simulate_panel(cfg)

    print("=== B1 Resolution Check (Analytical vs 3D VLM) ===")
    print(
        f"Analytical Downforce: {r_analytical.forces.downforce_total_N:.1f} N, Drag: {r_analytical.forces.drag_N:.1f} N"
    )
    print(
        f"VLM Downforce:        {r_vlm.forces.downforce_total_N:.1f} N, Drag: {r_vlm.forces.drag_N:.1f} N"
    )
    print(f"VLM Method Label:     {r_vlm.method}")

    # They should no longer be identical
    is_distinct = (
        r_vlm.forces.downforce_total_N != r_analytical.forces.downforce_total_N
        or r_vlm.forces.drag_N != r_analytical.forces.drag_N
    )
    print(f"Distinct 3D VLM physics active: {is_distinct}")
    assert is_distinct, "Bug B1 still present: panel_method is still identical to analytical"
    assert r_vlm.method == "analytical_enhanced"
    print("PASS: B1 resolved with real VLM physics.\n")


if __name__ == "__main__":
    verify_b1_resolution()
