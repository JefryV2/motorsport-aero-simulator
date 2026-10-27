"""Validation Case V1: Elliptic Planform vs Prandtl Lifting-Line Theory."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.vlm.forces import compute_vlm_forces
from backend.simulation.vlm.geometry import Panel
from backend.simulation.vlm.solver import solve_circulations


def run_v1_validation() -> dict:
    """Run Case V1: Elliptic wing planform validation."""
    span = 8.0
    c0 = 1.0  # Root chord
    # Elliptic area = pi * a * b / 4 = pi * span * c0 / 4
    area_half = (math.pi * span * c0) / 8.0
    ar = (span**2) / (2.0 * area_half)  # AR = 8 / (pi/4) ≈ 10.1859
    alpha_deg = 4.0
    alpha_rad = math.radians(alpha_deg)
    speed_ms = 50.0
    rho = 1.225

    # Discretize half-span with cosine distribution
    n_span = 24
    theta = np.linspace(0.0, np.pi / 2.0, n_span + 1)
    y_nodes = (span / 2.0) * (1.0 - np.cos(theta))

    panels: list[Panel] = []
    for j in range(n_span):
        y_left = y_nodes[j]
        y_right = y_nodes[j + 1]
        y_mid = 0.5 * (y_left + y_right)
        dy = y_right - y_left

        # Local chord from ellipse equation
        chord_mid = c0 * math.sqrt(max(0.0, 1.0 - (2.0 * y_mid / span) ** 2))
        chord_left = c0 * math.sqrt(max(0.0, 1.0 - (2.0 * y_left / span) ** 2))
        chord_right = c0 * math.sqrt(max(0.0, 1.0 - (2.0 * y_right / span) ** 2))

        # Panel coordinates
        p1 = np.array([0.0, y_left, 0.0])
        p2 = np.array([0.0, y_right, 0.0])
        p3 = np.array([chord_right, y_right, 0.0])
        p4 = np.array([chord_left, y_left, 0.0])

        bound_p1 = np.array([0.25 * chord_left, y_left, 0.0])
        bound_p2 = np.array([0.25 * chord_right, y_right, 0.0])
        cp = np.array([0.75 * chord_mid, y_mid, 0.0])

        # Normal vector with angle of attack alpha
        normal = np.array([-math.sin(alpha_rad), 0.0, math.cos(alpha_rad)])
        panel_area = 0.5 * (chord_left + chord_right) * dy

        panels.append(
            Panel(
                p1=p1,
                p2=p2,
                p3=p3,
                p4=p4,
                bound_p1=bound_p1,
                bound_p2=bound_p2,
                cp=cp,
                normal=normal,
                area=panel_area,
                span=dy,
                chord=chord_mid,
                surface_id="elliptic_wing",
            )
        )

    v_inf = np.array([speed_ms, 0.0, 0.0])
    gamma, _ = solve_circulations(panels, freestream_vel=v_inf, use_symmetry=True)

    forces = compute_vlm_forces(
        panels,
        gamma,
        speed_ms=speed_ms,
        rho=rho,
        use_symmetry=True,
        ref_area=2.0 * area_half,
    )

    result = forces["elliptic_wing"]

    # Theoretical values from Prandtl lifting-line
    cl_theory = (2.0 * math.pi * alpha_rad) / (1.0 + 2.0 / ar)
    cd_ind_theory = (cl_theory**2) / (math.pi * ar)
    e_theory = 1.000

    cl_actual = abs(result.cl)
    e_actual = result.oswald_e
    e_error_pct = abs(e_actual - e_theory) / e_theory * 100.0
    cl_error_pct = abs(cl_actual - cl_theory) / cl_theory * 100.0

    passed = e_error_pct <= 2.5 and cl_error_pct <= 3.0

    print("=== Case V1: Elliptic Planform Validation ===")
    print(f"Aspect Ratio: {ar:.2f}")
    print(f"Oswald Efficiency: {e_actual:.4f} (Theory: {e_theory:.4f}, Error: {e_error_pct:.2f}%)")
    print(
        f"Lift Coeff (CL):   {cl_actual:.4f} (Theory: {cl_theory:.4f}, Error: {cl_error_pct:.2f}%)"
    )
    print(f"Induced Drag (CDi):{result.cd_induced:.5f} (Theory: {cd_ind_theory:.5f})")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "case": "V1",
        "passed": passed,
        "oswald_e": e_actual,
        "e_error_pct": e_error_pct,
        "cl_error_pct": cl_error_pct,
    }


if __name__ == "__main__":
    res = run_v1_validation()
    assert res["passed"], f"V1 Validation failed: {res}"
