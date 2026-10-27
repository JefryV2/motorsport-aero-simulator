"""Validation Case V6: CFD Aerodynamic Benchmark vs Literature.

References:
- McCroskey, W. J. (1987). "A critical assessment of wind tunnel results for the NACA 0012 airfoil". AGARD-CP-429.
- Ladson, C. L. (1988). "Effects of independent variation of Mach and Reynolds numbers
  on the low-speed aerodynamic characteristics of the NACA 0012 airfoil section". NASA TM-4074.
- Zerihan, J., & Zhang, X. (2000). "Aerodynamics of a generic single-element Formula 1
  front wing in ground effect". AIAA J. Aircraft, 37(6): 1045–1052.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel
from backend.simulation.cfd.runner import run_cfd_case


def run_v6_validation() -> dict[str, float]:
    """Run Case V6: NACA 0012 polar & front-wing ground effect benchmarks."""
    print("=== Case V6: CFD Aerodynamic Benchmark vs Literature ===")

    # ---------------------------------------------------------
    # Part A: NACA 0012 polar vs Ladson (1988) / McCroskey (1987)
    # Re = 3.0e6, speed ≈ 45 m/s, chord = 1.0 m
    # ---------------------------------------------------------
    print("\n--- Part A: NACA 0012 Polar Benchmark (Re = 3.0M) ---")
    experimental_data = {
        0.0: {"cl": 0.000, "cd": 0.0082},
        4.0: {"cl": 0.440, "cd": 0.0090},
        8.0: {"cl": 0.865, "cd": 0.0118},
        10.0: {"cl": 1.060, "cd": 0.0145},
        12.0: {"cl": 1.230, "cd": 0.0185},
    }

    cfd_results: dict[float, dict[str, float]] = {}
    print(
        f"{'Alpha':>6} | {'Cl_CFD':>8} {'Cl_Exp':>8} {'Error%':>8} | {'Cd_CFD':>8} {'Cd_Exp':>8} {'Delta_Cd':>9}"
    )
    print("-" * 62)

    for alpha, exp in experimental_data.items():
        geom = AirfoilGeometry(name="NACA0012", chord_m=1.0, angle_of_attack_deg=alpha)
        res = run_cfd_case(geom, mesh_level=MeshLevel.FINE, speed_ms=45.0)

        cl_error_pct = (
            abs(res.cl - exp["cl"]) / exp["cl"] * 100.0
            if abs(exp["cl"]) > 0.05
            else abs(res.cl) * 100.0
        )
        cd_delta = abs(res.cd - exp["cd"])

        cfd_results[alpha] = {"cl": res.cl, "cd": res.cd}
        print(
            f"{alpha:>6.1f}° | {res.cl:>8.4f} {exp['cl']:>8.4f} {cl_error_pct:>7.2f}% | "
            f"{res.cd:>8.4f} {exp['cd']:>8.4f} {cd_delta:>9.5f}"
        )

    # Verification criteria Part A:
    # 1. Zero-lift drag within ±0.0015 of 0.0082
    cd0_delta = abs(cfd_results[0.0]["cd"] - 0.0082)
    assert cd0_delta < 0.0015, f"Cd0 error {cd0_delta:.5f} exceeds tolerance 0.0015"

    # 2. Linear lift slope dCl/dalpha error < 4.0%
    # Experimental slope (0 to 8 deg): (0.865 - 0.0) / 8 = 0.1081 / deg
    # CFD slope: (Cl(8) - Cl(0)) / 8
    cfd_slope = (cfd_results[8.0]["cl"] - cfd_results[0.0]["cl"]) / 8.0
    exp_slope = (0.865 - 0.000) / 8.0
    slope_error_pct = abs(cfd_slope - exp_slope) / exp_slope * 100.0
    print(f"\nLift-curve slope error: {slope_error_pct:.2f}% (Target: < 4.0%)")
    assert slope_error_pct < 4.0, f"Lift slope error {slope_error_pct:.2f}% exceeds 4.0%"

    # ---------------------------------------------------------
    # Part B: Front Wing in Ground Effect vs Zerihan & Zhang (2000)
    # Inverted wing at alpha = 5.0 deg
    # ---------------------------------------------------------
    print("\n--- Part B: Front Wing in Ground Effect vs Zerihan & Zhang (2000) ---")
    hc_benchmarks = {
        1.00: 1.000,
        0.50: 1.080,
        0.30: 1.180,
        0.20: 1.280,
        0.15: 1.340,
    }

    # Baseline isolated wing (far from ground)
    base_geom = AirfoilGeometry(
        name="FrontWingSection", chord_m=0.35, angle_of_attack_deg=5.0, ground_height_m=None
    )
    base_res = run_cfd_case(base_geom, mesh_level=MeshLevel.FINE, speed_ms=30.0)
    cl_inf = base_res.cl

    print(f"{'h/c':>6} | {'Cl_CFD':>8} {'Cl/Cl_inf':>10} {'Exp_Ratio':>10} {'Error%':>8}")
    print("-" * 50)

    for hc, exp_ratio in hc_benchmarks.items():
        geom_ge = AirfoilGeometry(
            name="FrontWingSection",
            chord_m=0.35,
            angle_of_attack_deg=5.0,
            ground_height_m=hc * 0.35,
        )
        res_ge = run_cfd_case(geom_ge, mesh_level=MeshLevel.FINE, speed_ms=30.0)
        sim_ratio = res_ge.cl / cl_inf
        err_pct = abs(sim_ratio - exp_ratio) / exp_ratio * 100.0
        print(
            f"{hc:>6.2f} | {res_ge.cl:>8.4f} {sim_ratio:>10.4f} {exp_ratio:>10.4f} {err_pct:>7.2f}%"
        )
        assert err_pct < 5.0, f"Ground effect ratio error {err_pct:.2f}% at h/c={hc} exceeds 5.0%"

    print("\n>>> Case V6: PASSED\n")

    return {
        "cd0_delta": cd0_delta,
        "slope_error_pct": slope_error_pct,
    }


if __name__ == "__main__":
    run_v6_validation()
