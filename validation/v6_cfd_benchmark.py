"""Validation Case V6: CFD Aerodynamic Benchmark vs Literature.

References:
- McCroskey, W. J. (1987). "A critical assessment of wind tunnel results for the NACA 0012 airfoil". AGARD-CP-429.
- Ladson, C. L. (1988). "Effects of independent variation of Mach and Reynolds numbers
  on the low-speed aerodynamic characteristics of the NACA 0012 airfoil section". NASA TM-4074.
- Zerihan, J., & Zhang, X. (2000). "Aerodynamics of a generic single-element Formula 1
  front wing in ground effect". AIAA J. Aircraft, 37(6): 1045-1052.

Part A is a real comparison against the NACA 0012 data.  It records the measured
errors; it does not claim the historical tight tolerances are met by a coarse,
fully turbulent RANS section.  Part B is trend-only: the modelled section is an
inverted NACA 4412, not Zerihan & Zhang's multi-element wing, so absolute ratios
must not be presented as a validation of their geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.cfd.models import AirfoilGeometry, ConvergenceStatus, MeshLevel
from backend.simulation.cfd.runner import run_cfd_case


def _require_solved(label: str, res) -> None:
    if res.convergence.status in {ConvergenceStatus.DIVERGED, ConvergenceStatus.FAILED}:
        raise RuntimeError(f"{label} did not produce a usable result: {res.convergence.status.value}")


def run_v6_validation() -> dict[str, float]:
    """Run Case V6: NACA 0012 polar & front-wing ground-effect trend."""
    print("=== Case V6: CFD Aerodynamic Benchmark vs Literature ===")

    # ---------------------------------------------------------
    # Part A: NACA 0012 polar vs Ladson (1988) / McCroskey (1987)
    # Re = 3.0e6, speed ≈ 45 m/s, chord = 1.0 m
    # ---------------------------------------------------------
    print("\n--- Part A: NACA 0012 Polar Benchmark (Re = 3.0M, medium mesh) ---")
    experimental_data = {
        0.0: {"cl": 0.000, "cd": 0.0082},
        4.0: {"cl": 0.440, "cd": 0.0090},
        8.0: {"cl": 0.865, "cd": 0.0118},
        10.0: {"cl": 1.060, "cd": 0.0145},
        12.0: {"cl": 1.230, "cd": 0.0185},
    }

    cfd_results: dict[float, dict[str, float]] = {}
    print(
        f"{'Alpha':>6} | {'Cl_CFD':>8} {'Cl_Exp':>8} {'Error%':>8} | "
        f"{'Cd_CFD':>8} {'Cd_Exp':>8} {'Delta_Cd':>9}"
    )
    print("-" * 68)

    for alpha, exp in experimental_data.items():
        geom = AirfoilGeometry(name="NACA0012", chord_m=1.0, angle_of_attack_deg=alpha)
        res = run_cfd_case(geom, mesh_level=MeshLevel.MEDIUM, speed_ms=45.0)
        _require_solved(f"NACA0012 alpha={alpha}", res)

        cl_error_pct = (
            abs(res.cl - exp["cl"]) / exp["cl"] * 100.0
            if abs(exp["cl"]) > 0.05
            else abs(res.cl) * 100.0
        )
        cd_delta = abs(res.cd - exp["cd"])

        cfd_results[alpha] = {"cl": res.cl, "cd": res.cd}
        print(
            f"{alpha:>6.1f}° | {res.cl:>8.4f} {exp['cl']:>8.4f} {cl_error_pct:>7.2f}% | "
            f"{res.cd:>8.4f} {exp['cd']:>8.4f} {cd_delta:>9.5f}   "
            f"({res.convergence.status.value})"
        )

    cd0_delta = abs(cfd_results[0.0]["cd"] - 0.0082)
    cfd_slope = (cfd_results[8.0]["cl"] - cfd_results[0.0]["cl"]) / 8.0
    exp_slope = (0.865 - 0.000) / 8.0
    slope_error_pct = abs(cfd_slope - exp_slope) / exp_slope * 100.0

    print(f"\nCd0 delta: {cd0_delta:.5f} (historical tight target: < 0.0015)")
    print(
        f"Lift-curve slope: {cfd_slope:.4f}/deg vs experimental {exp_slope:.4f}/deg "
        f"({slope_error_pct:.2f}% error; historical target: < 4%)"
    )

    # Sanity checks only.  The tight historical targets are reported above but
    # are not asserted: a fully turbulent coarse RANS section overpredicts Cd0
    # and this is a measured limitation, not a coding failure.
    assert cfd_slope > 0.0, "CFD lift slope is not positive"
    assert 0.005 < cfd_results[0.0]["cd"] < 0.05, "CFD Cd0 is outside a plausible RANS range"

    # ---------------------------------------------------------
    # Part B: Front Wing in Ground Effect vs Zerihan & Zhang (2000)
    # Inverted wing at alpha = 5.0 deg, coarse mesh, trend only
    # ---------------------------------------------------------
    print("\n--- Part B: Front Wing in Ground Effect (TREND ONLY, coarse mesh) ---")
    print(
        "NOTE: this is an inverted NACA 4412 section, not the Zerihan/Zhang "
        "multi-element wing; absolute ratios are not a geometry validation."
    )
    hc_benchmarks = {
        1.00: 1.000,
        0.50: 1.080,
        0.30: 1.180,
        0.20: 1.280,
        0.15: 1.340,
    }

    base_geom = AirfoilGeometry(
        name="FrontWingSection", chord_m=0.35, angle_of_attack_deg=5.0, ground_height_m=None
    )
    base_res = run_cfd_case(base_geom, mesh_level=MeshLevel.COARSE, speed_ms=30.0)
    _require_solved("ground-effect baseline", base_res)
    cl_inf = base_res.cl

    print(f"{'h/c':>6} | {'Cl_CFD':>8} {'Cl/Cl_inf':>10} {'Exp_Ratio':>10} {'Error%':>8}")
    print("-" * 50)

    ratios: dict[float, float] = {}
    for hc, exp_ratio in hc_benchmarks.items():
        geom_ge = AirfoilGeometry(
            name="FrontWingSection",
            chord_m=0.35,
            angle_of_attack_deg=5.0,
            ground_height_m=hc * 0.35,
        )
        res_ge = run_cfd_case(geom_ge, mesh_level=MeshLevel.COARSE, speed_ms=30.0)
        _require_solved(f"ground effect h/c={hc}", res_ge)
        sim_ratio = res_ge.cl / cl_inf if abs(cl_inf) > 1e-9 else 0.0
        ratios[hc] = sim_ratio
        err_pct = abs(sim_ratio - exp_ratio) / exp_ratio * 100.0
        print(
            f"{hc:>6.2f} | {res_ge.cl:>8.4f} {sim_ratio:>10.4f} {exp_ratio:>10.4f} "
            f"{err_pct:>7.2f}%   ({res_ge.convergence.status.value})"
        )

    trend_ok = ratios[0.15] > ratios[1.00]
    print(
        f"\nGround-effect trend: ratio(0.15)={ratios[0.15]:.3f} vs ratio(1.00)={ratios[1.00]:.3f} "
        f"-> {'increasing (matches the expected direction)' if trend_ok else 'NOT increasing'}"
    )
    assert trend_ok, "Ground-effect downforce did not increase as ride height decreased"

    print(
        "\n>>> Case V6: Part A measured (not a tight-tolerance pass); "
        "Part B trend-only, not a direct literature validation.\n"
    )

    return {
        "cd0_delta": cd0_delta,
        "slope_error_pct": slope_error_pct,
        "cfd_slope": cfd_slope,
        "ground_effect_ratio_015": ratios[0.15],
        "ground_effect_ratio_100": ratios[1.00],
        "ground_effect_trend": 1.0 if trend_ok else 0.0,
    }


if __name__ == "__main__":
    run_v6_validation()
