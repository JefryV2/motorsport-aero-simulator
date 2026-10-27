"""Validation Case V5: CFD Mesh Independence Study on 3 Levels (ASME GCI).

Reference:
- Roache, P. J. (1998). "Verification and Validation in Computational Science and Engineering".
- Celik, I. B., et al. (2008). "Procedure for estimation and reporting of uncertainty
  due to discretization in CFD applications". ASME J. Fluids Eng., 130(7), 078001.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.cfd.grid_convergence import GCIReport, compute_gci
from backend.simulation.cfd.models import AirfoilGeometry, MeshLevel
from backend.simulation.cfd.runner import run_cfd_case


def run_v5_validation() -> GCIReport:
    """Run Case V5: 3-level grid refinement study on NACA 0012 at alpha = 4.0 deg."""
    print("=== Case V5: CFD Mesh Independence Study (Celik / Roache GCI) ===")

    airfoil = AirfoilGeometry(
        name="NACA0012",
        chord_m=1.0,
        angle_of_attack_deg=4.0,
    )
    speed_ms = 45.0  # ~162 km/h

    # 1. Execute across 3 systematically refined levels
    print("Running Level 3 (Coarse)...")
    res_coarse = run_cfd_case(airfoil, mesh_level=MeshLevel.COARSE, speed_ms=speed_ms)
    print(f"  Coarse: cells={res_coarse.mesh_stats.num_cells}, Cl={res_coarse.cl:.5f}")

    print("Running Level 2 (Medium)...")
    res_medium = run_cfd_case(airfoil, mesh_level=MeshLevel.MEDIUM, speed_ms=speed_ms)
    print(f"  Medium: cells={res_medium.mesh_stats.num_cells}, Cl={res_medium.cl:.5f}")

    print("Running Level 1 (Fine)...")
    res_fine = run_cfd_case(airfoil, mesh_level=MeshLevel.FINE, speed_ms=speed_ms)
    print(f"  Fine:   cells={res_fine.mesh_stats.num_cells}, Cl={res_fine.cl:.5f}")

    # 2. Compute GCI metrics
    report = compute_gci(
        phi_fine=res_fine.cl,
        phi_medium=res_medium.cl,
        phi_coarse=res_coarse.cl,
        h_fine=res_fine.mesh_stats.grid_spacing_h,
        h_medium=res_medium.mesh_stats.grid_spacing_h,
        h_coarse=res_coarse.mesh_stats.grid_spacing_h,
        safety_factor=1.25,
    )

    print("\n--- ASME GCI Summary ---")
    print(f"Refinement ratios: r21 = {report.r21:.3f}, r32 = {report.r32:.3f}")
    print(f"Apparent order p:  {report.apparent_order_p:.3f}")
    print(f"Extrapolated Cl:   {report.phi_extrapolated:.5f}")
    print(f"Relative error:    {report.relative_error_fine_pct:.3f}%")
    print(f"GCI (Fine mesh):   {report.gci_fine_pct:.3f}% (Target: < 3.0%)")
    print(f"GCI (Medium mesh): {report.gci_medium_pct:.3f}%")
    print(f"Asymptotic ratio:  {report.asymptotic_range_ratio:.3f} (Target: 0.85 - 1.15)")
    print(f"In asymptotic range: {report.is_in_asymptotic_range}")

    assert report.gci_fine_pct < 3.0, f"GCI fine {report.gci_fine_pct:.2f}% exceeds 3.0% target"
    assert report.is_in_asymptotic_range, (
        f"Asymptotic ratio {report.asymptotic_range_ratio:.2f} out of range"
    )
    print("\n>>> Case V5: PASSED\n")

    return report


if __name__ == "__main__":
    run_v5_validation()
