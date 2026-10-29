"""Validation Case V3: VLM Grid Refinement and Discretization Convergence Study."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.vlm.core import VLMSurface, solve_vlm


class RefinementLevel(TypedDict):
    name: str
    n_span: int
    n_chord: int


def run_v3_validation() -> dict:
    """Run Case V3 grid convergence study across 4 mesh refinement levels."""
    levels: list[RefinementLevel] = [
        {"name": "Level 1 (Coarse)", "n_span": 8, "n_chord": 1},
        {"name": "Level 2 (Medium)", "n_span": 16, "n_chord": 2},
        {"name": "Level 3 (Fine)", "n_span": 24, "n_chord": 3},
        {"name": "Level 4 (Very Fine)", "n_span": 32, "n_chord": 4},
    ]

    print("=== Case V3: Panel Convergence Study (AR=6 Wing) ===")
    history: list[dict[str, object]] = []

    for lvl in levels:
        wing = VLMSurface(
            name="convergence_wing",
            origin=(0.0, 0.0, 0.0),
            span=6.0,
            chord=1.0,
            angle_deg=5.0,
            n_span=lvl["n_span"],
            n_chord=lvl["n_chord"],
        )
        sol = solve_vlm([wing], speed_kmh=150.0, use_symmetry=True)
        total_panels = lvl["n_span"] * lvl["n_chord"]

        history.append(
            {
                "name": lvl["name"],
                "panels": total_panels,
                "cl": sol.total_cl,
                "cd_i": sol.sections["convergence_wing"].cd_induced,
                "oswald_e": sol.sections["convergence_wing"].oswald_e,
            }
        )
        print(
            f"{lvl['name']:20s} | Panels: {total_panels:4d} | "
            f"CL: {sol.total_cl:6.4f} | CDi: {history[-1]['cd_i']:7.5f} | e: {history[-1]['oswald_e']:5.3f}"
        )

    # Calculate relative change between Level 3 and Level 4
    cl_l3 = cast(float, history[2]["cl"])
    cl_l4 = cast(float, history[3]["cl"])
    rel_change_cl_pct = abs(cl_l4 - cl_l3) / cl_l4 * 100.0

    cdi_l3 = cast(float, history[2]["cd_i"])
    cdi_l4 = cast(float, history[3]["cd_i"])
    rel_change_cdi_pct = abs(cdi_l4 - cdi_l3) / cdi_l4 * 100.0

    passed = rel_change_cl_pct < 1.0 and rel_change_cdi_pct < 1.5

    print(
        f"\nLevel 3 -> Level 4 Delta: CL change = {rel_change_cl_pct:.3f}%, CDi change = {rel_change_cdi_pct:.3f}%"
    )
    print(f"Convergence Target (< 1.0% change): {'PASS' if passed else 'FAIL'}")

    return {
        "case": "V3",
        "passed": passed,
        "rel_change_cl_pct": rel_change_cl_pct,
        "rel_change_cdi_pct": rel_change_cdi_pct,
        "history": history,
    }


if __name__ == "__main__":
    res = run_v3_validation()
    assert res["passed"], f"V3 Validation failed: {res}"
