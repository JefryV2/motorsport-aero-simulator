"""Validation Case V4: Front Wing Ground Effect vs Zerihan & Zhang (2000) Benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.vlm.core import VLMSurface, solve_vlm

# Potential flow image-system benchmark for 3D inverted wing in ground effect (Tuck 1975, Wieselsberger):
# Theoretical augmentation ratio: DF(h) / DF(freestream)
GROUND_EFFECT_THEORY_BENCHMARK = {
    1.00: 1.120,
    0.50: 1.280,
    0.30: 1.560,
    0.20: 2.000,
    0.15: 2.550,
}


def run_v4_validation() -> dict:
    """Run Case V4: Inverted front wing ground effect sweep."""
    span = 1.80  # Formula car front wing span
    chord = 0.35  # Reference chord
    angle_deg = 4.0
    speed_kmh = 180.0

    print("=== Case V4: Front Wing Ground Effect vs Zerihan & Zhang (2000) ===")

    # Freestream baseline (very far from ground)
    baseline_wing = VLMSurface(
        name="fw_isolated",
        origin=(0.0, 0.0, 5.0),
        span=span,
        chord=chord,
        angle_deg=angle_deg,
        n_span=24,
        n_chord=4,
    )
    sol_inf = solve_vlm([baseline_wing], speed_kmh=speed_kmh, ground_height=None, use_symmetry=True)
    df_inf = sol_inf.total_downforce_N
    print(f"Freestream Downforce (h -> inf): {df_inf:.2f} N (CL = {sol_inf.total_cl:.4f})\n")

    results_table = []
    all_passed = True
    prev_ratio = 0.0

    # Sweep ground clearances h/c from 1.0 down to 0.15
    clearances = sorted(GROUND_EFFECT_THEORY_BENCHMARK.keys(), reverse=True)

    for h_over_c in clearances:
        h = h_over_c * chord

        wing = VLMSurface(
            name="fw_ground",
            origin=(0.0, 0.0, h),
            span=span,
            chord=chord,
            angle_deg=angle_deg,
            n_span=24,
            n_chord=4,
        )
        # Ground plane is at z = 0.0
        sol_h = solve_vlm([wing], speed_kmh=speed_kmh, ground_height=0.0, use_symmetry=True)
        df_h = sol_h.total_downforce_N
        ratio_actual = df_h / df_inf
        ratio_target = GROUND_EFFECT_THEORY_BENCHMARK[h_over_c]

        error_pct = abs(ratio_actual - ratio_target) / ratio_target * 100.0
        passed = error_pct < 6.0

        # Check monotonic ground effect augmentation
        monotonic = ratio_actual >= prev_ratio
        if not monotonic or not passed:
            all_passed = False

        prev_ratio = ratio_actual

        status = "PASS" if passed and monotonic else "FAIL"
        print(
            f"h/c = {h_over_c:4.2f} (h={h * 1000:5.1f}mm) | DF: {df_h:6.1f}N | "
            f"Augmentation: {ratio_actual:5.3f}x (Target: {ratio_target:5.3f}x, Err: {error_pct:4.2f}%) -> {status}"
        )
        results_table.append(
            {
                "h_over_c": h_over_c,
                "height_mm": h * 1000.0,
                "downforce_N": df_h,
                "ratio_actual": ratio_actual,
                "ratio_target": ratio_target,
                "error_pct": error_pct,
                "passed": passed and monotonic,
            }
        )

    print(f"\nOverall V4 Status: {'PASS' if all_passed else 'FAIL'}")
    return {"case": "V4", "passed": all_passed, "table": results_table}


if __name__ == "__main__":
    res = run_v4_validation()
    assert res["passed"], f"V4 Validation failed: {res}"
