"""Validation Case V4: VLM ground-effect trend.

Two distinct checks are reported here:

* **Pass/fail:** the potential-flow image-system augmentation (Tuck 1975,
  Wieselsberger). This is the physics the VLM is built to reproduce, so it is
  the only check that sets ``passed``.
* **PENDING:** the Zerihan & Zhang (2000) experimental ratio. The model uses a
  different wing geometry and produces a much larger augmentation, so this
  comparison is printed for context but does not count as a validation pass.
  Replace it with digitized external data before claiming V4 externally valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.simulation.vlm.core import VLMSurface, solve_vlm

# Potential-flow image-system trend for a 3D inverted wing in ground effect.
# Augmentation ratio: DF(h) / DF(freestream). This is a self-consistency target.
POTENTIAL_FLOW_BENCHMARK = {
    1.00: 1.120,
    0.50: 1.280,
    0.30: 1.560,
    0.20: 2.000,
    0.15: 2.550,
}

# Indicative published experimental trend, NOT digitized data. Kept only to
# show the size of the gap; external validation remains pending.
ZERIHAN_ZHANG_2000_INDICATIVE = {
    1.00: 1.00,
    0.50: 1.08,
    0.30: 1.18,
    0.20: 1.28,
    0.15: 1.34,
}
EXTERNAL_REFERENCE_PENDING = True


def run_v4_validation() -> dict:
    """Run Case V4: Inverted front wing ground effect sweep."""
    span = 1.80  # Formula car front wing span
    chord = 0.35  # Reference chord
    angle_deg = 4.0
    speed_kmh = 180.0

    print("=== Case V4: Ground-Effect Trend vs Potential-Flow Image System ===")
    print("External Zerihan & Zhang (2000) comparison is PENDING (indicative only).\n")

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
    clearances = sorted(POTENTIAL_FLOW_BENCHMARK.keys(), reverse=True)

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
        ratio_target = POTENTIAL_FLOW_BENCHMARK[h_over_c]
        external_ratio = ZERIHAN_ZHANG_2000_INDICATIVE[h_over_c]

        error_pct = abs(ratio_actual - ratio_target) / ratio_target * 100.0
        external_error_pct = abs(ratio_actual - external_ratio) / external_ratio * 100.0
        passed = error_pct < 6.0

        # Check monotonic ground effect augmentation
        monotonic = ratio_actual >= prev_ratio
        if not monotonic or not passed:
            all_passed = False

        prev_ratio = ratio_actual

        status = "PASS" if passed and monotonic else "FAIL"
        print(
            f"h/c = {h_over_c:4.2f} (h={h * 1000:5.1f}mm) | DF: {df_h:6.1f}N | "
            f"Augmentation: {ratio_actual:5.3f}x "
            f"(Potential-flow target: {ratio_target:5.3f}x, Err: {error_pct:4.2f}%) -> {status}"
        )
        print(
            f"    [PENDING] Zerihan & Zhang indicative: {external_ratio:5.3f}x, "
            f"gap {external_error_pct:5.1f}% -- not a validation pass"
        )
        results_table.append(
            {
                "h_over_c": h_over_c,
                "height_mm": h * 1000.0,
                "downforce_N": df_h,
                "ratio_actual": ratio_actual,
                "ratio_target": ratio_target,
                "error_pct": error_pct,
                "external_reference_ratio": external_ratio,
                "external_error_pct": external_error_pct,
                "external_reference_pending": EXTERNAL_REFERENCE_PENDING,
                "passed": passed and monotonic,
            }
        )

    print(f"\nPotential-flow V4 Status: {'PASS' if all_passed else 'FAIL'}")
    print("External Zerihan & Zhang validation: PENDING (no digitized data supplied).")
    return {
        "case": "V4",
        "passed": all_passed,
        "external_reference_pending": EXTERNAL_REFERENCE_PENDING,
        "table": results_table,
    }


if __name__ == "__main__":
    res = run_v4_validation()
    assert res["passed"], f"V4 Validation failed: {res}"
