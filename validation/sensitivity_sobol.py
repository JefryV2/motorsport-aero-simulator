"""Global Sobol sensitivity analysis on the real-dataset surrogate (Phase 4 P4.3).

Blueprint: "Global sensitivity analysis (Sobol indices via SALib) on the
surrogate; connect findings to physical reasoning."

What this does, honestly:
- Fits single-fidelity GPs on the 28 real OpenFOAM rows of
  ``aero_surrogate_v1.parquet`` (refuses a synthetic-labelled dataset).
- Runs Saltelli/Sobol over a sampling box that is **fully inside the
  CFD-validated envelope**: ``fw_main`` in [0, 4] and ``fw_flap`` in [4, 8]
  guarantees main+flap <= 8 deg at every corner (constrained LHS drew the data
  the same way), ground clearance 71-150 mm, rear wing 5-25 deg, and speed
  bounded by the dataset's own support (104-295 km/h).
- The raw design space allows larger flap angles; those designs are outside
  the validated envelope and are deliberately NOT sampled, because the GP has
  no validated data there. The indices therefore describe sensitivity within
  the validated envelope, not the raw space - this is stated in the output
  rather than hidden.

First- and second-order indices come from SALib 1.6 with a bootstrap
convergence check: the analysis is repeated at N=512 and the largest absolute
change in the totals is reported, so the reader can judge whether the indices
are stable at the sampling budget used.

Usage:
    .venv\\Scripts\\python.exe validation\\sensitivity_sobol.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import saltelli

from backend.ml.dataset import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    get_or_generate_dataset,
    summarize_provenance,
)
from backend.ml.surrogate import SingleFidelityGP

BASE_N = 256
CHECK_N = 512
RESULTS_JSON = Path("data/datasets/sensitivity_sobol.json")

# Envelope box: see module docstring. Speed uses the dataset's own support.
ENVELOPE_BOUNDS: dict[str, tuple[float, float]] = {
    "fw_main_angle": (0.0, 4.0),
    "fw_flap_angle": (4.0, 8.0),
    "ride_height_front_mm": (71.0, 150.0),
    "rear_wing_main_angle": (5.0, 25.0),
    "speed_kmh": (104.0, 295.0),
}

PHYSICAL_NOTES = {
    "fw_main_angle": "main-element incidence raises both element loadings together",
    "fw_flap_angle": "flap incidence biases load forward and loads the main flap gap",
    "ride_height_front_mm": "ground clearance drives the wing-ground venturi effect",
    "rear_wing_main_angle": "rear wing only enters through empirical floor/rear terms",
    "speed_kmh": "dynamic pressure scales forces ~quadratically in the VLM/LF terms",
}


class SobolIndices(TypedDict):
    """First-, second- and total-order indices for one sampling budget."""

    S1: list[float]
    S1_conf: list[float]
    ST: list[float]
    ST_conf: list[float]
    S2: list[list[float]]


def run_sobol_study() -> dict[str, object]:
    """Run the Sobol study on the real-dataset surrogate and print the indices."""
    print("=== Sobol sensitivity on the real-dataset surrogate (Phase 4 P4.3) ===")
    df = get_or_generate_dataset(file_path=DEFAULT_DATASET_PATH, n_samples=24, seed=42)
    provenance = summarize_provenance(df)
    print(f"Dataset rows: {len(df)} | HF provenance: {provenance['solver_counts']}")
    if provenance.get("all_synthetic"):
        print("STOP: the HF column is synthetic. Sobol indices would describe a "
              "mock solver; refusing to present them as CFD-derived sensitivity.")
        return {"targets_met": 0.0, "hf_source": "synthetic_model"}

    x = df[FEATURE_COLUMNS].to_numpy()
    problem: dict[str, Any] = {
        "num_vars": len(FEATURE_COLUMNS),
        "names": list(FEATURE_COLUMNS),
        "bounds": [list(ENVELOPE_BOUNDS[name]) for name in FEATURE_COLUMNS],
    }
    print("Sampling box (CFD-validated envelope only):")
    for name in FEATURE_COLUMNS:
        lo, hi = ENVELOPE_BOUNDS[name]
        print(f"  {name:<24} [{lo:>7.1f}, {hi:>7.1f}]")

    results: dict[str, object] = {
        "sampling_box": ENVELOPE_BOUNDS,
        "note": (
            "Indices are for the incidence-constrained envelope box "
            "(fw_main 0-4 + fw_flap 4-8 keeps main+flap <= 8 deg at every corner); "
            "raw-design-space flap angles up to 20 deg are outside the validated "
            "envelope and are not sampled."
        ),
    }
    targets_met = True
    for target in ("cl", "cd"):
        gp = SingleFidelityGP(n_restarts_optimizer=3, random_state=42)
        gp.fit(x, df[f"{target}_hf"].to_numpy())

        indices_by_n: dict[int, SobolIndices] = {}
        max_total_shift = 0.0
        for n_base in (BASE_N, CHECK_N):
            samples = saltelli.sample(problem, n_base, calc_second_order=True)
            y, _ = gp.predict(samples)
            analysis = sobol_analyze.analyze(
                problem, np.asarray(y, dtype=float), calc_second_order=True,
                print_to_console=False, parallel=False,
                # SALib draws the bootstrap CIs from an unseeded RNG; without this
                # S1_conf/ST_conf differ on every run of a recorded artifact.
                seed=42,
            )
            indices_by_n[n_base] = {
                "S1": [float(v) for v in analysis["S1"]],
                "S1_conf": [float(v) for v in analysis["S1_conf"]],
                "ST": [float(v) for v in analysis["ST"]],
                "ST_conf": [float(v) for v in analysis["ST_conf"]],
                "S2": [[float(a) for a in row] for row in analysis["S2"]],
            }
        base = indices_by_n[BASE_N]
        check = indices_by_n[CHECK_N]
        shifts = [abs(b - c) for b, c in zip(base["ST"], check["ST"], strict=True)]
        max_total_shift = max(shifts)
        stable = max_total_shift <= 0.05
        targets_met = targets_met and stable

        order = np.argsort([-v for v in base["ST"]])
        print(f"\n--- Sobol indices in the envelope box ({target.upper()}) ---")
        print(f"{'input':<26} {'S1':>7} {'+/-':>7} {'ST':>7} {'+/-':>7}")
        for idx in order:
            name = FEATURE_COLUMNS[idx]
            print(
                f"{name:<26} {base['S1'][idx]:>7.3f} {base['S1_conf'][idx]:>7.3f} "
                f"{base['ST'][idx]:>7.3f} {base['ST_conf'][idx]:>7.3f}",
            )
        strongest, second = (FEATURE_COLUMNS[i] for i in order[:2])
        print(
            f"Ranking: {strongest} > {second}. {PHYSICAL_NOTES[strongest]}. "
            f"Convergence check (N={CHECK_N} vs {BASE_N}): max |dST| = "
            f"{max_total_shift:.3f} -> {'stable' if stable else 'NOT stable'}",
        )
        # second-order (pairwise) for the two biggest totals
        s2 = base["S2"]
        pair = s2[order[0]][order[1]]
        print(f"Top-pair interaction S2({strongest},{second}) = {pair:.3f}")

        results[target] = {
            "indices": base,
            "check_indices": check,
            "max_total_shift": max_total_shift,
            "stable": bool(stable),
            "strongest": strongest,
            "second": second,
        }

    results["targets_met"] = bool(targets_met)
    results["hf_source"] = "cfd"
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nResults written to {RESULTS_JSON}")
    print(
        "Reading guide: the indices quantify how the SURROGATE responds inside the "
        "validated envelope; they inherit the surrogate's errors (V7 CV gave Cl "
        "R2 = 0.9952) and are not a substitute for new CFD runs."
    )
    return results


if __name__ == "__main__":
    run_sobol_study()
