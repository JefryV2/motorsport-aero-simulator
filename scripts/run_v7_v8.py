"""Run V7 + V8 against the current HF dataset with honest guards.

This is the entry point for validating a freshly regenerated dataset (see
scripts/generate_real_dataset.py). It:

1. loads ``data/datasets/aero_surrogate_v1.parquet`` and prints where its
   high-fidelity column actually came from (real OpenFOAM vs synthetic);
2. refuses to run when zero high-fidelity points converged - there is nothing
   to validate and fabricating fill-in values is not an option;
3. runs V7 (5-fold surrogate cross-validation) and V8 (multi-fidelity sample
   efficiency) and writes both scorecards to JSON next to the dataset.

Every historical target is printed by the validators as PASS/FAIL against the
measured number; failing targets are reported, never converted into passes.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_v7_v8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.dataset import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_dataset_parquet,
    summarize_provenance,
)
from validation.v7_surrogate_cv import run_v7_validation  # noqa: E402
from validation.v8_multifidelity_gain import run_v8_validation  # noqa: E402

SCORECARD_PATH = Path("data/datasets/v7_v8_scorecards.json")


def main() -> int:
    print(f"Loading {DEFAULT_DATASET_PATH}", flush=True)
    df = load_dataset_parquet(DEFAULT_DATASET_PATH)
    provenance = summarize_provenance(df)

    print("\n=== High-fidelity provenance (measured) ===")
    for key in sorted(provenance):
        print(f"{key}: {provenance[key]}")

    n_converged = int(provenance.get("n_converged", 0) or 0)
    n_failed = int(provenance.get("n_failed", 0) or 0)
    if provenance.get("all_synthetic"):
        print(
            "\nSTOP: the HF column is the calibrated synthetic model, not "
            "OpenFOAM output. V7/V8 on it would only describe surrogate "
            "machinery. Regenerate the dataset with a real solver first.",
            file=sys.stderr,
        )
        return 2
    if n_converged == 0:
        print(
            f"\nSTOP: 0 of {len(df)} HF points converged ({n_failed} failed). "
            "Cross-validation has no usable targets; reporting V7/V8 numbers "
            "now would be meaningless. Keep the failed rows, fix the solver "
            "envelope, regenerate.",
            file=sys.stderr,
        )
        return 3

    print(
        f"\n{n_converged}/{len(df)} HF points converged "
        f"({n_failed} failed, retained with NaN targets). "
        "Running V7 and V8...",
        flush=True,
    )

    v7 = run_v7_validation()
    v8 = run_v8_validation()

    scorecard = {
        "dataset_path": str(DEFAULT_DATASET_PATH),
        "provenance": provenance,
        "v7": v7,
        "v8": v8,
    }
    SCORECARD_PATH.write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
    )
    print(f"\nScorecards written to {SCORECARD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
