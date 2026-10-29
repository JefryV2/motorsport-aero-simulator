"""Validation Case V8: Multi-Fidelity Sample Efficiency and Uncertainty Calibration.

Measures whether the Kennedy & O'Hagan (2000) AR1 multi-fidelity model:
1. Reduces error relative to a single-fidelity GP under a constrained number of
   high-fidelity evaluations.
2. Produces calibrated probabilistic uncertainty bounds.

IMPORTANT: "high fidelity" here means the dataset's HF column. When that column
is the calibrated synthetic model, the result describes surrogate machinery,
not agreement with real CFD. When it is a real OpenFOAM hybrid run, it still
inherits the 2D-section simplifications.

The historical targets are printed as a scorecard. They are not asserted: a
small real-CFD dataset can fail them, and that measured outcome is the honest
result. The script only refuses to run when there is not enough usable data to
compute the metrics at all.

References:
- Kennedy, M. C., & O'Hagan, A. (2000). "Predicting the output from a complex computer code
  when inputs are uncertain". Biometrika, 87(1): 1–13.
- Forrester, A. I., Sobester, A., & Keane, A. J. (2007). "Multi-fidelity optimization via
  surrogate modelling". Proc. R. Soc. A, 463(2088): 3251–3269.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend.ml.dataset import FEATURE_COLUMNS, get_or_generate_dataset, summarize_provenance
from backend.ml.metrics import compute_metrics, compute_reliability_curve
from backend.ml.surrogate import AR1MultiFidelitySurrogate, SingleFidelityGP


def _usable_rows(df):
    if "hf_failed" in df.columns:
        df = df[~df["hf_failed"].fillna(False).astype(bool)]
    return df.dropna(subset=["cl_hf", "cd_hf", "cl_lf", "cd_lf"]).reset_index(drop=True)


def run_v8_validation() -> dict[str, float | str | bool]:
    """Run Case V8: Multi-fidelity gain & uncertainty calibration analysis."""
    print("=== Case V8: Multi-Fidelity Efficiency & Uncertainty Calibration ===")

    dataset_path = Path("data/datasets/aero_surrogate_v1.parquet")
    df = get_or_generate_dataset(file_path=dataset_path, n_samples=24, seed=42)
    provenance = summarize_provenance(df)
    print(f"Dataset rows: {len(df)}")
    print(f"HF provenance: {provenance['solver_counts']} | synthetic={provenance['all_synthetic']}")
    print(f"Usable HF rows: {provenance['n_used']} | failed points: {provenance['n_failed']}")
    if provenance.get("warning"):
        print("\n" + "!" * 70)
        print(provenance["warning"])
        print("!" * 70 + "\n")

    df = _usable_rows(df)
    if len(df) < 12:
        raise RuntimeError(
            f"V8 needs at least 12 successful HF rows to measure sample efficiency; only {len(df)} are available."
        )

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    if len(df_train) < 6 or len(df_test) < 3:
        # Failed points can hollow out a small split. Fall back to a deterministic
        # held-out split and say so; do not silently leak test rows into training.
        rng = np.random.default_rng(42)
        order = rng.permutation(len(df))
        n_test = max(3, int(round(len(df) * 0.20)))
        test_idx = set(order[:n_test].tolist())
        train_idx = [i for i in range(len(df)) if i not in test_idx]
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[sorted(test_idx)].reset_index(drop=True)
        print(
            f"NOTE: dataset split was too small after removing failed rows; using a "
            f"deterministic {len(df_train)}/{len(df_test)} train/test fallback."
        )

    X_train_full = df_train[FEATURE_COLUMNS].to_numpy()
    cl_lf_train_full = df_train["cl_lf"].to_numpy()
    cl_hf_train_full = df_train["cl_hf"].to_numpy()

    X_test = df_test[FEATURE_COLUMNS].to_numpy()
    cl_lf_test = df_test["cl_lf"].to_numpy()
    cl_hf_test = df_test["cl_hf"].to_numpy()

    n_train = len(df_train)
    print(f"Train rows: {n_train} | Test rows: {len(df_test)}")

    # -------------------------------------------------------------------------
    # Part A: Sample Efficiency under Low High-Fidelity Budget
    # -------------------------------------------------------------------------
    print("\n--- Part A: Sample Efficiency vs High-Fidelity Budget ---")
    budgets = sorted({
        max(4, n_train // 3),
        max(6, (2 * n_train) // 3),
        n_train,
    })
    print(f"{'HF Budget':>10} | {'SF GP RMSE':>12} | {'MF AR1 RMSE':>12} | {'RMSE Reduction':>15} | {'MF R²':>8}")
    print("-" * 68)

    rmse_reductions: list[float] = []

    for budget in budgets:
        X_sub = X_train_full[:budget]
        cl_lf_sub = cl_lf_train_full[:budget]
        cl_hf_sub = cl_hf_train_full[:budget]

        gp_sf = SingleFidelityGP(n_restarts_optimizer=3, random_state=42)
        gp_sf.fit(X_sub, cl_hf_sub)
        pred_sf, _ = gp_sf.predict(X_test)
        metrics_sf = compute_metrics(cl_hf_test, pred_sf)

        mf_ar1 = AR1MultiFidelitySurrogate(n_restarts_optimizer=3, random_state=42)
        mf_ar1.fit(X_sub, cl_lf_sub, cl_hf_sub)
        pred_mf, std_mf = mf_ar1.predict(X_test, cl_lf_test)
        metrics_mf = compute_metrics(cl_hf_test, pred_mf, std_mf)

        reduction = (
            (metrics_sf.rmse - metrics_mf.rmse) / metrics_sf.rmse * 100.0
            if metrics_sf.rmse > 0
            else 0.0
        )
        rmse_reductions.append(reduction)

        print(
            f"{budget:>10} | {metrics_sf.rmse:>12.4f} | {metrics_mf.rmse:>12.4f} | "
            f"{reduction:>14.1f}% | {metrics_mf.r2:>8.4f}"
        )

    # -------------------------------------------------------------------------
    # Part B: Uncertainty Calibration Analysis on Full Test Set
    # -------------------------------------------------------------------------
    print("\n--- Part B: Epistemic Uncertainty Calibration ---")
    mf_full = AR1MultiFidelitySurrogate(n_restarts_optimizer=5, random_state=42)
    mf_full.fit(X_train_full, cl_lf_train_full, cl_hf_train_full)
    cl_pred, cl_std = mf_full.predict(X_test, cl_lf_test)

    final_metrics = compute_metrics(cl_hf_test, cl_pred, cl_std)
    rel_curve = compute_reliability_curve(cl_hf_test, cl_pred, cl_std)

    print(f"95% Confidence Interval Coverage: {final_metrics.coverage_95 * 100.0:.1f}%")
    print(f"Mean 95% Interval Width:         {final_metrics.mean_ci_width_95:.4f}")

    print("\nReliability Diagram Data:")
    print(f"{'Nominal Coverage':>18} | {'Empirical Coverage':>20} | {'Calibration Gap':>18}")
    print("-" * 62)
    for nom, emp in zip(rel_curve["nominal"], rel_curve["empirical"], strict=True):
        gap = emp - nom
        print(f"{nom * 100.0:>17.1f}% | {emp * 100.0:>19.1f}% | {gap * 100.0:>+17.1f}%")

    # -------------------------------------------------------------------------
    # Measured scorecard (not forced)
    # -------------------------------------------------------------------------
    low_budget_reduction = rmse_reductions[0]
    full_reduction = rmse_reductions[-1]
    cov_pct = final_metrics.coverage_95 * 100.0
    targets_met = low_budget_reduction > 10.0 and 85.0 <= cov_pct <= 100.0 and final_metrics.r2 > 0.90

    print("\n--- Historical scorecard (not asserted) ---")
    print(
        f"1. Smallest-budget (N={budgets[0]}) RMSE reduction = {low_budget_reduction:.1f}% "
        f"(> 10.0%) -> {'PASS' if low_budget_reduction > 10.0 else 'FAIL'}"
    )
    print(
        f"2. 95% CI Coverage = {cov_pct:.1f}% in [85.0%, 100.0%] -> "
        f"{'PASS' if 85.0 <= cov_pct <= 100.0 else 'FAIL'}"
    )
    print(
        f"3. Multi-Fidelity Test R² = {final_metrics.r2:.4f} (> 0.90) -> "
        f"{'PASS' if final_metrics.r2 > 0.90 else 'FAIL'}"
    )

    if targets_met:
        print("\n>>> Case V8: all historical targets met on this dataset.\n")
    else:
        print(
            "\n>>> Case V8: targets NOT all met. This is the measured result for the "
            "current HF dataset; it is reported as a limitation, not converted into a pass.\n"
        )

    hf_source = "synthetic_model" if provenance["all_synthetic"] else "cfd"
    return {
        "final_r2": final_metrics.r2,
        "final_rmse": final_metrics.rmse,
        "coverage_95": final_metrics.coverage_95,
        "rmse_reduction_small": low_budget_reduction,
        "rmse_reduction_full": full_reduction,
        "n_used": float(len(df)),
        "n_train": float(n_train),
        "n_test": float(len(df_test)),
        "targets_met": 1.0 if targets_met else 0.0,
        "hf_source": hf_source,
        "all_synthetic": bool(provenance["all_synthetic"]),
    }


if __name__ == "__main__":
    run_v8_validation()
