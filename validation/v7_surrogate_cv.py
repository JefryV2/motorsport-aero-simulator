"""Validation Case V7: Aerodynamic Surrogate Cross-Validation (R^2, RMSE, MAPE).

Evaluates 5-fold cross-validation performance of Single-Fidelity GP and
Kennedy & O'Hagan (2000) AR1 Multi-Fidelity surrogate on the LHS dataset.

Acceptance criteria:
- 5-fold Cross-Validation R^2 > 0.90 for both Cl and Cd
- 5-fold Cross-Validation MAPE < 5.0% for both Cl and Cd
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import KFold

from backend.ml.dataset import FEATURE_COLUMNS, get_or_generate_dataset
from backend.ml.metrics import compute_metrics
from backend.ml.surrogate import AR1MultiFidelitySurrogate, SingleFidelityGP


def run_v7_validation() -> dict[str, float]:
    """Run Case V7: 5-Fold Cross Validation of Single-Fidelity and Multi-Fidelity Surrogates."""
    print("=== Case V7: Aerodynamic Surrogate Cross-Validation ===")

    # 1. Load or generate the 100-sample LHS dataset
    dataset_path = Path("data/datasets/aero_surrogate_v1.parquet")
    print(f"\nLoading dataset from {dataset_path}...")
    df = get_or_generate_dataset(file_path=dataset_path, n_samples=100, seed=42)
    print(f"Dataset loaded: {len(df)} samples across 5 design dimensions.")

    X = df[FEATURE_COLUMNS].to_numpy()
    cl_lf = df["cl_lf"].to_numpy()
    cd_lf = df["cd_lf"].to_numpy()
    cl_hf = df["cl_hf"].to_numpy()
    cd_hf = df["cd_hf"].to_numpy()

    # 2. 5-Fold Cross-Validation Setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Accumulators for out-of-fold predictions
    cl_sf_preds = np.zeros(len(df))
    cd_sf_preds = np.zeros(len(df))
    cl_mf_preds = np.zeros(len(df))
    cd_mf_preds = np.zeros(len(df))
    cl_mf_stds = np.zeros(len(df))
    cd_mf_stds = np.zeros(len(df))

    print("\nExecuting 5-Fold Cross Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        cl_lf_tr, cl_lf_te = cl_lf[train_idx], cl_lf[test_idx]
        cd_lf_tr, cd_lf_te = cd_lf[train_idx], cd_lf[test_idx]
        cl_hf_tr = cl_hf[train_idx]
        cd_hf_tr = cd_hf[train_idx]

        # Fit models
        # Single-Fidelity GP
        gp_cl = SingleFidelityGP(n_restarts_optimizer=3, random_state=42 + fold)
        gp_cd = SingleFidelityGP(n_restarts_optimizer=3, random_state=42 + fold)
        gp_cl.fit(X_tr, cl_hf_tr)
        gp_cd.fit(X_tr, cd_hf_tr)

        # Multi-Fidelity AR1
        mf_cl = AR1MultiFidelitySurrogate(n_restarts_optimizer=3, random_state=42 + fold)
        mf_cd = AR1MultiFidelitySurrogate(n_restarts_optimizer=3, random_state=42 + fold)
        mf_cl.fit(X_tr, cl_lf_tr, cl_hf_tr)
        mf_cd.fit(X_tr, cd_lf_tr, cd_hf_tr)

        # Out-of-fold predictions
        cl_sf_p, _ = gp_cl.predict(X_te)
        cd_sf_p, _ = gp_cd.predict(X_te)
        cl_mf_p, cl_mf_s = mf_cl.predict(X_te, cl_lf_te)
        cd_mf_p, cd_mf_s = mf_cd.predict(X_te, cd_lf_te)

        cl_sf_preds[test_idx] = cl_sf_p
        cd_sf_preds[test_idx] = cd_sf_p
        cl_mf_preds[test_idx] = cl_mf_p
        cd_mf_preds[test_idx] = cd_mf_p
        cl_mf_stds[test_idx] = cl_mf_s
        cd_mf_stds[test_idx] = cd_mf_s

        print(f"  Fold {fold + 1}/5 completed.")

    # 3. Compute Metrics
    sf_cl_metrics = compute_metrics(cl_hf, cl_sf_preds)
    sf_cd_metrics = compute_metrics(cd_hf, cd_sf_preds)
    mf_cl_metrics = compute_metrics(cl_hf, cl_mf_preds, cl_mf_stds)
    mf_cd_metrics = compute_metrics(cd_hf, cd_mf_preds, cd_mf_stds)

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<22} | {'Target':<6} | {'R²':>8} | {'RMSE':>8} | {'MAPE %':>8} | {'Cov 95%':>8}")
    print("-" * 70)
    print(f"{'Single-Fidelity GP':<22} | {'Cl':<6} | {sf_cl_metrics.r2:>8.4f} | {sf_cl_metrics.rmse:>8.4f} | {sf_cl_metrics.mape:>7.2f}% | {'N/A':>8}")
    print(f"{'Single-Fidelity GP':<22} | {'Cd':<6} | {sf_cd_metrics.r2:>8.4f} | {sf_cd_metrics.rmse:>8.4f} | {sf_cd_metrics.mape:>7.2f}% | {'N/A':>8}")
    print("-" * 70)
    print(f"{'AR1 Multi-Fidelity':<22} | {'Cl':<6} | {mf_cl_metrics.r2:>8.4f} | {mf_cl_metrics.rmse:>8.4f} | {mf_cl_metrics.mape:>7.2f}% | {mf_cl_metrics.coverage_95 * 100.0:>7.1f}%")
    print(f"{'AR1 Multi-Fidelity':<22} | {'Cd':<6} | {mf_cd_metrics.r2:>8.4f} | {mf_cd_metrics.rmse:>8.4f} | {mf_cd_metrics.mape:>7.2f}% | {mf_cd_metrics.coverage_95 * 100.0:>7.1f}%")
    print("=" * 70)

    # 4. Verify Acceptance Criteria
    print("\n--- Verification of Acceptance Criteria ---")
    print(f"1. AR1 Cl R² = {mf_cl_metrics.r2:.4f} (> 0.90) -> {'PASS' if mf_cl_metrics.r2 > 0.90 else 'FAIL'}")
    print(f"2. AR1 Cd R² = {mf_cd_metrics.r2:.4f} (> 0.90) -> {'PASS' if mf_cd_metrics.r2 > 0.90 else 'FAIL'}")
    print(f"3. AR1 Cl MAPE = {mf_cl_metrics.mape:.2f}% (< 5.0%) -> {'PASS' if mf_cl_metrics.mape < 5.0 else 'FAIL'}")
    print(f"4. AR1 Cd MAPE = {mf_cd_metrics.mape:.2f}% (< 5.0%) -> {'PASS' if mf_cd_metrics.mape < 5.0 else 'FAIL'}")

    assert mf_cl_metrics.r2 > 0.90, f"Cl R² {mf_cl_metrics.r2:.4f} does not meet target > 0.90"
    assert mf_cd_metrics.r2 > 0.90, f"Cd R² {mf_cd_metrics.r2:.4f} does not meet target > 0.90"
    assert mf_cl_metrics.mape < 5.0, f"Cl MAPE {mf_cl_metrics.mape:.2f}% exceeds limit 5.0%"
    assert mf_cd_metrics.mape < 5.0, f"Cd MAPE {mf_cd_metrics.mape:.2f}% exceeds limit 5.0%"

    print("\nCase V7 Validation PASSED successfully!\n")
    return {
        "mf_cl_r2": mf_cl_metrics.r2,
        "mf_cd_r2": mf_cd_metrics.r2,
        "mf_cl_mape": mf_cl_metrics.mape,
        "mf_cd_mape": mf_cd_metrics.mape,
    }


if __name__ == "__main__":
    run_v7_validation()
