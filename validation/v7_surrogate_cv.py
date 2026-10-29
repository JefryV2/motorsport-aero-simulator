"""Validation Case V7: Aerodynamic Surrogate Cross-Validation (R^2, RMSE, MAPE).

Evaluates 5-fold cross-validation performance of a single-fidelity GP and a
Kennedy & O'Hagan (2000) AR1 multi-fidelity surrogate on the LHS dataset.

IMPORTANT: this case measures how well the surrogate reproduces the
"high-fidelity" column of the dataset. If that column came from the calibrated
synthetic fallback (no OpenFOAM), the result is a statement about surrogate
machinery, not aerodynamic accuracy. If the column came from real OpenFOAM
hybrid runs, the numbers still inherit the 2D-section simplifications.

The historical targets (R^2 > 0.90, MAPE < 5%) are printed as a scorecard, but
they are not forced. A small real-CFD dataset may not meet them; that measured
result is reported rather than hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import KFold

from backend.ml.dataset import FEATURE_COLUMNS, get_or_generate_dataset, summarize_provenance
from backend.ml.metrics import compute_metrics
from backend.ml.surrogate import AR1MultiFidelitySurrogate, SingleFidelityGP


def run_v7_validation() -> dict[str, float | str | bool]:
    """Run Case V7: 5-Fold Cross Validation of Single-Fidelity and Multi-Fidelity Surrogates."""
    print("=== Case V7: Aerodynamic Surrogate Cross-Validation ===")

    dataset_path = Path("data/datasets/aero_surrogate_v1.parquet")
    print(f"\nLoading dataset from {dataset_path}...")
    df = get_or_generate_dataset(file_path=dataset_path, n_samples=24, seed=42)
    provenance = summarize_provenance(df)
    print(f"Dataset loaded: {len(df)} rows across 5 design dimensions.")
    print(f"HF provenance: {provenance['solver_counts']} | synthetic={provenance['all_synthetic']}")
    print(f"Usable HF rows: {provenance['n_used']} | failed points: {provenance['n_failed']}")
    if provenance.get("warning"):
        print("\n" + "!" * 70)
        print(provenance["warning"])
        print("!" * 70 + "\n")

    # Failed points stay in the audit file but cannot train a surrogate.
    if "hf_failed" in df.columns:
        df = df[~df["hf_failed"].fillna(False).astype(bool)]
    df = df.dropna(subset=["cl_hf", "cd_hf", "cl_lf", "cd_lf"]).reset_index(drop=True)

    if len(df) < 10:
        raise RuntimeError(
            f"V7 needs at least 10 successful HF rows for 5-fold CV; only {len(df)} are available."
        )

    n_converged = int(df["convergence_flag"].fillna(False).astype(bool).sum()) if "convergence_flag" in df else 0
    print(f"Successful rows used: {len(df)} | converged: {n_converged}")

    X = df[FEATURE_COLUMNS].to_numpy()
    cl_lf = df["cl_lf"].to_numpy()
    cd_lf = df["cd_lf"].to_numpy()
    cl_hf = df["cl_hf"].to_numpy()
    cd_hf = df["cd_hf"].to_numpy()

    n_splits = min(5, len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cl_sf_preds = np.zeros(len(df))
    cd_sf_preds = np.zeros(len(df))
    cl_mf_preds = np.zeros(len(df))
    cd_mf_preds = np.zeros(len(df))
    cl_mf_stds = np.zeros(len(df))
    cd_mf_stds = np.zeros(len(df))

    print(f"\nExecuting {n_splits}-Fold Cross Validation...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        cl_lf_tr, cl_lf_te = cl_lf[train_idx], cl_lf[test_idx]
        cd_lf_tr, cd_lf_te = cd_lf[train_idx], cd_lf[test_idx]
        cl_hf_tr = cl_hf[train_idx]
        cd_hf_tr = cd_hf[train_idx]

        gp_cl = SingleFidelityGP(n_restarts_optimizer=3, random_state=42 + fold)
        gp_cd = SingleFidelityGP(n_restarts_optimizer=3, random_state=42 + fold)
        gp_cl.fit(X_tr, cl_hf_tr)
        gp_cd.fit(X_tr, cd_hf_tr)

        mf_cl = AR1MultiFidelitySurrogate(n_restarts_optimizer=3, random_state=42 + fold)
        mf_cd = AR1MultiFidelitySurrogate(n_restarts_optimizer=3, random_state=42 + fold)
        mf_cl.fit(X_tr, cl_lf_tr, cl_hf_tr)
        mf_cd.fit(X_tr, cd_lf_tr, cd_hf_tr)

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

        print(f"  Fold {fold + 1}/{n_splits} completed.")

    sf_cl_metrics = compute_metrics(cl_hf, cl_sf_preds)
    sf_cd_metrics = compute_metrics(cd_hf, cd_sf_preds)
    mf_cl_metrics = compute_metrics(cl_hf, cl_mf_preds, cl_mf_stds)
    mf_cd_metrics = compute_metrics(cd_hf, cd_mf_preds, cd_mf_stds)

    print("\n" + "=" * 70)
    print(f"{n_splits}-FOLD CROSS-VALIDATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<22} | {'Target':<6} | {'R²':>8} | {'RMSE':>8} | {'MAPE %':>8} | {'Cov 95%':>8}")
    print("-" * 70)
    print(f"{'Single-Fidelity GP':<22} | {'Cl':<6} | {sf_cl_metrics.r2:>8.4f} | {sf_cl_metrics.rmse:>8.4f} | {sf_cl_metrics.mape:>7.2f}% | {'N/A':>8}")
    print(f"{'Single-Fidelity GP':<22} | {'Cd':<6} | {sf_cd_metrics.r2:>8.4f} | {sf_cd_metrics.rmse:>8.4f} | {sf_cd_metrics.mape:>7.2f}% | {'N/A':>8}")
    print("-" * 70)
    print(f"{'AR1 Multi-Fidelity':<22} | {'Cl':<6} | {mf_cl_metrics.r2:>8.4f} | {mf_cl_metrics.rmse:>8.4f} | {mf_cl_metrics.mape:>7.2f}% | {mf_cl_metrics.coverage_95 * 100.0:>7.1f}%")
    print(f"{'AR1 Multi-Fidelity':<22} | {'Cd':<6} | {mf_cd_metrics.r2:>8.4f} | {mf_cd_metrics.rmse:>8.4f} | {mf_cd_metrics.mape:>7.2f}% | {mf_cd_metrics.coverage_95 * 100.0:>7.1f}%")
    print("=" * 70)

    targets_met = (
        mf_cl_metrics.r2 > 0.90
        and mf_cd_metrics.r2 > 0.90
        and mf_cl_metrics.mape < 5.0
        and mf_cd_metrics.mape < 5.0
    )
    print("\n--- Historical scorecard (not asserted) ---")
    print(f"1. AR1 Cl R² = {mf_cl_metrics.r2:.4f} (> 0.90) -> {'PASS' if mf_cl_metrics.r2 > 0.90 else 'FAIL'}")
    print(f"2. AR1 Cd R² = {mf_cd_metrics.r2:.4f} (> 0.90) -> {'PASS' if mf_cd_metrics.r2 > 0.90 else 'FAIL'}")
    print(f"3. AR1 Cl MAPE = {mf_cl_metrics.mape:.2f}% (< 5.0%) -> {'PASS' if mf_cl_metrics.mape < 5.0 else 'FAIL'}")
    print(f"4. AR1 Cd MAPE = {mf_cd_metrics.mape:.2f}% (< 5.0%) -> {'PASS' if mf_cd_metrics.mape < 5.0 else 'FAIL'}")

    if targets_met:
        print("\n>>> Case V7: all historical surrogate targets met on this dataset.\n")
    else:
        print(
            "\n>>> Case V7: targets NOT all met. This is the measured result for the "
            "current HF dataset; it is reported as a limitation, not converted into a pass.\n"
        )

    hf_source = "synthetic_model" if provenance["all_synthetic"] else "cfd"
    return {
        "mf_cl_r2": mf_cl_metrics.r2,
        "mf_cd_r2": mf_cd_metrics.r2,
        "mf_cl_mape": mf_cl_metrics.mape,
        "mf_cd_mape": mf_cd_metrics.mape,
        "n_used": float(len(df)),
        "n_converged": float(n_converged),
        "targets_met": 1.0 if targets_met else 0.0,
        "hf_source": hf_source,
        "all_synthetic": bool(provenance["all_synthetic"]),
    }


if __name__ == "__main__":
    run_v7_validation()
