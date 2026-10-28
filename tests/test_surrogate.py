"""Unit and property tests for Phase 3 ML surrogate pipeline and dataset DoE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.dataset import (
    DESIGN_SPACE,
    evaluate_design_point,
    load_dataset_parquet,
    sample_design_space,
    sample_vector_to_config,
    save_dataset_parquet,
)
from backend.ml.metrics import compute_metrics, compute_reliability_curve
from backend.ml.surrogate import (
    AeroSurrogatePipeline,
    AR1MultiFidelitySurrogate,
    SingleFidelityGP,
)


def test_sampling_bounds_and_distribution() -> None:
    """Verify that Latin Hypercube samples adhere strictly to parameter bounds."""
    n_samples = 50
    samples = sample_design_space(n_samples=n_samples, seed=123)

    assert samples.shape == (n_samples, len(DESIGN_SPACE))

    for col_idx, bound in enumerate(DESIGN_SPACE):
        col_vals = samples[:, col_idx]
        assert np.all(col_vals >= bound.min_val), f"{bound.name} has values below min {bound.min_val}"
        assert np.all(col_vals <= bound.max_val), f"{bound.name} has values above max {bound.max_val}"
        # Ensure reasonable variance / coverage
        assert np.std(col_vals) > 0.1 * (bound.max_val - bound.min_val)


def test_sample_vector_to_config_mapping() -> None:
    """Verify conversion of vector samples to typed FormulaAeroConfig."""
    sample = {
        "fw_main_angle": 4.5,
        "fw_flap_angle": 12.0,
        "ride_height_front_mm": 22.0,
        "rear_wing_main_angle": 15.0,
        "speed_kmh": 240.0,
    }
    cfg = sample_vector_to_config(sample)
    assert cfg.front_wing_main.angle_deg == 4.5
    assert cfg.front_wing_flap.angle_deg == 12.0
    assert cfg.ride_height_front_mm == 22.0
    assert cfg.speed_kmh == 240.0


def test_evaluate_design_point() -> None:
    """Verify evaluate_design_point generates complete low and high fidelity schema."""
    sample = {
        "fw_main_angle": 3.0,
        "fw_flap_angle": 8.0,
        "ride_height_front_mm": 25.0,
        "rear_wing_main_angle": 10.0,
        "speed_kmh": 200.0,
    }
    rec = evaluate_design_point(sample)

    assert "config_hash" in rec
    assert rec["convergence_flag"] is True
    assert rec["cl_lf"] > 0.0
    assert rec["cl_hf"] > 0.0
    assert rec["cd_lf"] > 0.0
    assert rec["cd_hf"] > 0.0


def test_parquet_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify saving and reloading dataset from Parquet maintains float precision."""
    df_orig = pd.DataFrame({
        "fw_main_angle": [1.0, 2.5, 5.0],
        "speed_kmh": [120.0, 220.0, 300.0],
        "cl_hf": [1.234567, 2.345678, 3.456789],
        "split": ["train", "val", "test"],
    })
    target = tmp_path / "test_aero.parquet"
    save_dataset_parquet(df_orig, target)

    assert target.exists()
    df_loaded = load_dataset_parquet(target)

    pd.testing.assert_frame_equal(df_orig, df_loaded)


def test_single_fidelity_gp_accuracy_and_uncertainty() -> None:
    """Verify SingleFidelityGP achieves high precision on smooth synthetic function."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 10.0, size=(40, 2))
    # Target function y = sin(x0) + 0.5 * x1
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

    gp = SingleFidelityGP(n_restarts_optimizer=2, random_state=42)
    gp.fit(X, y)

    # Predictions at training points should have near-zero error
    # (std includes WhiteKernel noise floor ~0.002–0.1 scale, so threshold is 0.15)
    y_pred, y_std = gp.predict(X)
    assert np.all(np.abs(y - y_pred) < 0.05)
    assert np.all(y_std < 0.15)

    # Uncertainty should increase when querying far from training data
    X_far = np.array([[25.0, 25.0]])
    _, y_std_far = gp.predict(X_far)
    assert float(y_std_far[0]) > 0.5


def test_ar1_multifidelity_surrogate() -> None:
    """Verify AR1 Kennedy & O'Hagan formulation on synthetic multi-fidelity pair."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 5.0, size=(30, 2))

    # Low fidelity: y_L = x0 + x1
    y_lf = X[:, 0] + X[:, 1]
    # High fidelity: y_H = 1.5 * y_L + 2.0 + 0.1 * sin(x0)
    y_hf = 1.5 * y_lf + 2.0 + 0.1 * np.sin(X[:, 0])

    mf = AR1MultiFidelitySurrogate(n_restarts_optimizer=2, random_state=42)
    mf.fit(X, y_lf, y_hf)

    # polyfit on 30 noisy samples has ~2% variance; allow 5% tolerance
    assert abs(mf.rho - 1.5) < 0.08, f"rho={mf.rho} not close to 1.5"
    assert abs(mf.c - 2.0) < 0.15, f"c={mf.c} not close to 2.0"

    X_test = np.array([[1.5, 2.5], [3.0, 1.0]])
    y_lf_test = X_test[:, 0] + X_test[:, 1]
    y_hf_test = 1.5 * y_lf_test + 2.0 + 0.1 * np.sin(X_test[:, 0])

    pred, std = mf.predict(X_test, y_lf_test)
    assert np.all(np.abs(y_hf_test - pred) < 0.05)
    assert np.all(std > 0.0)


def test_metrics_and_calibration() -> None:
    """Verify compute_metrics and reliability curve statistics."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.05, 1.95, 3.02, 3.98, 5.01])
    y_std = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    m = compute_metrics(y_true, y_pred, y_std)
    assert m.r2 > 0.99
    assert m.rmse < 0.06
    assert m.mape < 5.0
    assert m.coverage_95 == 1.0

    rel = compute_reliability_curve(y_true, y_pred, y_std)
    assert len(rel["nominal"]) == len(rel["empirical"])
    assert rel["empirical"][-1] == 1.0


def test_aero_pipeline_save_and_load(tmp_path: Path) -> None:
    """Verify end-to-end pipeline serialization and point prediction."""
    # Build minimal synthetic dataset
    data = {
        "fw_main_angle": [1.0, 3.0, 5.0, 7.0],
        "fw_flap_angle": [6.0, 10.0, 14.0, 18.0],
        "ride_height_front_mm": [20.0, 30.0, 40.0, 50.0],
        "rear_wing_main_angle": [8.0, 12.0, 16.0, 20.0],
        "speed_kmh": [120.0, 180.0, 240.0, 300.0],
        "cl_lf": [1.5, 2.0, 2.5, 3.0],
        "cd_lf": [0.4, 0.6, 0.8, 1.0],
        "cl_hf": [1.7, 2.2, 2.7, 3.2],
        "cd_hf": [0.45, 0.65, 0.85, 1.05],
    }
    df = pd.DataFrame(data)

    pipeline = AeroSurrogatePipeline(random_state=42)
    pipeline.fit_from_dataframe(df)

    save_path = tmp_path / "surrogate_pipeline.pkl"
    pipeline.save(save_path)
    assert save_path.exists()

    loaded = AeroSurrogatePipeline.load(save_path)
    assert loaded.is_fitted is True

    pred = loaded.predict_point({
        "fw_main_angle": 3.0,
        "fw_flap_angle": 10.0,
        "ride_height_front_mm": 30.0,
        "rear_wing_main_angle": 12.0,
        "speed_kmh": 200.0,
    })
    assert pred.cl_mean > 0.0
    assert pred.cd_mean > 0.0
    assert pred.downforce_N > 0.0
