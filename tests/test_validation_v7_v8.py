"""Test suite running surrogate validation cases V7 and V8.

These tests verify that the validation scripts run, use the real-CFD dataset,
return finite metrics, and report the historical scorecard honestly. They do
not assert that the historical R²/MAPE targets are met: on the current 28-point
real dataset they are partly unmet, and forcing a pass would defeat the point
of the validation.
"""

from __future__ import annotations

import pytest

from validation.v7_surrogate_cv import run_v7_validation
from validation.v8_multifidelity_gain import run_v8_validation


@pytest.mark.slow
def test_v7_surrogate_cross_validation():
    """V7 must report real-CFD CV metrics and a consistent scorecard."""
    results = run_v7_validation()
    cl_r2 = float(results["mf_cl_r2"])
    cd_r2 = float(results["mf_cd_r2"])
    cl_mape = float(results["mf_cl_mape"])
    cd_mape = float(results["mf_cd_mape"])

    assert 0.0 <= cl_r2 <= 1.0
    assert 0.0 <= cd_r2 <= 1.0
    assert cl_mape >= 0.0
    assert cd_mape >= 0.0
    assert float(results["n_used"]) >= 10.0
    assert results["hf_source"] == "cfd"
    assert results["all_synthetic"] is False

    scorecard = cl_r2 > 0.90 and cd_r2 > 0.90 and cl_mape < 5.0 and cd_mape < 5.0
    assert bool(results["targets_met"]) is scorecard


@pytest.mark.slow
def test_v8_multifidelity_gain():
    """V8 must report real-CFD sample efficiency and calibration honestly."""
    results = run_v8_validation()
    final_r2 = float(results["final_r2"])
    coverage = float(results["coverage_95"])
    reduction = float(results["rmse_reduction_small"])

    assert 0.0 <= final_r2 <= 1.0
    assert 0.0 <= coverage <= 1.0
    assert float(results["n_used"]) >= 12.0
    assert float(results["n_train"]) > 0.0
    assert results["hf_source"] == "cfd"
    assert results["all_synthetic"] is False

    scorecard = reduction > 10.0 and 0.85 <= coverage <= 1.0 and final_r2 > 0.90
    assert bool(results["targets_met"]) is scorecard
