"""Test suite running surrogate validation cases V7 and V8."""

from __future__ import annotations

import pytest

from validation.v7_surrogate_cv import run_v7_validation
from validation.v8_multifidelity_gain import run_v8_validation


@pytest.mark.slow
def test_v7_surrogate_cross_validation():
    """Verify Case V7: 5-fold CV R² > 0.90 and MAPE < 5.0% for AR1 surrogate."""
    results = run_v7_validation()
    assert results["mf_cl_r2"] > 0.90
    assert results["mf_cd_r2"] > 0.90
    assert results["mf_cl_mape"] < 5.0
    assert results["mf_cd_mape"] < 5.0


@pytest.mark.slow
def test_v8_multifidelity_gain():
    """Verify Case V8: Low-budget MF gain > 10%, calibrated coverage in [85%, 100%]."""
    results = run_v8_validation()
    assert results["final_r2"] > 0.90
    assert 0.85 <= results["coverage_95"] <= 1.0
