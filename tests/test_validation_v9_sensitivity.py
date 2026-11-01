"""Test suite running Phase 4 validation: V9 active learning and Sobol sensitivity.

Same contract as test_validation_v7_v8.py: the validators must run against the
real-CFD dataset, return finite/consistent metrics, and report their scorecard
honestly. The tests do NOT assert that active learning wins - the measured V9
result is that it does not pay for itself on this dataset at these budgets, and
forcing that pass would defeat the point of the validation.
"""

from __future__ import annotations

import pytest

from validation.sensitivity_sobol import run_sobol_study
from validation.v9_active_learning import run_v9_validation


@pytest.mark.slow
def test_v9_active_learning():
    """V9 must report acquisition learning curves on real HF data honestly."""
    results = run_v9_validation()
    assert results["hf_source"] == "cfd"
    assert float(results["al_rmse_lenient"]) >= 0.0
    assert float(results["random_rmse_lenient"]) >= 0.0
    assert float(results["maximin_rmse_lenient"]) >= 0.0
    assert float(results["strict_yield_al"]) >= 0.0

    scorecard = (
        float(results["reduction_vs_random_pct"]) > 5.0
        and float(results["reduction_vs_maximin_pct"]) > 5.0
        and float(results["strict_yield_al"]) > float(results["strict_yield_random"])
    )
    assert bool(results["targets_met"]) is scorecard


@pytest.mark.slow
def test_sobol_sensitivity():
    """Sobol study must run on the real dataset with a stable index set."""
    results = run_sobol_study()
    assert results["hf_source"] == "cfd"

    for target in ("cl", "cd"):
        block = results[target]
        if isinstance(block, dict):
            st = [float(v) for v in block["indices"]["ST"]]  # type: ignore[index]
            assert len(st) == 5
            assert max(st) <= 1.5  # totals are bounded for an additive surrogate
            assert block["stable"] is True  # type: ignore[index]
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected Sobol block for {target}: {block}")

    # The study's own convergence verdict must match the stored blocks.
    stable_all = all(
        results[t]["stable"] for t in ("cl", "cd") if isinstance(results[t], dict)
    )
    assert bool(results["targets_met"]) is bool(stable_all)
