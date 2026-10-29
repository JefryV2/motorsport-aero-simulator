"""Slow test suite for CFD validation cases V5 and V6.

Run explicitly with: pytest -m slow tests/test_validation_v5_v6.py
These execute real OpenFOAM cases and are intentionally excluded from the
default fast suite.
"""

from __future__ import annotations

import math

import pytest

from validation.v5_grid_independence import run_v5_validation
from validation.v6_cfd_benchmark import run_v6_validation

pytestmark = pytest.mark.slow


def test_v5_mesh_independence_gci():
    """Run Case V5 and record the measured GCI without forcing a pass."""
    report = run_v5_validation()
    assert math.isfinite(report.gci_fine_pct)
    assert report.gci_fine_pct >= 0.0
    assert math.isfinite(report.phi_extrapolated)


def test_v6_cfd_literature_benchmark():
    """Run Case V6 and verify only the measured sanity/trend conditions."""
    res = run_v6_validation()
    assert math.isfinite(res["cd0_delta"])
    assert res["cfd_slope"] > 0.0
    assert res["ground_effect_trend"] == 1.0
