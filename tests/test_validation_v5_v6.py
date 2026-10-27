"""Test suite running CFD validation cases V5 and V6."""

from __future__ import annotations

from validation.v5_grid_independence import run_v5_validation
from validation.v6_cfd_benchmark import run_v6_validation


def test_v5_mesh_independence_gci():
    """Verify Case V5: 3-level ASME GCI < 3.0% and asymptotic range condition."""
    report = run_v5_validation()
    assert report.gci_fine_pct < 3.0
    assert report.is_in_asymptotic_range


def test_v6_cfd_literature_benchmark():
    """Verify Case V6: NACA 0012 polar & front wing ground effect vs literature."""
    res = run_v6_validation()
    assert res["cd0_delta"] < 0.0015
    assert res["slope_error_pct"] < 4.0
