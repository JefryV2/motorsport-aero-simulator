"""Test suite running validation cases V1 to V4."""

from __future__ import annotations

from validation.v1_elliptic import run_v1_validation
from validation.v2_aspect_ratio import run_v2_validation
from validation.v3_convergence import run_v3_validation
from validation.v4_ground_effect import run_v4_validation


def test_v1_elliptic_planform():
    res = run_v1_validation()
    assert res["passed"], f"V1 validation failed: {res}"


def test_v2_aspect_ratio_sweep():
    res = run_v2_validation()
    assert res["passed"], f"V2 validation failed: {res}"


def test_v3_panel_convergence():
    res = run_v3_validation()
    assert res["passed"], f"V3 validation failed: {res}"


def test_v4_ground_effect_trend():
    res = run_v4_validation()
    assert res["passed"], f"V4 validation failed: {res}"
