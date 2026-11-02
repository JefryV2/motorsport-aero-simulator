"""Provenance integrity tests.

These lock in the rule that a result may never claim to be something it is not:
a synthetic fallback must not be labelled OpenFOAM, a convergence flag must
reflect an actual solver run, and a 2D RANS + empirical hybrid must not be
advertised as a full-car CFD solution.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.ai.advisor import _fallback_advice
from backend.main import app
from backend.ml.dataset import DEFAULT_DATASET_PATH, load_dataset_parquet
from backend.models.aero_config import AdvisorRequest, FormulaAeroConfig
from backend.simulation.analytical import geometric_rake_deg, simulate
from backend.simulation.cfd.models import (
    AirfoilGeometry,
    CFDResult,
    ConvergenceStatus,
    ConvergenceVerdict,
    MeshLevel,
    MeshStats,
)
from backend.simulation.cfd.openfoam_runner import run_cfd_with_provenance
from backend.simulation.cfd.runner import CFDRunError
from backend.simulation.panel_method import simulate_panel


def _fake_cfd_result(*, synthetic: bool) -> CFDResult:
    mesh = MeshStats(
        level=MeshLevel.MEDIUM,
        num_cells=21000,
        num_points=40800,
        grid_spacing_h=0.08,
        estimated_y_plus=120.0,
    )
    verdict = ConvergenceVerdict(
        status=ConvergenceStatus.MOCK_EVALUATED if synthetic else ConvergenceStatus.CONVERGED,
        iterations_run=0 if synthetic else 400,
        cl_mean=0.63 if not synthetic else 2.0,
        cl_std=0.0 if synthetic else 0.0004,
        cd_mean=0.11 if not synthetic else 0.2,
        cd_std=0.0 if synthetic else 0.0002,
        final_p_residual=0.0 if synthetic else 3.0e-5,
        final_u_residual=0.0 if synthetic else 4.0e-4,
        final_k_residual=0.0 if synthetic else 1.0e-4,
        final_omega_residual=0.0 if synthetic else 1.0e-5,
        is_converged=False if synthetic else True,
        cm_mean=-0.1,
        cm_std=0.0,
    )
    return CFDResult(
        case_id="case_test",
        case_dir="data/cfd_cases/case_test",
        mesh_stats=mesh,
        convergence=verdict,
        cl=verdict.cl_mean,
        cd=verdict.cd_mean,
        cm=verdict.cm_mean,
        downforce_N=1000.0,
        drag_N=150.0,
        execution_time_s=0.0 if synthetic else 12.5,
        solver_used="calibrated_synthetic" if synthetic else "openfoam_wsl",
        solver_version="none (calibrated-synthetic-v1)" if synthetic else "OpenFOAM 1912",
        is_synthetic=synthetic,
    )


def test_health_reports_actual_backend() -> None:
    client = TestClient(app)
    body = client.get("/api/health").json()
    assert body["cfd_backend"] in {"native", "wsl", "synthetic"}
    assert body["solver_version"]
    assert body["provenance"]["cfd_solver_installed"] == (
        body["cfd_backend"] in {"native", "wsl"}
    )
    if body["cfd_backend"] == "synthetic":
        assert "not a mesh solution" in body["provenance"]["note"]
    else:
        assert "hybrid" in body["provenance"]["note"]


def test_vlm_and_analytical_declare_their_fidelity() -> None:
    cfg = FormulaAeroConfig()
    vlm = simulate_panel(cfg)
    assert vlm.fidelity == "low_fidelity_vlm"
    assert vlm.provenance["validated_externally"] is False
    assert vlm.forces.downforce_total_N > 0.0

    empirical = simulate(cfg)
    assert empirical.fidelity == "empirical"
    assert empirical.provenance["validated_externally"] is False


def test_real_cfd_provenance_is_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.simulation.cfd.openfoam_runner.run_cfd_case",
        lambda *args, **kwargs: _fake_cfd_result(synthetic=False),
    )
    cfg = FormulaAeroConfig()
    sim, provenance = run_cfd_with_provenance(cfg)

    assert provenance.is_synthetic is False
    assert provenance.solver_used == "openfoam_wsl"
    assert provenance.solver_version == "OpenFOAM 1912"
    assert provenance.fidelity == "rans_hybrid"
    assert sim.method == "openfoam_rans_hybrid"
    assert sim.fidelity == "rans_hybrid"
    assert provenance.mesh_cells == 21000
    assert provenance.iterations == 400
    assert provenance.case_id == "case_test"
    assert "full-car CFD" in provenance.note
    assert "2D k-omega SST RANS" in provenance.note


def test_real_cfd_failure_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real solver failure must surface as 503, never as synthetic results."""

    def fail(*args, **kwargs):
        raise CFDRunError("forced solver failure")

    monkeypatch.setattr("backend.main.run_cfd", fail)
    client = TestClient(app)
    response = client.post(
        "/api/simulate?method=cfd",
        json=FormulaAeroConfig().model_dump(),
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "No synthetic fallback was substituted" in detail
    assert "forced solver failure" in detail


def test_synthetic_cfd_provenance_is_labelled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.simulation.cfd.openfoam_runner.run_cfd_case",
        lambda *args, **kwargs: _fake_cfd_result(synthetic=True),
    )
    cfg = FormulaAeroConfig()
    sim, provenance = run_cfd_with_provenance(cfg)

    assert provenance.is_synthetic is True
    assert provenance.solver_used == "calibrated_synthetic"
    assert provenance.fidelity == "synthetic_model"
    assert sim.method == "cfd_synthetic_model"
    assert "calibrated closed-form" in provenance.note


def test_synthetic_case_does_not_write_a_fake_solver_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.simulation.cfd.runner.detect_cfd_backend", lambda: "synthetic"
    )
    from backend.simulation.cfd.runner import run_cfd_case

    result = run_cfd_case(
        AirfoilGeometry(chord_m=0.5, angle_of_attack_deg=4.0, ground_height_m=0.03),
        mesh_level=MeshLevel.COARSE,
        speed_ms=40.0,
        base_dir=tmp_path,
    )

    case_dir = Path(result.case_dir)
    assert result.solver_used == "calibrated_synthetic"
    assert result.is_synthetic is True
    assert result.convergence.status is ConvergenceStatus.MOCK_EVALUATED
    assert result.convergence.is_converged is False
    assert (case_dir / "SYNTHETIC_MODEL.json").exists()
    assert not (case_dir / "log.simpleFoam").exists()


def test_dataset_rows_carry_provenance() -> None:
    if not DEFAULT_DATASET_PATH.exists():
        pytest.skip("dataset has not been generated")

    df = load_dataset_parquet(DEFAULT_DATASET_PATH)
    required = {
        "solver_used",
        "solver_version",
        "hf_fidelity",
        "is_synthetic",
        "convergence_flag",
        "convergence_status",
    }
    assert required.issubset(df.columns)
    assert set(df["solver_used"].astype(str)).issubset(
        {"calibrated_synthetic", "openfoam_native", "openfoam_wsl"}
    )
    assert set(df["hf_fidelity"].astype(str)).issubset(
        {"synthetic_model", "rans", "rans_hybrid"}
    )
    synthetic_rows = df[df["is_synthetic"].astype(bool)]
    if len(synthetic_rows):
        assert not synthetic_rows["convergence_flag"].astype(bool).any()
        assert set(synthetic_rows["convergence_status"].astype(str)) == {"mock_evaluated"}


def test_rake_term_is_not_reported_as_body_attitude() -> None:
    """`rake_deg` scales underbody suction; the body angle comes from the ride heights."""
    cfg = FormulaAeroConfig()
    assert cfg.rake_deg == pytest.approx(1.5)
    assert geometric_rake_deg(cfg) == pytest.approx(0.342, abs=1e-3)

    raised = cfg.model_copy(update={"ride_height_rear_mm": 125.0})
    assert geometric_rake_deg(raised) == pytest.approx(math.degrees(math.atan(100.0 / 3350.0)))

    advice = _fallback_advice(AdvisorRequest(config=cfg, simulation=simulate(cfg), question="what about rake"))
    assert "0.34" in advice
    assert "giving a rake of 1.5" not in advice
