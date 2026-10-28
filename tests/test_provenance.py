"""Provenance integrity tests.

These lock in the rule that a result may never claim to be something it is not:
a synthetic fallback must not be labelled OpenFOAM, and a convergence flag must
reflect an actual solver run.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.ml.dataset import evaluate_design_point
from backend.models.aero_config import FormulaAeroConfig
from backend.simulation.analytical import simulate
from backend.simulation.cfd.models import AirfoilGeometry, ConvergenceStatus, MeshLevel
from backend.simulation.cfd.openfoam_runner import run_cfd_with_provenance
from backend.simulation.cfd.runner import detect_cfd_backend, run_cfd_case
from backend.simulation.panel_method import simulate_panel


def test_health_reports_actual_backend() -> None:
    client = TestClient(app)
    body = client.get("/api/health").json()
    assert body["cfd_backend"] in {"native", "wsl", "synthetic"}
    assert body["provenance"]["cfd_solver_installed"] == (
        body["cfd_backend"] in {"native", "wsl"}
    )
    if body["cfd_backend"] == "synthetic":
        assert "not a mesh solution" in body["provenance"]["note"]


def test_vlm_and_analytical_declare_their_fidelity() -> None:
    cfg = FormulaAeroConfig()
    vlm = simulate_panel(cfg)
    assert vlm.fidelity == "low_fidelity_vlm"
    assert vlm.provenance["validated_externally"] is False

    empirical = simulate(cfg)
    assert empirical.fidelity == "empirical"
    assert empirical.provenance["validated_externally"] is False


def test_cfd_result_provenance_matches_the_backend() -> None:
    cfg = FormulaAeroConfig()
    sim, provenance = run_cfd_with_provenance(cfg)

    if detect_cfd_backend() == "synthetic":
        assert provenance.is_synthetic is True
        assert provenance.solver_used == "calibrated_synthetic"
        assert provenance.is_converged is False
        assert provenance.fidelity == "synthetic_model"
        assert sim.method == "cfd_synthetic_model"
        assert "synthetic" in provenance.solver_version
        assert "OpenFOAM" not in provenance.solver_version
    else:
        assert provenance.is_synthetic is False
        assert sim.method == "openfoam_rans"
        assert provenance.fidelity == "rans"


def test_synthetic_case_does_not_write_a_fake_solver_log(tmp_path: Path) -> None:
    if detect_cfd_backend() != "synthetic":
        return

    result = run_cfd_case(
        AirfoilGeometry(chord_m=0.5, angle_of_attack_deg=4.0, ground_height_m=0.03),
        mesh_level=MeshLevel.MEDIUM,
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


def test_dataset_rows_carry_real_provenance() -> None:
    sample = {
        "fw_main_angle": 3.0,
        "fw_flap_angle": 8.0,
        "ride_height_front_mm": 25.0,
        "rear_wing_main_angle": 10.0,
        "speed_kmh": 200.0,
    }
    row = evaluate_design_point(sample)

    assert row["solver_used"] in {"calibrated_synthetic", "openfoam_native", "openfoam_wsl"}
    assert row["hf_fidelity"] in {"synthetic_model", "rans"}
    if row["is_synthetic"]:
        assert row["convergence_flag"] is False
        assert row["convergence_status"] == "mock_evaluated"
        assert "synthetic" in row["solver_version"]
