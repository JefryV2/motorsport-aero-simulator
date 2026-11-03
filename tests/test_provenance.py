"""Provenance integrity tests.

These lock in the rule that a result may never claim to be something it is not:
a synthetic fallback must not be labelled OpenFOAM, a convergence flag must
reflect an actual solver run, and a 2D RANS + empirical hybrid must not be
advertised as a full-car CFD solution.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.ai.advisor import _fallback_advice, _wing_moves
from backend.main import app
from backend.ml.dataset import DEFAULT_DATASET_PATH, load_dataset_parquet
from backend.models.aero_config import (
    AdvisorRequest,
    AeroForces,
    FormulaAeroConfig,
    OptimizationResult,
    TrackContext,
)
from backend.simulation.analytical import (
    WHEELBASE_M,
    centre_of_pressure_m,
    geometric_rake_deg,
    lap_time_delta_s,
    simulate,
)
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


def _hybrid_rear_wing_cases() -> list[tuple[str, FormulaAeroConfig]]:
    cfg = FormulaAeroConfig()
    with_gurney = cfg.model_copy(
        update={
            "rear_wing_main": cfg.rear_wing_main.model_copy(update={"gurney_mm": 20.0})
        }
    )
    with_shorter_span = cfg.model_copy(
        update={
            "rear_wing_main": cfg.rear_wing_main.model_copy(update={"span_m": 0.85}),
            "rear_wing_flap": cfg.rear_wing_flap.model_copy(update={"span_m": 0.85}),
        }
    )
    cases = [
        ("default", cfg),
        ("drs_open", cfg.model_copy(update={"drs_active": True})),
        ("rear_gurney_20mm", with_gurney),
        ("rear_span_0.85m", with_shorter_span),
        ("speed_250kmh", cfg.model_copy(update={"speed_kmh": 250.0})),
    ]
    return cases


def test_hybrid_tier_does_not_restate_the_rear_wing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CFD tier solved a 2D front-wing section; the rear wing is this model's
    empirical term and must be the same number the analytical tier charges."""
    monkeypatch.setattr(
        "backend.simulation.cfd.openfoam_runner.run_cfd_case",
        lambda *args, **kwargs: _fake_cfd_result(synthetic=False),
    )

    for label, cfg in _hybrid_rear_wing_cases():
        hybrid, _ = run_cfd_with_provenance(cfg)
        assert hybrid.forces.downforce_rear_N == simulate(cfg).forces.downforce_rear_N, label


def test_hybrid_rear_load_moves_with_the_rear_wing_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gurney, span and DRS reach the hybrid rear wing through the shared term."""
    monkeypatch.setattr(
        "backend.simulation.cfd.openfoam_runner.run_cfd_case",
        lambda *args, **kwargs: _fake_cfd_result(synthetic=False),
    )
    cases = dict(_hybrid_rear_wing_cases())
    base = run_cfd_with_provenance(cases["default"])[0].forces

    with_gurney = run_cfd_with_provenance(cases["rear_gurney_20mm"])[0].forces
    assert with_gurney.downforce_rear_N > base.downforce_rear_N

    narrow = run_cfd_with_provenance(cases["rear_span_0.85m"])[0].forces
    assert narrow.downforce_rear_N < base.downforce_rear_N

    drs = run_cfd_with_provenance(cases["drs_open"])[0].forces
    assert drs.downforce_rear_N < base.downforce_rear_N
    assert drs.drag_N < base.drag_N


class TestOneBalanceLeverAcrossTiers:
    """Aero balance and centre of pressure are one split stated twice.

    The CFD tier reported `1.6 * balance_front_pct / 100` for `cop_x_m` while the
    other two tiers used the wheelbase lever, so the same 91.8 % front-loaded
    hybrid result printed a CoP 0.858 m aft of the front axle - and moved it
    rearward as the car became more front-heavy, the opposite of the definition
    `backend/models/aero_config.py` documents ("from front axle, metres").
    """

    def test_the_lever_endpoints_and_direction(self) -> None:
        assert centre_of_pressure_m(1000.0, 1000.0) == 0.0
        assert centre_of_pressure_m(0.0, 1000.0) == WHEELBASE_M
        assert centre_of_pressure_m(500.0, 1000.0) == WHEELBASE_M * 0.5
        assert centre_of_pressure_m(0.0, 0.0) == WHEELBASE_M * 0.5
        # More of the load ahead of the centre of pressure pulls it forward.
        assert centre_of_pressure_m(700.0, 1000.0) < centre_of_pressure_m(300.0, 1000.0)

    def test_every_tier_reports_the_shared_lever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "backend.simulation.cfd.openfoam_runner.run_cfd_case",
            lambda *args, **kwargs: _fake_cfd_result(synthetic=False),
        )
        cfg = FormulaAeroConfig()
        tiers = {
            "analytical": simulate(cfg),
            "panel_vlm": simulate_panel(cfg),
            "cfd": run_cfd_with_provenance(cfg)[0],
        }
        for label, sim in tiers.items():
            forces = sim.forces
            want = centre_of_pressure_m(forces.downforce_front_N, forces.downforce_total_N)
            assert forces.cop_x_m == pytest.approx(want, abs=1e-3), (
                f"{label} restates the lever: {forces.cop_x_m} vs {want:.3f}"
            )
            # The two fields must describe the same split, to their own rounding.
            assert forces.cop_x_m == pytest.approx(
                WHEELBASE_M * (1.0 - forces.balance_front_pct / 100.0), abs=2e-3
            ), label


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


def test_advisor_prices_drs_from_the_model() -> None:
    """DRS advice must quote this model's own delta, not a handbook percentage."""
    cfg = FormulaAeroConfig()
    closed = simulate(cfg.model_copy(update={"drs_active": False})).forces
    opened = simulate(cfg.model_copy(update={"drs_active": True})).forces

    advice = _fallback_advice(
        AdvisorRequest(config=cfg, simulation=simulate(cfg), question="what does DRS do")
    )
    assert f"{closed.drag_N:.0f}" in advice
    assert f"{opened.drag_N:.0f}" in advice
    assert "25-30%" not in advice


# The UI's own preset descriptions, kept verbatim because their wording is the
# point: each one contains a keyword a rule branch matches on.
TRACK_PRESETS = {
    "Silverstone": ("Balanced medium-high downforce circuit. High-speed corners.", 1.0, 1.0),
    "Monza": ("Temple of Speed. Ultra-low drag is required.", 2.2, 0.35),
    "Monaco": ("Tight street circuit. Maximum downforce setup.", 0.15, 2.4),
    "Spa-Francorchamps": ("Speed in Sector 1/3, cornering downforce in Sector 2.", 1.3, 0.95),
    "Singapore": ("Street circuit. High downforce and stability.", 0.6, 1.8),
}


def _track(name: str) -> TrackContext:
    description, drag_scale, df_scale = TRACK_PRESETS[name]
    return TrackContext(
        name=name,
        description=description,
        drag_multiplier=drag_scale,
        downforce_multiplier=df_scale,
    )


MOVE_RE = re.compile(
    r"(?P<element>Front wing flap|Rear wing flap|Rear wing main) "
    r"(?P<frm>-?[0-9.]+) to (?P<to>-?[0-9.]+) deg"
)
MOVE_FIELDS = {
    "Front wing flap": "front_wing_flap",
    "Rear wing flap": "rear_wing_flap",
    "Rear wing main": "rear_wing_main",
}


def _variant(cfg: FormulaAeroConfig, label: str) -> FormulaAeroConfig:
    """Rebuild the setup an advice line claims to have priced."""
    move = MOVE_RE.search(label)
    assert move, f"cannot read the move from {label!r}"
    field = MOVE_FIELDS[move.group("element")]
    assert getattr(cfg, field).angle_deg == pytest.approx(float(move.group("frm")))
    return cfg.model_copy(
        update={
            field: getattr(cfg, field).model_copy(update={"angle_deg": float(move.group("to"))})
        }
    )


def _advice(
    question: str, track: TrackContext | None = None, cfg: FormulaAeroConfig | None = None
) -> str:
    cfg = cfg or FormulaAeroConfig()
    return _fallback_advice(
        AdvisorRequest(config=cfg, simulation=simulate(cfg), question=question, track=track)
    )


def test_the_selected_track_cannot_steer_an_unrelated_question() -> None:
    """A circuit used to ride into the advisor as prose inside `question`, and the
    rule branches keyword-matched whatever arrived there. Measured before the fix:
    one question - "how do I read the pressure map?" - got five different answers,
    an understeer lecture under Silverstone (its description says "Balanced"), a
    drag lecture under Monza and Spa, and Monaco's canned degrees under Monaco and
    Singapore, because those descriptions say "drag" or "downforce".
    """
    question = "how do I read the pressure map?"
    # Only the part above the separator is chosen by keyword matching. The sections
    # below it are *meant* to know which track is selected now.
    answers = {
        _advice(question, _track(name)).split("\n---")[0] for name in TRACK_PRESETS
    } | {_advice(question, None).split("\n---")[0]}

    assert len(answers) == 1, "the selected track changed which answer a question got"
    answer = answers.pop()
    for tell in (
        "To resolve understeer",
        "To resolve oversteer",
        "you must reduce aerodynamic drag",
        "18-20",
    ):
        assert tell not in answer, f"the track's prose chose a branch: {tell!r}"

    sections = {
        _advice(question, _track(name)).split("\n---", 1)[1] for name in TRACK_PRESETS
    }
    assert len(sections) == len(TRACK_PRESETS), (
        "the selected track no longer reaches the summary and track sections"
    )


def test_track_setup_advice_is_priced_by_the_solver() -> None:
    """The old low-speed branch ended with fixed degrees no tier had ever produced.

    The replacement has to be reproducible from the model: the test parses the move
    the advice reports and recomputes its lap delta and force changes itself.
    """
    advice = _advice("what track setup should I run?", _track("Monaco"))
    assert "18-20" not in advice
    assert "25-30" not in advice

    move = re.search(
        r"Fastest: (?P<label>.+?) at (?P<lap>[-+][0-9.]+) s "
        r"\((?P<df>[-+][0-9.]+) N downforce, (?P<drag>[-+][0-9.]+) N drag\)",
        advice,
    )
    assert move, f"no move in the advice can be recomputed:\n{advice}"

    cfg = FormulaAeroConfig()
    track = _track("Monaco")
    base, moved = simulate(cfg).forces, simulate(_variant(cfg, move.group("label"))).forces

    def lap(f: AeroForces) -> float:
        return lap_time_delta_s(
            f.drag_N,
            f.downforce_total_N,
            drag_scale=track.drag_multiplier,
            downforce_scale=track.downforce_multiplier,
        )

    assert float(move.group("lap")) == pytest.approx(lap(moved) - lap(base), abs=5e-4)
    assert float(move.group("df")) == pytest.approx(
        moved.downforce_total_N - base.downforce_total_N, abs=0.5
    )
    assert float(move.group("drag")) == pytest.approx(moved.drag_N - base.drag_N, abs=0.5)


def test_track_weighting_actually_changes_the_verdict() -> None:
    """If the multipliers were decoration, both circuits would print the same line."""
    monaco = _advice("what track setup should I run?", _track("Monaco"))
    monza = _advice("what track setup should I run?", _track("Monza"))

    assert monaco != monza
    assert "drag x0.15 / downforce x2.4" in monaco
    assert "drag x2.2 / downforce x0.35" in monza


def test_a_track_question_does_not_price_the_same_moves_twice() -> None:
    """Otherwise the answer and the recommendation section print the same two
    numbered lines in one reply."""
    advice = _advice("what track setup should I run?", _track("Monaco"))
    assert advice.count("- Fastest:") == 1
    assert "Already priced above" in advice.split("## Track Recommendation")[1]

    other = _advice("what about balance?", _track("Monaco"))
    assert "- Fastest:" not in other
    assert "fastest one-knob move" in other.split("## Track Recommendation")[1]


def test_advise_endpoint_takes_the_track_as_fields(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = FormulaAeroConfig()
    track = _track("Monza")

    response = TestClient(app).post(
        "/api/advise",
        json={
            "config": cfg.model_dump(),
            "simulation": simulate(cfg).model_dump(),
            "question": "what track setup should I run?",
            "track": track.model_dump(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "rule_based"
    advice = payload["advice"]
    assert "### Answer to: 'what track setup should I run?'" in advice
    assert "Active track context" not in advice
    assert f"{track.name} weights the lap proxy" in advice


def test_efficiency_is_no_longer_graded_against_an_unreachable_target() -> None:
    """`efficiency < 2.5` decided the summary text, but the model cannot get there.

    Measured before the change: the best L/D out of a 100-point sweep of wing angles
    and front ride height was 1.45, so every setup was called "Efficiency is low"
    and the "Good aero efficiency" branch was dead code.
    """
    cfg = FormulaAeroConfig()
    f = simulate(cfg).forces
    advice = _advice("how efficient is this?", None, cfg)

    assert "Efficiency is low" not in advice
    assert "Good aero efficiency" not in advice
    assert "2.5" not in advice
    assert f"Efficiency is {f.efficiency} L/D on the analytical tier" in advice

    move = re.search(
        r"best one-knob move for efficiency is (?P<label>.+?) at (?P<ld>[0-9.]+) L/D "
        r"\((?P<df>[-+][0-9.]+) N downforce, (?P<drag>[-+][0-9.]+) N drag\)",
        advice,
    )
    assert move, advice
    moved = simulate(_variant(cfg, move.group("label"))).forces
    assert moved.efficiency == pytest.approx(float(move.group("ld")), abs=5e-3)
    assert moved.downforce_total_N - f.downforce_total_N == pytest.approx(
        float(move.group("df")), abs=0.5
    )


def test_balance_is_compared_with_the_objective_target_not_an_invented_window() -> None:
    """The old text called 40-44 % front "the optimal window" out of nothing.

    The same model reaches 21.4-82.2 % across the angles it can set, and the only
    balance number it owns as a target is the optimizer's `target_balance_pct`.
    """
    cfg = FormulaAeroConfig()
    f = simulate(cfg).forces
    untargeted = _advice("what about balance?", None, cfg)

    assert "optimal window" not in untargeted
    assert "40-44" not in untargeted
    assert "No balance target was given" in untargeted

    opt = OptimizationResult(
        best_config=cfg,
        best_forces=f,
        improvement_pct=0.0,
        trials_run=1,
        target_balance_pct=48.0,
    )
    targeted = _fallback_advice(
        AdvisorRequest(
            config=cfg,
            simulation=simulate(cfg),
            optimization=opt,
            question="what about balance?",
        )
    )
    assert f"{48.0 - f.balance_front_pct:.1f} points short at the front" in targeted

    correction = re.search(
        r"- (?P<label>.+?) closes that gap best on the analytical tier: "
        r"[0-9.]+% -> (?P<value>[0-9.]+)% front",
        targeted,
    )
    assert correction, targeted
    moved = simulate(_variant(cfg, correction.group("label"))).forces
    assert moved.balance_front_pct == pytest.approx(float(correction.group("value")), abs=0.05)


def test_the_track_recommendation_stops_being_a_speedometer() -> None:
    """The bands were absolute newtons, so they read the speed rather than the setup.

    Measured: one fixed setup at its defaults was "Low drag setup - ideal for
    high-speed circuits (Monza, Spa)" at 200 km/h, "Balanced" at 300 and "High
    downforce setup - ideal for twisty circuits (Monaco, Hungaroring)" at 350.
    """
    selected = _track("Silverstone")
    for speed in (50.0, 200.0, 350.0):
        advice = _advice(
            "what about balance?", selected, FormulaAeroConfig(speed_kmh=speed)
        )
        section = advice.split("## Track Recommendation")[1]
        for other in ("Monza", "Spa", "Monaco", "Barcelona", "Hungaroring"):
            assert other not in section, f"at {speed} km/h it recommended {other}"
        assert "Silverstone" in section
        assert "2500" not in section and "4000" not in section

    no_track = _advice("what about balance?", None)
    assert "No track is selected" in no_track.split("## Track Recommendation")[1]


SHIFT_RE = re.compile(
    r"(?P<frm>[0-9.]+)% -> (?P<to>[0-9.]+)% front, bought with "
    r"(?P<df>[+-][0-9]+) N total downforce and (?P<drag>[+-][0-9]+) N drag"
)
ADDITIVE_RE = re.compile(
    r"shifts (?P<frm>[0-9.]+)% -> (?P<to>[0-9.]+)% front while gaining "
    r"(?P<df>[+-][0-9]+) N of total downforce for (?P<drag>[+-][0-9]+) N of drag"
)
DRAG_CUT_RE = re.compile(
    r"deg: (?P<frm>[0-9]+) N -> (?P<to>[0-9]+) N, paid for with "
    r"(?P<df>[+-][0-9]+) N of total downforce"
)
HANDLING_QUESTIONS = (
    "how do I fix the understeer?",
    "the rear is loose through the slow corners",
    "how do I reduce drag on the straight?",
)


def _setup(front: float, rear_flap: float, rear_main: float) -> FormulaAeroConfig:
    """A config at given wing angles, the way the UI sliders would leave it."""
    cfg = FormulaAeroConfig()
    for field, angle in (
        ("front_wing_flap", front),
        ("rear_wing_flap", rear_flap),
        ("rear_wing_main", rear_main),
    ):
        cfg = cfg.model_copy(
            update={field: getattr(cfg, field).model_copy(update={"angle_deg": angle})}
        )
    return cfg


@pytest.mark.parametrize(
    "cfg", [_setup(8.0, 15.0, 5.0), _setup(14.0, 24.0, 12.0), _setup(2.0, 6.0, 2.0)]
)
def test_every_handling_move_reproduces_the_solver_it_quotes(cfg: FormulaAeroConfig) -> None:
    """A recommendation the model cannot re-solve is a guess with units on it.

    The branches used to answer every handling question with a fixed range - "by
    2-3°", "by 2-4°", "towards 10° or lower" - none of which came from the solver's
    own sensitivity to that knob.
    """
    base = simulate(cfg).forces
    for question in HANDLING_QUESTIONS:
        answer = _advice(question, None, cfg).split("\n---")[0]
        for line in answer.splitlines():
            label = MOVE_RE.search(line)
            if label is None:
                continue
            moved = simulate(_variant(cfg, label.group(0))).forces
            shift = SHIFT_RE.search(line) or ADDITIVE_RE.search(line)
            cut = DRAG_CUT_RE.search(line)
            assert shift or cut, f"the line names a move but prices nothing: {line!r}"

            if shift:
                assert float(shift["frm"]) == pytest.approx(
                    base.balance_front_pct, abs=0.05
                )
                assert float(shift["to"]) == pytest.approx(
                    moved.balance_front_pct, abs=0.05
                )
                assert float(shift["drag"]) == pytest.approx(
                    moved.drag_N - base.drag_N, abs=0.5
                )
            if cut:
                assert float(cut["frm"]) == pytest.approx(base.drag_N, abs=0.5)
                assert float(cut["to"]) == pytest.approx(moved.drag_N, abs=0.5)
            priced = shift or cut
            assert priced is not None
            assert float(priced["df"]) == pytest.approx(
                moved.downforce_total_N - base.downforce_total_N, abs=0.5
            )


def test_the_prescribed_degree_ranges_are_gone() -> None:
    """Locks the text the audit flagged, phrased exactly as it shipped for a while."""
    for cfg in (FormulaAeroConfig(), _setup(14.0, 24.0, 12.0)):
        for question in HANDLING_QUESTIONS:
            advice = _advice(question, None, cfg)
            for canned in ("by 2-3", "by 2-4", "towards 10", "18-20", "25-30"):
                assert canned not in advice, f"{canned!r} survived on {question!r}"


def test_a_saturated_setup_is_not_offered_a_move_that_changes_nothing() -> None:
    """Past the stall clamp a flap still takes an angle and returns the same forces.

    Measured at 35/35/35 deg: the model reported every one-knob variant with the same
    drag and downforce it already had, and the advice still called one of them the
    fastest move and the one that "moves the split most".
    """
    cfg = _setup(35.0, 35.0, 35.0)
    questions = HANDLING_QUESTIONS + ("what track setup should I run?",)
    for question in questions:
        advice = _advice(question, _track("Monaco"), cfg)
        assert MOVE_RE.search(advice) is None, f"{question!r} prescribed a no-op move"
        assert "No one-knob flap move" in advice.split("\n---")[0]


def test_an_unreachable_balance_target_is_not_called_a_correction() -> None:
    """`min` over every move used to name the least wrong one as the fix.

    Measured with every wing at -5 deg: the balance sits at 50.0% front because the
    car is producing lift, and all three flap moves change the forces while shifting
    the split by 0.000 points. Against a 20% target the old code still returned one of
    them as the move that "closes that gap best".
    """
    cfg = _setup(-5.0, -5.0, -5.0)
    f = simulate(cfg).forces
    assert _wing_moves(cfg), "this setup is only useful if it still has priced moves"
    opt = OptimizationResult(
        best_config=cfg,
        best_forces=f,
        improvement_pct=0.0,
        trials_run=1,
        target_balance_pct=20.0,
    )
    advice = _fallback_advice(
        AdvisorRequest(
            config=cfg,
            simulation=simulate(cfg),
            optimization=opt,
            question="what about balance?",
        )
    )
    assert "closes that gap best" not in advice
    assert "shifts this setup toward the objective's 20.0% target" in advice


def test_a_priced_move_names_the_tier_its_starting_number_came_from() -> None:
    """Measured live on /api/advise: the answer opened with "64.8% front" and then
    priced a move from 43.8%, because the UI sends an analytical_enhanced simulation
    while the moves are solved on the analytical tier. Two numbers, one label - and the
    reader had no way to see that they were two solvers disagreeing.
    """
    cfg = FormulaAeroConfig(speed_kmh=250.0)
    enhanced = simulate_panel(cfg)
    analytical = simulate(cfg).forces
    assert enhanced.method == "analytical_enhanced"
    assert analytical.balance_front_pct == pytest.approx(43.8, abs=0.1)

    for question in HANDLING_QUESTIONS:
        advice = _fallback_advice(
            AdvisorRequest(config=cfg, simulation=enhanced, question=question)
        )
        answer = advice.split("\n---")[0]
        assert "analytical_enhanced" in answer, f"{question!r} priced a move tier-blind"
        if "drag" in question:
            assert f"{enhanced.forces.drag_N}" in answer
        else:
            assert f"{enhanced.forces.balance_front_pct}%" in answer


def test_a_balance_gap_is_measured_on_the_tier_that_prices_its_correction() -> None:
    """Measured live: the UI simulates analytical_enhanced, where this setup reads
    64.8% front, while every priced move shifts the analytical tier's 43.8%. Against a
    42% target the advisor reported the gap from one solver and answered it with a move
    measured by another.
    """
    cfg = FormulaAeroConfig()
    enhanced = simulate_panel(cfg)
    analytical = simulate(cfg).forces
    opt = OptimizationResult(
        best_config=cfg,
        best_forces=enhanced.forces,
        improvement_pct=0.0,
        trials_run=1,
        target_balance_pct=42.0,
    )
    advice = _fallback_advice(
        AdvisorRequest(
            config=cfg,
            simulation=enhanced,
            optimization=opt,
            question="what about balance?",
        )
    )
    key = advice.split("## Key Changes")[1].split("\n##")[0]
    gap = 42.0 - analytical.balance_front_pct
    stated = f"{abs(gap):.1f} points {'short' if gap > 0 else 'over'} at the front"
    assert stated in key, key
    assert f"{abs(42.0 - enhanced.forces.balance_front_pct):.1f} points" not in key
    assert f"({enhanced.forces.balance_front_pct}% on the analytical_enhanced tier" in key

    correction = re.search(
        r"on the analytical tier: (?P<frm>[0-9.]+)% -> (?P<to>[0-9.]+)% front", key
    )
    assert correction, key
    assert float(correction["frm"]) == pytest.approx(analytical.balance_front_pct, abs=0.05)
