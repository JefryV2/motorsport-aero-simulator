"""Fast tests for the imported-mesh 3D CFD tier.

The mesh is measured, never fitted and never invented: these tests pin the
measurement maths, the deterministic case files, the log-parsing path, and the
HTTP contract (no synthetic fallback - 503 without a solver, 422 out of
envelope, real provenance on success).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app
from backend.simulation.cfd.parser import parse_residuals
from backend.simulation.cfd.runner import clear_cfd_backend_cache
from backend.simulation.cfd3d.case_builder import (
    CASE3D_FORMAT_VERSION,
    build_imported_case,
)
from backend.simulation.cfd3d.models import LEVELS, ImportedCFDLevel
from backend.simulation.cfd3d.runner import (
    ImportedCFDProvenance,
    ImportedCFDResult,
    run_imported_cfd,
)
from backend.simulation.cfd3d.surface import (
    axes_to_solver,
    measure_imported_surface,
)


@pytest.fixture()
def box_stl() -> bytes:
    """A watertight binary-STL box 1.2 m long x 0.5 m tall x 1.0 m wide."""

    def box_stl_bytes(Lx: float, Ly: float, Lz: float) -> bytes:
        v = {
            "a": (0.0, 0.0, 0.0),
            "b": (Lx, 0.0, 0.0),
            "c": (Lx, 0.0, Lz),
            "d": (0.0, 0.0, Lz),
            "e": (0.0, Ly, 0.0),
            "f": (Lx, Ly, 0.0),
            "g": (Lx, Ly, Lz),
            "h": (0.0, Ly, Lz),
        }
        tris = [
            ["a", "b", "c"],
            ["a", "c", "d"],
            ["e", "g", "f"],
            ["e", "h", "g"],
            ["a", "e", "f"],
            ["a", "f", "b"],
            ["d", "c", "g"],
            ["d", "g", "h"],
            ["a", "d", "h"],
            ["a", "h", "e"],
            ["b", "f", "g"],
            ["b", "g", "c"],
        ]
        out = bytearray()
        out += struct.pack("<80s", b"test box")
        out += struct.pack("<I", len(tris))
        for t in tris:
            out += struct.pack("<3f", 0.0, 0.0, 0.0)
            for name in t:
                out += struct.pack("<3f", *v[name])
            out += struct.pack("<H", 0)
        return bytes(out)

    return box_stl_bytes(1.2, 0.5, 1.0)


@pytest.fixture()
def box_surface(box_stl: bytes):
    return measure_imported_surface(box_stl, "stl", axes="solver")


class TestTheSurfaceMeasurement:
    """The numbers the case needs come from the file, not from a template."""

    def test_a_box_measures_exactly(self, box_surface) -> None:
        assert box_surface.frontal_area_m2 == pytest.approx(0.5, rel=1e-3)
        assert box_surface.watertight_edge_fraction == pytest.approx(1.0)
        assert box_surface.extent_x_m == pytest.approx(1.2, abs=1e-4)
        assert box_surface.extent_y_m == pytest.approx(0.5, abs=1e-4)
        assert box_surface.extent_z_m == pytest.approx(1.0, abs=1e-4)
        assert box_surface.source_triangle_count == 12
        assert box_surface.warnings == ()

    def test_overlapping_triangles_cannot_double_count(self, box_stl: bytes) -> None:
        # The 12-triangle box already projects 4 full quad-pairs per face; the
        # scanline union must still measure one 0.5 m^2 silhouette, not more.
        surface = measure_imported_surface(box_stl + box_stl, "stl", axes="solver")
        assert surface.frontal_area_m2 == pytest.approx(0.5, rel=1e-3)

    def test_three_frame_maps_to_solver_streamwise(self) -> None:
        import numpy as np

        # Three.js-style authoring: x lateral, y up, z rearward.
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -2.0], [0.0, 3.0, 0.0]])
        solver = axes_to_solver(v, "three")
        assert solver[1] == pytest.approx([2.0, 0.0, 1.0])
        assert solver[2] == pytest.approx([0.0, 3.0, 0.0])

    def test_unknown_axes_is_rejected(self, box_stl: bytes) -> None:
        with pytest.raises(ValueError, match="axes"):
            measure_imported_surface(box_stl, "stl", axes="six")

    def test_bad_format_is_rejected_with_a_format_message(self, box_stl: bytes) -> None:
        with pytest.raises(ValueError, match="Unsupported CFD format"):
            measure_imported_surface(box_stl, "ply")

    def test_zero_or_negative_scale_is_rejected(self, box_stl: bytes) -> None:
        with pytest.raises(ValueError, match="scale_m_per_unit"):
            measure_imported_surface(box_stl, "stl", scale_m_per_unit=0.0)

    def test_out_of_envelope_extents_are_refused_not_reinterpreted(
        self, box_stl: bytes
    ) -> None:
        # A 100x scale makes a 120 m long monster: ValueError, not a "fix".
        with pytest.raises(ValueError, match="scale_m_per_unit"):
            measure_imported_surface(box_stl, "stl", scale_m_per_unit=100.0)
        with pytest.raises(ValueError, match="long"):
            measure_imported_surface(box_stl, "stl", scale_m_per_unit=0.01)


class TestTheCaseBuilder:
    """Cases are real, complete, deterministic OpenFOAM directories."""

    def test_rebuild_is_deterministic(self, box_surface, tmp_path: Path) -> None:
        b1 = build_imported_case(
            box_surface, speed_ms=30.0, base_dir=tmp_path
        )
        b2 = build_imported_case(
            box_surface, speed_ms=30.0, base_dir=tmp_path
        )
        assert b1.case_hash == b2.case_hash
        assert b1.case_dir == b2.case_dir

    def test_a_different_speed_changes_the_hash(self, box_surface, tmp_path: Path) -> None:
        b1 = build_imported_case(box_surface, speed_ms=30.0, base_dir=tmp_path)
        b2 = build_imported_case(box_surface, speed_ms=40.0, base_dir=tmp_path)
        assert b1.case_hash != b2.case_hash

    def test_the_case_has_every_required_file(self, box_surface, tmp_path: Path) -> None:
        build = build_imported_case(box_surface, speed_ms=30.0, base_dir=tmp_path)
        case = build.case_dir
        required = [
            "system/blockMeshDict",
            "system/snappyHexMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "constant/triSurface/car.stl",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "0/U",
            "0/p",
            "0/k",
            "0/omega",
            "0/nut",
            "run_case.sh",
            "case_params.json",
        ]
        missing = [r for r in required if not (case / r).exists()]
        assert missing == [], f"case is incomplete: {missing}"

    def test_controlDict_carries_the_measured_reference_and_sign(
        self, box_surface, tmp_path: Path
    ) -> None:
        build = build_imported_case(box_surface, speed_ms=30.0, base_dir=tmp_path)
        control = (build.case_dir / "system" / "controlDict").read_text()
        assert "liftDir         (0 -1 0);" in control
        assert f"Aref            {box_surface.frontal_area_m2:.6f};" in control
        params = json.loads((build.case_dir / "case_params.json").read_text())
        assert params["cl_sign_convention"].startswith("positive Cl is downforce")
        assert params["case_hash"] == build.case_hash

    def test_snappy_dict_targets_the_uploaded_surface(
        self, box_surface, tmp_path: Path
    ) -> None:
        for level in ImportedCFDLevel:
            build = build_imported_case(
                box_surface, speed_ms=30.0, mesh_level=level, base_dir=tmp_path
            )
            snappy = (build.case_dir / "system" / "snappyHexMeshDict").read_text()
            params = LEVELS[level]
            assert 'file            "car.stl";' in snappy
            assert "triSurfaceMesh" in snappy
            assert "searchableSurfaceBox" not in snappy
            assert f"level ({params.surface_levels} {params.surface_levels});" in snappy
            assert f"nSurfaceLayers {params.n_layers};" in snappy

        build = build_imported_case(box_surface, speed_ms=30.0, base_dir=tmp_path)
        case_params = json.loads((build.case_dir / "case_params.json").read_text())
        assert case_params["format"] == CASE3D_FORMAT_VERSION


class TestTheRunner:
    """Logs are parsed, never fabricated; caching re-parses instead of solving."""

    @pytest.fixture()
    def solved_case(self, box_surface, tmp_path: Path) -> Path:
        """A case directory that looks like a real converged 1912 solve."""
        build = build_imported_case(
            box_surface,
            speed_ms=30.0,
            mesh_level=ImportedCFDLevel.MEDIUM,
            base_dir=tmp_path,
        )
        case = build.case_dir
        pp = case / "postProcessing" / "forceCoeffs" / "0"
        pp.mkdir(parents=True)
        header = (
            "# Time Cd Cs Cl CmRoll CmPitch CmYaw Cdr Clr Csrr Cmr Csyr Cmzr\n"
        )
        rows = "".join(f"{800 + i} 0.8 0 1.0 0 0.1 0 0.8 1.0 0 0.1 0 0\n" for i in range(40))
        (pp / "coefficient.dat").write_text(header + rows, encoding="utf-8")
        solver_tail = "".join(
            f"Time = {900 + i}\n"
            "smoothSolver:  Solving for Ux, Initial residual = 9.0e-07, Final residual = 8.1e-07\n"
            "smoothSolver:  Solving for Uy, Initial residual = 8.0e-07, Final residual = 7.1e-07\n"
            "smoothSolver:  Solving for Uz, Initial residual = 7.0e-07, Final residual = 6.1e-07\n"
            "GAMG:  Solving for p, Initial residual = 5.0e-07, Final residual = 4.1e-07\n"
            "smoothSolver:  Solving for k, Initial residual = 6.0e-07, Final residual = 5.1e-07\n"
            "smoothSolver:  Solving for omega, Initial residual = 3.0e-07, Final residual = 2.1e-07\n"
            for i in range(20)
        )
        (case / "log.simpleFoam").write_text(
            "Build : 1912\nOPENFOAM=1912\n" + solver_tail, encoding="utf-8"
        )
        (case / "log.checkMesh").write_text(
            "Mesh stats\n    cells:               30000\n"
            "    points:              31625\n"
            "Mesh non-orthogonality Max: 64.9\n"
            "    Number of severely non-orthogonal faces: 0\n"
            "Mesh OK.\n",
            encoding="utf-8",
        )
        (case / "postProcessing" / "yPlus" / "0").mkdir(parents=True, exist_ok=True)
        (case / "postProcessing" / "yPlus" / "0" / "yPlus.dat").write_text(
            "# Time patch min max average\n20 car 12.0 90.0 30.5\n", encoding="utf-8"
        )
        # Pretend a previous real solve wrote its provenance: the cached path
        # must re-parse instead of re-solving.
        (case / "case_meta.json").write_text(
            json.dumps(
                {
                    "solver_used": "openfoam_wsl",
                    "solver_version": "OpenFOAM 1912",
                    "is_converged": True,
                    "convergence_status": "converged",
                    "execution_time_s": 12.5,
                }
            ),
            encoding="utf-8",
        )
        return case

    def test_cached_cases_are_reparsed_not_solved(
        self, solved_case: Path, box_surface, monkeypatch
    ) -> None:
        # If the cached path reaches the solver, this explodes loudly.
        def must_not_run(*args: object, **kwargs: object) -> None:
            raise AssertionError("cached case must not launch a solver")

        monkeypatch.setattr(
            "backend.simulation.cfd3d.runner._run", must_not_run
        )
        monkeypatch.setattr(
            "backend.simulation.cfd3d.runner.detect_cfd_backend", lambda: "wsl"
        )
        result = run_imported_cfd(
            box_surface,
            30.0,
            mesh_level=ImportedCFDLevel.MEDIUM,
            base_dir=solved_case.parent,
        )
        assert result.cd == pytest.approx(0.8, abs=1e-6)
        assert result.cl == pytest.approx(1.0, abs=1e-6)
        q = 0.5 * 1.225 * 30.0**2
        assert result.drag_N == pytest.approx(0.8 * q * 0.5, abs=0.5)
        # positive Cl is downforce in this tier
        assert result.downforce_N == pytest.approx(1.0 * q * 0.5, abs=0.5)
        assert result.lift_N == pytest.approx(-result.downforce_N)
        assert result.provenance.iterations >= 40
        assert result.provenance.is_converged is True

    def test_a_failing_Uz_residual_is_not_hidden(
        self, solved_case: Path, box_surface, monkeypatch
    ) -> None:
        log = solved_case / "log.simpleFoam"
        content = log.read_text(encoding="utf-8")
        # one stuck Uz line appended after the converged series
        log.write_text(
            content
            + "smoothSolver:  Solving for Uz, Initial residual = 0.5, Final residual = 0.4\n",
            encoding="utf-8",
        )
        residuals = parse_residuals(solved_case / "log.simpleFoam")
        assert residuals["Ux"] == pytest.approx(9.0e-07)
        assert "Uz" not in residuals  # the 2D parser does not look for Uz...

        monkeypatch.setattr(
            "backend.simulation.cfd3d.runner.detect_cfd_backend", lambda: "wsl"
        )
        result = run_imported_cfd(
            box_surface,
            30.0,
            mesh_level=ImportedCFDLevel.MEDIUM,
            base_dir=solved_case.parent,
        )
        # ...but this tier does, and refuses to call that "converged".
        assert result.provenance.residual_u == pytest.approx(0.5)
        assert result.provenance.is_converged is False
        assert result.provenance.convergence_status == "unconverged"

    def test_the_tier_refuses_to_invent_numbers(
        self, box_surface, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "backend.simulation.cfd3d.runner.detect_cfd_backend", lambda: "synthetic"
        )
        with pytest.raises(Exception, match="refuses to invent"):
            run_imported_cfd(
                box_surface,
                30.0,
                mesh_level=ImportedCFDLevel.COARSE,
                base_dir=tmp_path,
            )


def _fake_result(cd: float = 0.8, cl: float = 1.2) -> ImportedCFDResult:
    speed_ms = 120.0 / 3.6
    q = 0.5 * 1.225 * speed_ms**2
    f_ref = q * 0.5
    prov = ImportedCFDProvenance(
        backend="wsl",
        solver_used="openfoam_wsl",
        case_hash="deadbeefdeadbeef",
        frontal_area_m2=0.5,
        mesh_triangles=12,
        watertight_edge_fraction=1.0,
        scale_m_per_unit=1.0,
        axes="solver",
        mesh_cells=30000,
        max_iterations=1500,
        iterations=900,
        convergence_status="converged",
        is_converged=True,
        cd_3d=cd,
        cl_3d=cl,
        cm_3d=0.4,
        cd_std=0.002,
        cl_std=0.004,
        residual_p=9e-7,
        residual_u=8e-7,
        residual_k=7e-7,
        residual_omega=6e-7,
        y_plus_avg=30.5,
        execution_time_s=99.0,
        solver_version="OpenFOAM 1912",
        case_dir="data/cfd_cases_3d/case3d_deadbeefdeadbeef",
        reynolds=2400000.0,
    )
    return ImportedCFDResult(
        drag_N=cd * f_ref,
        downforce_N=cl * f_ref,
        lift_N=-cl * f_ref,
        sideforce_N=0.0,
        cd=cd,
        cl=cl,
        cm=0.4,
        column_load_N=f_ref,
        q_dynamic_pa=q,
        speed_ms=speed_ms,
        frontal_area_m2=0.5,
        provenance=prov,
    )


class TestTheCFD3DEndpointOverHttp:
    """The API keeps the honesty rules: 503 without a solver, 422 bad meshes."""

    def _client(self) -> TestClient:
        return TestClient(app)

    def _upload(self, client: TestClient, data: bytes, name: str = "box.stl") -> str:
        resp = client.post(
            "/api/models/upload", files={"file": (name, data, "application/octet-stream")}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["model_id"]

    def test_measure_reports_the_real_mesh(self, box_stl: bytes) -> None:
        client = self._client()
        model_id = self._upload(client, box_stl)
        resp = client.post(
            "/api/cfd/3d/measure", json={"model_id": model_id, "axes": "solver"}
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["frontal_area_m2"] == pytest.approx(0.5, rel=1e-3)
        assert body["triangle_count"] == 12
        assert body["watertight_edge_fraction"] == pytest.approx(1.0)
        assert body["warnings"] == []

    def test_measure_unknown_model_is_404(self, box_stl: bytes) -> None:
        client = self._client()
        resp = client.post(
            "/api/cfd/3d/measure", json={"model_id": "nosuchid", "axes": "solver"}
        )
        assert resp.status_code == 404

    def test_measure_out_of_envelope_is_422(self, box_stl: bytes) -> None:
        client = self._client()
        model_id = self._upload(client, box_stl)
        resp = client.post(
            "/api/cfd/3d/measure",
            json={"model_id": model_id, "axes": "solver", "scale_m_per_unit": 100.0},
        )
        assert resp.status_code == 422
        assert "scale_m_per_unit" in resp.json()["detail"]

    def test_start_without_solver_is_503_never_simulated(
        self, box_stl: bytes, monkeypatch
    ) -> None:
        monkeypatch.setattr(main_mod, "solver_available", lambda: False)
        clear_cfd_backend_cache()
        try:
            client = self._client()
            model_id = self._upload(client, box_stl)
            resp = client.post(
                "/api/cfd/3d",
                json={"model_id": model_id, "speed_kmh": 120.0, "axes": "solver"},
            )
            assert resp.status_code == 503
            assert "refuses" in resp.json()["detail"]
        finally:
            clear_cfd_backend_cache()

    def test_unknown_job_is_404(self) -> None:
        client = self._client()
        resp = client.get("/api/cfd/3d/nosuchjob")
        assert resp.status_code == 404

    def test_job_flow_reports_the_solved_result(self, box_stl: bytes, monkeypatch) -> None:
        monkeypatch.setattr(main_mod, "solver_available", lambda: True)
        monkeypatch.setattr(main_mod, "run_imported_cfd", lambda *a, **kw: _fake_result())
        clear_cfd_backend_cache()
        try:
            client = self._client()
            model_id = self._upload(client, box_stl)
            resp = client.post(
                "/api/cfd/3d",
                json={
                    "model_id": model_id,
                    "speed_kmh": 120.0,
                    "axes": "solver",
                    "level": "coarse",
                },
            )
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["job_id"]
            assert resp.json()["status"] == "queued"

            payload = None
            for _ in range(200):
                poll = client.get(f"/api/cfd/3d/{job_id}")
                assert poll.status_code == 200
                payload = poll.json()
                if payload["status"] in ("done", "error"):
                    break
                assert poll.status_code == 200
            assert payload is not None and payload["status"] == "done", payload.get(
                "detail"
            )
            result = payload["result"]
            assert result is not None
            assert result["speed_kmh"] == pytest.approx(120.0)
            assert result["cd"] == pytest.approx(0.8)
            assert result["provenance"]["solver_used"] == "openfoam_wsl"
            assert result["provenance"]["is_converged"] is True
            assert result["provenance"]["case_hash"] == "deadbeefdeadbeef"
            # positive Cl labelled as downforce
            assert result["downforce_N"] > 0.0
            assert result["lift_N"] == pytest.approx(-result["downforce_N"])
        finally:
            clear_cfd_backend_cache()
