# Step 0 Reality Check Report

- **Date:** 2026-10-27
- **Baseline Git Tag:** `v0.1.0-baseline`
- **Status:** Complete — Awaiting Human Approval

---

## 1. Executive Summary

Step 0 (Reality Check) has been completed on the `motorsport-aero-simulator` repository. Every audit claim from `docs/AUDIT.md` was investigated and verified against the actual source code, the server was spun up, API endpoints were exercised, and golden regression tests (`tests/golden/test_baseline_golden.py`) were established and verified.

The core findings confirm that while the application provides a working interactive dashboard and basic analytical estimations, its advanced engineering claims (VLM panel method, OpenFOAM CFD execution, multi-objective Pareto front) are currently placeholders or relabeled analytical duplicates.

---

## 2. Audit Claims Verification Table

| Claim ID | Category | Claim Description | Verdict | Evidence (File, Line, Command) |
|---|---|---|---|---|
| **B1** | Aerodynamics | `panel_method.py` uses 1D horseshoe vortex lattice | **CONFIRMED** (Deficient) | `backend/simulation/panel_method.py:10-45`. Directly imports and calls `_wing_forces` and `_underbody_forces` from `analytical.py`. Outputs are byte-identical. Verified via `tests/golden/verify_b1.py`. |
| **B2** | CFD | `openfoam_runner.py` computes CFD force coefficients | **CONFIRMED** (Deficient) | `backend/simulation/cfd/openfoam_runner.py:71-73, 91-92`. Calls `simulate_analytical(config)` and relabels `result.method = "openfoam_cfd"`. |
| **B3** | Optimization | `optimizer.py` tracks true Pareto front | **CONFIRMED** (Deficient) | `backend/optimization/optimizer.py:65-72, 103`. Appends first 50 trials to a list and slices `[:20]` sorted by single-objective score. No non-dominated sorting. |
| **B4** | 3D IO | `mesh_io.py` supports FBX/GLB/PLY formats | **CONFIRMED** (Bug) | `backend/visualization/mesh_io.py:15-16, 159-163`. Lists non-STL/OBJ in supported set, but branches `else: parse_obj(data)`, causing parser crashes on binary GLB/FBX. |
| **B5** | CFD | Placeholder OpenFOAM result parser | **CONFIRMED** (Incomplete) | `backend/simulation/cfd/openfoam_runner.py:90`. Line comment `# Parse force coefficients from postProcessing (stub — extend with actual parser)`. |
| **B6** | Frontend | Canvas scoping / lifecycle issues in UI | **CONFIRMED** (Bug) | `frontend/js/app.js:324-354, 364, 493`. Repetitive ad-hoc canvas access without container lifecycle guard or null check if rendered before tab activation. |
| **B7** | Storage | Model uploads lack disk cleanup | **CONFIRMED** (Bug) | `backend/visualization/mesh_io.py:168-170`. Writes UUID-named files into `data/models/` without TTL or cleanup. 21 orphan `.obj` files (>10M lines) were accumulated. |
| **B8** | API | Untyped `method` query parameter in `/api/simulate` | **CONFIRMED** (Hygiene) | `backend/main.py:55, 62`. Parameter is untyped `method: str = "panel_vlm"`. Invalid strings trigger manual `HTTPException(400)` instead of FastAPI schema validation `422`. |
| **P1** | Concurrency | Optimizer blocks FastAPI event loop | **CONFIRMED** (Performance) | `backend/main.py:183-190`. `simulate_and_optimize` is `async def` but executes synchronous `optimize()` directly on main asyncio thread. |
| **P2** | Security | Unrestricted CORS policy | **CONFIRMED** (Security) | `backend/main.py:35-40`. Configured with `allow_origins=["*"]`. |
| **P3** | Security | CDN imports lack Subresource Integrity (SRI) | **CONFIRMED** (Security) | `frontend/index.html`. External script resources lack SHA integrity hashes. |
| **P4** | Docs/Physics | Undocumented coordinate conventions | **CONFIRMED** (Documentation) | `backend/visualization/aero_field.py:18` uses $x=\text{front}\to\text{rear}$, $y=\text{left}\to\text{right}$, $z=\text{ground}\to\text{top}$, conflicting with ISO 8855 ($x$ forward) and Three.js ($y$ up). |

---

## 3. Operational State & Diagnostics

| Component / Endpoint | Verification Method | Status | Notes |
|---|---|---|---|
| Server Startup | `uvicorn backend.main:app` | **WORKS** | Clean startup, binds to port 8000 |
| `GET /api/health` | `httpx.get` | **WORKS** | Returns `{status: ok, openfoam_available: false, ...}` |
| `POST /api/simulate` | `httpx.post` | **WORKS** | 200 OK for `analytical`, `panel_vlm`, fallback for `cfd` |
| `POST /api/optimize` | `httpx.post` | **WORKS** | 200 OK; runs 50 Optuna trials in ~1.2s |
| `POST /api/aero-field` | `httpx.post` | **WORKS** | Generates Cp map for 3D model vertices |
| Static UI (`/`) | HTTP GET + Browser inspection | **PARTIAL** | UI loads; Three.js imports need proper importmap / bundling |
| Git Baseline | `git tag v0.1.0-baseline` | **TAGGED** | Clean repository initialized with `.gitignore` |
| Golden Test Suite | `python -m tests.golden.test_baseline_golden` | **PASS** | 4 reference configs locked to $10^{-3}$ tolerance |

---

## 4. Module Classification (Keep / Wrap / Replace)

| Module / File | Classification | Rationale & Plan |
|---|---|---|
| `backend/simulation/analytical.py` | **KEEP, DEMOTE** | Keep intact as the low-fidelity reference model; protected by golden tests. |
| `backend/simulation/panel_method.py` | **RENAME, REPLACE** | Phase 0: rename method label to honest `analytical_enhanced`. Phase 1: replace with true VLM (Biot-Savart, image ground plane, Trefftz drag). |
| `backend/simulation/cfd/openfoam_runner.py` | **RENAME, REPLACE** | Phase 0: rename label to `cfd_stub`. Phase 2: replace with deterministic case builder, snappyHexMesh/blockMesh, and `forceCoeffs` parser. |
| `backend/optimization/optimizer.py` | **WRAP, REPLACE** | Phase 0: wrap in `asyncio.to_thread`. Phase 6: replace pseudo-Pareto with true multi-objective NSGA-II / MOTPE with regulation constraints. |
| `backend/visualization/mesh_io.py` | **KEEP, FIX** | Fix B4 (restrict or properly parse formats) and B7 (cleanup temporary model files). |
| `backend/visualization/aero_field.py` | **KEEP, EXTEND** | Document coordinate transformation (P4); connect to surrogate/VLM surface outputs in Phase 8. |
| `backend/ai/advisor.py` | **KEEP, EXTEND** | Phase 8: add conversation history and tool-calling for simulation/optimization. |
| `frontend/` (Three.js UI) | **KEEP, POLISH** | Retain layout and visualization; add SRI / importmap, SSE progress streaming (Phase 8), and configurable aero map grid. |

---

## 5. Reconciled Design Vector

The original `FormulaAeroConfig` exposes ~11 parameters. The ML study will focus on a **reduced, 8-dimensional core vector** with strict aerodynamic bounds. Parameters outside this core vector retain their standard analytical defaults and are explicitly labeled.

### 5.1 Core ML Design Vector ($\mathbf{x} \in \mathbb{R}^8$)

| Parameter Name | Target Pydantic Field | Physical Bounds | Units | Physics Influence |
|---|---|---|---|---|
| **$x_1$: Front Wing Main AoA** | `front_wing_main.angle_deg` | $[-2.0, 8.0]$ | $\mathrm{deg}$ | Front downforce, balance, vortex generation |
| **$x_2$: Front Wing Flap AoA** | `front_wing_flap.angle_deg` | $[2.0, 20.0]$ | $\mathrm{deg}$ | Front downforce, slot gap separation risk |
| **$x_3$: Rear Wing Main AoA** | `rear_wing_main.angle_deg` | $[0.0, 14.0]$ | $\mathrm{deg}$ | Rear downforce, main profile drag |
| **$x_4$: Rear Wing Flap AoA** | `rear_wing_flap.angle_deg` | $[5.0, 32.0]$ | $\mathrm{deg}$ | Total rear downforce, induced drag |
| **$x_5$: Front Ride Height ($h_f$)** | `ride_height_front_mm` | $[15.0, 50.0]$ | $\mathrm{mm}$ | Front wing ground effect, underbody entry |
| **$x_6$: Rear Ride Height ($h_r$)** | `ride_height_rear_mm` | $[25.0, 80.0]$ | $\mathrm{mm}$ | Rake, diffuser expansion ratio |
| **$x_7$: Diffuser Expansion Angle** | `diffuser_angle_deg` | $[8.0, 18.0]$ | $\mathrm{deg}$ | Underbody suction, stall threshold |
| **$x_8$: Freestream Speed ($V_\infty$)** | `speed_kmh` | $[80.0, 350.0]$ | $\mathrm{km/h}$ | Dynamic pressure $q$, Reynolds variations |

### 5.2 Fixed / Baseline Parameters (Excluded from Core Surrogate)

- `drs_active`: Handled as discrete operating mode (open/closed study).
- `front_wing_main.gurney_mm` / `rear_wing_main.gurney_mm`: Kept at standard race trim ($5\,\mathrm{mm}$ and $10\,\mathrm{mm}$ respectively).
- `front_wing_main.span_m`, `chord_m`, `rear_wing_main.span_m`, `chord_m`: Fixed by formula regulation envelopes.

---

## 6. Next Steps: Phase 0 Kickoff

With Step 0 completed and verified:
1. **User Approval Required:** Confirm acceptance of this Step 0 report to unlock Phase 0.
2. **Phase 0 Execution Plan:**
   - **P0.1**: Honest naming (`panel_vlm` $\to$ `analytical_enhanced`, `openfoam_cfd` $\to$ `cfd_stub`).
   - **P0.2**: Typed method Enum in Pydantic schemas (returning 422 for invalid values).
   - **P0.3**: `pyproject.toml`, ruff, mypy, pytest integration, fix B4, B6, B7.
   - **P0.4**: Wrap optimizer in `asyncio.to_thread` (fix P1), restrict CORS (fix P2).
   - **P0.5**: GitHub Actions CI workflow.
   - **P0.6**: Physics sanity tests (monotonicity with $V^2$, balance bounds).
   - **P0.7**: Document coordinate system convention in `docs/COORDINATES.md`.
