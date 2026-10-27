# Codebase Audit Report (v0.1.0 Baseline)

- **Date:** 2026-10-27
- **Target:** `motorsport-aero-simulator` (v0.1.0)
- **Status:** Baseline audit for Step 0 (Reality Check)

---

## 1. Executive Summary

`motorsport-aero-simulator` is a prototype web application designed for motorsport aerodynamic evaluation and optimization. It consists of a FastAPI backend (Python 3.11) and a Vanilla JavaScript / Three.js frontend.

While the user interface and basic API workflows are functioning smoothly for presentation purposes, the aerodynamic physics, optimization routines, and data management contain critical placeholders, simplified proxies, and architectural flaws that must be addressed before the platform can serve as a scientific aero-ML platform.

---

## 2. Inventory & File Statistics

| Path | Lines | Role | Current Status |
|---|---|---|---|
| `backend/main.py` | 218 | FastAPI entrypoint & route handlers | Functional, but blocks event loop on optimization & has untyped query params |
| `backend/models/aero_config.py` | 94 | Pydantic data schemas | Functional, contains full-car formula setup |
| `backend/simulation/analytical.py` | 155 | Semi-empirical analytical aero equations | Working baseline; lift-slope + ground effect + drag models |
| `backend/simulation/panel_method.py` | 68 | Claimed VLM / panel method | **Deficient (Bug B1)**: 100% duplicate of `analytical.py` |
| `backend/simulation/cfd/openfoam_runner.py` | 94 | Claimed OpenFOAM CFD runner | **Deficient (Bug B2, B5)**: controlDict stub only; relabels analytical output |
| `backend/optimization/optimizer.py` | 105 | Optuna TPE Bayesian optimization | **Deficient (Bug B3)**: Single objective; pseudo-Pareto list slicing |
| `backend/visualization/mesh_io.py` | 193 | 3D mesh import (STL/OBJ) | Functional with bugs (B4, B7) |
| `backend/visualization/aero_field.py` | 166 | Surface Cp & streamline generation | Synthetic rule-based approximation |
| `backend/ai/advisor.py` | 206 | LLM setup advice (Anthropic/OpenAI) | Functional; lacks conversation memory |
| `frontend/index.html` | 666 | Main single-page UI | Rich layout; missing SRI / CDN integrity headers |
| `frontend/js/app.js` | 841 | Frontend controller & state management | Functional |
| `frontend/js/viewer3d.js` | 1109 | Three.js wind tunnel visualization | Rich visualization; needs robust coordinate normalization |
| `frontend/js/flowEngine.js` | 411 | Particle & streamline physics | Synthetic canvas / particle engine |
| `frontend/js/meshFlow.js` | 291 | Mesh surface flow particles | Synthetic surface flow |
| `frontend/js/engineerTools.js` | 129 | Telemetry & engineer readout helpers | Functional |
| `frontend/js/theme.js` | 18 | Color palette constants | Functional |
| `data/models/*.obj` | 21 files | Uploaded 3D car models | Leftover uncleaned large binary/ASCII mesh uploads (>10M lines total) |

---

## 3. Detailed Bug & Limitation Catalogue

### Bug B1: `panel_method.py` is identical to `analytical.py`
- **Claim:** File claims to implement a "1D horseshoe vortex lattice along the span for each wing element".
- **Evidence:** `backend/simulation/panel_method.py` lines 10-45 imports and calls `_wing_forces`, `_underbody_forces`, and `_body_drag` directly from `analytical.py`. There are zero panels, zero horseshoe vortices, and zero Biot-Savart calculations.
- **Severity:** High (misleading naming; fake solver).

### Bug B2: `openfoam_runner.py` relabels analytical output
- **Claim:** Runs OpenFOAM CFD to compute force coefficients.
- **Evidence:** `backend/simulation/cfd/openfoam_runner.py` line 71-72 (fallback) and line 91-92 (even if `simpleFoam` succeeded) simply invoke `simulate_analytical(config)` and change `result.method = "openfoam_cfd"`.
- **Severity:** High (misleading CFD results).

### Bug B3: Pseudo-Pareto tracking in `optimizer.py`
- **Claim:** Returns true Pareto front from multi-objective optimization.
- **Evidence:** `backend/optimization/optimizer.py` lines 65-72 simply appends up to 50 trials to a flat list, and line 103 returns `sorted(pareto, key=lambda x: x["score"], reverse=True)[:20]`. This is single-objective scalar ranking of the first 50 iterations, not non-dominated sorting (NSGA-II / true Pareto).
- **Severity:** Medium.

### Bug B4: `mesh_io.py` fails on non-STL/OBJ uploads
- **Claim:** Supports `.fbx`, `.glb`, `.gltf`, `.dae`, `.ply`.
- **Evidence:** `backend/visualization/mesh_io.py` lines 15-16 lists these formats in `SUPPORTED_EXTENSIONS`, but lines 159-163 uses `if ext == ".stl": ... else: parse_obj(data)`. Any binary GLB or FBX file is fed into `parse_obj` which fails.
- **Severity:** Medium.

### Bug B5: Placeholder OpenFOAM result parser
- **Claim:** Parses forces from OpenFOAM `postProcessing`.
- **Evidence:** `backend/simulation/cfd/openfoam_runner.py` line 90: `# Parse force coefficients from postProcessing (stub — extend with actual parser)`.
- **Severity:** Medium.

### Bug B6: Frontend canvas variable & event scoping
- **Claim:** Canvas scoping and event handlers in `app.js` and `viewer3d.js`.
- **Evidence:** Multiple separate references to `document.getElementById('aeroMapCanvas')` without lifecycle guard; potential null reference if tab switched before render.
- **Severity:** Low.

### Bug B7: Model upload disk cleanup missing
- **Claim:** Server manages uploaded model files cleanly.
- **Evidence:** `backend/visualization/mesh_io.py` lines 168-170 generates a random UUID and saves files permanently to `data/models/`. No cleanup or TTL mechanism exists, resulting in 21 accumulated multi-hundred-megabyte `.obj` files.
- **Severity:** Medium.

### Bug B8: Untyped `method` parameter in `/api/simulate`
- **Claim:** Endpoint validates simulation method.
- **Evidence:** `backend/main.py` line 55 accepts `method: str = "panel_vlm"`. Invalid strings trigger manual `HTTPException(400)` rather than Pydantic typed enum returning `422 Unprocessable Entity`.
- **Severity:** Low (API hygiene).

---

## 4. Architectural & Production Issues (P1 to P4)

### Issue P1: Optimizer blocks FastAPI event loop
- **Evidence:** `backend/main.py` lines 183-190 defines `async def simulate_and_optimize(...)` but synchronously calls `optimize(request)` on the main asyncio thread. This freezes all concurrent requests for several seconds during Optuna trials.
- **Solution:** Wrap CPU-bound calls in `asyncio.to_thread`.

### Issue P2: Unrestricted CORS
- **Evidence:** `backend/main.py` lines 35-40: `allow_origins=["*"]`.
- **Solution:** Restrict CORS origins to localhost / configured environments.

### Issue P3: Missing Subresource Integrity (SRI) for frontend CDN dependencies
- **Evidence:** External library loads in frontend lack SRI `integrity` hashes.
- **Solution:** Vendor Three.js or pin exact versions with SHA-384 / SHA-512 SRI hashes.

### Issue P4: Undocumented Coordinate System Conventions
- **Evidence:** `backend/visualization/aero_field.py` line 18 uses $x=\text{front}\to\text{rear}$, $y=\text{left}\to\text{right}$, $z=\text{ground}\to\text{top}$. Meanwhile motorsport convention (ISO 8855) is $x=\text{forward}$, $y=\text{lateral left}$, $z=\text{up}$. Three.js uses $y=\text{up}$, $z=\text{depth}$.
- **Solution:** Formally document conventions in `docs/COORDINATES.md` and define explicit transformation functions.

---

## 5. What Works vs What Crashes

- **Working:**
  - FastAPI server startup and `/api/health`.
  - `/api/simulate?method=analytical` (returns forces, coefficients, balance, lap delta).
  - `/api/simulate?method=panel_vlm` (returns same analytical forces with label).
  - `/api/optimize` (runs Optuna optimization over analytical physics).
  - 3D Three.js scene viewer with OrbitControls, ground plane, lighting, and synthetic particle streamlines.
  - Telemetry sliders and preset buttons (Silverstone, Monza, Monaco, Spa, Singapore).

- **Non-working / Stubs:**
  - True Vortex Lattice Method (`panel_vlm` is a stub).
  - OpenFOAM CFD execution and force parser.
  - Lap time simulation (linear delta proxy rather than point-mass lap sim).
  - Non-STL/OBJ file uploads.
  - Multi-objective Pareto frontier tracking.
