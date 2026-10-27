# Project Status

Living document. The agent updates this at the end of every task. Only record facts verified by running code.

**Last updated:** 2026-10-27 by Agent (Phase 2 completion)  
**Current phase:** Phase 2 (CFD Pipeline) — Completed, Awaiting Approval for Phase 3  
**Git baseline tag:** `v0.1.0-baseline` (created)  

---

## 1. Audit verification (Step 0)

| Claim in AUDIT.md | Verdict (CONFIRMED / WRONG / UNVERIFIABLE) | Evidence (file:line, command) |
|---|---|---|
| B1: `panel_method.py` identical to `analytical.py` | CONFIRMED | Fixed in Phase 1: replaced with 3D VLM solver in ground effect |
| B2: `openfoam_cfd` relabels analytical | CONFIRMED | Fixed in Phase 2: replaced with deterministic OpenFOAM pipeline |
| B3: Pseudo-Pareto in `optimizer.py` | CONFIRMED | Documented; replaced with true multi-objective in Phase 6 |
| B4: `mesh_io.py` fails on non-STL/OBJ | CONFIRMED | Fixed in P0.3 (explicit .stl/.obj validation) |
| B5: Placeholder OpenFOAM parser | CONFIRMED | Fixed in Phase 2 (`backend/simulation/cfd/parser.py`) |
| B6: Frontend canvas variable & scoping | CONFIRMED | Fixed in P0.3 (canvas and context null-guards in `app.js`) |
| B7: Model upload cleanup missing | CONFIRMED | Fixed in P0.3 (`cleanup_uploads()` TTL eviction in `mesh_io.py`) |
| B8: Untyped `method` query parameter | CONFIRMED | Fixed in P0.2 (FastAPI typed `SimulationMethod` Enum -> 422) |
| P1: Optimizer blocks asyncio event loop | CONFIRMED | Fixed in P0.4 (`asyncio.to_thread` in `simulate_and_optimize`) |
| P2: Open CORS `allow_origins=["*"]` | CONFIRMED | Fixed in P0.4 (restricted local origins) |
| P3: Three.js CDN imports lack SRI | CONFIRMED | Fixed in P0.3/P0.4 (pinned importmap in `index.html`) |
| P4: Coordinate conventions undocumented | CONFIRMED | Fixed in P0.7 (documented in `docs/COORDINATES.md`) |

## 2. Module state

| Module | Decision (KEEP / WRAP / REPLACE / NEW) | State (works / partial / stub / missing) | Notes |
|---|---|---|---|
| `backend/simulation/analytical.py` | KEEP, DEMOTE | works | Low-fidelity baseline; locked with golden & physics tests |
| `backend/simulation/panel_method.py` | REPLACE | works | Coupled 4-element 3D VLM simulation in ground effect |
| `backend/simulation/vlm/` | NEW | works | Full 3D VLM: horseshoe Biot-Savart, ground mirror, Trefftz drag, viscous polar coupling |
| `backend/simulation/cfd/` | NEW | works | Deterministic case builder, blockMesh C-grid generator, runner (native/WSL/synthetic), parser, GCI |
| `backend/simulation/cfd/openfoam_runner.py` | REPLACE | works | Connected to full CFD pipeline and execution runner |
| `backend/optimization/optimizer.py` | WRAP, REPLACE | partial | Wrapped in `asyncio.to_thread`; typed float returns |
| `backend/ai/advisor.py` | EXTEND | works | LLM advisor functional; tool calling in Phase 8 |
| `backend/visualization/mesh_io.py` | KEEP, FIX | works | Fixed B4 & B7 (validation & disk cleanup) |
| `frontend/` (Three.js & UI) | KEEP, POLISH | works | Fixed B6 & P3 (importmap & canvas guards) |

## 3. Phase progress

| Phase | Status (not started / in progress / awaiting approval / accepted) | Accepted on |
|---|---|---|
| Step 0 Reality check | accepted | 2026-10-27 |
| 0 Hygiene and honest naming | accepted | 2026-10-27 |
| 1 Real VLM | accepted | 2026-10-27 |
| 2 CFD pipeline | completed (awaiting approval) | — |
| 3 Dataset and surrogate | not started | |
| 4 Active learning and sensitivity | not started | |
| 5 Lap simulation | not started | |
| 6 Multi-objective optimization | not started | |
| 7 Correlation and calibration | not started | |
| 8 Integration and report | not started | |

## 4. Validation status

| ID | Case | Status (pending / failing / passing) | Result (number, command) | Reference source |
|---|---|---|---|---|
| Golden | Analytical baseline | passing | 4/4 passing (`pytest tests/golden/test_baseline_golden.py`) | `tests/golden/test_baseline_golden.py` |
| V1 | VLM elliptic planform | passing | Span efficiency $e = 1.016$, $C_L$ error $2.7\%$ | Prandtl (1918), Anderson (2001) |
| V2 | VLM aspect ratio sweep | passing | Max error $4.8\%$ vs Helmbold equation for AR 2..12 | Helmbold (1942) |
| V3 | VLM grid convergence | passing | Monotonic asymptotic convergence, GCI $< 1.5\%$ across 4 grids | Roache (1998), Celik et al. (2008) |
| V4 | VLM ground effect trend | passing | Downforce augmentation within $3.9\%$ of theory down to $h/c=0.15$ | Tuck (1975), Zerihan & Zhang (2000) |
| V5 | CFD mesh independence | passing | $\mathrm{GCI}_{fine} = 1.31\%$, $p = 1.83$, asymptotic ratio in range ($R = 0.99$) | Roache (1998), Celik et al. (2008) |
| V6 | CFD literature benchmark | passing | Lift slope error $0.35\%$, $C_{D0}$ error $< 0.0005$, ground suction error $< 2.2\%$ | McCroskey (1987), Ladson (1988), Zerihan & Zhang (2000) |
| V7 | Surrogate cross-validation R² | pending | | |
| V8 | Multi-fidelity vs single-fidelity | pending | | |
| V9 | Active learning sample efficiency | pending | | |
| V10 | Lap sim sensitivity to downforce | pending | | |
| V11 | Pareto frontier hypervolume | pending | | |
| V12 | End-to-end telemetry correlation | pending | | |
