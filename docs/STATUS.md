# Project Status

Living document. The agent updates this at the end of every task. Only record facts verified by running code.

**Last updated:** 2026-10-27 by Agent (Step 0 completion)  
**Current phase:** Step 0 (Reality Check) — Completed, Awaiting Human Approval  
**Git baseline tag:** `v0.1.0-baseline` (created)  

---

## 1. Audit verification (Step 0)

| Claim in AUDIT.md | Verdict (CONFIRMED / WRONG / UNVERIFIABLE) | Evidence (file:line, command) |
|---|---|---|
| B1: `panel_method.py` identical to `analytical.py` | CONFIRMED | `backend/simulation/panel_method.py:10-45`, verified via `python -m tests.golden.verify_b1` |
| B2: `openfoam_cfd` relabels analytical | CONFIRMED | `backend/simulation/cfd/openfoam_runner.py:71-73, 91-92` calls `simulate_analytical` |
| B3: Pseudo-Pareto in `optimizer.py` | CONFIRMED | `backend/optimization/optimizer.py:65-72, 103` appends first 50 trials, sorts by score |
| B4: `mesh_io.py` fails on non-STL/OBJ | CONFIRMED | `backend/visualization/mesh_io.py:15-16, 159-163` routes non-STL directly to `parse_obj` |
| B5: Placeholder OpenFOAM parser | CONFIRMED | `backend/simulation/cfd/openfoam_runner.py:90` explicit stub comment |
| B6: Frontend canvas variable & scoping | CONFIRMED | `frontend/js/app.js:324, 364, 493` unguarded element queries |
| B7: Model upload cleanup missing | CONFIRMED | `backend/visualization/mesh_io.py:168-170` saves without TTL or eviction |
| B8: Untyped `method` query parameter | CONFIRMED | `backend/main.py:55` uses `method: str = "panel_vlm"`, manual 400 instead of 422 |
| P1: Optimizer blocks asyncio event loop | CONFIRMED | `backend/main.py:183-190` async endpoint calls sync `optimize()` directly |
| P2: Open CORS `allow_origins=["*"]` | CONFIRMED | `backend/main.py:35-40` wildcard origin configuration |
| P3: Three.js CDN imports lack SRI | CONFIRMED | `frontend/index.html` external imports lack integrity hashes |
| P4: Coordinate conventions undocumented | CONFIRMED | `backend/visualization/aero_field.py:18` conflicts with ISO 8855 and Three.js |

## 2. Module state

| Module | Decision (KEEP / WRAP / REPLACE / NEW) | State (works / partial / stub / missing) | Notes |
|---|---|---|---|
| `backend/simulation/analytical.py` | KEEP, DEMOTE | works | Low-fidelity baseline; locked with golden tests |
| `backend/simulation/panel_method.py` | RENAME, REPLACE | stub | Identical to analytical; rename in P0, replace in P1 |
| `backend/simulation/cfd/openfoam_runner.py` | RENAME, REPLACE | stub | Relabels analytical; rename in P0, replace in P2 |
| `backend/optimization/optimizer.py` | WRAP, REPLACE | partial | Single-objective Optuna works; wrap in thread in P0, replace in P6 |
| `backend/ai/advisor.py` | EXTEND | works | LLM advisor functional; lacks memory / tool calling |
| `backend/visualization/mesh_io.py` | KEEP, FIX | partial | Works for STL/OBJ; fix B4 & B7 in Phase 0 |
| `frontend/` (Three.js & UI) | KEEP, POLISH | works | UI functional; add SRI, importmap, SSE in Phase 8 |

## 3. Phase progress

| Phase | Status (not started / in progress / awaiting approval / accepted) | Accepted on |
|---|---|---|
| Step 0 Reality check | awaiting approval | — |
| 0 Hygiene and honest naming | not started | |
| 1 Real VLM | not started | |
| 2 CFD pipeline | not started | |
| 3 Dataset and surrogate | not started | |
| 4 Active learning and sensitivity | not started | |
| 5 Lap simulation | not started | |
| 6 Multi-objective optimization | not started | |
| 7 Correlation and calibration | not started | |
| 8 Integration and report | not started | |

## 4. Validation status

| ID | Case | Status (pending / failing / passing) | Result (number, command) | Reference source |
|---|---|---|---|---|
| Golden | Analytical baseline | passing | 4/4 passing (`python -m tests.golden.test_baseline_golden`) | `tests/golden/test_baseline_golden.py` |
| V1 | VLM elliptic planform | pending | | |
| V2 | VLM vs lifting-line | pending | | |
| V3 | VLM ground effect image system | pending | | |
| V4 | VLM multi-element wing | pending | | |
| V5 | OpenFOAM NACA 0012 polar | pending | | |
| V6 | OpenFOAM front wing in ground effect | pending | | |
| V7 | Surrogate cross-validation R² | pending | | |
| V8 | Multi-fidelity vs single-fidelity | pending | | |
| V9 | Active learning sample efficiency | pending | | |
| V10 | Lap sim sensitivity to downforce | pending | | |
| V11 | Pareto frontier hypervolume | pending | | |
| V12 | End-to-end telemetry correlation | pending | | |
