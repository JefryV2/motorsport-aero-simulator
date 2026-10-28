# Project Status

Living document. The agent updates this at the end of every task. Only record facts verified by running code.

**Last updated:** 2026-10-28 by Agent (provenance correction)
**Current phase:** Phase 3 (Dataset & Surrogate) — code complete, evidence re-labelled, awaiting approval
**Git baseline tag:** `v0.1.0-baseline` (created)
**Working branch:** `fix/provenance-honesty`

> **Read this first:** on this machine `detect_cfd_backend()` returns `synthetic`.
> There is no OpenFOAM binary, so V5, V6 and the high-fidelity column of the
> Phase 3 dataset are **not CFD**. They are a deterministic calibrated model.
> Every result is now labelled accordingly in code, data, API and UI.

---

## 1. Audit verification (Step 0)

| Claim in AUDIT.md | Verdict | Evidence (file:line, command) |
|---|---|---|
| B1: `panel_method.py` identical to `analytical.py` | CONFIRMED (fixed in Phase 1) | Replaced by 3D VLM in `backend/simulation/panel_method.py` |
| B2: `openfoam_cfd` relabels analytical | CONFIRMED (partially fixed) | Pipeline exists, but no solver runs here; now tagged `cfd_synthetic_model` |
| B3: Pseudo-Pareto in `optimizer.py` | CONFIRMED | `optimizer.py:67` still appends the first 50 trials; Phase 6 |
| B4: `mesh_io.py` fails on non-STL/OBJ | CONFIRMED (fixed) | `mesh_io.py` extension validation |
| B5: Placeholder OpenFOAM parser | CONFIRMED (fixed for real runs) | `backend/simulation/cfd/parser.py`; unused on synthetic path |
| B6: Frontend canvas variable & scoping | CONFIRMED (fixed) | `app.js` null guards |
| B7: Model upload cleanup missing | CONFIRMED (fixed) | `mesh_io.py:152` `cleanup_uploads()` TTL eviction |
| B8: Untyped `method` parameter | CONFIRMED (fixed) | `SimulationMethod` enum -> 422 |
| P1: Optimizer blocks event loop | CONFIRMED (fixed in one route) | `main.py:236-239` uses `asyncio.to_thread`; `/api/optimize` still sync |
| P2: Open CORS | CONFIRMED (fixed) | `main.py:46-59` origin allowlist |
| P3: Three.js without SRI | PARTIAL | Importmap pinned, but duplicate map removed only now; no SRI |
| P4: Coordinate conventions undocumented | CONFIRMED (fixed) | `docs/COORDINATES.md` |
| **NEW:** HF dataset labelled OpenFOAM | CONFIRMED | 100/100 rows were `OpenFOAM-v2406`, `converged=True`; now corrected |
| **NEW:** V4 contradicts its cited literature | CONFIRMED | Model 2.585 vs published 1.34 at h/c=0.15 |

---

## 2. Module state

| Module | Decision | State | Notes |
|---|---|---|---|
| `backend/simulation/analytical.py` | KEEP, DEMOTE | works | `fidelity="empirical"`; sign bug still present (`abs(lift)`) |
| `backend/simulation/panel_method.py` | REPLACE | works | Real 3D VLM; `fidelity="low_fidelity_vlm"`; sign bug in `vlm/forces.py:194` |
| `backend/simulation/vlm/` | NEW | works | Horseshoe Biot–Savart, image ground, Trefftz drag |
| `backend/simulation/cfd/` | NEW | partial | Case builder + parser + GCI are real; execution tier is synthetic |
| `backend/simulation/cfd/openfoam_runner.py` | REPLACE | partial | `run_cfd_with_provenance()` now reports the run that actually happened |
| `backend/ml/dataset.py` | NEW | works | Provenance columns derived from the run, not hardcoded |
| `backend/ml/surrogate.py` | NEW | works | GP + AR1; not wired into API or UI |
| `backend/optimization/optimizer.py` | WRAP, REPLACE | partial | Pseudo-Pareto; balance target not met (Phase 6) |
| `backend/ai/advisor.py` | EXTEND | works | No memory, no tool calling (Phase 8) |
| `backend/visualization/mesh_io.py` | KEEP, FIX | works | Upload validation + TTL cleanup |
| `frontend/` | KEEP, POLISH | works | Fidelity badge, provenance banner, method note; duplicate importmap removed |

---

## 3. Phase progress

| Phase | Status | Accepted on |
|---|---|---|
| Step 0 Reality check | accepted | 2026-10-27 |
| 0 Hygiene and honest naming | accepted (naming residue remains) | 2026-10-27 |
| 1 Real VLM | accepted (V4 reference claim corrected here) | 2026-10-27 |
| 2 CFD pipeline | **awaiting approval — real solver never executed** | — |
| 3 Dataset and surrogate | in progress — awaiting approval | — |
| 4 Active learning and sensitivity | not started | |
| 5 Lap simulation | not started | |
| 6 Multi-objective optimization | not started | |
| 7 Correlation and calibration | not started | |
| 8 Integration and report | not started | |

Phase 2 and 3 cannot be marked accepted until either (a) OpenFOAM is installed
and V5/V6/the campaign are re-run, or (b) an ADR explicitly accepts
"Tier A deferred / synthetic-only" as a documented limitation.

---

## 4. Validation status

Measured on 2026-10-28 with `.venv\Scripts\python.exe validation\<case>.py`.

| ID | Case | Status | Result | Reference source |
|---|---|---|---|---|
| Golden | Analytical baseline | passing | 4/4 (`pytest tests/golden/test_baseline_golden.py`) | internal |
| V1 | VLM elliptic planform | passing | span efficiency e = 1.016, Cl error 2.7 % | Prandtl (1918), Anderson (2001) |
| V2 | VLM aspect-ratio sweep | passing | max error 4.8 % vs Helmbold, AR 2–12 | Helmbold (1942) |
| V3 | VLM grid convergence | passing | monotonic, GCI < 1.5 % over 4 grids | Roache (1998) |
| V4 | Ground-effect trend | **passing vs internal target / FAILS external** | model 1.120, 1.313, 1.618, 2.055, 2.585 vs hardcoded 1.12, 1.28, 1.56, 2.00, 2.55; published Zerihan/Zhang is 1.00, 1.08, 1.18, 1.28, 1.34 → **1.93× at h/c=0.15** | Zerihan & Zhang (2000) |
| V5 | CFD mesh independence | **invalid on this machine** | `detect_cfd_backend() == "synthetic"`; "refinement" replays a hardcoded `grid_bias` table, no mesh is generated | Roache (1998) — not applicable until solver exists |
| V6 | CFD literature benchmark | **invalid on this machine** | Compares the synthetic formula against constants fitted to it; circular | McCroskey (1987), Ladson (1988) |
| V7 | Surrogate cross-validation | passing **against synthetic generator** | AR1: Cl R² 0.9953, Cd R² 0.9993; MAPE 1.36 % / 0.10 %; coverage 99 % / 100 %; SF GP: 0.9978 / 0.9995 | Kennedy & O'Hagan (2000) |
| V8 | Multi-fidelity vs single-fidelity | **mixed** | N=15: +25.9 % RMSE; N=25: −13.3 %; N=45: −16.6 %; N=70: −35.6 %; 95 % coverage 100 %; test R² 0.9918 | Forrester et al. (2007) |
| V9 | Active learning sample efficiency | pending | | |
| V10 | Lap-sim analytic cases | pending | | |
| V11 | Pareto candidates re-verified in CFD | pending | | |
| V12 | Calibration recovers injected bias | pending | | |

V7/V8 measure the surrogate against a deterministic synthetic generator. They
are valid evidence about the **ML machinery** and are **not** evidence about
aerodynamic accuracy. The scripts now print that warning automatically.

---

## 5. Stubs, approximations, and known limitations

Everything below is not the real thing and is labelled as such in code and UI.

- **CFD execution is synthetic on this machine.** `detect_cfd_backend()` → `synthetic`.
  `backend/simulation/cfd/runner.py` fits coefficients; it does not mesh or solve.
  Case directories now contain `SYNTHETIC_MODEL.json` instead of a fake `log.simpleFoam`.
- **CFD mesh has no airfoil geometry.** `mesh_builder.py` declares an `airfoil`
  patch but blockMesh never inserts the section.
- **Reference-area mismatch in the CFD path.** `case_builder.py:339` writes
  `Aref = chord * 0.1`, while `runner.py:220` dimensionalises with `sref = chord * 1.0`.
- **VLM/analytical sign handling.** `vlm/forces.py:194` and `analytical.py` use
  `abs()`-style logic, so a positive-lift wing can still report positive downforce.
- **V4 target data is not the cited literature.** Targets were set from the model,
  not from `validation/references.md`.
- **Optimizer "balanced" objective does not hit its target.** Downforce term
  (×0.6, ~1200) dominates the balance penalty (×5, ~100).
- **"Pareto front" is the first 50 random trials**, not a nondominated set.
- **Lap-time delta is a linear proxy**, computed client-side in `app.js`.
- **Aero map is fixed at 5×5** in the UI and runs live simulations.
- **Surrogate is not wired into the API or UI.** No uncertainty view yet.
- **`aeroField.js` Cp field is a heuristic normal-based function**, not CFD.
- **Advisor has no conversation memory and no tool calling.** Quoted numbers come
  from the request payload, not from a fresh simulation.
- **No plots, no DVC/MLflow, no `experiments/` folder.** Learning curves are text only.
- **Heavy jobs are synchronous** except `/api/simulate-and-optimize`; a 40-trial
  optimize took 74.6 s and blocks the request.

---

## 6. Open questions for the human

1. Can OpenFOAM be installed (native or WSL) on the target machine? This unblocks Phase 2.
2. If not, do we accept a documented "synthetic-only" limitation (ADR) and move to Phase 4?
3. Which published ground-effect dataset should V4 use, and can you supply digitized values?
4. Should `panel_vlm` remain as a public alias, or be removed now that it equals `analytical_enhanced`?

## 7. Reference data still needed from the human

- Digitized Zerihan & Zhang (2000) ground-effect augmentation at h/c = 1.0 … 0.15.
- A validated 2D/3D CFD benchmark case with forces and mesh details.
- Track centerline data for Phase 5 (no invented coordinates).

---

## 8. Change log

| Date | Task | Summary | Commit |
|---|---|---|---|
| 2026-10-27 | Step 0 | Audit verification, tag, golden tests, docs | `8e17fc0` |
| 2026-10-27 | Phase 0 | Hygiene, typed enum, CORS, CI, coordinate doc | `3fc2691` |
| 2026-10-27 | Phase 1 | 3D VLM, image ground, V1–V4 | `fbba6d2` |
| 2026-10-27 | Phase 2 | CFD pipeline scaffolding, GCI, V5–V6 | `899d3fe` |
| 2026-10-28 | Phase 3 checkpoint | Surrogate pipeline committed as-is | `f6e939d` |
| 2026-10-28 | Provenance fix | Real solver labels, no fake logs, dataset regenerated, UI badges | pending |
