# Project Status

Living document. The agent updates this at the end of every task. Only record
facts verified by running code.

**Last updated:** 2026-10-28 by Agent (real OpenFOAM + UI/optimizer finalization)
**Current phase:** Phase 2 real CFD implemented; Phase 3 dataset/surrogate
re-measured with real CFD. Acceptance pending.
**Git baseline tag:** `v0.1.0-baseline`
**Working branch:** `fix/provenance-honesty`

> **Read this first:** OpenFOAM 1912 is installed inside WSL2 Ubuntu 22.04 on
> this machine. `detect_cfd_backend()` returns `wsl` and `/api/health` reports
> `solver_version="OpenFOAM 1912"`. The `cfd` method now executes a real 2D
> front-wing-section RANS solve; the total force is combined with empirical
> floor and rear-wing terms and is labelled `rans_hybrid`. It is not full-car
> CFD. If the solver is unavailable, the fallback is explicitly labelled
> `synthetic_model`; a real-run failure returns HTTP 503 rather than silently
> producing synthetic numbers.

---

## 1. Audit verification (Step 0)

| Claim in AUDIT.md | Verdict | Evidence |
|---|---|---|
| B1: `panel_method.py` identical to `analytical.py` | CONFIRMED (fixed in Phase 1) | 3D VLM in `backend/simulation/panel_method.py` |
| B2: `openfoam_cfd` relabels analytical | CONFIRMED (fixed) | Real OpenFOAM WSL path; `rans_hybrid` provenance; 503 on failure |
| B3: Pseudo-Pareto in `optimizer.py` | CONFIRMED (fixed) | `_pareto_front()` filters non-dominated drag/downforce points |
| B4: `mesh_io.py` fails on non-STL/OBJ | CONFIRMED (fixed) | Extension validation |
| B5: Placeholder OpenFOAM parser | CONFIRMED (fixed for real runs) | `backend/simulation/cfd/parser.py`; exercised by V5/V6 |
| B6: Frontend canvas variable & scoping | CONFIRMED (fixed) | `app.js` null guards |
| B7: Model upload cleanup missing | CONFIRMED (fixed) | `mesh_io.py` TTL eviction |
| B8: Untyped `method` parameter | CONFIRMED (fixed) | `SimulationMethod` enum -> 422 |
| P1: Optimizer blocks event loop | CONFIRMED (fixed for one-shot) | `/api/simulate-and-optimize` uses `asyncio.to_thread`; `/api/optimize` runs in FastAPI's worker pool |
| P2: Open CORS | CONFIRMED (fixed) | `main.py` origin allowlist |
| P3: Three.js without SRI | PARTIAL | Importmap pinned; no SRI |
| P4: Coordinate conventions undocumented | CONFIRMED (fixed) | `docs/COORDINATES.md` |
| **NEW:** HF dataset labelled OpenFOAM | CONFIRMED (fixed) | 100 synthetic rows backed up; 28 real rows now in `aero_surrogate_v1.parquet` |
| **NEW:** V4 contradicts its cited literature | CONFIRMED | Model 2.585 vs published 1.34 at h/c=0.15 |

---

## 2. Module state

| Module | Decision | State | Notes |
|---|---|---|---|
| `backend/simulation/analytical.py` | KEEP, DEMOTE | works | `fidelity="empirical"`; sign handling corrected |
| `backend/simulation/panel_method.py` | KEEP | works | 3D VLM; `fidelity="low_fidelity_vlm"` |
| `backend/simulation/vlm/` | KEEP | works | Horseshoe Biot-Savart, image ground, Trefftz drag |
| `backend/simulation/cfd/` | REAL | works | STL, blockMesh, snappyHexMesh, simpleFoam, parsers, cache |
| `backend/simulation/cfd/openfoam_runner.py` | REAL + HYBRID | works | 2D section + empirical floor/rear; manual API path |
| `backend/ml/dataset.py` | REAL HF | works | 28 real WSL points; per-point failure rows; medium mesh |
| `backend/ml/surrogate.py` | VALIDATION ONLY | works | GP + AR1; not wired into the live UI |
| `backend/optimization/optimizer.py` | FIXED | works | Normalised balanced objective; true drag/downforce Pareto front |
| `backend/ai/advisor.py` | KEEP | works | No memory; optimization advice now describes the optimized setup, not the baseline |
| `backend/visualization/mesh_io.py` | KEEP | works | Upload validation + TTL cleanup |
| `frontend/` | REDESIGNED | works | Method tabs, solver pill, manual CFD, progress, case card, provenance |

---

## 3. Phase progress

| Phase | Status | Notes |
|---|---|---|
| Step 0 Reality check | accepted 2026-10-27 | |
| 0 Hygiene and honest naming | accepted 2026-10-27 | |
| 1 Real VLM | accepted 2026-10-27 | V4 external reference still fails |
| 2 CFD pipeline | **implemented with real solver; measured limitations; acceptance pending** | V5 NOT DEMONSTRATED, V6 Part A partly fails, Part B trend-only |
| 3 Dataset and surrogate | **real 28-point HF dataset; V7/V8 targets partly unmet** | 4/28 solver-converged points |
| 4 Active learning and sensitivity | not started | |
| 5 Lap simulation | not started | |
| 6 Multi-objective optimization | partial | True Pareto + normalized balance; no 3D Pareto/constraints |
| 7 Correlation and calibration | not started | |
| 8 Integration and report | in progress | UI and docs updated |

---

## 4. Validation status

Measured on 2026-10-28 with `.venv\Scripts\python.exe validation\<case>.py`.
The V5/V6/V7/V8 runs used the real WSL OpenFOAM 1912 backend.

| ID | Case | Status | Result | Reference |
|---|---|---|---|---|
| Golden | Analytical baseline | passing | 4/4 | internal |
| V1 | VLM elliptic planform | passing | span efficiency e = 1.016, Cl error 2.7 % | Prandtl (1918), Anderson (2001) |
| V2 | VLM aspect-ratio sweep | passing | max error 4.8 % vs Helmbold, AR 2-12 | Helmbold (1942) |
| V3 | VLM grid convergence | passing | monotonic, GCI < 1.5 % over 4 grids | Roache (1998) |
| V4 | Ground-effect trend | **FAILS external reference** | model 1.120, 1.313, 1.618, 2.055, 2.585 vs published 1.00, 1.08, 1.18, 1.28, 1.34 -> **1.93x at h/c=0.15** | Zerihan & Zhang (2000) |
| V5 | CFD mesh independence | **NOT DEMONSTRATED** | coarse 8,215 cells Cl 0.45158; medium 21,078 Cl 0.40384; fine 43,971 Cl 0.34433 (unconverged). r21=1.444, r32=1.602, p=0.500, GCI fine 107.049 %, asymptotic ratio 0.432 | Roache (1998), Celik et al. (2008) |
| V6A | NACA 0012 polar | **measured; tight targets not met** | Cl 0.0615/0.4038/0.6925/0.8967/0.7628 at 0/4/8/10/12 deg; Cd 0.01128/0.01894/0.03838/0.05913/0.10052; Cd0 delta 0.00308; lift slope 0.0789/deg vs 0.1081/deg (27.05 % error) | McCroskey (1987), Ladson (1988) |
| V6B | Ground-effect trend | **trend only** | Cl ratio 1.187 / 1.292 / 1.509 / 1.917 / 2.439 at h/c 1.00/0.50/0.30/0.20/0.15; direction matches, geometry is an inverted NACA 4412 section, not Zerihan/Zhang | Zerihan & Zhang (2000), trend only |
| V7 | Surrogate cross-validation | **targets partly unmet** | real 28 rows, 4 converged. AR1 Cl R² 0.6879, Cd R² 0.9605; MAPE 7.08 % / 0.92 %; 95 % coverage 82.1 % / 89.3 % | Kennedy & O'Hagan (2000) |
| V8 | Multi-fidelity vs single-fidelity | **targets partly unmet** | train 19 / test 5. Budgets N=6/12/19: SF RMSE 0.1294/0.1576/0.1933, AR1 0.1050/0.1566/0.1765, reduction 18.9 %/0.6 %/8.7 %; 95 % coverage 100 %, width 0.5833; test R² 0.5751 | Forrester et al. (2007) |
| V9 | Active learning sample efficiency | pending | | |
| V10 | Lap-sim analytic cases | pending | | |
| V11 | Pareto candidates re-verified in CFD | pending | | |
| V12 | Calibration recovers injected bias | pending | | |

V7/V8 now measure the surrogate against **real OpenFOAM hybrid rows**, not the
old synthetic generator. They are still surrogate-machinery evidence, not a
full-car aerodynamic validation. The scripts print their own caveats and do not
force a pass.

---

## 5. Stubs, approximations, and known limitations

- **The real CFD tier is a 2D section simplification.** The front-wing section
  is an inverted NACA 4412 with a moving ground plane; floor and rear-wing
  contributions are empirical. The total is labelled `rans_hybrid`, not
  `rans`.
- **Mesh independence is not demonstrated.** `snappyHexMesh` produces severe
  non-orthogonal faces; the fine NACA 0012 case is force-unconverged and GCI is
  107 %. A mesh-family rebuild is required before claiming mesh independence.
- **V6 Part A uses fully turbulent RANS at Re=3.0M.** It overpredicts drag and
  underpredicts lift near stall; the old tight tolerances are not met.
- **V6 Part B is trend-only.** The modelled section is not the Zerihan/Zhang
  multi-element wing, so absolute ratios are not a geometry validation.
- **V4 still misses the external reference.** The VLM ground-effect augmentation
  is roughly 1.93x the published value at h/c=0.15.
- **The 28-point HF dataset is small and mostly unconverged.** 4/28 points have
  `convergence_flag=True`; the rest are real solves that hit the force-stability
  threshold. Rows are retained with their measured status, not relabelled.
- **V7/V8 targets are partly unmet.** Reported above. The dataset is too small
  and the hybrid/VLM relationship is not smooth enough for the historical
  R² > 0.90 / MAPE < 5 % targets.
- **Optimizer "balanced" is a weighted score, not a physical optimum.**
  Downforce, efficiency, and balance are normalised (0.35 / 0.25 / 0.40) before
  scoring. A 30-trial run found 3,239.8 N downforce, L/D 1.34, and 44.1 % front
  balance against a 42 % target; results vary with the sampled range. The API
  reports the sampled balance range and whether the target was reachable. The
  Pareto front is a real non-dominated drag/downforce set. The UI caps
  optimization at 30 VLM trials so the request stays interactive.
- **Advisor optimizer text was corrected.** The rule-based advisor previously
  described baseline efficiency/balance while showing optimized recommendations,
  and printed the fixed-backend lap proxy as an "impact". It now describes the
  optimized forces and explicitly labels the improvement as a normalized
  objective score, not a physical or lap-time percentage.
- **Lap-time delta is a relative proxy.** The frontend compares only runs from
  the same fidelity family (VLM vs VLM, CFD vs CFD), or a saved baseline. It is
  not a predicted lap time. The backend still carries a crude fixed-baseline
  proxy for API consumers.
- **Dashboard geometry:** at a measured 1000x700 viewport the header is 80 px,
  provenance banner 75 px, viewer 215 px, and bottom panel 320 px (internally
  scrolling); at 1620x1165 the viewer is 727 px. At widths <= 1100 px the bottom
  panel is capped at 320 px.
- **The aero map is fixed at 5x5** and always uses the fast VLM path, even when
  CFD is selected.
- **The surrogate is not wired into the API or UI.** No uncertainty view yet.
- **`aeroField.js` Cp field is a heuristic normal-based function**, not CFD.
- **Advisor has no conversation memory and no tool calling.** Quoted numbers come
  from the request payload.
- **Heavy jobs:** a medium real CFD run took 45.8 s; the 28-point dataset took
  26.9 min serial. A 100-trial API optimization took roughly 100 s on this
  machine; the UI uses 30 trials (about 50 s) and shows an elapsed-time counter.
- **Synthetic fallback remains only for missing solvers.** It is labelled
  `synthetic_model` / `cfd_synthetic_model`; a real-run failure is a 503 and a
  failed UI card.

---

## 6. Open questions for the human

1. Is the 2D-section + empirical floor/rear hybrid acceptable as the Phase 2
   CFD deliverable, or is full-car/multi-element CFD required before acceptance?
2. Should the VLM/CFD front-rear balance split be calibrated against external
   data before Phase 4?
3. Which digitized external ground-effect dataset should replace the V4 target?
4. Should the surrogate be wired into the live optimizer, or remain a validation
   artifact until the dataset is larger and more converged?
5. Track centerline data for Phase 5 (no invented coordinates).

## 7. Reference data still needed

- Digitized Zerihan & Zhang (2000) ground-effect augmentation at h/c = 1.0 ... 0.15.
- A validated full-car or multi-element CFD benchmark with forces and mesh details.
- Track centerline data for Phase 5.

---

## 8. Change log

| Date | Task | Summary | Commit |
|---|---|---|---|
| 2026-10-27 | Step 0 | Audit verification, tag, golden tests, docs | `8e17fc0` |
| 2026-10-27 | Phase 0 | Hygiene, typed enum, CORS, CI, coordinate doc | `3fc2691` |
| 2026-10-27 | Phase 1 | 3D VLM, image ground, V1-V4 | `fbba6d2` |
| 2026-10-27 | Phase 2 | CFD pipeline scaffolding, GCI, V5-V6 | `899d3fe` |
| 2026-10-28 | Phase 3 checkpoint | Surrogate pipeline committed as-is | `f6e939d` |
| 2026-10-28 | Provenance fix | Real solver labels, no fake logs, synthetic dataset regenerated, UI badges | `5a1c681` |
| 2026-10-28 | Real CFD + UI | OpenFOAM 1912 WSL, real V5/V6 runs, manual CFD UI, 503 path, 28-point real dataset, V7/V8 re-run, true Pareto, layout redesign, 30-trial UI, consistent optimizer advice; 27 fast + 4 slow tests pass, ruff/mypy clean | pending |
