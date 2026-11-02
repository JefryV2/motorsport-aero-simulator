# Project Status

Living document. The agent updates this at the end of every task. Only record
facts verified by running code.

**Last updated:** 2026-11-02 by Agent (analytic smoke realism + flow-field gate)
**Current phase:** Phase 2 real CFD implemented (O-grid family); Phase 3
dataset/surrogate re-measured on the validated envelope - V7 all targets met,
V8 2 of 3; Phase 4 measured - **V9 targets NOT met (1 of 3 checks)**, P4.3
Sobol sensitivity passing and reproducible. The live wind-tunnel smoke was
upgraded (coherent turbulence, floor boundary layer, wake, tip vortices) and is
labelled in the UI as an analytic visualization. Acceptance for Phases 2-4
pending.
**Git baseline tag:** `v0.1.0-baseline`
**Working branch:** `master` (Phase 2/3/4 work fast-forwarded from
`fix/provenance-honesty`; that branch is now fully contained in master)

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
| **NEW:** HF dataset labelled OpenFOAM | CONFIRMED (fixed) | Snappy-family 28 rows (4/28 converged) superseded 2026-11-01 by the O-grid envelope run in `aero_surrogate_v1.parquet`: 28 real rows, 13 converged, 0 diverged; snappy set committed as `_snappy_backup.parquet` |
| **NEW:** V4 contradicts its cited literature | CONFIRMED (reporting fixed 2026-10-29) | Model 2.585 vs published 1.34 at h/c=0.15. V4 is now split into a potential-flow self-check (pass) and a PENDING external comparison. |

---

## 2. Module state

| Module | Decision | State | Notes |
|---|---|---|---|
| `backend/simulation/analytical.py` | KEEP, DEMOTE | works | `fidelity="empirical"`; sign handling corrected |
| `backend/simulation/panel_method.py` | KEEP | works | 3D VLM; `fidelity="low_fidelity_vlm"` |
| `backend/simulation/vlm/` | KEEP | works | Horseshoe Biot-Savart, image ground, Trefftz drag |
| `backend/simulation/cfd/` | REAL | works | Body-fitted O-grid polyMesh, simpleFoam, checkMesh + y+ parsers, result cache |
| `backend/simulation/cfd/openfoam_runner.py` | REAL + HYBRID | works | 2D section + empirical floor/rear; manual API path |
| `backend/ml/dataset.py` | REAL HF | works | 28 real WSL points sampled inside the V6-validated envelope (incidence <= 8 deg, h/c 0.15-0.32); per-point failure rows; medium mesh |
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
| 2 CFD pipeline | **implemented with real solver; mesh independence demonstrated on the O-grid family** | V5 PASSED (GCI 0.615 %); V6 re-measured on the O-grid family, see section 4 |
| 3 Dataset and surrogate | **real 28-point O-grid envelope dataset; V7 all targets met, V8 2 of 3** | 13/28 solver-converged, 0 diverged; V7 Cl R² 0.9952 / Cd R² 0.9907; V8 small-budget (N=6) AR1 FAIL |
| 4 Active learning and sensitivity | **measured - V9 targets NOT met (1 of 3); P4.3 Sobol passing** | AL beats random (+17.6 % RMSE at N=13) but loses to maximin (-16.8 %) and only ties random on strict-oracle yield (5.0 vs 5.0); Sobol indices stable to N=512, ride height dominates both outputs |
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
| V2 | VLM aspect-ratio sweep | passing with documented low-AR caveat | AR>=4: max error 3.2 % (tolerance 3.5 %); AR=2: 15.4 % (tolerance 16 %, Helmbold is not a low-AR reference) | Helmbold (1942) |
| V3 | VLM grid convergence | passing | monotonic, GCI < 1.5 % over 4 grids | Roache (1998) |
| V4a | Ground-effect trend vs potential-flow image system | passing | 1.120, 1.313, 1.618, 2.055, 2.585; max error 3.7 % vs image-system targets; monotonic | Tuck (1975), Wieselsberger |
| V4b | External ground-effect comparison | **PENDING** | model 2.585 vs published 1.34 at h/c=0.15 (~1.93x); modelled section is not the Zerihan & Zhang geometry; no digitized data | Zerihan & Zhang (2000) |
| V5 | CFD mesh independence | **PASSED (O-grid family, 2026-10-30)** | coarse 7,616 cells Cl 0.37922 (converged); medium 17,184 Cl 0.36170 (converged); fine 38,736 Cl 0.35696 (iter-cap; force-stability flag unconverged). r21=1.501, r32=1.502, p=3.219, extrapolated Cl 0.35520, fine rel. error 1.327 %, GCI fine 0.615 %, asymptotic ratio 0.985 -> in asymptotic range | Roache (1998), Celik et al. (2008) |
| V6A | NACA 0012 polar (O-grid family) | **measured; tight targets partly met** | Cl 0.0000/0.3617/0.7216/0.9009/1.0731 at 0/4/8/10/12 deg; Cd 0.0075/0.0117/0.0245/0.0347/0.0486; all five cases converged. Cd0 delta 0.00071 (target < 0.0015, met); lift slope 0.0902/deg vs 0.1081/deg (16.58 % error; target < 4 %, not met) | McCroskey (1987), Ladson (1988) |
| V6B | Ground-effect trend (O-grid coarse) | **trend only** | Cl ratio 1.652 / 2.100 / 2.486 / 3.063 / 3.446 at h/c 1.00/0.50/0.30/0.20/0.15, all converged; direction matches, geometry is an inverted NACA 4412 section, not Zerihan/Zhang | Zerihan & Zhang (2000), trend only |
| V7 | Surrogate cross-validation | **all four targets met (O-grid envelope, 2026-11-01)** | real 28 rows, 13 converged, 0 diverged; incidence <= 8 deg, h/c 0.15-0.32. AR1 Cl R² 0.9952 (RMSE 0.0674, MAPE 1.15 %), Cd R² 0.9907 (RMSE 0.0034, MAPE 0.23 %); coverage 100.0 % / 96.4 %. SF baselines: Cl R² 0.9898, Cd R² 0.9937 | Kennedy & O'Hagan (2000) |
| V8 | Multi-fidelity vs single-fidelity | **targets partly met (O-grid envelope, 2026-11-01)** | train 19 / test 5. Budgets N=6/12/19: SF RMSE 0.1545/0.1616/0.0930, AR1 0.7893/0.1492/0.0566, reduction -410.8 %/+7.7 %/+39.1 %; N=6 FAIL (correlation un-estimable at 6 HF points; AR1 wins from N=12 up), final R² 0.9979 PASS, coverage 95 % = 100 % PASS (intervals conservative: 80 % empirical at 50 % nominal, mean width 0.4368) | Forrester et al. (2007) |
| V9 | Active learning sample efficiency | **targets NOT met (1 of 3, O-grid envelope, 2026-11-01)** | AR1 learning curves, lenient + strict oracle, test rows 1/18/22/23/27, budgets N=4/6/9/13. Lenient RMSE_Cl: AL 0.616/0.846/0.226/0.048, maximin 0.322/0.306/0.156/0.042, random 0.601/0.345/0.131/0.059. Checks: AL beats random **+17.6 %** PASS, AL beats space-filling **-16.8 %** FAIL (maximin wins at every budget), strict yield AL 5.0 vs random 5.0 FAIL (tie). Measured answer stands: variance acquisition does not pay for itself on 28 real rows at these budgets | Jones, Schonlau & Welch (1998), Forrester et al. (2007) |
| P4.3 | Sobol global sensitivity on the surrogate | **passing (2026-11-01)** | Saltelli N=256 + N=512 bootstrap check on SF GPs, sampled only inside the CFD-validated envelope box. Total-order ST: Cl - ride_height_front_mm 0.575 +/- 0.112, speed_kmh 0.226 +/- 0.040, rear_wing 0.139 +/- 0.029; Cd - ride_height_front_mm 0.850 +/- 0.130, rear_wing 0.112 +/- 0.024, speed 0.034 +/- 0.008. Max abs dST between budgets 0.0008 (Cl) / 0.0056 (Cd) -> stable, targets_met True. Caveat: indices describe the surrogate inside the validated envelope, not the raw design space; bootstrap CIs are seed-pinned so the artifact is reproducible | Sobol (1993), Saltelli et al. (2010) |
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
- **The wind-tunnel smoke is an analytic visualization, not CFD output.** The
  live viewer advects tracers through a potential-flow model (blockage
  doublets, trailing wingtip vortex pairs, a 1/7-power floor boundary layer,
  wake momentum deficit with rear-wing downwash) plus divergence-free
  curl-noise turbulence. Nothing in it is sampled from the OpenFOAM solves, and
  the panel says so beside the controls. `scripts/check_flow_field.mjs` gates
  the field (finiteness, bounded speed, divergence-free turbulence, spatial
  coherence, streamline integration, floor profile, wake deficit); it passes
  11/11 on Node 24 and costs ~1 ms/frame for 400 tracers.
- **Mesh independence is demonstrated on the body-fitted O-grid family**
  (ADR 0006): GCI fine 0.615 %, asymptotic ratio 0.985, fine Cl within 1.33 %
  of the Richardson extrapolation. The fine solve reached the 2000-iteration
  cap and is flagged unconverged by the force-stability test, which is stated
  alongside the result rather than hidden.
- **V6 Part A uses fully turbulent RANS at Re=3.0M.** On the O-grid family it
  still underpredicts lift (12.8-17.8 % per point) and overpredicts drag; the
  Cd0 target is now met, the lift-curve slope target is not (16.58 % error).
- **V6 Part B is trend-only.** The modelled section is not the Zerihan/Zhang
  multi-element wing, so absolute ratios are not a geometry validation.
- **V4 external validation is still pending.** The potential-flow image-system
  check passes, but the VLM ground-effect augmentation is roughly 1.93x the
  published Zerihan & Zhang value at h/c=0.15. The script reports the external
  gap separately and does not count it as a pass. A digitized external dataset
  and a matching geometry are required before any external accuracy claim.
- **V2's low-AR point is weak.** At AR=2 the error is 15.4% against Helmbold;
  the script's 16% low-AR tolerance is documented in `validation/references.md`
  and the tight 3.5% target applies only to AR>=4.
- **The 28-point HF dataset (O-grid family, 2026-11-01) is real, envelope-
  constrained, and 13/28 solver-converged.** Final sampling envelope: total
  front-wing incidence main+flap <= 8 deg and front ride height 71-150 mm
  (h/c 0.15-0.32) - exactly the region V6 Part A/B measured. All 28 points ran
  the real WSL OpenFOAM hybrid path; 13 converged, 15 stayed
  force-statistically unstable (plausible Cl 3.7-6.8, above the stability
  threshold) and none diverged. Failed rows are retained with NaN HF targets,
  never filled in. It took two aborted attempts to find this envelope, both
  kept as evidence: `generate_real_dataset_ogrid_aborted.log` (0/18 usable:
  ride heights 15-50 mm put h/c at 0.032-0.106, below the mesh-validated
  range, and the coarse O-grid radial spacing in the wing-ground gap let Cl
  reach 8-20 or blow up) and `generate_real_dataset_ogrid_attempt2.log`
  (3/26 usable: the incidence axis was still unsampled-safe, total
  main+flap up to 28 deg, i.e. deep post-stall for a steady RANS section -
  every converged point was at 8.4 deg or less, every diverged point above
  15.7 deg). The pydantic ride-height ceilings were widened (200/250 mm) to
  match the envelope; user-facing VLM sliders stay wide because the live
  analytic path is valid everywhere. Re-running the final point set costs
  about 8 minutes serial thanks to the log-based CFD cache; a cold run is
  roughly 70 minutes.
- **V7 targets are all met; V8 is 2 of 3 (measured 2026-11-01).** On the O-grid
  envelope dataset, 5-fold cross-validation gives AR1 Cl R² 0.9952 / MAPE
  1.15 % and Cd R² 0.9907 / MAPE 0.23 % (targets R² > 0.90, MAPE < 5 %: all
  four PASS). V8's multi-fidelity test R² is 0.9979 and 95 % coverage is
  100 % (PASS), but the smallest-budget (N=6) AR1 run is worse than the
  single-fidelity GP (RMSE reduction -410.8 %, target > +10 %: FAIL). At
  N=6 the Kennedy & O'Hagan correlation coefficient is estimated from almost
  no HF information and collapses to a poor scaling; from N=12 the AR1 model
  wins (+7.7 %), and at N=19 it saves 39.1 % RMSE. The intervals are
  conservative, not sharp (empirical coverage >= nominal at every level:
  80 % at the 50 % nominal, 100 % from 68 % nominal up; mean 95 % width 0.4368). Scorecards:
  `data/datasets/v7_v8_scorecards.json`,
  run log `v7_v8_ogrid_run.log`, runner `scripts/run_v7_v8.py` (refuses a
  synthetic or zero-yield dataset instead of validating one).
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
- **Heavy jobs:** a medium real CFD run took 45.8 s; the 28-point envelope
  dataset took 8.0 min on the warmed log cache (5 fresh solves + 23 cached
  re-parses; a cold run is roughly 70 min serial). A 100-trial API optimization
  took roughly 100 s on this machine; the UI uses 30 trials (about 50 s) and
  shows an elapsed-time counter.
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
| 2026-10-28 | Real CFD + UI | OpenFOAM 1912 WSL, real V5/V6 runs, manual CFD UI, 503 path, 28-point real dataset, V7/V8 re-run, true Pareto, layout redesign, 30-trial UI, consistent optimizer advice; 27 fast + 4 slow tests pass, ruff/mypy clean | `4bd531e` |
| 2026-10-29 | Validation reporting | V4 split into potential-flow pass and PENDING external reference; V2 error corrected to 15.4% at AR=2; generated dataset logs ignored; 27 fast tests, ruff/mypy clean | `dac74eb` |
| 2026-10-30 | Mesh hardening | snappyHexMesh replaced by body-fitted O-grid polyMesh family (r=1.5, y+ 30), moving-ground wall, y+ instrumentation; V5 re-measured: GCI fine 0.615 %, asymptotic 0.985, PASSED; V6 re-measured (all cases converge, Cd0 target met); ADR 0006; flat engineering UI redesign; 33 fast + 4 slow tests, ruff/mypy clean | `3fb7498` |
| 2026-10-31 | UI audit pass | OpenDesigner review: 7 errors/5 warnings reduced to 0/0 (contrast tokens >= 4.5:1, 31 label associations, 4 px spacing grid, 12 px type floor, reduce-motion support, one-line labels, main landmark); tools + tokens documented in docs/DESIGN.md; O-grid dataset regeneration aborted 0/18 with h/c-envelope diagnosis (unchanged dataset); probe artifacts gitignored | `bc12812` |
| 2026-11-01 | Dataset envelope + V7/V8 | LHS constrained to the V6-validated envelope (incidence <= 8 deg, ride height 71-150 mm); two aborted attempts kept as evidence (0/18, then 3/26); final run: 28 real rows, 13 converged, 0 diverged, 8 min warmed; guarded V7/V8 runner (`scripts/run_v7_v8.py`); V7 all 4 targets met (AR1 Cl R² 0.9952 / Cd 0.9907), V8 2 of 3 (final R² 0.9979 PASS, coverage PASS, N=6 reduction -410.8 % FAIL with cause); 33 fast tests, ruff/mypy clean | `9ba54a8` |
| 2026-11-01 | Branch recovery + Phase 4 | `master` fast-forwarded over the 11 stranded commits (`899d3fe..9ba54a8`), restoring `backend/ml/` and clearing the pytest collection error that had made the suite uncollectable; Phase 4 scripts + V9/sensitivity test committed as tracked work; removed the unused `pool` parameter from `_run_scheme` and typed the Sobol payload via a `SobolIndices` TypedDict to clear 9 mypy errors (S1/ST/S2 and every verdict verified byte-identical after refactoring); pinned SALib's bootstrap seed, making `sensitivity_sobol.json` reproducible (byte-identical across two consecutive runs; previously only `*_conf` drifted); `*_fit_warnings.txt` logs gitignored; V9 recorded as targets NOT met (1 of 3); 33 fast + 2 slow tests pass, ruff/mypy clean | `d79d58f` |
| 2026-11-02 | Smoke realism + flow gate | Tracer field: per-frame white-noise jitter replaced by divergence-free curl noise (central-difference vector potential, eddy length ~0.22 car length, time-evolving); blockage doublet sign corrected (it was accelerating the inflow instead of decelerating it); 1/7-power floor boundary layer; wake momentum deficit + rear-wing downwash; trailing Lamb-Oseen wingtip vortex pairs; smoke billboards get fake directional lighting and age-based growth/dilution; the panel now states the visualization is analytic, not CFD; new `scripts/check_flow_field.mjs` gate (11/11 pass, ~1 ms/frame for 400 tracers); app verified in the browser with no console errors | this commit |
