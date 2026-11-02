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
  coherence, streamline integration, floor profile, wake deficit, inflow
  seeding coverage); it passes 16/16 on Node 24 and costs ~1 ms/frame for 400
  tracers. Inlet tracers are seeded from an R2 low-discrepancy sequence so the
  curtain tiles the inflow plane instead of clumping; the rake feeds 45 tubes
  (5 heights x 9 lateral) and the Surround source feeds all four faces. The
  default look is a smoke-rake photograph: 56-point streak tubes (~3.9 m of
  trail, cross-section atlas with alpha constant along the flow axis), thin
  near-white threads that stay visible well into the wake. A volumetric style
  and the classic dot style remain selectable.
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
| 2026-11-02 | Uniform inflow seeding | R2 low-discrepancy inlet curtain for front grid/wide and top/side sources; rake upgraded from 7 streams to 24 tubes (a counter-parity bug meant only alternate tubes were reachable); new Surround source seeds front, top and both sides; inlet-heavy sources give 45% of tracers to the inflow instead of 35%; particle cap raised to 1000; flow gate extended with seeding checks (16/16 pass) | `c0f79d2` |
| 2026-11-02 | Smoke tubes | Smoke no longer reads as particles: trail is 20 history points at 0.025 model lengths (~1.7 m) instead of 8 at 0.07; new cross-section streak atlas keeps density constant along the flow axis so overlapping quads form a tube; segments overlap 1.5x; cross-axis shading; degenerate respawn segments are hidden (they stamped a bright blob at the inlet); tube width 1.05x and softer dilution so the wake stays visible. Verified in a browser frame: continuous plume, no console errors | `a13bfba` |
| 2026-11-02 | Smoke-rake filaments | New default particle style matching a tunnel rake photograph: thin (4-6 cm) near-white threads instead of a cloud; 56-point trails (~3.9 m), rake widened to 45 tubes (5 heights x 9 lateral) with 72% of the tracer budget in the inflow, respawn moved to 1.1 model lengths behind the tail so filaments persist into the wake, reduced wake growth/dilution for thread-like behaviour. Volumetric and dot styles remain; flow gate still 16/16, 33 fast tests pass | this commit |
| 2026-11-02 | Filament continuity | Smoke strands no longer read as bead chains: the streak cell's noise now varies only across the streak (length-wise noise repeated the same mottle in every segment quad), its end taper is capped at 6% with a 0.7 floor (a taper that reached zero made every joint a dark gap), and quads overlap 1.2x so a neighbour's flat middle always covers the taper; atlas mip chain disabled (each mip averaged the 9 differently-oriented cells into a cross-hatch grid on minified strands). Verified by offscreen render + readPixels: 0 GL errors, strands continuous, neutral smoke mean saturation 0.022 vs 0.651 when tinting, coverage smoke 44% / filament 22% / dots 1.4% | this commit |
| 2026-11-02 | Volumetric smoke rewrite | The streak cell cannot serve the volumetric style: its quads are wider than long, so marching them along the trail drew a ribbed fan. Added a second 9-cell atlas of isotropic noise-warped puffs; the smoke style now draws square quads that advance through the cells and rotate along the trail (no repeated stamp), with alpha gain 0.17 -> 0.075 and neutral base 0.70-1.00 -> 0.50-0.72 so overlapping puffs accumulate instead of clipping (blown-out pixels 0.03% of the frame), slower along-trail width growth, and recycling moved 0.55 model lengths past the tail so plumes reach the wake. Volumetric smoke is now the default style. Verified by offscreen render + readPixels: 0 GL errors, filament and dots styles unaffected, flow gate 16/16 | this commit |
| 2026-11-02 | Pre-publication honesty pass | README rewritten against HEAD and corrected on 6 stale claims (V5 now PASSED at GCI 0.615% / asymptotic 0.985, dataset 13 of 28 converged not 4, V6a lift slope 16.58% miss with Cd0 pass, V7 all 4 met, V8 2 of 3, V9 not met 1 of 3, regeneration ~70 min cold not 25-30, parameter ranges now quoted from `aero_config.py` not the optimizer space, upload `.stl`/`.obj` server-side vs 7 formats loaded in-browser); UI labels corrected to stop calling the analytic field CFD (`Volumetric 3D Tubes (CFD / CAD)` -> `Volumetric 3D tubes`, `CFD Slice Plane` -> `Slice Plane`) and the rake 24 -> 45 tubes; new hero render captured from the current build via the offscreen harness. Verified: page reloads with 0 console messages, DOM reads back the corrected labels, default style `smoke` | this commit |
| 2026-11-02 | CI: undeclared dependency | The first GitHub Actions run failed on Python 3.10 and 3.11 during collection with `ModuleNotFoundError: No module named 'SALib'`: `validation/sensitivity_sobol.py` (V9 Sobol study) imports SALib and `tests/test_validation_v9_sensitivity.py` imports that module, but SALib was declared in neither `requirements.txt` nor `pyproject.toml`. It only worked locally because the venv had it installed. Added `SALib>=1.5.0` to `requirements.txt` and to the `dev` extra; an AST scan of every tracked `.py` confirms SALib was the only undeclared third-party import | this commit |
| 2026-11-02 | CI: numpy scalar type error | The second Actions run got past collection and failed at `mypy backend` (mypy 2.4.1, numpy 2.2.6): `backend/simulation/vlm/geometry.py:128` passed `np.linalg.norm`'s result to `Panel(area: float)`, reported as `floating[Any] | float64` is incompatible with `float`. Wrapped the computed area in `float()`; the arithmetic is unchanged. It was invisible locally because this venv has newer numpy stubs. CI is now reproduced locally by extracting `git archive master` into a clean directory (this also excludes the untracked Phase 5 lap files) and running the three CI steps there: ruff `All checks passed`, mypy `Success: no issues found in 32 source files`, pytest `33 passed, 6 deselected` | this commit |
| 2026-11-03 | Parametric built-in car | The drawn body and the smoke were decorative fractions of a bounding box: rake, diffuser angle, rear ride height and chord/span changed nothing on screen. New `frontend/js/carSpec.js` is the three-free source of truth (wing stations and heights from `panel_method.py:38-39`, floor line from the two ride heights at the two axles, diffuser throat/exit from `diffuserGeometry`, all 12 aero terms the field mirrors in TRUE metres); `carGeometry.js` builds the mesh from it, and `toFlowGeo()` publishes the solved stations in world metres so the doublets, tip filaments, floor sink and diffuser source sit on the body that was actually rendered. Uploaded CAD keeps the bounding-box fallback. Fixed three drawn-body defects found by measuring the mesh, not by looking at it: front-wing endplates crossed the tunnel floor at a 25 mm ride height (a plate is centred on its own height, so anchor by its bottom), diffuser strakes dipped to y = -0.0177 (a rotated constant-height box cannot follow a widening channel; now lofted wedges), and `extents.maxY` declared a 1.1 m car whose drawn top is the rear-wing endplate crown at 0.91 m, inflating wake centre, sigma, boundary layer and seeding by ~20% (now derived from shared `REFERENCE.rwEndplateHeightM`/`rwEndplateDropM`). Flow similarity: the field must not change with display scale, which required scaling two previously fixed lengths (doublet/source softening radius², ground band) - deviation now exactly 0.00%. Verified: gate 37/37, offscreen render through the app's own viewer gives world minY 0.0000 / maxY 0.910 == declared height, 0 vertices below the floor across 4 configs including 0 mm ride heights, geometry rebuild 20.2 ms, 0 console messages | this commit |
| 2026-11-03 | Flow gate in CI | `.github/workflows/ci.yml` gains a Node 20 step running `node scripts/check_flow_field.mjs` (the viewer modules have no build step and no dependencies), taking the gate from 16 to 37 assertions: the geo path, a golden mirror of `analytical.py` lift and underbody terms through the JS (so a backend change breaks both sides), a 720-point probe cloud, six setup knobs that must move the field, underbody gap speed, wake deficit and upwash sign, rake seeding at true heights, display-scale invariance and yaw axis tracking. Found and fixed a CI-unsafe flake first: the turbulence divergence ratio is a mean over randomly placed probes, and at 40 probes it swung 0.115-0.206 between runs against a 0.2 bound (1 failure in ~10 local runs); raised to 300 probes, where 25 trials give 0.142-0.173. Verified: 20 consecutive gate runs, 0 failures, 182 ms each | this commit |
| 2026-11-03 | Live-preview stall | Moving one slider blocked the main thread ~4 s, so the tunnel froze mid-drag. Profiling the app's own viewer showed two causes: `refreshVisualization` calls `setFlowSettings` + `setModelScale` + `setModelRotation` + `updateAero`, and each of those re-traced and re-tessellated every streamline on its own (4 full rebuilds per move); and the tube style re-sampled the analytic field at all ~500 vertices of every tube, which cost more than integrating the tubes did. Added `beginFlowBatch`/`endFlowBatch` + `_rebuildFlowVisuals` so a setup change collapses to one rebuild (tracers respawn only when a setting actually needs it - a view-mode change no longer restarts the smoke), and tube vertices now take their speed from the traced station they belong to by ring index instead of sampling the field (validated at 0.056 m vertex offset, 51.07 vs 49.19 m/s). Debounce 16 -> 90 ms, because one rebuild is no longer free. Measured: `_updateStreamlines` 1374 -> 482 ms; harness A/B batched 495 ms vs unbatched 2054 ms; live app 543-628 ms per config change. Verified with the flow gate (37/37, exit 0), an offscreen tube render (112 tube meshes, strands continuous with the speed gradient), and 0 console messages | this commit |
