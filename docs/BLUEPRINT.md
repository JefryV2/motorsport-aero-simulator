# Aero-ML Project Blueprint



Target plan for raising the Motorsport Aero Simulator from a v0.1.0 prototype to a deep, validated aerodynamics + machine learning engineering project.



**Reading order for the agent:** `AUDIT.md` (claims to verify) -> this file -> `STATUS.md`.

**Principle:** build on the existing project. Do not start from zero and do not rewrite working modules without an ADR and human approval.



---



## 1. Thesis



Everything serves one question:



> Can a multi-fidelity surrogate (a cheap physics model plus a small number of CFD runs) predict aerodynamics accurately enough to optimize lap time, and how much CFD budget does it save compared with naive sampling?



Headline results the project must produce, whatever the numbers turn out to be:



- Surrogate error versus number of CFD runs (learning curves), with and without the low-fidelity model.

- Active learning versus random and space-filling sampling at equal CFD budget.

- Calibration of the uncertainty (does a 95% interval cover about 95% of truth?).

- A Pareto front of lap time versus balance versus drag, with the best candidates re-verified by real CFD.



## 2. Scope



**In scope:** vortex lattice physics, scripted OpenFOAM pipeline, multi-fidelity surrogate with uncertainty, active learning, sensitivity analysis, point-mass lap simulation, multi-objective optimization with a regulation envelope, CFD-to-measurement correlation, reproducible experiments, a written report.



**Out of scope:** auth, multi-tenancy, billing, SSO, on-prem deployment, mobile layout, full-car CFD, transient aerodynamics. Anything else goes to `docs/BACKLOG.md`.



---



## 3. Current state and migration



### 3.1 Starting point (to be verified in Step 0)



The repo already has: a FastAPI backend with Pydantic schemas; a semi-empirical analytical aero model; an Optuna TPE optimizer (11 parameters); a rule-based plus LLM advisor; a Three.js viewer with model import, particles, streamlines, Cp coloring, probe, slice plane, force vectors; a metrics dashboard, baseline comparison, JSON/CSV export, DRS toggle, 5 track profiles, and a 5x5 aero map.



### 3.2 Keep / wrap / replace



| Module | Decision | Notes |

|---|---|---|

| Three.js viewer, UI, model import, camera presets, probe, slice, force vectors | **KEEP** | Add uncertainty and Pareto views later (Phase 8) |

| `flowEngine.js`, `meshFlow.js`, `aeroField.js` | **KEEP** | Label as qualitative potential flow, not CFD |

| Metrics dashboard, baseline comparison, exports, DRS toggle, track selector | **KEEP** | Extend, do not rewrite |

| FastAPI app, Pydantic schemas | **KEEP / EXTEND** | Version any API change |

| `analytical.py` | **KEEP, DEMOTE** | Becomes the low-fidelity baseline in the multi-fidelity model |

| `panel_method.py` ("panel_vlm") | **RENAME NOW, REPLACE LATER** | Currently identical to analytical (bug B1) |

| `openfoam_runner.py` ("openfoam_cfd") | **RENAME NOW, REPLACE LATER** | Currently relabels analytical output (bug B2) |

| Pseudo-Pareto in `optimizer.py` | **REPLACE** | True multi-objective (Phase 6) |

| Linear lap time proxy | **REPLACE** | Point-mass lap sim (Phase 5) |

| `advisor.py` | **KEEP, EXTEND** | Add conversation memory and tool calling (Phase 8) |

| `mesh_io.py` | **KEEP, FIX** | Fix B4, B5, B7 in Phase 0 |



### 3.3 Folder migration rule



Extend the existing folders (`simulation/`, `optimization/`, `ai/`, `visualization/`, `models/`). Add new packages beside them. Do not mass-rename or reorganize. Move a file only when a task requires it, in its own commit, with imports updated and tests passing.



```text

motorsport-aero-simulator/

  pyproject.toml                 NEW (ruff, mypy, pytest, packaging)

  backend/

    main.py                      existing

    models/aero_config.py        existing (extend)

    simulation/

      analytical.py              existing (low-fidelity baseline)

      panel_method.py            existing (rename, later remove)

      vlm/                       NEW: panels, influence matrices, solver, trefftz, image ground

      lowfi.py                   NEW: unified low-fidelity model

      cfd/

        openfoam_runner.py       existing (rewrite behind ADR)

        templates/               NEW: 2D and 3D OpenFOAM case templates

        case_builder.py  parser.py  mesh_study.py    NEW

    ml/                          NEW: doe, dataset, models/, active_learning, calibration, sensitivity

    vehicle/                     NEW: tire, track, ggv, laptime

    optimization/

      optimizer.py               existing (extend)

      moo.py  constraints.py  regulations.py    NEW

    correlation/                 NEW: telemetry, tunnel, calibrate

    ai/advisor.py                existing (extend)

    visualization/               existing

  frontend/                      existing (extend)

  experiments/                   NEW: configs/*.yaml, run scripts, outputs (DVC or MLflow)

  validation/                    NEW: one script per V-case, references.md, results/

  tests/                         NEW: unit, property, regression, golden, validation-in-CI

  docs/                          AUDIT.md, BLUEPRINT.md, STATUS.md, decisions/, REPORT.md, BACKLOG.md

```



### 3.4 Protecting working code



- Tag `v0.1.0-baseline` before any change.

- Capture golden outputs of the current analytical model and optimizer (fixed seeds, reference configs) in `tests/golden/`.

- No rewrite or deletion of a working module without: a short ADR, human approval, and a golden or regression test showing behavior is preserved or the change is intended.

- Imported car meshes stay visualization-only. VLM, CFD, and the surrogate work on the parametric geometry in the design vector.



### 3.5 Audit items the plan must not drop



| Audit item | Where it goes |

|---|---|

| Optimizer blocks the event loop (P1) | Phase 0 (`asyncio.to_thread`) |

| No optimization progress streaming | Phase 6 backend, Phase 8 UI (SSE) |

| Chat has no conversation memory | Phase 8 |

| Advisor should call simulate/optimize as tools | Phase 8 |

| Aero map fixed at 5x5 in UI | Phase 8 (configurable, backed by surrogate) |

| Model upload cleanup (B7) | Phase 0 |

| Three.js loaded without SRI (P3) | Phase 0 or Phase 8 |

| Coordinate convention undocumented (P4) | Phase 0 |



---



## 4. Design vector



The current config is a full-car formula setup with about 11 optimizer parameters. The ML study uses a **reduced, documented subset**; parameters outside it keep their analytical behavior and are labeled as such. The exact subset and bounds are proposed in the Step 0 report, mapped to existing Pydantic fields.



| Group | Candidate parameters |

|---|---|

| Front wing | main plane angle, flap 1 angle, flap 2 angle, slot gap, overlap, chord, endplate height |

| Rear wing | angle of attack (and DRS state as a separate study) |

| Floor / diffuser | front ride height, rear ride height or pitch, diffuser angle, diffuser length |

| Conditions | speed, yaw angle |

| Outputs | CL, CD, CoP (aero balance), per-component forces, always with uncertainty |



**Compute tiers (choose by hardware, record in an ADR):**



- **Tier A, laptop-friendly:** 2D multi-element wing in ground effect, 5 to 6 parameters, 100 to 300 runs, minutes per run. Prove the whole pipeline and ML study here first.

- **Tier B, 3D:** half-model front wing with endplate over a ground plane, 8 to 10 parameters, 60 to 150 runs. Measure run time in Phase 2 before committing.



## 5. System map



```text

Design vector (geometry params + operating conditions)

        |

        v

+----------------+      +-----------------------+

| Parametric     |----->| Low fidelity: VLM +   |--> CL, CD, CoP in milliseconds

| geometry       |      | ground effect + visc. |

+----------------+      +-----------+-----------+

        |                           | residual learning (AR1: yH = rho*yL + delta)

        v                           v

+----------------+      +-----------------------+

| CFD pipeline   |----->| Multi-fidelity        |--> mean + uncertainty

| mesh>solve>    | HF   | surrogate (GP/ens.)   |

| parse          | data +-----------+-----------+

+----------------+                  |

        ^                           v

        |  next CFD run chosen    uncertainty

        +--- by active learning <---+

                                    |

              +---------------------+--------------------+

              v                                          v

   Multi-objective optimizer (NSGA-II)        Vehicle model (point-mass lap sim,

   constraints = regulation envelope  <----->  tires, aero map from surrogate)

              |

              v

   Pareto front -> top candidates re-run in CFD -> report and API/UI

```



---



## 6. Phases



Each phase has a Definition of Done. Do not start the next phase until the current one is accepted by the human.



### Step 0: Reality check (before Phase 0, read-only)



- Verify every audit claim against the code (CONFIRMED / WRONG / UNVERIFIABLE with file and line evidence).

- Run the app and existing scripts; record what works and what crashes.

- Tag `v0.1.0-baseline`; create `tests/golden/`.

- Produce the keep/wrap/replace table and the reconciled design vector.

- **Done when:** the human approves the Step 0 report.



### Phase 0: Hygiene and honest naming (1 to 2 weeks)



- P0.1 Rename the mislabeled modes (`panel_vlm` -> `analytical_enhanced` or hidden, `openfoam_cfd` -> `cfd_stub`).

- P0.2 Typed method enum in schemas so invalid values return 422 (B8).

- P0.3 `pyproject.toml`, ruff, mypy, pytest; `.gitignore`; remove OBJ files from git history; fix B4 (non-STL/OBJ uploads), B6 (canvas variable), B7 (upload cleanup).

- P0.4 Logging, restricted CORS, optimizer in `asyncio.to_thread`.

- P0.5 GitHub Actions: lint, type-check, tests.

- P0.6 Physics-based tests for the current analytical model (monotonicity with speed squared, bounded balance, limiting cases) plus golden tests.

- P0.7 Document the coordinate convention (x forward, y lateral, z up) and its mapping to the Three.js scene.

- **Done when:** CI is green, no mode name claims more than it does, a fresh clone runs with one documented command.



### Phase 1: Real vortex lattice physics (3 to 5 weeks)



- P1.1 VLM: horseshoe vortices per panel, bound vortex at quarter chord, control point at three-quarter chord, Biot-Savart influence matrix, linear solve, Kutta-Joukowski forces, induced drag in the Trefftz plane, cosine spacing.

- P1.2 Ground effect by image system (mirror vortices with opposite circulation).

- P1.3 Multi-element handling (main plane plus flaps as separate surfaces); slot-gap effect as a documented empirical correction.

- P1.4 Viscous profile drag and Reynolds dependence from section data. ADR: 2D CFD-derived table (reuses Phase 2) versus a published correlation.

- P1.5 Envelope warnings for conditions outside the attached-flow range (VLM has no stall).

- **Done when:** V1 to V4 pass with documented tolerances; the old analytical model is demoted to a baseline.



### Phase 2: CFD pipeline (3 to 5 weeks)



- P2.1 Deterministic case builder: config in, OpenFOAM case directory out; hash the config and cache results.

- P2.2 Meshing (blockMesh for 2D, snappyHexMesh for 3D), `simpleFoam`, k-omega SST, wall treatment matched to the y+ target; record y+ statistics.

- P2.3 Runner: local process pool, timeouts, resume, failure classification (diverged, not converged, mesh failed).

- P2.4 Parser: `forceCoeffs` history, average of the last N iterations with standard deviation as the convergence indicator; optional VTK export.

- P2.5 Mesh independence study on 3 levels with Richardson extrapolation and GCI.

- P2.6 Reproduce a published benchmark before trusting other results.

- **Done when:** one command takes a config to forces with a convergence verdict; V5 and V6 documented.



### Phase 3: Dataset and multi-fidelity surrogate (4 to 6 weeks)



- P3.1 Sobol or Latin hypercube DoE over the design vector, bounded by the regulation envelope.

- P3.2 Parquet dataset: config hash, mesh level, solver version, convergence flag, HF result, LF prediction at the same point. Fixed train/validation/test splits; DVC or MLflow versioning.

- P3.3 Models: (a) GP on HF only, (b) AR1 multi-fidelity (yH = rho*yL + delta, GP on delta), (c) deep ensemble. Stretch: neural operator on field data.

- P3.4 Metrics: R squared, RMSE, MAPE per output; k-fold and leave-one-out; learning curves.

- P3.5 Uncertainty calibration: coverage and reliability curves; report over- or under-confidence.

- **Done when:** V7 and V8 results exist for all models on identical splits, with plots.



### Phase 4: Active learning and explainability (2 to 3 weeks)



- P4.1 Acquisition functions (predictive variance, then diverse batch selection) and a loop that proposes the next CFD runs.

- P4.2 Compare to random and Sobol sampling at equal budget across several seeds with confidence bands.

- P4.3 Global sensitivity analysis (Sobol indices via SALib) on the surrogate; connect findings to physical reasoning.

- **Done when:** V9 complete. Report the result even if active learning does not win.



### Phase 5: Vehicle model and lap simulation (3 to 4 weeks)



- P5.1 Track from centerline (x, y) to curvature; speed profile by forward and backward passes with a g-g limit.

- P5.2 Load-sensitive tire model; power-limited acceleration; aero drag and downforce from the aero map (surrogate mean versus speed, ride height, pitch).

- P5.3 Aero balance via CoP so axle loads and grip distribution respond to balance.

- P5.4 Track data from user-supplied or public centerlines. Never invent coordinates.

- **Done when:** V10 passes; the linear lap-time proxy is removed.



### Phase 6: Multi-objective optimization (3 to 4 weeks)



- P6.1 NSGA-II (pymoo or Optuna NSGAIISampler); objectives: lap time, balance error, drag.

- P6.2 Constraints from `regulations.py`: geometry envelope, minimum ride height, stall-margin limit.

- P6.3 Robust mode: optimize mean minus k times standard deviation.

- P6.4 Re-run the top Pareto candidates in CFD and report surrogate error on exactly those designs.

- P6.5 SSE progress stream for optimization runs.

- **Done when:** V11 complete with the verification table in the report.



### Phase 7: Correlation and calibration (2 to 3 weeks)



- P7.1 CSV importers for tunnel and telemetry data; MoTeC `.ld` as a stretch.

- P7.2 Synthetic data generator with a known injected bias and noise so calibration can be verified against ground truth. Label as synthetic everywhere.

- P7.3 Calibration model: additive or multiplicative correction as a function of ride height and yaw, with uncertainty.

- **Done when:** V12 shows the injected bias recovered within tolerance, with a before-and-after gap plot.



### Phase 8: Integration and report (2 to 3 weeks)



- P8.1 API: surrogate prediction with uncertainty, aero map from the surrogate (configurable grid), optimization progress via SSE, Pareto results.

- P8.2 UI: fidelity badge on every result, uncertainty bands, Pareto plot, out-of-training-range warnings.

- P8.3 Advisor: conversation memory and tool calling into simulate/optimize so quoted numbers come from code.

- P8.4 Docker, full CI including fast validation cases, a script that regenerates every figure from stored data.

- P8.5 Final report (section 9) and a short demo script.

- **Done when:** a new reader can reproduce the headline figures from the README.



---



## 7. Validation matrix



Reference values must come from cited sources recorded in `validation/references.md`. The agent must not recall or invent reference numbers; if a reference is missing, mark the case **pending** and ask.



| ID | Case | Pass criterion (initial target) |

|---|---|---|

| V1 | VLM, elliptic planform: induced drag factor near 1 | within a few percent of theory |

| V2 | VLM lift-curve slope versus lifting-line theory across aspect ratios | documented error per aspect ratio |

| V3 | VLM panel-count convergence | change below 1% at chosen resolution |

| V4 | Ground-effect trend versus published data | trend and magnitude error stated |

| V5 | CFD mesh independence on 3 levels | GCI reported, fine-mesh uncertainty stated |

| V6 | CFD benchmark case from literature | error versus reference stated |

| V7 | Surrogate accuracy on held-out CFD runs | target: downforce MAPE under 5% (hypothesis) |

| V8 | Uncertainty calibration | 95% interval coverage between 90% and 98% (target) |

| V9 | Active learning versus random versus Sobol | learning curves with seeds; claim only what the data shows |

| V10 | Lap sim analytic cases (skidpad speed, straight-line limits) | within numerical tolerance |

| V11 | Optimizer candidates verified in CFD | surrogate versus CFD error on chosen designs |

| V12 | Calibration recovers injected synthetic bias | within stated tolerance |



## 8. Quality and reproducibility rules



- SI units everywhere; units documented in signatures or docstrings.

- Every experiment is a YAML config plus a seed; outputs are never hand-edited.

- Every non-obvious choice gets a short ADR in `docs/decisions/`.

- Tests ship with each module: unit, property-based physics invariants, regression, golden.

- Type hints on core code; ruff and mypy must pass.

- No new heavy dependency without an ADR.

- `STATUS.md` is updated at the end of every task.

- Never fabricate reference data, results, or track coordinates.



## 9. Final report outline



1. Problem and thesis

2. Physics models and assumptions (and where they break)

3. CFD pipeline and mesh study

4. Dataset design

5. Surrogate models and comparison

6. Uncertainty calibration

7. Active learning results

8. Sensitivity analysis

9. Lap simulation

10. Optimization and CFD verification

11. Correlation and calibration

12. Limitations and negative results

13. Future work



## 10. Main risks



| Risk | Mitigation |

|---|---|

| CFD too slow for the dataset size | Start with Tier A (2D); measure run time in Phase 2 before committing to Tier B |

| Surrogate does not beat the baseline | Report it; the comparison is still a valid result |

| Reference data hard to find | Mark cases pending, use analytic cases first, ask the human to supply digitized data |

| Audit is stale or wrong | Step 0 verification before any plan relies on it |

| Scope creep | Phase gates; anything outside section 2 goes to `BACKLOG.md` |

| Rewriting working code by accident | Golden tests, ADR plus approval rule, small commits |





---
