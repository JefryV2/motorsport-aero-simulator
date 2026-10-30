# ADR 0004: Provenance labelling and the synthetic CFD fallback

- **Status:** accepted, updated 2026-10-28 after OpenFOAM was installed
- **Date:** 2026-10-28
- **Phase / task:** Phase 2-3 remediation

## Context

Originally no OpenFOAM binary was available on the development machine
(`detect_cfd_backend()` returned `synthetic`). The Phase 2 runner fell back to
`evaluate_calibrated_synthetic_cfd()`, a closed-form fit.

The Phase 3 dataset nevertheless stamped every row with
`solver_version="OpenFOAM-v2406"` and `convergence_flag=True`, the runner wrote a
fake `log.simpleFoam`, and V5/V6 "passed" against constants derived from the same
formula. That violated the project's honesty rule: do not name a method after
something it does not do.

OpenFOAM 1912 is now installed in WSL2 Ubuntu 22.04. `detect_cfd_backend()`
returns `wsl`; `simpleFoam`, `checkMesh`, and `yPlus` execute real cases on the
body-fitted O-grid family (ADR 0006).
The question is no longer "can we label a synthetic model honestly?" but "what
exactly is the real path, and what happens when it fails?"

## Decision

1. **Use the real solver when it is detected.** The `cfd` method runs a 2D
   single-element inverted NACA 4412 section with a moving ground plane,
   a body-fitted O-grid polyMesh (ADR 0006), and steady k-omega-SST RANS. It
   is a section simplification, not the full multi-element wing or the full car.

2. **Label the total honestly as `rans_hybrid`.** The 2D section supplies the
   front-wing contribution; the floor and rear wing remain empirical terms. The
   API, dataset, and UI call this `rans_hybrid` / `OpenFOAM + empirical`, never
   plain `rans` or full-car CFD.

3. **Keep the synthetic fallback only for missing solvers.** When backend
   detection returns `synthetic`, the result is `synthetic_model` /
   `cfd_synthetic_model`, `is_synthetic=True`, `convergence_flag=False`, and a
   `SYNTHETIC_MODEL.json` marker. No fake `log.simpleFoam` is written.

4. **A real-run failure is a failure.** `CFDRunError` propagates to the API as
   HTTP 503. The UI shows a failed case card with no synthetic substitution.
   Unexpected CFD exceptions also return 503 rather than falling back.

5. **Provenance is data, not a badge bolted on afterwards.** `CFDProvenance` is
   derived from the run that actually executed. The dataset stores
   `solver_used`, `solver_version`, `hf_fidelity`, `is_synthetic`,
   `convergence_status`, `hf_failed`, and `hf_error`. Failed points keep NaN
   high-fidelity targets instead of invented values.

6. **CFD is a manual action in the UI.** A medium case takes roughly one minute.
   VLM remains the live preview; the CFD tab requires an explicit **Run CFD**
   press. Optimization and the ride-height map use the fast VLM path.

7. **Validation reports measured results, not forced passes.** V5 reports
   "NOT DEMONSTRATED" because the mesh family is not asymptotic. V6 Part A
   reports its measured errors; Part B is explicitly trend-only because the
   section geometry is not the Zerihan/Zhang wing. V7/V8 report unmet targets
   rather than asserting success.

## Evidence

- `/api/health` reports `cfd_backend="wsl"`, `solver_version="OpenFOAM 1912"`,
  `cfd_ready=true`.
- A front-wing section run completed in 45.8 s with 21,155 cells,
  `Cl_section=2.7899`, `Cd_section=0.13716`, `Cm=-0.05677`, and
  `convergence_status="unconverged"`.
- V5: coarse/medium/fine Cl = 0.45158/0.40384/0.34433; GCI fine 107.049 %;
  asymptotic ratio 0.432 -> NOT DEMONSTRATED.
- V6 Part A: Cl and Cd measured at 0/4/8/10/12 deg; Cd0 delta 0.00308 and lift
  slope error 27.05 %; historical targets not met. Part B ratios increase with
  lower h/c but are trend-only.
- The real dataset has 28 rows from `openfoam_wsl`; 4 are solver-converged and
  24 are real but force-unconverged. The previous 100-row synthetic file is
  preserved at `aero_surrogate_v1_synthetic_backup.parquet`.
- V7 (real data): AR1 Cl R² 0.6879, Cd R² 0.9605; MAPE 7.08 % / 0.92 %.
- V8 (real data): smallest-budget RMSE reduction 18.9 %, 95 % coverage 100 %,
  test R² 0.5751; overall targets not all met.

## Consequences

- **Easier:** a reader can tell whether a number came from VLM, an empirical
  correlation, a synthetic fit, or a real solver, and can see the case metadata.
- **Harder:** the hybrid total is not full-car CFD; it must not be used as one.
  The dataset is smaller and less converged than the old synthetic one, so the
  surrogate metrics are worse but real.
- **Must be tested:** `tests/test_provenance.py` locks the real/synthetic labels,
  the absence of fake logs, the health payload, and the dataset checks.
- **Affected code:** `cfd/models.py`, `cfd/runner.py`, `cfd/openfoam_runner.py`,
  `ml/dataset.py`, `models/aero_config.py`, `main.py`, `frontend/index.html`,
  `frontend/js/app.js`.
- **Known risk:** the 2D section is mesh-sensitive (V5) and the hybrid balance
  is front-biased relative to real F1 targets. Labelling does not make the
  physics accurate; it makes the limitation visible.

## Approval

This ADR records a correction and a scope decision requested by the project
owner on 2026-10-28. It does not rewrite working physics modules; it defines the
real/synthetic boundary, the hybrid label, the failure behaviour, and the
validation reporting rules.
