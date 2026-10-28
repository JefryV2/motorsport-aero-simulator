# ADR 0004: Provenance labelling and the synthetic CFD fallback

- **Status:** accepted
- **Date:** 2026-10-28
- **Phase / task:** Phase 3 remediation (post-audit)

## Context

No OpenFOAM binary is available on the development machine
(`detect_cfd_backend()` returns `synthetic`). The Phase 2 runner therefore falls
back to `evaluate_calibrated_synthetic_cfd()`, a closed-form fit.

Despite that, the Phase 3 dataset stamped every row with
`solver_version="OpenFOAM-v2406"` and `convergence_flag=True`, the runner wrote a
fake `log.simpleFoam`, and V5/V6 "passed" against constants derived from the same
formula. This violated the project's honesty rule: do not name a method after
something it does not do.

## Options considered

1. **Remove the synthetic path entirely.** Pros: impossible to mislabel.
   Cons: the ML pipeline and tests cannot run on a laptop; Phase 3 stops
   completely until OpenFOAM is installed.
2. **Keep the path but label it everywhere** (code, data, API, UI). Pros:
   pipeline remains runnable and testable; the limitation is explicit and
   visible. Cons: requires discipline to propagate the label; a careless
   consumer could still ignore it.
3. **Keep the label only in logs.** Pros: smallest diff. Cons: the UI and dataset
   would still mislead; rejected.

## Decision

Option 2. The synthetic fallback is retained as an explicitly-labelled
low-fidelity surrogate:

- `CFDResult` carries `solver_used`, `solver_version` and `is_synthetic`.
- `run_cfd_with_provenance()` returns a `CFDProvenance` derived from the run that
  actually executed — never from solver availability.
- The synthetic verdict is `ConvergenceStatus.MOCK_EVALUATED` with
  `is_converged=False`; no residuals are invented.
- Case directories contain `SYNTHETIC_MODEL.json`; no fake `log.simpleFoam` is written.
- Dataset columns `solver_used`, `hf_fidelity`, `is_synthetic`, `convergence_status`
  are read from the run. `solver_version` is `none (calibrated-synthetic-v1)`.
- `SimulationResult.fidelity` is one of `empirical`, `low_fidelity_vlm`,
  `synthetic_model`, `rans`; `/api/health` reports `cfd_backend` and a note.
- The UI shows a provenance banner, a per-result fidelity badge, a method note,
  and a provenance line under the metrics.

## Consequences

- **Easier:** a reader can tell instantly whether a number came from VLM,
  an empirical correlation, a synthetic fit, or a solver.
- **Harder:** any downstream code that assumed `solver_version=="OpenFOAM-v2406"`
  must now branch on `is_synthetic`.
- **Must be tested:** `tests/test_provenance.py` locks the labels, the absence of
  fake logs, and the health payload.
- **Affected code:** `cfd/models.py`, `cfd/runner.py`, `cfd/openfoam_runner.py`,
  `ml/dataset.py`, `models/aero_config.py`, `main.py`, `frontend/index.html`,
  `frontend/js/app.js`, `frontend/js/engineerTools.js`.
- **Known risk:** the underlying coefficients are still a fit. Labelling does not
  make them accurate; V5/V6 remain invalid until a real solver runs.

## Approval

This ADR records a correction requested by the project owner on 2026-10-28. It
does not rewrite a working physics module; it changes labels, provenance
plumbing, and UI only.
