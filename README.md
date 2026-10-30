# Motorsport Aero Simulator

Formula open-wheel aerodynamic simulator with a 3D vortex-lattice preview, an
optional real OpenFOAM front-wing section solve, Bayesian optimization, and
AI-powered setup advice.

## Features

- **Three simulation methods**
  - `analytical` — fast semi-empirical model (wings + ground effect + diffuser)
  - `analytical_enhanced` / `panel_vlm` — 3D vortex lattice with image ground
    plane and Trefftz drag; this is the live preview
  - `cfd` — manual OpenFOAM RANS run on a 2D front-wing section, combined with
    empirical floor and rear-wing terms (hybrid estimate; not full-car CFD)
- **Bayesian optimizer** (Optuna TPE) — tunes wing angles, ride height, rake,
  diffuser, and gurney flaps. Optimization always uses the fast VLM path; real
  CFD is a single-point verification, not a 100-case inner loop. The UI runs 30
  trials and shows an elapsed-time counter; the API trial count is configurable.
- **AI advisor** — LLM-powered recommendations (OpenAI-compatible API) with a
  rule-based fallback
- **Web UI** — interactive 3D viewer with STL/OBJ/GLB import, a heuristic Cp
  pressure map, streamlines, force vectors, provenance badges, and a CFD case
  details card

## Quick Start

```bash
cd motorsport-aero-simulator
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # optional: add OPENAI_API_KEY for the LLM advisor
python -m uvicorn backend.main:app --port 8000
```

Open <http://localhost:8000>. Drag and drop a car model (STL, OBJ, GLB/GLTF) or
use the built-in formula car.

## Real CFD (optional)

The app works without OpenFOAM: the VLM preview runs immediately, and the CFD
tab reports an explicitly-labelled synthetic fallback when no solver is found.

To run real CFD on Windows, install OpenFOAM inside WSL2 (for example,
Ubuntu 22.04 with OpenFOAM 1912), then expose `simpleFoam`, `checkMesh`, and
`yPlus` inside the distro. Restart the server after installing; the
health endpoint reports the detected backend and solver version.

The CFD tab is deliberately **manual**, because a medium case takes roughly one
minute:

1. Select the CFD tab.
2. Press **Run CFD**.
3. Wait for the mesh + solve progress; the case card reports cell count,
   residuals, iterations, execution time, and `CONVERGED` / `UNCONVERGED`.

What the real path does:

- Generates a 2D inverted NACA 4412 section from the current front-wing chord,
  total angle, and front ride height.
- Writes the body-fitted O-grid directly as an OpenFOAM `polyMesh` (wall
  facets exactly on the NACA section, first-cell height set from the target
  y+, moving-ground wall patch) and solves steady k-omega-SST RANS.
- Reports measured y+ statistics and `checkMesh` quality metrics per run.
- Combines the section coefficients with empirical floor and rear-wing terms.

What it does **not** do:

- It is not a full-car, multi-element CFD solution. The total is a hybrid
  estimate.
- A real solver failure returns HTTP 503 and a failed case card. It never
  silently replaces the run with synthetic numbers.
- The mesh family is a 2D O-grid with y+ ≈ 30 wall treatment and a constant
  1.5 refinement ratio between levels. The measured GCI on this family is
  0.615 % (fine) with an asymptotic ratio of 0.985 — mesh independence is now
  demonstrated for the section Cl. See `docs/STATUS.md` for V5/V6 numbers.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health, CFD backend, solver version, readiness |
| POST | `/api/models/upload` | Upload a 3D car model (STL/OBJ/GLB) |
| POST | `/api/aero-field` | Heuristic Cp field on mesh vertices |
| POST | `/api/visualization` | Streamlines + force vectors |
| POST | `/api/simulate?method=analytical_enhanced` | Fast VLM preview |
| POST | `/api/simulate?method=cfd` | Real 2D RANS + empirical hybrid, or 503 |
| POST | `/api/optimize` | Bayesian parameter optimization |
| POST | `/api/aero-map` | 5x5 ride-height map (fast VLM path) |
| POST | `/api/advise` | AI setup recommendations |
| POST | `/api/simulate-and-optimize` | Simulate -> optimize -> advise |

Interactive docs: <http://localhost:8000/docs>

## Validation and Dataset

```bash
.venv\Scripts\python.exe validation\v5_grid_independence.py
.venv\Scripts\python.exe validation\v6_cfd_benchmark.py
.venv\Scripts\python.exe validation\v7_surrogate_cv.py
.venv\Scripts\python.exe validation\v8_multifidelity_gain.py
```

The V5/V6 scripts run real meshes/solves and report measured results, including
failure to meet historical targets. The V7/V8 surrogates use
`data/datasets/aero_surrogate_v1.parquet`, which is now a 28-point real-CFD
dataset (OpenFOAM WSL, medium mesh). The pre-existing synthetic dataset is kept
at `data/datasets/aero_surrogate_v1_synthetic_backup.parquet`.

To regenerate the real dataset (about 25-30 minutes for 28 serial medium cases):

```bash
.venv\Scripts\python.exe scripts\generate_real_dataset.py --samples 28 --max-workers 1
```

Failed solver points are retained with `hf_failed=True` and NaN high-fidelity
targets; they are never filled with invented numbers.

## Architecture

```
Setup Config -> Simulator (analytical / VLM / real CFD hybrid)
                    |
              Force Results (DF, drag, balance, L/D)
                    |
         Optimizer (Optuna, VLM path) <- objective: balanced / max DF / min drag
                    |
              AI Advisor (LLM or rules)
                    |
           Recommendations + optimized setup
```

## Tunable Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Front wing main/flap angle | -5 to 35 deg (UI narrower) | Front downforce, balance |
| Rear wing main/flap angle | -5 to 35 deg (UI narrower) | Total downforce, drag |
| Ride height F/R | 10-70 mm | Ground effect sensitivity |
| Rake | 0-3.5 deg | Diffuser expansion, rear downforce |
| Diffuser angle | 8-18 deg | Underbody downforce |
| Gurney flaps | 0-30 mm | Local lift boost |

## LLM Advisor

Set `OPENAI_API_KEY` in `.env` for GPT-powered advice. The advisor also works
with any OpenAI-compatible API (Ollama, Azure, etc.) via `LLM_BASE_URL` and
`LLM_MODEL`. Without a key, a rule-based advisor provides recommendations. When
an optimization result is present, the rule-based advice describes the optimized
forces/balance and labels the improvement as a normalized objective score; the
lap delta remains the UI's same-fidelity relative proxy.

## Honest Limitations

- The graphical Cp field is a heuristic visualisation, not a CFD solution.
- The real CFD tier is a 2D section simplification with empirical floor/rear
  terms; it is labelled `rans_hybrid`.
- V4 misses the external Zerihan/Zhang ground-effect reference. V5 is not in
  the asymptotic range. V6 Part B is trend-only because the section geometry
  differs from the cited wing.
- The 28-point dataset has only 4 solver-converged points; V7/V8 targets are
  partly unmet and the scripts report that result rather than forcing a pass.
- The lap-time delta is a relative proxy within the same fidelity family, not a
  predicted lap time.
- The surrogate is used by the validation scripts, not yet by the live UI.

See `docs/STATUS.md` and `docs/decisions/` for measured numbers and decisions.

## Next Steps

- Full-car or multi-element CFD with a validated mesh family
- Calibrate the VLM/CFD balance split against wind-tunnel data
- Wire the surrogate and its uncertainty into the live optimizer
- Track-specific lap simulation with real centerline data
- Validate the hybrid totals against a full-car reference
