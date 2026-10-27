# Motorsport Aero Simulator

Formula open-wheel aerodynamic simulator with Bayesian optimization and AI-powered setup advice.

## Features

- **Three simulation methods**
  - `analytical` — fast semi-empirical model (wings + ground effect + diffuser)
  - `panel_vlm` — vortex lattice method for wings (recommended)
  - `cfd` — OpenFOAM integration (falls back to analytical if OpenFOAM not installed)

- **Bayesian optimizer** (Optuna TPE) — tunes wing angles, ride height, rake, diffuser, gurney flaps

- **AI advisor** — LLM-powered recommendations (OpenAI-compatible API) with rule-based fallback

- **Web UI** — interactive 3D viewer with STL/OBJ/GLB import, real-time Cp pressure map, streamlines, and force vectors

## Quick Start

```bash
cd motorsport-aero-simulator
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # optional: add OPENAI_API_KEY for LLM advisor
python -m uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

Open http://localhost:8000 — drag & drop your car model (STL/OBJ/GLB) or use the built-in formula car.

## 3D Visualization

| Feature | Description |
|---------|-------------|
| **Model import** | Drag & drop or browse STL, OBJ, GLB/GLTF files (up to 50 MB) |
| **Pressure map** | Real-time Cp coloring — blue = suction/downforce, red = pressure/drag |
| **Streamlines** | Airflow paths deflected around the car |
| **Force vectors** | Red = drag, teal = downforce arrows scaled to magnitude |
| **Live updates** | Move any slider — pressure map updates instantly |
| **Optimize** | Bayesian search finds best setup, updates 3D view + AI advice |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + available methods |
| POST | `/api/models/upload` | Upload 3D car model (STL/OBJ) |
| POST | `/api/aero-field` | Cp field on mesh vertices |
| POST | `/api/visualization` | Streamlines + force vectors |
| POST | `/api/simulate?method=panel_vlm` | Run aero simulation |
| POST | `/api/optimize` | Bayesian parameter optimization |
| POST | `/api/advise` | AI setup recommendations |
| POST | `/api/simulate-and-optimize` | Full pipeline: simulate → optimize → advise |

Interactive docs: http://localhost:8000/docs

## Architecture

```
Setup Config → Simulator (analytical / VLM / CFD)
                    ↓
              Force Results (DF, drag, balance, L/D)
                    ↓
         Optimizer (Optuna) ← objective: balanced / max DF / min drag
                    ↓
              AI Advisor (LLM or rules)
                    ↓
           Recommendations + optimized setup
```

## Tunable Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Front wing main/flap angle | -2° to 20° | Front downforce, balance |
| Rear wing main/flap angle | 0° to 30° | Total downforce, drag |
| Ride height F/R | 10–70 mm | Ground effect sensitivity |
| Rake | 0° to 3.5° | Diffuser expansion, rear DF |
| Diffuser angle | 8° to 18° | Underbody downforce |
| Gurney flaps | 0–30 mm | Local Cl boost |

## OpenFOAM CFD (optional)

Install [OpenFOAM](https://www.openfoam.com/) and ensure `simpleFoam` is on PATH.
The simulator generates case files in `data/cfd_cases/` and runs CFD automatically
when method=`cfd` is selected.

## LLM Advisor

Set `OPENAI_API_KEY` in `.env` for GPT-powered advice. Works with any OpenAI-compatible
API (Ollama, Azure, etc.) via `LLM_BASE_URL` and `LLM_MODEL`.

Without an API key, a rule-based advisor provides setup recommendations automatically.

## Next Steps

- Import real car geometry (STL → OpenFOAM mesh)
- Calibrate model against wind tunnel / CFD validation data
- Add track-specific lap time simulation (Maplesoft-style)
- Train a surrogate ML model on CFD results for real-time optimization
- Multi-element DRS simulation
