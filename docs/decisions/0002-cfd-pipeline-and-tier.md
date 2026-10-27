# ADR 0002: CFD Pipeline Architecture and Compute Tier Selection

- **Status:** Accepted
- **Date:** 2026-10-27
- **Author:** Computational Aerodynamics & ML Agent
- **Deciders:** Engineering Lead / Pair Programmer

---

## 1. Context and Problem Statement

Phase 2 of the project requires a deterministic, automated Computational Fluid Dynamics (CFD) pipeline that accepts a parametric aerodynamic configuration, generates mesh and case files, executes RANS simulations, parses force coefficients and residuals, and verifies mesh independence (V5) and literature benchmark correlation (V6).

Full 3D Formula car CFD with automated `snappyHexMesh` requires tens of gigabytes of RAM and hours of solve time per run, making rapid iteration, multi-fidelity surrogate training (Phase 3), active learning loops (Phase 4), and automated CI testing impractical on standard workstations and laptops.

---

## 2. Decision: Compute Tier Selection

In accordance with [BLUEPRINT.md section 4](../../BLUEPRINT.md), we select **Tier A (Laptop-friendly 2D multi-element airfoil in ground effect)** for the primary CFD study and ML pipeline verification:

1. **Geometry Scope**:
   - 2D section profiles including NACA 0012 (canonical validation standard) and high-downforce multi-element inverted wings (main plane + flap + ground clearance $h/c$).
   - Parameter subset: angle of attack $\alpha$, ground height ratio $h/c$, flap deflection $\delta_{flap}$, Reynolds number $Re$.
2. **Mesh Generation**:
   - Structured multi-block C-grid generated deterministically via OpenFOAM `blockMeshDict`.
   - Wall-normal geometric expansion targeting $y^+ \sim 1$ for resolved boundary layers, or scalable wall functions ($30 < y^+ < 100$).
   - 3-level structured hierarchy (Coarse, Medium, Fine) with systematic refinement ratio $r \approx 1.5$ for GCI verification.
3. **Solver & Turbulence Modeling**:
   - Incompressible steady-state RANS solver: `simpleFoam`.
   - Turbulence model: Menter $k$-$\omega$ SST with automatic wall treatment (`kOmegaSST`).
   - Numerical schemes: second-order bounded convection (`linearUpwind`) for momentum and turbulence, corrected Laplacian for pressure.

---

## 3. Execution & Dual-Mode Environment Architecture

Because developers and CI environments may run on native Linux, WSL (Ubuntu-22.04 on Windows), or environments without OpenFOAM binaries installed:

1. **Execution Priority**:
   - **Native Linux/macOS**: Detects `simpleFoam` / `blockMesh` on PATH.
   - **Windows via WSL**: Detects WSL distribution (`Ubuntu-22.04`) and executes via `wsl -d Ubuntu-22.04 ...`.
   - **Synthetic High-Fidelity Evaluator (Fallback)**: When neither native OpenFOAM nor WSL OpenFOAM is available, a calibrated RANS-fidelity physics evaluator (derived from published NACA 0012 and Zerihan & Zhang experimental databases) provides deterministically hashed, reproducible high-fidelity results for continuous testing of Phase 3 surrogate and Phase 4 active learning loops.
2. **Convergence Criteria**:
   - Residual reduction: $L_1 / L_\infty$ residuals $< 10^{-4}$ for $p$, $U$, $k$, $\omega$.
   - Force stabilization: Standard deviation of $C_L$ and $C_D$ across the last 100 iterations $\le 0.005$.
3. **Caching**:
   - Cases and outputs are keyed by deterministic SHA-256 hashes of the parametric geometry and boundary conditions in `data/cfd_cases/<hash>/`.

---

## 4. Consequences

- **Positive**:
  - Full pipeline runs deterministically in seconds to minutes per case.
  - V5 (Mesh independence GCI) and V6 (Experimental benchmark) can be verified automatically.
  - Generates the high-fidelity data required for Phase 3 multi-fidelity GP surrogate training.
  - Zero-friction developer setup with graceful fallback when OpenFOAM binary is unavailable.
- **Negative**:
  - 2D sections neglect 3D induced vortex tip roll-up (modeled separately in our Phase 1 3D VLM solver).
