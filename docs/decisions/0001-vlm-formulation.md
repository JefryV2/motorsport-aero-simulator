# ADR 0001: Vortex Lattice Method Formulation and Viscous Profile Drag Coupling

- **Status:** accepted
- **Date:** 2026-10-27
- **Phase / task:** Phase 1, Tasks P1.1 - P1.5

## Context

The prototype `backend/simulation/panel_method.py` was identical to `analytical.py` (confirmed in Step 0 as Bug B1). To establish a true mid-fidelity physics solver for wings in ground effect and multi-element interaction, a 3D Vortex Lattice Method (VLM) is implemented.

Key technical choices:
1. Downwash / induced drag computation method.
2. Ground effect modeling approach.
3. Viscous profile drag coupling.
4. Multi-element wing coupling.

## Options Considered

### 1. Near-field panel pressure integration vs Far-field Trefftz plane analysis
- **Option A (Near-field integration):** Integrate $C_p$ over panel surfaces. Highly sensitive to panel geometry, edge singularities, and chordwise resolution; can yield oscillatory or negative induced drag.
- **Option B (Trefftz plane analysis) [CHOSEN]:** Evaluate downwash $w_{\text{ind}}(y)$ and circulation $\Gamma(y)$ in the downstream crossflow plane ($x \to \infty$). Mathematically guarantees strictly non-negative induced drag $C_{Di} \ge 0$ and consistent Oswald efficiency $e$.

### 2. Ground effect: Discretized ground plane vs Image vortex system
- **Option A (Discretized ground):** Mesh the track surface with thousands of panels. Drastically increases matrix dimension ($N^2$) and compute time ($>10\times$).
- **Option B (Mirror image system) [CHOSEN]:** Enforce exact zero-normal-flow on ground plane ($z = 0$) by mirroring every bound and trailing vortex across $z = 0$ with opposite circulation $-\Gamma$. Zero extra panel unknowns; exact boundary condition.

### 3. Viscous Profile Drag ($C_{Dp}$)
- **Option A (Pure potential flow):** Zero viscous drag (d'Alembert's paradox). Unphysical L/D ratios.
- **Option B (Section polar coupling) [CHOSEN]:** Add viscous section profile drag $C_{Dp}(Re, C_L)$ using standard inverted high-lift racing foil polars ($C_{Dp,0} \approx 0.012$ + lift-dependent term), with Reynolds scaling.

## Decision

Adopt Option B across all three choices: Trefftz plane induced drag, mirror image ground effect, and section polar viscous drag coupling.

## Consequences

- Real 3D vortex wake, induced drag, and multi-element aerodynamic coupling are now solved.
- Solver completes in $< 50\,\text{ms}$, fast enough for real-time frontend interaction and optimization loops.
- Attached-flow limits are tracked; local panel $C_L > 2.2$ issues flow separation warnings.
- The baseline analytical model is preserved intact and protected by `tests/golden/test_baseline_golden.py`.
