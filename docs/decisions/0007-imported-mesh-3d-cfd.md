# ADR 0007: Full 3D CFD Tier for Imported Meshes

- **Status:** Accepted
- **Date:** 2026-11-05
- **Author:** Computational Aerodynamics & ML Agent
- **Deciders:** Engineering Lead / Pair Programmer

---

## 1. Context and Problem Statement

Users can upload meshes (STL/OBJ) for visualization, but every solver tier
computes forces from the parametric `FormulaAeroConfig` — an uploaded shape
affects the picture, never the numbers. ADR 0002 deliberately scoped CFD to the
2D section; assuming that scope would stay valid was never stated, and it would
now block the user's request: *the simulator should work on objects they
import*.

## 2. Decision

Add an **imported-mesh 3D CFD tier** (`backend/simulation/cfd3d/`), which
measures and solves the mesh the user actually supplied:

1. **Geometry ownership.** The uploaded triangles are the geometry. Nothing is
   inferred from the parametric car. The mesh is measured (frontal area by
   silhouette rasterization, watertight edge fraction, extents) and reported
   verbatim; out-of-envelope results raise an error (HTTP 422) rather than
   being silently rescaled or reoriented.
2. **Frame conventions.** The solver frame is +x streamwise, +y up, +z lateral
   (as in the 2D cases). `axes="three"` converts the Three.js authoring frame
   (+x lateral, +y up, +z rearward) via `solver.x = -z_three`, documented in
   `docs/COORDINATES.md`. Positive `Cl` is **downforce** (`liftDir (0 -1 0)`),
   matching car-mesh expectations.
3. **Case construction.** `snappyHexMesh` pipeline: `blockMesh` background box
   sized in units of the mesh's own extents (inlet 1.5, outlet 2.5 car
   lengths; tunnel height 2.5 car heights; sides 3.0 car widths), surface
   refinement levels + boundary layers on the `car` walls.
4. **Solver.** Steady `simpleFoam`, `kOmegaSST`, second-order bounded
   convection; `forceCoeffs` on the `car` patches; `checkMesh` records measured
   cell count and quality.
5. **No synthetic fallback.** Uniquely among tiers, an imported mesh with no
   OpenFOAM returns **503**, never invented numbers: the whole point of the
   tier is "numbers reflect *this* shape", which a synthetic model cannot
   honestly provide.
6. **Execution.** Async job API (`POST /api/cfd/3d`,
   `GET /api/cfd/3d/{job_id}`) because snappyHexMesh + RANS runs take minutes
   to hours. Cases are keyed by SHA-256 of the triangles + run parameters
   (`data/cfd_cases_3d/case3d_<hash>/`); a repeated request re-parses the
   stored logs instead of re-solving.
7. **Convergence honesty.** The shared 2D verdict machinery is reused with the
   3D streamwise `Uz` residual folded in; a stuck `Uz` residual downgrades the
   label to unconverged, it is not hidden.

## 3. Consequences

- **Positive**: uploaded shapes get real, provenance-labelled RANS forces;
  measurement-only preview (`POST /api/cfd/3d/measure`) works without a
  solver.
- **Negative**: hours-scale runtime for real cars at Medium/Fine levels; a
  box-coarse smoke solve is the expected first execution. Level Coarse is
  recommended for interactive use.
- **Validation**: V13 (Ahmed body, coarse level, single Reynolds number) is the
  benchmark gate for this tier; results are only committed to the ledger after
  a measured run, never estimated.

## 4. Alternatives Considered

- **Mesh measurement into the parametric tiers** (cheap, honest, but does not
  answer "solve my shape").
- **Point-in-flow visual-only field refinement** (visual only; forces unchanged).
- **Full-3D solve of the parametric car** (deferred; the built-in car's value
  comes from its parametric sweeps, which the 2D + VLM tiers already serve).
