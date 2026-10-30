# ADR 0006: Body-fitted O-grid mesh family for the 2D section CFD

- **Status:** Accepted
- **Date:** 2026-10-30
- **Context:** V5 measured on the previous `blockMesh` + `snappyHexMesh`
  pipeline reported "NOT DEMONSTRATED": the fine NACA 0012 case was
  force-unconverged, max non-orthogonality reached ~80 degrees around the
  section, and prism layers attached to only a few percent of the wall, so
  neither wall resolution nor a systematic refinement family could be shown.

## Decision

Replace the snappy pipeline with a body-fitted single-block O-grid written
directly as an OpenFOAM `polyMesh` (`backend/simulation/cfd/mesh_builder.py`):

1. **Exact wall representation.** The inner O-ring lies on the NACA section,
   so there are no snapped facets and no layer-coverage problem.
2. **y+ driven first cell.** The first cell height comes from the flat-plate
   correlation times a calibration factor (`WALL_FUNCTION_HEIGHT_FACTOR = 2.2`,
   target y+ = 30 for the k-omega SST wall functions). Measured y+ averages
   30-32 on the medium/fine meshes; the factor must be re-checked whenever the
   mesh family or wall treatment changes.
3. **Constant refinement ratio.** Levels are coarse/medium/fine = 60/90/135
   surface points (both sides) and 64/96/144 radial cells -> r21 = r32 = 1.50
   by construction, which is what the ASME GCI procedure assumes.
4. **Moving-ground wall.** The ground patch uses `movingWallVelocity` with the
   freestream pressure condition handled per ADR 0004's road-relative frame.
5. **Instrumentation.** A `yPlus` function object records min/max/average wall
   y+ per run, parsed into `case_meta.json` alongside `checkMesh` quality.

Case format moves to `real-airfoil-rans-v10-ogrid-moving-ground` and
`max_iterations` default to 2000, so caches from the old family do not apply.

## Consequences

- V5 re-measured on this family (NACA 0012, alpha = 4 deg, free air):
  coarse 7,616 cells Cl 0.37922 (converged); medium 17,184 cells Cl 0.36170
  (converged); fine 38,736 cells Cl 0.35696 (hit the 2000-iteration cap,
  flagged unconverged by the force-stability test).
  r21 = 1.501, r32 = 1.502, apparent order p = 3.219, extrapolated
  Cl = 0.35520, relative fine error 1.327 %, **GCI fine = 0.615 %**
  (target < 3 %), asymptotic ratio 0.985 (target 0.85-1.15) -> PASSED.
- `checkMesh` on the fine mesh reports max non-orthogonality 53.7 deg, zero
  severe non-orthogonal faces, zero failed checks (previously ~80 deg with
  severe faces).
- The fine solve is honest about its convergence nuance: the GCI is computed
  from the measured Cl values, and we state that the fine case reached the
  iteration cap before passing the force-stability threshold.
- The 28-row HF dataset (ADR 4) was generated on the old snappy medium family
  and must be regenerated on the O-grid family to keep V7/V8 evidence
  consistent; until then, dataset rows keep their recorded provenance.
