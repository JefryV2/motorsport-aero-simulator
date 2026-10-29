# ADR 0005: Split V4 into a potential-flow check and a pending external reference

- **Status:** accepted
- **Date:** 2026-10-29
- **Phase / task:** Phase 2 follow-up, V4 reporting

## Context

`validation/v4_ground_effect.py` was titled as a comparison against Zerihan &
Zhang (2000), but its pass/fail targets were the potential-flow image-system
augmentation (Tuck 1975 / Wieselsberger), hardcoded as 1.120, 1.280, 1.560,
2.000, and 2.550. The same file reported PASS while the measured 2.585 at
h/c=0.15 is roughly 1.93x the published experimental 1.34. The test suite was
green because it checked the self-consistent targets, not the external data.

## Options considered

1. **Leave the script as-is and rely on STATUS.md to explain the gap.**
   Pros: no code churn. Cons: the test name and printed header still claim an
   external validation that is not performed; the plan's honesty rules forbid
   naming a check after something it does not do.
2. **Fit or loosen the targets until the external curve matches.**
   Pros: a green external test. Cons: fabricates agreement and violates the
   rule against tuning tests to pass.
3. **Split V4 into V4a (potential-flow self-check, pass/fail) and V4b
   (external reference, PENDING until digitized data and matching geometry
   exist).**
   Pros: honest; keeps the useful self-consistency test; makes the missing
   external evidence explicit. Cons: the external claim stays open.

## Decision

Option 3. The script's pass/fail criterion is now the potential-flow trend
(monotonic, <6% against the image-system targets). The Zerihan & Zhang values
are printed as an indicative gap only and are returned with
`external_reference_pending=True`. `validation/references.md` and
`docs/STATUS.md` record V4b as PENDING rather than passing.

## Consequences

- V1-V4 tests still pass, but V4 no longer implies external accuracy.
- Phase 2/3 acceptance must not cite V4 as external validation.
- To close V4b we need digitized Zerihan & Zhang data and a geometry that
  matches the referenced wing, or a different cited benchmark.
- V2's low-AR tolerance is documented at the same time: the tight 3.5% target
  applies to AR>=4; AR=2 is reported at its measured 15.4% error.
