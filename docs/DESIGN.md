# Design system

Recorded so the next session can extend the UI without re-deriving it. The
current skin was audited and re-measured with the OpenDesigner skill
(`~/.config/opencode/skills/opendesigner*`, installed from
`github.com/ckryptickunal/OpenDesigner`); probe runs live in the ignored
`opendesigner/review/` folder.

## Direction

Flat engineering instrument, not a marketing page. Analogous product: a
modern race-team telemetry tool (dark, dense, single accent, data-first).
No gradients, no glows, no pill badges; hairline borders, mono numerals,
one red used for the primary action and small status marks.

## Tokens (see the `:root` block in `frontend/index.html`)

| Token group | Values | Rule |
|---|---|---|
| Surfaces | `--bg #0e0f12`, `--surface #15161a`, `--surface-2 #1b1d22`, `--surface-3 #22252b`, `--surface-4 #2a2d34` | One grey per elevation level; never invent a new grey inline. |
| Borders | `--border #2a2d34`, `--border-strong #3a3e47` | Hairlines everywhere; `--border-strong` only for interactive affordances. |
| Text | `--text #eceef1`, `--muted #a3aab5`, `--faint #8b919c` | All three pass WCAG 4.5:1 on every surface above (worst pair 4.85:1 `--faint` on `--surface-3`). Never use text below 12 px. |
| Signal | `--accent #e13018` (white text = 4.53:1), `--ok #3fb27f`, `--warn #d99a2b`, `--info #6ea8dc` | Accent: primary action and ACTIVE CFD tab only. Status colours: badges, dots, left border ticks. |
| Type | `--fs-1 12px`, `--fs-2 12px`, `--fs-3 13px` (body), `--fs-4 15px` (values) | Zero text below 12 px. Uppercase micro-labels use `--fs-1` + letter-spacing ≤ 0.06em. Mono (`--mono`) for every number a user compares. |
| Spacing | `--s1 4px`, `--s2 8px`, `--s3 12px`, `--s4 16px`, `--s5 20px`, `--s6 24px`, `--s7 32px` | Spacing must be a multiple of 4 px (2 px is allowed only for text-level offsets). |
| Corners | `--r1 3px` controls, `--r2 4px` cards, `--r3 6px` panels | No radius larger than 6 px. |
| Motion | 0.14–0.16 s ease transitions; one progress sweep, `3` iterations then static | Honour `prefers-reduced-motion: reduce` (global kill switch plus a static progress bar). Nothing decorative moves forever. |

## Verified quality bar (OpenDesigner review, 1296x932, dark)

- Home page probe: **0 errors, 0 warnings**, 4 info notes (2 px text offsets,
  the visually-hidden select's -1 px margin, ellipsis kept under review, canvas
  motion the probe cannot read).
- Every text pair measured ≥ 4.5:1; 31 form controls got explicit
  `label for=` associations; `<main>` landmark added; chips/tabs/buttons never
  wrap their labels; metric units moved into a small `.u` suffix so labels
  stay on one line.
- Audience floor: every control ≥ 24 px effective target, reachable via Tab,
  visible 2 px focus ring.

## Rules for future changes

1. One decision at a time, measured: after any visual change re-run the probe
   (`review.py plan/probe/check`) and keep the scoreboard at 0 errors/0 warnings.
2. Never hard-code a colour, size, or spacing value that exists as a token.
3. New text minimum is 12 px; new controls minimum 24 px with a focus ring.
4. Copy is neutral engineer-speak; no hype, no exclamation marks, no AI slop.
5. Method honesty precedes style: provenance banners, failed CFD cards, and
   synthetic labels must stay visually loud (left-border ticks + status colour).
