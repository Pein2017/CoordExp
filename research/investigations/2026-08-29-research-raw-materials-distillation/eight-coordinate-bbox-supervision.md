---
title: Eight-coordinate bbox supervision raw-material synthesis
type: investigation
role: archival-synthesis
authority: non_normative_research
status: complete_with_image2299_dependency
updated: 2026-08-29
---

# Eight-coordinate bbox supervision

## Session context and result

Session `019fbdc9-b62a-7461-b20e-23619ebc36a0` closed the arity/order screen.
Eight-coordinate quadrilaterals were learnable, but no useful advantage over
four coordinates was established. The actionable bounded result was the
x-then-y ordering signal (`0.434394` versus retained old-order values), with no
authorization to silently change legacy `geo_sorted` semantics.

Canonical result:
`/data/CoordExp/.worktrees/coordexp-infras/research/investigations/eight-coordinate-bbox-supervision/results.md`.

## Retention exception

The Image2299 unit explicitly consumes:

`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`

as a frozen source substrate. The closeout root is only 781M, has a
`CLOSEOUT.txt` and `SHA256SUMS`, and is retained as `KEEP-DEPENDENCY` in this
pass rather than splitting a provenance bundle. It can be narrowed to the
exact checkpoint in a later, separately verified cleanup.

