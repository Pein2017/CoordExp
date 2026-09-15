---
id: decision.prefer-x-then-y-object-ordering
type: decision
status: active
updated: 2026-08-05
topic: eight-coordinate-bbox-supervision
evidence:
  - research/investigations/eight-coordinate-bbox-supervision/results.md
relations:
  supports: []
  narrows: []
  supersedes: []
---
# Prefer X-then-Y Ordering for Sorted Detection Targets

## Decision

For future sorted CoordExp detection experiments, prefer four-coordinate
`x1 y1 x2 y2` targets ordered by top-left x then y through an explicit
`geo_sorted_xy` key. Preserve legacy `geo_sorted` as y then x for compatibility.
Do not promote redundant eight-coordinate bbox supervision as the default
axis-aligned representation.

## Evidence

- The new four-coordinate x-then-y run reached `0.434394` val-200 COCO average
  precision, compared with `0.411179` in the operator-owned old step-917
  benchmark and `0.415552` in the old eight-epoch y-then-x run.
- The eight-coordinate x-then-y run reached `0.430679`, close to but slightly
  below the new four-coordinate result.
- Of 1,407 eight-coordinate predictions, 1,405 were exact rectangles and the
  remaining two deviated by only one coordinate bin, showing that redundancy
  was learnable but not observably advantageous.

## Belief Update

The original compound result should no longer be interpreted mainly as an
eight-coordinate representation gain. The matched-direction four-coordinate
result makes object serialization order the stronger explanation. The shortest
representation remains the better default research candidate unless a task
requires general quadrilaterals.

This decision updates the research route only. It does not authorize a product
default, config migration, parser change, or removal of legacy behavior.

## Next Discriminator

If exact attribution becomes decision-critical, run one matched four-coordinate
ablation that changes only y-then-x versus x-then-y ordering. Otherwise, reuse
the current result and spend training budget on downstream localization or
enumeration questions rather than another eight-coordinate arm.
