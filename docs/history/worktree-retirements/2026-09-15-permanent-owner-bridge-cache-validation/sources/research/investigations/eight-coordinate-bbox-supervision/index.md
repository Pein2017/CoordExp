# Eight-coordinate BBox Supervision and Object Ordering

This investigation records the completed COCO val-200 comparison between
legacy four-coordinate targets ordered by top-left y then x, redundant
eight-coordinate clockwise targets ordered by top-left x then y, and a later
four-coordinate x-then-y run.

## Current Result

- [Closeout results and evidence boundary](results.md)
- [Research decision: prefer x-then-y ordering](../../decisions/prefer-x-then-y-object-ordering.md)

## Status

The experiment is closed. Four-coordinate `x1 y1 x2 y2` remains the preferred
representation because eight-coordinate supervision did not show an advantage
large enough to justify its doubled coordinate sequence. Future sorted runs
should treat top-left `(x1, y1)` ordering as the leading candidate while
retaining the legacy `(y1, x1)` behavior for compatibility.

This is non-normative research evidence. It does not by itself change current
configuration defaults, parsers, schemas, prompts, or stable contracts.
