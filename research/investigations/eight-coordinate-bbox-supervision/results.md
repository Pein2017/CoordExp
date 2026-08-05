---
title: Eight-coordinate bbox supervision and x-then-y ordering closeout
description: Val-200 results, geometry census, provenance, and bounded conclusions for four- versus eight-coordinate supervision.
type: investigation
role: results
authority: non_normative_research
status: complete
evidence_status: verified
updated: 2026-08-05
---
# Closeout Results

## Decision-bearing observations

1. The model learned the redundant rectangle constraints in the clockwise
   eight-coordinate target. Of 1,407 decoded quadrilaterals, 1,405 were exact
   axis-aligned rectangles. The other two disagreed across a repeated edge by
   one coordinate bin out of 1,000. Thus the model can learn the symmetry and
   equality constraints in `x1 y1 x2 y1 x2 y2 x1 y2`.
2. Eight-coordinate supervision did not establish a useful advantage over
   four-coordinate `x1 y1 x2 y2`. With x-then-y object ordering, four-coordinate
   bbox average precision was `0.434394`, while eight-coordinate mask average
   precision was `0.430679`. The `0.003715` difference slightly favors the
   shorter representation on this slice, but the training recipes were not a
   matched arity-only control.
3. Changing the sorted object sequence from top-left `(y1, x1)` to `(x1, y1)`
   is the larger and more actionable signal. The operator-owned benchmark table
   reports `0.411179` for the old four-coordinate step-917 control and
   `0.434394` for the new four-coordinate x-then-y run, an absolute increase of
   `0.023215` COCO average precision on val-200.

Here COCO average precision is the standard `AP@[0.50:0.95]` metric, commonly
called mAP in the experiment notes.

## Benchmark table

| Run | Epochs and coordinates | Ordering and objective | Val-200 COCO AP |
| --- | --- | --- | ---: |
| Old four-coordinate control, step 917 | 4 epochs, `x1 y1 x2 y2` | legacy `geo_sorted` by `(y1, x1)`, pure cross-entropy, gate 0, effective batch size 64 | 0.411179 |
| Old four-coordinate long run, step 4887 | 8 epochs, `x1 y1 x2 y2` | legacy `geo_sorted`, pure cross-entropy plus type gate 0.2, effective batch size 24 | 0.415552 |
| Eight-coordinate run, step 5529 | 8 epochs, clockwise quadrilateral | `geo_sorted_xy` by `(x1, y1)`, type gate 0.2, effective batch size 24 | 0.430679 |
| New four-coordinate run, step 2444 | 4 epochs, `x1 y1 x2 y2` | `geo_sorted_xy`, pure cross-entropy, gate 0, effective batch size 24 | 0.434394 |

The first value is preserved from the operator-supplied benchmark summary. A
locally retained legacy evaluator artifact for the same named step-917 run
reports `0.414944` instead of `0.411179`. The exact evaluator or filtering
identity behind that discrepancy was not recovered during closeout. It does
not reverse the direction of the x-then-y result: using the local value would
make the increase `0.019450` rather than `0.023215`.

## What is supported

- Redundant clockwise coordinate equality is learnable rather than routinely
  producing arbitrary quadrilaterals.
- On this slice, four- and eight-coordinate supervision are operationally
  near-equivalent in localization quality.
- There is no current evidence-based reason to pay the longer-sequence cost of
  eight-coordinate supervision for axis-aligned bbox detection.
- X-then-y sorted supervision is the preferred candidate for subsequent sorted
  training because both the four-coordinate and eight-coordinate x-then-y runs
  exceeded the two retained y-then-x results.

## What is not claimed

- The table is not a strict one-variable causal estimate of ordering. The old
  step-917 run used effective batch size 64, while the new four-coordinate run
  used 24. The old long run and eight-coordinate run also differ in coordinate
  arity, and the four- versus eight-coordinate runs differ in epochs and
  type-gate objective.
- The inference caps were not identical: the new four-coordinate run used 512
  maximum new tokens, while the eight-coordinate evaluation used 3,084. Both
  completed the 200-row decode, but this remains a recipe difference.
- Approximate performance equality does not prove identical internal geometry
  representations or generalization outside this COCO val-200 slice.
- This result does not authorize silently changing legacy `geo_sorted`
  semantics. Compatibility requires a distinct `geo_sorted_xy` key.

## Durable evidence

The compact source snapshots, completed checkpoints, resolved configs,
inference outputs, scored JSONL, metrics, data manifests, benchmark image, and
representative visualizations were moved out of the retired worktree to:

```text
/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout
```

Important content identities:

- operator benchmark image SHA-256:
  `95020fd03f9a0a703e7adadd473509952ed92e85cc767d0b42a48506c4c73f3f`;
- eight-coordinate scored JSONL SHA-256:
  `afb855fa6a65a1e02d79fb0ecc330daeb8295b2dda835157b3b4c2e7411c685a`;
- eight-coordinate metrics SHA-256:
  `ee481368eb1ebf25fd7bb3cfabc541fe2df56460bb987bd810b40e6f1180bfe7`;
- four-coordinate x-then-y scored JSONL SHA-256:
  `1f80bb1ab907f28b038bf76516f938cae14e28cf8019a50b0c09d4d3cd04cad9`;
- four-coordinate x-then-y metrics SHA-256:
  `bd049f2c57a0f725e886ea4210aa74183172424b188cc93ff57158293aad884b`;
- old four-coordinate long-run metrics SHA-256:
  `02e390b511f8665f071b38f5cfcf29136ec16530be6ea654b1465be279d5b06b`;
- locally retained old step-917 metrics SHA-256:
  `70421fd5188680e5cb1b7c73cc9b331cc22f4ce39c375ffcee658103667f6fe9`.

The evidence root contains a final `SHA256SUMS` manifest. The processed data
roots remain under `/data/CoordExp/public_data/coco/`; deleting the experiment
worktree does not delete those shared datasets.

## Next discriminator

Only run another ordering study if a decision requires a causal effect size.
That run should hold epochs, effective batch size, objective, prompt, seed,
checkpoint selection, decode cap, and evaluator identity fixed while changing
only `(y1, x1)` versus `(x1, y1)`. No further eight-coordinate training is
needed unless a downstream task specifically requires non-axis-aligned
quadrilateral output.
