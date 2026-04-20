# Human Findings

These notes summarize the human interpretation that was provided after reviewing the bbox-audit panels.

## Case 1: `1000:9:4->5`

- `baseline_pred` is a correct localization for one person.
- `chosen_gt_next` also localizes that same person.
- `source_anchor` is also correctly localized, but it is the other left-side person.

Implication:

- This case is not a clean semantic wrong-instance failure.
- The selected pred-centered basin appears to be centered on a correct instance.
- It should be interpreted as evidence that a strong pred-centered basin does not automatically imply wrong-instance collapse.

## Case 2: `776:4:0->2`

- `baseline_pred` localizes the lower-left teddy bear.
- `chosen_gt_next` localizes the middle teddy bear.
- `source_anchor` localizes the upper-right teddy bear.
- The scene contains exactly three teddy bears, so all three boxes land on different valid same-class instances.

Implication:

- This is a genuine same-class wrong-instance failure.
- But the failure does not reduce to a simple copy of the immediate `source_anchor`.
- Instead, the scene supports a stronger repeated-object competition interpretation: the model can be pulled toward an alternative same-class local basin that is neither the GT target nor the previous anchor instance.

## Takeaway

- The human review currently supports a mixed picture rather than a single mechanism.
- Some pred-centered basins are semantically correct and simply sharper around the model's own estimate.
- Other pred-centered basins are true wrong-instance failures in repeated-object scenes.
- Therefore, bad-basin analysis should keep separating:
  - semantically correct pred-centered basins
  - wrong-instance repeated-object basins
  - source-anchor-driven shifts
