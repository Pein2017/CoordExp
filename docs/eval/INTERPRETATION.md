---
doc_id: docs.eval.interpretation
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Distinguish detection metrics, assignment policies, category namespaces and annotation-relative errors.
tags: [eval, coco, matching, metrics, diagnosis]
updated: 2026-09-09
---

# Detection metric interpretation

Use this page when interpreting or changing a detection metric. The
[stable evaluator contract](../../openspec/specs/coordexp-infras-detection-evaluator/spec.md)
owns supported pipeline behavior; the research unit owns any different scientific
metric. This page explains distinctions without choosing a universal matcher.

## Name the measured quantity

| Quantity | Required interpretation |
| --- | --- |
| COCO AP / AR | Use the actual scored prediction ordering and evaluator configuration, including IoU, area, max detections, category and crowd/ignore scope. AP is not fixed-threshold F1. |
| One-to-one owner recall | Matched annotated owners divided by the declared GT population at a specified category/IoU/matching policy. |
| Precision / F1 | Define eligible predictions, GT, TP/FP/FN, dropped/empty cases and aggregation. Micro counts and a mean of per-image ratios need not agree. |
| Match rate | State the numerator and denominator; prediction match rate and GT owner recall answer different questions. |
| Owner-set gain | Report gains and losses against the same baseline/population. Gains alone hide owner replacement; token changes alone do not imply owner changes. |

An empty valid model output and a technically invalid/missing run have different
meanings. Follow the frozen denominator and missing-case policy rather than
silently dropping difficult images.

## Three matching owners

- [Research global assignment](../../src/eval/assignment.py): category-constrained,
  cardinality first, then quantized IoU and deterministic ties. It expects the
  caller's category normalization and geometry; it does not normalize labels.
- [Visualization matching](../../src/vis/matching.py): greedy matching for its
  declared overlay contract. Do not infer research metrics from overlay colors.
- [COCO consumer](../../src/eval/detection_consumer.py): converts the supported
  scored artifact and calls `pycocotools.COCOeval`. Its score-ranked metric is
  not the research global assignment objective.

The [two-owner counterexample](../../tests/eval/test_assignment.py) yields two
matches under global assignment and one under the retained greedy visualization
rule. “Hungarian-like” is not a sufficient specification: state the objective,
threshold, tie/category rules and unmatched treatment. Replacing an algorithm
can change the scientific result even when every box is unchanged.

## Geometry, labels and COCO namespaces

Bind image dimensions and `xyxy`/`xywh`/coordinate-bin interpretation before IoU.
The current Swift consumer converts inline GT bins to pixels and consumes scored
prediction boxes already in pixels. Do not apply the conversion twice.

The [category registry](../../src/eval/detection_categories.py) exposes separate
evaluator-local and official COCO IDs. For example, `stop sign` is local 12 but
official 13; `bottle` is local 40 but official 44. The
[namespace tests](../../tests/eval/test_detection_categories.py) preserve this
distinction. Export against the target annotation registry; never infer official
IDs from contiguous class indices. Current V1 name normalization does not
authorize arbitrary alias expansion.

The current direct consumer constructs a COCO-shaped GT from inline objects and
sets their `iscrowd` to zero. It uses the COCO metric implementation but does not
thereby recover crowd, ignore or missing-object information discarded upstream.
An official-dataset claim must verify raw annotation/category scope and the
appropriate export path; see [submission scope](COCO_TEST_SUBMISSION.md).

## Explain an unmatched row before naming its cause

Effective 2026-09-16, use the project-wide
[TIDE-aligned unmatched review vocabulary](UNMATCHED_REVIEW.md) for new analyses.
It separates reference-relative `Cls/Loc/Both/Dupe/Bkg/Miss`, physical evidence,
geometry, detector-proxy support and training admission. This adds diagnostic
terminology; it does not replace frozen matching or make detector output GT.

Keep these distinct: malformed object span; invalid geometry; valid box below the
IoU threshold; one-to-one competition for an already matched owner; class mismatch;
and missing/ambiguous annotation. Near-duplicate boxes and multiple boxes referring
to one owner are also different definitions. State dedup order and threshold.

Unmatched to annotations is not sufficient evidence of hallucination. Inspect
entity/category separately from geometry, with image/crop evidence when needed.
Model-generated review labels are candidates, not replacement ground truth.
Do not rename a parser-valid unmatched row simply “invalid”: that can suggest
the wrong loss or intervention.

For a counter change, retain parser/drop and population counts and compare the
actual owner sets before recomputing derived metrics. Follow
[the evaluation workflow](WORKFLOW.md) for execution; no model rerun is implied
by a correction confined to a derived evaluator with complete valid raw inputs.
