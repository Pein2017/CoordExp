# Completed density2048 control

Scientific disposition: mitigates the matched narrow-data CE degradation,
but no positive result over original Source on the registered dev128 outcome.
Technical disposition: lead-accepted completed training, checkpoint integrity,
three natural inference artifacts, and repaired full reduction. No promotion.

| Fixed panel / checkpoint | Source IoU50/60/80 owners | Density2048 | Same-step old full-CE |
|---|---|---|---|
| Dev128 / step64 | 614 / 585 / 451 | 608 / 583 / 452 | 609 / 578 / 444 |
| Dev128 / step256 | 614 / 585 / 451 | 611 / 578 / 449 | 596 / 563 / 418 |
| Original train256 / step256 | 1259 / 1190 / 908 | 1319 / 1239 / 943 | 1394 / 1317 / 1046 |

Dev denominators are128 images /891 annotated owners. At terminal IoU50,
the broader course is+15 owners versus same-step old CE and-3 versus Source
(19 gains,22 losses). This is consistent with narrow-cohort repetition being
part of the original degradation, not proof of a unique mechanism or an
intrinsic CE limitation. The intervention jointly changes breadth and repeat
exposure:2048 images /15296 owners repeated8 times instead of256 /1955 repeated64
times. Image presentations, updates, optimizer recipe and full DoRA are matched;
annotated-owner and target-token presentations are not exactly matched.

All128 dev decodes stop naturally at both points, with zero parser/score failures.
Terminal valid prediction count is1159 versus1103 for Source; strict physical
duplicate candidates rise26 versus9. There are9 dropped predictions versus58
for Source. Valid unmatched predictions are548 and remain unknown, not false
positives or hallucinations. The fixed train256 panel has1 length-capped decode
versus4 for Source, and is not the whole2048 training population. Full threshold,
density, gain/loss and common-owner geometry evidence remains in the aggregate.

Evidence scope is this single seed and historically used development panel,
not untouched-test generalization, repeatability or complete-scene precision.
The registered course is closed without selecting another dose or checkpoint.

## Receipts and reducer recovery

- Authoritative aggregate: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/analysis-density2048-v2/aggregate.json
- Aggregate SHA-256:6440e2d1f0884617ff3dc1457d0202c01ad09e332f4e59499dff70359fa1e5b0.
- Checkpoint acceptance: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-06-ce-controls-rloo-successor/native-ce/checkpoint-acceptance-density2048-v1.json.
- All256 training updates were finite/applied, all four checkpoints have the
  exact588 Source tensor keys with196 changed A/B/m components, and selected
  embeddings are byte-identical to Source. Natural reads remain FP32/SDPA,
  unmerged DoRA, world8/B2/max3084/greedy with frozen canonical geometry.
- The first reduction failed after inference because its helper expected a
  whole plan while its caller passed the resolved baseline table. No aggregate
  was published and no scientific claim used it. The two-line helper correction
  retains same-step pairing; a real-data full-reducer self-comparison failed
  before correction and gives zero old-CE delta at every point afterward.
  All5 focused tests and the actual density2048 CLI reduction passed. Only
  reduction was rerun; inference and training artifacts were not replaced.
