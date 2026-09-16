# Co-DETR-only proxy: retained historical replay

> Historical evidence only. The three displayed-case judgments in this replay were later corrected by the user. Read [results.md](results.md) for the current accepted result; preserve this file for provenance and reproduction.

## Decision

Use Co-DETR as the preferred primary screening path without a default VLM judge.
Keep entity/category support separate from trainable box quality. The retained
single-context rule is useful for candidate nomination but is **not accepted as
an automatic GT/teacher writer**. Residual decisions belong to lead/subagent
review under the [shared vocabulary](../../../docs/eval/UNMATCHED_REVIEW.md).

This is an accepted CPU replay of historical evidence plus a registered design.
No new detector/VLM inference, independent validation or training occurred.
The 22-image SFT result remains closed and unchanged: fixed-teacher offline SFT,
first sustained saved completion128, full authorized dose256; no online rollout
correction loop.

## Retained 64-image contrast

Only remove the semantic-VLM filter; preserve detector score>=.50 and same-class
candidate/detector IoU>=.75, candidate identities and source observations.
The table uses the previously lead-adjudicated provisional visual reference.

| Method | Supported | Clean | Defective | Gray | Clean retention |
|---|---:|---:|---:|---:|---:|
| Co-DETR only | 20 | 13 | 3 | 4 | 41.9% |
| Prior Co-DETR + VL | 14 | 11 | 1 | 2 | 35.5% |

Removing VL adds6 supported candidates:2 clean,2 defective boxes and2 gray.
All3 definite detector-only errors retain positive entity/category judgments;
their error is extent. This supports separating entity discovery from geometry,
but does not establish a calibrated probability of physical correctness.

| Defective case | Agreement IoU | Reference geometry |
|---|---:|---|
| 237954:p11 | 0.8709 | too_tight |
| 307814:p0 | 0.9178 | too_loose |
| 134520:p0 | 0.9152 | too_tight |

High agreement, including IoU>.91, does not establish correct extent. The two
predictors can agree on the same too-tight or too-loose region. Raising the
agreement threshold alone is not a demonstrated solution.

Against the ORIGINAL blind reference, detector-only supports13 clean,4 defective
and3 gray; combined supports11 clean,2 defective and1 gray. Both reference versions
are retained. Historical lead adjudication happened after unblinding and checked
combined-accepted cases; this panel is not neutral fresh validation for the new
method. No reference changes were made for this replay.

The64 candidates were image-balanced and already revealed. Their counts are not
population prevalence, not a new holdout, not human gold, and not proof of
cross-prediction owner uniqueness. In particular, the result does not establish
that VL optimism caused previous errors; on this panel its filter removed both
useful candidates and defective boxes.

## Concrete follow-on candidate

Keep the detector, canonical COCO80 mapping and source-pixel frame. Cache one
full-image detection per image and add candidate-centered context crop+resize
only as needed. Compare the views after exact inverse geometry mapping; keep
all competing objects rather than just the nearest same-class box. Use the
inherited3x/min128 context as a starting recipe, not a proven optimum.

Produce separate evidence against trusted annotations and detector references:
TIDE-aligned class/localization/duplicate/background/miss candidates; owner
identity/uniqueness; edge error; view consistency; crop truncation; visibility;
and scope/annotation-convention ambiguity. Do not blindly average detector and
rollout boxes or pass crop-only detections as reference truth.

Routing:

- Consistent single-owner evidence: supported candidate, prioritized for admission.
- Supported entity with extent mismatch: localization/repair candidate; no training
  promotion of the original box.
- Multiple plausible owners, class/view conflict, prior-only tiny evidence or
  annotation ambiguity: HOLD and bounded full-image/overlay/context visual review.
- No detector support: unknown; never automatic hallucination or negative reward.

A new image-disjoint calibration should include supported/conflict/unknown strata
and a frozen independent test portion, with per-image isolation of repeated
proposals. First estimate distribution using probability sampling and weights;
only fit a more complex proxy if it improves the declared conservative operating
point. No automatic-admission threshold or safety guarantee is invented here.
An initial200–300-case calibration is a proposal, not an executed workload.

## Reproduction and acceptance

Run the bound [CPU replay script](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/replay.py) in a fresh output copy; its
`replay-v1` destination deliberately refuses overwrite. It generates detector-only
predictions before opening reference or combined decisions and verifies the
unchanged combined baseline against the old summary. New model calls:0.

[Per-case outputs](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/replay-v1/cases.json) · [Summary and exact input bindings](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/replay-v1/summary.json) ·
[Lead receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-16-codetr-only-review-proxy/lead-receipt.json).

Scientific disposition: retained-data contrast accepted; automatic teacher
admission unqualified. Implementation disposition: shared terminology registered,
replay completed, new multi-view runtime not implemented by this record.
Stop: no additional models, threshold sweep, labels or teacher writes in this unit.
