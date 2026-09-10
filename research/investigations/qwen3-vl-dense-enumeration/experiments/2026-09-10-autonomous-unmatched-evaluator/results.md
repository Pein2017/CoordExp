# Co-DETR context + independent category observation

## Decision and evidence scope

A **conservative screening proxy**, not a replacement for GT or a hard reward
oracle, is retained at a bounded operating point. The internal callable passed
a real eight-candidate consumer replay as well as its fail-closed CPU checks.

The user's already-working Co-DETR route was the useful new ingredient:
use it on a candidate-centered **3x context crop**, not only the full image,
then ask the official Qwen3-VL-8B to identify an object independently without
seeing the detector's proposed category or box edges. Candidate generation,
GT and original evaluation versions were not changed.

## Frozen method

- Existing Co-DETR ViT-L checkpoint, official MMDetection image API, COCO80
  class-index mapping, original-pixel box coordinates.
- Crop: 3x candidate width/height, minimum128 pixels/side, image-clipped.
  No box outline, candidate category or GT enters this detector.
- Candidate must agree with a same-category detector box at score>=.50,
  IoU>=.75.
- Only eligible candidates receive two8B point observations: full source image
  and fixed half-image context crop. At least one valid resolved independent
  category must match, using the small frozen alias list.
- Otherwise return **unknown**, not a negative label. No model-triggered GT
  edits, repair promotion or lower-IoU duplicate-identity ruling.

The [frozen specification](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/selected-candidate-v1.json) predates holdout
predictions. No threshold, prompt, model or case exception was changed after
unblinding. The original explicit strict IoU>.95 repeat rule remains separate.

## Development observation

The54 reused visual-review cases are development, not validation.

| Route | Accepted clean | Accepted defective | Accepted gray |
|---|---:|---:|---:|
| Full-image Co-DETR, score.50 / IoU.70 | 1 | 5 | 3 |
| 3x-context Co-DETR, same thresholds | 9 | 2 | 2 |
| Frozen context + semantic support, IoU.75 | 7 | 0 | 0 |

The change tests a combination of object magnification and altered context;
it does not isolate which causes the improvement. COCO AP does not establish
unmatched-candidate judging quality.

## One frozen image-disjoint validation

64 real candidates, one per image, sampled from91 source images outside the39
development images. The underlying pool contains268 non-strict-repeat unmatched
predictions. This is **image-balanced**, not whole-FP candidate-weighted
prevalence. The images are held out from proxy development/prompt tuning, not
claimed held out from Co-DETR or VLM training. Protected confirmation512 was
not opened.

The reference was collected once by a blinded visual-review agent, then the
lead checked all14 accepted cases. It is a **provisional visual reference**,
not human gold or new COCO annotations.

| Reference version | Accepted clean | Accepted defective | Accepted gray | Precision among definite accepts |
|---|---:|---:|---:|---:|
| Original blind reference | 11 | 2 | 1 | 84.6% |
| Lead-adjudicated uncertainty | 11 | 1 | 2 | 91.7% |

**The initial reference fails the90% criterion.** One label was downgraded
from confidently defective to **uncertain**, not upgraded to correct:
`540567:p2` shows only a white bathroom-fixture fragment at the right image
edge. The original reference incorrectly identified a separate sink at left
(the left fixture is a bathtub), and cannot establish the alleged
category error or displaced box. A second change corrects descriptive text
for the red vehicle in `540694:p3`, without changing its labels.

The [explicit overrides](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/lead-reference-overrides.jsonl),
[original evaluation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/holdout-evaluation-v1/summary.json), and
[adjudicated evaluation](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/holdout-evaluation-v2/summary.json) are all retained.
The64 predicted decisions are byte-identical across both reference versions.

### What the result supports

- Automatic support:14/64, **21.9% coverage**.
- Clearly usable:11; clear box error:1; reference-uncertain:2.
- Retention of definite clean candidates:11/31,
  **35.5%**.
- Definite-accept precision:11/12,
  **91.7%**; descriptive Wilson95%
  interval **64.6–98.5%**.
- Conservatively counting every unverified acceptance in the denominator:
  **11/14 = 78.6% verified**,
  unchanged by adjudication.
- Reference uncertainty among accepts:2/14,
  **14.3%**.
- The known error `237954:p11` captures only part of a depicted dog.
  The frame-edge teddy-bear fragment `504353:p16` remains unresolved.

All predeclared rough-use gates pass **only with the adjudicated uncertainty
treatment**. This is a fragile, small-sample screening result, not a90%
population truth guarantee. Use it to nominate supported candidates and retain
unknowns; do not use it to silently rewrite labels, penalize model non-detections,
claim benchmark accuracy, or supply unvalidated hard RL rewards.

## Measured resource cost

The64-case frozen validation used64 Co-DETR crop forwards and40 conditional8B
responses (20 geometry-eligible candidates).

- Hot amortized model work: **0.628s/candidate**.
- Sum of the two measured model initializations:
  **42.44s**.
- Sum of model-stage allocated GPU time:
  **0.0231 GPU-hours**.
- Co-DETR measured peak allocated memory:15.13GB.
- Models run sequentially in the existing `mmdet` and `ms` environments.
  No GPU service is kept resident.

These model-stage figures exclude process startup and input preparation;
they are not single-request interactive latency. The reusable callable's
separate8-case consumer took **80.824 seconds end to end**, including full
model/code fingerprint checks, process startup and input preparation. Both
stages exited0; all8 decisions matched the frozen validation outputs with zero
candidate-IoU drift. This is a cold-batch measurement, not0.63-second interactive
latency. A subsequent filename-alias preflight guard was verified RED/GREEN
and replayed on these exact8 inputs; model scripts and runtime path did not change.

Earlier failed4B/8B judging, point re-grounding and interrupted Source2B crop
runs remain evidence of their own bounded outcomes. The Grounding DINO download
was stopped when the user offered Co-DETR; its incomplete files are preserved
and no model-quality conclusion is drawn from that branch.

## Reproduction and status

All raw inputs, model/config/code identities, responses, per-case decisions,
reference versions and timing receipts live in the
[artifact root](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator). `evaluate_holdout.py` reproduces the reference comparison
without rerunning models. `profile_rule.py` does not read reference labels.

Scientific status: bounded screening evidence, no architecture promotion.
Implementation status: internal callable retained, real consumer verified.

- [Internal callable](/data/CoordExp/.worktrees/research-probes/probes/unmatched_judge/codetr_profile/runner.py)
- [Use boundary](/data/CoordExp/.worktrees/research-probes/probes/unmatched_judge/codetr_profile/README.md)
- [Real consumer receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator/consumer-smoke-v1/summary.json)

The12 focused tests cover preservation on collisions, filename aliases/path
escape, declared image identity, frozen threshold despite ambient environment,
technical failure propagation, and legitimate model abstentions. Named faults
were reproduced before correction. No shared model code or environment was
changed, no GT/training write occurred, and no model service remains resident.
