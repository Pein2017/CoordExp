# Natural-continuation quick check: no net owner benefit

Date: 2026-09-08. Technical status: verified. Scientific status: negative under
the frozen clean-benefit rule. Architecture/training promotion: none. Closed.

## Decision

One block-27 direction graft changed the immediate coordinate token on **all
13 images**, but produced **5 gained / 5 lost owners, net 0** at primary
IoU >=0.50. Annotation-relative FP increased by 7 and strict duplicate
candidates by 1. Local coordinate-choice transfer therefore did not yield
clean net owner recovery in this fixed-prefix natural-continuation pilot.

Do not reinterpret this as either a universal impossibility result or evidence
for a useful natural-generation method. The user-requested quick check is
finished; no layer/dose search, further training or successor is queued.

## Exact comparison

The [frozen unit](unit.md) reuses all 13 Human13 training images and the same
Source step-2444 and magnitude-only CE-overfit checkpoints as the
[radius-direction study](../2026-09-08-logit-lens-radius-direction/results.md).
For each image the preselected site is the first coordinate of the middle
completed row in the saved overfit trajectory. No outcome-based reselection.

Both Source branches receive the exact same generated prefix ending before
that token; overfit sees exactly this prefix/image, with no future token.
Baseline continues natively. Treatment receives overfit direction at Source
radius once at decoder output 27, current position only. The hook then removes
itself. Each branch freely follows its own tokens with normal KV cache; the
one-time intervention's downstream cache consequences are retained.

Greedy FP32/SDPA, RP1.0; total output cap 768 including the common prefix.
All 26 scientific branches ended with im_end before the cap. This is natural
continuation **conditional on an overfit-generated prefix**, not native Source
generation from the original prompt.

Canonical compact-object parser, integer-bin-to-pixel GT conversion and the
existing category-compatible one-to-one matcher score complete prefix plus
continuation. The 392 GT owners are human-refined annotations; stable identities
are `(image_id, coco_ann_id)`, including negative human-added IDs. Shared
prefix owners are not counted as new gains; gains/losses compare owner sets.

## Aggregate results

| IoU threshold | Baseline TP / FP / FN | Direction graft TP / FP / FN | Gained / lost | Net TP |
| --- | --- | --- | --- | ---: |
| >=0.50 (primary) | 277 / 39 / 115 | 277 / 46 / 115 | 5 / 5 | 0 |
| >=0.60 (descriptive) | 268 / 48 / 124 | 267 / 56 / 125 | 4 / 5 | -1 |
| >=0.80 (descriptive) | 236 / 80 / 156 | 235 / 88 / 157 | 0 / 1 | -1 |

Valid prediction count rises from 316 to 323, not unique owners. Strict
duplicate candidates rise from 7 to 8: unambiguous same-category GT attribution
at IoU >=0.5 plus an earlier prediction-pair pixel IoU **strictly >0.95**.
Both arms have zero dropped/invalid predictions and zero cap hits. These FP
and duplicate counts are annotation-relative geometry metrics, not manual
hallucination/physical-identity adjudications.

Primary per-image net: 3 positive, 3 negative, 7 tied. Changed-owner IDs:

| Image | Gained annotation IDs | Lost annotation IDs | Net |
| --- | --- | --- | ---: |
| 1584 | 541559 | 540946, 542102 | -1 |
| 4134 | none | -95 | -1 |
| 7511 | -169, -171 | none | +2 |
| 13348 | 1198133 | none | +1 |
| 14038 | none | -131, -140 | -2 |
| 16228 | 1441803 | none | +1 |

Other seven images have identical matched-owner sets. All 13 suffix token
sequences differ, but token divergence does not imply beneficial owner change.

## Interpretation

**Observation:** Immediate coordinate decisions change consistently; unique
owner gains and losses cancel, while unmatched predictions and strict repeats
increase. **Inference:** The tested single local intervention propagates into
free continuation but is insufficient for a clean net owner benefit.

The pilot does not identify whether each loss comes from subsequent coordinates,
row choice or stopping; no further mechanism decomposition is needed for its
negative acceptance decision. It cannot establish held-out generalization,
training dynamics, full natural-rollout coverage from the prompt, or the value
of other interventions. Donor training-set fitting remains a limitation even
for the individual positive cases.

## Reproducibility and acceptance

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-logit-lens-natural-continuation`.

- Raw: `run-v1/receipt.json`; 13 per-image inputs/states/trajectories/mechanics.
- Independent CPU reduction: `analysis-v1/summary.json` and
  `analysis-v1/gt_vs_pred.jsonl`, produced by local `analyze.py`.
- Runner: `scripts/research/probe_logit_lens_natural_continuation.py`.
- Lead replay: 8 runner tests + 3 reducer tests passed. Duplicate-boundary
  sensitivity rejects an inclusive >=0.95 mutation.
- Lead checked all 53 execution artifact hashes, all 13 image receipts/checks,
  current runner identity, one-shot hook implementation, exact-prefix assembly,
  independent scoring and gained/lost arithmetic.
- 13 scientific pairs plus one reused Image2299 self-graft smoke; 13 direction
  hooks, 1 self hook; smoke self-graft continuation exactly equals baseline.
  Native/text and first-token cache parity passed; all generated follow-ups
  use normal KV cache and receive no additional graft.
- 26 full-model + 26 direct-text no-cache captures; 2,406 generation forwards
  and new tokens including self smoke; 53 vision calls.
- Runtime 240.78 seconds; peak GPU allocation 18,195,857,408 bytes; peak RSS
  14,229,080 KiB; artifact payload 328,706,542 bytes excluding top receipt.
  One attempt, no training or GPU rerun; GPU0 returned to 3 MiB.

Exact evidence hashes and final disposition: [lead-acceptance.json](lead-acceptance.json).
