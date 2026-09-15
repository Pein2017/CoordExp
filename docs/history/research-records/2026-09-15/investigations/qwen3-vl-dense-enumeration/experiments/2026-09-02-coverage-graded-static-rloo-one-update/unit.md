---
title: Coverage-Graded Static RLOO One-Update
description: One frozen-K4 train-first update that keeps IoU50 coverage dominant while making IoU60 and IoU80 localization weakly credit-bearing.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-coverage-graded-static-rloo-one-update
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Coverage-graded static RLOO one-update

Executed result: [results.md](results.md).  The fresh unmerged candidate is
train-positive at IoU50/60/80 (`+1/+2/+3`, summed `+6`), but reproduces the
prior binary-reward behavior closely enough that the reward variants are not
behaviorally separated at this dose.

## Frozen question

> From exact frozen C, does one static K4 RLOO update whose reward is 90%
> IoU50 owner coverage and 10% higher-IoU localization improve natural-greedy
> annotated-owner behavior on the registered training surfaces relative to C?

The immediate goal is **train improvement**.  Natural behavior is read first
on the unchanged eight-image panel and then on the registered 248-image
training cohort.  Every IoU50/60/80 gain, loss, and geometry movement is
recorded; this unit has no perfection requirement.  A positive train movement
is useful evidence even if a later image-disjoint read does not improve.
Train plus image-disjoint improvement would be stronger, but image-disjoint
performance is not a precondition for retaining the train result.

Use the summed IoU50/60/80 annotated-owner hits as the compact train summary,
with all three thresholds separately visible.  This summary is not mAP.  The
unit stops after one update and the two train reads, without an in-unit reward,
dose, seed, or step sweep.

## Contrast and interpretation

- **Anchor:** exact C (`base + step-2444 embedding delta + unmerged C DoRA`).
- **Intervention:** one shared language-DoRA update from the exact frozen K4
  actions using the coverage-graded reward below.
- **Primary evidence:** C versus the fresh cold-loaded candidate under natural
  greedy decode on panel8 and train248.
- **Monitors:** per-threshold owner exchanges, matched and common-owner IoU,
  predictions, duplicates, invalid/dropped spans, EOS/caps, length, and natural
  ordering.  Unmatched valid predictions remain unknown; ordering violations
  remain legal monitor events.
- **Strongest alternative:** the prior train gain and geometry movement may be
  selected-panel noise or generic finite-step drift rather than improved
  trajectory credit.

Any positive aggregate train movement is labeled
**`TRAIN_GAIN_RECORDED_COVERAGE_GRADED_RLOO`** and may open one separately
registered image-disjoint read.  A flat or negative result is recorded and
stops this exact profile; it does not reject policy gradient, DoRA, or graded
credit generally.  An identity, finite-value, collective, persistence, or
cold-read failure is `MECHANICAL_INVALID`, publishes no scientific candidate,
and permits only a mechanics repair from immutable C.

## Immutable specimen packet

Implementation is owned by
[OpenSpec change `add-coverage-graded-static-rloo-update`](/data/CoordExp/outputs/research/restructure-research-probe-development/preservation-20260909/historical-links/c-anchored-owner-mechanism-audit/openspec/changes/add-coverage-graded-static-rloo-update/proposal.md).
All sources are inherited unchanged from the completed binary-IoU50 vertical:

```text
C adapter fingerprint:
  5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95

Base:
/data/Qwen3-VL/model_cache/models/Qwen/
  Qwen3-VL-2B-Instruct-coordexp-natural-adjacent

Embedding delta fingerprint:
  635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6

Frozen K4 analysis:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-02-current-policy-owner-set-signal-census/analysis-v3.json
sha256: 0569bdc818d69ddb221d18b57ae6ac22d7e2cb4938e3ed53e73ddbc2fd65496d

Input JSONL:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/
  train-common-base.jsonl
sha256: 86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd

C train-248 natural anchor:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-02-c-anchored-owner-mechanism-audit/full/infer/c/
  qwen3-vl-2b-c-anchored-owner-audit-c-train248/gt_vs_pred.jsonl
sha256: 9f8aa4f478883468ace45daa4a4de9b90d055938057ca8e2a1add83eff2077ed
```

The annotation authority remains
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`
at SHA-256
`4bc00be57d78d4d76e0874cd83f89329d1dc2cd471c208f3f2e30e0365dd96ee`.
The eight panel images, seeds `2026090201..2026090204`, prompts, media,
complete action tokens, appended natural `im_end`, FP32/SDPA surface, and C
composition are byte-for-byte the prior vertical's frozen specimen.  Start
again from C, never from the binary-RLOO candidate.

## Reward and intervention

For image `i`, trajectory `k`, annotated owners `O_i`, and
category-consistent global one-to-one matched-owner counts
`c_ik(τ)`:

```text
R50_ik = c_ik(0.50) / |O_i|
R60_ik = c_ik(0.60) / |O_i|
R80_ik = c_ik(0.80) / |O_i|

r_ik = 0.90 R50_ik + 0.05 R60_ik + 0.05 R80_ik
A_ik = 4/3 * (r_ik - mean_k(r_i))
L_i  = -(1/4) * sum_k A_ik * sum_t log pi_C(a_ikt | h_ikt)
```

Offline exact rescoring supplies the reason for this single frozen profile.
Across all 218 complete-clean K4 groups, it changes 132 advantage vectors,
breaks 252 of 852 IoU50-tied trajectory pairs, and activates 30 previously
flat groups, with zero observed preference for fewer IoU50 owners.  Its global
advantage cosine to binary IoU50 is `0.997424`.  Equal averaging of the three
thresholds breaks the same ties but reverses IoU50-count preference in 14 of
456 differing-count pairs, so it is not used.  These are offline design facts,
not evidence that the update will succeed.

Reuse the existing one-image-per-rank, eight-rank replicated DDP path: 32
complete-action forwards/backwards, global mean over trajectories, one fresh
AdamW step at learning rate `2.5e-6`, zero weight decay, no scheduler, and
global gradient clipping at one.  Train only the 588 registered language-DoRA
tensors; freeze base, vision, aligner, embedding delta, tied embeddings, and
output head.  Persist only an unmerged shared DoRA adapter and compact
receipts.  QP, PPO, critic, KL, reference penalties, online sampling, and a
merged checkpoint are forbidden.

## Bounds and outputs

The longest action remains 295 tokens.  The update is bounded to 32 forwards
and backwards, one optimizer step, no saved optimizer state, and one hour
absent a concrete runtime fault.  Abort rather than change precision or DDP
topology on OOM.  The expected output root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-02-coverage-graded-static-rloo-one-update/
```

One machine-readable evaluation receipt will bind the plan, gradient/update,
unmerged cold read, panel behavior, and train-248 comparison.  Human records
will summarize that receipt without promoting train behavior into
image-disjoint generalization, repeatability, full precision, F1, mAP, or
production readiness.
