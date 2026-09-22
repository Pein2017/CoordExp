---
title: Broad-Support Coverage-Graded Static RLOO One-Update
description: One train-first shared-DoRA update using every complete-clean K4 image group while eliding only mathematically zero RLOO terms.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-03-broad-support-coverage-graded-static-rloo-one-update
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-03
---

# Broad-support coverage-graded static RLOO

## Frozen question

> From exact C, does one shared language-DoRA update using the full 218-image
> complete-clean K4 support improve natural-greedy annotated-owner behavior on
> train248 under the unchanged coverage-graded reward and mechanical gates?

The decision-owning outcome is the change in summed category-consistent global
one-to-one annotated-owner hits at IoU50, IoU60, and IoU80 on the frozen
train248 cohort, with every threshold also reported separately.  Any positive
sum is retained as train-side evidence; zero or negative change stops this
exact profile.  This unit runs one update only: no reward, learning-rate,
seed, exposure, or step sweep.  It does not require image-disjoint improvement
and does not claim F1, mAP, generalization, or production readiness.

## Contrast

- **Anchor:** exact C: the registered base, step-2444 embedding delta, and
  unmerged C DoRA.
- **Intervention:** one fresh AdamW step at `2.5e-6` on the same 588 shared
  language-DoRA tensors, using frozen natural K4 complete actions and
  `0.90*IoU50 + 0.05*IoU60 + 0.05*IoU80` annotated-owner reward.
- **Primary evidence:** fresh cold-loaded candidate versus C under the same
  train248 natural-greedy decode and matcher.
- **Strongest alternative:** averaging broader trajectory credit may cancel
  the sparse selected-eight direction; any apparent change may remain a small
  threshold crossing rather than scalable owner-set learning.
- **Monitors:** matched-owner gains/losses, common-owner IoU, prediction and
  duplicate counts, invalid/dropped spans, EOS/caps, length, and natural
  ordering.  Unmatched valid predictions remain unknown.

The seven complete-clean images carrying dense repeated book, broccoli, or cup
review flags remain in the registered population; the flags are reported as a
quality monitor and do not silently alter the objective.

## Exact support and zero-term simplification

The immutable census is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-current-policy-owner-set-signal-census/analysis-v3.json`
at SHA-256
`0569bdc818d69ddb221d18b57ae6ac22d7e2cb4938e3ed53e73ddbc2fd65496d`.
It admits 218 complete-clean K4 groups, 1,388 annotated owners, 872 frozen
trajectories, and 58,684 action tokens.  For image `i` and trajectory `k`:

```text
r_ik = (0.90*c_ik(0.50) + 0.05*c_ik(0.60) + 0.05*c_ik(0.80)) / |O_i|
A_ik = r_ik - mean_{j != k}(r_ij)
L    = -(1/218) * sum_i (1/4) * sum_k A_ik * sum_t log pi(a_ikt | h_ikt)
```

Seventy-one groups have four exactly equal rewards, hence a mathematically
zero RLOO term.  They remain in the plan and denominator but require no model
forward.  The live update executes 147 groups, 588 trajectories, 51,287
generated body tokens, and 588 appended `im_end` action tokens.  Eight-rank
DDP keeps each image group intact, balances generated body
tokens by deterministic LPT assignment, performs one synchronized backward per
rank, and compensates DDP rank averaging by `8/218`.  This is the same
218-image mean objective, not a 147-image redefinition.

## Bounds, gates, and stop

- eight FP32/SDPA ranks; one resident model and one image materialization per
  rank at a time;
- 588 forwards and backwards, one optimizer step, no online generation,
  optimizer checkpoint, merged checkpoint, QP, PPO, critic, KL, or reference
  model;
- 18--19 executed groups and at most 6,434 action tokens per rank;
- expected peak below 30 GiB CUDA and 14 GiB host RSS per rank, artifact payload
  below 2 GiB, and a 75-minute wall ceiling absent a concrete fault.

Before publication, source, plan, assignment, action/EOS, reward, normalization,
finite-gradient, collective, unmerged-adapter, and fresh cold-read identities
must pass.  Mechanical failure publishes no candidate and permits only a repair
from immutable C.  After one mechanically valid update and train248 read, this
round stops and reports to the user.

## Registered completion

The one-update and train248 read are complete.  IoU50/60/80 owner counts move
`+7/+11/+6` versus C, for +24 summed threshold-owner hits, and all three counts
also exceed the prior selected-eight graded update.  The seven dense repeated
category quality-monitor images contribute zero summed net; excluding the
largest single cap-recovery contributor still leaves `+3/+7/+2`.

The exact mechanics, partitions, sensitivity, and claim boundary are recorded
in [results](results.md) and the externally stored analysis at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-broad-support-coverage-graded-static-rloo-train248-read/analysis.json`
(SHA-256
`d81dd644cc4bc401bd9c059e4edee49c29dbcba7769a2e5d3af7bba7eab1de0a`).
No image-disjoint read or further training belongs to this completed unit.
