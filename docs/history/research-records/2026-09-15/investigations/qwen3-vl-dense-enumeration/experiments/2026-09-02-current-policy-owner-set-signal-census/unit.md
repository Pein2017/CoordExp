---
title: Current-Policy Annotated-Owner Set Signal Census
description: A full 248-image K=4 census of whether frozen C supplies within-image annotated-owner utility variation for a minimal on-policy set-reward update.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-02-current-policy-owner-set-signal-census
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Current-policy annotated-owner set signal census

Executed result: [results.md](results.md).  The mechanically valid census
returns **`GO_MINIMAL_RLOO_VERTICAL`** with 109 fully clean, dominance-bearing
IoU50 image groups; no training occurred in this unit.

## Decision and outcome

> From the frozen censored-transcript cross-entropy adapter C, do four
> current-policy natural completions per image provide enough within-image
> variation in final annotated-owner sets to identify one minimal eight-GPU
> leave-one-out policy-gradient update, without labeling unmatched rows as
> negatives?

The decision-owning observation is **signal supply**, not trained model quality:
whether the current policy samples materially different numbers and identities
of category-consistent, globally one-to-one annotated owners at intersection
over union (IoU) at least 0.50.  This unit performs inference and analysis only.
It writes no model checkpoint, executes no backward pass, and promotes no
training architecture.

This is the shortest open discriminator after the completed
[S2 observed-alias stop](../2026-09-02-s2-observed-alias-handbag-continuation-rescue/results.md).
It does not revive winner-trajectory cross-entropy, random-order transcript
training, target-versus-STOP row objectives, positive sampled-row replay, or
static K-union credit.  Those materially different routes already have
negative or non-promoting evidence.

## Immutable specimen

The model is the universal base plus the existing **unmerged**, shared language
DoRA adapter C and the registered step-2444 embedding delta.  No merged export,
output residual, per-image module, or inference-time memory is allowed.

```text
C adapter:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-01-annotated-owner-direct-c-d0-pilot/train/c/runs/
  qwen3_vl_2b_annotated_owner_direct_c_26_steps_ebs64_seed19/
  checkpoints/step-26/adapter

Inference config:
/data/CoordExp/.worktrees/c-anchored-owner-mechanism-audit/configs/
  coordexp_infras/infer/
  qwen3_vl_2b_c_anchored_owner_audit_c_train248.yaml

Executed 248-image input:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/
  train-common-base.jsonl

Frozen C greedy anchor:
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-02-c-anchored-owner-mechanism-audit/full/infer/c/
  qwen3-vl-2b-c-anchored-owner-audit-c-train248/
  gt_vs_pred.jsonl
```

The C adapter fingerprint remains
`5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`.
The population is the same 248 mechanically eligible optimization images and
1,798 annotated owners.  The 256-image candidate universe and its eight prior
exclusions remain provenance only; excluded owners are not imputed as misses.

The annotation authority remains
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl`
with SHA-256
`4bc00be57d78d4d76e0874cd83f89329d1dc2cd471c208f3f2e30e0365dd96ee`.
The geometry-sorted derivative, SHA-256
`ecf07a40adca1c85fa677d22970b1ad73e2299380660417520adf660e3674ad1`,
is transcript-order provenance only.  Natural decode is permitted to be
unordered.

## Sampling intervention

For every one of the 248 images, sample four independent completions from the
frozen current policy with seeds
`2026090201,2026090202,2026090203,2026090204`:

- Hugging Face sampling with temperature `1.0` and top-p `1.0`;
- repetition penalty `1.0`;
- maximum 3,084 new tokens and natural Qwen `im_end` stopping;
- exact current C prompt, image, tokenizer, base, embedding delta, and adapter;
- eight disjoint 31-image shards, one process and one physical GPU per shard.

At these knobs, no temperature, nucleus, or repetition transformation changes
the model logits before categorical sampling.  A length stop is right
censoring and a safety event; it is not an annotated-owner miss.

The existing sampler
`scripts/research/run_current_seeded_sampled_rollouts.py` is reused unchanged.
Its current focused test must pass before the first launch.  A one-image K=4
artifact must then pass model/prompt/image identity, natural-stop evidence,
parser readback, and owner matching before the 248-image wave.

### Pre-analysis semantic correction

The raw 248-image acquisition was launched from Git commit `d0ed592d7`, whose
unit hash is preserved in `launch-plan.json`.  After the non-scientific smoke
but before any full-shard output was inspected, an independent reducer-contract
check found that the initial wording allowed a two-clean-rollout dominance pair
to authorize a four-rollout RLOO group.  The corrected analysis requires all
four rollouts to be clean, uses image-normalized reward and sample variance,
defines any fully clean identity-diverse image as sufficient for the bounded K8
branch, and makes the annotation-quality check an explicit route
counterfactual.  Sampling seeds, population, model, decode policy, horizon,
IoU gate, and unmatched-row semantics did not change.

No `analysis-v1` was executed.  The unchanged raw acquisition is retained as
attempt 1.  `analysis-v2` then failed before metrics because its C-anchor join
did not canonicalize the twelve-digit zero-padded image filename.  That
technical failure is immutable and scientifically neutral.  The one-line key
normalization repair is replayed as `analysis-v3` against the same raw shards.
The semantic correction is provenance, not evidence from the still-unseen
full-cohort outcomes.

## Primary observation

For image `i` and rollout `k`, let `S_ik` be its category-consistent global
one-to-one IoU50 annotated-owner set.  Let `G_i` be the corresponding frozen C
greedy owner set.  For every mechanically analyzable image report:

1. the four counts `|S_ik|`, their range and sample variance, and the image-normalized
   reward `r_ik = |S_ik| / |O_i|`, where `O_i` is the annotated-owner set;
   its leave-one-out advantage is
   `A_ik = r_ik - sum_(j != k) r_ij / (K - 1)`;
2. whether any pair has strict inclusion `S_ia` strictly containing `S_ib`;
3. sampled union size, best-single size, and the union-only gap;
4. incumbent retention `|S_ik intersect G_i|`, rescue
   `|S_ik minus G_i|`, and loss `|G_i minus S_ik|`;
5. whether a sampled set weakly contains `G_i` and adds at least one owner;
6. natural-stop, cap, malformed/drop, valid-row, exact-duplicate, and ordering
   monitors.

IoU60 and IoU80 repeat the owner-set report as robustness monitors; IoU50 owns
the current decision.  Unmatched valid rows remain unknown and receive neither
positive nor negative owner identity.  Ordering violations are legal natural
behavior and monitor-only.

## Annotation-quality monitor

Sparse book, fruit, vegetable, and cup cases remain ordinary reliable cases.
Only scenes marked `high` by the frozen cohort ledger and having at least four
annotations from one of
`book`, `cup`, `apple`, `banana`, `orange`, `broccoli`, or `carrot` are flagged
for conclusion-triggered visual review.  They are not excluded and do not
silently change a denominator.  Review is required only if this flagged subset
alone changes the route decision.

## Decision matrix and stop rule

A rollout is clean for this gate when it reaches natural `im_end`, its parser
status is `accepted` or `empty`, and it has no dropped span.  A group is
**dominance-bearing** when all four rollouts are clean and at least one pair
has a strict annotated-owner set inclusion at IoU50.  Strict
inclusion implies nonzero raw coverage reward range and therefore nonzero
leave-one-out advantage.

- **`GO_MINIMAL_RLOO_VERTICAL`**: at least eight images are dominance-bearing.
  This supplies one full world-size-eight update in which every rank receives
  a genuine within-image coverage comparison.  It authorizes only a separate
  one-update base-plus-unmerged-DoRA vertical slice with current-policy
  generation, exact behavior-policy log probabilities, and a frozen-reference
  KL term.
- **`GO_OWNER_IDENTITY_UTILITY_ANALYSIS`**: fewer than eight images are
  dominance-bearing, but at least eight have nonzero owner-count range.  The
  raw reward signal is exchange-heavy; use this same artifact to compare
  C-anchored retention/rescue or per-owner utilities before any training.
- **`GO_K8_SUPPORT_EXPANSION`**: fewer than eight images have nonzero count
  range, but any fully clean unresolved image has equal-count owner-identity
  diversity or a positive union-only gap, showing that K=4 may hide set
  signal.  Expand only those unresolved images to K=8; do not resample the
  full panel.
- **`SCIENTIFIC_STOP_RAW_CURRENT_POLICY_SET_REWARD`**: fewer than eight images
  have nonzero count range, and no fully clean image has equal-count identity
  diversity or a positive union-only gap.  Stop the raw current-policy
  K-sample reward route.  This is not a general negative about DoRA capacity
  or annotated-owner objectives.
- **`MECHANICAL_INVALID`**: incomplete shard coverage, identity mismatch,
  duplicate image/seed cells, missing parser evidence, or unreadable artifacts
  prevents interpretation.  Repair mechanics and rerun from frozen C under a
  new immutable attempt.

If caps or malformed completions are the only source of an apparent decision,
hold the scientific verdict and report that safety confound instead.  Recompute
the route after excluding the dense repeated-category review flags; visually
review those images only if that counterfactual changes the route.  A positive
census does not establish optimization success, greedy transfer, held-out
generalization, or missing-annotation robustness.

## Artifact handle and cost

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-09-02-current-policy-owner-set-signal-census/
```

Expected primary payload is eight immutable sampled-rollout shards totaling
992 trajectories, one smoke artifact, one compact analysis receipt, and one
results record.  Expected peak memory is below the prior C natural-decode
observation of about 20 GB reserved per 80 GB A100.  No reusable trainer or
OpenSpec is needed unless this census licenses the on-policy update.
