---
title: C-Anchored Two-Rank Two-Step DDP Smoke
description: Test the accepted quarter-dose actual-prefix objective for two canonical DDP updates, unmerged persistence, and cold natural behavior without adding a trainer.
status: completed; scientific stop
---

# C-Anchored Two-Rank Two-Step DDP Smoke

Result: [two-step DDP result](results.md).

## Frozen question

> Can the existing canonical rollout-calibration trainer execute the selected
> eleven-image C-anchored objective for two equal-credit DDP updates, persist
> unmerged language-DoRA adapters with an identity-copy of the required frozen
> embedding delta, and retain the positive twelve-image natural behavior after
> cold reload?

This is a production-shaped mechanics/effect smoke between the accepted
single-step probe and a larger image screen.  It is not an 8-GPU throughput
test, a COCO run, or a held-out generalization estimate.

## Frozen identities

Start again from the C adapter, not from the accepted single-step adapter:

- C adapter fingerprint:
  `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`;
- source checkpoint ID:
  `f75949ad231c1846399a1a8c5470bf7bda71c930a0a76474dbccedb551a16036`;
- small-dual plan SHA-256:
  `f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`;
- accepted single-step evaluation SHA-256:
  `399d090787e46dbdd58e6909c7ea7ee75d1ac87ccea4e12773cc0b2cd76de213`;
- universal base, step-2444 selected-token embedding delta, processor,
  tokenizer, `geo_sorted_xy` prompt contract, complete-row targets, C natural
  prefixes, and eleven owner identities remain unchanged.

The training JSONL is the exact `1024 * global max len 12000`-lineage artifact
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/train-common-base.jsonl`,
SHA-256
`86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd`.
Only the eleven annotated positive owners are supervised.  Unmatched or
unannotated objects remain unknown, never negatives.

## Shortest implementation

Use the existing `accelerate launch -m src.train` path with
`training.mode=rollout_calibration` and
`rollout_calibration.profile=annotated_complete_row_imitation_only`.  The
existing complete-row CE primitive, DDP reducer, optimizer, scheduler factory,
checkpoint writer, and adapter loader are unchanged.  The StateBank adds one
truthful annotated-row family rather than falsely relabeling canonical COCO
rows as sampled or greedy model generations.  The only new experiment
executable is a CPU materializer from the frozen plan to that public schema,
plus child YAML configs.  No generic adapter layer is warranted for this
bounded seam.

The materializer must fail closed on every plan, data, image, prompt, prefix,
target-row, token-type, adapter, embedding-delta, and checkpoint identity.  It
must call the existing public `assemble_state_bank`; it must not reproduce the
StateBank validator.

## Exact two-rank objective

Materialize two lexically ordered 12-record windows.  Each window contains the
eleven plan events once plus a second copy of `positive:88854:1`.  Across the
full 24 records:

- each nonduplicated image has two records with raw
  `image_balanced_event_weight=1.0`;
- image `88854` has four records with raw weight `0.5`.

The public StateBank mean normalization therefore maps singleton weights to
`12/11` and duplicate weights to `6/11`.  In each 12-record update window every
image contributes total weight `12/11`; division by the global eligible
denominator 12 gives exactly `1/11` objective credit per image.  The
materializer must verify these realized weights, both window memberships, and
equal per-image credit before launch.

Run two processes with effective batch size 12, hence six accumulation
microsteps per rank and one global update per 12-record window.  Require exact
world size 2 and global denominator 12 at both updates.

## Optimizer and persistence

- language DoRA only: 588 tensors, 18,006,016 scalars;
- embedding delta frozen;
- AdamW betas `(0.9,0.999)`, epsilon `1e-8`, weight decay 0, global clip 1;
- adapter learning rate `2.5e-6` at both updates;
- existing `cosine_with_warmup` factory with `num_cycles: 0.0`, which makes
  the two registered effective learning rates constant without new code;
- BF16 canonical distributed runtime;
- checkpoint after step 1 and final checkpoint after step 2;
- no merged checkpoint or base-weight export.  The canonical writer may save
  the already-required frozen special-token embedding payload beside each
  adapter only if it is tensor-identical to the input payload and has zero
  trainable embedding parameters.

The DDP smoke deliberately does not add per-step Armijo rollback or protected
margin projection.  Its purpose is to test the mature training path and a
second update.  If behavior fails, that observation decides whether a
transactional line search or preservation term is worth implementing.

## Cold behavior reads

Cold-load both step-1 and step-2 unmerged adapters in fresh inference
processes.  Require exact saved-to-materialized adapter tensor equality,
`merged_adapters=[]`, frozen embedding-delta identity, RP1, temperature zero,
max 3084, natural EOS, and the same twelve-image panel.

For C, step 1, and step 2, report category-consistent global one-to-one owner
coverage at IoU50/60/80, the eleven selected targets, all three protected
owners, total gains/losses, prediction/duplicate/drop/invalid counts, natural
ordering violations, and EOS.  IoU50 is decision-owning; unmatched predictions
remain unknown.

## Annotation-quality monitor

Do not relax an entire semantic category.  Sparse `book`, cup, fruit, or
vegetable instances remain ordinary reliable GT.  Mark an owner as
`dense_category_annotation_uncertain` only when that owner belongs to one of
those families **and** the image contains many repeated, overlapping instances
of the family whose individual ownership or boundary cannot be visually
separated.  Small, occluded, or truncated owners from other categories are a separate
`visual_difficulty` monitor, not an annotation exemption.

Both strata keep their original training weight and stay in every IoU50/60/80
denominator.  Report both selected-target uptake and all owner gains/losses
separately for reliable, `dense_category_annotation_uncertain`, and other
visually difficult owners.  Selected-target uptake observed only in the
dense-category uncertain stratum is valid mechanics evidence but cannot by
itself trigger the effect GO.

## Decision and stop

1. Any identity, StateBank weight/window, world/denominator, objective,
   trainable-surface, finite-gradient, checkpoint, unmerged-load, or decode
   mismatch is mechanical invalidity and stops the unit.
2. Step 1 must retain all three protected owners, recover at least one selected
   target, and keep total IoU50 coverage at least the C value 137.  Otherwise
   return `SCIENTIFIC_STOP_C_ANCHORED_DDP_STEP1` and do not scale.
3. Any step-2 protected loss, invalid/capped decode, or total IoU50 coverage
   below 137 returns `SCIENTIFIC_STOP_C_ANCHORED_DDP_TWO_STEP`.
4. If step 2 retains `3/3`, has selected-target uptake at least 1 including at
   least one target outside `dense_category_annotation_uncertain`, total IoU50
   coverage at least 138, and no invalid/capped row, return
   `GO_C_ANCHORED_MULTIIMAGE_SCREEN`.
5. Otherwise return `HOLD_C_ANCHORED_DDP_EFFECT`.

Stop after this two-step, two-rank panel read.  Do not opportunistically add
steps, doses, GPUs, owners, projection, or a held-out cohort.

## Claim boundary

Success establishes only canonical two-rank execution, two finite shared DoRA
updates, unmerged DoRA persistence with a frozen embedding identity-copy, and
same-panel cold behavior.  It does not
establish DDP equivalence to the FP32 probe, held-out transfer, full-dataset
benefit, missing-label precision, or production readiness.
