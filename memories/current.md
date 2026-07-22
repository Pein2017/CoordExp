# Current Project Memory

Last verified: 2026-07-22T16:00:53Z. At that check, eight sampled workers were
active, no sampled JSON shard was complete, and the outer launcher remained
paused. Recheck all live execution claims before acting.

## Active objective

Run the constant-dose image-breadth treatment screen for Qwen3-VL dense object
enumeration. Hold sampled-route event count, Source-preservation event count,
optimizer updates, object-count-band allocation, and event difficulty as fixed
as feasible while comparing supervision concentrated in 118 images with the
same dose spread across 496 images.

This is a bounded test of whether narrow image breadth caused the previous
owner exchange. It is not full-size promotion and not an architecture
commitment.

## Most recent closed evidence

The 118-image Source-route-preservation screen is complete. Its treatment was
learnable but narrow:

* selected Source-missed owners were recovered approximately four to six times
  as often as non-selected missed owners;
* admitted images could gain owners and box quality and standard detection
  metrics improved;
* non-admitted images lost owners at every milestone;
* more updates did not repair transfer.

The result supports route-conditioned owner redistribution, not safe final-set
expansion. The two live alternatives are insufficient image breadth versus an
intrinsic owner-exchange limitation of positive complete-row imitation.

## Frozen current unit

The unit is:

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`

The 2,432-image label-only pool was frozen before route inspection and split by
image identity and object-count band into:

* 2,048 training-candidate images, 512 per band;
* 256 development images, 64 per band;
* 128 held-out images, 32 per band.

Each treatment arm will contain 496 sampled-route rows plus 496 exact Source
rows: 992 unique events, 31 optimizer updates, and effective event batch size
32. The broad arm uses 496 physical images; the concentrated arm uses a nested
118-image subset. The twelve human-refined images remain safety evidence only
and never supply gradients.

The selector must record one of three matching modes:

1. exact object-count-band by selection-rank matching;
2. coarse object-count-band by rank-one, rank-two, rank-three, and
   rank-four-or-deeper matching;
3. policy-only rank-one fallback.

Interpretation must follow the emitted mode. Only exact matching supports a
breadth-effect claim at fixed ordinal-rank distribution. Coarse matching has a
weaker claim, and policy-only fallback cannot identify physical image breadth
as the cause.

## Live execution state

The missing 2,176-image sampled panel is running at:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-22-constant-dose-image-breadth-treatment-screen/trajectory-panel-2432-v1/`

Eight GPUs are processing independent two-seed shards for seeds 31,001 through
31,016. Each request contains one image, uses temperature 0.4, nucleus
probability 0.95, repetition penalty 1.0, and a 512-token generation limit, and
resets random state per image and seed. At the last live check all eight
workers were healthy and no sampled JSON shard had completed.

The outer launcher is intentionally paused so it cannot start its original
single-GPU greedy stage after the sampled workers finish. The sampled child
workers are not paused. Leave them running.

## Current understanding

Established or bounded-supported:

* Low-temperature sampling exposes real physical owners that greedy decoding
  misses.
* Prefix order is causally active; no order-free covered-object ledger has been
  demonstrated.
* Entity discovery and complete box geometry are separate outcomes.
* Fixed-prefix coordinate correction can be learned without improving clean
  rollout.
* Positive complete-row imitation can strongly favor selected owners while
  exchanging away other Source owners.
* Standard mean Average Precision can improve even when unique physical-owner
  coverage does not.

Tentative or unresolved:

* Whether greater physical image breadth turns narrow owner redistribution
  into a transferable rule.
* Whether complete-row positive imitation intrinsically exchanges owners even
  after matching training dose and event difficulty.
* Whether any explicit covered-set carrier, object slot, cursor, or other
  architecture change is necessary.

Rejected or held:

* Do not interpret longer output, terminal suppression, selected-owner
  recovery, or a mean Average Precision gain alone as set expansion.
* Do not scale the previous 118-image treatment unchanged.
* Do not add a second cohort, external detector, object slot, terminal
  suppression, canonical supervised-fine-tuning mixture, or online refresh to
  this pilot.

## Immediate next actions

1. Wait for all eight sampled shard files and confirm the sampled workers have
   exited cleanly.
2. Prevent the paused outer launcher from falling through to one-GPU greedy;
   generate greedy seed 31,000 as eight image shards under the same physical
   batch-one semantics.
3. Validate the old-plus-new trajectory union, run the full 2,432-image route
   analysis, and assemble both StateBanks.
4. Read the actual `matching_mode` and `interpretation_scope` before making any
   claim or launching training.
5. Run one real mixed-step StateBank loader and gradient smoke. If it passes,
   train broad and concentrated arms under matched optimizer seeds 19 and 23
   for 31 updates, saving steps 10, 20, 30, and 31.
6. Choose one shared step on development, then evaluate held-out once under
   the unit's owner-ledger and output-health rules.

The worktree is intentionally dirty with current research implementation,
configs, tests, and documents. Do not clean, revert, stage, or commit unrelated
changes while resuming this run.

## Minimum reading path

1. `memories/notes/2026-07-22-constant-dose-image-breadth-screen-checkpoint.md`
2. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-source-route-preservation-and-multiple-sampled-route-training-screen/results.md`
3. `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-constant-dose-image-breadth-treatment-screen/unit.md`
4. `research/investigations/qwen3-vl-dense-enumeration/compass.md`
5. `memories/notes/019f4a19-d81c-75a2-84b0-2c20379e686e-comprehensive-recap.md`

The comprehensive recap remains the long historical reconstruction. The new
checkpoint note is the operational handoff for the running breadth screen.
