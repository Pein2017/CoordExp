---
title: PVCI Sequential Control-Variable Causal Schedule
description: Tests whether continuation, current-instance identity, and geometry require distinct causal interventions at their natural row boundaries.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-10-pvci-sequential-control-variable-causal-schedule
topic: qwen3-vl-painted-gt-transcription-probe
status: completed
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - causal-patching
  - free-continuation
updated: 2026-07-10
---

# PVCI Sequential Control-Variable Causal Schedule

## Question

Does a Qwen3-VL object row require temporally distinct causal states for
continuation, current-instance identity, and geometry, or does successful
full-row control require persistently available visual conditioning?

The tested residual states are experimental handles. This unit does not assume
that they should become runtime modules or the final training interface.

## Decision Relevance

- Functional capability or uncertainty: whether continuation, instance
  binding, and geometry can be established by boundary-local causal states.
- Costly or hard-to-reverse choice this result could change: whether later
  training research should target sequential boundary-specific control or a
  visual designation that remains available throughout the row.
- Outside this unit's scope: learned selector, renderer, slots, coverage
  ledger, STOP training, mixture training, final architecture, and production
  detection metrics.

## Established Predecessor Evidence

The predecessor result is currently artifact provenance, not a migrated
current-worktree research claim:

- source worktree: `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`;
- source branch baseline: `codex/qwen3-vl-painted-gt-transcription-probe` at
  `36ca29a9`;
- source probe and closeout were untracked relative to that baseline;
- result artifact:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_row_trajectory/e1_wrong_object_jitter_medium_strict18_20260710/`.

That panel reported a family split: a row-entry intervention reliably changed
STOP to continuation but did not reliably preserve object identity, whereas an
intervention after `<|object_ref_start|>` reliably changed the phrase and
strongly pulled later coordinates toward the post-scatter object. This unit
must revalidate the source ledger and runtime seam before using those events.

## Competing Hypotheses

- H1: sequential boundary-specific control.
  - Expected signature: a continuation intervention opens the row; a later
    identity intervention selects the intended object; a geometry-boundary
    intervention further improves coordinate recovery.
  - Meaningful falsifier: an appropriately timed second intervention does not
    improve the downstream span relative to the corresponding one-anchor
    condition.
- H2: persistent visual drive.
  - Expected signature: persistent post-scatter conditioning remains
    substantially stronger than any small schedule of one-time residual
    interventions.
  - Meaningful falsifier: a predeclared finite intervention schedule produces
    the intended phrase and geometrically equivalent row without persistent
    conditioning.
- H3: generic prefix-continuation mediation.
  - Expected signature: the row-entry intervention changes continuation but
    the generic row-opening token, rather than a retained visual identity,
    explains later behavior.
  - Meaningful falsifier: after controlling for the shared textual prefix, an
    independently captured marked-object state changes identity and geometry.
- H4: one identity state already controls usable geometry.
  - Expected signature: coordinate tokens differ from persistent post-scatter
    token-for-token, but decoded geometry is already equivalent and a geometry
    re-anchor gives no meaningful gain.
  - Meaningful falsifier: the geometry re-anchor improves coordinate distance
    or IoU without changing the already-correct phrase.

## Completion Promise

This unit is complete when:

- Evidence gate: the predecessor event ledger and relevant runtime identities
  are verified in this checkout, and both continuation-then-identity and
  identity-then-geometry panels have clean, persistent-post, one-anchor,
  two-anchor, no-op, and early negative controls.
- Acceptable evidence: event-level free continuations with hook/cache receipts,
  phrase attribution, `x1/y1/x2/y2`, geometric distance or IoU, row closure,
  fresh replay eligibility, and exact intervention-application counts.
- Insufficient evidence: next-token recovery alone; teacher-forced scores with
  no free continuation; or an aggregate row metric that hides the two event
  families or phrase-versus-coordinate behavior.

## Outcome Interpretation

- If continuation plus identity fixes wrong-object continuations: distinct
  existence and identity decisions become more likely.
- If identity plus geometry re-anchor materially improves localization: a
  distinct geometry-boundary state becomes more likely.
- If identity alone is geometrically equivalent: exact coord-token replay is
  demoted as a success criterion, and a separate geometry state becomes less
  necessary at this handle.
- If only persistent post-scatter conditioning works: continuously available
  visual drive becomes more likely.
- If a second intervention harms a correct first-stage trajectory: state
  compatibility or donor-prefix mismatch becomes more likely than a missing
  causal variable.
- If the result is negative at this handle: do not infer that the functional
  capability is absent; layer, duration, target position, or donor state may be
  mismatched.
- Possibilities this probe cannot distinguish: how useful states should be
  learned, represented, selected, committed, or stopped at runtime.

## Evidence Scope

- Checkout or branch:
  `/data/CoordExp/.codex/worktrees/69ed/CoordExp`,
  `codex/continue-handoff-session`.
- Baseline commit: `e07c6b73` (`Merge CoordExp-Swift infrastructure`).
- Integration constraint: reuse current `src/inference/` and artifact
  contracts; do not merge the divergent painted-GT branch wholesale or import
  another worktree at runtime.
- Checkpoint: the E1 anti-copy step-484 checkpoint recorded by the predecessor
  post-scatter artifacts, subject to fresh identity verification.
- Planned artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/`.
- Current-runtime prerequisite gate:
  `outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json`,
  SHA-256
  `b46af47a270e7e616577cf07719ce4c533ef671c648c6878ace1cf0941c938d3`.
- Historical checkpoint compatibility: the E1 checkpoint's neighboring
  `checkpoint_handoff.json` is a post-hoc incomplete manifest and the current
  production validator correctly holds it. The research replay uses
  byte-identical hardlinks under
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/legacy_e1_step484_replay/`;
  it does not rewrite or promote the historical handoff.
- Verified CPU plan:
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/debug4_plan.json`,
  SHA-256
  `7c41f4a77ab1bd36ec3d3b7ae2dea1022c6eaff329001e4625ff02cf8a5c70ee`.
- Planned sample window:
  - A-hard: `coco2017_val_000000001503__stepwise_geo_sorted_b22e7e1ee03e0ce8__step_001`, divergence `0`;
  - A-nonregression: `coco2017_val_000000002153__stepwise_geo_sorted_b22e7e1ee03e0ce8__step_001`, divergence `0`;
  - B-hard: `coco2017_val_000000000139__stepwise_geo_sorted_b22e7e1ee03e0ce8__step_008`, divergence `1`;
  - B-clean-ceiling: `coco2017_val_000000000802__stepwise_geo_sorted_b22e7e1ee03e0ce8__step_000`, divergence `1`.
  Expand only if controls and intervention receipts pass.
- Verified single-event integration smoke:
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/a_hard_bridge_smoke.json`,
  SHA-256
  `0f0740b884daaaad540a46c21cac2c6b8a97fba8d2a2b703f62f6875b3d286b6`.
  This is a call-boundary and exact-replay gate, not hypothesis evidence.
- Completed debug4 result root:
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/debug4/`.
  - `summary.json`: SHA-256
    `5f5a92bc0d68804e499b05ebd69bffefc955b45e1854e5134d7b2ba953af86fd`;
  - `predecessor_replay.jsonl`: SHA-256
    `0f34f6abb7264128358999fa50168644677257fc49c329d32254782d6b7f4dfb`;
  - `donor_snapshots.jsonl`: SHA-256
    `d70edf5547e467a9ee825f5762d6ea0361f0d539c160d19f7f18e6906f99b355`;
  - `conditions.jsonl`: SHA-256
    `818e99359cf3ae28bb6fd08848b9cc0d081bb321d86368db7b100035c664d522`.
  A forced repeat produced the same four hashes.
- Metrics: immediate transition, phrase identity, exact coordinate slots,
  coordinate L1 and IoU to clean/post references, row closure, replay gates,
  and hook/cache counters.
- Known limitations: one checkpoint, one val32-derived counterfactual family,
  residual-state interventions as causal handles, and no autonomous detection
  claim.

## Procedure

1. Verify predecessor event, feature-store, config, checkpoint, tokenizer, and
   runtime identities before loading the model.
2. Inventory and selectively reuse current inference/artifact seams; port only
   the missing research-specific adapters.
3. Run a bounded continuation-then-identity debug panel: row-entry layer-19
   continuation state followed, only after native
   `<|object_ref_start|>`, by a layer-23 identity state.
4. Run a bounded identity-then-geometry debug panel: layer-23 identity state
   followed, only after native phrase closure and `<|box_start|>`, by a
   layer-23 pre-`x1` refresh. This is a same-layer refresh handle; geometry
   onset has not been localized.
5. Compare with persistent post-scatter, one-anchor, no-op, and early controls.
6. Expand only if fresh replay, hook/cache, and phrase/geometry parsing gates
   pass.
7. Interpret against the predeclared outcome map and stop. Do not continue
   automatically into training or architecture work.

## Observations

- Direct integration observation: the A-hard replay used the metadata-declared
  nine-token assistant prefix, reproduced the predecessor 1329-token prompt
  exactly, reproduced clean `<|im_end|>` exactly, and reproduced the persistent
  post-scatter `mouse` row and all four coordinate tokens exactly. The stored
  image delta was applied once, native cached steps were not modified, the
  layer-19 row-entry state and layer-23 post-`<|object_ref_start|>` state were
  captured with shapes `[1, 1, 2048]`, and all hooks and compile state were
  restored.
- Infrastructure deviation found and resolved within the research runner: the
  current core renderer intentionally no longer injects legacy painted-GT
  assistant prefixes. Omitting that prefix changed A-hard clean behavior from
  STOP to a `tv` row. The research runner now restores the prefix explicitly
  from immutable row metadata and receipts its token ids and prompt hash; core
  renderer behavior remains unchanged.
- Counterexample or negative result: an identity state did not generally carry
  sufficient geometry, and the Panel-A clean identity donor was not a literal
  no-op on A-nonregression. The result supports causal staging, not a universal
  reusable-state claim.
- Integration artifact handle:
  `/data/CoordExp/outputs/painted_gt/pvci_sequential_control_schedule/a_hard_bridge_smoke.json`.
- Debug4 execution gate: all four clean/post predecessor replay pairs passed
  before interventions; eight clean/post donor snapshots used identical textual
  prefixes; 24 outcome conditions completed; all 28 scheduled anchors applied
  exactly once; no hook, cache, or native-boundary receipt failed.
- Panel A, A-hard: clean stopped; row-entry layer-19 alone opened a valid
  `laptop` row with post-box IoU `0.005`; adding the layer-23 state after native
  `<|object_ref_start|>` changed the row to `mouse` with post-box IoU `0.983`
  and coordinate L1 `2`. The layer-13 and clean-donor controls did not produce
  the post identity.
- Panel A, A-nonregression: row-entry layer-19 already produced the post
  `person` identity with IoU `0.922`; the layer-23 identity state improved it
  to IoU `0.953` and coordinate L1 `7`. The layer-13 control matched the
  one-anchor result. The clean layer-23 donor was not behaviorally neutral: it
  changed the row to `sports ball`, so this condition is interpreted as source
  specificity rather than a literal no-op.
- Panel B, B-hard: layer-23 identity alone changed `dining table` to the post
  `refrigerator`, but its geometry had post-box IoU `0.0` and coordinate L1
  `297`. A second layer-23 state at native `<|box_start|>` improved post-box IoU
  to `0.753` and L1 to `48`; the layer-13 and clean-donor controls exactly
  matched the identity-only geometry.
- Panel B, B-clean-ceiling: layer-23 identity alone changed `refrigerator` to
  post `oven` with IoU `0.824` and L1 `62`; the native pre-`x1` layer-23 state
  improved IoU to `0.963` and L1 to `14`. Layer-13 and clean-donor controls
  again matched identity-only.

## Interpretation

- Supported reading: at these handles, row continuation, instance identity,
  and usable geometry are causally separable in time. A row-entry state can
  open continuation without selecting the marked instance; a later identity
  state can select the instance; and a state at the geometry boundary can
  substantially improve localization while preserving the selected phrase.
- H1 is supported on the bounded debug4 panel. The A-hard and both Panel-B
  events supply the decisive contrasts; the controls show layer and donor
  specificity rather than a generic perturbation effect.
- H2 is weakened but not globally falsified. A finite two-anchor schedule was
  sufficient for the post phrase in all four events and recovered high-IoU
  geometry, so persistent visual conditioning is not necessary at these
  events. Persistent post-scatter remained the exact-token and exact-geometry
  ceiling, especially for B-hard.
- H3 is falsified at A-hard: the continuation state alone produced the
  scheduled `laptop`; the later post identity state was required to produce
  the marked `mouse` and its geometry.
- H4 is not a general rule. Identity alone carried useful geometry for
  B-clean-ceiling, but failed to localize the post object in B-hard; the
  geometry-boundary state improved both events.
- Alternative reading: these residual vectors may be trajectory-specific
  steering handles rather than reusable semantic variables. Prefix-controlled
  donor capture establishes causal timing, not how the model could synthesize
  the states on a clean image.
- Remaining uncertainty: whether clean Qwen visual features contain enough
  instance-specific information to predict these identity and geometry states
  without a painted/post-scatter teacher, and whether the result generalizes
  beyond four events and one checkpoint.
- Candidate implementations remain unresolved: selector, renderer,
  pseudo-token, feature delta, slot, coverage memory, and stop mechanism.

## Research Unit Closeout

Observed:

The current-runtime call boundary, all four exact predecessor replay pairs,
eight matched-prefix donor snapshots, and the 24-condition debug4 intervention
matrix passed. A temporally staged continuation/identity/geometry signature was
observed with layer/source controls.

Evidence gate:

Passed for this bounded unit: exact replays, donor-prefix parity, native free
continuation, boundary receipts, cache use, hook restoration, phrase
attribution, four-coordinate parsing, geometry distance/IoU, and deterministic
artifact regeneration all passed.

Supported:

Continuation, current-instance identity, and geometry can be controlled at
distinct natural row boundaries in this four-event E1 panel. Persistent visual
conditioning is not required to recover the post phrase or useful geometry at
these events, although it remains the exact ceiling.

Not supported yet:

Generalization beyond the debug4 panel; a learned or endogenous state
generator; autonomous selection, commit, coverage, or STOP; and any candidate
runtime architecture.

Architecture update:

No architecture is promoted. The evidence makes staged boundary-local control
a viable capability target and makes persistent conditioning a ceiling/control,
not an assumed requirement.

Next decider:

Run an endogenous-state-synthesis probe: from clean Qwen visual features and a
GT-selected instance support used only as a research oracle, predict the
identity-boundary and pre-`x1` residual states with a tiny held-out linear or
low-rank mapper. Compare causal row recovery against exact post donors,
clean-state donors, wrong-instance shuffles, and norm-matched random controls.
Do not train the Qwen backbone or introduce an external detector.

Promotion decision:

Not promoted. The mechanism-level capability is supported, but a reusable
clean-image state generator and broader event evidence are still missing.
