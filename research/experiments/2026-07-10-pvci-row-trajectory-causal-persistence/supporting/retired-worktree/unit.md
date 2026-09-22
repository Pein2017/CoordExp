---
title: PVCI Row-Trajectory Causal Persistence
description: Tests whether a one-time late-middle residual intervention changes a complete free-generated object row or only the immediate divergent token.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-10-pvci-row-trajectory-causal-persistence
topic: qwen3-vl-painted-gt-transcription-probe
status: completed
tags:
  - coordexp-swift
  - research-unit
  - painted-gt
  - pvci
  - causal-patching
  - free-continuation
updated: 2026-07-10
---

# PVCI Row-Trajectory Causal Persistence

## Question

When a post-scatter-to-clean late-middle residual patch changes the first
clean/post divergent token, does that intervention place the clean run onto a
coherent post-scatter object-row trajectory, or does its effect disappear after
the immediate token?

The unit keeps phrase identity, coordinate slots, box closure, and row
termination separate. It does not assume that a residual patch, post-scatter
delta, pseudo-token, or any other current experimental handle is the final
runtime interface.

## Decision Relevance

- Functional capability or uncertainty: whether the known causal
  current-object handle persists across a free-generated structured row.
- Costly or hard-to-reverse choice this result could change: whether later
  internalization work should target a one-time row-entry state, a
  coordinate-specific re-anchor, continuously available visual context, or
  prefix-robust trajectory training.
- Outside this unit's scope: learning a cursor, fitting a renderer, autonomous
  object selection, coverage, STOP, mixture training, final detection mAP, and
  production cache/interface design.

## Competing Hypotheses

- H1: Persistent trajectory or basin entry.
  - Expected signature: a one-time patch flips the first divergent token and
    the unpatched continuation remains aligned with the post-scatter phrase,
    coordinate span, box closure, and row termination.
  - Meaningful falsifier: continuation returns to the clean trajectory or loses
    object-row coherence immediately after the patched token.
- H2: Span-split control.
  - Expected signature: phrase or continuation behavior follows post-scatter,
    but geometry diverges at `<|box_start|>`, `x1`, or a later coordinate slot.
  - Meaningful falsifier: one patch controls all recorded row spans without a
    geometry-specific failure boundary.
- H3: Immediate-token-only mediation.
  - Expected signature: immediate post-token recovery remains strong, but exact
    post-trajectory prefix length is approximately one token and later output is
    clean-like or unstable.
  - Meaningful falsifier: multi-token post-like free continuation after the
    intervention is removed.
- H4: Common-prefix replay dependence.
  - Expected signature: teacher/common-prefix next-token replay succeeds, while
    free continuation after the patch is unstable or highly prefix-sensitive.
  - Meaningful falsifier: stable event-level free continuation on the strict
    replay panel.

## Completion Promise

This unit is complete when:

- Evidence gate: a verified event ledger links the existing layer-onset,
  residual-patch, layer-boundary, and post-scatter generation artifacts; clean,
  persistent post-scatter, one-time post-to-clean patch, clean-to-clean no-op,
  and early-boundary negative-control continuations are evaluated first on a
  debug panel and then on all eligible strict replay events.
- Acceptable evidence: event-level token trajectories and parsed rows report
  immediate recovery, exact clean/post prefix lengths, phrase identity,
  `x1/y1/x2/y2`, box closure, and row termination separately, with hook/cache
  receipts and replay eligibility retained.
- Insufficient evidence: next-token margins alone; teacher-forced scoring with
  no free continuation; an aggregate row score that hides phrase-versus-geometry
  failure; or a panel without no-op and early-boundary controls.

## Outcome Interpretation

- If one patch carries a complete post-like row: a compact persistent
  row-trajectory state becomes more plausible; internal synthesizability remains
  unresolved.
- If phrase follows but geometry drifts: identity and geometry likely have
  different temporal control requirements; a coordinate-boundary causal probe
  becomes the next decider.
- If only the immediate token changes: persistent visual drive becomes more
  plausible and the residual patch remains a local causal handle rather than a
  demonstrated row memory.
- If free continuation is unstable despite replay success: prefix-distribution
  dependence becomes more plausible and paired teacher/self-prefix persistence
  becomes the next decider.
- If event families disagree: preserve separate conclusions for continuation
  gating, description identity, coordinate binding, and termination instead of
  averaging them into one verdict.
- If the result is negative at this handle: do not infer that current-object
  conditioning is impossible; the tested boundary, duration, prefix, or row
  slot may be mismatched.
- Possibilities this probe cannot distinguish: how to synthesize the useful
  state, whether visual tokens or slots support autonomous selection, and what
  mechanism should implement commit or STOP.

## Evidence Scope

- Checkout or branch:
  `/data/CoordExp/.codex/worktrees/eb1c/CoordExp`,
  `codex/qwen3-vl-painted-gt-transcription-probe`.
- Baseline commit: `36ca29a9` (`Add PVCI post-scatter causal localization probes`).
- Intended code scope:
  `scripts/probes/painted_gt/run_post_scatter_row_trajectory_probe.py` plus a
  focused test file; reuse existing probe helpers rather than changing stable
  inference behavior.
- Checkpoint or model version: the same E1 anti-copy step-484 checkpoint and
  base model recorded by the source post-scatter units.
- Source artifacts:
  - `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_image_token_equivalence/e1_wrong_object_jitter_medium_val32`;
  - `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_onset/e1_wrong_object_jitter_medium_val32_step0_step1_all`;
  - `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_residual_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all`;
  - `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_layer_boundary_patch/e1_wrong_object_jitter_medium_val32_step0_step1_all`.
- Planned artifact root:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_row_trajectory/`.
- Planned commands:
  - focused unit tests for event joining, patch lifetime, stopping, and trajectory
    metrics;
  - explicit balanced `debug4` strict-event run (not positional
    `--max-events 4`, which includes a non-strict event);
  - full eligible strict-event run only after debug receipts pass.
- Metrics or counters: selected and strict event counts, hook counts, patch
  application counts, generated-token counts, immediate recovery, exact
  clean/post prefix lengths, slot-wise matches, parser status, and row outcomes.
- Sample window: the existing `20` selected step-0/step-1 divergence events,
  with the causal claim restricted to the existing `18` strict margin replay
  events unless fresh replay changes eligibility.
- Known limitations: one checkpoint, one val32-derived wrong-object
  jitter-medium panel, row-level diagnostic intervention, and no production
  selection/coverage/STOP claim.

## Planned Probe

1. Build a read-only event ledger from source artifact identities and verify
   one-to-one joins before loading the model.
2. Pre-register one strong patch boundary per divergence family from the
   existing boundary results; do not reopen a broad layer sweep.
3. Run a four-event debug panel with:
   - clean free continuation;
   - persistent post-scatter free continuation;
   - one-time post-scatter-to-clean residual patch followed by unpatched free
     continuation;
   - clean-to-clean no-op patch;
   - established early-boundary negative control.
4. Stop generation at the first complete row boundary or a recorded safety
   token limit. Do not silently repair malformed output.
5. Inspect hook/cache receipts and event-level trajectories before scaling.
6. If the debug gate passes, run all eligible strict events.
7. Compare results with the predeclared outcome map and stop. Do not
   automatically continue into renderer, selector, attention-head, coverage, or
   mixture work.

## Research Unit Closeout

Observed:

- The balanced debug panel passed its hook/cache controls and justified the
  pre-registered strict-panel expansion. The complete run evaluated all `18`
  prior strict next-token events; exact fresh clean/post row replay retained
  `10` events for the narrow causal claim.
- On those `10` events, the strong one-time layer-input patch recovered the
  post-scatter immediate divergent token in `100%` of cases, while exact full
  post-row recovery was `0%` and exact four-coordinate recovery was `0%`.
- The aggregate hides a decisive family split:
  - step-0 stop-versus-continue: `5/5` immediate continuation flips, but only
    `2/5` post-scatter phrase matches;
  - step-1 description competition: `5/5` post-scatter phrase matches; among
    the `5` coordinate-comparable events, `4/5` were closer to post-scatter
    geometry, with mean coordinate post-recovery `0.8591` across all five.
- Clean-to-clean no-op and the layer-13 early-boundary control preserved the
  clean row in `100%` of fresh replay-eligible events. Every accepted one-shot
  run recorded one prefill patch followed only by single-token cached calls.
- Primary artifacts:
  `/data/CoordExp/outputs/painted_gt/pvci_post_scatter_row_trajectory/e1_wrong_object_jitter_medium_strict18_20260710/`.

Interpretation:

- H1 is supported only for description-conditioned identity and approximate
  geometry, not as exact full-row trajectory copying.
- H2 receives the strongest support: continuation gating and current-instance
  identity/geometry are temporally and functionally distinct. A layer-19
  row-entry write can open continuation without reliably specifying the next
  object; a layer-23 description-boundary write reliably controls phrase and
  strongly pulls later coordinates toward the post-scatter object.
- H3 is rejected for the description family because effects persist beyond the
  immediate token, but remains a reasonable description of three of five
  stop-boundary events after the generic continue token.
- H4 remains a material limitation: only `10/18` strict next-token events had
  exact fresh full-row replay in both clean and persistent-post conditions.
  Results are therefore event-conditional and must not be generalized to all
  rows or checkpoints.

Next experiment seed:

- Test a two-anchor causal schedule rather than a final architecture: for
  stop-boundary events, write the established continuation state at row entry
  and then an independently captured identity state after
  `<|object_ref_start|>`; for description events, compare no geometry re-anchor
  with one post-scatter re-anchor at `<|box_start|>` or pre-`x1`. This directly
  tests whether the desired natural loop requires sequential control variables
  (`continue`, `current instance`, `geometry`) rather than one monolithic
  cursor. Keep renderer, selector, coverage, and training architecture outside
  that unit.

- Pending.

Evidence gate:

- Pending.

Supported:

- Pending.

Not supported yet:

- Full-row causal persistence, coordinate control, free-rollout stability, and
  any candidate runtime architecture remain unproven at unit creation.

Architecture update:

- None at unit creation. The candidate architecture space remains deliberately
  open.

Next decider:

- Complete the verified event ledger and four-event debug continuation panel.

Promotion decision:

- Not promoted. This is an active non-normative research unit.
