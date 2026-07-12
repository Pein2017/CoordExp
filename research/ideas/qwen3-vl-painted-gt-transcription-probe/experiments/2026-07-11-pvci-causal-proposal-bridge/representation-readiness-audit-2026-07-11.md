---
title: PVCI Representation Screen Readiness Audit
description: Pre-launch claim-validity audit for complete proposal controls, cohort production, and strict representation gating.
type: idea
role: research-audit
authority: non_normative_research
status: active
topic: qwen3-vl-painted-gt-transcription-probe
updated: 2026-07-12
---

# PVCI representation-screen readiness audit — 2026-07-12

## Verdict

`GO` for representation promotion and for proceeding to the bounded
metric-bearing own-prefix panel. The frozen A/B/C 512-step runs completed, the
in-sample v2 gate is `in_sample_learnable`, and the held-out v2 gate is
`promote`. This is a representation-only result: it does not promote the
causal bridge, own-prefix behavior, detection recall, or a final architecture.

The earlier `HOLD` below is retained as prelaunch history. It was resolved by
the producer, cohort, and tie-aware gate corrections documented in the
post-run result note.

This hold is not a model-quality result.  It was issued before metric-bearing
training because the old path could not distinguish learned object-specific
visual support from position, prefix-depth, global-image, or content-position
shortcuts without accepting unproven control artifacts.

## Accepted evidence

- A4/B2/C1 two-step VF6 passed on the shared cache/view identity; see
  `vf6-results-2026-07-11.md`.
- C-on/C-off forced replay proved an active bounded pulse and exact C-off
  no-op mechanics.
- The real-runtime one-row C producer smoke wrote a finite normalized native
  proposal with checkpoint/config/cache/view provenance:
  `/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_vector_producer/C_step2_rank0_row1_20260711T1900Z`.
- The current strict evaluator correctly returns `rerun` for incomplete
  evidence and `contract_fail` for stale metric-code fingerprints.  Script
  execution `PASS` is not representation-gate `promote`.

## Readiness gaps found

1. The producer loaded only the first rank-local microstep and one native row;
   it could not stream the frozen cohort.
2. Non-native representation controls were named but not executable.
3. Their exact formulas, seed/domain separation, and provenance were not
   frozen before scores.
4. Additional gate-input metric rows were not bound to matching producer
   headers, checkpoint/config/cache/view/cohort, or control-generator identity.
5. Cohort equality omitted actual support/grid identity.
6. Jitter and duplicate-history groups were required to exist but were not
   substantive gate terms.
7. Same-area/center/aspect distractors were recorded but not decision-bearing.
8. The compact view receipt exposed a declared full-manifest digest but did
   not itself contain enough payload to recompute that digest.

## Frozen correction

The active unit now freezes `pvci-representation-controls-v1`:

- native aligned query/content;
- fixed coordinate-only sinusoidal visual states;
- observed-prefix-depth-only sinusoidal query with future-count/target leakage
  forbidden;
- repeated FP32 image-mean states as a global/uniform null;
- deterministic nonidentity content-to-cell shuffle;
- score-blind matched-object specificity diagnostics.

All A/B/C controls must share exact row/grid/support identity and carry
content-addressed transform receipts.  Missing evidence is `rerun`, observed
threshold failure is `hold`, and malformed/stale/mismatched evidence is
`contract_fail`.

Representation evaluation is now explicitly two-stage:

1. native A/B/C training-monitor events over the frozen 12,288 views:
   in-sample learnability diagnostic only, without full-vector control
   reserialization over roughly 93k supervised rows;
2. complete five-control panel on frozen val200 plus 120 dense validation
   images, with eligible images expanded
   into canonical/jitter/duplicate-history views: held-out binding confirmation.

A pilot subset is mechanics-only.  It cannot replace either gate.

## Launch requirements

Before an independent audit can return `LAUNCH`:

1. stream the full declared row cohort without reusing one packed forward;
2. execute every frozen control with exact native parity and hook restoration;
3. require provenance sidecars for every gate input;
4. bind support/grid/cohort/control recipe identities across A/B/C;
5. enforce noncanonical retention and matched-distractor specificity;
6. materialize and hash the held-out validation view manifest before any
   candidate score is inspected;
7. pass focused adversarial tests and one real multi-row/multi-control GPU
   smoke;
8. obtain a fresh independent innovation-risk audit verdict.

No 512-step run was launched under this hold.

## Held-out cohort preflight update

The score-blind held-out materializer subsequently passed at:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_bridge_heldout_preflight`

It reproduced the frozen 120-image digest `e7929e...`, combined those images
with the exact disjoint val200 set, and materialized 906 views over 320 unique
images.  Counts are canonical `320`, jitter `293`, duplicate-history `293`;
the 27 singleton images are canonical-only with both history-dependent
conditions explicitly unavailable.  The full view payload/file SHA256 is
`2554969aaebb43f86a03e85a639f28cf06c0b7ff81b2763bf4c46e2cc474b75c`.
This closes cohort selection/view construction, but not evaluation-only
encoding/packing or proposal production.

## Producer/control attestation update

The training-cache producer index and a bounded real all-control smoke now
pass.  The full index covers 12,288 views and 93,108 supervised rows.  The
real C1 step-2 smoke is:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_vector_producer/bounded_controls_two_microsteps_20260711T2015Z`

For two distinct cached microsteps, one Qwen forward per microstep recovered
the exact layer-0 query/visual states and produced all five controls by FP32
bridge rescoring.  A second native repeat was byte-identical; content shuffle
was nonidentity; image-mean maximum uniform error was `1.45e-11`; row-depth
used no forbidden future/target field; hooks and gradient state were restored;
and CUDA allocation returned from an 8.50 GB peak to a 9.6 MB context
residual.  Streaming output omits redundant pooled vectors by default while
retaining their hash/norm receipt.

This closes the producer/control mechanics gap.  It does not close the
held-out evaluation-cache or final gate audit.

## Held-out evaluation-cache update

Canonical Qwen tokenization, image planning, packing, supervision, position
inputs, proposal plans, and cache serialization were reused to build the
separate diagnostic cache referenced by:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_bridge_heldout_preflight/cache_descriptor.json`

Its fingerprint is `eda43789...`; it contains 906 segments in 114 microsteps
and 3,639 proposal rows, with zero zero-positive bags.  The descriptor binds
the source/view hashes and tokenizer/processor identity and is distinct from
the frozen training cache.

Producer support for this arm-neutral evaluation-cache descriptor subsequently
passed.  The complete held-out index attestation is:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_vector_producer/heldout_descriptor_index_smoke_20260711T2145Z`

It binds 906 views, 320 source images, 3,639 proposal rows, all five controls,
the explicit held-out descriptor, and the frozen unsharded `rank=0,
world_size=1` execution contract.  A real one-microstep GPU integration then
produced 46 rows for every control with one Qwen forward and passed the strict
evaluator at:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_proposal_vector_producer/heldout_descriptor_one_microstep_smoke_20260711T2150Z`

The evaluator returned execution `PASS` and gate decision `rerun`, which is the
required result for an intentionally incomplete C-only, one-microstep smoke.
The producer/evaluator contract therefore no longer blocks launch by itself.
The remaining launch gate is the executable in-sample run-root provenance join
plus a fresh independent audit of the complete path.

The earlier root
`production_format_two_microsteps_all_controls_20260711T2110Z` is retained as
schema-only historical diagnostics.  It was produced before evidence-stage
hardening and labels training-cache COCO-train rows as `heldout_binding`; the
current producer correctly rejects that combination.  It is not held-out
evidence and must not be used for a representation claim.

## Final prelaunch verdict

The executable VF6 run-root join passed in explicit provenance-only mode over
the authoritative A4/B2/C1 roots.  It validated 1,371 rank-0 monitor rows,
world size 8, completed two-step schedules, arm-specific checkpoint/config
identity, and shared cache/view/cohort/code identity.  It correctly returned
`rerun`, admitted no decision evidence, and reported the absent steps 385–512.
The durable artifact is:

`/data/CoordExp/outputs/probes/coordexp_swift/pvci_in_sample_learnability_preflight_A4_B2_C1_20260711`

The strict post-run path still requires a resolved, completed, latest
checkpoint at exactly step 512 and is covered by a synthetic 512-root
integration test.  Independent innovation-risk re-audit therefore returned
`LAUNCH/GO` for the matched A/B/C screen while retaining interpretation `HOLD`
until the post-run gates execute.

Arm A was launched on all eight A100 GPUs from the frozen production config as:

`/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_A-pvci512-A-20260711T194750Z`

The launch uses the shared frozen cache and view identities.  B and C remain
unlaunched until A completes and its runtime/checkpoint receipts pass.

## Post-run representation verdict — 2026-07-12

The matched A/B/C 512-step runs completed under the frozen contract:

- A:
  `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_A-pvci512-A-20260711T194750Z`
- B:
  `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_B-pvci512-B-20260711T220028Z`
- C:
  `/data/CoordExp/outputs/research/pvci_causal_proposal_bridge/pvci_causal_proposal_bridge_C-pvci512-C-20260712T002318Z`

The in-sample v2 decision is `in_sample_learnable`; the regenerated
v2-compatible output decision retained the exact prior decision digest
`f98bf3805c077e1dc1d700483037e4b089fb1cf2f1b40e5db3275d3ab5e1b956`.

The complete held-out representation panel is recorded at
`/data/CoordExp/outputs/probes/coordexp_swift/pvci_heldout_representation_gate_ABC_512_v2_20260712`.
Its decision artifact is `representation_gate_decision.json` (SHA256
`94e3fee01514eaedb55bed44cf9d62f18296d102e5684dfdc048d5e3caa010ec`) and
its producer/evaluator receipt is `representation_probe.receipt.json` (SHA256
`b9d82bcdff6a97703eb3650611a5a7b39d3d9d40531d9821a821060a0f980529`). The
gate selected C by the frozen candidate priority `(C,B)`, not by a
demonstrated C-over-B superiority: both B and C passed all 29 applicable
held-out checks.

The support-quartile policy is tie-aware and value-based. The observed
cutpoints are `[972, 972, 1014]`; q2 is unattainable because the tied support
value cannot be split, while q1, q3, and q4 are attainable and required. The
gate records this explicitly rather than treating an unreachable nominal q2 as
missing evidence. The completed held-out cohort has 320 images, 906 views, and
3,639 proposal rows; its source/view and cache identities remain those declared
in the unit.

An independent promotion audit returned `GO` for this representation-only
scope. The audit also records remaining provenance caveats: the resolved
configuration hash is retained as a provenance field rather than independently
replayed here; the primary receipt SHA is a receipt binding, not a full replay
of every producer tensor; and no independent full A/B/C replay was performed.
These caveats limit the claim but do not invalidate the executed gate.

This result promotes the learned proposal representation as a follow-up input
to the natural own-prefix causal/safety panel. It does **not** promote
`causal_bridge_supported`, rollout improvement, detection improvement, or any
final architecture. Those remain `not_promoted` pending the separate
metric-bearing own-prefix controls and safety gates.
