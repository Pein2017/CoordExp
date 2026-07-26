---
title: Source versus Transition Step 36 Forced-Opener Owner Selection
description: Same-boundary checkpoint comparison that fixes the continue decision and tests whether recent transition training improves one-row uncovered-owner realization.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-26-source-versus-transition-step36-forced-opener-owner-selection
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: executed_and_verified
updated: 2026-07-26
---

# Source versus Transition Step 36 Forced-Opener Owner Selection

Execution is complete. The decision-bearing interpretation is owned by
[results.md](results.md).

## Decision and Primary Question

At the same fixed 200 Source natural-terminal boundaries, after the canonical
new-row opener is supplied to both checkpoints, does recent transition step 36
produce a better strict set of verified uncovered physical owners than Source?

This isolates behavior after the choice to continue. The preceding Source-only
release established that opener forcing exposes 44 strict paired owner gains
relative to Source native release, while a provisional visual audit suggests
that the frozen owner ledger undercounts some visible objects. That result is a
Source capability estimate, not a recent-training effect. This unit measures
the missing checkpoint contrast.

## Strongest Alternative

Transition step 36 may primarily increase the probability of starting another
row without improving the conditional choice, category, instance, or geometry
of that row. Under this alternative, forced transition outputs will be equal to,
exchange with, or underperform forced Source outputs even if transition is more
likely to continue natively.

## Frozen Panel and Checkpoints

The panel is exactly the 200 distinct-image
`untouched_terminal_with_remaining_owner` boundaries frozen in:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality/panel-v1/manifest.json`

Its SHA-256 is
`3f0fb0a56463e66ace4fab89fc670d4113f0a727ed841a292f265e3f67523e87`.
Each boundary retains its literal Source-generated assistant prefix, frozen
covered-owner set, and frozen remaining-owner set.

The control is the already executed Source receipt set and reduction from the
paired natural-terminal forced-opener unit. The treatment checkpoint is the
first-divergence transition objective at step 36, loaded from:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-24-prefix-local-and-on-policy-owner-set-training/train-first-divergence-1440-long-learning-rate-1e-5-v1/runs/qwen3_vl_2b_first_divergence_transition_1440_event_long_learning_rate_1e-5/checkpoints/step-36`

The transition adapter tensor SHA-256 is
`7e49e8c31aa4d8dc72a309c5ba238dc50a517e51ce09acaa499386ba3362f61d`;
the transition special-token embedding tensor SHA-256 is
`33d65d575df398375e72994e2a197aaff0a257e26c088717e3b7689b3f494c3b`.

## Exact Contrast

For every boundary, execute transition step 36 twice under the same current
runtime used for Source:

1. **Transition native:** release at most one row from the literal boundary.
2. **Transition forced opener:** append only token `151646`
   (`<|object_ref_start|>`) and release the remainder of at most one row.

The decision-owning checkpoint contrast is forced transition versus forced
Source on the same boundary. Native transition is retained to complete the
two-by-two description and to distinguish a gate change from conditional-row
behavior. No description, coordinate, covered-set text, alternate prompt, or
intended owner is supplied.

## Primary and Secondary Outcomes

For boundary `i`, let `U_i` be the frozen remaining-owner set, `S_i` the strict
owners produced by forced Source, and `T_i` the strict owners produced by
forced transition step 36.

The primary owner sets are:

- checkpoint gains: `(T_i intersect U_i) minus (S_i intersect U_i)`;
- checkpoint losses: `(S_i intersect U_i) minus (T_i intersect U_i)`; and
- checkpoint net: total gained owners minus total lost owners.

Report the paired any-uncovered-owner transition table, discordant boundary
counts, an exact paired sign test for gain-only versus loss-only boundaries,
and a deterministic bootstrap interval for mean per-boundary owner net. A
positive aggregate without paired owner identities is insufficient.

Secondary outcomes are native and forced outcome distributions, exact row and
owner-set agreement, covered-owner repeats, valid unmatched rows, invalid rows,
description changes, match geometry, and results by prefix depth, scene-density
band, remaining-owner count, and earlier diagnostic-margin band.

The strict owner result remains the reproducible primary result. Because the
Source visual audit exposed raw-COCO omissions, checkpoint-changed unmatched
cases must be routed to an entity-first and geometry-second visual review before
claiming that a strict difference equals a visible-object difference. Human
review supplements rather than overwrites the strict ledger.

## Meaning-Bearing Invariants and Contract Gate

- The panel, candidate-pool, image, prompt-token, prefix-token, covered-owner,
  and remaining-owner identities must match the frozen Source receipts.
- Source and transition must share the base model, tokenizer, processor,
  template, image transform, full-model 32-bit floating point runtime, scaled
  dot-product attention, physical batch size one, greedy decode, parser, and
  owner matcher.
- The transition authored config must be
  `row-local-long-transition-lr1e5-step-36-64-hf.yaml`, SHA-256
  `1bcf9c0be2d5fb4889f67e5deef59e5a863c899f750f35f4ef9912e04b8a5018`.
- Only checkpoint adapter and special-token embedding delta may differ across
  the decision-owning forced arms. Runtime receipts must prove the loaded
  transition paths and tensor identities.
- Each arm uses repetition penalty `1`, temperature `0`, top-p `1`, at most 64
  new tokens, and malformed-row limit 2.
- A valid row is not a verified owner unless strict matching assigns a member
  of the frozen remaining-owner set. An unmatched row is not automatically a
  hallucination.
- A one-row conditional gain is not a final free-rollout set gain and does not
  establish preservation or natural termination.

## Smoke, Launch Gate, and Cost

Run at least one previously strict Source recovery and one Source unmatched
case through transition native and forced release. The smoke passes only if
the expected step-36 adapter and embedding delta are loaded, the base and
frontend identities match Source, prompt and prefix hashes match the manifest,
the forced token is exactly the canonical opener, raw outputs are attributable,
and a repeated smoke case is deterministic.

If the smoke passes, run eight stable boundary-ID shards across GPUs 0 through
7. This loads eight copies of the 2-billion-parameter checkpoint in full-model
32-bit floating point and performs two one-row releases per boundary. No
optimizer state or training is involved.

## Artifact Root and Stop

The immutable artifact root is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-26-source-versus-transition-step36-forced-opener-owner-selection/`

Expected products are smoke receipts, eight transition production receipts,
a transition-only reduction, a paired Source-versus-transition case ledger,
and a bounded visual-review packet for conclusion-changing unmatched cases.

Stop after the complete paired reduction, necessary visual adjudication,
durable bounded result, and user discussion. Do not launch training, alter the
prompt or output contract, promote an inference policy, run a longer forced
trajectory, or compare another checkpoint automatically.
