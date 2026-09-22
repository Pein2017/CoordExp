---
title: Image2299 protected-null output distillation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-30-image2299-protected-null-output-distillation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: cold_not_reached_stage1_active_competitor_exhausted
updated: 2026-08-30
---

# Image2299 protected-null output distillation

## Disposition

`stage_exhausted`: mechanically valid bounded negative. Stage 1 exhausted its
active-competitor closure budget before any warm candidate or cold augmented
model greedy evaluation. The unit closes the two-closure target+competitor
family at position 301; it does not execute or claim the target-only cold
success successor.

## Frozen question and contrast

From the frozen r32 checkpoint, does one persistent sparse output-only linear
residual distill the five controlled decisions in the terminal-EOS witness into
an unforced ordinary-greedy trajectory with all 38 person owners, while being
numerically null under one frozen FP64 apply contract on every observed
non-intervention hidden state?

The contrast is the unchanged r32 model versus the same model plus this one
checkpointed residual.  Both use the original image and prompt, HF greedy
`argmax`, no sampling, beam, forced token, prefix table, controller, or logits
processor.  Only a naturally generated row-aligned EOS counts.

The strongest alternative is not another small DoRA dose.  The first safe r32
branch remains about 7.85 logits away while prior DoRA motion was small and
owner-exchange-prone.  The decisive alternative here is that the intervention
hidden states have too little component outside the protected-state span, or
that a bounded output residual cannot close the live full-vocabulary
competitors without changing a protected decision. This run supports the
latter only for the active-competitor family tested below.

## Immutable specimen

- Terminal witness receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-terminal-eos-seal/20260830T-image2299-terminal-eos-seal-v1/receipt.json`
  with SHA-256
  `d9cb113ae3f7cbe4b080fbe4f8faf76f68d36d9ca2844940b27560d330ab4495`.
- Witness runner snapshot SHA-256:
  `1247d4cc901f2b95116cfd8b8c7fa5166ad42a5e0ff9c97d316b902c336baf0a`.
- Frozen r32 checkpoint:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step`.
- r32 model receipt SHA-256:
  `a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c`.
- r32 DoRA surface SHA-256:
  `2ac0e8a963d91ac7a4ba42322272889efd74bd46578c1ac44e16e531d1246676`;
  frozen surface SHA-256:
  `c16f0b52a2d5ce5ccfa1b239a0934a9ed27945278fc65e97d56cb8d50c4877d5`.
- Prompt-token SHA-256:
  `33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb`;
  image SHA-256:
  `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`.
- Controlled route: 370 tokens, route SHA-256
  `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`,
  38 persons, 3 ties, 41 strict owners, zero hard debt.

The controlled route, its four source-node routes, intervention positions, and
final row are read only from that one receipt.  Independently valid rows or
trajectory unions are not permitted.

## Algorithm

Capture the final-RMSNorm hidden state and frozen full-vocabulary logits for the
ordinary r32 route and the controlled route.  The five intervention blocks are
the four recorded appended prefixes
`[row_open, person, object_ref_end, box_start, x1]` and the final complete
gt18 row plus EOS.  A block position is positive only when its controlled token
is not already strict top-1 under its exact controlled prefix; there are at
most 30 positives.

Every other controlled-route state, every non-positive ordinary-r32 state, and
every accepted earlier-stage state is protected.  If `H0` is the protected
hidden-state matrix, form an orthonormal basis `Q0` of its row span and project
each positive state through

```text
P0 = I - Q0 Q0^T.
```

Let `B` be an orthonormal basis of the projected positive span, with rank at
most 30.  For only the selected target and active-competitor token rows, solve
one sparse residual

```text
delta_z_s(h) = a_s^T B^T h
```

by minimum normalized L2 norm, subject to every accumulated target beating its
current full-vocabulary competitors by `0.01`.  The runtime stores the
collapsed rows `D = A B^T` in FP64, casts output-head hidden states to FP64,
computes `h @ D.T` in FP64, and only then casts the correction to the logits
dtype.  It hashes the protected hidden matrix and requires the maximum absolute
protected correction to be at most `1e-10` in both warm and cold execution.
This is the complete meaning of *numerically null*; no bit-exact real-arithmetic
claim is made.  The normalized residual norm is capped at 1, and there is no
input-embedding path.

Proceed through the five intervention stages.  At each stage, solve all
accumulated constraints from zero, full-vocabulary recheck them, then run
ordinary greedy from the original prompt.  A newly exposed competitor or the
earliest changed protected decision may be added and the total system re-solved
at most twice per stage.  Rejection never mutates the incumbent payload.

Intermediate exact source-node replay is an execution precondition because the
later hidden states are bound to that prefix; it is not the final scientific
estimand.  Final acceptance is match-level and may use an owner-equivalent
route.

## Acceptance, bounds, and stop

Final success requires a fresh-process load of the residual-bearing augmented
model followed by ordinary unforced greedy decoding with:

- all 38 person owners and the same 3 retained ties;
- exactly 41 strict matched predictions/owners;
- zero duplicate, malformed, ambiguity, unmatched, unsupported, unknown,
  parser-drop, and token-budget debt;
- natural row-aligned EOS;
- identical warm/cold route, owner ledger, and residual hashes.

This is one GPU, world size 1, at most 30 positive states, rank at most 30,
15 total solves, 15 warm greedy candidates, two model loads, 1,800 seconds,
16 GiB peak reserved memory, and 100 MB of artifacts.  DoRA, tied embeddings,
MLP aligner, vision tower, and base weights remain bit-identical.

Stop immediately on cold success, numerical-null infeasibility, normalized norm
above 1, two exhausted closures at any stage, loss of an accepted owner/tie,
new hard debt, non-finite or failed full-vocabulary constraints, non-natural
EOS, or warm/cold mismatch.  Do not widen rank, cap, surface, or data under this
unit.

The only permitted positive claim is a single-specimen augmented-model greedy
result.  It does not establish transfer, general enumeration learning, a base
r32 greedy result, or 8/8 tie recovery.

For this completed run, no positive augmented-model greedy claim is made:
ordinary greedy remained unexecuted in this unit after stage-1 exhaustion. The
distinct [target-only successor](../2026-08-30-image2299-target-only-protected-null-distillation/results.md)
later executed and began the global/norm-release chain; it does not change this
unit's bounded negative.
