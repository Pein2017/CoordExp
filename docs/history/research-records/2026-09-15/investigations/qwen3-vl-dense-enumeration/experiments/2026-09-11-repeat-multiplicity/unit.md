---
title: Fixed-prefix repeat multiplicity and conditional next-row selection
description: Synthetic matched-history diagnosis of whether earlier exact-row multiplicity reweights recurrence to already-covered instances.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: packet_frozen_pending_gpu_grant
unit_id: 2026-09-11-repeat-multiplicity
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: unexecuted
updated: 2026-09-11
---

## Frozen question

From unchanged Stable50 on the original image, does reallocating a fixed number
of older, exact complete-row occurrences between two already-covered candidate
instances A and B change the next-row A-versus-B preference, while holding the
unique covered-row set, total prefix token count, most-recent A and B positions,
and final complete row fixed?

This is a **synthetic fixed-prefix conditional diagnosis**. It is not evidence
about natural policy frequency, physical-owner memory, learning, training
origin, KV or attention circuits, or generalization. Exact-row likelihood is
not the probability mass of an IoU neighborhood.

The strongest alternative is recency/order/last-geometry dependence rather
than an independent multiplicity effect. The paired block reversals below
counterbalance one simple older-order axis, but do not identify multiplicity
against every possible function of the older history. A consistent result is
therefore reported as bounded conditional history reweighting unless the
matched invariants and reversal pairs support the narrower description.

## Candidate admission

The cohort is fixed to `9813,158044,417044,502725`, with no backfill. Each case
must have visually admitted, same-description, valid, geometrically distinct
A and B rows that plausibly mark different visible instances. IoU at most 0.95
is necessary but not sufficient. Candidate rows are literal model rows from
the prior natural anchor or original-image translated continuation, never
edited coordinates or donor-image rows. A common literal final row C is used
when available. Unresolved cases are `HOLD`, not negatives.

Root owns visual admission. The draft packet and full-image overlays are under
the new output root; no model call is allowed until root freezes `packet.json`
and grants an exact command.

Root admitted `9813` and `417044`. Case `158044` is HOLD because A and B cover
book groups/stacks rather than resolved individual books; `502725` is HOLD
because A covers an overlapping knife/utensil group. There is no backfill. The
admission receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-recovery-parallel/mechanism-visual-admission.json`
with SHA256
`89ec11d4a9f2506aa5c3315dc237f1d8bab4c49ea9bac8a2759512e2088c777f`.

## Prefix family

Every admitted case uses six nine-row histories. Let the common base be
`[A,B]`, the common local suffix be `[A,B,C]`, and let `V` be one of:

| cell | varied block V | total A | total B |
|---|---|---:|---:|
| `a5_b3_fwd` | `A,A,A,B` | 5 | 3 |
| `a5_b3_rev` | `B,A,A,A` | 5 | 3 |
| `a4_b4_fwd` | `A,A,B,B` | 4 | 4 |
| `a4_b4_rev` | `B,B,A,A` | 4 | 4 |
| `a3_b5_fwd` | `A,B,B,B` | 3 | 5 |
| `a3_b5_rev` | `B,B,B,A` | 3 | 5 |

Thus each prefix is `[A,B] + V + [A,B,C]`. The final three rows, last A and B
positions, final row, row count, token count, and unique literal-row set are
identical within a case. The reversal pairs expose sensitivity to the order of
the varied block. If a case lacks a visually admitted distinct C, the same
admitted A or B row may be the common final row, but that choice must be frozen
before launch.

## Evidence and falsifier

For each prefix, native causal replay scores exact full-row A, B, C and EOS
separately. Preserve summed log likelihood and per-token mean; A and B must
have equal serialized length. Then acquire one fresh greedy continuation with
at most 512 free tokens and total action length at most 3084. The primary
behavioral readout is the first complete free row's exact/native-pixel-IoU>0.95
same-description recurrence to A or B, plus class-blind native-pixel strict
repeats, geometry-invalid complete rows, parser
drops, and native EOS versus horizon. Artificial prefix rows receive zero
natural owner credit.

The concrete falsifier for a directional multiplicity account is either:

1. the A-minus-B full-row log-odds does not move monotonically from A-heavy to
   balanced to B-heavy after averaging each reversal pair; or
2. the sign changes are dominated by disagreement within reversal pairs, or
   fresh free next-row behavior repeatedly contradicts the scored direction.

Failure of that falsifier would support only a bounded conditional reweighting
interpretation on admitted cases. Mixed or HOLD outcomes close this finite
panel without another regime.

## Identity, cost, and stop

- Stable50 adapter and Source embeddings/model; HF native FP32 SDPA.
- Greedy `temperature=0`, `repetition_penalty=1`, `top_p=1`, `top_k=0`, model
  defaults disabled, native EOS `151645`.
- One fresh exact natural anchor per admitted case.
- Six conditional prefixes per admitted case, four score replays and one free
  continuation per prefix: 14 generation calls and 48 score replays for the
  two admitted cases (below the lane caps 36 and 160).
- The conservative generated-token hard maximum is `2*3084 + 12*512 = 12,312`.
  `9,328` is only the expected total conditional on both fresh natural anchors
  exactly matching their saved lengths (100 and 3084); failures are accounted
  against the hard maximum.
- Two model loads in the proposed one-case-per-worker route; 1500 s,
  peak CUDA 12 GiB and RSS 16 GiB per worker. No training.
- One full-case production-shaped smoke is reused as that case's final evidence;
  the other admitted cases launch only after its native parity, scoring,
  durable readback, and resource receipts pass.
- Stop after the frozen six-cell panel for every admitted case, including
  HOLDs. No donor/image swap, extra cohort, regime, hyperparameter, special
  token or KV patch, or novelty claim.

## Owners

The output root is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-repeat-multiplicity`.
This unit and its task-local scripts own the synthetic contrast and receipts.
Root owns visual admission, launch authorization, final interpretation, and
acceptance.
