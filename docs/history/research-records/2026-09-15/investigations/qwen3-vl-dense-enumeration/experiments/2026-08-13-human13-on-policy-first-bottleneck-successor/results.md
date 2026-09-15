---
title: Human-13 On-Policy First-Bottleneck Successor Results
description: Decision-grade same-panel result for transactional full-row and first-greedy-blocker native consolidation.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-08-13-human13-on-policy-first-bottleneck-successor
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified_bounded_panel
updated: 2026-08-13
---

# Result

## Disposition

**HOLD both frozen objectives. Do not promote either arm or continue this fixed
dose.** First-bottleneck remains the narrower primitive, but the completed
pilot does not show that it is safe.

The production-shaped force-reject vertical passed candidate scoring, packed
update, private proposal checkpoint, HF fp32/SDPA clean decode, rejection, and
complete model/optimizer/scheduler/counter/CPU+CUDA-RNG rollback. The bounded
pilot then made eight independent one-update proposals from restored Source per
arm. All 16 were rejected; no accepted checkpoint exists.

## Decision-owning result

| Arm | Attempts | Accepted | Selected target compiled | H gained / proposal | G lost / proposal | Unique owners | Duplicates | Malformed | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Source | — | — | — | 0 | 0 | 173 | 9 | 11 | 244 |
| O-First-Safe | 8 | 0 | 4/8 | 3–6 | 1–4 | 174–177 | 9–34 | 1–14 | 243–299 |
| O-Full-Safe | 8 | 0 | 5/8 | 1–5 | 1–5 | 172–177 | 8–54 | 0–15 | 249–310 |

Every proposal lost at least one protected Source owner. Both arms therefore
hit the declared eight-attempt cap with zero accepted updates. All 16 rejected
proposals restored an identical within-attempt transaction digest and their
rollback decodes reproduced the frozen Source behavior.

The candidate sequence covered eight distinct aliases, five owner identities,
and four images. It included all four native aliases for `gt:6040:14`, then
`gt:7511:42`, `gt:14038:10`, `gt:2685:28`, and `gt:7511:39`. Thus the negative
result is no longer only a top-candidate result. It remains bounded by the
eight-attempt shortlist policy, the fixed optimizer dose, and this panel.

## What the evidence says

1. **K-native support is usable but not sufficient.** Every proposal recovered
   some frozen K-hit H owners, but the specifically selected target became part
   of clean greedy only 4/8 times for O-First and 5/8 for O-Full. A successful
   forced continuation is therefore not a reliable estimator of post-update
   greedy compilation at this dose.
2. **Owner exchange is the invariant failure.** All 16 proposals lost protected
   G owners. Duplicate, malformed, and row burden varied substantially, but
   protected-owner loss did not.
3. **Full-row CE is not the remedy.** It sometimes compiled one additional
   selected target, but reached 54 duplicates, 15 malformed rows, 310 rows, and
   twice decreased total unique owners. O-First avoided those worst tails yet
   still lost G owners in every attempt.
4. **Net recall is an unsafe selector.** Some proposals reached 177 owners, but
   only by exchanging protected owners for H/M owners. None satisfied the
   owner-preservation contract.

The strongest next discriminator, if a new unit is authorized, is not another
suffix search. It is lower/adaptive dose or an explicit Source-owner
preservation constraint/watch that is evaluated by the same post-update clean
greedy gate. This result does not establish that such a successor will work.

## Artifacts

- Verified bounded-pilot summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-on-policy-first-bottleneck-successor/matrix-retry-v5/bounded-pilot-summary.json`,
  SHA256 `9b71d436e501948f1232e2c10d9d2c582b0828038c2610f6d7cd2c9e5a3e41e7`.
- The summary binds all 16 attempt-receipt hashes and is mechanically replayed
  against proposal decodes, manifest matching, selected-target identity, gate
  reasons, and transaction equality.
- Full gained/lost owner identities:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-on-policy-first-bottleneck-successor/matrix-retry-v5/bounded-pilot-owner-identities.json`,
  SHA256 `359e364d98da20e93d129c46ef2469ed0a2af2e4700cb4ee0abf08afdeaad70c`.
- Vertical mechanics summary:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-13-human13-on-policy-first-bottleneck-successor/vertical/o-first-safe-force-reject-v3/vertical-summary.json`,
  SHA256 `64d688a07a5f3979339903d789777b85bef69c2cd3cd2dba03b91c41f8601a61`.
- The earlier one-proposal summaries under `matrix/` remain immutable historical
  evidence but are superseded for the bounded-pilot conclusion by
  `matrix-retry-v5/bounded-pilot-summary.json`.
- Peak GPU allocation was not durably recorded and is not reconstructed from
  interactive monitoring. No private proposal files remain.

Each arm processed 2,444 packed aliases in 329 zero-padding packed forwards,
recomputed 1,019 candidates on HF, released 32 forced continuations, and made
eight proposed optimizer steps. O-Full/O-First summed 3,169.5/3,127.3 clean
decode seconds over 221 image-decode rows and occupied 4,505.8/4,463.6 seconds
from Source artifact to final attempt receipt. These are summed per-image decode
times and artifact wall spans, not GPU-kernel timings.

## Claim boundary

This is exact Human-13 same-panel, fresh-Source, one-update evidence at a fixed
optimizer dose. It does not establish validation/generalization, K-miss support
expansion, population prevalence, architecture sufficiency, or the outcome of
lower/adaptive dose and explicit preservation objectives. It is sufficient to
reject promotion or further same-dose execution of these two frozen arms.
