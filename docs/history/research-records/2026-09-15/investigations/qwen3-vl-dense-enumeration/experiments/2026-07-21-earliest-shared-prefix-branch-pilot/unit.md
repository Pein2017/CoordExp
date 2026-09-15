---
title: Earliest Shared-Prefix Branch Pilot
description: Test whether a sampled branch introduced at the first exact token divergence can be transported to a greedy rollout and whether a later sampled-only owner requires its preceding sampled history.
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-07-21-earliest-shared-prefix-branch-pilot
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: executed_with_native_replay_parity
updated: 2026-07-21
---

# Question

The three selected low-temperature routes contain physical owners that the
native greedy route does not cover within a fixed row budget.  Their full
generated token sequences first diverge very early, inside the first row.
This pilot asks two bounded questions:

1. Does forcing only the sampled branch from that first divergence through the
   remaining boundaries of the first row change later greedy coverage?
2. If that is insufficient, does supplying the sampled complete history before
   the first sampled-only owner make the owner path portable to greedy
   continuation?

The pilot is causal and local.  It is not a proposal for a new architecture,
does not inject a detector or object slot, and does not train a model.

# Frozen routes

The cases are derived from the reviewed route-local certificates in the
preceding individual-trajectory audit.  They use the same prompt, checkpoint,
coordinate representation, and low-temperature route files.  The first
sampled-only owner at a sixteen-complete-row budget is used as the target:

| Case | Seed | First sampled-only owner row | Target owner |
| --- | ---: | ---: | --- |
| `person-5001-first-sampled-only` | 21015 | 5 | `5001:1269467` |
| `person-7511-first-sampled-only` | 21012 | 4 | `7511:-169` |
| `wine-glass-2685-first-sampled-only` | 21008 | 6 | `2685:-78` |

The exact route and artifact hashes are recorded in [cases.json](cases.json).
The source checkpoint and reviewed route certificates are inherited from the
preceding experiment; no new annotation is silently promoted to truth.

# Intervention levels

## Level A: earliest branch in the first row

Find the exact longest common generated-token prefix between the full greedy
and sampled sequences.  In all three cases it ends before the first differing
coordinate token.  Force the sampled row from that first differing token
through successive boundaries (`x1`, `y1`, `x2`, `y2`, and row close), then
release native greedy generation for the remaining fixed budget.

## Level B: sampled history before the target owner

If Level A does not produce a non-negative fixed-budget set change, supply the
complete sampled rows before the target row, then force the target row only
through the same semantic boundaries and release greedy generation.  The
target row is not counted as a forced-context owner.  The purpose is to test
whether the route needs a persistent preceding state rather than to claim that
the injected row was discovered by the decoder.

# Measurements

Every result reports separately:

- owners in the forced sampled context;
- owners in the partially or completely supplied current row;
- owners found in the released suffix only;
- the fixed-budget union and its change from native greedy;
- target acquisition in the current row versus the released suffix;
- duplicate, unresolved, malformed, and token-budget receipts; and
- exact prompt and native replay parity.

An owner appearing only in forced context is never treated as a decoder
discovery.  A completely supplied target row is retained as a state-response
observation but is not a training or causal-acquisition claim.

The first sampled row is not uniformly better geometry than the greedy first
row: reviewed intersection-over-union is approximately `0.870` versus
`0.750` (image `5001`), `0.711` versus `0.789` (image `7511`), and `0.968`
versus `0.935` (image `2685`).  Therefore a Level-A change only demonstrates
that an early branch can influence later decoding.  It does not authorize
training the sampled coordinate tokens.  A training-positive must remain an
independently reviewed uncovered-owner path with non-negative downstream set
value.

# Interpretation boundary

A positive Level A result supports a local decision-ranking explanation.  A
positive Level B result with no Level A result supports a path-history or
state-transport explanation.  A result that merely repeats forced owners but
does not improve the released suffix is not a treatment signal.  Three cases
are intentionally too small for a population claim; they only decide whether
the next collection/training screen should use early branch events or a
history-aware state.
