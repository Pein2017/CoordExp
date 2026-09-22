---
title: Equal Mandatory Repair Gate Before Shared Real Smoke
description: One semantic qualification gate applied to all three implementation-race candidates before the shared real Smoke A and Smoke B.
type: investigation
role: launch-gate
authority: non_normative_research
gate_status: frozen
updated: 2026-07-20
---

# Equal Mandatory Repair Gate Before Shared Real Smoke

## Purpose

The three as-submitted heads remain immutable evidence of implementation
quality. Each implementation receives one bounded repair round before the
shared real smokes. Repairs may only make the frozen scientific contract
executable and verifiable. They may not add a new objective, change optimizer
values, weaken the fixture, alter the Qwen3-VL inference graph, or import an
entire competing implementation.

Passing this gate qualifies an implementation for Smoke A. It does not erase
the defects found in the as-submitted implementation and does not determine
the final ranking by itself.

## Universal Semantic Requirements

### Exact candidate provenance

- Exactly one producer-declared greedy candidate exists per event.
- The sole harmful branch is that exact greedy candidate.
- Every positive branch is a producer-declared same-image, same-prefix sampled
  branch.
- Generation mode, seed, temperature, top-p value, repetition penalty,
  checkpoint identity, prompt identity, and prefix identity are typed and
  checked rather than stored as an arbitrary mapping.

### Prefix coverage and physical owners

- The record stores the unique physical owners represented by every prior
  object row in the exact prefix.
- A transition positive owner is absent from that set.
- A physical-duplicate harmful owner is present in that set.
- Every claimed owner is a nonempty stable per-image physical-entity
  identifier with typed review provenance.
- Candidate aliases are grouped only after this owner identity is validated.

### Entity score path

- Every nonterminal owner-resolution interval begins at candidate offset zero,
  the shared next-row decision.
- The interval ends at the shortest token prefix that uniquely resolves the
  owner.
- Every included token has one canonical intended token type.

### First-wrong-coordinate evidence

- A geometry event has a trusted candidate owner, and the correction owner is
  the same physical owner.
- The record stores an ordered prefix of boundary observations in the fixed
  order `x1`, `y1`, `x2`, `y2` through the selected boundary.
- Each observation stores its candidate token offset, actual coordinate value,
  accepted coordinate values, tolerance axis, and review provenance.
- At runtime, every actual and accepted coordinate value is resolved through
  the active ordered coordinate-token identity. Independently supplied token
  identifiers are not trusted.
- Every earlier boundary is inside its own accepted set. The final stored
  boundary is outside its accepted set and is therefore the first wrong
  coordinate. Later coordinates receive no direct loss.

### Objective isolation and exposure

- A single-objective arm gates only sites selected by that active objective.
- Joint-arm duplicate declarations for the same causal segment and logits
  position are deduplicated; conflicts fail.
- Event scheduling may reorder the frozen bank but may not duplicate events,
  omit events, or change per-event exposure to force a family into every
  optimizer step.
- If one joint step has no eligible event for one objective, the absence is
  visible and handled without inventing gradient. The complete frozen schedule
  must still exercise both objective families.

### Exact Qwen3-VL replay

- Stored prompt, prefix, and candidate token identifiers are replayed without
  decoding and retokenizing.
- Image bytes, dimensions, executed prompt, image-token placeholder identity,
  one contiguous placeholder run, and expected merged visual-token count are
  checked against the active processor.
- Source adapter, selected-token embedding delta, tokenizer, special-token,
  prompt, processor, and inference-policy identities are bound before any
  gradient-bearing forward.

### Trainable surface

- The source selected-token embedding delta is loaded to reproduce the source
  checkpoint but is frozen.
- The optimizer owns only language-tower Weight-Decomposed Low-Rank Adaptation
  parameters.
- Vision Tower, multimodal MLP aligner, base language parameters, and selected
  token delta have no optimizer ownership and no gradient-bearing parameters.
- Step-zero source logits on the shared fixture match the frozen source
  composition within the declared tolerance.

### Durable smoke evidence

- Persist the validated trainable-surface receipt.
- Persist the post-backward finite-gradient report, including finite nonzero
  gradient norm and optimizer-update status.
- Persist pre-update and post-update transition and coordinate target margins
  in the same evaluation mode.
- Smoke A passes only if every declared margin increases by more than the
  frozen numerical epsilon after the tiny update.
- Reload the produced checkpoint through ordinary inference and prove no state
  bank or calibration controller enters inference.

## Repair and Stop Rule

1. Preserve the three original heads.
2. Apply one bounded repair commit per worktree against this same gate.
3. Freeze all three repaired heads before the neutral fixture is compiled into
   native schemas.
4. An implementation that fails this pre-smoke gate or Smoke A is disqualified
   with a visible reproducible reason; its numerical output is not interpreted.
5. After repaired heads are frozen, no candidate-specific repair is allowed.
   A genuine neutral-fixture defect may be corrected once for all candidates,
   followed by restarting every Smoke A run.
6. The implementation race ends when each candidate has passed Smoke A and
   Smoke B or has one visible qualification failure. If no candidate passes,
   select no implementation.
