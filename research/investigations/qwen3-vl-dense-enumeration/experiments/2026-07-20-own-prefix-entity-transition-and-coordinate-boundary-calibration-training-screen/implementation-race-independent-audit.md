---
title: Independent Fixed-Point Audit of the Three Calibration Implementations
description: The lead agent's pre-external-review, pre-real-smoke comparison of the three implementations forked from the same frozen research contract.
type: investigation
role: implementation-audit
authority: non_normative_research
audit_status: static_audit_complete_real_smoke_pending
updated: 2026-07-20
---

# Independent Fixed-Point Audit of the Three Calibration Implementations

## Independence Boundary

This audit was completed before reading either external audit:

- `agent_review/codex-audit.md`; or
- `agent_review/opus-audit.md`.

The three worktrees share base commit
`2d3ea09ee988c9babb02ba5ae75fc640a2a80417` and were clean at the audited
heads:

| Implementation | Audited head |
| --- | --- |
| Command-line interface implementation | `68c4ea6f98b0e928180ed795f08568c9305d785a` |
| Application implementation | `485e5a4b8c2d1d9c498bc94e5b75d716830abbeb` |
| ClaudeX implementation | `b8a4689089c4eca432dd6ee8c0fb522f308afaac` |

The audit compared behavior against the frozen research unit and OpenSpec
change. It did not compare presentation quality or test count alone. No real
Qwen3-VL training smoke existed in any implementation at this point.

## Static Ranking

1. **Command-line interface implementation**
2. **Application implementation**
3. **ClaudeX implementation**

This is the frozen static ranking. The final execution ranking remains pending
the same lead-owned real Smoke A and Smoke B evidence.

## Why the Command-Line Interface Implementation Ranks First

It is the strongest scientific backbone because it has the most complete
chain from producer-owned exact rollout evidence to immutable state-bank
records:

- a standalone state-bank assembler;
- exact prompt, prefix, candidate, image, and checkpoint identity checks;
- explicit producer-declared greedy versus sampled candidate roles;
- an explicit set of physical entities already represented in the prefix;
- exact image-token placeholder validation against the active Qwen token
  identity and expected merged visual-token count;
- strong exclusion of blind evaluation images; and
- the clearest separation between exact rollout facts and reviewer-owned
  declarations.

Its larger code surface is a cost, but the added surface mostly protects
scientific identity rather than adding unrelated framework machinery.

### Blocking fixes before real training

1. Bind a coordinate correction to the same physical owner as the candidate
   trajectory receiving the gradient.
2. Validate the coordinate-value-to-token mapping against the active ordered
   coordinate token vocabulary.
3. Preserve and validate enough ordered boundary evidence to establish that
   the selected boundary is the first rejected coordinate.
4. Require an entity owner-resolution interval to begin at the shared
   next-row decision, candidate offset zero.
5. Guarantee that every enabled objective family contributes to every joint
   optimizer step, or build the state-bank stream so this property is proven
   and receipted.
6. Add one executable state-bank-to-replay-to-packing-to-loss-to-backward test
   without monkeypatching the central assembly seam.

## Why the Application Implementation Ranks Second

Its strongest contributions are:

- a concrete per-image physical-entity ledger;
- runtime coordinate-value-to-token validation;
- a family-balanced stream for joint optimizer steps; and
- a compact package that keeps state-bank, planning, replay, and training
  behavior close together.

It ranks below the command-line interface implementation because several facts
that decide candidate admission are still reviewer assertions rather than
producer-bound evidence. In particular, the schema does not prove that the
harmful branch is the actual greedy branch, does not reconstruct the set of
physical owners represented by the exact prefix, and leaves candidate
generation policy in a weakly typed mapping. The token-type gate also consumes
all declared sites even when a single-objective arm should select only that
objective's sites.

### Blocking fixes before real training

1. Bind the sole harmful branch to a producer-declared exact greedy candidate;
   require positives to be producer-declared same-prefix sampled branches.
2. Store and validate the physical-owner set represented by prior prefix rows;
   derive covered versus uncovered status from it.
3. Require owner-resolution paths to begin at candidate offset zero.
4. Establish first-wrong-coordinate ordering, not only one opaque reviewed
   coordinate declaration.
5. Filter token-type-gate sites by the objective enabled in the current arm.
6. Require the requested DoRA target surface to preserve the source adapter
   exactly at step zero and reject newly initialized target modules.
7. Validate the image-token placeholder against the active token identity and
   expected visual-token count.

## Why the ClaudeX Implementation Ranks Third

It is the smallest and easiest implementation to read. It correctly binds the
harmful candidate to an `actual_greedy_candidate_id`, filters selected gate
sites by enabled objective, reuses the existing trainer, and implements the
core 32-bit floating-point loss formulas correctly.

It ranks third because its state-bank evidence model is too thin for the
review-heavy scientific contract. It has no explicit physical-entity ledger
or prefix-covered-owner set, weak review provenance, no standalone state-bank
assembler, weaker image-token placeholder identity checks, and no real replay
parity path. Its receipt also marks a replay task complete using a synthetic
compact-versus-full forward comparison rather than a stored source event.

### Blocking fixes before real training

1. Require a trusted physical owner for geometry, or a typed independent
   same-owner correction reference.
2. Establish first-wrong-coordinate ordering and exact value-to-token binding.
3. Require owner-resolution intervals to begin at candidate offset zero.
4. Bind premature terminal output to the exact active terminal token.
5. Add typed, nonempty physical-owner and review provenance.
6. Add an explicit prefix-covered-owner ledger and validate duplicate versus
   uncovered status against it.
7. Validate the image-token placeholder against the active Qwen image token
   identity and expected visual-token count.

## Shared Findings Across All Three Implementations

All three correctly implement the main mathematical shape:

- exact candidate path log probability is a sum, not a token mean;
- candidate aliases are reduced by physical owner before the normalized
  smooth maximum over distinct positive owners;
- premature terminal output is a one-token null-owner branch;
- the coordinate objective compares accepted coordinate mass with the actual
  wrong token;
- objective math is performed in 32-bit floating point;
- candidate groups remain atomic within a micro-step;
- canonical supervised fine-tuning replay and Gaussian coordinate smoothing
  are absent; and
- the existing trainer, optimizer, gradient clipping, checkpoint, and normal
  inference paths are reused.

All three also share two launch-evidence gaps:

1. The selected special-token embedding delta is installed as a separately
   trainable parameter. This contradicts the newly frozen treatment of
   **language-tower Weight-Decomposed Low-Rank Adaptation only**. The source
   delta must still be loaded, but it must be excluded from optimization and
   step-zero parity must be re-established.
2. The trainable-surface and post-backward finite-gradient reports are checked
   in memory but not persisted as decision-grade run artifacts.

## Decision

Use the command-line interface implementation as the likely merge backbone,
subject to the mandatory fixes and real smokes. The application
implementation's joint-step family balancing is the clearest evidence-backed
cherry-pick. Do not combine entire packages.

The next gate is not another synthetic test-count comparison. It is one
lead-owned neutral fixture containing exactly one reviewed entity-transition
event and one reviewed first-wrong-coordinate event, compiled into each
implementation's native schema and executed through the real Qwen3-VL
forward, backward, optimizer, checkpoint, and ordinary inference paths.
