---
title: Image2299 target-only protected-null distillation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-30-image2299-target-only-protected-null-distillation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: stage1_exact_stage2_future_positive_leakage
updated: 2026-08-30
---

# Image2299 target-only protected-null distillation

Results: [results.md](results.md).

## Final disposition

The staged target-only protected-null unit is complete and stage-exhausted.
It is a mechanically valid bounded negative of the staged schedule. Stage 1
is exact; Stage 2 satisfies its solver and full-vocabulary runtime checks but
fails the ordinary-greedy gate. Stages 3--5 were correctly not run.

## Question and frozen contrast

Can target-only output-row corrections preserve the frozen source surface while
releasing the next ordinary-greedy Image2299 owners, without changing
non-target vocabulary rows? The protected-null correction fixes every
non-target row and constrains each selected target against the other movable
targets and the strongest frozen competitor. The model, prompt, image, and
decode remain frozen; no training or checkpoint promotion is authorized.

## Evidence scope

The authoritative receipt is the v1 receipt named in `results.md`. Stage 1
has nine constraints, normalized norm `0.43629900998978716`, exact route
`7a39d314548b8f8a7cbae23727a110e836b0c449aa459a0fe3e31fe93d4147d0`, and a
passing gate. Stage 2 has 27 constraints, normalized norm
`0.7897506392158977`, full-vocabulary recheck passed, and retained the parent
plus `gt:2299:32` and `gt:2299:35`; its route has 352 tokens rather than the
expected 343 and one unmatched/unsupported person, so its gate fails.

## Interpretation and successor

The positive at position 342 was deliberately excluded from the protected set
until a future stage. A global nine-row residual can therefore activate its
row-open before its `x1` is constrained. The result supports a staged-gate /
global-residual mismatch, not target-only capacity exhaustion. At this unit's
stop, the one-shot all-108-constraint successor remained pending. It
later executed as [Global Target-Only Protected-Null Distillation](../2026-08-30-image2299-global-target-only-protected-null-distillation/results.md),
followed by the successful [dyadic norm release](../2026-08-30-image2299-dyadic-norm-release-distillation/results.md);
those successors do not alter this unit's own bounded negative.

The claim is limited to one frozen Image2299 augmented-model ordinary-greedy
result. It is not transfer, general enumeration learning, base-r32 greedy
evidence, or 8/8 tie recovery.
