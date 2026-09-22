---
title: Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility Crossover Results
description: Four-anchor evidence that direct row reads can switch geometry and same-category spatial ownership, but do not reproduce the image-139 cross-category semantic-plus-geometry phenotype without earlier-query intervention.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility Crossover Results

## Verdict

Accept the executed evidence and close the unit with the frozen automatic
classification:

```text
support_dependence_on_earlier_query_computation
```

Interpret that classification narrowly. Restricting regional image-key access
only at the response-row scoring queries is a potent causal intervention. It
is sufficient for geometry and complete-row spatial-owner switching in some
cases, including the same-category image-`632` book-to-book anchor. It is not
sufficient to reproduce the parent image-`139` cross-category vase-to-clock
semantic-plus-geometry switch. Some computation at earlier query positions is
therefore required for that parent phenotype.

This result does not show that direct row reads are unimportant, and it does
not localize the missing earlier computation to image-token recompilation.
Earlier prefix-text queries and nonlinear interaction between earlier state
and the row read remain live explanations. The smallest next discriminator is
one image-`139` earlier-query-only arm completing a baseline, row-only,
earlier-only, and all-query two-by-two factorial.

Two independent read-only scientific reviews agreed that the automatic verdict
is contract-valid but must be bounded this way. Both rejected the stronger
interpretation that the result identifies one unique earlier-state mechanism
or makes direct row reads causally irrelevant.

## Evidence Boundary

The conclusion-owning four-anchor receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/
  cohort-four-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
4aba45f16a90d032191a51f1c9c35ad9b71ae4d0ec97f943d9e0ddf217055fa4
```

The accepted image-`139` mechanics smoke is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover/
  smoke-image139-float32-20260715a/receipt.json
```

Its `SHA-256` digest is:

```text
b0b2f8704b15ec2dfc5497239aca59ced4c6c09dd87f3317e5be7ae325c8bfad
```

The cohort receipt contains exactly images `139`, `632`, `12120`, and `12639`.
It binds the accepted all-query hard parent receipt with digest
`6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee`
and the canonical-row soft-bias receipt with digest
`468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a`.
The resolved configuration, source JavaScript Object Notation Lines (`JSONL`),
and audit-ledger digests are respectively:

```text
80aa7fdca59dae988d4dfbe55277bd2c39f11fd9990017a17815eff6b11f9381
9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4
52e9f21eb32f7c3793d1356125931cb4c2a1fa5a12647d0d437dec217668d8df
```

## Executed Contract and Trust Gates

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: Institute of Electrical and Electronics Engineers 754 32-bit
  floating point (`float32`), physical batch size one, and Scaled Dot Product
  Attention (`SDPA`).
- Scoring: teacher-forced, no cache, and no repetition penalty.
- Visual state: one fixed full-image encoding per case with exact primary and
  DeepStack feature replay across arms.
- Changed factor: regional hard image-key eligibility only at hidden-state
  query positions whose logits predict the canonical response row. Every
  image-token query and every unscored prefix query retained unrestricted
  image-key access.
- Parent comparison: the accepted all-query hard endpoint was copied from the
  bound parent receipt; it was not reselected or silently rerun.

All four cases passed row identity, parent continuity, feature continuity,
ordinary-path parity, and structural mask gates. Maximum absolute no-op drift
was `5.2452e-5`, `4.9591e-5`, `5.1975e-5`, and `2.3842e-5` for images `139`,
`632`, `12120`, and `12639`, respectively, below the frozen `1e-4` ceiling.
Every regional arm had a nonempty scoring-query range, blocked image-key cells,
zero changed non-image-key cells, zero changed off-scope query cells, and
unchanged causal future-key blocking.

The one-image smoke receipt is mechanically accepted even though its panel
classifier reports missing anchors: that classifier is expected to be
incomplete for a one-case smoke and does not own the four-anchor conclusion.

## Observed

For each phase, `gamma target` is target-row minus competitor-row mean log
likelihood under target-region eligibility, and `gamma competitor` is the same
difference under competitor-region eligibility. `Crossover` is `gamma target`
minus `gamma competitor`. Values are natural-log units per token. `Recovery`
is query-only crossover divided by the all-query parent crossover and is shown
only where the parent denominator exceeds `0.10`.

| Image | Target to competitor | Phase | Gamma target | Gamma competitor | Crossover | Parent crossover | Recovery |
|---:|---|---|---:|---:|---:|---:|---:|
| `139` | vase to clock | description | `-3.486` | `-2.391` | `-1.095` | `+12.655` | `-0.087` |
| `139` | vase to clock | geometry | `+0.564` | `-4.292` | `+4.856` | `+1.961` | `2.476` |
| `139` | vase to clock | complete row | `-0.078` | `-2.970` | `+2.892` | `+2.826` | `1.023` |
| `632` | book to book | geometry | `+1.630` | `-2.822` | `+4.451` | `+1.815` | `2.452` |
| `632` | book to book | complete row | `+1.086` | `-1.881` | `+2.967` | `+1.210` | `2.452` |
| `12120` | person to tennis racket | description | `+1.548` | `+0.634` | `+0.914` | `+0.061` | not interpreted |
| `12120` | person to tennis racket | geometry | `+2.126` | `-2.030` | `+4.156` | `+2.801` | `1.484` |
| `12120` | person to tennis racket | complete row | `+1.284` | `-1.286` | `+2.570` | `+1.714` | `1.499` |
| `12639` | person to person | geometry | `-0.143` | `-4.174` | `+4.031` | `+2.279` | `1.769` |
| `12639` | person to person | complete row | `-0.095` | `-2.783` | `+2.687` | `+1.519` | `1.769` |

Same-description anchors have no description-owner contrast, so their
description crossover is exactly zero and is not treated as a semantic test.

Owner-row non-destruction relative to the all-query hard endpoint was:

| Image | Target owner-row delta | Target passed | Competitor owner-row delta | Competitor passed |
|---:|---:|---|---:|---|
| `139` | `-0.355` | no | `+1.044` | yes |
| `632` | `+0.785` | yes | `+0.217` | yes |
| `12120` | `-0.119` | no | `+0.996` | yes |
| `12639` | `+0.009` | yes | `+1.173` | yes |

Image `139` retained a strong geometry reversal and a large complete-row
crossover, but lost the semantic reversal: target-region description gamma
changed from the parent `+8.226` to `-3.486`, and target full-row gamma changed
from `+1.071` to `-0.078`. Its target-row release was `-0.297`, compared with
the parent `+0.057`, and its target owner row failed the non-destruction gate.

Image `632` changed from a one-sided all-query hard endpoint to a clean
same-category geometry and complete-row owner reversal under direct row reads.
Its complete-row target release was `+0.321`. Image `12120` also reversed
geometry and complete-row ownership, but missed the target owner-row
non-destruction tolerance by `0.069` natural-log units per token beyond the
allowed `-0.05`. Image `12639` exceeded half-parent crossover recovery and had
positive target release `+0.541`, but both target-region geometry and
complete-row gammas remained negative, so its frozen near-rescue predicate
failed.

The receipt's automatic panel fields are: zero complete-switch cases, zero
retained cases, zero secondary-predicate cases, no invalid cases, and
`direct_read_gate_passed: false`.

## Supported

- Direct response-row reads are a potent spatial-routing locus. They can
  produce geometry and complete-row owner reversals without modifying
  image-token queries or unscored prefix queries.
- Direct row reads are sufficient for same-category spatial owner switching in
  image `632`; they also produce a strong but formally non-retained geometry
  and complete-row reversal in image `12120`.
- The cross-category image-`139` parent phenotype depends on some restriction
  at earlier query positions in addition to, or interacting with, the direct
  row read. The evidence supports this dependency claim, not a unique internal
  localization.
- The all-query and row-only effects are not ordered by a simple scalar
  strength. Row-only improves several geometry crossovers while removing the
  image-`139` semantic switch, which keeps an earlier-state-by-row-read
  nonlinear interaction live.

## Ruled Out

- A response-row-query-only restriction is not sufficient to reproduce the
  complete image-`139` semantic-plus-geometry parent phenotype.
- The frozen direct-row-read promotion gate does not pass: image `139` is not a
  complete switch, and neither image `12120` nor image `12639` satisfies its
  full secondary predicate.
- No balanced target-plus/competitor-minus bias, additional dose, layer sweep,
  larger cohort, or free-row replay follows automatically from this unit.

## Unresolved

- Whether the required earlier computation is image-token-to-image-token
  recompilation, earlier prefix-text query computation, another all-query
  trajectory effect, or an interaction among them.
- Why row-only restriction strengthens same-category and geometry routing while
  losing image-`139` semantic ownership.
- Whether the image-`632` same-category switch generalizes beyond the selected
  anchor. This four-case mechanism panel does not estimate prevalence.
- Free-generation behavior, autonomous object selection, commit, coverage,
  termination, and downstream detection metrics.

## Not Claimed

This unit does not establish a pure object pointer, a universal direct-read
mechanism, the causal irrelevance of direct row reads, native necessity,
population prevalence, Average Precision or recall gain, a trainable loss, or
a final architecture. It authorizes no architecture, training, production
code, configuration, or stable-contract change.

## Receipt Limitations

The following limitations are real but do not change the bounded verdict:

- The receipt does not bind the executed runner file or command-line argument
  vector. Artifact hashes, frozen parent identities, row identities, feature
  continuity, structural mask receipts, and numeric outputs are present, but
  the exact invocation is not recoverable from the receipt alone.
- Per-phase recovery is derivable from the stored raw query-only and parent
  crossover values, but the receipt emits recovery fields only for selected
  frozen predicates rather than every phase.
- The frozen threshold boundary's exact equality convention is not material to
  any observed result; all decision-bearing values are away from the relevant
  boundary.

These are receipt-hardening opportunities for a future execution, not grounds
to invalidate or broaden this unit.

## Next Discriminator

Run one fixed image-`139` earlier-query-only arm and combine it with the already
available baseline, row-only, and all-query conditions to form a two-by-two
factorial:

| Earlier queries restricted | Row-scoring queries restricted | Condition |
|---|---|---|
| no | no | baseline |
| no | yes | row-only |
| yes | no | earlier-only |
| yes | yes | all-query |

Keep the checkpoint, canonical vase and clock rows, regional key sets, fixed
visual encoding, explicit position identifiers, scoring semantics, and
`float32` execution invariant. The primary discriminator is whether the
earlier-only condition restores semantic ownership and whether the all-query
effect is additive or interactive relative to the row-only condition. This is
an experiment seed only; it is not implementation or Graphics Processing Unit
(`GPU`) launch authorization.
