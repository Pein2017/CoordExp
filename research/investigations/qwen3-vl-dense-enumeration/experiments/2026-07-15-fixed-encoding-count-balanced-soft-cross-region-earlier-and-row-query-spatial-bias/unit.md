---
title: Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query and Row-Scoring-Query Spatial Bias
description: One-image four-arm test of whether crossed semantic and geometric behavior survives a finite phase-dependent regional bias while every causally visible key remains readable.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Count-Balanced Soft Cross-Region Earlier-Query and Row-Scoring-Query Spatial Bias

## Question

On image `139`, does the phase-dependent crossed behavior observed under hard
spatial-key eligibility survive when every causally visible key remains
readable and the declared twenty-key region receives only a finite additive
attention-logit bias?

This unit is a discriminator between two explanations of the completed hard
cross-region result:

1. **Phase-Dependent Regional Routing**: earlier queries and row-scoring
   queries make distinct object-specific contributions, so crossing their
   softly favored regions can still split phrase and geometry ownership.
2. **Hard Phase-Switch Artifact**: the crossed behavior was manufactured by
   excluding all nonselected image keys and abruptly changing the only
   readable region at the row boundary. If this explanation is sufficient,
   the behavior should disappear when all visible keys remain readable.

The unit does not optimize a controller. It tests one dose, one image, and one
operator, then stops.

## Frozen Evidence Basis

The conclusion-owning hard-cross receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-cross-region-earlier-and-row-query-spatial-key-eligibility-hybrid/
  image139-float32-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
a50820cbc8dde6d92b964cfe9233e0e897da54e14d817dc48bd79787c8380bec
```

That hard panel found:

- Clock-Earlier and Vase-Row produced `clock` as the actual top first
  description token while aggregate geometry favored the vase row;
- Vase-Earlier and Clock-Row produced `person`, not `vase`, as the actual top
  first description token while aggregate geometry favored the clock row;
- both crossed arms nearly neutralized complete-row owner margin; and
- matched hard arms were strongly suppression-heavy.

The completed uniform soft spatial-key bias panel used doses up to `2.0` and
found graded regional actuation without a reliable complete owner switch. Its
receipt is historical context only:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/
  cohort-four-float32-20260715a/receipt.json
```

Its `SHA-256` digest is:

```text
468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a
```

The prior dose `2.0` is neither a required execution arm nor a required
comparator. This unit does not rerun it or use it to tune the new dose.

## Frozen Case and Execution Contract

Reuse exactly:

- image `139`;
- target annotation `1669970`, whose category is `vase`;
- competitor annotation `1666628`, whose category is `clock`;
- the hard-cross receipt's exact twenty-key vase support and exact twenty-key
  clock support;
- the frozen vase and clock canonical rows, prompt, prefix, input identifiers,
  explicit position identifiers, image grid, merge size, and full-image visual
  features; and
- the same source data, resolved configuration, audit ledger, base model, and
  adapter identities.

Execution uses Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter, Institute of
Electrical and Electronics Engineers 754 32-bit floating point (`float32`),
Scaled Dot Product Attention (`SDPA`), physical batch size one, no cache,
teacher-forced canonical-row scoring, raw model logits, and no repetition
penalty. The image is encoded once. Exact primary and DeepStack visual features
are replayed for every arm.

No crop, resize, pixel edit, arm-specific image encoding, candidate change,
support change, prompt change, prefix change, sampling, generation, training,
layer intervention, head intervention, architecture change, or additional
image is permitted.

## Count-Balanced Finite Bias

The frozen image-key set contains `1014` keys and each selected regional
support contains exactly `20` keys. Define the **full-support count-balanced
finite bias** as the single value

\[
\lambda_{\star}
=
\ln\left(\frac{1014-20}{20}\right)
=
\ln\left(\frac{994}{20}\right)
=
3.9060049331.
\]

This value is derived only from the frozen cardinalities. Under equal baseline
logits and when all `1014` image keys are causally visible, the selected
twenty-key set and the remaining `994` image keys have equal aggregate
unnormalized exponential weight:

\[
20\exp(\lambda_{\star})=994.
\]

This is a cardinality control, not a fitted dose and not a claim that actual
attention mass becomes one half. Baseline logits are not equal. Moreover,
early causal queries may not yet see all image keys, so the formula does not
guarantee query-local mass balance there. That limitation must remain visible
in the interpretation.

Let `P` be prefix length and `K` the length of the canonical row being scored.
Earlier queries are:

\[
q\in[0,P-2],
\]

and row-scoring queries are:

\[
q\in[P-1,P+K-2].
\]

Query `P+K-1` is excluded because its logit does not score a canonical-row
token. The two query ranges are disjoint and must be constructed separately
for the vase and clock rows because their token counts may differ.

For a selected region `M` and a query in that region's declared range, the
attention-mask value for a causally visible selected image key is increased by
exactly the one `float32` representation of `lambda_star`. For every arm:

- causal future keys remain negative infinity;
- causally visible nonselected image keys remain finite with additive value
  zero;
- causally visible non-image keys remain finite with additive value zero;
- selected image keys receive positive `lambda_star` only in their declared
  earlier-query or row-scoring-query range; and
- every off-range, nonselected, and non-image visible cell remains unchanged.

No causally visible key may be blocked. The operator changes only intended
finite bias cells; it does not convert finite cells to negative infinity.

## Exact Four-Arm Panel

Each arm is scored against both canonical rows.

| Earlier-query biased region | Row-scoring-query biased region | Complete arm name |
|---|---|---|
| vase | vase | Vase-Earlier and Vase-Row Matched Soft Arm |
| vase | clock | Vase-Earlier and Clock-Row Crossed Soft Arm |
| clock | vase | Clock-Earlier and Vase-Row Crossed Soft Arm |
| clock | clock | Clock-Earlier and Clock-Row Matched Soft Arm |

The first region name always owns the earlier-query bias range. The second
region name always owns the row-scoring-query bias range. No opaque arm code is
used.

Also score one unrestricted all-image-allowed four-dimensional no-operation
(`no-op`) arm for each canonical row. Here, no-operation means a
behavior-preserving control whose additive mask changes no causally visible
logit. It is a trust reference and the sole baseline for owner release. It is
not a fifth biased arm.

## Scores and Operational Definitions

For phase `p` and arm `a`, let

\[
\Gamma_{p,a}
=
S_{p,a}^{\mathrm{vase}}
-
S_{p,a}^{\mathrm{clock}},
\]

where each `S` is the mean raw selected-token log probability for that
canonical row and phase. Positive gamma favors the vase row; negative gamma
favors the clock row. Gamma is a pairwise score and does not identify the
actual top token when a third token wins.

For canonical owner `o`, define owner release only relative to unrestricted
scoring:

\[
R_{p,a}^{o}
=
S_{p,a}^{o}
-
S_{p,\mathrm{unrestricted}}^{o}.
\]

At the shared first-differing description position, both canonical-row scoring
paths have the same preceding token prefix. Record the actual top-token
identifier from both paths separately and require them to agree before one
semantic top token is declared. Compare exact token identifiers; decoded text
is a human-readable diagnostic only and cannot satisfy a gate.

For a crossed arm whose row-scoring region belongs to owner `o`, define the
**matched-row owner compatibility difference** as:

\[
C_{p,a}^{o}
=
S_{p,a}^{o}
-
S_{p,\mathrm{matched}(o,o)}^{o}.
\]

A mismatch penalty is present when this difference is at most `-0.05`
natural-log units per token. Absence of a material mismatch penalty requires
the difference to be strictly greater than `-0.05`. Equality at `-0.05`
belongs to the penalty side.

Record at least:

1. the first differing description token;
2. complete description;
3. left x coordinate (`x1`), top y coordinate (`y1`), right x coordinate
   (`x2`), and bottom y coordinate (`y2`);
4. complete geometry; and
5. complete row.

For every selected token, preserve the raw logit, log probability, rank, and
actual top token identifier and decoded token text. Preserve phase sums,
means, token counts, gamma, owner releases, and matched-row owner compatibility
differences without rounding in the receipt.

### Owner activation and suppression

For a phase whose pairwise gamma favors one canonical owner by an absolute
margin of at least `0.10`, define exactly two booleans:

```text
owner_activated := preferred-owner release >= +0.05
alternative_suppressed := alternative-owner release <= -0.05
```

Assign exactly one mutually exclusive mechanism label:

| `owner_activated` | `alternative_suppressed` | Mechanism label |
|---|---|---|
| true | false | `activation_only` |
| false | true | `suppression_only` |
| true | true | `mixed_activation_and_suppression` |
| false | false | `neither_activation_nor_suppression` |

Separately record `preferred_owner_non_destructive`, defined as preferred-owner
release at least `-0.05`. This boolean is not a fifth mechanism label.

If the actual top first description token is neither the vase canonical first
token nor the clock canonical first token, no canonical semantic owner may be
declared from pairwise gamma alone.

### Phrase-geometry chimera

A **phrase-geometry chimera** means that the actual top first description
token belongs to one canonical object while complete-geometry gamma favors the
other canonical row by an absolute margin of at least `0.10`.

Therefore:

- Clock-Earlier and Vase-Row is a clock-phrase and vase-geometry chimera only
  if the actual top first description token is the clock canonical first token
  and complete-geometry gamma is at least `+0.10`;
- Vase-Earlier and Clock-Row is a vase-phrase and clock-geometry chimera only
  if the actual top first description token is the vase canonical first token
  and complete-geometry gamma is at most `-0.10`.

Pairwise first-token gamma without the required actual top token is never a
chimera. Each chimera must additionally carry exactly one of the four exclusive
mechanism labels plus the separate `preferred_owner_non_destructive` boolean.
Deterministic tests must cover all four boolean combinations and both threshold
equalities.

## Matched-Control Adjudication Gate

Interpret crossed arms only after both matched soft controls pass all of the
following requirements:

1. Vase-Earlier and Vase-Row must have the vase canonical first description
   token identifier as the actual top first description token in both
   canonical-row scoring paths, and those two observed top-token identifiers
   must agree exactly.
2. The vase canonical first-description-token owner release must be at least
   `-0.05` relative to unrestricted scoring.
3. Vase-Earlier and Vase-Row complete-geometry gamma must be at least `+0.10`,
   and vase complete-geometry owner release must be at least `-0.05`.
4. Clock-Earlier and Clock-Row must have the clock canonical first description
   token identifier as the actual top first description token in both
   canonical-row scoring paths, and those two observed top-token identifiers
   must agree exactly.
5. The clock canonical first-description-token owner release must be at least
   `-0.05` relative to unrestricted scoring.
6. Clock-Earlier and Clock-Row complete-geometry gamma must be at most `-0.10`,
   and clock complete-geometry owner release must be at least `-0.05`.

If either matched control fails any requirement, the only scientific
classification is:

```text
no_adjudication_close_count_balanced_soft_cross_operator
```

This complete label means: the count-balanced finite soft operator failed to
reproduce both matched owner controls, so the crossed arms cannot adjudicate
phase routing and the operator is closed. Record all raw observations, stop,
and do not interpret crossed behavior.

The classifier must have deterministic tests in which the correct canonical
token remains top but its owner release is below `-0.05`; the matched-control
gate must fail. It must also fail when the two canonical-row scoring paths do
not agree on the top-token identifier.

## Crossed-Arm Decision Signatures

These signatures apply only after the matched-control adjudication gate passes.

For crossed arm `a`, phase `p`, and the canonical owner `o` named by that arm's
row-scoring region, define:

\[
\operatorname{penalty}(a,p)
\iff
C_{p,a}^{o}\le -0.05.
\]

Define an arm-level geometry-or-row penalty as:

\[
\operatorname{penalty\_either}(a)
\iff
\operatorname{penalty}(a,\mathrm{geometry})
\lor
\operatorname{penalty}(a,\mathrm{full\_row}).
\]

Absence of a material geometry-or-row penalty requires both compatibility
differences to be strictly greater than `-0.05`.

### Bidirectional soft phase split

Classify a bidirectional soft phase split only if:

- Clock-Earlier and Vase-Row is a clock-phrase and vase-geometry chimera; and
- Vase-Earlier and Clock-Row is a vase-phrase and clock-geometry chimera.

This supports phase-dependent regional routing under an all-keys-readable
operator. It does not establish native object binding.

### Asymmetric hard-cross phenotype persists

Classify bounded persistence of the asymmetric hard-cross phenotype only if:

- Clock-Earlier and Vase-Row remains a clock-phrase and vase-geometry chimera;
- Vase-Earlier and Clock-Row retains clock-favoring geometry with gamma at
  most `-0.10` but its actual top first description token is not the vase
  canonical first token; and
- `penalty_either` is true for at least one crossed arm.

This preserves the directional hard-cross pattern without claiming symmetric
semantic handoff.

### Hard mismatch penalty persists without a phase split

Classify a mismatch-sensitive compatibility effect without a phrase-geometry
phase split when:

- neither crossed arm is a phrase-geometry chimera;
- both matched controls passed; and
- `penalty_either` is true separately for both crossed arms.

This supports mismatch sensitivity under finite bias, but not a temporal
owner transfer.

### Crossed behavior disappears into row-region control

Classify disappearance into row-region control only if:

- Clock-Earlier and Vase-Row has the vase canonical first description token as
  the actual top token and complete-geometry gamma at least `+0.10`;
- Vase-Earlier and Clock-Row has the clock canonical first description token
  as the actual top token and complete-geometry gamma at most `-0.10`; and
- neither crossed arm has a material matched-row owner penalty in complete
  geometry or complete row.

This weakens the hard phase-split explanation and supports direct row-region
control under the finite all-keys-readable operator.

### Crossed behavior follows the earlier region

Classify earlier-region control only if Clock-Earlier and Vase-Row has the
clock canonical first token as its actual top token and clock-favoring
geometry, while Vase-Earlier and Clock-Row has the vase canonical first token
as its actual top token and vase-favoring geometry, each with absolute geometry
gamma at least `0.10`, and neither crossed arm has a material matched-row owner
penalty in complete geometry or complete row.

Any gate-passing result that satisfies none of these exact signatures is
classified as mixed. Report raw owner activation, suppression, actual top
tokens, and mismatch penalties without inventing an additional mechanism.

The executable classifier must apply this strict precedence; the first
satisfied label owns the result and no later label may also be emitted:

1. bidirectional soft phase split;
2. asymmetric hard-cross phenotype persists;
3. crossed behavior disappears into row-region control;
4. crossed behavior follows the earlier region;
5. hard mismatch penalty persists without a phase split; and
6. mixed.

Truth-table tests must cover every label, every threshold boundary, and cases
that would satisfy more than one structural predicate without precedence.

## Execution Trust Gate

Before the matched controls or crossed arms can be interpreted, require all of
the following:

1. Validate the hard-cross receipt path and exact `SHA-256` digest before model
   loading.
2. Match source data, resolved configuration, audit ledger, image, annotation,
   canonical-row token arrays, regional key arrays, image grid, merge size,
   input identifiers, explicit position identifiers, base model, adapter, and
   fixed visual-feature fingerprints to the frozen hard-cross contract.
3. Require exactly `1014` image keys and exactly `20` unique selected keys in
   each frozen support.
4. Reproduce the frozen unrestricted selected-token log probabilities within
   `1e-4` and reproduce every selected-token rank exactly for both canonical
   rows.
5. Require implicit-position, explicit-position, and all-image-allowed no-op
   paths to agree within `1e-4` with exact selected-token ranks.
6. Construct every arm from an independently inspectable arm plan that binds
   its earlier support, row support, earlier query range, row-scoring query
   range, and canonical-row length from one source of truth.
7. Prove the earlier and row-scoring biased-cell sets are disjoint and their
   union is exactly the intended changed-cell set.
8. Prove that every intended selected-and-visible cell changes by exactly the
   single `float32(lambda_star)` mask value, and that no other causally visible
   cell changes.
9. Prove that every causally visible nonselected image key and every causally
   visible non-image key remains finite and has zero additive change. No
   causally visible key may be blocked.
10. Prove that every causal future cell remains negative infinity and exactly
    matches the unrestricted causal future mask.
11. Prove zero changes to off-range queries and prohibit any positive bias on
    the final unscored query.
12. Record raw logits, log probabilities, ranks, actual top token identifiers,
    decoded top-token text, phase aggregates, gamma, owner releases, and
    matched-row owner compatibility differences for all four arms and both
    canonical rows.
13. Bind the new runner digest, every imported scorer digest, normalized
    argument vector, resolved configuration path and digest, source and ledger
    identities, hard-parent receipt digest, base model identity, adapter
    identity, execution precision, and library versions in the immutable
    receipt.
14. Bind effective runtime attestation showing that the executed attention
    implementation is exactly Scaled Dot Product Attention (`sdpa`), model
    parameters and scoring logits use `torch.float32`, `use_cache` is false,
    scoring calls model forward directly without a logits processor or
    repetition penalty, and the intervention mask is passed to the executed
    model forward for every biased and no-operation arm.
15. For every executed arm and canonical row, record the observed intervention
    mask shape, data type, content digest, intended finite changed-cell count,
    and observed finite changed-cell count. Bind these observed masks to the
    same masks whose structural gates pass; a separately checked but unconsumed
    mask construction is insufficient.

Any failed trust gate invalidates the execution. A zero behavioral effect after
all gates pass is valid negative evidence. A matched-control scientific failure
does not invalidate execution; it triggers the frozen no-adjudication close
label.

The one-image real execution is also the required live runtime-consumption
smoke. Source inspection and unit tests alone cannot attest mask consumption or
authorize interpretation.

## Minimal Route and Stop Rule

Create at most one experiment-local runner and one focused test file only after
separate implementation authorization. Reuse the frozen hard-cross receipt,
feature replay, canonical-row scorer, exact query ranges, arm-plan mapping,
phase aggregation, and raw-token reporting. Add only the finite crossed-bias
operator, exact structural receipts, matched-control gate, and frozen
classifier required above.

Run exactly one `float32` execution for image `139` on one graphics processing
unit. Produce one immutable receipt, obtain one independent artifact audit and
one independent scientific audit, then stop.

Logical artifact root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-count-balanced-soft-cross-region-earlier-and-row-query-spatial-bias/
  image139-float32-20260715a/
```

Do not add another dose, image, region, object pair, layer, head, rollout,
training screen, architecture module, free generation, or test-time controller
inside this unit. Do not promote any result to a training or architecture
claim.

## Confounds and Falsification Boundaries

- The finite bias reweights keys but does not directly prove that attention
  weights explain downstream logits.
- The twenty-key windows are object-centered, not object-pure.
- Fixed visual features are globally contextualized before the intervention.
- The cardinality-derived dose is only nominally count-balanced under equal
  logits and full image-key visibility.
- Actual top-token checks prevent a third-token winner from being mislabeled as
  canonical semantic ownership, but one first token cannot prove whole-phrase
  identity.
- Failure of the matched controls closes this operator; it does not falsify the
  hard-cross observation.
- Survival of a crossed signature under finite bias weakens the hard-exclusion
  artifact explanation but does not eliminate every mask-transition artifact.
- Disappearance under finite bias shows operator dependence, not that earlier
  computation is generally irrelevant.

## Not Claimed

This unit cannot establish native object binding, an object pointer, causal
attention interpretation, population prevalence, free-rollout utility,
commit, coverage, stopping, Average Precision, recall gain, a trainable loss,
or a final architecture. It does not authorize training or any stable code,
configuration, schema, or runtime contract.

## Closure

Execution and bounded interpretation are complete. The verified receipt passed
the execution trust gate, but both matched soft controls failed the
prespecified adjudication gate. The unit therefore closes with classification
`no_adjudication_close_count_balanced_soft_cross_operator`; crossed arms remain
uninterpreted, and no architecture or training claim is promoted.

See [results.md](results.md) for the evidence boundary, matched-control
observations, bounded verdict, rejected cross-operator dose synthesis, and stop
decision.
