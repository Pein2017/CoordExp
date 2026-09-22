---
title: Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility Crossover
description: Four-anchor causal discriminator testing whether response-row query restriction is sufficient or whether earlier-query computation is required.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-row-scoring-query-only-spatial-key-eligibility-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Row-Scoring-Query-Only Spatial-Key Eligibility Crossover

## Closure

Execution and bounded interpretation are complete. The conclusion-owning
evidence, automatic frozen verdict, limitations, and next discriminator are in
[results.md](results.md). No architecture or training work is authorized.

## Question

Does the accepted hard spatial-eligibility phenotype survive when regional
image-key restriction is applied only to the causal query positions that
predict the current canonical row, while every image-token query and every
unscored prefix query retains unrestricted full-image key access?

This unit separates two levels of explanation left confounded by both
completed all-query interventions:

1. **Direct Row-Read Competition**: the response-row text queries need a
   restricted regional read surface, while the already compiled full-image
   representation is adequate.
2. **Earlier-Query Dependence**: hard eligibility requires changing some
   computation before the response-row scoring queries. Image-token state
   recompilation is one possibility, but earlier prefix-text queries and other
   all-query trajectory effects remain alternatives.

Success establishes the sufficiency of row-scoring-query restriction; it does
not show that earlier computation is irrelevant. Failure establishes that
response-row restriction alone is insufficient; it does not uniquely localize
the required earlier computation to image-token states.

It is a new bounded causal discriminator, not a stronger controller search.
It does not add bias doses, layers, training, generation, or new candidates.

## Evidence Basis

The completed [all-query hard-eligibility
panel](../2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md)
found one complete semantic-plus-geometry switch, one geometry-specific case,
and three one-sided or destructive cases.

The completed [uniform positive soft-bias
panel](../2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/results.md)
found strong dose-dependent regional actuation but zero complete-row or
geometry owner reversals. Both operators changed selected image-key columns
for every causal query, including image-token queries. The remaining question
is therefore intervention locus, not bias magnitude.

Two independent read-only scientific reviews ranked this query-scope
discriminator above balanced target-plus/competitor-minus bias. Balanced bias
would add explicit anti-competitor force while retaining the same all-query
confound; it could demonstrate steering strength without identifying the
native bottleneck.

## Frozen Anchors

Reuse the exact target key set, competitor key set, all-query hard endpoint,
and feature contract from the conclusion-owning all-query hard receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/
  cohort-six-float32-20260715b/receipt.json
```

The four frozen mechanism anchors are:

| Image | Target to competitor | Parent phenotype | Purpose |
|---:|---|---|---|
| `139` | vase to clock | complete semantic-plus-geometry switch | Required positive-control phenotype. |
| `632` | book to book | differential destruction | Negative control for target-independent suppression. |
| `12120` | person to tennis racket | geometry and complete-row switch without semantic switch | Phase-separation anchor. |
| `12639` | person to person | one-sided near-rescue | Same-category spatial-address anchor. |

No image, candidate, row, window, halo, annotation, or dose may be reselected.

The parent artifact is immutable for this unit. Its required Secure Hash
Algorithm 256-bit (`SHA-256`) digest is:

```text
6148fa75998385eeab0ac0118d3fe540acaa6c7dfece85ee6116907ed3a703ee
```

The runner must fail before loading rows, regions, or endpoints if this digest
does not match.

The hard receipt predates canonical row-token persistence. Exact target and
competitor row token arrays therefore come from the accepted soft-bias receipt,
which used the same frozen annotations and stores those arrays explicitly:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response/
  cohort-four-float32-20260715a/receipt.json
```

Its required `SHA-256` digest is:

```text
468af3e49962b4e78c69ecac0c004e774c38da93e47e4bd5e19c05814c12580a
```

Before model scoring, the runner must verify that this row-contract receipt
points to the frozen hard-receipt digest, matches source, config, ledger, image,
annotation, and mask identities, and that freshly derived target and competitor
row token arrays exactly equal its stored arrays. Missing or mismatched fields
are launch failures; the old artifact is not mutated.

## Fixed Execution Contract

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: full Institute of Electrical and Electronics Engineers 754
  32-bit floating point (`float32`), physical batch size one, and Scaled Dot
  Product Attention (`SDPA`).
- Prefix: the same fixed base/reset detection prompt used by both parent units.
- Scoring: teacher-forced, no cache, and no repetition penalty.
- Visual path: one full-image encoding per case with exact primary and
  DeepStack feature replay for every arm.
- Position path: explicit Multimodal Rotary Position Embedding (`M-RoPE`)
  identifiers derived from the ordinary two-dimensional mask.
- Prohibited changes: crop, resize, pixel editing, arm-specific visual
  re-encoding, candidate change, region change, generation, training, layer or
  head sweep, phase controller, and architecture modification.

## Exact Query Scope

Let `P` be the prefix token count and `K` the canonical row token count. The
only query indices receiving regional hard image-key eligibility are:

\[
q\in[P-1,\ P+K-2].
\]

These are exactly the hidden-state positions whose logits predict the row-
entry token through the final row token under the one-token causal shift.

For every other query index:

- all causally visible image keys remain eligible;
- all non-image keys remain unchanged; and
- normal causal future-key blocking remains unchanged.

The target and competitor rows may have different token counts. Their query
ranges must be constructed independently from their own `K` values.

This operator preserves image-token-to-image-token language-decoder
propagation and all unscored prefix computation. It restricts only the current
row's direct reads from image-key columns.

## Arms

For each target and competitor canonical row, execute:

1. **Implicit-Position Standard Two-Dimensional Attention Mask**;
2. **Explicit-Position Standard Two-Dimensional Attention Mask**;
3. **All-Image-Allowed Four-Dimensional Attention Mask**;
4. **Target-Region Row-Scoring-Query-Only Hard Eligibility**; and
5. **Competitor-Region Row-Scoring-Query-Only Hard Eligibility**.

Copy the corresponding all-query hard endpoint from the parent receipt for
comparison. Do not rerun it unless prerequisite continuity fails.

## Trust Gate

Before interpreting one case:

1. source JSON Lines (`JSONL`), audit ledger, resolved config, parent receipt,
   canonical rows, regional indices, and feature fingerprint must match the
   frozen parent contract exactly;
2. the explicit-position ordinary path and all-image-allowed four-dimensional
   path must preserve selected-token ranks and keep maximum absolute selected-
   token log-probability drift at or below `1e-4` relative to the implicit
   ordinary path;
3. zero-arm selected-token log probabilities must match the parent
   all-image-allowed arm within `1e-4`; and
4. every target-region and competitor-region arm must emit a structural mask-
   execution receipt for each scored row. It must prove a nonempty query range,
   at least one blocked image-key cell, no changed non-image-key cell, no
   changed off-scope query row, and unchanged causal future-key blocking.

The image-`139` real smoke must execute and validate both regional arms. A zero
behavioral effect is a valid negative result after the structural receipt and
no-op gates pass; behavioral movement is not an execution gate. If image `139`
fails a structural or no-op gate, stop. A later failing case is excluded
without threshold relaxation; valid cases remain interpretable.

## Phase Scores and Comparisons

Reuse the parent scores:

1. row-entry-versus-terminal log odds;
2. first differing description token, when defined;
3. complete description span;
4. `x1` (left x coordinate), `y1` (top y coordinate), `x2` (right x
   coordinate), and `y2` (bottom y coordinate);
5. complete geometry span; and
6. complete row.

For phase `p`, preserve target-region gamma, competitor-region gamma,
crossover, target-row release, and competitor-row release. A complete owner
switch requires target-region gamma at or above the case effect floor and
competitor-region gamma at or below the negative case effect floor for complete
description, geometry, and complete row when the description contrast is
defined.

Define the case effect floor as:

\[
\operatorname{effect\_floor}
=
\max(10\times\operatorname{no\_op\_drift},\ 0.01).
\]

Every interpreted crossover must additionally be at least `0.10` natural-log
units per token and exceed the case effect floor.

For parent phases whose all-query hard crossover exceeds `0.10` natural-log
units per token, additionally report:

\[
\operatorname{recovery}_p
=
\frac{
  \Delta^{\text{row-scoring-query-only hard}}_p
}{
  \Delta^{\text{all-query hard}}_p
}.
\]

Always report the raw numerator and denominator. Do not interpret a ratio when
the parent denominator is at or below `0.10`.

For an owner row, query-scoped hard eligibility is considered no more
destructive than the parent all-query endpoint when its mean owner-row log
likelihood is at least the parent value minus `0.05` natural-log units per
token. The two owner rows are exactly the target row under target-region
eligibility and the competitor row under competitor-region eligibility.

The frozen secondary-anchor phenotype predicates are:

- image `12120` preserves or strengthens its parent phenotype when geometry
  and complete-row gamma satisfy the owner-reversal effect floors under both
  regional arms; description may remain unswitched or may strengthen to a
  complete semantic reversal;
- image `12639` preserves its parent near-rescue when complete-row and geometry
  crossover each recover at least half their parent value, target-region gamma
  is no more than `0.05` below its parent gamma, competitor-region gamma is at
  or below the negative effect floor, and target-row release is positive.

A **retained case** is image `139` after its complete predicate passes or one of
these secondary anchors after its exact predicate passes, with both owner rows
also satisfying the non-destruction rule.

## Frozen Decision Rule

### Support a direct row-read contribution

Require all of the following:

1. image `139` preserves complete description, geometry, and complete-row
   owner reversal;
2. at least one of images `12120` or `12639` satisfies its frozen phenotype
   predicate;
3. every retained case recovers at least half of its parent full-row and
   geometry crossover; and
4. its target and competitor owner rows are not more destructive than the
   corresponding all-query hard endpoint.

This supports a compact direct row-read routing seam but does not authorize
training or architecture promotion.

### Promote one bounded free-row replay

Promotion additionally requires at least two independent complete owner
switches, including one different-category case with semantic reversal and
target-row release of at least `+0.05` natural-log units per token. Promote only
the strongest one case to one no-cache free-row replay. This still does not
authorize training.

### Support dependence on earlier pre-row query computation

If image `139` loses its complete switch and neither image `12120` nor image
`12639` satisfies its frozen predicate after all structural and no-op gates
pass, conclude only that response-row query restriction is insufficient and
that some earlier query restriction is required for the parent hard phenotype.
Close direct response-row read restriction and do not add balanced bias. Do
not choose among image-token recompilation, earlier prefix-text computation,
or another all-query trajectory effect from this negative alone.

### Mixed or isolated result

Any other result is mixed. Record it, close uniform spatial-key eligibility as
a general owner-selection architecture candidate, and do not add balanced
bias, more doses, a layer sweep, or a larger cohort.

The conclusion-owning cohort receipt must contain exactly the four frozen image
identifiers `139`, `632`, `12120`, and `12639`. The earlier-query-dependence
verdict additionally requires images `139`, `12120`, and `12639` to pass all
structural and no-op gates. A missing, absent-source, or invalid decision-
critical anchor yields an incomplete or mixed result, never mechanistic
negative evidence.

## Falsification and Confounds

- A no-op-parity or structural mask-execution failure invalidates the case
  rather than supporting a mechanism. Zero behavioral movement after those
  gates pass is valid negative evidence.
- A target-versus-competitor margin can grow through non-owner destruction
  without recovering the owner row.
- The regions are object-centered, not object-pure; a positive result remains
  region-conditioned routing rather than an instance pointer.
- The visual features were globally contextualized before every intervention.
- Teacher-forced row scoring does not establish free-rollout utility,
  autonomous selection, commit, coverage, or termination.
- Four phenotype-selected anchors cannot estimate population prevalence.

## Minimal Implementation Route

Create one experiment-local runner and focused test file. Reuse parent model
loading, receipt validation, feature capture and replay, explicit positions,
canonical row scoring, phase metrics, hard endpoints, and classifier helpers.

Add only:

- a boolean four-dimensional causal mask constructor that restricts
  nonselected image-key columns on the exact row-scoring query indices;
- query-range and non-image-key invariance tests; and
- the frozen panel classifier above.

Run image `139` as a real smoke. Run images `632`, `12120`, and `12639` only if
the smoke passes. Stop after one four-anchor receipt and one independent
scientific audit.

## Not Claimed

This unit cannot establish a pure object pointer, native causal necessity,
population prevalence, Average Precision or recall gain, commit, coverage,
termination, a trainable mechanism, or a final architecture.
