---
title: Fixed-Encoding Soft Spatial-Key Bias Dose Response
description: Four-anchor causal panel testing whether finite post-vision attention-logit bias can route an object-centered region while preserving globally useful visual context.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-soft-spatial-key-bias-dose-response
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Soft Spatial-Key Bias Dose Response

Executed evidence and the bounded verdict are recorded in
[results.md](results.md).

## Question

With every full-image visual key still available, can a finite additive bias on
one object-centered set of already-computed image-token keys improve its owner
row without destroying the globally useful context removed by hard key
eligibility?

This is a new bounded discriminator motivated by the completed
[Fixed-Encoding Object-Centered Spatial-Eligibility Crossover](../2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/results.md).
It is not an unregistered extension of that panel and does not reinterpret its
inconclusive verdict.

## Mechanism and Competing Explanations

Let (A) and (B) be the frozen target and competitor rows, and let (M_A)
and (M_B) be their frozen equal-cardinality regional key sets. For every
causally visible key (k), add a finite bias to the attention logit:

\[
L'_{qk}
=
L_{qk}+\lambda\,\mathbf{1}[k\in M],
\]

while preserving all nonlocal keys, causal order, visual features, text,
positions, weights, and execution precision.

The hypotheses are:

1. **Soft Competition Reweighting**: unrestricted visual evidence contains the
   owner row, but its probability is diluted. A finite shared bias increases
   owner likelihood without the context loss of hard exclusion.
2. **Hard-Restriction-Only Contrast**: the preceding crossover arose mainly by
   suppressing non-owner evidence. Finite bias does not create a recurrent
   context-preserving owner switch.
3. **Global-Context Dependence**: even a moderate local bias degrades the owner
   or preserves the same one-sided behavior because the row needs distributed
   evidence.
4. **Generic Continuation or Salience**: local bias changes row-entry-versus-
   terminal margin or both candidate rows similarly without choosing the owner.
5. **Phase-Specific Routing**: description and geometry react differently,
   indicating that one uniform all-phase operator is insufficient.

## Frozen Mechanism Anchors

The four cases are selected from the completed hard-eligibility panel for
different predefined phenotypes. They are mechanism anchors, not a prevalence
sample:

| Image | Target to competitor | Parent phenotype | Purpose |
|---:|---|---|---|
| `139` | vase to clock | complete regional switch with weak target release | Test whether finite bias preserves the switch while reducing competitor destruction. |
| `632` | book to book | differential destruction | Negative-control case for global-context dependence. |
| `12120` | person to tennis racket | geometry switch without description switch | Test whether finite bias changes the phase split. |
| `12639` | person to person | strong one-sided near-rescue | Test whether a context-preserving dose crosses the instance-geometry boundary. |

Image `9400` remains excluded because its parent no-op gate failed. Image
`2299` is omitted because it duplicates the differential-destruction phenotype
already represented by image `632`.

The exact target identifier, competitor identifier, target key indices,
competitor key indices, source hash, ledger hash, config hash, and feature-grid
contract come from the parent conclusion-owning receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/
  cohort-six-float32-20260715b/receipt.json
```

No candidate or mask is reselected after observing this unit.

## Fixed Execution Contract

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Execution: full Institute of Electrical and Electronics Engineers 754
  32-bit floating point (`float32`), physical batch size one, Scaled Dot
  Product Attention (`SDPA`).
- Prefix: the same base/reset detection prompt used by the parent unit.
- Scoring: teacher-forced, no cache, and no repetition penalty.
- Visual path: one full-image encoding per case; exact primary and DeepStack
  feature replay for every arm.
- Position path: explicit Multimodal Rotary Position Embedding (`M-RoPE`)
  identifiers derived from the ordinary two-dimensional mask.
- Prohibited changes: crop, resize intervention, pixel masking, visual
  re-encoding by arm, changed candidate rows, changed regional indices,
  generation, training, layer sweep, and architecture modification.

## Frozen Bias Doses and Arms

The finite bias values are:

\[
\lambda\in\{0.5,1.0,2.0\}.
\]

These correspond to attention-weight multipliers of approximately `1.65`,
`2.72`, and `7.39` before renormalization. No per-case bias is selected.

The arms are:

1. **Implicit-Position Standard Two-Dimensional Attention Mask**;
2. **Explicit-Position Standard Two-Dimensional Attention Mask**;
3. **Zero-Bias Custom Four-Dimensional Attention Mask**;
4. **Target-Region Bias 0.5** and **Competitor-Region Bias 0.5**;
5. **Target-Region Bias 1.0** and **Competitor-Region Bias 1.0**; and
6. **Target-Region Bias 2.0** and **Competitor-Region Bias 2.0**.

The four-dimensional additive mask is zero for every causally visible key,
negative infinity for every future key, and positive (lambda) only on the
selected image-token key columns. All image and non-image keys remain
available.

The bias is deliberately uniform across language-decoder layers, attention
heads, and row phases. A null closes only this uniform operator.

This is a decoder-wide post-vision spatial-key reweighting operator. Because
the selected image-token columns are biased for every causal query, it changes
image-token-to-image-token reads as well as later text-token-to-image-token
reads. It is not a text-query-only cross-modal attention intervention.

## Trust Gate

For both owner rows, the explicit-position standard path and zero-bias custom
path must preserve selected-token ranks and keep maximum absolute selected-row
token log-probability drift at or below `1e-4` relative to the ordinary path.
Every interpreted effect must be at least ten times that case's no-op drift.

If image `139` fails, stop. Any later case that fails is excluded without
relaxing the threshold; valid cases remain interpretable.

The parent target and competitor rows, mask indices, feature fingerprint, and
resolved config hash must match exactly before scoring. Feature continuity is
defined by the feature fingerprint content hash, shape, and data type; a CUDA
device-label difference is not a continuity failure. Here, Compute Unified
Device Architecture (`CUDA`) names the accelerator runtime only.

For each owner row, the new zero-bias selected-token log probabilities must
also match the parent's `all_allowed_4d` arm within `1e-4`. The image `139`
real smoke must additionally show that bias `0.5` changes at least one selected
row token log probability by more than ten times the measured no-op drift;
otherwise stop before the cohort because the finite-bias execution path has not
been demonstrated to actuate the model.

## Phase Scores and Causal Contrasts

Preserve the same phase scores as the parent unit:

1. row-entry-versus-terminal log odds;
2. first differing description token, when defined;
3. complete description span;
4. each of (x_1,y_1,x_2,y_2);
5. complete geometry span; and
6. complete row.

For dose (lambda) and phase (p):

\[
\Gamma_{p,\lambda}(M)
=
\ell_p(Y_A\mid M,\lambda)-\ell_p(Y_B\mid M,\lambda),
\]

\[
\Delta_{p,\lambda}
=
\Gamma_{p,\lambda}(M_A)-\Gamma_{p,\lambda}(M_B).
\]

Preserve target and competitor absolute release relative to zero bias. A
positive crossover without target-positive and competitor-negative gamma is
one-sided redistribution, not an owner switch.

Copy the parent hard-eligibility phase scores into the new receipt without an
additional inference pass. For every finite dose, report owner-gamma sign
agreement and signed distance toward the corresponding hard-eligibility
endpoint. For every owner-gamma field (g), record:

\[
\text{movement-from-zero}_{\lambda}=g_{\lambda}-g_{0},
\qquad
\text{signed-distance-to-hard}_{\lambda}=g_{\lambda}-g_{\mathrm{hard}}.
\]

Sign agreement means that \(g_{\lambda}\) and \(g_{\mathrm{hard}}\) have the
same strict sign; zero agrees with neither sign. The hard endpoint is
diagnostic only and cannot
satisfy this unit's promotion rule.

## Decision Rule

### Promote one bounded soft-bias free-row replay

Promotion requires at least two independent valid cases at the **same frozen
bias value** satisfying:

1. full-row and geometry crossovers are each at least `0.10` natural-log units
   per token and at least ten times no-op drift;
2. target-region gamma is positive and competitor-region gamma is negative for
   both complete row and geometry;
3. target full-row release is at least `+0.05`;
4. competitor full-row release is at least `-0.05`, preventing promotion based
   on severe destruction of the already preferred competitor; and
5. for different-category cases, description and first-differing-token gamma
   also switch ownership when the latter is defined.

At least one of the two promoting cases must be a different-category case with
semantic owner reversal. Two same-category geometry reversals alone are
insufficient because they may reflect positional or address steering rather
than object-specific routing.

If the first shared dose satisfying the rules above is `2.0`, every promoting
case must already have the same target-positive and competitor-negative owner-
gamma signs at dose `1.0`. A sign-discontinuous `2.0`-only spike is
inconclusive. When more than one shared dose passes, select the lowest passing
dose rather than the largest-effect dose.

Promote only the strongest one case to one bounded no-cache free-row replay.
This does not authorize training.

### Route to one phase-specific discriminator

If at least two valid cases at one shared bias show the same description-only
or geometry-only owner reversal, stop and define one phase-specific unit.

### Close uniform soft spatial-key bias

Close this operator if no shared bias yields two complete or recurrent phase-
specific cases, or if all apparent crossovers require substantial competitor
destruction.

### Inconclusive

One isolated complete case remains a mechanism anchor. Do not add doses,
change masks, or expand the cohort in this unit.

## Falsification and Confounds

- A zero-bias no-op failure invalidates the corresponding case.
- A monotonic continuation shift without owner gamma reversal supports generic
  salience, not object routing.
- A response only at `2.0` that approaches the destructive hard-mask phenotype
  does not establish a context-preserving operating point.
- The regions remain object-centered rather than object-pure; dense windows
  contain multiple official object centers.
- The visual features were globally contextualized before the intervention.
- The cases are deliberately phenotype-selected and cannot estimate
  prevalence or detector performance.
- Multiple doses are diagnostic. The decision uses one shared dose and does
  not select a separate best value for each case.

## Minimal Implementation Route

Add one experiment-local runner and focused tests. Reuse parent receipt
loading, runtime assembly, prompt construction, exact feature capture and
replay, explicit position construction, canonical row scoring, phase metrics,
and owner-crossover semantics.

Add only:

- a validated additive four-dimensional causal attention-bias constructor;
- exact continuity checks against the parent receipt; and
- a shared-dose panel classifier.

Run image `139` first, then the remaining three cases only if zero-bias parity
passes. Stop after the four-case receipt and independent scientific audit.

## Not Claimed

This unit cannot establish population prevalence, Average Precision or recall
improvement, native causal necessity, an instance pointer, commit or coverage,
autonomous traversal, a final architecture, or training learnability.
