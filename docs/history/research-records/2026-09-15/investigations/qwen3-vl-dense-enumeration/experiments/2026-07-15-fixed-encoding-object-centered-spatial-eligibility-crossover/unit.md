---
title: Fixed-Encoding Object-Centered Spatial-Eligibility Crossover
description: Six-case causal panel testing whether symmetric post-vision image-token key eligibility redistributes support between two real object rows without cropping, resizing, masking pixels, or re-encoding the image.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Fixed-Encoding Object-Centered Spatial-Eligibility Crossover

## Question

With the full image encoded exactly once, can object-centered eligibility over
already-computed image-token positions redistribute language-decoder support
between two valid object rows?

This unit tests a narrow post-vision routing hypothesis:

\[
\text{many eligible image-token routes}
\rightarrow
\text{candidate probability fragmentation or suppression}.
\]

It does not test pixel masking, crop-resize inference, a detector architecture,
or detector-level performance.

## Motivation and Competing Explanations

The preceding Full-Canvas Masked Region with Per-Region Reset study showed that
local spatial restriction can recover objects missed by Full-Image K-Rollout
Independent Bagging, but pixel masking also caused fragmentation, category
errors, and unsupported hallucinations. That intervention simultaneously
changed the image, vision-tower computation, visible context, and language
horizon.

The current hypotheses are:

1. **Object-Specific Post-Vision Routing**: target evidence exists in the
   fixed visual encoding, but unrestricted decoder-side image-token
   competition suppresses or fragments the corresponding object row.
2. **Generic Context Reduction**: any smaller visual scope changes likelihoods
   similarly, without favoring the object occupying that scope.
3. **Generic Continuation Activation**: restriction changes the decision to
   emit another row but does not select its owner.
4. **Phase-Limited Routing**: the intervention controls semantic identity but
   not geometry, or geometry but not semantic identity.
5. **Global-Context Dependence or Invalid Seam**: hard eligibility destroys
   useful context, or this particular language-decoder key mask is not a usable
   post-vision routing surface.

## Frozen Selected Targets

The six targets are official Common Objects in Context (`COCO`) annotations
that the prior masked arm matched but the sixteen-rollout full-image bagging
union did not match. Every target was selected before the present intervention
was observed.

| COCO image identifier | Target annotation identifier | Target category | Pair stratum | Selection reason |
|---:|---:|---|---|---|
| `139` | `1669970` | vase | different-category | Small vase amid many furniture and container instances. |
| `632` | `1661908` | book | same-category | Small book inside a dense same-class bookcase. |
| `2299` | `2008221` | person | same-category | One person inside a heavily overlapping crowd. |
| `9400` | `1292246` | person | different-category | Person amid laptops, keyboards, cups, and other people. |
| `12120` | `2030099` | person | different-category | Small person in a dense tennis and spectator scene. |
| `12639` | `543629` | person | same-category | One player inside a dense same-class sports group. |

Target rows come from the frozen source JSON Lines (`JSONL`) record, including
its exact category text and four normalized coordinate tokens. The audit ledger
owns object validity and source-canvas geometry.

## Competitor Selection Before Intervention

For each image, define target object \(A\) and choose competitor object \(B\)
without inspecting any restricted-arm score.

1. Build canonical rows for every other official object in the same frozen
   source record.
2. Keep candidates whose object center can own an exact translated copy of the
   target eligibility window without clipping or overlapping the target
   window.
3. For a `same-category` case, retain only objects with the target category.
   For a `different-category` case, retain only objects with a different
   category.
4. Under the ordinary unrestricted two-dimensional attention mask, score every
   remaining candidate row and select the row with the highest mean full-row
   log likelihood.
5. Require the selected competitor row to have higher unrestricted mean
   full-row log likelihood than the target row. Otherwise the case is not a
   valid base-prefix competition case and is reported as ineligible rather
   than replaced after restricted-arm inspection.

This makes \(B\) the model-preferred valid competitor within the declared
stratum while avoiding a post-hoc choice based on the causal intervention.

## Fixed Execution Contract

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the geometry-sorted
  step-4887 Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter.
- Model execution: full Institute of Electrical and Electronics Engineers 754
  32-bit floating point (`float32`).
- Attention implementation: Scaled Dot Product Attention (`SDPA`).
- Physical batch size: one.
- Prefix: the standard reset/base detection prompt with no prior object row.
- Decode policy for scoring: teacher-forced, no cache, repetition penalty not
  applied.
- Image: original full-resolution processed canvas from the frozen source
  record.
- Prohibited changes: crop, resize intervention, pixel masking, per-arm visual
  re-encoding, external detector, training, long rollout, and End-of-Sequence
  suppression.

The primary and every DeepStack visual-feature stream are computed once per
image, cloned, hashed, and replayed byte-identically for competitor selection
and every causal arm.

## Position and Attention Trust Gate

A custom four-dimensional attention mask would otherwise change the position
mask inferred by the outer Qwen3-VL model. Therefore, the runner derives
Multimodal Rotary Position Embedding (`M-RoPE`) position identifiers once from
the ordinary two-dimensional mask and supplies those exact position identifiers
to all explicit-position and restricted arms.

Three no-op paths are compared before any restriction is interpreted:

1. **Implicit-Position Standard Two-Dimensional Attention Mask**;
2. **Explicit-Position Standard Two-Dimensional Attention Mask**; and
3. **All-Image-Allowed Custom Four-Dimensional Attention Mask** with the same
   explicit position identifiers.

For every target and competitor row, both no-op transitions must preserve token
ranks and keep the maximum absolute selected-row token log-probability drift at
or below `1e-4`. Every interpreted intervention effect must additionally be at
least ten times the matching no-op drift. If the one-case smoke fails this
gate, stop without running the remaining cohort.

## Spatial-Eligibility Arms

The target window is the merged visual-token support intersecting target
object \(A\)'s source-canvas box, dilated by one merged-token halo. The
competitor window is an exact translated copy of that binary shape centered on
competitor object \(B\). It must preserve token count and shape, remain inside
the merged grid, contain \(B\)'s center cell, and not overlap the target
window.

The four-dimensional boolean attention mask preserves causal order and all
non-image key visibility. It changes only whether an image-token key is
eligible:

\[
\operatorname{allow}(q,k)
=
\operatorname{causal}(q,k)
\land
\left[
\neg\operatorname{image}(k)
\lor
\operatorname{eligible}(k)
\right].
\]

The causal arms are:

1. **Full Image-Token Eligibility**: the validated all-image-allowed no-op;
2. **Target-Object Spatial Eligibility**: only target window image-token keys
   remain eligible; and
3. **Competitor-Object Spatial Eligibility**: only competitor window
   image-token keys remain eligible.

This intervention changes image-token propagation inside the language decoder.
It does not make the already contextualized vision-tower features local.

## Candidate Rows and Phase Scores

Let \(Y_A\) and \(Y_B\) be the canonical rows for target and competitor. For
each arm, preserve summed and per-token mean log likelihood for:

1. row-entry token and row-entry-versus-terminal log odds;
2. first description token at which \(Y_A\) and \(Y_B\) differ, when defined;
3. complete description span;
4. each of \(x_1,y_1,x_2,y_2\) separately;
5. complete geometry span; and
6. complete row.

The four coordinate slots are reported separately because later slots are
teacher-forced after earlier row tokens and cannot be interpreted as one
unconditioned geometry decision.

For phase \(p\) and eligibility mask \(M\), define:

\[
\Gamma_p(M)
=
\ell_p(Y_A\mid M)-\ell_p(Y_B\mid M),
\]

and the symmetric crossover interaction:

\[
\Delta_p
=
\Gamma_p(M_A)-\Gamma_p(M_B).
\]

Also preserve absolute competition release:

\[
R_{A,p}
=
\ell_p(Y_A\mid M_A)-\ell_p(Y_A\mid M_{\mathrm{full}}),
\]

\[
R_{B,p}
=
\ell_p(Y_B\mid M_B)-\ell_p(Y_B\mid M_{\mathrm{full}}).
\]

Same-category cases do not have a description-identity contrast; they test
instance geometry only.

## Decision Rule

### Promote one bounded free-row switch replay

Promote only the strongest single case if at least two independent eligible
cases satisfy all applicable conditions:

1. full-row mean crossover \(\Delta_{\mathrm{row}}\ge 0.10\) natural-log units
   per token and at least ten times no-op drift;
2. geometry-span mean crossover \(\Delta_{\mathrm{geometry}}\ge 0.10\);
3. for different-category cases, the first differing description token and
   description-span crossovers both favor the window owner; and
4. at least one owner row has positive absolute release of at least `0.05`
   natural-log units per token over full eligibility.

This promotes one no-cache free-row causal switch. It does not authorize
training or architecture promotion.

### Route to a phase-specific unit

If recurrent effects appear only in description or only in geometry, stop this
unit and formulate exactly one phase-specific discriminator.

### Close this operator

Close hard post-vision image-token key eligibility if every valid case has
full-row crossover below `0.10`, or target and competitor windows move both
candidate rows in the same direction without owner-specific reversal.

### Inconclusive

One isolated owner-specific case remains a case study. Do not expand the
cohort or launch training inside this unit.

## Interpretation Table

| Observation | Bounded interpretation |
|---|---|
| A no-op path exceeds the trust tolerance | Invalid intervention path; stop. |
| Each object-centered window favors its owner, with positive absolute release | Object-specific post-vision routing and competition release are supported. |
| Symmetric owner reversal without positive release over full eligibility | A routing seam exists, but harmful unrestricted competition is not established. |
| Only row-entry-versus-terminal changes | Generic continuation or context effect. |
| Target and competitor windows affect both rows similarly | Generic restriction or normalization. |
| Description reverses but geometry does not | Semantic routing without whole-row geometry binding. |
| Geometry reverses but description does not | Spatial address routing without semantic owner binding. |
| Both rows degrade strongly under both restricted windows | Global context is required or hard gating is destructive. |
| All cases are null with a valid no-op | Close this operator, not all possible post-vision competition mechanisms. |

## Minimal Implementation Route

Use one experiment-local runner and targeted unit tests. Reuse current runtime
assembly, prompt construction, source JSONL loading, exact coordinate tokens,
fixed visual-feature capture and replay, merged-grid support mapping, and model
identity receipts.

Add only the experiment-local utilities required to:

- translate one target window to a declared competitor center;
- build and validate the four-dimensional causal key-eligibility mask;
- derive and freeze explicit M-RoPE position identifiers; and
- score candidate row phases in one no-cache forward pass.

Do not modify the shared inference backend, model forward implementation,
training configuration, or production artifact contract.

The first real smoke is image `139` under full-model `float32`. Run the other
five images only after the three-path no-op trust gate, exact feature replay,
mask invariants, and row round-trip checks pass.

## Stop Rule

Stop after the six-case crossover panel is classified. Do not start a
two-hundred-and-fifty-six-image training screen, layer sweep, slot or query
architecture, coverage ledger, long rollout, or detector evaluation in this
unit.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-fixed-encoding-object-centered-spatial-eligibility-crossover/<run-id>/
```

The compact receipt must preserve source and ledger hashes, target and selected
competitor identities, every candidate's unrestricted selection score,
primary and DeepStack feature hashes, position-identifier hashes, mask indices
and invariants, no-op drift, raw selected-token log probabilities, phase
scores, crossover interactions, absolute release terms, and the final bounded
classification.

## Not Claimed

This unit cannot establish population prevalence, mean Average Precision
improvement, native autonomous traversal, causal necessity of unrestricted
competition, absence of vision-tower global mixing, or a final architecture.

## Executed Result

The frozen unit is complete. The [verified result](results.md) accepts five
valid cases, excludes one no-op failure, and stops as an isolated complete
regional-switch case rather than promoting the preregistered free-row replay.
