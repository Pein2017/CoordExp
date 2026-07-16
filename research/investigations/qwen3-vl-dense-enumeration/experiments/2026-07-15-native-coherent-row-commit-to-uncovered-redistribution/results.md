---
title: Native Coherent-Row Commit-to-Uncovered Redistribution Factorial Results
description: Verified one-case evidence for geometry-conditioned visual repair and batch-sensitive same-class instance selection, with stable native commit held after the execution-invariance gate failed.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-native-coherent-row-commit-to-uncovered-redistribution
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: narrowed_after_failed_execution_invariance_gate
updated: 2026-07-15
---

# Native Coherent-Row Commit-to-Uncovered Redistribution Factorial Results

## Scope and Evidence Identity

This unit used one exact Qwen3 Vision-Language (`Qwen3-VL`) state on Common
Objects in Context validation image `7574`. The target was the standalone white
bowl on the upper-left cabinet. The natural sampled trajectory first emitted
that bowl and immediately emitted the same physical bowl again. The causal
recipient contained the exact original prompt and zero generated-prefix tokens.

The factorial appended either no row or one exact ten-token model-native row,
then evaluated only the first complete free action. The four equal-length rows
crossed:

- the target white-bowl description and the source clear-or-gray-bottle
  description; and
- the target white-bowl geometry and the source clear-or-gray-bottle geometry.

No ground-truth output token was injected. The full panel used one greedy call
and eight paired sampled calls per condition at temperature `0.4`, top-p
nucleus threshold `0.95`, repetition penalty `1.0`, maximum generation horizon
`512`, and physical batch size four. All 45 first free actions were valid
complete rows. There were no immediate terminal, invalid, unsupported, or
unresolved actions under direct image review.

The conclusion-owning full-panel receipt is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/
  paired-panel-20260715a/receipt.json
```

Its Secure Hash Algorithm 256-bit (`SHA-256`) digest is:

```text
34d907ecc0e274d62ced87354e5e83e9b2dec0110e90476b5502b9bd96c0741c
```

## Full-Panel Observation at Physical Batch Size Four

Every greedy action and every one of the eight paired sampled actions followed
the same physical-object outcome within each condition:

| Appended row | Greedy outcome | Eight paired sampled outcomes | Manual physical-object interpretation |
|---|---:|---:|---|
| No Appended Row Baseline | 1 of 1 | 8 of 8 | original white target bowl |
| Target Bowl Description with Target Bowl Geometry | 1 of 1 | 8 of 8 | distinct orange bowl on the upper cabinet |
| Other Bottle Description with Other Bottle Geometry | 1 of 1 | 8 of 8 | distinct adjacent blue bottle, annotation `88785` |
| Target Bowl Description with Other Bottle Geometry | 1 of 1 | 8 of 8 | source clear-or-gray bottle, annotation `90913` |
| Other Bottle Description with Target Bowl Geometry | 1 of 1 | 8 of 8 | original white target bowl, annotation `1535235` |

The orange bowl is a visible Common Objects in Context 80-category object but
is not annotated in this image. It is therefore a verified uncovered object,
not a hallucination.

Within this batch-four panel, the smallest rule covering all five conditions
is:

```text
geometry identifies the visual object currently being discussed
  -> if the phrase agrees with that object, accept the row and advance
  -> if the phrase disagrees, re-identify or repair the object at that geometry
```

The two factual transitions allow a successor of the same category:
white bowl to orange bowl and clear-or-gray bottle to blue bottle. Category-
wide lexical inhibition is therefore rejected. The two crossed rows also show
that phrase or geometry alone is insufficient to reproduce factual-row
advancement.

## Execution-Invariance Gate

The unit's primary interpretation required the coherent target row to have one
stable immediate successor. It did not. The exact same coherent-target prompt
selected different bowl instances solely as physical batch execution changed.

| Physical batch size | Maximum new tokens `16` | Maximum new tokens `512` |
|---:|---|---|
| 1 | repeats white target bowl; first horizontal coordinate token `181` | repeats white target bowl; first horizontal coordinate token `181` |
| 4 | advances to orange successor bowl; first horizontal coordinate token `438` | advances to orange successor bowl; first horizontal coordinate token `438` |

The first-coordinate selected-token log probabilities were also stable within
each batch regime:

- batch size one: approximately `-3.329308` and `-3.329306`;
- batch size four: approximately `-3.604852` and `-3.604850`.

The diagonal receipts are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/
  execution-equivalence-greedy-max512-batch1-20260715a/receipt.json

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/
  execution-equivalence-greedy-max16-batch4-20260715a/receipt.json
```

Their `SHA-256` digests are:

```text
e15082d9e2a6a663ec7b12e905fba024f958fe54c2f70a03ecf99661c983bcf8
ed07e37500846c108aba846a9df3e26b4bfe8770e00a8a9c519ad2504c2155d3
```

The maximum generation horizon is therefore ruled out as the cause. Physical
batch execution changes which same-class coordinate mode wins. Divergence
starts only at the first coordinate after both regimes have generated the
same `bowl` phrase and box-start wrapper.

The sampled requests strengthen the batch-cardinality diagnosis. Sampling was
condition-major: each physical batch of four contained four copies of the same
prompt with different paired random seeds. The coherent-target condition still
selected the orange bowl in all eight requests. Mixed neighboring prompts are
therefore not required for the observed batch-four mode, although a one-token
greedy homogeneous-versus-mixed logit panel remains necessary to separate
floating-point kernel sensitivity from padding, mask, position, or collation
semantics.

## Supported

1. **Prior geometry can act as a visual address.** Both crossed rows caused
   the model to name and localize the physical object present at the supplied
   geometry rather than blindly preserve the supplied phrase.
2. **Phrase-geometry compatibility affects the transition.** Equal-length
   factual and crossed rows do not produce one generic complete-row effect.
3. **The first-coordinate decision is multimodal and execution-sensitive.**
   Physical batch size changes the selected bowl instance while phrase and
   syntax remain stable.
4. **Generation horizon is not the cause of the discrepancy.** Changing the
   maximum horizon from `16` to `512` leaves the result unchanged within each
   batch regime.
5. **The full batch-four factorial is a valid description of batch-four
   phenomenology.** Its paired calls are internally exact and free from parser
   or invalid-output asymmetry.

## Held or Rejected

- **Stable native physical-object commit-to-uncovered redistribution is held.**
  The coherent white-bowl row does not robustly advance under batch size one.
- **An explicit or implicit order-free object ledger is not identified.**
- **A category-inhibition explanation is rejected** because both coherent
  factual rows advance to another object of the same category.
- **A phrase-only, geometry-only, or uniform equal-length-row explanation is
  rejected as sufficient** for the five-arm interaction.
- **A generic model corruption explanation is rejected as sufficient.** Four
  of five greedy condition outcomes are unchanged across the diagonal reruns;
  the sensitive decision is the competing same-class bowl geometry.
- **Architecture and training promotion remain rejected.** This event cannot
  select a ledger, object slot, cursor, bridge, or loss.

## Mechanism Update

The narrowest durable conclusion is:

> Qwen3-VL can use a previously emitted bounding-box region to re-identify and
> correct the object associated with that region, but its choice between
> competing same-class visual instances at the first coordinate token is
> sensitive to physical batch execution. Coherent-row advancement is therefore
> not yet a stable native commit mechanism.

This result shifts the immediate bottleneck. Before asking whether a row writes
an object into a native ledger, the experiment harness must establish that the
same prompt produces the same coordinate distribution across semantically
equivalent batching surfaces. The instability may be:

1. bfloat16 batch-shape-dependent floating-point or kernel variation amplified
   by a near-tied coordinate distribution;
2. batch-dependent padding, attention mask, multimodal rotary position, or
   collation semantics; or
3. an unintended cross-request interaction.

The current evidence does not distinguish these explanations.

## Exactly One Next Discriminator

Run a **Homogeneous and Mixed Batch Coordinate-Logit Invariance Probe** on the
exact coherent-target prompt.

Force the common row-start token, `bowl` phrase, object-reference close token,
and box-start token sequentially through the ordinary generation key-value
cache, then score exactly one first-coordinate step under:

1. physical batch size one;
2. physical batch size four with four identical coherent-target prompts; and
3. physical batch size four with one coherent-target prompt and the three
   equal-length factorial control prompts, rotating the target through all four
   batch positions.

Cached forced-suffix replay owns the conclusion and must first reproduce
coordinate bin `181` at batch size one and coordinate bin `438` at homogeneous
batch size four. A secondary direct full-prefix forward pass may localize a
cached-generation difference but cannot replace that gate.

Capture the complete coordinate-logit vector as float32 data and report:

- full-vector differences;
- probability mass near the white-bowl and orange-bowl horizontal-coordinate
  regions;
- top-coordinate margin;
- dependence on batch cardinality, neighboring prompt content, and target
  batch position.

This is a scoring probe, not a rollout. Use the source-consistent Brain Floating
Point 16-bit (`bfloat16`) model lineage first. Full-model float32 execution is a
separately declared diagnostic lineage and should be entered only if the
source-consistent panel shows a floating-point-scale perturbation without
semantic batch-position effects.

Decision rules:

- homogeneous batch size four differs from batch size one: batch cardinality or
  kernel execution is active;
- only mixed neighbors differ: neighbor content, request ordering, collation,
  or cross-request contamination is active;
- only target batch position differs: request collation or position indexing is
  suspect;
- vectors are nearly equal but their maximum coordinate flips: a near tie
  amplifies a batch-dependent perturbation without identifying its origin.

Do not move to a second image, training screen, or architecture until this gate
identifies the execution surface.

## Verification

- full-panel first actions: `45 of 45` valid complete rows;
- focused tests: `9 passed`;
- source-bundle, image, prompt, model, adapter, tokenizer, runtime, generation
  configuration, and sampled-runtime identity checks passed;
- direct manual review classified every first action;
- independent read-only artifact recount reproduced all condition counts;
- independent scientific arbitration localized the discrepancy to physical
  batch execution and rejected the horizon explanation;
- research-local runner and test files compile and pass without shared runtime
  modifications.

The visual review montage is stored under the full-panel artifact root at:

```text
review/sample-cell-00-five-panel.png
```
