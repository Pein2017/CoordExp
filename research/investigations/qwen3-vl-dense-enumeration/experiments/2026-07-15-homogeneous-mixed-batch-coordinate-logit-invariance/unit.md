---
title: Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance Probe
description: One-token scoring test that separates physical batch cardinality, mixed prompt lengths, neighboring prompt content, target batch position, model precision, and coordinate modes for one exact coherent-row state.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Mixed-Length, Homogeneous, and Equal-Length Batch Coordinate-Logit Invariance Probe

## Question

Why does the exact coherent white-bowl prompt select the original white bowl at
physical batch size one but the distinct orange bowl at physical batch size
four?

The unit asks whether the first-coordinate distribution changes because of:

1. physical batch cardinality or batch-shape-dependent numerical kernels;
2. the predecessor batch's shorter no-row companion and mixed prompt lengths;
3. neighboring prompt content;
4. the target request's batch position;
5. an unintended cross-request or collation interaction; or
6. a genuine near-tie whose maximum flips under an otherwise negligible
   floating-point perturbation.

This is an execution-semantics and model-distribution probe. It does not test a
ledger, coverage mechanism, final architecture, or long-rollout performance.

## Frozen Recipient State

The source is the completed [Native Coherent-Row Commit-to-Uncovered
Redistribution Factorial](../2026-07-15-native-coherent-row-commit-to-uncovered-redistribution/results.md)
on Common Objects in Context validation image `7574`.

The target prompt is the exact original prompt plus the exact model-native
coherent target-bowl row. Its prompt-token Secure Hash Algorithm 256-bit
(`SHA-256`) digest is:

```text
d24725c7eca4b74b39043d2e6f52501decef760a0f95c270907745c7d741b351
```

Both divergent greedy executions then generate the same five-token suffix
before the first horizontal coordinate:

```text
<|object_ref_start|> b owl <|object_ref_end|> <|box_start|>
```

The corresponding token identifiers are:

```text
[151646, 65, 9605, 151647, 151648]
```

The scoring recipient contains no ground-truth token. The primary path begins
from the frozen target prompt and runs ordinary greedy generation for at most
16 new tokens through the real key-value cache. Interpretation requires the
target recipient to naturally generate the declared five tokens. The probe then
records the raw sixth-step logits before any logits processor and the ordinary
greedy sixth token. This exactly replays the predecessor's generation horizon
and avoids changing the trajectory through a prefix constraint.

A secondary teacher-forced path appends all five tokens to the full prefix and
uses one ordinary multimodal forward pass to record the next-token logits. This
direct path tests whether full-prefix recomputation agrees with cached
generation; it cannot own claims about the original greedy divergence.

Neither path samples or inspects any token after the first coordinate.

## Frozen Coordinate Modes

The prior diagonal gate identified two competing first-coordinate maxima:

- white-bowl mode: coordinate token `181`;
- orange-bowl mode: coordinate token `438`.

For descriptive probability-mass summaries, freeze two non-overlapping windows
before execution:

- white-bowl window: coordinate bins `149` through `213`, inclusive;
- orange-bowl window: coordinate bins `406` through `470`, inclusive.

Each window is the observed mode plus or minus 32 coordinate bins. The complete
1,000-coordinate vector remains the primary evidence; these windows cannot own
the conclusion by themselves.

## Batch Layouts

Every target row uses the exact same target prompt, image, model inputs, and
five-token recipient suffix.

| Layout | Physical batch | Target copies | Companion requests | Purpose |
|---|---:|---:|---|---|
| Single Target | 1 | 1 | none | Reference coordinate distribution. |
| Predecessor Mixed-Length Batch Rotation | 4 | 1 | the no-row prompt and two declared appended-row prompts | Reproduces the exact predecessor batch and rotates its fixed prompt-length multiset through all four target positions. |
| Homogeneous Target Copies | 4 | 4 | none; all four rows are byte-identical target recipients | Isolates batch cardinality and batch-shape-dependent execution without padding or semantic neighbors. |
| Equal-Length Mixed Rotation | 4 | 1 | the three other equal-length factorial recipients | Tests neighboring semantic content while rotating the target through all four batch positions without prompt padding. |

The **predecessor mixed-length batch** expands to:

1. No Appended Row Baseline, with 1,320 prompt tokens;
2. Target Bowl Description with Target Bowl Geometry, with 1,330 prompt tokens;
3. Other Bottle Description with Other Bottle Geometry, with 1,330 prompt tokens;
4. Target Bowl Description with Other Bottle Geometry, with 1,330 prompt tokens.

All four cyclic rotations preserve that condition multiset while moving the
target recipient through every physical batch position. This layout can show
that the shorter companion matters, but it cannot by itself distinguish prompt
padding from the semantic absence of an appended row.

The **three other equal-length factorial recipients** expand to:

1. Other Bottle Description with Other Bottle Geometry;
2. Target Bowl Description with Other Bottle Geometry; and
3. Other Bottle Description with Target Bowl Geometry.

Each companion receives the same five-token coordinate-recipient suffix. Their
own logits are stored for audit, but only the exact coherent-target row owns the
primary comparisons.

Every layout is executed twice in the same loaded-model process. Repeat
variation is measured rather than silently assumed to be zero.

## Competing Hypotheses and Predictions

### Hypothesis 1: Batch cardinality or numerical-kernel sensitivity

Changing only the physical batch dimension changes the target coordinate
distribution.

Predictions:

- all four Homogeneous Target Copies agree with each other;
- they differ from Single Target in the same white-versus-orange contrast seen
  by greedy decoding; and
- Equal-Length Mixed Rotation adds little beyond the homogeneous batch-four
  effect.

Falsification:

- Homogeneous Target Copies match Single Target while a mixed layout changes
  the target distribution.

### Hypothesis 2: Mixed prompt lengths or shorter-companion execution

The predecessor's shorter no-row prompt changes physical collation or a
batch-shape-dependent numerical path beyond the generic batch-four effect.

Predictions:

- every Predecessor Mixed-Length Batch Rotation differs from the equal-length
  batch-four layouts;
- the effect follows the preserved prompt-length multiset rather than target
  position; and
- full floating-point-32 execution substantially reduces the difference if it
  is numerical rather than semantic.

Falsification:

- predecessor and equal-length batch-four distributions agree within repeat
  variation.

### Hypothesis 3: Neighbor-content or mixed-batch semantics

Batch cardinality alone is harmless; the target changes only in the presence
of different equal-length neighboring recipients.

Predictions:

- Homogeneous Target Copies match Single Target;
- Equal-Length Mixed Rotation differs; and
- the change follows neighbor composition rather than target batch position.

Falsification:

- homogeneous batch size four already reproduces the full distribution shift.

### Hypothesis 4: Batch-position or request-collation defect

The target coordinate distribution depends on whether it occupies batch slot
zero, one, two, or three.

Prediction:

- the target vector and white-versus-orange contrast vary systematically as the
  identical target recipient rotates through mixed batch positions.

Falsification:

- all four rotated target positions agree within repeat variation.

### Hypothesis 5: Near-tie amplification of a batch-dependent perturbation

The white and orange coordinate modes have such a small margin that an
execution-dependent perturbation changes the maximum. A near tie is an
amplifier, not evidence that the perturbation itself is harmless.

Predictions:

- centered coordinate-vector differences are small relative to the single-
  target top-mode margin and same-layout repeat variation;
- the white-minus-orange contrast changes sign near zero; and
- no broad shift appears across unrelated coordinate regions.

Falsification:

- large structured differences affect many coordinate modes or depend on
  semantic neighbors or target position.

### Hypothesis 6: No reproducible one-token difference

Natural cached scoring is invariant even though the prior autoregressive
generation diverged.

Prediction:

- all cached layouts reproduce one coordinate distribution within repeat
  variation.

Implication:

- the new replay did not reproduce the conclusion-owning execution state, so
  interpretation stops before attributing the discrepancy to batch
  cardinality.

## Primary Measurements

For every target recipient, store the complete next-token vector as float32
data after the source-lineage Brain Floating Point 16-bit (`bfloat16`) model
forward pass. Report:

1. the complete 1,000-coordinate raw-logit vector;
2. the complete 1,000-coordinate conditional log-probability vector after
   normalizing only across coordinate tokens;
3. the full-vocabulary log probabilities for all coordinate tokens;
4. top coordinate tokens and their margins;
5. white-bowl-window and orange-bowl-window probability mass;
6. white-minus-orange mode contrast;
7. centered pairwise maximum absolute difference, root-mean-square difference,
   and Jensen-Shannon divergence relative to Single Target;
8. white-versus-orange logit-margin shift divided by same-layout repeat noise;
9. residual centered shift outside the two frozen coordinate windows; and
10. repeat, batch-position, and companion-layout variation.

Float32 artifact storage is mandatory. It does not imply that the full model
executes in float32. Full-model float32 is a separate diagnostic lineage and is
entered only if source-lineage results isolate a numerical batch-cardinality
effect without semantic neighbor or position dependence.

## Minimal Implementation Path

1. Reuse the current model loader, prompt composer, model-input materializer,
   model identity, and target/control rows from
   `scripts/research/run_native_commit_redistribution.py`.
2. Build research-local scoring recipients by appending the declared five
   tokens to each composed prompt.
3. Reuse the inference backend's prompt padding, multimodal input collation,
   generation configuration, and stock cached generation path; do not modify
   shared inference infrastructure.
4. In the primary path, run ordinary cached generation, require the target to
   naturally emit the declared five-token suffix, and capture the raw sixth-step
   logits before logits processing.
5. In the secondary path, run `model.eval()` under `torch.inference_mode()` and
   extract only `logits[:, -1, :]` from the full-prefix forward pass.
6. Convert detached logits to contiguous Central Processing Unit float32 data
   before metrics or serialization.
7. Write one receipt containing every batch layout, exact input token hashes,
   batch order, tensor shapes, runtime identity, complete coordinate vectors,
   cached-versus-direct comparisons, and predeclared summaries.

One research-local script and focused deterministic tests are sufficient. No
OpenSpec change, generic scoring interface, sampling attestation, or shared
runtime extension is authorized.

## Trust Gates

Stop interpretation if:

- the target prompt hash differs from the frozen value;
- the five-token recipient suffix differs from the declared vector;
- any layout changes the target image or target prompt;
- equal-length mixed companions are not exactly the three declared factorial
  conditions;
- a target row is not found at its declared batch position;
- multimodal inputs are not independently materialized or correctly repeated
  for every physical batch row;
- the model is not in evaluation and inference mode;
- source-lineage model, adapter, tokenizer, image, runtime, attention
  implementation, or model dtype changes;
- any target logit is non-finite;
- cached replay does not reproduce coordinate bin `181` for Single Target and
  coordinate bin `438` for the exact predecessor mixed-length batch with the
  target at its original batch position one;
- coordinate token identifiers are not exactly the declared 1,000-token set;
  or
- the serialized float32 vector cannot reproduce every reported summary.

Differences among homogeneous copies, same-layout repeats, batch positions, or
neighbor compositions are observations, not automatic trust-gate failures.

## Decision Rules

- If cached Homogeneous Target Copies differ from cached Single Target and all
  four copies agree, physical batch cardinality or numerical kernel execution
  is active.
- If the predecessor mixed-length rotations differ from homogeneous and
  equal-length layouts while remaining position invariant, the shorter
  companion or mixed-length execution contributes an additional effect.
- If homogeneous copies match Single Target but equal-length mixed layouts differ,
  neighbor-content, request ordering, or collation semantics are active.
- If rotated target positions differ, request collation or batch-position
  handling becomes the next code-level suspect.
- If centered vectors differ only slightly yet the white-minus-orange contrast
  changes sign near zero, conclude only that a near tie amplifies a batch-
  dependent perturbation. Do not use the near tie to rule out collation or
  execution defects.
- If cached scoring reproduces the prior divergence but direct full-prefix
  scoring is invariant, localize the effect to cached generation execution.
- If natural cached replay does not reproduce the prior coordinate bins, stop the unit
  as an execution-state mismatch.

No result from this one event authorizes training, architecture promotion, a
second image, or a population metric.

## Closeout

The unit completed with verified source-lineage `bfloat16` and full-model
`float32` receipts. See [results.md](results.md) for the evidence-backed verdict,
the corrected predecessor batch contract, and the next discriminator.

## Artifact Handle

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/
  <immutable-run-identifier>/
```

## Non-Goals

- no isolated padding-causality claim; the predecessor mixed-length layout
  jointly changes one companion's semantics and padding, while the equal-length
  layouts remove both;
- no long rollout, sampling, mean average precision, recall, or hallucination
  estimate;
- no hidden-state, attention-head, residual-patch, or layer sweep;
- no full-model float32 run before the source-lineage gate justifies it;
- no model training, data modification, ledger, slot, cursor, or final
  architecture;
- no claim that one batching symptom explains dense-scene low recall.
