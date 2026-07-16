---
title: Selected-Transition Batch-Precision Prevalence Screen
description: Six-case screen for whether a post-vision physical-batch divergence recurs at fixed model-native dense-image transitions and disappears under float32 execution.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-07-15-selected-transition-batch-precision-prevalence-screen
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-15
---

# Selected-Transition Batch-Precision Prevalence Screen

## Question

Does the post-`get_image_features` physical-batch divergence observed at one
white-bowl coordinate boundary recur at fixed model-native transitions from
several dense or same-class-ambiguous images, and is any recurrence conditioned
on Brain Floating Point 16-bit (`bfloat16`) rather than stable model semantics?

This is a selected-case mechanism screen. It is not an estimate of population
prevalence.

## Competing Explanations

The working explanation is that low-precision downstream execution can move a
recipient between distinct next-object or coordinate basins when physical batch
shape changes.

The strongest alternative is that image `7574` exposed a local numerical edge
case with no useful recurrence. A second alternative is a batch-shape effect
that persists under Institute of Electrical and Electronics Engineers 754
32-bit floating point (`float32`), which would reject precision conditioning
and keep collation, position construction, scatter, or other downstream batch
semantics open.

The discriminating controls are:

- one recipient versus four homogeneous copies of the exact same recipient;
- exact equality among the four homogeneous copies;
- a matched full-model `float32` rerun only for recipients that diverge under
  `bfloat16`.

## Frozen Selected Cohort

Each case comes from the manually reviewed dense-union cohort and uses canonical
Full-Image K-Rollout Independent Bagging (`FULL_BAG_K`) cell zero. Here `K`
means the frozen set of `16` independent full-image sampled rollouts, and cell
zero is the first member of that set. The source transition is the first
complete model-native row from that immutable call.

| Common Objects in Context image identifier | Selection reason | Source terminal bundle |
|---:|---|---|
| `7574` | Bowl, bottle, and tableware ambiguity; image-level predecessor anchor, not the exact earlier cell-three partial-row recipient. | `calls/f56e79e626cee96fc8b7f1b7d98db0fb8d7e338bf243d372ec3d9259a3e91562/terminal-output-bundle.json` |
| `8629` | Six visually similar pizza instances. | `calls/3c97ac12bcfce1787d56733b928ef25166c78178ca24c8fe7d99df0247495154/terminal-output-bundle.json` |
| `9891` | Repeated suitcases, cars, and persons across a broader scene. | `calls/6bca67d00fceac633b20fa8773a0ac74c373d8817b3f013bdfb3b259b7e86683/terminal-output-bundle.json` |
| `12576` | Dense food and tableware scene with repeated cups, pizzas, and forks. | `calls/b3006ff584a22ef984bf6922c9f3c91d6292eb405aae9b323fbba181f396b936/terminal-output-bundle.json` |
| `13659` | Repeated chairs and persons plus small indoor objects. | `calls/ac6c352af660dc3586381ed29229dbb22b2a29fbc87a8e167d97bfaf45ae8d8a/terminal-output-bundle.json` |
| `17714` | Repeated cups, knives, and forks with same-class competition. | `calls/1409ebbeb330a82dfe895af91d47e3d1f9221d98c98058d9e66e85fc5f40b8a8/terminal-output-bundle.json` |

The common bundle root is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-13-spatial-scope-history-disentanglement/executions/
  dense-union-51-primary-after-wave-local-tail-contract/artifacts/
```

Cell zero is used for every image to avoid selecting a source trajectory after
observing the batch comparison.

## Primary Observation

For each image:

1. reconstruct the exact source base prompt;
2. append the first complete source-generated object row;
3. greedily generate the next free action once as a physical batch of one;
4. repeat the byte-identical recipient four times in one physical batch;
5. compare action type, complete first-row token span, object description, and
   geometry.

The two named arms are:

- **Single Recipient**: one fixed image, prompt, committed source row, and
  decoder request in a physical batch of one;
- **Homogeneous Four-Copy Recipient**: four byte-identical copies of the same
  fixed recipient in one physical batch.

A primary divergence requires a difference in the complete first free action.
A difference only after the first complete action is secondary trajectory
evidence and does not count toward the recurrence gate.

## Decision Rule

- **Recurrent precision-conditioned phenomenon**: at least two independent
  images have a `bfloat16` first-action divergence, all four homogeneous first
  actions agree within each image, and matched `float32` removes or materially
  attenuates each divergence.
- **Local numerical edge case**: zero or one independent image diverges. Close
  the numeric branch for the current program rather than launching a layer
  sweep.
- **Batch-shape effect without precision localization**: divergence recurs but
  persists under `float32`. Hold precision as the cause and localize the
  downstream batch-semantic seam before any model-mechanism claim.
- **Invalid execution**: homogeneous first actions disagree, source
  reconstruction fails, source lineage changes beyond the declared dtype, or
  an action cannot be attributed. Later rows may diverge and are retained as
  secondary trajectory evidence, but they do not invalidate this first-action
  screen. Repair or invalidate primary-action failures; do not interpret them.

These rules choose the next experiment. They do not estimate a confidence
interval or claim a dataset-wide rate.

## Scope and Invariants

- Model: Qwen3 Vision-Language (`Qwen3-VL`) 2B with the step-4887
  Weight-Decomposed Low-Rank Adaptation (`DoRA`) adapter used by the source
  `FULL_BAG_K` artifacts.
- Ordering: geometry-sorted, matching adapter training.
- Decode: greedy, repetition penalty `1.0`, first-free-action interpretation,
  maximum `64` generated tokens.
- Inputs: original full images; no crop, resize intervention, visual mask, or
  external detector.
- Changed factor: physical batch shape; full-model dtype changes only in the
  promoted control run.
- Fixed factors: image, image processing, source base prompt, exact first
  source-generated row, tokenizer, adapter, coordinate-token surface, and
  decode policy.
- Precision upgrade: all model parameters and executed layers use `float32` in
  the control, following the predecessor precision diagnostic.

## Outline and Reused Infrastructure

The experiment-local runner reuses:

- current inference config and runtime assembly;
- existing source JSON Lines (`JSONL`) data loading;
- current image-plan materialization;
- `HFGenerateBackend` greedy batched decoding;
- current compact object-row parser and first-action extraction.

No shared inference module, model forward path, architecture, training loss,
slot, ledger, or decoder constraint is added.

Execution proceeds in two stages:

1. run all six recipients once under `bfloat16`;
2. run only divergent recipients under `float32`.

Representative smoke: image `7574`, followed by the remaining five cases only
after prompt reconstruction, source row extraction, and homogeneous-copy
equality pass.

Rough cost: one `bfloat16` model load plus thirty short requests; one additional
`float32` model load and five short requests per promoted divergent case.

## Stop Rule

Stop after the decision rule classifies the selected cohort. Do not start a
layer-onset probe, training screen, architecture implementation, or population
benchmark inside this unit.

## Artifact Handle

Logical root:

```text
outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-selected-transition-batch-precision-prevalence-screen/<run-id>/
```

Resolved durable root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-07-15-selected-transition-batch-precision-prevalence-screen/<run-id>/
```

The compact receipt must retain source bundle hashes, source prompt and row
hashes, model and tokenizer identity, dtype, exact raw first-action outputs,
homogeneous-copy equality, and batch-one-versus-batch-four comparison.

## Not Claimed

This unit cannot establish:

- a population prevalence estimate;
- that low precision causes dense-scene low recall;
- that `float32` should be used for production inference;
- the first downstream layer responsible for a divergence;
- a native object ledger, commit operation, or complete enumeration mechanism;
- any training or architecture recommendation.
