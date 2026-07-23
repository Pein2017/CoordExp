---
title: Unified vLLM Sampled-Only Trajectory Panel Receipt
description: Evidence and claim boundaries for the complete 2,432-image, sixteen-trajectory-per-image source-model sampling panel.
type: investigation
role: execution-receipt
authority: non_normative_research
unit_id: 2026-07-22-constant-dose-image-breadth-treatment-screen
topic: qwen3-vl-dense-enumeration
status: completed
evidence_status: verified
updated: 2026-07-23
---

# Unified vLLM Sampled-Only Trajectory Panel Receipt

## Purpose

This artifact records the complete high-throughput trajectory collection used
to estimate which physical objects appear anywhere in sixteen low-temperature
source-model trajectories for each candidate image. `vLLM` refers to the
high-throughput inference runtime used by this collection. The panel is a
sampling-support artifact; it is not a matched greedy-versus-sampling
comparison.

## Frozen Collection Contract

- source checkpoint: geometry-sorted, description-first, pure cross-entropy
  plus token-type gate, step 4,887;
- candidate pool: 2,432 distinct images;
- trajectories: exactly sixteen sampled completions per image;
- trajectory identity: `sample_index` from 0 through 15 within each image;
- temperature: 0.4;
- nucleus probability (`top_p`): 0.95;
- repetition penalty: 1.0;
- maximum generated tokens: 1,024;
- maximum prompt-plus-generation model length: 4,096;
- vLLM scheduler capacity: 32 simultaneous sequences per worker;
- execution: eight graphics-processing-unit workers, each owning a disjoint
  304-image shard and processing 16 images per persisted batch;
- truncation rule: any completion ending because of the token limit stops that
  worker after preserving diagnostic evidence.

Request seed is not an experimental variable and is absent from the
production routes. `sample_index`, not a seed value, is the trajectory
identifier.

## Why Greedy Was Removed from the Production Contract

The first targeted 4,096-token smoke used images 1,757, 18,688, 47,739, and
57,213. All four greedy completions generated the full 1,024-token allowance
and ended by length. Direct inspection showed exact or near-exact repeated
object rows: bottle, person, bottle, and orange respectively. The matched 64
low-temperature completions all ended naturally.

Increasing the greedy allowance would collect a longer repeated-row loop, not
establish a complete trajectory. Allowing greedy truncation would violate the
zero-truncation contract. Changing its repetition penalty would introduce a
different decoding intervention. Greedy was therefore removed from the
production support panel and retained only as bounded mechanism evidence.

This decision implies a strict claim boundary: the production panel measures
sampled object support and frequency. It does not measure full-panel greedy
recall, sampled rescue relative to a matched greedy trajectory, or improvement
over greedy decoding.

## Infrastructure Changes

The reusable model-length option was implemented in the upstream
`CoordExp-swift` infrastructure worktree and then synchronized as one bounded
commit into `research-probes`.

- upstream commit `f263beaa0`: configurable positive
  `backend.vllm.max_model_len`, default 2,048;
- research synchronization commit `6fa61cf4d`;
- research configuration commit `c8259aedf`: use 4,096 for this panel;
- collector commit `cfc24ea61`;
- sampled-only collector commit `41e530c01`.

The configuration value is passed both to the vLLM engine and to the
prompt-plus-generation length guard. Focused inference and collector tests
passed before the production run.

## Failure Sequence and Corrective Evidence

1. The first production attempt at a 2,048-token model length exposed four
   context-limited greedy completions. Their prompt plus generated token counts
   were exactly 2,048. This established that the old fixed model length was a
   real infrastructure limit.
2. After raising the model length to 4,096, those four greedy routes still used
   all 1,024 generated tokens, now because of repeated-row loops rather than
   context exhaustion.
3. The sampled-only smoke on the same four images produced 64 of 64 natural
   closures with no length stop and complete sample indices 0 through 15.
4. Only then was the complete sampled-only panel launched.

Failed and diagnostic runs remain separate from the production artifact:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
trajectory-panel-2432-vllm/
```

The authoritative production subdirectory is `production-v2`.

## Production Result

Authoritative root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-22-constant-dose-image-breadth-treatment-screen/
trajectory-panel-2432-vllm/production-v2
```

Observed totals:

- workers completed: 8 of 8;
- images completed: 2,432 of 2,432;
- persisted image batches: 152;
- sampled trajectories: 38,912;
- natural end markers: 38,912;
- length-finished trajectories: 0;
- generated tokens: 3,522,816;
- parser accepted without drops: 37,707;
- parser accepted after dropping one or more malformed spans: 1,205;
- maximum single-worker wall time: 215.41 seconds;
- stored size: approximately 1.8 gibibytes.

## Independent Recalculation

The following properties were recomputed from the persisted artifacts rather
than trusted from the worker status field:

- all eight worker manifests report `completed`;
- all workers use the same execution-model identity, configuration
  fingerprint, 4,096-token model length, and sampled-only contract;
- all 152 sampled artifact SHA-256 checksums match their manifests;
- the candidate-pool image set and output image set are identical, with 2,432
  distinct identities and zero set difference;
- every image has exactly sixteen routes;
- every `(image_id, sample_index)` pair is unique;
- every image has sample indices 0 through 15;
- every route uses sampled decoding and ends with the natural image-end token;
- no route ends because of length.

`SHA-256` is the Secure Hash Algorithm 256-bit checksum used to detect artifact
changes.

## Consequence for the Training Screen

This panel is now ready for object-support mining and sampled-trajectory event
construction. It does not itself provide the source greedy owner set required
by the original route-admission wording in `unit.md`. A `StateBank` is the
persisted collection of model states, prefixes, target rows, and token-level
training supervision consumed by this experiment's training pipeline. Before
freezing a training StateBank, the research lead must either:

1. define and collect a separate finite source-baseline decoding policy; or
2. revise route admission so it does not claim a matched greedy rescue.

That choice changes the scientific meaning and must not be filled implicitly
by treating a truncated greedy loop, a changed repetition penalty, or one of
the sampled trajectories as if it were the original matched greedy baseline.
