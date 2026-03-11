---
title: Full Idea v2
status: active
scope: stage2-channel-b
topics: [stage2, channel-b, clean-prefix, duplicate-ul]
supersedes: progress/directions/stage2_emish_set_supervision_v1.md
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE2_DESIGN.md
  - docs/training/STAGE2_RUNBOOK.md
  - openspec/specs/stage2-ab-training/spec.md
---

# Full Idea v2

This note is the main research-history summary for the current clean-prefix Stage-2 direction.

Read it after the stable docs layer when you want the motivation, contract, and failure-mode framing behind the landed design.

## Status

- The stable docs-layer design overview lives at:
  - [docs/training/STAGE2_DESIGN.md](../../docs/training/STAGE2_DESIGN.md)
- Normative behavior still lives in:
  - [docs/PROJECT_CONTEXT.md](../../docs/PROJECT_CONTEXT.md)
  - [Stage-2 Runbook](../../docs/training/STAGE2_RUNBOOK.md)
  - [Stage-2 AB spec](../../openspec/specs/stage2-ab-training/spec.md)
- This file is the concise historical companion that explains why the current contract looks the way it does.

## Relationship To v1

- [stage2_emish_set_supervision_v1.md](stage2_emish_set_supervision_v1.md) remains important historical background.
- Its Stage-2 Channel-B description is not the current repo contract anymore.
- The key v2 change is semantic:
  - generic unmatched accepted predictions stay neutral by default
  - duplicate-certified continuations are not neutral and receive targeted negative supervision

## Core Design Contract

### Hard constraints

- Base model remains a pretrained V-LLM in the Qwen3-VL family.
- Coordinates remain existing `<|coord_k|>` tokens with `k in [0, 999]`.
- No DETR-style detection head, query tower, or separate objectness classifier is added.
- Training stays compatible with standard SFT infrastructure:
  - teacher forcing
  - LM logits over the normal vocabulary
  - extra losses derived from those logits

### Simplified philosophy

The design is centered on three ideas:

1. Stage-1 learns the language of boxes: structured JSON, coord-token protocol, and basic geometry.
2. Stage-2 should not solve every pathology with one objective: Channel-A stabilizes geometry and self-context, while Channel-B handles rollout-time set/cardinality failures.
3. Generic unmatched predictions are not the same as self-collision duplicates: unmatched accepted extras stay neutral by default, while duplicate-certified continuations get a targeted negative signal.

## Output / Parsing Invariants

- Dense mode uses one top-level JSON object: `{"objects": [{...}, {...}]}`
- Each object record contains:
  - required `desc`
  - exactly one geometry field: `bbox_2d` or `poly`
- Canonical key order is `desc` first, then geometry.
- Coordinate tokens are emitted as bare `<|coord_k|>` tokens and mapped to `[0, 1]` by `k / 999`.
- Channel-B uses strict parsing:
  - top-level object must contain exactly one key, `objects`
  - `objects` must be an array
  - each record must contain exactly `desc + one geometry field`
  - invalid records are dropped rather than repaired

## Canonical Stage-2 Partition

Given a raw rollout, parsed valid objects are partitioned in two steps.

First, sequential de-duplication over rollout order creates:

- `A`: accepted clean-sequence objects
- `D`: duplicate objects removed from the positive prefix

Then Hungarian matching on `A` against GT creates:

- `M`: matched accepted objects
- `U`: accepted but unmatched objects
- `FN`: GT objects not explained by `A`

The canonical flow is:

```text
raw parsed rollout
    -> sequential dedup
         -> A (accepted clean sequence)
         -> D (duplicate bursts attached to clean boundaries)
    -> Hungarian on A vs GT
         -> M, U, FN
```

This partition is the semantic heart of v2:

- `U` means "not judged yet"
- `D` means "judged redundant"

## What Is Supervised

- `M`:
  - geometry supervised
  - structure supervised
  - matched `desc` CE remains off by default in Channel-B
- `U`:
  - kept as clean-prefix context
  - no geometry loss
  - no token CE inside their spans
  - treated as possible unlabeled positives or benign extras
- `D`:
  - removed from the positive prefix
  - converted into duplicate unlikelihood at the clean boundary where they appeared
- `FN`:
  - injected into the same `objects[]` container
  - receive positive structure, description, and geometry supervision

## Stage-1 Role

Stage-1 remains the scalable foundation and is not thrown away in v2.

- It teaches the JSON protocol and `desc_first` record format.
- It teaches coord-token vocabulary confinement and basic geometric calibration.
- It keeps the standard coord-distribution recipe:
  - hard coord CE anchor
  - soft CE
  - W1 / ordinal shaping
  - leakage / gate control where applicable

The working belief in v2 is that hard coord CE and soft/distributional shaping are complementary rather than mutually exclusive.

## Stage-2 Layers

### Channel-A: geometry and self-context stabilization

Channel-A is the hot path. Its job is not to solve duplicate bursts directly, but to keep geometry and self-conditioning stable while Channel-B remains sparse.

Recommended defaults and intent:

- iterative soft/ST coord self-context
- `n_softctx_iter = 2`
- `coord_ctx_embed_mode = st`
- `coord_decode_mode = exp` or `st`
- `softctx_grad_mode = unroll`
- fallback to `softctx_grad_mode = em_detach` if self-conditioning gradients get noisy

An optional weak A1 anchor can be useful in crowded scenes to reduce noisy first-iterate coord beliefs.

### Channel-B: clean-prefix rollout correction

Channel-B is the cold path and the canonical v2 novelty. It treats near-duplication as a local autoregressive continuation problem under a specific clean prefix.

The recommended operating policy is:

- start A-hot / B-cold
- increase `b_ratio` only after duplication and truncation are under control

## Clean-Prefix Training

Channel-B uses two views of the same rollout:

1. Raw rollout view: discover duplicate continuations.
2. Clean rollout view: define the positive teacher-forced prefix.

The clean teacher-forced sequence `y_in` is built by:

1. keeping accepted objects `A` in their rollout order
2. keeping accepted unmatched objects `U` as context
3. removing duplicates `D` from the positive prefix
4. appending missing GT objects `FN` inside the same top-level `objects[]` container

This is the critical training answer in v2:

- later correct objects are teacher-forced under the clean deduplicated prefix, not under the duplicate-contaminated raw prefix
- generic accepted extras are not deleted just to make the prefix look GT-pure

## Duplicate Unlikelihood

The duplicate fix is not a blanket punishment over all unmatched spans.

For each clean boundary `k`:

- `p_k` is the clean prefix ending after accepted object `a_k`
- `c_k^+` is the canonical positive continuation from that boundary
- `D_k` contains raw duplicate objects observed after that clean prefix

For each duplicate `d in D_k`, compare:

- `T_pos(d)`: token sequence of the canonical positive continuation
- `T_dup(d)`: token sequence of the duplicate continuation

Let:

- `l = LCP(T_pos(d), T_dup(d))`
- `u(d)` be the first token in `T_dup(d)` after that common prefix

Then apply unlikelihood at the first divergence point:

```text
L_dup = sum_{k} sum_{d in D_k} alpha(d) * [ -log( 1 - p_theta(u(d) | x, p_k + T_pos(d)[:l]) ) ]
```

Why this matters:

- it targets the local decision that created redundancy
- it avoids the naive mistake of always suppressing the first `desc` token
- when the next valid object is also the same class, the divergence point often lands on the first differing coord token instead

Default weighting should stay simple at first:

- `alpha(d) = 1`

## Channel-B Masking / Loss Summary

Canonical CE masks on the clean teacher-forced sequence:

- accepted matched objects:
  - `CE_struct = 1`
  - `CE_desc = 0`
  - `CE_coord = 0`
- accepted unmatched objects:
  - `CE_struct = 0`
  - `CE_desc = 0`
  - `CE_coord = 0`
- injected FN objects:
  - `CE_struct = 1`
  - `CE_desc = 1`
  - `CE_coord = 0`
- global closure / EOS remain supervised

Geometry losses from the same forward are applied to:

- matched accepted objects `M`
- injected `FN`

They are not applied to:

- accepted unmatched `U`
- duplicates `D`

The compact v2 Channel-B objective is:

```text
L_B = L_struct_clean + L_desc_FN + lambda_dup * L_dup + lambda_geo_B * L_geo_B
```

What is intentionally not in the core objective:

- no generic FP penalty
- no extra objectness head
- no repulsive set prior by default
- no soft-OT entropy term by default
- no RL reward shaping by default

## Why v2 Matters

The clean-prefix formulation is the minimal mechanism that actually changes model behavior rather than merely post-processing outputs.

- The raw rollout is still needed so the model's real failure mode can be observed.
- Duplicates must be identified from that raw rollout.
- But positive CE should not train later correct objects under a polluted duplicate prefix.
- The clean deduplicated prefix lets positive CE and negative UL coexist without conflating generic extras with certified self-collisions.

## Diagnostics / Acceptance Criteria

The most useful near-dup metrics are:

- `dup/max_desc_count`
- `dup/near_iou90_pairs_same_desc`
- `dup/near_iou90_pairs_any_desc`
- `dup/saturation_rate`
- `rollout/pred_per_sample`
- `rollout/rollout_len_mean`
- `rollout/parse_truncated_rate`
- `rollout/matched_maskiou_mean`

A good fix should satisfy all of the following:

1. `dup/near_iou90_pairs_same_desc` drops sharply.
2. `pred_per_sample` and `rollout_len_mean` stop drifting upward.
3. `rollout/parse_truncated_rate` does not grow over training.
4. `rollout/matched_maskiou_mean` stays stable.
5. Recall does not collapse while duplicate enumeration is suppressed.

Useful bookkeeping that should remain visible in analysis:

- `extra_dup_certified`
- `extra_generic_unmatched`

The target is not to minimize all extras. The target is to suppress redundant self-collision while leaving room for unlabeled true positives.

## Planned But Not Core

Unlabeled annotation / missing positives is treated as a future extension rather than part of the core v2 objective.

The intended order of research is:

1. remove duplicate instability cleanly
2. only then study accepted unmatched objects as latent positives

That future direction should remain likelihood-based and should not require adding a separate binary objectness head.

## One-Sentence Summary

Keep Stage-1 as the language-of-boxes foundation, keep Channel-A as the cheap geometry stabilizer, and make Channel-B distinguish between neutral generic unmatched objects and duplicate-certified continuations that get clean-boundary unlikelihood.
