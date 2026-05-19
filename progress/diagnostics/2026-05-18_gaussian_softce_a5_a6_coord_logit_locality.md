---
title: Gaussian SoftCE A5/A6 Coordinate-Logit Locality Follow-Up
date: 2026-05-18
status: active-reference
owner: codex
depends_on:
  - outputs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200/report.md
  - outputs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200/summary.json
  - outputs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200/report.md
  - outputs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200/summary.json
  - outputs/infer/recursive_detection_ce_latest/a5_instance_trie_gaussian_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_1gpu/eval/metrics_guarded.json
  - outputs/infer/recursive_detection_ce_latest/a6_ce_gaussian_mix0p2_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_1gpu/eval/metrics_guarded.json
---

# Gaussian SoftCE A5/A6 Coordinate-Logit Locality Follow-Up

## Scope

This note extends the hard-CE coordinate-logit and token-embedding locality
diagnosis to two Gaussian SoftCE checkpoints. It uses the same `val200`
diagnostic scope as
`progress/diagnostics/2026-05-18_hard_ce_coord_logit_embedding_locality.md`.

Dataset:

`public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

A5 Gaussian/default checkpoint:

`outputs/stage1_2b/compact_full_et_rmp_instance_trie_gaussian_softce_a5_support2_bsz16_8gpu_4epoch_tokenrows_v2/compact-full-et-rmp-instance_trie_gaussian_softce_a5-support2-bsz16-8gpu-4epoch-tokenrows-v2/v0-20260514-152238/checkpoint-3664`

A6 CE-anchored Gaussian mix-0.2 checkpoint:

`outputs/stage1_2b/compact_full_et_rmp_ce_gaussian_mix0p2_a6_support2_bsz16_8gpu_4epoch_tokenrows_v2/compact-full-et-rmp-ce-gaussian-mix0p2-a6-support2-bsz16-8gpu-4epoch-tokenrows-v2/v0-20260516-064815/checkpoint-3664`

Important provenance nuance:

- A6 explicitly records `objective.coord_soft_ce.gaussian_mixture_weight: 0.2`.
- A5 records `objective.coord_soft_ce.target_distribution:
  instance_trie_gaussian` but does not store an explicit
  `gaussian_mixture_weight` in `resolved_config.json`.
- I therefore label A5 as the Gaussian/default member. If the training-code
  default for that run was 0.5, this is the user-described 0.5 member, but the
  checkpoint metadata itself does not encode the number.

This is a mechanism diagnosis, not a full-validation benchmark.

## Artifact Roots

Analysis configs:

- `configs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200.yaml`
- `configs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200.yaml`

Analysis roots:

- A5:
  `outputs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200`
- A6:
  `outputs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200`

Self-rollout configs:

- `configs/infer/recursive_detection_ce_latest/a5_instance_trie_gaussian_ckpt3664_val200_bsz4_rep1p10_max3084_chatfix_1gpu.yaml`
- `configs/infer/recursive_detection_ce_latest/a6_ce_gaussian_mix0p2_ckpt3664_val200_bsz4_rep1p10_max3084_chatfix_1gpu.yaml`

Self-rollout roots:

- A5:
  `outputs/infer/recursive_detection_ce_latest/a5_instance_trie_gaussian_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_1gpu`
- A6:
  `outputs/infer/recursive_detection_ce_latest/a6_ce_gaussian_mix0p2_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_1gpu`

The self-rollout jobs were launched as one-GPU jobs because another 8-GPU
Stage-2 sweep was already occupying the machine. They used
`temperature=0.0`, `repetition_penalty=1.10`, `max_new_tokens=3084`,
`batch_size=4`, compact grammar, and `do_resize=false`.

## Rollout Detection Metrics

Guarded metrics are the better comparison surface because duplicate-control
guarding was enabled consistently in these runs.

| Run | Raw AP | Raw AP50 | Raw AP75 | Raw F1@0.50 | Guard AP | Guard AP50 | Guard AP75 | Guard F1@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ET-RMP-CE baseline | 0.425 | 0.575 | 0.448 | 0.614 | 0.412 | 0.557 | 0.435 | 0.600 |
| Random-SFT hard CE | 0.399 | 0.558 | 0.414 | 0.443 | 0.383 | 0.527 | 0.402 | 0.548 |
| A5 Gaussian/default | 0.366 | 0.550 | 0.372 | 0.574 | 0.359 | 0.539 | 0.364 | 0.573 |
| A6 Gaussian mix-0.2 | 0.400 | 0.543 | 0.418 | 0.526 | 0.388 | 0.528 | 0.407 | 0.551 |

Read:

- A5 is worse than the hard-CE ET-RMP baseline and worse than Random-SFT on AP.
- A6 recovers most of the Random-SFT-level AP and beats A5 clearly.
- ET-RMP-CE remains the best `val200` checkpoint in this comparison.
- A6 has higher AP than A5 but lower raw F1@0.50; it improves localization
  quality more than the F1-ish count/semantic balance.

## Teacher-Forced Coordinate Locality

All numbers are conditional coordinate-vocabulary probabilities over
`<|coord_0|>` through `<|coord_999|>`.

| Run | Slot | n | mass@4 | mass@8 | GT top1 | mean top1 dist | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| A5 | x1 | 598 | 0.160 | 0.286 | 0.067 | 52.7 | 4.78 |
| A5 | y1 | 1436 | 0.183 | 0.328 | 0.049 | 23.7 | 4.53 |
| A5 | x2 | 1444 | 0.227 | 0.402 | 0.053 | 12.9 | 4.19 |
| A5 | y2 | 1444 | 0.194 | 0.349 | 0.046 | 15.2 | 4.37 |
| A6 | x1 | 598 | 0.287 | 0.447 | 0.120 | 65.0 | 4.32 |
| A6 | y1 | 1436 | 0.303 | 0.497 | 0.088 | 23.9 | 4.05 |
| A6 | x2 | 1444 | 0.364 | 0.565 | 0.172 | 13.5 | 3.70 |
| A6 | y2 | 1444 | 0.289 | 0.478 | 0.102 | 14.2 | 3.94 |

For reference, the original ET-RMP-CE teacher-forced hard-CE rows had:

| Slot | ET-RMP-CE mass@4 | ET-RMP-CE mass@8 | ET-RMP-CE GT top1 | ET-RMP-CE mean top1 dist |
|---|---:|---:|---:|---:|
| x1 | 0.339 | 0.510 | 0.119 | 68.1 |
| y1 | 0.343 | 0.547 | 0.087 | 25.0 |
| x2 | 0.405 | 0.609 | 0.161 | 14.2 |
| y2 | 0.319 | 0.517 | 0.105 | 14.8 |

Read:

- A5 is not a healthy SoftCE result under this diagnostic. It is much more
  diffuse than ET-RMP-CE and A6 across every slot.
- A6 restores most of the ET-RMP-CE teacher-forced local basin, especially for
  `x2`, `y1`, and `y2`.
- A6 still trails ET-RMP-CE in mean mass within +/-4 and +/-8 bins.
- `x1` remains the special case: A6's `x1` GT-top1 rate matches ET-RMP-CE, but
  its mean top1 distance remains large because the wrong peak is often far
  away when `x1` fails.

## Self-Prefix Coordinate Locality

| Run | Slot | n | mass@4 | mass@8 | GT top1 | mean top1 dist | entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| A5 | x1 | 920 | 0.071 | 0.128 | 0.033 | 177.9 | 5.54 |
| A5 | y1 | 920 | 0.164 | 0.294 | 0.038 | 33.9 | 4.65 |
| A5 | x2 | 920 | 0.206 | 0.366 | 0.042 | 20.8 | 4.35 |
| A5 | y2 | 920 | 0.188 | 0.340 | 0.036 | 16.4 | 4.43 |
| A6 | x1 | 925 | 0.115 | 0.181 | 0.052 | 207.3 | 5.37 |
| A6 | y1 | 925 | 0.274 | 0.448 | 0.098 | 36.0 | 4.20 |
| A6 | x2 | 925 | 0.342 | 0.524 | 0.172 | 21.4 | 3.86 |
| A6 | y2 | 925 | 0.285 | 0.472 | 0.093 | 14.4 | 4.00 |

For reference, the original ET-RMP-CE self-prefix hard-CE rows had:

| Slot | ET-RMP-CE mass@4 | ET-RMP-CE mass@8 | ET-RMP-CE GT top1 | ET-RMP-CE mean top1 dist |
|---|---:|---:|---:|---:|
| x1 | 0.147 | 0.226 | 0.051 | 184.9 |
| y1 | 0.301 | 0.483 | 0.086 | 37.8 |
| x2 | 0.383 | 0.570 | 0.171 | 21.8 |
| y2 | 0.314 | 0.509 | 0.094 | 15.1 |

Read:

- A6 self-prefix behavior is meaningfully better than A5 on every slot.
- A6 is close to ET-RMP-CE on GT-top1 and mean top1 distance for `x2/y2`,
  but still lower in local mass.
- A6 does not solve the `x1` onset problem. Its self-prefix `x1` mass@4 is
  only `0.115`, and its mean top1 distance is `207.3`.
- A5 breaks badly under self-prefixes. The `x1` row becomes very diffuse, and
  even closure slots lose substantial local mass.

## Embedding Geometry

The effective-output coordinate rows remain strongly ordered in both Gaussian
checkpoints.

| Run | Effective-output Spearman | Pearson | kNN +/-4 | kNN +/-8 | kNN +/-16 | mean rank of +/-1 |
|---|---:|---:|---:|---:|---:|---:|
| ET-RMP-CE | 0.778 | 0.706 | 0.496 | 0.962 | 1.000 | 3.80 |
| Random-SFT hard CE | 0.828 | 0.803 | 0.493 | 0.955 | 1.000 | 4.33 |
| A5 Gaussian/default | 0.781 | 0.711 | 0.497 | 0.959 | 1.000 | 3.66 |
| A6 Gaussian mix-0.2 | 0.777 | 0.707 | 0.496 | 0.961 | 1.000 | 3.86 |

Interpretation:

- The coordinate-token row manifold is robust across hard CE, A5, and A6.
- A5 and A6 do not differ meaningfully in static coordinate-row adjacency.
- Static token geometry therefore cannot explain A5's weaker logit locality.
  The difference is in hidden-state/readout dynamics produced by the objective.
- The base expanded model already has moderate distance-vs-numeric correlation,
  but the coord-offset/effective rows impose a much stronger ordered kNN
  structure.

## Shape Diagnostics

Neither A5 nor A6 produces clean Gaussian logit landscapes at most coordinate
positions. The shape labels remain dominated by `multimodal_irregular`, with
some `smooth_or_irregular`, `target_local`, `wrong_object_local`, and `diffuse`
rows.

A5 teacher-forced hard-CE rows:

- `multimodal_irregular`: 4418
- `smooth_or_irregular`: 174
- `wrong_object_local`: 213
- `target_local`: 65
- `diffuse`: 52

A6 teacher-forced hard-CE rows:

- `multimodal_irregular`: 3440
- `smooth_or_irregular`: 863
- `wrong_object_local`: 287
- `target_local`: 276
- `diffuse`: 54
- `bimodal_target_other`: 2

This matters for the SoftCE decision. A Gaussian target does not make the
actual model distribution Gaussian. A6 mostly improves by making the dynamic
distribution less diffuse and more locally ranked, not by producing a textbook
single Gaussian peak.

## Diagnosis

A5 Gaussian/default is a negative or weak result for coordinate-logit locality.
It weakens local mass, increases entropy, and lowers rollout AP relative to the
hard-CE ET-RMP baseline. It also performs worse than A6 on every coordinate
slot under both teacher forcing and self-prefixes.

A6 Gaussian mix-0.2 is the more promising SoftCE shape. The hard-CE anchor
appears important: it preserves enough exact-token pressure while adding some
neighborhood awareness. Under teacher forcing, A6 recovers most of the
ET-RMP-CE local basin; under self-prefixes, it remains much stronger than A5
for `y1/x2/y2`.

However, A6 does not beat the original ET-RMP-CE baseline in this `val200`
study. ET-RMP-CE still has higher guarded AP and higher local coordinate mass
on the comparable hard-CE rows. A6 should be treated as a plausible modified
SoftCE candidate, not as evidence to replace the current CE baseline.

The embedding analysis narrows the mechanism. All checkpoints have a strong
coordinate-row numeric manifold, so "nearby coordinate tokens exist in the row
geometry" is already true. The remaining question is whether the contextual
hidden state points into the right basin. A5 says a soft target can still make
that worse; A6 says a small CE-anchored Gaussian can preserve much of the basin.

## Decision Read

Do not deprecate SoftCE globally from this evidence, but also do not promote
the broad/default Gaussian A5 setting.

Recommended state:

- Keep hard CE / ET-RMP-CE as the current production reference.
- Treat A5 Gaussian/default as not worth further direct investment unless the
  exact default mixture or variance is audited and changed.
- Keep A6-style CE-anchored Gaussian mix-0.2 as the only currently plausible
  Gaussian SoftCE candidate.
- If continuing SoftCE, focus on `x1` onset and prefix robustness, not on
  static token embedding geometry.
- Avoid overclaiming from the older A5/A6 IoU-Gibbs negative result. That note
  rules out the IoU/CIoU-Gibbs target shapes, not every Gaussian or ordinal
  coordinate objective.

## Follow-Ups

Most valuable next checks:

- Run a matched full-val or larger slice comparison for ET-RMP-CE versus A6
  mix-0.2, because `val200` is enough for mechanism but not enough for final
  promotion.
- Add a `x1`-focused analysis that separates first generated object, repeated
  description rows, and false-positive prefixes.
- Audit the A5 default Gaussian mixture/variance in the training code and
  record the actual default in resolved configs for future runs.
- Try a smaller CE-anchored Gaussian mix grid, for example `0.05/0.1/0.2`,
  while keeping the exact-token anchor and the same decode/eval surface.
- Compare dynamic hidden-state-to-row geometry: whether `h_t` moves toward the
  correct coordinate row under teacher forcing but toward another object row
  under self prefixes.

## Commands Run

Unit tests:

```bash
conda run -n ms python -m pytest tests/test_hard_ce_coord_logit_locality.py -q
```

Teacher-forced plus embedding analysis:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200.yaml \
  --stages teacher_forced,embeddings,plots,report

CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200.yaml \
  --stages teacher_forced,embeddings,plots,report
```

Self-rollout generation:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/data/CoordExp \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  conda run --no-capture-output -n ms python scripts/run_infer.py \
  --config configs/infer/recursive_detection_ce_latest/a5_instance_trie_gaussian_ckpt3664_val200_bsz4_rep1p10_max3084_chatfix_1gpu.yaml

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/data/CoordExp \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
  conda run --no-capture-output -n ms python scripts/run_infer.py \
  --config configs/infer/recursive_detection_ce_latest/a6_ce_gaussian_mix0p2_ckpt3664_val200_bsz4_rep1p10_max3084_chatfix_1gpu.yaml
```

Self-prefix analysis:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/a5_instance_trie_gaussian_ckpt3664_val200.yaml \
  --stages self_prefix,plots,report

CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/a6_ce_gaussian_mix0p2_ckpt3664_val200.yaml \
  --stages self_prefix,plots,report
```
