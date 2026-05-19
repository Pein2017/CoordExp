---
title: Hard-CE Coordinate-Logit And Embedding Locality Diagnosis
date: 2026-05-18
status: active-reference
owner: codex
depends_on:
  - outputs/analysis/hard_ce_coord_logit_locality/ckpt3664_val200/report.md
  - outputs/analysis/hard_ce_coord_logit_locality/ckpt3664_val200/summary.json
  - outputs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200/report.md
  - outputs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200/summary.json
  - outputs/analysis/hard_ce_coord_logit_locality/comparison_random_sft_vs_et_rmp_ckpt3664_val200.md
  - outputs/analysis/hard_ce_coord_logit_locality/comparison_random_sft_vs_et_rmp_ckpt3664_val200.json
---

# Hard-CE Coordinate-Logit And Embedding Locality Diagnosis

## Why This Note Exists

This note records the `val200` mechanism diagnosis for coordinate-token
probability landscapes under hard cross-entropy supervision. The central
question was whether explicit coordinate SoftCE is necessary, or whether hard CE
already induces useful local probability basins around GT coordinate tokens.

The study has two axes:

- output-distribution locality: raw model logits over `<|coord_0|>` through
  `<|coord_999|>` at coordinate prediction positions;
- token-row locality: static coordinate-token row geometry in input embedding,
  output head, coord-offset, and effective post-offset rows.

This is a diagnostic mechanism note, not a full-validation benchmark record.
All headline numbers below are `val200` only.

## Scope

Dataset:

`public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

Primary ET-RMP-CE checkpoint:

`outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`

Pure random-order hard-CE SFT checkpoint:

`outputs/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664`

Analysis configs:

- `configs/analysis/hard_ce_coord_logit_locality/ckpt3664_val200.yaml`
- `configs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200.yaml`

Analysis roots:

- ET-RMP-CE:
  `outputs/analysis/hard_ce_coord_logit_locality/ckpt3664_val200`
- Random-SFT-CE:
  `outputs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200`
- Comparison:
  `outputs/analysis/hard_ce_coord_logit_locality/comparison_random_sft_vs_et_rmp_ckpt3664_val200.md`

Self-rollout sources:

- ET-RMP-CE:
  `outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu`
- Random-SFT-CE:
  `outputs/infer/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_8gpu`

## Procedure

Teacher-forced prefixes:

- render compact-full GT assistant text;
- preserve slot order `x1, y1, x2, y2`;
- extract logits at `logits[p - 1]` for the coordinate teacher token at
  `input_ids[p]`;
- analyze full coordinate vocabulary `<|coord_0|>` through `<|coord_999|>`.

Self-prefix prefixes:

- use each checkpoint's own generated compact rollout artifact;
- cut generated prefixes at complete compact-row boundaries;
- strip chat stop markers;
- force the GT continuation for the target object after the generated prefix;
- label prefix quality from match artifacts as `empty_prefix`, `clean_prefix`,
  or `fp_prefix`.

Probability views:

- `coord_vocab_mass`: full-vocab softmax mass assigned to the 1000 coordinate
  tokens;
- `p_cond`: coordinate-only conditional distribution, normalized over the 1000
  coordinate bins.

Locality claims use `p_cond`; `coord_vocab_mass` is a leakage diagnostic.

Embedding analysis:

- extract coordinate rows for base input, base output, coord-offset, effective
  input, and effective output surfaces;
- compute pairwise Euclidean and cosine distances over coordinate bins;
- test whether numeric neighbors are geometric neighbors.

## Validity Notes

The Random-SFT teacher-forced analysis uses:

`teacher_target_kind_mode: all_hard_ce`

That correction matters. The ET-RMP checkpoint has recursive-detection sidecar
states where some `x1` rows are intentionally `trie_multi_positive` rather than
one-hot hard CE. The pure SFT baseline does not have those target kinds, so all
teacher-forced coordinate slots should be counted as hard CE.

The self-prefix extractor was also corrected during this study. Confidence
sidecars can skip ambiguous generated objects. Cutting at the Nth valid
confidence object can accidentally include more than `depth` complete generated
rows. The fixed extractor enforces exactly `depth` complete compact rows after
token-boundary cutting.

These corrections are reflected in:

- `src/analysis/hard_ce_coord_logit_locality.py`
- `tests/test_hard_ce_coord_logit_locality.py`

## Aggregate Results

| Slice | ET-RMP n | ET-RMP mass@4 | ET-RMP mass@8 | ET-RMP mass@16 | ET-RMP top1 dist | ET-RMP GT top1 | Random-SFT n | Random-SFT mass@4 | Random-SFT mass@8 | Random-SFT mass@16 | Random-SFT top1 dist | Random-SFT GT top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher hard-CE only | 4922 | 0.354 | 0.552 | 0.744 | 24.067 | 0.118 | 5776 | 0.299 | 0.473 | 0.656 | 58.502 | 0.109 |
| Teacher all coord targets | 5776 | 0.314 | 0.490 | 0.662 | 59.325 | 0.106 | 5776 | 0.299 | 0.473 | 0.656 | 58.502 | 0.109 |
| Self hard-CE | 3612 | 0.286 | 0.447 | 0.612 | 64.904 | 0.100 | 3748 | 0.279 | 0.439 | 0.612 | 65.388 | 0.107 |

Interpretation:

- ET-RMP looks meaningfully better than Random-SFT when restricted to ET-RMP's
  hard-CE coordinate rows.
- When ET-RMP's broad trie/multipositive coordinate states are included, global
  teacher-forced locality is much closer to Random-SFT.
- Under self-prefixes, the two checkpoints are also close in aggregate.

## Slot Results

| Slice | ET-RMP mass@4 | ET-RMP top1 dist | Random-SFT mass@4 | Random-SFT top1 dist |
|---|---:|---:|---:|---:|
| Teacher `x1` | 0.339 | 68.114 | 0.155 | 183.648 |
| Teacher `y1` | 0.343 | 24.979 | 0.331 | 22.743 |
| Teacher `x2` | 0.405 | 14.190 | 0.392 | 13.113 |
| Teacher `y2` | 0.319 | 14.796 | 0.316 | 14.502 |
| Self `x1` | 0.147 | 184.910 | 0.122 | 193.331 |
| Self `y1` | 0.301 | 37.773 | 0.304 | 33.551 |
| Self `x2` | 0.383 | 21.797 | 0.376 | 20.483 |
| Self `y2` | 0.314 | 15.136 | 0.311 | 14.188 |

Main symptom:

`x1` is the fragile slot. The pure random-SFT baseline is especially weak at
teacher-forced `x1`: only `0.155` conditional mass within +/-4 bins and mean
top-1 distance `183.648`. Its `y1`, `x2`, and `y2` behavior is close to ET-RMP.

This supports the earlier mechanism read that object-row onset is the hard
commitment point. The model can know the coordinate token manifold, but still
fail to point the hidden state into the right horizontal basin when starting a
new object row.

## Shape Diagnostics

The coordinate distributions are not clean Gaussians. Both checkpoints show
many rows labeled `multimodal_irregular`, with smaller numbers of
`target_local`, `wrong_object_local`, `diffuse`, and `sharp_delta` rows.

Random-SFT teacher-forced hard-CE shape counts:

- `multimodal_irregular`: 3734
- `smooth_or_irregular`: 991
- `diffuse`: 394
- `wrong_object_local`: 325
- `target_local`: 257
- `sharp_delta`: 65
- `bimodal_target_other`: 10

ET-RMP teacher-forced hard-CE shape counts:

- `multimodal_irregular`: 2935
- `smooth_or_irregular`: 1233
- `target_local`: 340
- `wrong_object_local`: 302
- `sharp_delta`: 65
- `diffuse`: 45
- `bimodal_target_other`: 2

The model has local structure, but the learned distribution is often
multimodal or irregular rather than a simple one-peak Gaussian around GT.

## Embedding Geometry

Effective-output coordinate rows are the actual post-offset rows used by the
LM head surface for coordinate-token scoring. The study found strong numeric
adjacency in both checkpoints:

| Checkpoint | Effective-output distance-vs-numeric Spearman | kNN recall +/-4 | kNN recall +/-8 | kNN recall +/-16 | Contiguous cluster purity |
|---|---:|---:|---:|---:|---:|
| ET-RMP-CE | 0.778 | 0.496 | 0.962 | 1.000 | 1.000 |
| Random-SFT-CE | 0.828 | 0.493 | 0.955 | 1.000 | 1.000 |

This means the static coordinate-token rows form a strong numeric manifold in
both models. Nearby coordinate bins are geometrically close and distant bins
tend to be farther away.

The Random-SFT checkpoint has slightly higher effective-output Spearman than
ET-RMP, while still having much worse dynamic `x1` locality. Therefore the
`x1` weakness is not caused by failure to learn coordinate-token adjacency.
The bottleneck is contextual hidden-state routing at object onset.

## Diagnosis

Hard CE is not merely learning isolated one-hot coordinate rows. It produces:

- high coordinate-vocabulary mass at coordinate positions;
- strong effective coordinate-token embedding adjacency;
- useful local output distributions for `y1`, `x2`, and `y2`;
- especially strong closure-coordinate locality for `x2`.

However, hard CE is not uniformly sufficient:

- random-shuffled pure SFT has weak `x1` locality even under teacher forcing;
- self-prefixes further weaken `x1` for both checkpoints;
- distribution shapes are often multimodal or irregular, not clean Gaussian
  basins;
- strong static token-row adjacency does not guarantee correct dynamic
  grounding at coordinate prediction time.

The most likely mechanism is:

1. Coord-offset training creates a robust ordered coordinate-token manifold.
2. Once the hidden state is already near the correct object-side basin,
   hard CE can place mass locally around nearby bins.
3. At object-row onset, especially `x1`, hidden states often encode an
   ambiguous object/posterior or wrong-object basin.
4. ET-RMP improves the hard-CE subset of `x1` behavior, but its broad
   trie/multipositive states should not be interpreted as one-hot hard-CE
   coordinate rows.

## SoftCE Decision Read

Do not keep SoftCE merely to teach coordinate-token adjacency. The effective
coordinate rows already learn adjacency under hard CE.

Do not fully deprecate all soft or ordinal coordinate supervision based on this
evidence. The remaining problem is targeted:

- `x1` onset;
- prefix drift and false-positive prefixes;
- repeated or crowded object settings;
- hidden-state ambiguity rather than static row geometry.

The most promising follow-up is not a blanket Gaussian target over every
coordinate slot. A better direction is an onset-aware or prefix-dynamics-aware
objective that helps route the hidden state into the correct object/side basin
before or at `x1`.

## Follow-Ups

Highest-value follow-up experiments:

1. Split metrics by object ordinal and repeated-description count, focused on
   `x1`.
2. Compare clean self-prefixes against FP-prefixes with the same target object
   and coordinate bin.
3. Add targeted `x1` hidden-state probes to test whether wrong peaks match
   another GT object in the same image.
4. Test a slot-specific soft or ordinal objective only for onset coordinates,
   rather than all four bbox slots.
5. Re-run the same diagnostics on full validation only after the `val200`
   mechanism questions stabilize.

## Verification

Commands completed during this study:

```bash
conda run -n ms python -m pytest tests/test_hard_ce_coord_logit_locality.py -q
conda run -n ms python -m py_compile src/analysis/hard_ce_coord_logit_locality.py scripts/analysis/run_hard_ce_coord_logit_locality.py
CUDA_VISIBLE_DEVICES=0 conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200.yaml \
  --stages teacher_forced,self_prefix,embeddings,plots,report
CUDA_VISIBLE_DEVICES=0 conda run -n ms python scripts/analysis/run_hard_ce_coord_logit_locality.py \
  --config configs/analysis/hard_ce_coord_logit_locality/random_sft_ckpt3664_val200.yaml \
  --stages teacher_forced,plots,report
```

Final helper tests:

- `10 passed`
