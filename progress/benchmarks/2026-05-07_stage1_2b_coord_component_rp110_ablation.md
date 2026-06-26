---
title: Stage-1 2B Coord-Component RP1.10 Ablation Benchmark
date: 2026-05-07
status: completed-benchmark
topics: [stage1, coord-token, coord-components, rp1.10, val200, lvis-proxy, adapter-eval]
tags: [benchmarks, stage1, coord-token, loss-ablation, repetition-penalty, val200]
summary: Records the Stage-1 2B coord-token component ablation results on COCO 1024 LVIS-proxy val200, with special focus on the rp=1.10 decode comparison and the hard-CE-only rescue.
---

# Stage-1 2B Coord-Component RP1.10 Ablation Benchmark (2026-05-07)

This note archives the important results from the Stage-1 2B coord-token loss
component ablation sequence.

The key update is:

```text
hard_ce_only + repetition_penalty=1.10 is now the strongest current val200
anchor among the compared coord-component checkpoints.
```

This supersedes the earlier interim read where `soft_ce_only` and
`smooth_l1_hard_ce` looked like the two most promising points before
`hard_ce_only` had been re-evaluated at `rp=1.10`.

## Scope

Shared benchmark contract:

- model family:
  `Qwen3-VL-2B-Instruct-coordexp`
- checkpoint loading:
  adapter checkpoint loading, not merged full-model export
- dataset:
  `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`
- slice:
  first 200 validation examples (`val200`)
- coordinate surface:
  coord tokens, not raw-text integer coordinates
- prompt:
  `prompt_variant=coco_80`
- object serialization:
  `object_field_order=desc_first`, `object_ordering=sorted`
- bbox format:
  `xyxy`
- decoding:
  `temperature=0.0`, `top_p=0.9`, `max_new_tokens=3084`,
  `batch_size=4`
- scoring:
  confidence post-op with `bbox_logprob_confidence_exp`
- evaluation:
  proxy bundle evaluation over `coco_real`, `coco_real_strict`, and
  `coco_real_strict_plausible`
- headline metric:
  `coco_real` `bbox_AP`

`coco_real` is the standard COCO-aligned headline. The strict and plausible
views are additive proxy-supervision analysis views and should not be compared
as standard COCO scores.

## Included Runs

### hard_ce_only

Training checkpoint:

- `output/stage1_2b/ablation/coord_components/coord_token_hard_ce/ablation_stage1_2b_coord_components_coord_token_hard_ce/v1-20260424-085234/checkpoint-684`

Eval artifacts:

- `rp=1.05`:
  [coord_components_2b_coord_token_hard_ce_val200_adapter](/data/CoordExp/output/infer/coord_components_2b/coord_components_2b_coord_token_hard_ce_val200_adapter)
- `rp=1.10`:
  [coord_components_2b_coord_token_hard_ce_val200_adapter_rp110](/data/CoordExp/output/infer/coord_components_2b/coord_components_2b_coord_token_hard_ce_val200_adapter_rp110)

The `rp=1.10` hard-CE run was launched as four manual contiguous shards and
merged back into the standard run directory:

- shard0: val indices `0-49`
- shard1: val indices `50-99`
- shard2: val indices `100-149`
- shard3: val indices `150-199`

### soft_ce_only

Training checkpoint:

- `output/stage1_2b/ablation/coord_components/soft_ce_only/ablation_stage1_2b_coord_components_soft_ce_only/v0-20260430-145942/checkpoint-684`

Eval artifact:

- [coord_components_2b_soft_ce_only_val200_adapter_rp110](/data/CoordExp/output/infer/coord_components_2b/coord_components_2b_soft_ce_only_val200_adapter_rp110)

### soft_ce_hard_ce

Training checkpoint:

- `output/stage1_2b/ablation/coord_components/soft_ce_hard_ce/ablation_stage1_2b_coord_components_soft_ce_hard_ce/v0-20260430-145941/checkpoint-684`

Eval artifact:

- [coord_components_2b_soft_ce_hard_ce_val200_adapter_rp110](/data/CoordExp/output/infer/coord_components_2b/coord_components_2b_soft_ce_hard_ce_val200_adapter_rp110)

### smooth_l1_hard_ce

Training checkpoint:

- `output/stage1_2b/ablation/coord_components/smooth_l1_hard_ce/ablation_stage1_2b_coord_components_smooth_l1_hard_ce/v0-20260430-145941/checkpoint-684`

Eval artifact:

- [coord_components_2b_smooth_l1_hard_ce_val200_adapter_rp110](/data/CoordExp/output/infer/coord_components_2b/coord_components_2b_smooth_l1_hard_ce_val200_adapter_rp110)

## Main COCO Real Scoreboard

All rows below are `val200`, `coco_real`, adapter-loaded, coord-token outputs.

| Run | RP | AP | AP50 | AP75 | AR100 | F1@0.50 | P@0.50 | R@0.50 | Pred total | Infer errors | Eval degenerate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hard_ce_only` | `1.05` | `0.3307` | `0.4242` | `0.3356` | `0.3770` | `0.4617` | `0.4169` | `0.5173` | `1805` | `4` | `13` |
| `hard_ce_only` | `1.10` | `0.3901` | `0.5042` | `0.3954` | `0.4386` | `0.5582` | `0.5425` | `0.5748` | `1555` | `0` | `0` |
| `soft_ce_only` | `1.10` | `0.3859` | `0.5043` | `0.4046` | `0.4355` | `0.5390` | `0.5102` | `0.5713` | `1651` | `0` | `0` |
| `smooth_l1_hard_ce` | `1.10` | `0.3853` | `0.4962` | `0.4019` | `0.4316` | `0.5546` | `0.5609` | `0.5485` | `1441` | `2` | `2` |
| `soft_ce_hard_ce` | `1.10` | `0.3455` | `0.4451` | `0.3625` | `0.3898` | `0.5454` | `0.5837` | `0.5118` | `1306` | `0` | `2` |

## Hard-CE Decode Delta

Changing only the hard-CE decode from `rp=1.05` to `rp=1.10` produced a large
improvement:

| Metric | Delta: `rp1.10 - rp1.05` |
|---|---:|
| AP | `+0.0594` |
| AP50 | `+0.0800` |
| AP75 | `+0.0598` |
| AR100 | `+0.0616` |
| F1@0.50 | `+0.0965` |
| Precision@0.50 | `+0.1256` |
| Recall@0.50 | `+0.0575` |
| Pred total | `-250` |
| Infer errors | `-4` |
| Eval degenerate | `-13` |

Interpretation:

- `rp=1.10` does not merely suppress outputs.
- It reduces duplicate or degenerate prediction pressure while increasing
  recall.
- This makes hard-CE substantially healthier than it looked under the earlier
  `rp=1.05` decode.

## Hard-CE RP1.10 Proxy Views

For the selected hard-CE `rp=1.10` run:

| View | AP | AP50 | AP75 | AR100 | F1@0.50 full micro |
|---|---:|---:|---:|---:|---:|
| `coco_real` | `0.3901` | `0.5042` | `0.3954` | `0.4386` | `0.5582` |
| `coco_real_strict` | `0.3811` | `0.4924` | `0.3878` | `0.4263` | `0.5564` |
| `coco_real_strict_plausible` | `0.3716` | `0.4815` | `0.3777` | `0.4164` | `0.5575` |

The monotonic AP decrease across proxy-expanded views is expected: the views
add more GT objects and therefore evaluate against a larger target set.

## Prediction Health

Hard-CE `rp=1.10` inference summary:

```json
{
  "total_read": 200,
  "total_emitted": 200,
  "errors_total": 0
}
```

Hard-CE `rp=1.10` confidence post-op summary:

```json
{
  "total_samples": 200,
  "total_pred_objects": 1555,
  "kept_pred_objects": 1555,
  "dropped_pred_objects": 0,
  "kept_fraction": 1.0
}
```

Compared with hard-CE `rp=1.05`, the `rp=1.10` decode removed:

- all `invalid_geometry` inference errors in this slice
- all `degenerate` evaluator counters in `coco_real`
- many extra predictions, while improving recall

This is the most important diagnostic signal in the benchmark: `rp=1.10`
cleans the output distribution rather than simply making it conservative.

## Runtime Notes

The hard-CE `rp=1.10` inference was launched on four GPUs as four contiguous
manual shards because the user requested a four-GPU run. Shard runtimes:

| Shard | Runtime | Notes |
|---|---:|---|
| shard0 | `20:08` | slowest shard, had long-generation samples |
| shard1 | `12:27` | moderate long tail |
| shard2 | `09:01` | healthy |
| shard3 | `08:54` | healthy |

Observed behavior:

- GPU utilization stayed high during the long shard0 period.
- The delay was not a hang; it was long autoregressive generation in the first
  shard.
- GPU0 KV/cache memory rose much higher than other shards before shard0
  completed.

Operational caveat:

- The merged hard-CE `rp=1.10` artifact restores the original val200 order.
- However, the manual four-shard launch changes batch boundaries relative to a
  single-process val200 run.
- For paper-table strictness, either re-run top candidates with the same
  sharded launcher or re-run the final selected top candidates in one unified
  inference mode.

## Current Decision

The current val200 decision ranking is:

1. `hard_ce_only + rp=1.10`
2. `soft_ce_only + rp=1.10`
3. `smooth_l1_hard_ce + rp=1.10`
4. `soft_ce_hard_ce + rp=1.10`

Recommended interpretation:

- Use `rp=1.10` as the decode default for this coord-token Stage-1 family.
- Treat `hard_ce_only + rp=1.10` as the strongest current clean Stage-1 2B
  coord-token anchor.
- Keep `soft_ce_only + rp=1.10` as a strong research comparator because it is
  close on AP and highest on AP75 in this slice.
- Keep `smooth_l1_hard_ce + rp=1.10` as the geometry-regularized comparator
  because it remains near the top and has strong precision.
- Do not prioritize `soft_ce_hard_ce` as the main path right now; its precision
  is high but AP and recall trail the other candidates.

## Configs Added For Hard-CE RP1.10

These configs define the canonical post-merge hard-CE `rp=1.10` evaluation
surface:

- [configs/infer/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110.yaml](/data/CoordExp/configs/infer/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110.yaml)
- [configs/postop/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110.yaml](/data/CoordExp/configs/postop/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110.yaml)
- [configs/eval/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110_bundle.yaml](/data/CoordExp/configs/eval/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110_bundle.yaml)

## Verification Commands

The canonical hard-CE `rp=1.10` post-op and eval commands were:

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110.yaml
```

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_proxy_detection_bundle.py \
  --config configs/eval/coco_1024/coord_components_2b/coord_token_hard_ce_val200_adapter_rp110_bundle.yaml
```

Artifacts checked:

- `gt_vs_pred.jsonl`: 200 lines
- `pred_token_trace.jsonl`: 200 lines
- `gt_vs_pred_scored.jsonl`: 200 lines
- `summary.json`: exists
- `confidence_postop_summary.json`: exists
- `proxy_eval_bundle_summary.json`: exists
- `eval_coco_real/metrics.json`: exists
- `eval_coco_real/per_image.json`: exists

## Follow-Up

Before promoting this beyond val200:

- re-run the top candidates with identical launch shape if strict ordering is
  required
- optionally run a larger validation slice only after the loss-family shortlist
  is frozen
- inspect representative samples where `soft_ce_only` wins AP75 but hard-CE
  wins AP/F1, because that contrast may reveal whether soft CE mainly improves
  localization quality or proposal diversity
