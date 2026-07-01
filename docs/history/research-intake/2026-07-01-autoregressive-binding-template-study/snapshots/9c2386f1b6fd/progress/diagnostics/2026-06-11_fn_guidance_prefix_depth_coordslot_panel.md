# FN Guidance Prefix-Depth and Coord-Slot Panel

Date: 2026-06-11

## Scope

Follow-up to the wrong-control `x1` stratified FN guidance panel. This panel
uses the same six selected no-aligner parent false-negative cases, but varies:

- existing autoregressive prefix depth: `0` predicted objects vs `all` predicted
  objects;
- coordinate guidance tier: `desc_only`, `desc_x1`, `desc_x1_y1`,
  `desc_x1_y1_x2`, and `desc_x1_wrong_control`.

The question is whether misses are blocked by the late autoregressive state,
whether `x1` alone is enough, and whether adding subsequent coordinate slots
collapses the local coordinate basin.

This is still a deterministic smoke panel, not a rescue-rate benchmark.

## Inputs

Guidance rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl
```

Scored rollout:

```text
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl
```

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Decode command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_decode_smoke.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1 \
  --per-bucket-cap 1 \
  --stratify-wrong-control-distance \
  --guidance-tiers desc_only,desc_x1,desc_x1_y1,desc_x1_y1_x2,desc_x1_wrong_control \
  --prefix-object-limits 0,all \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

## Artifacts

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1
```

Files:

```text
fn_guidance_decode_smoke_rows.jsonl
fn_guidance_decode_smoke_case_rows.jsonl
phase4_fn_guidance_decode_smoke_summary.json
phase4_fn_guidance_decode_smoke_report.md
```

## Results

Scope:

- Selected FN cases: `6`
- Decode rows: `60`
- Case rows: `12`
- Prefix limits: `0`, `all`

Primary rescue success by prefix and tier:

| prefix | desc_only | desc_x1 | desc_x1_y1 | desc_x1_y1_x2 | desc_x1_wrong_control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | 0/6 | 5/6 | 5/6 | 5/6 | 1/6 |
| `all` | 0/6 | 4/6 | 5/6 | 5/6 | 2/6 |

Case outcome counts by prefix, using the legacy case classifier over
`desc_only`, `desc_x1`, and `desc_x1_wrong_control`:

| prefix | clean desc-x1 specific | both x1 and wrong-control success | no rescue |
| ---: | ---: | ---: | ---: |
| `0` | 4 | 1 | 1 |
| `all` | 2 | 2 | 2 |

Key case transitions:

| image | gt | desc | prefix=0 read | prefix=all read |
| ---: | ---: | --- | --- | --- |
| 139 | 7 | vase | `desc_x1` rescues at IoU 0.6516; wrong-control fails at 0.4808 | `desc_x1` fails at 0.1083, but `desc_x1_y1` rescues at 0.6154 |
| 139 | 8 | microwave | `desc_x1` rescues at 0.5685; wrong-control fails | `desc_x1` rescues at 0.6266; wrong-control fails |
| 139 | 11 | chair | `desc_x1` rescues; wrong-control IoU is 0.9065 but is duplicate-rejected | `desc_x1` and wrong-control both rescue |
| 139 | 15 | chair | target and wrong-control both rescue | target and wrong-control both rescue |
| 139 | 17 | book | no tier rescues; `desc_x1_y1_x2` reaches only 0.1818 | no tier rescues; `desc_x1_y1_x2` reaches only 0.2091 |
| 724 | 3 | truck | `desc_x1` cleanly rescues; far wrong-control fails | `desc_x1` cleanly rescues; far wrong-control fails |

## Interpretation

The panel separates three mechanisms that were blended in the previous smoke:

1. Prefix-state lock-in: the vase case is recoverable with an empty prediction
   prefix but not with the full prefix under `desc_x1`. The full prefix has
   already pushed the model into a local vertical/box-history basin; adding
   `y1` breaks that basin and rescues the case.
2. Coordinate-slot basin collapse: for all recoverable cases, `desc_x1_y1` and
   `desc_x1_y1_x2` make the remaining coordinates much more deterministic.
   They reach 5/6 primary rescue at both prefix depths.
3. Hard residual FN: the book case remains unrescued even after `x1,y1,x2`.
   The model follows the supplied left/top/right edge but predicts a short box
   height around `y2 ~= 736-739` against target `826`, suggesting missing
   extent/visibility evidence rather than simple category or x-origin failure.

This supports the working hypothesis that some false negatives are not visual
blindness. They are late-prefix or coordinate-basin failures where partial
coordinate context can unlock the missing object. At the same time, the book
case remains a useful counterexample: not every FN is recoverable through
language-side coordinate guidance.

## Next Mechanistic Hook

The most promising hidden-state/attention target is now the transition between
`desc_x1` and `desc_x1_y1` on the vase case under `prefix=all`.

Concrete contrast:

- Same image, desc, target, checkpoint, and full prefix.
- `desc_x1`: generated `[526, 497, 553, 540]`, IoU `0.1083`.
- `desc_x1_y1`: generated `[526, 468, 546, 520]`, IoU `0.6154`.

That isolates a single additional coordinate token (`y1=468`) that moves the
decoder from a wrong local vertical basin into the target basin. This should be
a better hidden-state or attention patch target than a broad FN aggregate.

## Guardrails

- Six-case smoke only.
- Prefix `0` removes all raw predicted object rows but keeps the same image,
  prompt template, target desc, and continuation style.
- Prefix `all` appends after the full raw prediction prefix from the existing
  rollout, so it tests late continuation from an already formed autoregressive
  state.
- `desc_x1_y1` and `desc_x1_y1_x2` use target coordinate slots by construction;
  they test basin accessibility and downstream coordinate completion, not an
  intervention that would be available at evaluation time.
- `desc_x1_wrong_control` uses only the wrong-control `x1`; it is not a full
  wrong-box control for later coordinate slots.
