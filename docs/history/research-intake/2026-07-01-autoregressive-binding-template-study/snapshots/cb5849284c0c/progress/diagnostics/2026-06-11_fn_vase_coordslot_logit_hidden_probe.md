# FN Vase Coord-Slot Logit and Hidden Probe

Date: 2026-06-11

## Scope

Follow-up to the FN prefix-depth and coordinate-slot decode panel. The decode
panel identified a sharp contrast for image `139`, GT `7`, desc `vase`:

- full prefix + `desc_x1` fails: generated `[526, 497, 553, 540]`, IoU `0.1083`;
- full prefix + `desc_x1_y1` rescues: generated `[526, 468, 546, 520]`, IoU `0.6154`;
- empty prefix + `desc_x1` also rescues: generated `[526, 472, 546, 513]`, IoU `0.6516`.

This probe asks whether that failure is already visible before generation, at
the prompt-end next-coordinate distribution and hidden/logit-lens trajectory.

This is a one-case mechanistic target selector, not a general FN metric.

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

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot \
  --image-id 139 \
  --gt-idx 7 \
  --guidance-tiers desc_x1,desc_x1_y1,desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --layers all \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

## Artifacts

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot
```

Files:

```text
fn_coordslot_logit_condition_rows.jsonl
fn_coordslot_logit_layer_rows.jsonl
fn_coordslot_hidden_delta_rows.jsonl
phase4_fn_coordslot_logit_probe_summary.json
phase4_fn_coordslot_logit_probe_report.md
```

## Prompt-End Coord Distribution

The readout is taken at the prompt end before any new coordinate token is
generated.

| prefix | tier | known coords | next slot | target | target rank | target prob | top1 | top1 distance | top bins |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `desc_x1` | `[526]` | `y1` | 468 | 12 | 0.018688 | 472 | 4 | `[472, 471, 474, 479, 473]` |
| `all` | `desc_x1` | `[526]` | `y1` | 468 | 52 | 0.005097 | 501 | 33 | `[501, 498, 497, 494, 496]` |
| `all` | `desc_x1_y1` | `[526, 468]` | `x2` | 542 | 14 | 0.032332 | 548 | 6 | `[548, 546, 549, 545, 550]` |
| `all` | `desc_x1_y1_x2` | `[526, 468, 542]` | `y2` | 508 | 15 | 0.026050 | 516 | 8 | `[516, 517, 520, 513, 519]` |

## Layerwise Read

Layerwise logit-lens rows were emitted for all hidden-state tuple indices
(`174` rows total for six conditions). The strongest hidden deltas are in the
late tuple indices:

| comparison | top layer | cosine | L2 delta | left next-slot rank | right next-slot rank | left top1 | right top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix lock, `0/desc_x1` vs `all/desc_x1` | 28 | 0.94788 | 1240.9897 | 12 | 52 | 472 | 501 |
| y1 unlock, `all/desc_x1` vs `all/desc_x1_y1` | 28 | 0.62938 | 2963.2822 | 52 | 14 | 501 | 548 |

Layer `28` is also the best layerwise logit-lens readout for the prompt-end
`y1` target in both prefix states:

- prefix `0`, `desc_x1`: best target rank `12`, target prob `0.01868`, top1
  `472`;
- prefix `all`, `desc_x1`: best target rank `52`, target prob `0.005062`,
  top1 `498`.

## Interpretation

The full-prefix vase failure is not only a bad generation tail. It is already
present at the prompt-end coordinate distribution:

- Empty prefix + `desc_x1` places target `y1=468` near the top local basin
  (`rank=12`, top1 `472`).
- Full prefix + `desc_x1` shifts the next-token basin downward to
  `y1 ~= 494-501`; target `468` falls to `rank=52`.
- Supplying `y1=468` manually moves the next slot (`x2`) back into a near-target
  basin (`target rank=14`, top1 `548`), matching the downstream decode rescue.

Mechanistically, this is the cleanest current FN path:

1. The full autoregressive prefix has created a vertical coordinate basin around
   the previously emitted/local vase proposal.
2. The model can still complete a plausible target box if the correct `y1`
   token is forced.
3. Therefore the miss is not simple category blindness or pure x-origin
   blindness; it is a coordinate-slot basin failure visible before generation.

## Next Hook

The most promising causal target is the full-prefix `desc_x1` prompt at the
next `y1` site. A good next intervention would patch late-layer residual or
attention components from the empty-prefix `desc_x1` condition into the
full-prefix `desc_x1` condition and measure whether the `y1=468` rank/top1
recovers away from the `501` basin.

Recommended first patch scope:

- source: prefix `0`, tier `desc_x1`, prompt-end hidden state;
- target: prefix `all`, tier `desc_x1`, prompt-end hidden state;
- metric: target `y1=468` rank/prob, top1 bin, and mass within radius 8;
- candidate layers: late tuple indices around `24-28`, with layer `28` as the
  first smoke target.

## Guardrails

- One image/object only: image `139`, GT `7`, `vase`.
- This is a prompt-end readout over fixed partial prefixes; it does not prove
  the same mechanism for every FN.
- `desc_x1_y1` and `desc_x1_y1_x2` use target coordinates by construction. They
  reveal basin accessibility and downstream completion, not an eval-time
  decoding method.
- Hidden-delta comparisons are between different prompt texts, so they identify
  candidate layers/sites, not yet causal sufficiency.
