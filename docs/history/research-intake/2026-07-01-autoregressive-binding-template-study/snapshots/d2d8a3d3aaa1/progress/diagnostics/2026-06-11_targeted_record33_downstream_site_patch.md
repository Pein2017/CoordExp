# Targeted Record-33 Downstream-Site Patch

Date: 2026-06-11

## Scope

This slice follows the source-component expansion for the cross-phase top case:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`

Prior targeted result:

- duplicate-basin key+value state patch recovery: `0.010027`
- whole-head key+value state patch recovery: `0.013037`
- duplicate-basin route/content route-only output patch recovery: `0.018168`

The question here is whether the remaining gap appears downstream of
attention score/value assembly, especially after `o_proj`, residual addition,
or norm.

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Downstream-site output:

```text
whole_head_downstream_site_patch_layer17_head1/route_content_patch_rows.jsonl
whole_head_downstream_site_patch_layer17_head1/phase4_route_content_patch_summary.json
whole_head_downstream_site_patch_layer17_head1/route_content_patch_reduced_summary.json
whole_head_downstream_site_patch_layer17_head1/route_content_patch_reduced_report.md
```

Reduced synthesis:

```text
targeted_downstream_site_synthesis_layer17_head1/targeted_downstream_site_synthesis_summary.json
targeted_downstream_site_synthesis_layer17_head1/targeted_downstream_site_synthesis_report.md
```

Run scale:

- target cases: `5`
- rows: `492`
- reduced groups: `42`
- layer/head: `17/1`
- component: `whole_head`
- direction: `masked_to_control`
- sites: `self_attn_output`, `post_attention_residual`,
  `post_attention_norm`
- effect kinds: `total_delta`, `route_delta_masked_values`
- GPU used: `CUDA_VISIBLE_DEVICES=0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_route_content_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/whole_head_downstream_site_patch_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-sites self_attn_output,post_attention_residual,post_attention_norm \
  --patch-components whole_head \
  --effect-kinds total_delta,route_delta_masked_values
```

## Result

Main case downstream sites:

| site | effect | probability recovery | probability damage | rank recovery | abs-error recovery | mass16 recovery |
|---|---|---:|---:|---:|---:|---:|
| `self_attn_output` | `total_delta` | 0.012785 | -0.000947 | 18.5000 | -7.3170 | 0.217755 |
| `self_attn_output` | `route_delta_masked_values` | 0.014328 | 0.000596 | 19.8333 | -7.5994 | 0.234689 |
| `post_attention_residual` | `total_delta` | 0.007498 | -0.006234 | 13.7500 | -4.9413 | 0.142660 |
| `post_attention_residual` | `route_delta_masked_values` | 0.006914 | -0.006818 | 14.8333 | -4.3245 | 0.130616 |
| `post_attention_norm` | `total_delta` | -0.005306 | -0.019038 | -28.5000 | 12.2589 | -0.142663 |
| `post_attention_norm` | `route_delta_masked_values` | -0.004277 | -0.018009 | -20.8333 | 13.8124 | -0.113182 |

Main case references:

| probe | probability recovery | rank recovery | abs-error recovery |
|---|---:|---:|---:|
| whole-head key+value source-state patch | 0.013037 | 18.7500 | -7.2255 |
| duplicate-basin route/content route-only output | 0.018168 | 19.3333 | -9.8141 |
| duplicate-basin route/content total output | 0.018371 | 19.4167 | -9.8000 |

Top cross-case downstream groups:

| rank | checkpoint | record | phase | site | effect | probability recovery |
|---:|---|---:|---|---|---|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `self_attn_output` | `route_delta_masked_values` | 0.014328 |
| 2 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `self_attn_output` | `total_delta` | 0.012785 |
| 3 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `self_attn_output` | `total_delta` | 0.009036 |
| 4 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `self_attn_output` | `route_delta_masked_values` | 0.008090 |
| 5 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `post_attention_residual` | `total_delta` | 0.007498 |

## Interpretation

The missing effect does not appear after residual addition or post-attention
norm.

For the main record-33 failure:

- `self_attn_output` is the strongest downstream patch site.
- `post_attention_residual` is substantially weaker than `self_attn_output`.
- `post_attention_norm` is actively harmful in the main case.
- `self_attn_output` whole-head route patch (`0.014328`) is only modestly
  above whole-head key+value source-state patch (`0.013037`), but remains below
  the earlier duplicate-basin route/content route-only output patch
  (`0.018168`).

This narrows the mechanism again:

- The remaining gap is not rescued by post-attention residual addition.
- The remaining gap is not a norm-side coordinate-slot amplification.
- The best downstream site is still the attention output itself, which points
  back into how the route/content vector is constructed and projected, rather
  than later layer plumbing.

The surprising contrast is that the earlier duplicate-basin route/content
patch was stronger than this whole-head route/content patch. That suggests the
bucket-level projected effect may not be additive in the naive way: including
the full head may introduce cancelling directions, while the duplicate-basin
route subcomponent is unusually aligned with the coordinate-basin repair.

## Next Deterministic Step

Run a focused projected-direction decomposition for the same target rows:

- compare duplicate-basin, whole-head, non-region complement, near-ring, and
  far-background projected vectors against the coordinate target direction;
- inspect cancellation between duplicate-basin repair and other whole-head
  components at `self_attn_output`;
- preserve the `post_y1/pre_x2` slot focus.

The concrete question is whether whole-head underperforms duplicate-basin
because other source buckets cancel the coordinate-target direction in
post-`o_proj` residual space.

## Verification

- GPU run completed with:
  `route_content_patch_row_count=492`, `checkpoint_count=3`,
  `replay_case_count=5`,
  `patch_sites=["self_attn_output","post_attention_residual","post_attention_norm"]`,
  `effect_kinds=["total_delta","route_delta_masked_values"]`.
- Reducer completed with:
  `row_count=492`, `group_count=42`.
- Reduced synthesis completed with:
  `main_downstream_groups=6`.
- No code changes were made in this slice.
