# Targeted Record-33 Key+Value Component Expansion

Date: 2026-06-11

## Scope

This slice tests whether the remaining gap after duplicate-basin key+value
patching comes from source competition outside the duplicate-basin bucket.

Target case:

- checkpoint: `none_latest_ckpt32`
- record: `33`
- phase: `post_y1/pre_x2`
- layer/head: `17/1`

Prior targeted result:

- route/content route-only output patch recovery: `0.018168`
- score-bias per-source proxy recovery: `0.010221`
- direct duplicate-basin key+value-state patch recovery: `0.010027`

This run expands `key_value_state_control` from `duplicate_basin` to:

- `duplicate_basin`
- `visual_near_ring`
- `visual_far_background`
- `non_region_complement`
- `whole_head`

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Component-expansion output:

```text
key_value_component_expansion_layer17_head1/attention_score_bias_patch_rows.jsonl
key_value_component_expansion_layer17_head1/phase4_attention_score_bias_patch_summary.json
```

Reduced synthesis:

```text
targeted_key_value_component_expansion_synthesis_layer17_head1/targeted_key_value_component_expansion_synthesis_summary.json
targeted_key_value_component_expansion_synthesis_layer17_head1/targeted_key_value_component_expansion_synthesis_report.md
```

Run scale:

- target cases: `5`
- rows: `410`
- reduced groups: `35`
- layer/head: `17/1`
- direction: `masked_to_control`
- mode: `key_value_state_control`
- GPU used: `CUDA_VISIBLE_DEVICES=0`

## Command

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/key_value_component_expansion_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin,visual_near_ring,visual_far_background,non_region_complement,whole_head \
  --score-bias-modes key_value_state_control
```

## Result

Main case source components:

| component | probability recovery | attention mass recovery | rank recovery | abs-error recovery | mass16 recovery |
|---|---:|---:|---:|---:|---:|
| `duplicate_basin` | 0.010027 | 0.237000 | 16.2500 | -6.1240 | 0.180497 |
| `visual_near_ring` | 0.002492 | -0.089919 | 6.2500 | -0.8846 | 0.048246 |
| `visual_far_background` | 0.000198 | -0.006539 | 0.7500 | -0.2141 | 0.007676 |
| `non_region_complement` | 0.003225 | -0.007435 | 9.3333 | -1.2429 | 0.062394 |
| `whole_head` | 0.013037 | 0.000165 | 18.7500 | -7.2255 | 0.221317 |

Main case reference probes:

| probe | probability recovery | attention mass recovery | rank recovery | abs-error recovery |
|---|---:|---:|---:|---:|
| route/content total output | 0.018371 | n/a | 19.4167 | -9.8000 |
| route/content route-only output | 0.018168 | n/a | 19.3333 | -9.8141 |
| score-bias per-source proxy | 0.010221 | 0.292988 | 15.3333 | -6.2925 |
| direct key-state patch | 0.009652 | 0.237000 | 16.0000 | -5.9484 |
| duplicate-basin key+value-state patch | 0.010027 | 0.237000 | 16.2500 | -6.1240 |

Top cross-case groups by probability recovery:

| rank | checkpoint | record | phase | component | probability recovery |
|---:|---|---:|---|---|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `whole_head` | 0.013037 |
| 2 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `duplicate_basin` | 0.010027 |
| 3 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | `visual_near_ring` | 0.003950 |
| 4 | `no_aligner_parent_ckpt3668` | 36 | `post_y1/pre_x2` | `whole_head` | 0.003593 |
| 5 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `non_region_complement` | 0.003225 |

## Interpretation

Source competition outside `duplicate_basin` explains part of the missing
effect, but not all of it.

For the main record-33 failure:

- `whole_head` key+value patch is stronger than duplicate-basin-only:
  `0.013037` vs `0.010027`.
- `non_region_complement` and `visual_near_ring` each provide small positive
  probability recovery despite negative attention-mass recovery. This suggests
  they are not simple "more mass into target bucket" repairs; their effect is
  likely through value/output composition or normalization.
- `visual_far_background` is negligible.
- `whole_head` still remains below the route/content route-only output patch:
  `0.013037` vs `0.018168`.

This narrows the remaining gap again. It is not just:

1. duplicate-basin key state;
2. duplicate-basin value state;
3. duplicate-basin key+value state; or
4. source competition among the tested source buckets inside the attention
   score/value assembly.

The next likely site is downstream of assembled attention output:

- `o_proj`;
- post-attention residual addition;
- post-attention norm;
- coordinate-slot residual basin amplification.

## Next Deterministic Step

Run a same-target downstream-site comparison that separates:

- key+value whole-head state patch at the attention-score/value level;
- direct `self_attn_output` patch;
- `post_attention_residual` patch;
- `post_attention_norm` patch.

The concrete question is whether the missing `~0.005` probability recovery
appears when patching after `o_proj` or after residual/norm, which would place
the mechanism in output projection / residual-space geometry rather than
inside source-level attention routing.

## Verification

- GPU run completed with:
  `attention_score_bias_patch_row_count=410`, `checkpoint_count=3`,
  `replay_case_count=5`,
  `score_bias_modes=["key_value_state_control"]`.
- Reduced synthesis completed with:
  `rows=410`, `groups=35`.
- No code changes were made in this slice.
