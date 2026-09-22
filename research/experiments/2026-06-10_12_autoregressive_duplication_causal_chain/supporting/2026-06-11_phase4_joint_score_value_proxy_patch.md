# Phase 4 Joint Score/Value Proxy Patch

Date: 2026-06-11

Scope: add a small causal-proxy probe that composes two already supported
Phase 4 interventions for the primary layer 17 head 1 duplicate-basin route:

1. patch the attention score pattern over a selected source bucket; and
2. swap the corresponding precomputed value-source contribution at `o_proj`.

This probe is deliberately labeled as a proxy. It does not recompute a clean
matched K/V state under the patched route. Instead, it asks whether a joint
score-pattern plus value-source substitution can recover more of the coordinate
basin than either diagnostic surface alone.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_joint_score_value_patch.py
scripts/analysis/run_autoregressive_duplication_phase4_joint_score_value_patch_shard.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_joint_score_value_patch.py
```

Every emitted row uses:

```text
task = phase4_joint_score_value_patch
patch_kind = joint_score_pattern_plus_value_source_proxy
proxy_caveat = score route is patched in the attention kernel, while the value-source swap subtracts a precomputed old contribution and adds a precomputed replacement contribution; this is not a clean recomputed K/V-state patch
```

## Smoke Artifact

Input token windows:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_primary_smoke_token_windows.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_joint_score_value_patch_layer17_head1_primary_smoke
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_joint_score_value_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_value_source_layer17_head1_primary_smoke_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase2_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_joint_score_value_patch_layer17_head1_primary_smoke \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 17 \
  --attention-head 1 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin \
  --score-bias-modes per_source_delta
```

Summary:

```text
checkpoint_count = 1
replay_case_count = 1
region_row_count = 7
joint_score_value_patch_row_count = 12
```

Mean row deltas over the 12-row primary smoke:

| field | mean | min | max |
|---|---:|---:|---:|
| `prob_recovery_from_masked` | 0.011344 | -0.000939 | 0.027071 |
| `coord_mass_radius_4_recovery_from_masked` | 0.076090 | -0.003046 | 0.174056 |
| `coord_mass_radius_8_recovery_from_masked` | 0.134690 | -0.005025 | 0.321216 |
| `coord_expected_abs_error_recovery_from_masked` | -6.486229 | -16.416286 | 0.234165 |
| `attention_mass_recovery_from_masked` | 0.292988 | 0.000007 | 0.793335 |
| `prob_damage_from_control` | -0.002388 | -0.015488 | 0.008776 |
| `coord_mass_radius_4_damage_from_control` | -0.020523 | -0.101908 | 0.040545 |

## Read

The joint proxy recovers a visibly larger fraction of the duplicate-basin
coordinate distribution than the routing-only readout on this small primary
window, especially in radius-4/radius-8 mass and expected absolute error.

The result should not be promoted as clean causal evidence. Its value is that
it narrows the next hidden-state/attention analysis target: if joint route and
value content has this much rescue power, the final picture should focus on how
the basin-attracting route is built, selected, and overwritten across the
coordinate slot sequence rather than only on whether the visual region is
present.

## Next Use

- Use this proxy as a triage surface for promising windows and heads.
- Prefer a cleaner recomputed K/V or residual-stream patch before making a
  mechanistic claim about exact sufficiency.
- Keep the dynamic exploration policy active: when a path is promising and has
  more influence over the final picture, it is acceptable to dive deeper and
  update the task sequence.

## Verification

```text
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_joint_score_value_patch.py \
  scripts/analysis/run_autoregressive_duplication_phase4_joint_score_value_patch_shard.py

python -m pytest tests/analysis/autoregressive_duplication_mechanism/test_phase4_joint_score_value_patch.py
```
