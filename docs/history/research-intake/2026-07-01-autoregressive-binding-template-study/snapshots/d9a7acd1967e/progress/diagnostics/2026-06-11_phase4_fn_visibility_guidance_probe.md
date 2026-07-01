# Phase 4 FN Visibility Guidance Probe

Date: 2026-06-11

Scope: add a false-negative probe panel for the current val128 loss-only
auxiliary checkpoint. The question is whether missing objects are truly not
visually perceived, or whether they are available in the visual/prediction
state but fragile to language-side prefix/context guidance.

## Probe Design

The probe is intentionally two-stage:

1. CPU post-hoc selection from scored rollout artifacts.
2. Later continuation decode using the emitted guidance tiers.

The selector does not claim visual perception. It creates a test panel by
joining unmatched GT objects from `matches.jsonl` with model predictions from
`gt_vs_pred_scored.jsonl`, then assigning a visual-proxy bucket:

- `artifact_invalid_or_parse_blocked`: rollout/parser artifact is not reliable
  enough for model-limitation interpretation.
- `likely_language_or_prefix_guidance_fragile`: an unmatched GT object has a
  nearby same-desc prediction, suggesting the object/category is at least
  partially represented but not correctly enumerated/localized.
- `likely_visual_available_but_binding_or_enumeration_failed`: a nearby
  wrong-desc or spatially close prediction exists, suggesting some visual
  proposal is available but binding/enumeration failed.
- `likely_visual_low_salience_small_object`: no useful proposal and the object
  is tiny.
- `likely_visual_blind_or_no_object_proposal`: no useful nearby proposal.

For every selected FN candidate, the manifest emits three guidance tiers:

- `desc_only`
- `desc_x1`
- `desc_x1_wrong_control`

Interpretation rule for the later decode:

- `desc_x1` succeeds while `desc_only` fails: language-side coordinate/context
  guidance fragility.
- both fail and no visual proxy exists: stronger evidence for visual miss or
  low salience.
- `desc_x1_wrong_control` succeeds similarly to target `desc_x1`: coordinate
  hint leakage or weak binding, not true target rescue.

## Code Surface

```text
src/analysis/autoregressive_duplication_mechanism/phase4_fn_visibility_guidance_probe.py
scripts/analysis/run_autoregressive_duplication_phase4_fn_visibility_guidance_probe.py
tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_visibility_guidance_probe.py
```

The implementation handles the current artifact schema mismatch where
`matches.jsonl` uses row-local `image_id` while `gt_vs_pred_scored.jsonl` uses
COCO image id; it falls back to `file_name`/`image` as the stable join key.

## Current Artifact

Checkpoint:

```text
aux_latest_ckpt32
/data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32
```

Rollout:

```text
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_fn_visibility_guidance_probe_aux_latest_ckpt32_val128
```

Files:

```text
fn_visibility_guidance_probe_rows.jsonl
phase4_fn_visibility_guidance_probe_summary.json
phase4_fn_visibility_guidance_probe_report.md
```

## Initial Manifest Read

Scope: `sample_limit=96` unique FN candidates, three guidance rows per
candidate.

```text
unique_fn_case_count = 96
row_count = 288
```

Bucket counts:

| bucket | count |
|---|---:|
| artifact_invalid_or_parse_blocked | 51 |
| likely_language_or_prefix_guidance_fragile | 24 |
| likely_visual_available_but_binding_or_enumeration_failed | 13 |
| likely_visual_blind_or_no_object_proposal | 4 |
| likely_visual_low_salience_small_object | 4 |

Read:

- A large fraction of selected FNs are parse-blocked; these should be separated
  from genuine model-limit claims.
- Among interpretable cases, the panel contains more visual-proxy/guidance
  candidates (`24 + 13`) than no-proposal/low-salience candidates (`4 + 4`).
- This makes the language-guidance/prefix-fragility hypothesis worth testing
  directly before concluding the model cannot visually perceive the missing
  objects.

## Verification

Syntax:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_fn_visibility_guidance_probe.py \
  scripts/analysis/run_autoregressive_duplication_phase4_fn_visibility_guidance_probe.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_visibility_guidance_probe.py
```

Direct tests passed:

```text
test_bbox_iou_handles_overlap_and_empty_boxes
test_build_guidance_probe_rows_emits_three_tiers
test_classify_guidance_fragile_when_same_desc_prediction_nearby
test_classify_visual_available_when_wrong_desc_prediction_overlaps
```

Repo pytest wrapper note: `python -m pytest ...` returns `Pytest: No tests
collected`, consistent with the local test wrapper behavior observed in the
mechanism worktree.
