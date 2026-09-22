# FN Visibility Guidance Joint Summary

Date: 2026-06-11

## Scope

Self-driven false-negative branch for the mechanistic diagnosis roadmap.

Question:

- Are false negatives primarily cases where the model cannot perceive the missing object, or do many FNs retain nearby visual/proposal evidence and therefore plausibly need better language-side or prefix/context guidance?

This is a manifest-level proxy analysis. It does not run actual `desc_only` / `desc_x1` continuation decoding.

## Inputs

Parent rollout without MLP aligner tuned:

- Rollout:
  `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`
- Scored rows:
  `gt_vs_pred_scored.jsonl`
- Matches:
  `eval/matches@0.30_guarded.jsonl`

Parent rollout with MLP aligner tuned:

- Rollout:
  `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_original_latest_aligner_parent_val128_freegreedy_ckpt1824_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`
- Scored rows:
  `gt_vs_pred_scored.jsonl`
- Matches:
  `eval/matches@0.30_guarded.jsonl`

Existing aux manifest:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_fn_visibility_guidance_probe_aux_latest_ckpt32_val128`

## New Artifacts

Parent manifests:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668`
- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aligner_parent_ckpt1824`

Joint summary:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/joint_summary/phase4_fn_visibility_guidance_joint_summary.json`
- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/joint_summary/phase4_fn_visibility_guidance_joint_summary.md`

Regeneration note:

- The parent manifests were regenerated after fixing `hint_x1` to use norm1000 coordinate-token values rather than scored pixel x1 values.
- Rows now preserve `gt_bbox_xyxy` in the scored coordinate frame and add `gt_bbox_norm1000_xyxy` for prompt-side coordinate hints.
- `desc_only` rows now carry `hint_x1=null`; only `desc_x1` and `desc_x1_wrong_control` carry coordinate hints.
- Example: image `139`, GT pixel x1 `657` at width `1248` now becomes norm1000 hint x1 `526`.

## Counts

Across the two parent manifests plus the existing aux manifest:

- Total FN cases: `1106`
- Invalid / parse-blocked / non-metric cases: `490` (`44.3%`)
- Valid measurable FN cases: `616`
- Language/proxy-evidence cases: `436` (`70.8%` of valid)
- Visual-miss cases: `180` (`29.2%` of valid)

Checkpoint breakdown:

| checkpoint | total FN | invalid | valid | language/proxy valid % | visual-miss valid % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | 467 | 130 | 337 | 69.7% | 30.3% |
| `aligner_parent_ckpt1824` | 543 | 309 | 234 | 70.1% | 29.9% |
| `aux_latest_ckpt32` | 96 | 51 | 45 | 82.2% | 17.8% |

Bucket meanings:

- `likely_language_or_prefix_guidance_fragile`: a same-description nearby prediction exists.
- `likely_visual_available_but_binding_or_enumeration_failed`: a nearby/overlapping prediction exists but with wrong description or imperfect binding.
- `likely_visual_blind_or_no_object_proposal`: no nearby proposal found.
- `likely_visual_low_salience_small_object`: missing object is tiny enough that low salience is plausible.
- `artifact_invalid_or_parse_blocked`: not interpretable as a clean metric-bearing FN case.

## Interpretation

Among valid measurable FNs, most are not clean "model could not see it" cases. About 71% have same-description or nearby visual-proxy evidence. This supports the hypothesis that many false negatives are hidden behind locally plausible predictions, weak binding, delayed enumeration, or prefix/context fragility.

The aux checkpoint has fewer valid FN cases and a larger share of language/proxy-evidence cases among valid FNs (`82.2%`), but its invalid/parse-blocked share remains high. This is consistent with changed mechanics after the auxiliary objective, but not by itself sufficient to say the auxiliary objective improves FN perception.

## Guardrails

- This is a proxy manifest, not actual rescue decoding.
- `language_proxy_cases` says "visual/proposal evidence exists nearby"; it does not prove `desc_x1` guidance will recover the object.
- The parent manifests used `matches@0.30_guarded.jsonl`; the existing aux manifest used its available `matches.jsonl`, so comparisons should be treated as directional until regenerated under identical matching contracts.
- High invalid shares mean valid-only fractions are more meaningful for mechanism hypotheses than all-FN percentages.
- Continuation probes must use `gt_bbox_norm1000_xyxy` / norm1000 `hint_x1`, not scored pixel coordinates.

## Next Probe

Use the generated manifests to sample matched buckets for continuation decoding:

- `desc_only`: description hint only, tests whether semantic guidance is enough.
- `desc_x1`: description plus correct x1 coordinate, tests coordinate/prefix guidance fragility.
- `desc_x1_wrong_control`: description plus nearest wrong-control x1, tests coordinate leakage.

Prioritize valid FN cases in `likely_language_or_prefix_guidance_fragile` and `likely_visual_available_but_binding_or_enumeration_failed`, because those are the cases most likely to distinguish visual unavailability from language-side guidance fragility.
