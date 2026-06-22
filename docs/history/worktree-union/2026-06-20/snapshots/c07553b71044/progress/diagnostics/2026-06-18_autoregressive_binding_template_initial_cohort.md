---
doc_id: progress.diagnostics.autoregressive_binding_template_initial_cohort
layer: progress
doc_type: diagnostic-artifact-note
status: branch-provenance
domain: research-history
summary: Initial val200 cohort manifest for the desc-first versus geometry-first compact object/box-closed binding study.
tags: [progress, diagnostics, autoregressive-binding, cohort, val200, desc-first, geometry-first]
updated: 2026-06-18
branch: codex/autoregressive-binding-template-study
---

# Autoregressive Binding Template Initial Cohort

## Scope

Evidence scope: `val200` artifact analysis, CPU-only, no hidden-state or causal
claim yet.

This note records the first deterministic cohort manifest for the
`desc_first` versus `geometry_first` `compact_object_box_closed` checkpoint-928
study. The cohort uses existing `rp=1.10` artifacts to select high-information
cases, while the canonical operational validation source for future reruns and
probes remains `bbox_len12000`.

## Inputs

Config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Desc-first artifact:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

Geometry-first artifact:

```text
/data/CoordExp/outputs/infer/recursive_detection_ce/geometry_first_object_box_closed_ckpt928_val200_bsz4_temp0_rep1p10_max3084_8gpu_diag
```

The two artifacts have `200` rows with matching image IDs in the same order and
zero GT-count mismatches. Known caveats:

- desc-first summary records requested adapter under singular
  `/data/CoordExp/output/...`, while the verified checkpoint path is under
  `/data/CoordExp/outputs/...`;
- geometry-first diagnostic artifact used a max60-named validation root, but
  the first-200 image IDs and GT counts match the desc-first `bbox_len12000`
  artifact.

## Artifact Root

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_initial_cohort
```

Files:

- `cohort_manifest.json`
- `pair_scene_metrics.jsonl`
- `cohort_summary.md`

## Cohort Summary

Rows analyzed: `200`

Selected primary cases: `38`

Held-out reserve cases: `24`

Bucket counts:

| Bucket | Count | Leading image IDs |
| --- | ---: | --- |
| `empty_or_parse` | 3 | `7511`, `3255`, `1761` |
| `desc_first_duplication_burst` | 3 | `15254`, `885`, `12120` |
| `geometry_first_duplication_burst` | 3 | `2685`, `2299`, `632` |
| `desc_first_overemission` | 3 | `5586`, `13659`, `9400` |
| `geometry_first_overemission` | 3 | `19109`, `14038`, `8762` |
| `low_recall_desc_first` | 3 | `16010`, `12639`, `17714` |
| `low_recall_geometry_first` | 3 | `12670`, `19432`, `18380` |
| `shared_low_recall_dense` | 2 | `17959`, `9590` |
| `same_class_crowded` | 3 | `4134`, `16228`, `14439` |
| `dense_or_tiny_scene` | 3 | `139`, `11197`, `13923` |
| `geometry_first_extreme_box` | 3 | `2157`, `15335`, `8277` |
| `balanced_disagreement` | 3 | `7574`, `9891`, `10707` |
| `clean_shared_success_control` | 3 | `1000`, `5001`, `17627` |

## Interpretation Boundary

This is a target-selection artifact, not a mechanism result. It may be used to
choose the first hidden-state, attention, logit, and prefix-counterfactual
panels. It should not be used to claim that a bucket label is the final failure
taxonomy for an image.

## Next Use

Recommended first deep-probe panel:

- `15254`: desc-first duplication-heavy, geometry-first compact.
- `2685`: geometry-first duplication-heavy, desc-first also nontrivial.
- `1761`: desc-first empty/parse failure versus geometry-first long burst.
- `3255`: geometry-first empty/parse failure versus desc-first predictions.
- `16010`: desc-first low-recall contrast.
- `12670`: geometry-first low-recall contrast.
- `2157`: geometry-first extreme-box candidate.
- `1000`: clean shared-success control.

Use the held-out reserve only after a candidate signal appears on the primary
panel.
