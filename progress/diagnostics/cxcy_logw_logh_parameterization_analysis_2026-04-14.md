---
title: Center Log-Size Parameterization Analysis
date: 2026-04-14
status: completed
owner: codex
---

# Center Log-Size Parameterization Analysis

## Scope

This note investigates the trained Stage-1 checkpoint:

- `configs/stage1/profiles/2b/cxcy_logw_logh_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml`
- model: `output/stage1_2b/cxcy_logw_logh-pure_ce-2b-ckpt_2600-merged`

The goal is to compare its behavior against the prior duplication-collapse study
surfaces, with emphasis on:

- standard detection performance,
- duplication-collapse behavior,
- and whether the center-based parameterization changes optimization dynamics in
  a way that is visible both in trainer telemetry and in rollout artifacts.

Prior context reused here:

- `openspec/changes/add-duplication-collapse-analysis-study/`
- `openspec/changes/add-center-size-bbox-supervision/`
- `progress/diagnostics/duplication_collapse_final_analysis_2026-04-13.md`

## Experimental Setup

New scaffolding added for this analysis:

- infer config:
  `configs/infer/coco_1024/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce-2b-ckpt2600-merged.yaml`
- scored-artifact materialization config:
  `configs/infer/coco_1024/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce-2b-ckpt2600-materialize_scored.yaml`
- bundle-eval config:
  `configs/eval/coco_1024/valfull_lvis_proxy_cxcy_logw_logh_pure_ce_2b_ckpt2600_bundle.yaml`
- duplication panel config:
  `configs/analysis/duplication_collapse/contrastive_cxcy_logw_logh_pure_ce_panel.yaml`

The duplication panel reuses the previous study’s pinned case anchors:

- `stage1_ce_ciou_ckpt1564`: lines `2128`, `3892`
- `stage1_2b_center_param_ckpt1564`: lines `19`, `59`, `172`, `221`, `323`,
  `500`, `1040`, `3079`, `4119`

and replays those source cases onto:

- `stage1_cxcy_logw_logh_pure_ce_2b_ckpt2600`
- `stage1_pure_ce_ckpt1932`

This produced a 33-case panel:

- 11 source cases
- 22 replay cases

## Format Contract Verification

The current model emits bbox coordinates in `cxcy,logw,logh` form, so exact
inversion to canonical `xyxy` is required before any metric or probe-stage
artifact consumes the boxes.

Verified code path:

- `src/infer/engine.py`
  - `InferenceConfig.bbox_format`
  - `InferenceEngine.__init__` passes `bbox_format` into
    `CoordinateStandardizer`
- `src/common/coord_standardizer.py`
  - `_scale_points(...)` detects `bbox_format == "cxcy_logw_logh"`
  - calls
    `cxcy_logw_logh_norm1000_to_xyxy_norm1000(...)`
- `src/common/geometry/bbox_parameterization.py`
  - `cxcy_logw_logh_norm1000_to_xyxy_norm1000(...)`

Verified artifact surface:

- sampled live infer output already emitted canonical pixel boxes such as
  `[814, 445, 856, 698]` in `gt_vs_pred.jsonl`, not raw center/log-size tuples
- infer summary confirms `bbox_format: cxcy_logw_logh`
- the duplication-analysis config now resolves the new alias with
  `bbox_format: cxcy_logw_logh`

Two workflow fixes were required to make the study consistent with that
contract:

1. `src/analysis/duplication_collapse_analysis.py` was updated so replay and
   probe paths pass per-checkpoint `bbox_format` into `InferenceEngine`.
2. `src/infer/pipeline.py` was updated so `cxcy_logw_logh` infer runs also
   materialize `gt_vs_pred_scored.jsonl` via the built-in constant-score path,
   even when the run is infer-only.

One incorrect attempt was discarded:

- generic confidence post-op (`scripts/postop_confidence.py`) is not valid for
  `cxcy_logw_logh` in V1 and dropped every object with
  `pred_alignment_mismatch`
- those artifacts were removed and replaced with the pipeline’s constant-score
  `gt_vs_pred_scored.jsonl`

## Trainer Telemetry

Closest on-disk telemetry comparison surfaces:

- current run:
  `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce/.../checkpoint-2724/trainer_state.json`
- historical 2B center-parameterization continuation:
  `output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization/.../checkpoint-1564/trainer_state.json`
- historical 4B `ce_ciou` continuation:
  `output/stage1/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-ce_ciou/.../checkpoint-1564/trainer_state.json`

Important caveat:

- the current run is a from-base 2B training run
- both comparison surfaces are warm-start continuation runs
- raw loss values are therefore not directly comparable across families

More stable telemetry reads:

- `cxcy_logw_logh_2b` best checkpoint is `2600`
  - `eval_token_acc = 0.85342`
  - `eval_coord_diag/coord_vocab_mass = 0.99659`
  - `eval_coord_diag/expected_bin_mae = 19.28`
  - `eval_coord_diag/expected_bin_abs_err_p90 = 52.04`
- `center_param_2b` best checkpoint is `1564`
  - `eval_token_acc = 0.85094`
  - `eval_coord_diag/coord_vocab_mass = 0.99710`
  - `eval_coord_diag/expected_bin_mae = 29.11`
  - `eval_coord_diag/expected_bin_abs_err_p90 = 69.68`
- `ce_ciou_4b` best observed eval step is `80`
  - `eval_token_acc = 0.85275`
  - `eval_coord_diag/coord_vocab_mass = 0.99933`
  - `eval_coord_diag/expected_bin_mae = 21.69`
  - `eval_coord_diag/expected_bin_abs_err_p90 = 40.12`

Convergence-shape differences:

- `cxcy_logw_logh_2b` starts cold
  - step 40: `coord_vocab_mass = 0.0050`, `expected_bin_mae = 256.54`
  - reaches `coord_vocab_mass > 0.95` by step `200`
  - reaches `expected_bin_mae < 25` by step `640`
  - reaches `expected_bin_mae < 20` by step `1920`
  - reaches `eval_token_acc > 0.85` by step `1720`
- `center_param_2b` and `ce_ciou_4b` both begin near their eventual token
  regime at step `40`, consistent with continuation rather than cold-start
  optimization

Interpretation:

- the current center-log-size objective does converge to a sharp coord-token
  regime
- by token-centric telemetry alone it does not look obviously broken at the end
- but the downstream detection behavior diverges sharply from what those
  trainer metrics would lead you to expect

## Standard Detection Results

Artifacts:

- infer run:
  `output/infer/coco1024_valfull_lvis_proxy_cxcy_logw_logh_pure_ce_2b_ckpt2600_merged/`
- bundle summary:
  `output/infer/coco1024_valfull_lvis_proxy_cxcy_logw_logh_pure_ce_2b_ckpt2600_merged/proxy_eval_bundle_summary.json`

The decode was mostly syntactically valid:

- `summary.json`: `4951` samples read, `4951` emitted
- only `17` `invalid_geometry` errors at infer time
- bundle-eval counters show `0` `invalid_geometry`, `0` `degenerate`, and `0`
  `empty_pred` after artifact standardization

Despite that, detection quality collapsed:

- `coco_real`
  - `bbox_AP = 0.0000748`
  - `bbox_AP50 = 0.0002810`
  - `bbox_AP75 = 0.0000212`
  - `f1ish@0.50 precision = 0.006666`
  - `f1ish@0.50 recall = 0.007278`
  - `f1ish@0.50 f1 = 0.006959`
- `coco_real_strict`
  - `bbox_AP = 0.0000741`
  - `bbox_AP50 = 0.0002684`
  - `bbox_AP75 = 0.0000212`
  - `f1ish@0.50 precision = 0.006729`
  - `f1ish@0.50 recall = 0.007122`
  - `f1ish@0.50 f1 = 0.006920`
- `coco_real_strict_plausible`
  - `bbox_AP = 0.0000741`
  - `bbox_AP50 = 0.0002689`
  - `bbox_AP75 = 0.0000212`
  - `f1ish@0.50 precision = 0.006878`
  - `f1ish@0.50 recall = 0.006794`
  - `f1ish@0.50 f1 = 0.006836`

Against the prior comparison surfaces:

- vs `stage1_ce_ciou_ckpt1564` (`coco_real`)
  - `bbox_AP`: `-0.28761`
  - `f1ish@0.50 f1`: `-0.46915`
- vs `stage1_2b_center_param_ckpt1564` (`coco_real`)
  - `bbox_AP`: `-0.37310`
  - `f1ish@0.50 f1`: `-0.57909`

The failure is not simply “the model emits far too many objects on average”:

- current run mean predictions/image: `8.319`
- `stage1_2b_center_param_ckpt1564`: `8.590`
- `stage1_ce_ciou_ckpt1564`: `10.067`

However, the new run still shows obvious bursty failure signatures:

- `53` images hit the 128-object cap exactly
- `108` images produced `>= 64` predictions
- `131` images had a max same-desc burst of at least `32`
- `106` images had a max same-desc burst of at least `64`
- `98` images had a max same-desc burst of at least `100`

Important caveat:

- AP for this family is based on the officially documented constant-score scored
  artifact, not confidence-fused per-object scores
- that scoring choice can further depress AP relative to confidence-scored
  baselines
- but it does **not** explain the near-zero F1ish numbers, so the collapse is
  not just a ranking artifact

### Transform Sanity Check

Because `bbox_AP` fell far below the prior `0.28` baseline, the bbox
transformation itself was stress-tested before accepting the result.

Checks performed:

- verified that training samples really serialize `bbox_2d` as
  `cx,cy,logw,logh` bins after dataset preprocessing
- verified that live inference uses the documented
  `cxcy_logw_logh_norm1000_to_xyxy_norm1000(...)` inversion path before
  metric evaluation
- materialized a diagnostic alternate artifact that reinterprets the same raw
  generated `bbox_2d` tuples as linear `cxcywh`

Diagnostic alternate-eval artifact:

- materializer:
  `temp/materialize_alt_bbox_decode.py`
- bundle-eval config:
  `temp/valfull_lvis_proxy_cxcy_logw_logh_pure_ce_2b_ckpt2600_linear_cxcywh_bundle.yaml`
- alternate summary:
  `output/infer/coco1024_valfull_lvis_proxy_cxcy_logw_logh_pure_ce_2b_ckpt2600_linear_cxcywh_decode/proxy_eval_bundle_summary.json`

Alternate `cxcywh` reinterpretation does improve overlap metrics, but not nearly
enough to explain the collapse away:

- official `cxcy_logw_logh` decode (`coco_real`)
  - `bbox_AP = 0.0000748`
  - `bbox_AP50 = 0.0002810`
  - `f1ish@0.50 f1 = 0.006959`
- alternate linear `cxcywh` decode (`coco_real`)
  - `bbox_AP = 0.002644`
  - `bbox_AP50 = 0.012204`
  - `f1ish@0.50 f1 = 0.032844`

That is a nontrivial lift:

- `bbox_AP`: `+0.00257` (`35.4x` relative)
- `f1ish@0.50 f1`: `+0.02589` (`4.72x` relative)

but it still remains vastly below:

- `stage1_ce_ciou_ckpt1564`: `bbox_AP = 0.28768`
- `stage1_2b_center_param_ckpt1564`: `bbox_AP = 0.37317`

Additional slot-level diagnostic:

- same-desc greedy center matching over current raw generations produced
  `26,564` matched pairs
- mean absolute error of emitted size slots vs GT encodings:
  - log-size target space:
    - `w_log = 181.87`
    - `h_log = 138.74`
  - linear-width target space:
    - `w_lin = 593.74`
    - `h_lin = 519.84`

Interpretation:

- the model’s emitted size slots are still substantially closer to the intended
  log-size target space than to linear width/height bins
- a transform mismatch contributes to the headline metric sensitivity, but the
  collapse is **not** primarily explained by “the model was actually outputting
  cxcywh”
- the current pure-CE center-log-size recipe appears to learn a poor size
  surface even though the official inversion path is correct

## Duplication Collapse Results

Completed study artifacts:

- run root:
  `research/duplication_collapse_cxcy_logw_logh/duplication-collapse-contrastive-center-log-size-pure-ce-panel/`
- probe rows:
  `research/duplication_collapse_cxcy_logw_logh/duplication-collapse-contrastive-center-log-size-pure-ce-panel/probe/case_rows.jsonl`
- compare rows:
  `research/duplication_collapse_cxcy_logw_logh/duplication-collapse-contrastive-center-log-size-pure-ce-panel/compare/case_rows.jsonl`
- report:
  `research/duplication_collapse_cxcy_logw_logh/duplication-collapse-contrastive-center-log-size-pure-ce-panel/report/report.md`

Top-line report summary:

- selected cases: `33`
- reproduced checkpoints: `4`
- overall classification counts:
  - `mixed = 14`
  - `insufficient-evidence = 7`
  - `internal-state-dominant = 2`
  - `coordinate-dominant = 1`

Per-checkpoint cohort summary:

- `stage1_ce_ciou_ckpt1564`
  - `2` cases
  - classification: `1 internal-state-dominant`, `1 mixed`
  - `gt_next_minus_duplicate` mean: `-1.94`
  - final `history_minus_visual` mean: `0.364`
- `stage1_2b_center_param_ckpt1564`
  - `9` cases
  - classification: `1 coordinate-dominant`, `2 mixed`, `6 insufficient-evidence`
  - `gt_next_minus_duplicate` mean: `-2.28`
  - final `history_minus_visual` mean: `0.257`
- `stage1_cxcy_logw_logh_pure_ce_2b_ckpt2600`
  - `7` cases
  - classification: `7 mixed`
  - `gt_next_minus_duplicate` mean: `-1.26`
  - no stable `history_minus_visual` / `prior_coord_minus_visual` aggregate was
    emitted for this cohort in the final report
- `stage1_pure_ce_ckpt1932`
  - `6` cases
  - classification: `4 mixed`, `1 internal-state-dominant`,
    `1 insufficient-evidence`
  - `gt_next_minus_duplicate` mean: `-2.17`
  - final `history_minus_visual` mean: `0.344`

Most important contrastive control result:

- CE-like comparison surfaces remain uniformly duplicate-favoring on the
  `gt_next_minus_duplicate` control
  - `stage1_ce_ciou_ckpt1564`: `2/2` negative
  - `stage1_2b_center_param_ckpt1564`: `3/3` negative
  - `stage1_pure_ce_ckpt1932`: `3/3` negative
- the current center-log-size run does **not** stay uniformly duplicate-favoring
  on that control
  - `stage1_cxcy_logw_logh_pure_ce_2b_ckpt2600`: `3/6` positive,
    `3/6` negative

Representative center-log-size control margins:

- `...from_stage1_2b_center_param_ckpt1564-line_03079`
  - `gt_next_minus_duplicate = +0.617`
  - `predicted_minus_duplicate = +1.552`
- `...from_stage1_2b_center_param_ckpt1564-line_04119`
  - `gt_next_minus_duplicate = +0.282`
  - `predicted_minus_duplicate = +0.973`
- `...from_stage1_2b_center_param_ckpt1564-line_00323`
  - `gt_next_minus_duplicate = +0.736`
  - `predicted_minus_duplicate = -0.818`
- but strongly duplicate-favoring cases still remain
  - `...from_stage1_2b_center_param_ckpt1564-line_00500`
    - `gt_next_minus_duplicate = -4.617`

Family-comparison readout from the final report:

- `stage1_ce_ciou_ckpt1564`
  - mechanism readout:
    `ce-side-family-still-shows-copy-basin-or-mixed-failure`
- `stage1_2b_center_param_ckpt1564`
  - mechanism readout:
    `ce-side-family-still-shows-copy-basin-or-mixed-failure`
- `stage1_cxcy_logw_logh_pure_ce_2b_ckpt2600`
  - mechanism readout:
    `mixed-or-unknown-family-behavior`
  - diagnostic note:
    `insufficient-case-evidence`

Important limitation:

- the center-log-size cohort did not yield the same depth of onset-probe
  surfaces as the legacy `xyxy` families
- in the report, the center-log-size case rows largely show
  `deep_onset_probe = false`, so the study can say that duplicate preference is
  less uniform, but it cannot cleanly prove a new replacement mechanism with the
  same confidence as the CE-family traces

## Takeaway

The completed evidence supports three conclusions:

1. `cxcy_logw_logh` token training can look numerically healthy in trainer
   telemetry while still failing catastrophically at full-val detection time.
2. The official `cxcy,logw,logh -> xyxy` inversion path is necessary and
   correct, and although a linear `cxcywh` reinterpretation improves metrics
   somewhat, it does **not** rescue the run or explain the collapse.
3. Relative to the previous CE-like duplication families, the current
   center-log-size run shows a weaker and less uniform duplicate-preference
   signal in case-control margins, but the mechanism evidence is still
   incomplete enough that the safest classification is “mixed / ambiguous,” not
   “successfully solved duplication collapse.”

Practical implication:

- the current pure-CE center-log-size recipe is not a viable improvement over
  the prior baselines
- if this parameterization is revisited, the next iteration should focus on
  recovering stable size supervision and stronger rollout-probe compatibility
  before treating duplication changes as a primary success signal
