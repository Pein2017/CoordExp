# Coordinate Family Comparison Report

This is a comparative progress scaffold for later family-level synthesis.

> Note: the mechanism layer is mixed (basin layer=smoke, recall layer=real).
> The matched val200 eval ranking below is real; family verdicts stay provisional until the remaining non-smoke basin/recall summaries land.

## Families

- `base_xyxy_merged`: `strong` (mass@4=1.0, wrong-anchor=0.0, vision_lift=4.5098)
- `center_parameterization`: `mixed` (mass@4=None, wrong-anchor=None, vision_lift=None)
- `cxcywh_pure_ce`: `strong` (mass@4=1.0, wrong-anchor=0.0, vision_lift=None)
- `raw_text_xyxy_pure_ce`: `mixed` (mass@4=None, wrong-anchor=None, vision_lift=4.8757)

## Recall Status

- `base_xyxy_merged`: status=verifier_complete_oracle_pending
- `center_parameterization`: status=oracle_and_verifier_complete, baseline_recall_loc=0.6163410301953819, oracle_k_recall_loc=0.6998223801065719, oracle_k_recovery_rate=0.25, competitive_fn_rate=0.041666666666666664, weak_visual_fn_rate=0.9583333333333334
- `cxcy_logw_logh_pure_ce`: status=verifier_complete_oracle_pending
- `cxcywh_pure_ce`: status=verifier_complete_oracle_pending
- `hard_soft_ce_2b`: status=verifier_complete_oracle_pending
- `raw_text_xyxy_pure_ce`: status=oracle_and_verifier_complete, baseline_recall_loc=0.47424511545293074, oracle_k_recall_loc=0.7069271758436945, oracle_k_recovery_rate=0.4527027027027027, competitive_fn_rate=0.013513513513513514, weak_visual_fn_rate=0.9797297297297297

## Matched val200 Eval

| Rank | Family | bbox_AP | bbox_AP50 | F1@0.50 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `center_parameterization` | 0.4220818648401396 | 0.600747198341825 | 0.6107707191228636 | 947.0 | 710.0 | 497.0 |
| 2 | `raw_text_xyxy_pure_ce` | 0.34401092701713487 | 0.44998226893484994 | 0.5899960614415125 | 749.0 | 346.0 | 695.0 |
| 3 | `cxcy_logw_logh_pure_ce` | 0.27915398081379983 | 0.42921469838416487 | 0.5083388925950635 | 762.0 | 792.0 | 682.0 |
| 4 | `cxcywh_pure_ce` | 0.27252623412406707 | 0.4003027885168362 | 0.46526249607041814 | 740.0 | 997.0 | 704.0 |
| 5 | `hard_soft_ce_2b` | 0.05679256527630405 | 0.14066405488087785 | 0.15785554728220402 | 212.0 | 1030.0 | 1232.0 |
| 6 | `base_xyxy_merged` | 0.05494939499748911 | 0.14049510410819718 | 0.14068692206076616 | 213.0 | 1371.0 | 1231.0 |

## Sources

- Basin summary: `/data/CoordExp/output/analysis/coord-family-basin-smoke/summary.json`
- Recall summary: `/data/CoordExp/output/analysis/coord-family-recall-progress-2026-04-20/summary.json`
- Vision rows: `2` inline rows
- Eval snapshot: `/data/CoordExp/output/analysis/coord-family-comparison-progress-2026-04-20/current_eval_metrics_snapshot.json`
