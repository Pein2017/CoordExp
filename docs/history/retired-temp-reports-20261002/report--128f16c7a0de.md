# Rollout FN-Factor Study: rollout-fn-factor-2b-smoke-20260316

## Fixed Inputs

- `original`: `/data/home/xiaoyan/AIteam/data/CoordExp/output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332` (prompt=`coco_80`, field_order=`desc_first`)
- `a_only`: `/data/home/xiaoyan/AIteam/data/CoordExp/output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900` (prompt=`coco_80`, field_order=`desc_first`)
- `train`: `/data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `val`: `/data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## Bootstrap Hard-Case Cohorts

### train
- image_id `217268` `images/train2017/000000217268.jpg`: mean_unresolved_gt=49.50, mean_unmatched_pred=32.50, gt_count=59
- image_id `124004` `images/train2017/000000124004.jpg`: mean_unresolved_gt=40.50, mean_unmatched_pred=4.50, gt_count=56
- image_id `365206` `images/train2017/000000365206.jpg`: mean_unresolved_gt=38.50, mean_unmatched_pred=24.50, gt_count=56
- image_id `247790` `images/train2017/000000247790.jpg`: mean_unresolved_gt=37.00, mean_unmatched_pred=18.50, gt_count=58
- image_id `161367` `images/train2017/000000161367.jpg`: mean_unresolved_gt=36.50, mean_unmatched_pred=19.00, gt_count=59

### val
- image_id `303566` `images/val2017/000000303566.jpg`: mean_unresolved_gt=39.00, mean_unmatched_pred=14.00, gt_count=53
- image_id `210273` `images/val2017/000000210273.jpg`: mean_unresolved_gt=33.50, mean_unmatched_pred=10.00, gt_count=41
- image_id `31296` `images/val2017/000000031296.jpg`: mean_unresolved_gt=32.50, mean_unmatched_pred=21.50, gt_count=52
- image_id `316666` `images/val2017/000000316666.jpg`: mean_unresolved_gt=32.00, mean_unmatched_pred=3.50, gt_count=40
- image_id `470924` `images/val2017/000000470924.jpg`: mean_unresolved_gt=30.50, mean_unmatched_pred=17.50, gt_count=55

## Recovery Waterfall

### train
- `a_only` / `Hard-16`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `a_only` / `Hard-32`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `original` / `Hard-16`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `original` / `Hard-32`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0

### val
- `a_only` / `Hard-16`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `a_only` / `Hard-32`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `original` / `Hard-16`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0
- `original` / `Hard-32`:
  total_gt=0, deterministic_hit=0, decode_selection_miss=0, prefix_sensitive_miss=0, length_bias_miss=0, persistent_unrecovered=0
  annotation_mismatch_candidate=0, semantic_confusion_candidate=0

## Prefix Findings

### train / a_only
- `oracle_gt_prefix_random_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_random_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=1: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=2: health_valid=False, invalid_reason=parse_invalid

### train / original
- `oracle_gt_prefix_random_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_random_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=1: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=2: health_valid=False, invalid_reason=parse_invalid

### val / a_only
- `oracle_gt_prefix_random_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_random_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=1: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=2: health_valid=False, invalid_reason=parse_invalid

### val / original
- `oracle_gt_prefix_random_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_random_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=1: health_valid=False, invalid_reason=parse_invalid
- `oracle_gt_prefix_train_order` n=2: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=1: health_valid=False, invalid_reason=parse_invalid
- `self_prefix` n=2: health_valid=False, invalid_reason=parse_invalid

## Stress Findings

- `train` / `original` / `switched_prefix` / `switched_prefix`: health_valid=False reason=parse_invalid
- `train` / `original` / `broken_prefix` / `broken_prefix` / delete: health_valid=False reason=parse_invalid
- `train` / `original` / `broken_prefix` / `broken_prefix` / adjacent_swap: health_valid=False reason=parse_invalid
- `train` / `a_only` / `switched_prefix` / `switched_prefix`: health_valid=False reason=parse_invalid
- `train` / `a_only` / `broken_prefix` / `broken_prefix` / delete: health_valid=False reason=parse_invalid
- `train` / `a_only` / `broken_prefix` / `broken_prefix` / adjacent_swap: health_valid=False reason=parse_invalid
- `val` / `original` / `switched_prefix` / `switched_prefix`: health_valid=False reason=parse_invalid
- `val` / `original` / `broken_prefix` / `broken_prefix` / delete: health_valid=False reason=parse_invalid
- `val` / `original` / `broken_prefix` / `broken_prefix` / adjacent_swap: health_valid=False reason=parse_invalid
- `val` / `a_only` / `switched_prefix` / `switched_prefix`: health_valid=False reason=parse_invalid
- `val` / `a_only` / `broken_prefix` / `broken_prefix` / delete: health_valid=False reason=parse_invalid
- `val` / `a_only` / `broken_prefix` / `broken_prefix` / adjacent_swap: health_valid=False reason=parse_invalid

## Notes

- `decode_selection_miss` means same-prompt union-of-K recovered the object before any prefix/length intervention.
- `prefix_sensitive_miss` is scored from continuation-only recovery; injected prefix objects do not count.
- `length_bias_miss` means default deterministic/image-only rollout missed the object but the extended-length control recovered it.
- `persistent_unrecovered` should be read as unrecovered under tested interventions, not proven incapacity.
