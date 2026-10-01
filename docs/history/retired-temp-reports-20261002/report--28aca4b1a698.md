# Rollout FN-Factor Study: rollout-fn-factor-2b-hard32-extension-20260317

## Fixed Inputs

- `original`: `/data/home/xiaoyan/AIteam/data/CoordExp/output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332` (prompt=`coco_80`, field_order=`desc_first`)
- `a_only`: `/data/home/xiaoyan/AIteam/data/CoordExp/output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900` (prompt=`coco_80`, field_order=`desc_first`)
- `train`: `/data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
- `val`: `/data/home/xiaoyan/AIteam/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`

## Bootstrap Hard-Case Cohorts

### train
- image_id `22484` `images/train2017/000000022484.jpg`: mean_unresolved_gt=44.00, mean_unmatched_pred=33.00, gt_count=51
- image_id `403358` `images/train2017/000000403358.jpg`: mean_unresolved_gt=41.00, mean_unmatched_pred=13.50, gt_count=53
- image_id `124004` `images/train2017/000000124004.jpg`: mean_unresolved_gt=40.50, mean_unmatched_pred=5.00, gt_count=56
- image_id `164485` `images/train2017/000000164485.jpg`: mean_unresolved_gt=38.50, mean_unmatched_pred=20.50, gt_count=49
- image_id `378962` `images/train2017/000000378962.jpg`: mean_unresolved_gt=38.00, mean_unmatched_pred=13.50, gt_count=50

### val
- image_id `303566` `images/val2017/000000303566.jpg`: mean_unresolved_gt=39.50, mean_unmatched_pred=12.00, gt_count=53
- image_id `316666` `images/val2017/000000316666.jpg`: mean_unresolved_gt=34.50, mean_unmatched_pred=2.00, gt_count=40
- image_id `210273` `images/val2017/000000210273.jpg`: mean_unresolved_gt=33.00, mean_unmatched_pred=9.50, gt_count=41
- image_id `470924` `images/val2017/000000470924.jpg`: mean_unresolved_gt=31.00, mean_unmatched_pred=23.00, gt_count=55
- image_id `416991` `images/val2017/000000416991.jpg`: mean_unresolved_gt=30.50, mean_unmatched_pred=22.00, gt_count=38

## Recovery Waterfall

### train
- `a_only` / `Hard-16`:
  total_gt=838, deterministic_hit=252, decode_selection_miss=105, prefix_sensitive_miss=30, length_bias_miss=0, persistent_unrecovered=451
  annotation_mismatch_candidate=45, semantic_confusion_candidate=10
- `a_only` / `Hard-32`:
  total_gt=1667, deterministic_hit=563, decode_selection_miss=237, prefix_sensitive_miss=61, length_bias_miss=0, persistent_unrecovered=806
  annotation_mismatch_candidate=118, semantic_confusion_candidate=17
- `original` / `Hard-16`:
  total_gt=838, deterministic_hit=225, decode_selection_miss=108, prefix_sensitive_miss=34, length_bias_miss=0, persistent_unrecovered=471
  annotation_mismatch_candidate=49, semantic_confusion_candidate=6
- `original` / `Hard-32`:
  total_gt=1667, deterministic_hit=514, decode_selection_miss=247, prefix_sensitive_miss=54, length_bias_miss=0, persistent_unrecovered=852
  annotation_mismatch_candidate=125, semantic_confusion_candidate=13

### val
- `a_only` / `Hard-16`:
  total_gt=651, deterministic_hit=208, decode_selection_miss=97, prefix_sensitive_miss=15, length_bias_miss=0, persistent_unrecovered=331
  annotation_mismatch_candidate=49, semantic_confusion_candidate=6
- `a_only` / `Hard-32`:
  total_gt=1210, deterministic_hit=449, decode_selection_miss=169, prefix_sensitive_miss=27, length_bias_miss=0, persistent_unrecovered=565
  annotation_mismatch_candidate=102, semantic_confusion_candidate=13
- `original` / `Hard-16`:
  total_gt=651, deterministic_hit=185, decode_selection_miss=138, prefix_sensitive_miss=15, length_bias_miss=0, persistent_unrecovered=313
  annotation_mismatch_candidate=39, semantic_confusion_candidate=10
- `original` / `Hard-32`:
  total_gt=1210, deterministic_hit=402, decode_selection_miss=223, prefix_sensitive_miss=35, length_bias_miss=0, persistent_unrecovered=550
  annotation_mismatch_candidate=93, semantic_confusion_candidate=17

## Prefix Findings

### train / a_only
- `oracle_gt_prefix_random_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_random_order` n=2: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=2: health_valid=True, invalid_reason=None
- `self_prefix` n=1: health_valid=True, invalid_reason=None
- `self_prefix` n=2: health_valid=True, invalid_reason=None

### train / original
- `oracle_gt_prefix_random_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_random_order` n=2: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=2: health_valid=True, invalid_reason=None
- `self_prefix` n=1: health_valid=True, invalid_reason=None
- `self_prefix` n=2: health_valid=True, invalid_reason=None

### val / a_only
- `oracle_gt_prefix_random_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_random_order` n=2: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=2: health_valid=True, invalid_reason=None
- `self_prefix` n=1: health_valid=True, invalid_reason=None
- `self_prefix` n=2: health_valid=True, invalid_reason=None

### val / original
- `oracle_gt_prefix_random_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_random_order` n=2: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=1: health_valid=True, invalid_reason=None
- `oracle_gt_prefix_train_order` n=2: health_valid=True, invalid_reason=None
- `self_prefix` n=1: health_valid=True, invalid_reason=None
- `self_prefix` n=2: health_valid=True, invalid_reason=None

## Notes

- `decode_selection_miss` means same-prompt union-of-K recovered the object before any prefix/length intervention.
- `prefix_sensitive_miss` is scored from continuation-only recovery; injected prefix objects do not count.
- `length_bias_miss` means default deterministic/image-only rollout missed the object but the extended-length control recovered it.
- `persistent_unrecovered` should be read as unrecovered under tested interventions, not proven incapacity.
