# Coord Family Contract Audit

- Family count: 6
- Runtime-ready families: 6
- Checkpoint types: {"adapter": 3, "merged": 3}

| Alias | Checkpoint Path | Type | Load Pattern | Infer Mode | BBox Format | Pred Coord Mode | Eval Path | Headline 2B | Runtime Ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_xyxy_merged | output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged | merged | model_checkpoint only | coord | xyxy | pixel | confidence_postop | True | True |
| raw_text_xyxy_pure_ce | output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552 | adapter | model_checkpoint + adapter_checkpoint | text | xyxy | norm1000 | confidence_postop | True | True |
| cxcywh_pure_ce | output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcywh-pure_ce/epoch_4-cxcywh-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v0-20260415-060451/checkpoint-692 | adapter | model_checkpoint + adapter_checkpoint | coord | cxcywh | norm1000 | constant_score_scored_jsonl | True | True |
| cxcy_logw_logh_pure_ce | output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-cxcy_logw_logh-pure_ce/epoch_4-cxcy_logw_logh-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v3-20260414-102133/checkpoint-716 | adapter | model_checkpoint + adapter_checkpoint | coord | cxcy_logw_logh | norm1000 | constant_score_scored_jsonl | True | True |
| center_parameterization | output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged | merged | model_checkpoint only | coord | center_parameterization | pixel | family_specific_projection_audit | True | True |
| hard_soft_ce_2b | output/stage1_2b/coco_bbox_max60-hard_soft_ce-2b-merged | merged | model_checkpoint only | coord | xyxy | pixel | confidence_postop | True | True |
