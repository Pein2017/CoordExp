# Detection Run Comparison

Baseline: `pure_ce_1932_merged_coco_val_limit200_res768_temp0`

## Summary Metrics

| run | samples | mAP | AR1 | F1@0.50(full) | pred_total | pred_mean | pred_p90 | empty_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pure_ce_1932_merged_coco_val_limit200_res768_temp0 | 200 | 0.2712 | 0.2198 | 0.5638 | 1040.0 | 5.2000 | 11.00 | 0.00% |
| softce_w1_1832_merged_coco_val_limit200_res768_temp0 | 200 | 0.2400 | 0.1924 | 0.5606 | 977.0 | 4.8850 | 12.00 | 0.50% |
| softce_hardce_mixed_1344_merged_coco_val_limit200_res768_temp0_v2 | 200 | 0.4212 | 0.3725 | 0.6479 | 1358.0 | 6.7900 | 16.00 | 0.00% |

## Decode And Confidence Diagnostics

| run | trace_rows | eff_tok_mean | eff_tok_p90 | coord_frac_mean | score_fusion_mean | score_geom_mean | score_desc_mean | tp_fp_gap_fusion | tp_fp_gap_desc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pure_ce_1932_merged_coco_val_limit200_res768_temp0 | 200 | 130.9 | 279 | 0.1521 | 0.3377 | 0.1411 | 0.7965 | 0.0418 | 0.0493 |
| softce_w1_1832_merged_coco_val_limit200_res768_temp0 | 200 | 123.4 | 293 | 0.1520 | 0.2767 | 0.0683 | 0.7629 | 0.0233 | 0.0289 |
| softce_hardce_mixed_1344_merged_coco_val_limit200_res768_temp0_v2 | 200 | 170.5 | 390 | 0.1540 | 0.2730 | 0.0762 | 0.7324 | 0.0543 | 0.1216 |

## Diffs Vs Baseline

### `softce_w1_1832_merged_coco_val_limit200_res768_temp0` vs `pure_ce_1932_merged_coco_val_limit200_res768_temp0`

Per-class AP (delta = other - base), top improved:

| class | delta_ap |
| --- | --- |
| donut | 1.0000 |
| stop sign | 0.1723 |
| train | 0.1716 |
| knife | 0.1332 |
| skateboard | 0.1331 |
| zebra | 0.1293 |
| remote | 0.1119 |
| laptop | 0.0821 |
| pizza | 0.0732 |
| car | 0.0590 |
| bed | 0.0581 |
| handbag | 0.0545 |

Per-class AP (delta = other - base), top degraded:

| class | delta_ap |
| --- | --- |
| giraffe | -1.0000 |
| fire hydrant | -0.4059 |
| carrot | -0.2922 |
| cow | -0.2574 |
| suitcase | -0.2021 |
| frisbee | -0.1860 |
| surfboard | -0.1649 |
| refrigerator | -0.1480 |
| vase | -0.1461 |
| spoon | -0.1188 |
| tennis racket | -0.1178 |
| apple | -0.1158 |

Per-image delta TP_full@0.50 (other - base), top improved:

| file_name | delta_tp |
| --- | --- |
| images/val2017/000000009400.jpg | 5 |
| images/val2017/000000007818.jpg | 4 |
| images/val2017/000000002006.jpg | 3 |
| images/val2017/000000010707.jpg | 3 |
| images/val2017/000000012576.jpg | 3 |
| images/val2017/000000016451.jpg | 3 |
| images/val2017/000000018380.jpg | 3 |
| images/val2017/000000018837.jpg | 3 |
| images/val2017/000000005001.jpg | 2 |
| images/val2017/000000006213.jpg | 2 |
| images/val2017/000000007386.jpg | 2 |
| images/val2017/000000008844.jpg | 2 |

Per-image delta TP_full@0.50 (other - base), top degraded:

| file_name | delta_tp |
| --- | --- |
| images/val2017/000000003845.jpg | -4 |
| images/val2017/000000006040.jpg | -4 |
| images/val2017/000000006471.jpg | -3 |
| images/val2017/000000008277.jpg | -3 |
| images/val2017/000000012639.jpg | -3 |
| images/val2017/000000017714.jpg | -3 |
| images/val2017/000000018575.jpg | -3 |
| images/val2017/000000002157.jpg | -2 |
| images/val2017/000000002431.jpg | -2 |
| images/val2017/000000003156.jpg | -2 |
| images/val2017/000000005600.jpg | -2 |
| images/val2017/000000006954.jpg | -2 |

### `softce_hardce_mixed_1344_merged_coco_val_limit200_res768_temp0_v2` vs `pure_ce_1932_merged_coco_val_limit200_res768_temp0`

Per-class AP (delta = other - base), top improved:

| class | delta_ap |
| --- | --- |
| donut | 1.0000 |
| orange | 0.9000 |
| dog | 0.8505 |
| baseball glove | 0.4923 |
| oven | 0.4455 |
| tv | 0.4455 |
| kite | 0.3645 |
| stop sign | 0.3584 |
| tennis racket | 0.3422 |
| mouse | 0.3356 |
| fire hydrant | 0.3281 |
| sandwich | 0.3190 |

Per-class AP (delta = other - base), top degraded:

| class | delta_ap |
| --- | --- |
| carrot | -0.1305 |
| vase | -0.0945 |
| cell phone | -0.0751 |
| suitcase | -0.0726 |
| apple | -0.0644 |
| truck | -0.0374 |
| cat | -0.0330 |
| umbrella | -0.0277 |
| bird | -0.0132 |
| book | -0.0015 |
| airplane | -0.0010 |
| bear | 0.0000 |

Per-image delta TP_full@0.50 (other - base), top improved:

| file_name | delta_tp |
| --- | --- |
| images/val2017/000000019432.jpg | 11 |
| images/val2017/000000002157.jpg | 8 |
| images/val2017/000000009400.jpg | 7 |
| images/val2017/000000017959.jpg | 7 |
| images/val2017/000000013923.jpg | 6 |
| images/val2017/000000010707.jpg | 5 |
| images/val2017/000000012576.jpg | 5 |
| images/val2017/000000015660.jpg | 5 |
| images/val2017/000000005001.jpg | 4 |
| images/val2017/000000007386.jpg | 4 |
| images/val2017/000000013729.jpg | 4 |
| images/val2017/000000014038.jpg | 4 |

Per-image delta TP_full@0.50 (other - base), top degraded:

| file_name | delta_tp |
| --- | --- |
| images/val2017/000000007281.jpg | -2 |
| images/val2017/000000009891.jpg | -2 |
| images/val2017/000000013546.jpg | -2 |
| images/val2017/000000019109.jpg | -2 |
| images/val2017/000000001675.jpg | -1 |
| images/val2017/000000002149.jpg | -1 |
| images/val2017/000000004134.jpg | -1 |
| images/val2017/000000006040.jpg | -1 |
| images/val2017/000000016010.jpg | -1 |
| images/val2017/000000000285.jpg | 0 |
| images/val2017/000000000724.jpg | 0 |
| images/val2017/000000000776.jpg | 0 |
