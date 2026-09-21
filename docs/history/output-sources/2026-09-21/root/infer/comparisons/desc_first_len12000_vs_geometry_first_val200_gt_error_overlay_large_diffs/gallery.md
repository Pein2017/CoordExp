# Desc-first vs Geometry-first GT Error Overlay Gallery

Each PNG is a strict 1x2 comparison. Within each checkpoint panel: green = matched prediction, red = false-positive prediction, yellow dashed = false-negative missed GT.

Geometry handling: Rendered from canonical comparison_scenes.json. Raw type+points objects were normalized by src.vis.comparison/canonicalize_gt_vs_pred_record before drawing; this avoids ad hoc bbox_2d-only assumptions.

Decode caveat: desc-first existing val200 artifact is temp0/rp1.00; geometry-first artifact is temp0/rp1.10 diagnostic. Both are checkpoint-928 selected val200 artifacts.

| rank | image_id | desc matched/fp/fn | geom matched/fp/fn | reason |
|---:|---:|---:|---:|---|
| 0 | 17959 | 7/143/17 | 7/12/17 | largest total disagreement; desc-first dense overprediction |
| 1 | 1761 | 2/5/5 | 2/119/5 | largest geometry-first-only burst |
| 2 | 12120 | 3/112/21 | 7/8/17 | desc-first dense overprediction |
| 3 | 1268 | 7/67/4 | 6/24/5 | desc-first-heavy mixed disagreement |
| 4 | 4134 | 14/51/12 | 14/39/12 | large balanced disagreement; both checkpoints dense |
| 5 | 2685 | 9/27/10 | 10/66/9 | geometry-first-heavy mixed disagreement |
| 6 | 8762 | 5/36/2 | 4/16/3 | low-match desc-first-heavy disagreement |
| 7 | 18380 | 30/14/23 | 23/35/30 | geometry-first-heavy crowded scene |
| 8 | 14038 | 9/29/15 | 9/23/15 | balanced disagreement with many objects |
| 9 | 19109 | 14/16/13 | 15/36/12 | geometry-first-heavy moderate crowd |
| 10 | 632 | 3/22/14 | 4/27/13 | balanced medium disagreement |
| 11 | 2299 | 6/19/16 | 6/17/16 | balanced low-imbalance disagreement |
| 12 | 3255 | 3/29/7 | 0/0/10 | geometry-first empty/invalid row versus desc-first predictions |
| 13 | 3934 | 11/23/3 | 11/1/3 | desc-first-only extras with geometry-first compact output |
| 14 | 19042 | 1/20/2 | 1/1/2 | low-match desc-first-only burst |
| 15 | 5586 | 4/20/10 | 5/3/9 | desc-first-heavy moderate disagreement |
| 16 | 13923 | 10/2/9 | 9/16/10 | geometry-first-only extras with high overlap core |
| 17 | 12670 | 14/15/14 | 15/14/13 | equal pred counts but different boxes/descriptions |

## 00 image 17959

desc_first matched=7 fp=143 fn=17 | geometry_first matched=7 fp=12 fn=17

![image 17959](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0000_image_000000017959.png)

## 01 image 1761

desc_first matched=2 fp=5 fn=5 | geometry_first matched=2 fp=119 fn=5

![image 1761](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0001_image_000000001761.png)

## 02 image 12120

desc_first matched=3 fp=112 fn=21 | geometry_first matched=7 fp=8 fn=17

![image 12120](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0002_image_000000012120.png)

## 03 image 1268

desc_first matched=7 fp=67 fn=4 | geometry_first matched=6 fp=24 fn=5

![image 1268](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0003_image_000000001268.png)

## 04 image 4134

desc_first matched=14 fp=51 fn=12 | geometry_first matched=14 fp=39 fn=12

![image 4134](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0004_image_000000004134.png)

## 05 image 2685

desc_first matched=9 fp=27 fn=10 | geometry_first matched=10 fp=66 fn=9

![image 2685](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0005_image_000000002685.png)

## 06 image 8762

desc_first matched=5 fp=36 fn=2 | geometry_first matched=4 fp=16 fn=3

![image 8762](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0006_image_000000008762.png)

## 07 image 18380

desc_first matched=30 fp=14 fn=23 | geometry_first matched=23 fp=35 fn=30

![image 18380](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0007_image_000000018380.png)

## 08 image 14038

desc_first matched=9 fp=29 fn=15 | geometry_first matched=9 fp=23 fn=15

![image 14038](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0008_image_000000014038.png)

## 09 image 19109

desc_first matched=14 fp=16 fn=13 | geometry_first matched=15 fp=36 fn=12

![image 19109](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0009_image_000000019109.png)

## 10 image 632

desc_first matched=3 fp=22 fn=14 | geometry_first matched=4 fp=27 fn=13

![image 632](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0010_image_000000000632.png)

## 11 image 2299

desc_first matched=6 fp=19 fn=16 | geometry_first matched=6 fp=17 fn=16

![image 2299](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0011_image_000000002299.png)

## 12 image 3255

desc_first matched=3 fp=29 fn=7 | geometry_first matched=0 fp=0 fn=10

![image 3255](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0012_image_000000003255.png)

## 13 image 3934

desc_first matched=11 fp=23 fn=3 | geometry_first matched=11 fp=1 fn=3

![image 3934](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0013_image_000000003934.png)

## 14 image 19042

desc_first matched=1 fp=20 fn=2 | geometry_first matched=1 fp=1 fn=2

![image 19042](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0014_image_000000019042.png)

## 15 image 5586

desc_first matched=4 fp=20 fn=10 | geometry_first matched=5 fp=3 fn=9

![image 5586](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0015_image_000000005586.png)

## 16 image 13923

desc_first matched=10 fp=2 fn=9 | geometry_first matched=9 fp=16 fn=10

![image 13923](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0016_image_000000013923.png)

## 17 image 12670

desc_first matched=14 fp=15 fn=14 | geometry_first matched=15 fp=14 fn=13

![image 12670](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_gt_error_overlay_large_diffs/gt_error_overlay_0017_image_000000012670.png)
