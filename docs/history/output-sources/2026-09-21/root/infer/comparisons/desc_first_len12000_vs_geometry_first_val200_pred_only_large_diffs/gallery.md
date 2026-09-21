# Desc-first vs Geometry-first Large-Difference Prediction Gallery

Color semantics: green = exact-description match at IoU >= 0.5; red = desc-first-only; yellow = geometry-first-only. No GT boxes are drawn.

Decode caveat: desc-first existing val200 artifact is temp0/rp1.00; geometry-first artifact is temp0/rp1.10 diagnostic. Both are checkpoint-928 selected val200 artifacts.

| rank | image_id | matched | desc-only | geom-only | reason |
|---:|---:|---:|---:|---:|---|
| 0 | 17959 | 7 | 143 | 12 | largest total disagreement; desc-first dense overprediction |
| 1 | 1761 | 2 | 5 | 119 | largest geometry-first-only burst |
| 2 | 12120 | 4 | 111 | 11 | desc-first dense overprediction |
| 3 | 1268 | 7 | 67 | 23 | desc-first-heavy mixed disagreement |
| 4 | 4134 | 16 | 49 | 37 | large balanced disagreement; both checkpoints dense |
| 5 | 2685 | 16 | 20 | 60 | geometry-first-heavy mixed disagreement |
| 6 | 8762 | 3 | 38 | 17 | low-match desc-first-heavy disagreement |
| 7 | 18380 | 25 | 19 | 33 | geometry-first-heavy crowded scene |
| 8 | 14038 | 9 | 29 | 23 | balanced disagreement with many objects |
| 9 | 19109 | 16 | 14 | 35 | geometry-first-heavy moderate crowd |
| 10 | 632 | 9 | 16 | 22 | balanced medium disagreement |
| 11 | 2299 | 6 | 19 | 17 | balanced low-imbalance disagreement |
| 12 | 3255 | 0 | 32 | 0 | geometry-first empty/invalid row versus desc-first predictions |
| 13 | 3934 | 12 | 22 | 0 | desc-first-only extras with geometry-first compact output |
| 14 | 19042 | 1 | 20 | 1 | low-match desc-first-only burst |
| 15 | 5586 | 4 | 20 | 4 | desc-first-heavy moderate disagreement |
| 16 | 13923 | 10 | 2 | 15 | geometry-first-only extras with high overlap core |
| 17 | 12670 | 17 | 12 | 12 | equal pred counts but different boxes/descriptions |

## 00 image 17959

matched=7 desc_only=143 geom_only=12 total_unmatched=155

![image 17959](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0000_image_000000017959.png)

## 01 image 1761

matched=2 desc_only=5 geom_only=119 total_unmatched=124

![image 1761](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0001_image_000000001761.png)

## 02 image 12120

matched=4 desc_only=111 geom_only=11 total_unmatched=122

![image 12120](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0002_image_000000012120.png)

## 03 image 1268

matched=7 desc_only=67 geom_only=23 total_unmatched=90

![image 1268](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0003_image_000000001268.png)

## 04 image 4134

matched=16 desc_only=49 geom_only=37 total_unmatched=86

![image 4134](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0004_image_000000004134.png)

## 05 image 2685

matched=16 desc_only=20 geom_only=60 total_unmatched=80

![image 2685](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0005_image_000000002685.png)

## 06 image 8762

matched=3 desc_only=38 geom_only=17 total_unmatched=55

![image 8762](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0006_image_000000008762.png)

## 07 image 18380

matched=25 desc_only=19 geom_only=33 total_unmatched=52

![image 18380](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0007_image_000000018380.png)

## 08 image 14038

matched=9 desc_only=29 geom_only=23 total_unmatched=52

![image 14038](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0008_image_000000014038.png)

## 09 image 19109

matched=16 desc_only=14 geom_only=35 total_unmatched=49

![image 19109](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0009_image_000000019109.png)

## 10 image 632

matched=9 desc_only=16 geom_only=22 total_unmatched=38

![image 632](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0010_image_000000000632.png)

## 11 image 2299

matched=6 desc_only=19 geom_only=17 total_unmatched=36

![image 2299](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0011_image_000000002299.png)

## 12 image 3255

matched=0 desc_only=32 geom_only=0 total_unmatched=32

![image 3255](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0012_image_000000003255.png)

## 13 image 3934

matched=12 desc_only=22 geom_only=0 total_unmatched=22

![image 3934](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0013_image_000000003934.png)

## 14 image 19042

matched=1 desc_only=20 geom_only=1 total_unmatched=21

![image 19042](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0014_image_000000019042.png)

## 15 image 5586

matched=4 desc_only=20 geom_only=4 total_unmatched=24

![image 5586](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0015_image_000000005586.png)

## 16 image 13923

matched=10 desc_only=2 geom_only=15 total_unmatched=17

![image 13923](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0016_image_000000013923.png)

## 17 image 12670

matched=17 desc_only=12 geom_only=12 total_unmatched=24

![image 12670](/data/CoordExp/outputs/infer/comparisons/desc_first_len12000_vs_geometry_first_val200_pred_only_large_diffs/large_diff_pred_only_0017_image_000000012670.png)
