# Coord Family Comparison Interim Metrics

## Matched val200 Families

| Family | Records | bbox_AP | bbox_AP50 | F1@0.50 full micro | TP | FP | FN | Pred total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_xyxy_merged | 200 | 0.0549 | 0.1405 | 0.1407 | 213 | 1371 | 1231 | 1625 |
| hard_soft_ce_2b | 200 | 0.0568 | 0.1407 | 0.1579 | 212 | 1030 | 1232 | 1287 |
| raw_text_xyxy_pure_ce | 200 | 0.3440 | 0.4500 | 0.5900 | 749 | 346 | 695 | 1115 |
| cxcywh_pure_ce | 200 | 0.2725 | 0.4003 | 0.4653 | 740 | 997 | 704 | 1774 |
| cxcy_logw_logh_pure_ce | 200 | 0.2792 | 0.4292 | 0.5083 | 762 | 792 | 682 | 1586 |
| center_parameterization | 200 | 0.4221 | 0.6007 | 0.6108 | 947 | 710 | 497 | 1724 |
