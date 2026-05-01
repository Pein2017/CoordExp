# Core6 FN Latent Probe Summary

- completed job summaries: 18
- candidate score rows: 13398
- pair rows: 10518

## Close vs Continue-With-FN

| reference_label | scoring_label | prefix_mode | n | mean_continue_first_mass_pair | frac_continue_first_margin_positive | mean_continue_minus_close_first_logprob |
| --- | --- | --- | --- | --- | --- | --- |
| base_rp105_t020_s029 | base1332 | empty | 27 | 1.0000 | 1.0000 | 144.6528 |
| base_rp105_t020_s029 | base1332 | oracle_before_fn | 27 | 0.9815 | 1.0000 | 44.2020 |
| base_rp105_t020_s029 | base1332 | oracle_late_all_except_fn | 27 | 0.5152 | 0.6296 | 0.0629 |
| base_rp105_t020_s029 | base1332 | oracle_matched_gt | 27 | 0.5001 | 0.6296 | 0.0002 |
| base_rp105_t020_s029 | base1332 | self_pred_all | 27 | 0.5001 | 0.6296 | 0.0002 |
| base_rp105_t020_s029 | et_rmp_step300 | empty | 27 | 1.0000 | 1.0000 | 102.5000 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_before_fn | 27 | 0.8816 | 0.9630 | 27.6432 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_late_all_except_fn | 27 | 0.5038 | 0.8889 | 0.0151 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_matched_gt | 27 | 0.5106 | 0.8889 | 0.0426 |
| base_rp105_t020_s029 | et_rmp_step300 | self_pred_all | 27 | 0.5017 | 0.7407 | 0.0069 |
| et_rp115_t020_s017 | base1332 | empty | 32 | 1.0000 | 1.0000 | 145.5664 |
| et_rp115_t020_s017 | base1332 | oracle_before_fn | 32 | 0.9841 | 1.0000 | 48.6501 |
| et_rp115_t020_s017 | base1332 | oracle_late_all_except_fn | 32 | 0.5495 | 0.6250 | 1.5232 |
| et_rp115_t020_s017 | base1332 | oracle_matched_gt | 32 | 0.7545 | 1.0000 | 8.2213 |
| et_rp115_t020_s017 | base1332 | self_pred_all | 32 | 0.9368 | 1.0000 | 8.6555 |
| et_rp115_t020_s017 | et_rmp_step300 | empty | 32 | 1.0000 | 1.0000 | 103.8281 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_before_fn | 32 | 0.8982 | 0.9688 | 30.6781 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_late_all_except_fn | 32 | 0.5348 | 0.9688 | 0.2425 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_matched_gt | 32 | 0.8508 | 1.0000 | 2.9721 |
| et_rp115_t020_s017 | et_rmp_step300 | self_pred_all | 32 | 0.5531 | 1.0000 | 0.2153 |
| et_rp118_t035_s017 | base1332 | empty | 37 | 1.0000 | 1.0000 | 83.0695 |
| et_rp118_t035_s017 | base1332 | oracle_before_fn | 37 | 0.9865 | 1.0000 | 27.5631 |
| et_rp118_t035_s017 | base1332 | oracle_late_all_except_fn | 37 | 0.5032 | 1.0000 | 0.0130 |
| et_rp118_t035_s017 | base1332 | oracle_matched_gt | 37 | 0.6258 | 1.0000 | 0.8414 |
| et_rp118_t035_s017 | base1332 | self_pred_all | 37 | 0.8867 | 1.0000 | 3.8003 |
| et_rp118_t035_s017 | et_rmp_step300 | empty | 37 | 1.0000 | 1.0000 | 59.0154 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_before_fn | 37 | 0.9302 | 1.0000 | 18.3887 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_late_all_except_fn | 37 | 0.5332 | 1.0000 | 0.1613 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_matched_gt | 37 | 0.8226 | 1.0000 | 1.9166 |
| et_rp118_t035_s017 | et_rmp_step300 | self_pred_all | 37 | 0.6620 | 1.0000 | 0.9294 |


## Visual Sensitivity For Continue-With-FN

| reference_label | scoring_label | prefix_mode | mask_condition | n | mean_raw_mean_drop_when_masked | frac_raw_mean_drop_positive |
| --- | --- | --- | --- | --- | --- | --- |
| base_rp105_t020_s029 | base1332 | empty | mask_fn_bbox | 27 | 0.1029 | 0.7778 |
| base_rp105_t020_s029 | base1332 | empty | mask_other_gt_bbox | 27 | -0.0085 | 0.3704 |
| base_rp105_t020_s029 | base1332 | oracle_before_fn | mask_fn_bbox | 27 | 0.1205 | 0.8519 |
| base_rp105_t020_s029 | base1332 | oracle_before_fn | mask_other_gt_bbox | 27 | 0.0253 | 0.4444 |
| base_rp105_t020_s029 | base1332 | oracle_late_all_except_fn | mask_fn_bbox | 27 | 0.0715 | 0.7407 |
| base_rp105_t020_s029 | base1332 | oracle_late_all_except_fn | mask_other_gt_bbox | 27 | 0.0161 | 0.4444 |
| base_rp105_t020_s029 | base1332 | oracle_matched_gt | mask_fn_bbox | 27 | 0.0718 | 0.7778 |
| base_rp105_t020_s029 | base1332 | oracle_matched_gt | mask_other_gt_bbox | 27 | 0.0185 | 0.4444 |
| base_rp105_t020_s029 | base1332 | self_pred_all | mask_fn_bbox | 27 | 0.0601 | 0.6667 |
| base_rp105_t020_s029 | base1332 | self_pred_all | mask_other_gt_bbox | 27 | 0.0187 | 0.4444 |
| base_rp105_t020_s029 | et_rmp_step300 | empty | mask_fn_bbox | 27 | 0.1551 | 0.8148 |
| base_rp105_t020_s029 | et_rmp_step300 | empty | mask_other_gt_bbox | 27 | 0.0088 | 0.2963 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_before_fn | mask_fn_bbox | 27 | 0.1900 | 0.8519 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_before_fn | mask_other_gt_bbox | 27 | 0.0162 | 0.3704 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_late_all_except_fn | mask_fn_bbox | 27 | 0.1946 | 0.8889 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_late_all_except_fn | mask_other_gt_bbox | 27 | 0.0284 | 0.6296 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_matched_gt | mask_fn_bbox | 27 | 0.1931 | 0.8519 |
| base_rp105_t020_s029 | et_rmp_step300 | oracle_matched_gt | mask_other_gt_bbox | 27 | 0.0336 | 0.4815 |
| base_rp105_t020_s029 | et_rmp_step300 | self_pred_all | mask_fn_bbox | 27 | 0.1836 | 0.8889 |
| base_rp105_t020_s029 | et_rmp_step300 | self_pred_all | mask_other_gt_bbox | 27 | 0.0347 | 0.5926 |
| et_rp115_t020_s017 | base1332 | empty | mask_fn_bbox | 32 | 0.1517 | 0.8125 |
| et_rp115_t020_s017 | base1332 | empty | mask_other_gt_bbox | 32 | -0.0174 | 0.3438 |
| et_rp115_t020_s017 | base1332 | oracle_before_fn | mask_fn_bbox | 32 | 0.2095 | 0.9062 |
| et_rp115_t020_s017 | base1332 | oracle_before_fn | mask_other_gt_bbox | 32 | 0.0312 | 0.4688 |
| et_rp115_t020_s017 | base1332 | oracle_late_all_except_fn | mask_fn_bbox | 32 | 0.0872 | 0.6562 |
| et_rp115_t020_s017 | base1332 | oracle_late_all_except_fn | mask_other_gt_bbox | 32 | 0.0098 | 0.3438 |
| et_rp115_t020_s017 | base1332 | oracle_matched_gt | mask_fn_bbox | 32 | 0.1312 | 0.7500 |
| et_rp115_t020_s017 | base1332 | oracle_matched_gt | mask_other_gt_bbox | 32 | 0.0185 | 0.5938 |
| et_rp115_t020_s017 | base1332 | self_pred_all | mask_fn_bbox | 32 | 0.1462 | 0.7500 |
| et_rp115_t020_s017 | base1332 | self_pred_all | mask_other_gt_bbox | 32 | 0.0211 | 0.5938 |
| et_rp115_t020_s017 | et_rmp_step300 | empty | mask_fn_bbox | 32 | 0.2402 | 0.8125 |
| et_rp115_t020_s017 | et_rmp_step300 | empty | mask_other_gt_bbox | 32 | 0.0143 | 0.4062 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_before_fn | mask_fn_bbox | 32 | 0.2776 | 0.8750 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_before_fn | mask_other_gt_bbox | 32 | 0.0221 | 0.4688 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_late_all_except_fn | mask_fn_bbox | 32 | 0.2693 | 0.9062 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_late_all_except_fn | mask_other_gt_bbox | 32 | 0.0338 | 0.5625 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_matched_gt | mask_fn_bbox | 32 | 0.2671 | 0.8750 |
| et_rp115_t020_s017 | et_rmp_step300 | oracle_matched_gt | mask_other_gt_bbox | 32 | 0.0082 | 0.4375 |
| et_rp115_t020_s017 | et_rmp_step300 | self_pred_all | mask_fn_bbox | 32 | 0.2773 | 0.9062 |
| et_rp115_t020_s017 | et_rmp_step300 | self_pred_all | mask_other_gt_bbox | 32 | 0.0223 | 0.4375 |
| et_rp118_t035_s017 | base1332 | empty | mask_fn_bbox | 37 | 0.1150 | 0.8108 |
| et_rp118_t035_s017 | base1332 | empty | mask_other_gt_bbox | 37 | -0.0490 | 0.3514 |
| et_rp118_t035_s017 | base1332 | oracle_before_fn | mask_fn_bbox | 37 | 0.1908 | 0.8919 |
| et_rp118_t035_s017 | base1332 | oracle_before_fn | mask_other_gt_bbox | 37 | -0.0014 | 0.5135 |
| et_rp118_t035_s017 | base1332 | oracle_late_all_except_fn | mask_fn_bbox | 37 | 0.0394 | 0.6216 |
| et_rp118_t035_s017 | base1332 | oracle_late_all_except_fn | mask_other_gt_bbox | 37 | -0.0079 | 0.2432 |
| et_rp118_t035_s017 | base1332 | oracle_matched_gt | mask_fn_bbox | 37 | 0.0510 | 0.6757 |
| et_rp118_t035_s017 | base1332 | oracle_matched_gt | mask_other_gt_bbox | 37 | -0.0024 | 0.3514 |
| et_rp118_t035_s017 | base1332 | self_pred_all | mask_fn_bbox | 37 | 0.0939 | 0.7838 |
| et_rp118_t035_s017 | base1332 | self_pred_all | mask_other_gt_bbox | 37 | 0.0001 | 0.4865 |
| et_rp118_t035_s017 | et_rmp_step300 | empty | mask_fn_bbox | 37 | 0.1885 | 0.8378 |
| et_rp118_t035_s017 | et_rmp_step300 | empty | mask_other_gt_bbox | 37 | -0.0113 | 0.3784 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_before_fn | mask_fn_bbox | 37 | 0.2280 | 0.8919 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_before_fn | mask_other_gt_bbox | 37 | -0.0159 | 0.3784 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_late_all_except_fn | mask_fn_bbox | 37 | 0.2258 | 0.9459 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_late_all_except_fn | mask_other_gt_bbox | 37 | 0.0023 | 0.4865 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_matched_gt | mask_fn_bbox | 37 | 0.2324 | 0.8919 |
| et_rp118_t035_s017 | et_rmp_step300 | oracle_matched_gt | mask_other_gt_bbox | 37 | -0.0212 | 0.4595 |
| et_rp118_t035_s017 | et_rmp_step300 | self_pred_all | mask_fn_bbox | 37 | 0.2353 | 0.8378 |
| et_rp118_t035_s017 | et_rmp_step300 | self_pred_all | mask_other_gt_bbox | 37 | -0.0170 | 0.3514 |

