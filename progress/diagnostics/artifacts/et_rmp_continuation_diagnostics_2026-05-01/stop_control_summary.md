# Core6 Stop-Control Sweep Summary

| model | variant | tp50 | fp50 | fn50 | pred_objects_parsed | recovered_baseline_fn_gt | lost_baseline_tp_gt | raw_invalid_json_count | empty_pred_count | hit_max_new_tokens_count | generated_token_mean | dup_pair_iou85_same_desc | dup_max_cluster_iou85_same_desc | bbox_AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base1332 | baseline | 55 | 19 | 32 | 77 | 0 | 0 | 0 | 0 | 0 | 447.6667 | 0 | 0 | 0.4590 |
| base1332 | suppress_first_close | 9 | 9 | 78 | 18 | 0 | 46 | 5 | 5 | 0 | 456.3333 | 0 | 0 | 0.0378 |
| base1332 | steer_array_b1 | 0 | 0 | 87 | 0 | 0 | 55 | 6 | 6 | 0 | 447.6667 | 0 | 0 | 0.0000 |
| base1332 | suppress_all_terminators | 9 | 11 | 78 | 20 | 0 | 46 | 5 | 5 | 6 | 3084 | 0 | 0 | 0.0362 |
| et_rmp_step300 | baseline | 42 | 12 | 45 | 59 | 0 | 0 | 0 | 0 | 0 | 315 | 0 | 0 | 0.4399 |
| et_rmp_step300 | suppress_first_close | 11 | 3 | 76 | 17 | 0 | 31 | 4 | 4 | 0 | 315 | 0 | 0 | 0.1408 |
| et_rmp_step300 | steer_array_b1 | 0 | 0 | 87 | 0 | 0 | 42 | 6 | 6 | 0 | 315 | 0 | 0 | 0.0000 |
| et_rmp_step300 | suppress_all_terminators | 21 | 7 | 66 | 31 | 0 | 21 | 3 | 3 | 4 | 2171.6667 | 0 | 0 | 0.1884 |
