# Core6 Stop-Control Salvage Summary

Diagnostic-only parse salvage from raw token traces. It repairs terminal syntax or extracts complete emitted object entries, then reruns the same evaluator contract.

| model | variant | tp50 | fp50 | fn50 | bbox_AP | salvaged_pred_objects | official_recovered_baseline_fn_gt | official_lost_baseline_tp_gt | local_recovered_baseline_fn_gt | local_lost_baseline_tp_gt | dup_pair_iou85_same_desc | dup_max_cluster_iou85_same_desc | repair_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base1332 | baseline | 55 | 19 | 32 | None | 77 | 0 | 0 | 0 | 0 | 0 | 1 | {"strict": 6} |
| base1332 | suppress_first_close | 55 | 19 | 32 | None | 77 | 0 | 0 | 0 | 0 | 0 | 1 | {"remove_stray_quote_before_close": 5, "strict": 1} |
| base1332 | steer_array_b1 | 55 | 19 | 32 | None | 77 | 0 | 0 | 0 | 0 | 0 | 1 | {"remove_stray_quote_before_close": 6} |
| base1332 | suppress_all_terminators | 55 | 22 | 32 | None | 80 | 0 | 0 | 0 | 0 | 0 | 1 | {"extract_complete_entries": 5, "strict": 1} |
| et_rmp_step300 | baseline | 42 | 12 | 45 | None | 59 | 0 | 0 | 0 | 0 | 1 | 2 | {"strict": 6} |
| et_rmp_step300 | suppress_first_close | 42 | 12 | 45 | None | 59 | 0 | 0 | 0 | 0 | 1 | 2 | {"extract_complete_entries": 1, "remove_stray_quote_before_close": 3, "strict": 2} |
| et_rmp_step300 | steer_array_b1 | 42 | 12 | 45 | None | 59 | 0 | 0 | 0 | 0 | 1 | 2 | {"extract_complete_entries": 1, "remove_stray_quote_before_close": 5} |
| et_rmp_step300 | suppress_all_terminators | 40 | 12 | 47 | None | 57 | 2 | 4 | 2 | 4 | 1 | 2 | {"extract_complete_entries": 3, "strict": 3} |
