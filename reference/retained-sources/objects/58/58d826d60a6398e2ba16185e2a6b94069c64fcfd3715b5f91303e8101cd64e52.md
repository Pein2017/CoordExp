# Candidate-row score delta summary

This is paired score reporting only. Candidate rows are not normalized into one shared probability distribution. The terminal margin is reported separately from row scores.

## Validation

- Both scoring receipts reference the same manifest hash and unit identifier.
- Image identifiers, boundary identifiers, candidate identifiers, prefix token hashes, and score metric fields matched the manifest.
- Both runs used full precision (`fp32`) model parameters and score accumulation.

## Paired results

Primary candidate ranks use full-row mean log probability (higher is better), with full-row sum as the tie-breaker.

| Pair | Score checkpoint | Observed alternate owner | Alternate-owner relative sum change | Favours alternate? | Left ranks | Right ranks | Terminal margin change |
|---|---|---|---:|---|---|---|---:|
| 18380::common_objects_depth_6::sorted::current_sorted_rollout_order_vs_seeded_shuffle_earlier_rows_with_fixed_final_two | sorted | entity-gt_0018 | 1.165058 | True | L1/R2 | L2/R1 | -0.124973 |
| 18380::common_objects_depth_6::sorted::current_sorted_rollout_order_vs_seeded_shuffle_earlier_rows_with_fixed_final_two | random | entity-gt_0018 | -0.050805 | False | L1/R2 | L1/R2 | -0.082022 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_historical_random_relative_order_with_fixed_final_two | sorted | entity-gt_0010 | 1.219803 | True | L1/R2 | L1/R2 | 0.357466 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_historical_random_relative_order_with_fixed_final_two | random | entity-gt_0010 | -0.199268 | False | L3/R1 | L3/R1 | 0.093454 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_one | sorted | entity-gt_0010 | 4.063498 | True | L1/R2 | L2/R1 | 0.995920 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_one | random | entity-gt_0010 | 0.112284 | True | L3/R1 | L3/R1 | 0.006479 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_two | sorted | n/a | n/a | ambiguous | n/a | n/a | 0.488411 |
| 19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_two | random | n/a | n/a | ambiguous | n/a | n/a | 0.067606 |
| 19109::common_objects_depth_6::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_two | sorted | entity-gt_0010 | 0.037067 | True | L3/R2 | L3/R2 | 0.121267 |
| 19109::common_objects_depth_6::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_two | random | entity-gt_0010 | 0.154117 | True | L2/R1 | L2/R1 | 0.012196 |
| 19109::common_objects_depth_6::random::current_sorted_rollout_order_vs_seeded_shuffle_earlier_rows_with_fixed_final_two | sorted | entity-gt_0010 | 0.126938 | True | L3/R2 | L3/R2 | 0.104481 |
| 19109::common_objects_depth_6::random::current_sorted_rollout_order_vs_seeded_shuffle_earlier_rows_with_fixed_final_two | random | entity-gt_0010 | -0.030546 | False | L2/R1 | L2/R1 | 0.006695 |

## Ambiguities

- `19109::common_objects_depth_10::random::current_sorted_rollout_order_vs_reverse_earlier_rows_with_fixed_final_two`: no_greedy_right_owner; sampled-owner fallback candidates are not treated as an observed greedy alternate.

The JSON companion contains every candidate's right-minus-left delta for full row, description, each coordinate, geometry, and closure, plus the complete rankings.
