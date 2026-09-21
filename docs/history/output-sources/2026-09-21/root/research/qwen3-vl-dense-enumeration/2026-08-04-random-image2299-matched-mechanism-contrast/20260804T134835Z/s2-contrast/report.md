# Image 2299 random versus sorted matched contrast

## Native rollout

- Sorted: 19 TP owners from 24 rows.
- Random: 3 TP owners from 14 rows.
- Random emits 10 exact full-canvas duplicate rows and 11 unmatched rows.
- Retained/gained/lost: {'lost_by_random': 16, 'missed_by_both': 27, 'retained': 3}.

## Identical root context

- Owner-rank 1: sorted 2/46; random 2/46.
- Category-candidate top 3: sorted 4/46; random 3/46.
- Median random-minus-sorted deltas: `{"best_local_candidate_score": 4.441345930099487, "continue_vs_stop_margin": 0.6759589910393515, "exact_anchor_score": 4.471486210823059, "local_concentration": 1.1712346374988556, "owner_margin_to_best": 2.338074564933777, "peak_lift": 2.477595339595858}`.

## Conditional free-box behavior

- Any-context strict owner hit: sorted 20/46; random 4/46.

Binary frozen-threshold FN prevalence is not claimed across checkpoints.
