# Manual Audit Pack v1

This pack is the recommended first-pass human audit for the unmatched proposal verifier study.

Scope:
- runs included: t=0.5 and t=0.7 only
- checkpoints included: ul-res_1024-ckpt_300_merged and ul-res_1024-v2-ckpt_300_merged
- full set: 96 proposals
- priority set: 48 proposals (top 12 per run)

Recommended labeling schema:
- real_visible_object
- duplicate_like
- wrong_location
- dead_or_hallucinated
- uncertain

Recommended workflow:
1. Open `index_recommended96.html` in a browser for the main 96-sample review set.
2. If you want the fastest first pass, start with `manual_audit_priority48.csv`.
3. For the full recommended study set, fill `manual_audit_recommended96.csv`.
4. Only use `manual_audit_full96.csv` if you want the larger exploratory pool.
5. Fill `audit_label` and optionally `audit_notes`.

Important:
- `nearest_gt_iou` is weak evidence only, not a gold label.
- High `counterfactual` can still correspond to oversized boxes. Use the overlay.
- Duplicate handling is a separate design surface; still mark `duplicate_like` when appropriate.
