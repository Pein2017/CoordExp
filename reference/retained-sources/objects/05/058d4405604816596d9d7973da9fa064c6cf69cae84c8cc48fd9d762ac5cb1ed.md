# Reviewed unlabeled annotations

`annotations.jsonl` contains the current eleven training-image records. All
original fields and 169 GT objects are unchanged. The added `unlabeled` list
contains 77 confirmed separate physical owners with reviewed full bbox geometry:
63 prior owners and 14 third-fit additions. No unreviewed proposal is a positive.

Each item has `stable_owner_id`, native `bbox_2d` coordinate tokens, explicit
`bbox_2d_bins_1000`, `physical_status`, `geometry_status`, separate `class_status`,
and locally bound visual/decision provenance. The 18 unknown classes keep
`category_name` and `desc` null; known geometry does not authorize class CE.

`review-source-index.json` maps each owner to its actual recorded decision,
original image, bbox overlay, crop and immutable source hashes. Root acceptance
is recorded in `root-acceptance.json`. The seven reviewed but unresolved
candidates remain in `../third-fit-root-rulings-v1/reviewed-pending.jsonl` with
positive annotation disabled. Full current raw review results are in
`../third-fit-review-extraction-v1/full-review-results.jsonl`.

This is the enriched annotation artifact. It does not modify the currently
running fourth-fit manifest, its frozen 232-owner target, or retroactive metrics.
The user requests finishing that round's training/evaluation and then stopping.
