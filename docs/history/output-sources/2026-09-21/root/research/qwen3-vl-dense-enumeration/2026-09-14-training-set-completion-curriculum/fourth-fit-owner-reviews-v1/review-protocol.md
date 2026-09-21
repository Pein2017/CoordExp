# Final-round paired single-image review

## User clarification: unmatched-only visual review (authoritative v2)

On 2026-09-15 the user clarified that `view_image` focuses only on unmatched
predictions. Already IoU-matched rows inherit the existing matching result and
do not receive a separate physical-extent visual judgment. This supersedes
the broader matched-row inspection language below. Prior viewed judgments
remain on disk as historical evidence, but cannot veto an inherited match.

`match-inheritance-v2.json` binds the exact rows: existing class-agnostic,
cardinality-first one-to-one IoU >= 0.5 v4 matches, then the same matcher for
remaining predictions against v5-only owners. The existing review selector
already uses 0.5; 0.8 remains diagnostic. Matched owner and reasonable extent
are accepted by this rule. Only remaining valid unmatched rows need original,
bbox overlay and crop review. Invalid parser rows are mechanically retained.
Class certainty remains separate; geometric match alone cannot verify class.
After unmatched owner decisions, recompute route repetition by generated
order. Matched repeats still cover their owner but have masked direct CE.
Add `matching_inheritance` source binding and `basis: iou_matched_inherited`
to inherited decisions. No new GPU work, training, or held-out evaluation.

User requests completing this already-running round and then stopping. No new
training or research arm. Review one assigned image only at immediate parent
third-fit64 and final fourth-fit256. Use original image, generated bbox overlay,
target/reference overlay and context/tight crops for each unresolved distinct
valid geometry. Persist every view interpretation after viewing as local
review-notes.jsonl. All raw rows including parser-dropped rows must survive.
Exact-signature cached physical owner/extent/class may be reused only with
its exact image/geometry/literal source and evidence hashes; recompute current
route duplicate status by physical owner and order. Cached drafts are not
accepted judgments. Class stays separate from geometry-qualified coverage.

Output parent64-review.json and final256-review.json in your single-image
review directory. Schema fourth_fit_paired_owner_review.v1. Top fields:
image_id, checkpoint_step (64 or256), phase (parent64 or final256), source_packet
(path+sha256), target_catalog (v4 path+sha256), annotation_catalog (v5 path+sha256),
status candidate_ready, raw_row_count, decisions flatlist, summary,
new_owner_candidates, validation. Keep a decision per raw prediction ID/order.

Decision fields: prediction_id, generated_order, visual_group_id, owner_id
(stable owner ID from current v5 if known, pending-new ID ornull otherwise),
physical_status true_unique|repeat|false|unknown|invalid_output,
extent reasonable|wrong|unknown, class verified|wrong|unknown,
coverage_eligible (parser valid AND true_unique/repeat AND reasonable AND
member of fixedv4), annotation_coverage_eligible (same using currentv5),
direct_CE {bbox:positive|mask,description:positive|mask}, reason, evidence_paths,
reuse_source if reused. A known unique v5 owner with reasonable geometry can
have positive bbox; description positive only when verified. Repeats are
always both CE masked, but one reasonable repeat can qualify ownercoverage.
Unknown/false/invalid/wrongextent rows both masked. Do not canonicalize rawaxes.

Summary fields: target_owner_count, covered_owner_ids, missing_owner_ids,
covered_owner_count, missing_owner_count (fixedv4); annotation_target_owner_count,
annotation_covered_owner_ids, annotation_missing_owner_ids (currentv5);
physical_repeat_row_count, confirmed_false_row_count, physical_unknown_row_count,
invalid_output_row_count, class_wrong_row_count, class_unknown_row_count,
raw_stop_reason, cap_debt. For final256 also include retained_parent_owner_ids,
lost_parent_owner_ids,newly_covered_fixed_owner_ids from your parent64 review.
Report raw physical repetition even for changed boxes below IoU.95. Unknown
is not FP. All complete-image stop/structural debt remains visible.

Targetv4=232 fixed trainingowners. Currentv5=246 known owners, adding14 root-
reviewed objects in annotationJSONL v3 (169GT+77unlabeled). Keep both denominators
explicit; currentv5 does not retroactively change training's primary contrast.
Use current full geometry references and separate class masking. Newconfirmed
objects outsidev5 must have candidate fullbbox/categorycertainty and evidence
saved for root admission into annotationJSONL before closing this eval. Never
admit or alter sharedcatalog yourself. No whole-stage pass from oneimage.

Validate all raw IDs/group memberships/source hashes, both owner partitions,
CE masks, native EOS/cap, and viewed evidence before returning candidate.
