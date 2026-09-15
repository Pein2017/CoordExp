# COCO22 cumulative expansion: authorized preparation contract

## Authority and state

Root lead owns scope, admission, execution coordination and final acceptance.
The latest user conversation authorizes the following22-image stage and its
conditional Source control. The user requested a handoff before context
compression; no new data collection, annotation mutation or GPU launch occurred
in the handoff turn. Current lifecycle is in [state.json](state.json).
This outline is approved in research direction; image IDs, new teacher contents
and production execution bindings remain to be prepared and frozen before launch.

Latest user rulings, superseding the earlier two-hour/five-hour proposals:

- Extra verified, in-scope real objects are accepted; manual visual review is
  necessary. The user explicitly chose this rule.
- Accepted newly discovered unlabeled owners must be written into the matching
  original-format JSONL image record's `unlabeled` field, following the existing
  export/provenance convention. Reports alone are insufficient.
- No experiment wall-time cap. Maximize useful throughput with all8 GPUs.
- Conditional Source joint-fit is preauthorized; do not ask again at its trigger.
- The fixed scientific dose remains256 full-cohort updates per arm. Removing
  hour limits does not authorize an unbounded dose/seed/architecture sweep.

Pro's advice is background, not authority. The source packet is
`/data/CoordExp/.codex/attachments/8c9a8099-f5f0-4381-a656-b4a72b3a39e2/pasted-text.txt`,
SHA256 `af9d786d648bd60c299ed041a3bc7ae4ca602211f0ed8b1013ecc6b2fa6752f2`.
User rulings above take precedence over that packet's5-hour cap and80-GPU-hour
worst-case illustration. Predecessor [CE results](../2026-09-15-coco227-ce-normalization/results.md)
remain closed and unchanged in scientific meaning.

## Question and contrast

Can the completed227-owner continuation model absorb11 new image conditions
while preserving all old11-image tasks, using full replay and the same recipe?
This is cumulative joint fitting with full old-data replay. Generalization and
memory-limited continual learning are outside the claim.

Main: latest COCO227 **Sample arm final new-step256** adapter, fresh optimizer.
Conditional control: original `geo_sorted_xy` step2444 DoRA adapter with the
same frozen model/embedding components, teacher and full256-update dose.
Run the control only if the main arm finishes its dose and remains incomplete
under the criteria below, after excluding teacher, execution, loading and
truncation defects. Pending physical review is not a learning failure.
If main succeeds, stop this unit without a control. A failed main and successful
control implicate initialization/optimization path under this recipe; neither
outcome alone proves capacity limits or irreversible plasticity loss.

## Cohort and teacher

-22 total images: the original11 plus11 different images from the same COCO
 train2017 source, outside this specialized fitting history and near-duplicates.
 Verify the fitting-history exclusion before claiming novelty; no claim that
 the base model or original Source has never encountered those images.
- Match rough object count/density, same-class repetition, small-object share
 and teacher-length range. Use annotations/metadata as selection proxies, with
 declared teacher-quality exclusion rules. Freeze selection before training
 outcomes; do not replace images because the model fails or already succeeds.
- Preserve the old227 teacher sequences exactly: tokens, descriptions, boxes,
 masks, order and EOS. New owner count follows visual review; do not force227
 new owners. New teachers use the same compact protocol and fixed order rule.
- Trusted GT plus independently reviewed exploration candidates supplies the
 new teacher. Current greedy output alone cannot define the new obligations.
- Freeze teacher and owner identities before training. Newly verified owners
 discovered during evaluation update the annotation ledger immediately, and
 are candidate obligations for the next teacher version; they do not silently
 refresh this run or change its target denominator.
- The19 historical category-unknown and2 known non-COCO owners stay in the full
 ledger outside the active COCO-80 teacher. Their blanket adjudication is not
 a prerequisite to expanding this versioned task.

## Objective, dose and runtime

For each image i, use active-token mean CE plus0.01 times its existing per-image
complete-box expected-axis-validity hinge (margin1/999). Average over22 images:
`L = (1/22) sum_i [S_i/T_i + 0.01 H_i]`.
Binary masks affect numerator and denominator; EOS supervised; no guessed class
text in a supposedly trusted teacher. Preserve the tested numerical recipe.

- Same language DoRA surface; base/vision/embedding/lm_head frozen.
- Fresh AdamW, seed42, lr1e-5, betas(.9,.999), eps1e-8, wd0, foreachFalse, clip1.
-256 updates, every update covers all22 images:5632 logical image exposures
 per arm,256 exposures/image. No learning-rate doubling to compensate for1/22.
- Use8 GPUs for the main arm. Verify uneven3/3/3/3/3/3/2/2 rank contributions
 form a true global22-image mean; clip/step only after the full gradient sum.
- Qualify batching and checkpointing against actual per-image loss/gradients
 with predeclared quantity-appropriate tolerances. Measure throughput, padding,
 rank imbalance, memory, token counts and actual model calls. Favor faster
 correct execution; idle memory or utilization alone is not a throughput proof.
- Read back the22-image cold step0, then8/16/32/64/128/256:154 primary requests
 per arm if fully run. If step0 already completes the entire task, record
 zero-update completion instead of manufacturing a training-success claim.
- Natural greedy on original image and empty assistant prefix; inherit exact
 prompt/media/coordinate/EOS/decode identity, including the current3084-token
 per-request cap. Teacher sequences must fit; do not silently widen decode
 budgets to admit over-budget data.
- No2h/5h or inherited7200s experiment kill cap. Monitoring expiry is an
 observation deadline, not a producer stop or algorithm failure. Operational
 crash/OOM/hang handling remains necessary. Jobs run in named tmux with durable
 identities, logs, receipts and missing-only recovery.

## Success and physical review: user-accepted semantics

Publish two separate decisions at each saved checkpoint:

1. **Frozen task completion:** all old227 and new trusted targets are retained/
   covered with correct descriptions. Match each image once (class-agnostic,
   cardinality-first, IoU>=.5), then project old/new and old218/prior-new9
   partitions from that same assignment. IoU>=.8 remains a fixed diagnostic.
2. **Complete-output review closed:** no confirmed false objects, repeats,
   malformed/invalid geometry, wrong/out-of-scope classes, unresolved output
   identity/category, or EOS/cap problems. An additional verified real COCO-80
   owner with valid geometry is permitted and must be recorded in `unlabeled`.

Continue reporting original annotation-relative F1 with all valid predictions
in its denominator. F1<1 from a verified extra real object does not itself make
the main arm fail or trigger Source. Unknown is not confirmed FP, but output
review remains open until adjudicated. An actual observed category-unknown
owner may need targeted review; this does not require resolving all historical19.

The sustained-completion milestone is the earliest scheduled checkpoint for
which BOTH decisions hold and continue to hold at all later saved checkpoints.
Record interim old-owner losses/recovery; do not halt merely on transient
regression. No claim about unsaved intervals or exact convergence time.

## Required JSONL writeback

Use the established per-image `objects` + `unlabeled` schema. Preserve original
GT and unrelated original record fields. Preserve stable owner IDs and previous
accepted owner records. Deduplicate by physical identity as well as stable ID;
coordinate closeness is only candidate evidence.

Every accepted non-GT owner records `stable_owner_id`, `bbox_2d` native coord
strings, `bbox_2d_bins_1000` (xyxy0..999), `coordinate_convention`, `category_id`
(historically null), `category_name`, `desc`, `class_status`, `physical_status`,
`geometry_status`, reference fields and `provenance`.

Existing literals: `physical_status="valid_unlabeled"`,
`geometry_status="reasonable"`; class is `verified` only after class review.
A real physical owner with unresolved category remains `class_status="unknown"`,
`category_name=null`, `desc=null`, outside trusted category supervision.
Physical-identity uncertainty alone must not be promoted to valid_unlabeled.

Review is per-owner visual examination of original image, bbox overlay and crop,
plus a persisted decision and lead admission. Automated matches/proposals alone
are not manual verification. Follow the previous native visual-review/root-
ruling workflow; ordinary resolved owner decisions do not require repeated
user permission. Keep genuinely unresolved cases neutral and explicit.

The current enriched dataset is versioned. Extend/update the corresponding
original-format image record in the current annotation JSONL and publish a
new version with predecessor identity/receipt. Do not leave additions only in
notes or detached proposals. Retain prior hash-bound snapshots and frozen
training banks; this is the existing writeback convention, not refusal to update
annotations. The source raw public COCO dataset is not silently rewritten.

## Next action and stop

Prepare the cohort/teacher ledger, exact bindings, measured execution layout and
review-ready launch packet. Root admits these then launches within this already
approved scope; no repeated confirmation of the above rulings is needed.
Stop at the bounded main result, or the conditional matched Source result if
triggered. Further doubling, label-policy changes, extra doses/seeds and public
publication require their own next-stage decision.
