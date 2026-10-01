# Physical evaluation: what is being counted?

## Question and evidence unit

The unit of inference is an explicitly defined physical owner or declared group, at the actually executed image scale. Category, geometry, physical identity, visibility and annotation coverage are separate axes. Algorithmic unmatched rows and near-IoU rows are nomination signals, not verified truth.

**Question:** are we seeing real owner gains/losses, annotation-relative matching changes, category/extent changes, or a changed review policy? A dataset annotation and a physical instance are related but not interchangeable.

## Evidence chain

Human-audited rare-object genealogy (catalog:2026-07-16-human-audited-rare-object-trajectory-genealogy) separated semantic support from part-sized, multi-instance and axis-wise box failures. FP visual distribution (catalog:2026-09-09-fp-visual-distribution) later separated strict repetition, visible objects lacking current GT coverage, class/extent errors and unresolved cases. These sampled audits are not an exhaustive physical census.

The blind physical accounting (catalog:2026-09-14-label-vs-compilation) is a decisive measurement counterexample: annotation-relative F1 could improve while reviewed atomic-owner presence decreased. Many lost old predictions were GT50-unmatched, but that status alone does not prove that the instance had no annotation. Threshold, extent, class and assignment also matter. A union-of-predictions audit cannot see instances all compared models missed.

The frozen supply audit (catalog:2026-09-14-label-vs-compilation) records nomination, candidate-successor and admission funnels. A no-nomination image or unknown-neutral HOLD group is not a missing-label count. Evaluation bias, information available to today's acquisition, and the causal effect of historical SFT annotation omissions are separate questions.

## Evaluation identities must remain explicit

The accepted fourth-fit report (catalog:2026-09-14-training-set-completion-curriculum) inherited class-agnostic one-to-one IoU≥0.5 matches and visually reviewed only residual unmatched valid predictions. Earlier stricter matched-row visual judgments remain historical observations; they do not retroactively veto that experiment's inherited matches. This is an experiment-specific convention, not a timeless physical truth rule.

The fresh128 accepted result (catalog:2026-09-17-readout-norm-fresh128) also uses class-agnostic one-to-one IoU≥0.5, whereas its human comparison export uses class-aware matching. Keep both named. The renderer's same-category pixel-IoU≥.30 pair count includes matched/matched pairs; the study's strict counter counts each later valid row once against any earlier bin-IoU>.95 box irrespective of category. Neither count certifies repeated physical identity. Human-review discussion and CPU census (catalog:2026-09-17-readout-norm-fresh128) preserve the definitions, examples and raw annotation context.

Keep the frozen training population and the later all-known supplemental population separately versioned. New discoveries do not retroactively become training targets. One predicted row credits at most one atomic owner; group boxes remain separate. Unknown class does not necessarily mean unknown physical identity, and a physical match does not establish category correctness.

Confirmed false objects, repeats, wrong category, wrong extent, malformed/invalid rows and unknown support are distinct debts. Unknown rows stay neither automatic positives nor negatives. They also limit any claim of physical-zero error; silence or a favorable aggregate is not exhaustive verification.

The2026-09-16 user clarification permits dense-scene group annotations when the
unit is explicitly a group. Keep individual owners, groups, body/object parts and
unresolved granularity separate. A valid individual box may contain neighbors;
overlap alone is not cross-owner failure. A group box may earn declared group
coverage, but cannot silently credit an unverified number of atomic owners.
Mixed annotation granularity is a measurement/supervision condition to record,
not a reason to tighten every box until existing valid owners become negatives.
The hand-sized person rejection remains an identity/part-as-whole example.

## Human evidence and review priority

The user's2026-09-17 review finds many real objects among nonduplicate annotation-unmatched predictions. This is useful evidence for protecting credible unlabeled owners, not a measured population rate or blanket label admission. Weakly visible objects, part/group ambiguity and reference-IoU misses remain separate from clear false objects. Judge at the actual unpainted model-input resolution; overlay strokes can obscure tiny evidence. No visual evidence sufficient for a decision means HOLD, not an invented absence or positive.

Original annotations remain intact. Any future visibility/granularity exclusions need a versioned symmetric evaluation, not raw GT deletion or retrospective score improvement. In image542582, a raw traffic-light crowd annotation is absent from the ordinary processed bank; this explains a missing evaluation context, not the truth of every unmatched prediction. A tiny box, a clipped boundary or a dense overlapping group is not automatically an error.

For the proposed recurrence study, review a fixed small set of changed owner clusters and suppression conflicts, including possible losses of credible unlabeled owners. Do not make exhaustive raw-proposal adjudication or teacher completion the gate to another bounded inference question. The previously stopped selective review still supplies no population physical-recall estimate.

## Reopening condition

User ruling2026-09-16: future unmatched diagnosis follows the project-wide
[TIDE-aligned review vocabulary](../../docs/eval/UNMATCHED_REVIEW.md). Co-DETR is
the preferred primary proxy, with no default VLM judge; unresolved instances go
to lead/subagent review. Detector agreement supports nomination, not automatic GT
or training admission. The
detector-only retained-output diagnostic and user adjudication (catalog:2026-09-16-codetr-only-review-proxy)
is closed. Crop/context detector support may assist selective review, but new
calibration remains unproven and is not the next research gate.

When a proposed result changes its matching/review rule, apply it symmetrically to the intended paired raw outputs or keep the result non-comparable. Preserve the previous evaluation and publish a new evaluation identity. A model judge remains a screening instrument until the intended error/admission boundary is independently tested; the automated evaluator pilot (catalog:2026-09-10-autonomous-unmatched-evaluator) did not create an oracle or a hard reward authority.

Do not move confirmation images into training, infer hallucination from FP alone, or remove uncertain owners to make a stage pass. The current user's latest explicit task rule takes precedence over an earlier scientific convention, without silently rewriting the earlier experiment's meaning.

## Current reopening boundary

Reopen a physical label only with identified new visual evidence and a versioned adjudication. Keep common ambiguity shared across compared arms. A finite audit is not a population precision/FN estimate, and candidate-only review cannot reveal objects missed by every generator.

## COCO/LVIS proxy review: missing target is a different population

The 2026-09-27 exploratory review was produced from main by session
`01a0e2e2-ef29-7291-9045-f631930359c1`. This section distills those recorded
findings; it is not a new visual review or independent precision validation.
The retained artifact root is
`/data/CoordExp/.worktrees/research-probes/outputs/research/coco-lvis-proxy-exploration/`.
Its original input/sample hashes remain in the unchanged JSON contracts.

In `audit-20260927-v1`, each relation has 32 fixed source-positive,
COCO-target-unannotated images, stratified with up to eight COCO validation
images. Human presence judgments were made before consulting target status.
`sample.jsonl`, `contract.json`, source-specific `*.labels.jsonl`, and `results.json`
separate image presence from source-box usability; the exact source box is not
assumed to localize another entity. Recorded yes/no/uncertain counts were:
keyboard 30/0/2 (30 usable boxes); tablecloth-to-dining-table 7/22/3 (6 usable);
faucet-to-sink 4/28/0 (0 usable); ski-pole-to-skis 11/16/5 (0 usable);
license-plate-to-car 0/30/2 (0 usable); soap-to-bottle 5/20/7 (4 usable).
These are unweighted finite image-level counts from one review, not population
precision or missing-object estimates. Uncertain/negative or embedded cases
require notes, and nonpositive presence has no applicable target-box judgment.

The strongest counterexample to co-occurrence-based admission is the selection
shift: in `person-and-stable-mappings-20260927`, all 16 sampled dress and all 16
sampled hat source instances in person-unannotated images failed to establish a
real wearer, despite about 94% co-occurrence in the annotated population. Eight
sampled person-proxy cases without COCO person context contained five rejected
part/depiction cases and three unresolved tiny people. The old 1,766 person
proxies were not a verified census of new physical owners. An overlap threshold
alone cannot separate two occluding people from a part of one person.

The keyboard geometry readback in `audit-20260927-v1/keyboard_coco_style.json`
records 453/1,512 embedded keyboards and 1,567/1,795 separate keyboards matched
to COCO keyboard boxes. This is an observed annotation-style difference, not a
universal COCO rule. Tablecloth scene inference (14 high/7 medium/11 low) is a
separate subjective judgment: eight high-inference cases still had no visible
table boundary. Unknown LVIS target status is neither a negative nor a positive;
verified-negative and non-exhaustive category flags remain distinct.

### Export version boundary and reopening

The first two review rounds did not export v2. The later `v2-export-20260927`
records a separate export and its `validation_receipt.json`. Weighted v2 retains
weak person/context candidates; hard v2 excludes them and includes selected
car/truck subclasses at the current trainer's full object weight. The weighted
supervision sidecar is not consumed by that trainer. Armchair was withheld after
counterexamples; soap-to-bottle and embedded-laptop keyboard proxies were removed.
No object-count cap was introduced. Recorded train/val hard-view counts are
884,035/37,430 objects, including 34,088/1,095 proxies; recorded weighted counts
are 938,123/40,500 objects. These are export counts, not calibrated physical truth
or an observed training gain. Dataset payloads and the pre-existing v2 exporter
were not changed by the output migration.

Reopen hard-label admission only with versioned evidence on the target-absent
population, separating independent owner, visible extent, depiction, category
and source-box localization. Keep weak nomination apart from hard supervision.
The original four Markdown source identities and verified migration/recovery
paths are in the root `route-owned-outputs-and-migrate-legacy` change; current
meaning lives here rather than in a parallel report archive.

## Closed fixed-dose row-feedback pilot

The 2026-09-13 pilot's final scientific record is recoverable from
`74665e6c710518cf0298850f6f967795e19da701` at
`research/ideas/qwen3-vl-row-feedback/experiments/2026-09-13-fixed-dose-feedback/{unit,results}.md`.
Its seven commits were integrated as historical parents without restoring the
obsolete one-off probe framework; no current caller uses that implementation.
This synthesis is historical evidence, not a fresh rerun or promotion.

S inserted the ordinary box-end contextual slot; F added the just-completed
row final hidden-state vector, RMS-matched at fixed scale 1. Both started from
N16, used 64 updates, the same 16 successor packages across 11 images and
54-record teacher protection, with a 3,084-visible-token limit. On the frozen
32-image / 280-annotation panel, IoU .50 owner matches were S193/F197:
+4/280 (+1.43 percentage points), with 9 gains, 5 losses and 188 retained.
At IoU .60 the counts were 177/180; at .80 they were 129/125. This single-seed,
fixed-dose result is sensitive to localization precision.

The preselected dense8 proposal review covered 306 proposals in 230 display
groups over eight images. Clearly attributable owners were S110/F113; the
admitted sets retained 107, gained 6 and lost 3. Extra proposals for the same
owner were 3/3; uncertain proposals were 36/32 and remain neutral. This is
candidate-owner review, not an exhaustive scene census, physical recall or
population precision measurement.

The cup/spoon/donut content diagnostic transplanted a future-completion vector
captured after h+c+w to an earlier C boundary. W remained the first emitted
row in all three correct/wrong-source pairs; spoon was unchanged, both cup
runs capped, and changed donut regions included donut-hole groups. Sensitivity
in two cases does not establish selective visited-owner suppression, a memory
ledger, native-memory incapacity, storage/readout isolation or sample efficiency.
The bounded round is closed with no checkpoint promotion or authorized next
dose. Known GPU allocation was at least 12.858515 hours, with six early timing
intervals missing; this is not an exact cost. The old artifact root is absent;
Git recovery of the record does not establish payload availability or qualify
continuation.

## Provenance

Catalog IDs resolve through [the existing catalog](../experiments/catalog.jsonl), which retains original evidence labels, artifact locators and exact Git recovery paths. Detailed source records are recoverable at `108dede0154abfd90a54d18234d9e0bac780a3ba`. Historical entries are unsupported for continuation; recovery is not execution qualification.
