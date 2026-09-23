# Ruling 01: epoch16 endpoint and matched CE-only comparison

2026-09-22. Lead dose ruling following the user's question whether8/16 epochs
suffice instead of32. The user has authorized retraining and comparison; the
lead selects16 as this round's definitive endpoint, with8 for a trend check.
The worker confirmed zero model-entry processes, zero scientific updates,
zero allocated GPU-seconds and no model clock at the time of this ruling.
No executed evidence is being relabeled or discarded.

This amendment supersedes the exposure, checkpoint, evaluation-denominator,
queue and GPU-scheduling clauses of [unit.md](unit.md). The original unit stays
immutable. All other recipe, guardrail, source, ownership, qualification,
Luna-learning, budget and reporting requirements remain in force. The temporary
new-model-admission hold is lifted: mechanical checks passing authorize
immediate execution without another lead ACK.

## Training and qualification

Train fresh from the identical mature live untied+axis001 source, seed1729,
same1024 training identities and492-pack cache, effective batch8packs,
four ranks x accumulation2. Use exactly the first984 updates of the predecessor
sample/pack schedule:16epochs =7872global packs =16384image presentations
=984optimizer calls. Epoch8 is step492,3936packs,8192presentations.
Retain checkpoint saves62/123/246 and add492, with984 final. No step1968 or
automatic continuation to32. Step62 remains1033presentations, not an exact epoch.

Keep the existing `constant_with_warmup` scheduler and10 warmup steps, all LRs,
optimizer and parameter groups. This scheduler does not decay as a function of
the total horizon; verify actual per-step LRs and pack IDs match the preceding
trajectory's prefix. Bind the amended effective config and schedule before
execution. Do not change a32-epoch cosine schedule into16 or assume arbitrary
schedulers have horizon-independent equivalence.

The only objective difference remains CE1/typegate0.2(all four groups)/
raw_axis_hinge0.01 margin1/999/gaussian0, segment_balanced. Qualification and
real-caller falsifications from unit.md remain mandatory. The shorter dose
does not relax them or establish that fitting should already be complete.

## Evaluation and comparison

Epoch8: evaluate only the original96-image training sentinel. It is descriptive
and cannot choose the final checkpoint. No epoch4 native evaluation this round.
Epoch16: evaluate the complete fixed1024/256 panels, including that sentinel.
Keep natural empty-prefix generation and all prior parsing/scoring semantics.

Use CE-only epoch16, NOT CE-only epoch32, for the primary equal-dose comparison.
The exact historical checkpoint is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-22-coordinate-codebook-scale/scale1024-seed1729-32epoch-repair-v2/checkpoints/step-984`

The accepted predecessor has96 epoch16 sentinel cells bound to that checkpoint.
Reuse those96 after identity checks. Evaluate the remaining928training plus
256validation images (1184new cells) from that immutable checkpoint using its
original payload/loader bindings. Write supplementary comparison outputs only
under this new package; do not modify or extend the closed predecessor's queue,
results or acceptance. No CE-only retraining is needed. Fresh-process payload
readback must prove the supplementary cells use the bound historical tensors.
Existing ordinary full-vocabulary teacher CE remains comparable; the new
composite training objective is reported separately.

Denominators are explicitly:

| Evidence | New native cells | Reused native cells |
|---|---:|---:|
| Mature source, full1024/256 |0|1280|
| CE-only epoch16, full1024/256 |1184|96|
| Three-loss epoch8, sentinel96 |96|0|
| Three-loss epoch16, full1024/256 |1280|0|
| Total distinct cells |2560|1376|

Thus3936 distinct cells support this package. The three-loss source/trained
trajectory alone has2656 cells (1280source+96epoch8+1280epoch16); matched final
source/CE-only/three-loss triplets cover1280images. Do not inflate denominators
by counting pairwise comparisons, shared source references, old epoch32 cells
or repeated appearances of sentinel images as new independent observations.
Old epoch32 remains historical context, not the matched primary control.

Apply the original source-relative new-bad/cap/owner-recurrence/severe limits
to the NEW final epoch16:51/10/51/10 train and12/2/12/2 validation. Keep paired
severity and CE-only-epoch16 repaired/persistent/new failures separate. Training
natural fit remains primary; validation coverage/CE alone cannot veto/select.
All incomplete cells remain HOLD in these frozen denominators. No loss-specific
causality, transfer or universal-capacity claim follows from this single round.

## Execution and resource accounting

Reuse all source cells and keep the original retained32 membership independent
of reuse provenance. Use separate fresh atomic queue directories for
`ce_only_epoch16`, `three_loss_epoch8` and `three_loss_epoch16`; do not share a
parent queue-state.json across conditions. Frozen order follows existing panel
and sentinel order, excluding only already-reused cells from new-work queues.
Bind exact IDs, conditions, checkpoint paths and original hashes before launch.

Training uses GPUs0..3; GPUs4..7 can immediately evaluate the fixed historical
CE-only checkpoint and then eligible new checkpoints. After training, use all8
for remaining fixed work. Prioritize complete final comparison pairs over the
epoch8 diagnostic at the budget tail; no output-dependent case selection.

The same NEW8wall-hour/64allocatedGPU-hour envelope remains, including new
historical-checkpoint evaluations, qualifications, loading and failed attempts.
First model-entry process starts the clock even if it is a comparator evaluator.
Source-reuse and old training costs are labeled historical, not newly incurred.
No clock reset when training starts. Enforce the existing in-flight reservations
and7.75hour admission cutoff; more decode work means halving training exposure
does not imply halving end-to-end cost.

Continue the assigned Luna prompting/acceptance learning in the single
delegation-notes.md, with actual model/effort and attributable changes. Return
the first production-shaped evidence and final candidate directly to lead.
Stop at this fixed16-epoch package; extending to32 needs a new lead decision.
