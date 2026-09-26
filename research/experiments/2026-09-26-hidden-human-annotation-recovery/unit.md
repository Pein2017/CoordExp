# Hidden human-annotation recovery

Status: autonomous stage completed and lead-accepted. The full frozen189-candidate experiment adds bounded annotation support and favors enlarged-view localization/repair over mandatory joint-view verification. No automatic positive admission, further query package or production training is released.

## Authority and ownership

The user endorsed hiding human-added annotations and testing recovery, asked to
include the additional five refined images, and explicitly requested persistent
`926-worker` with Astra / low on 2026-09-26. This is a new bounded acquisition
experiment, not continuation of a historical training or mechanism run.

- Lead: `01a0dd7c-0899-7b81-90a2-2f50da3476d1`; owns interpretation, scientific
  choices, launch release and acceptance.
- Worker: requested sidebar task `926-worker`, model `gpt-6-astra`, effort `low`.
  Actual task UUID: `01a0de41-cc56-7a62-8c56-c2d9850b95b5`; live transport inspection verified Astra / low.
- Cwd: `/data/CoordExp/.worktrees/research-probes`.
- Return directly to the lead through the installed lead-worker transport.
  Read `/data/CoordExp/.codex/skills/lead-worker/SKILL.md`; the dispatch appends
  the exact pair and direct-return command. Do not create another main worker.

## Question and outcome

At one frozen Qwen3-VL checkpoint, can spatially targeted queries plus bounded
verification produce additional trustworthy human-added instance reports beyond
ordinary full-image sampling, at a measured acquisition and verification cost?

The final destination remains better original-image, empty-history natural
greedy coverage. This package tests supervision acquisition only. Hidden-label
recovery, candidate admission, greedy behavior and learned transfer are distinct.
Hiding labels is an information boundary for acquisition/admission; it does not
erase model exposure or modify image pixels.

## Decisive inputs

Read `research/index.md`, `research/assets.md`, `research/CONVENTIONS.md`,
`docs/OUTPUT_STORAGE_POLICY.md`, and the discovery/physical-evaluation question
pages. Use research-flow and qwen3-vl-execution as needed. Reuse current maintained
source; historical outputs are data/evidence, never executable imports.

Human13 frozen input:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl`.
Its adjacent `human-refined-13.geo_sorted_xy.coord.receipt.json` binds SHA256
`5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`.
Lead checked 13 images / 392 reports: 197 negative human-added annotation IDs,
195 existing nonnegative IDs. Retained boxes can themselves be human-refined;
filtering IDs does not recreate the original COCO annotation version.

Refined5 frozen reference:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl`.
Lead checked image/count pairs `7116:5,309264:14,351017:49,417044:63,477415:47`.
Its pre-edit source has 83 reports versus 178 now. Do not infer 95 newly added
owners from that subtraction: resolve additions, retained edits, deletions and
unresolved identity from annotation provenance. Resolve snapshot image references
through the actual source workspace; these relative paths are not automatically
relative to the copied snapshot directory.

The union is 18 distinct images / 570 annotated reports. Keep Human13 and refined5
separate in reporting. Human13 is development data, with historical panel-specific
fitting. Refined5 is a proposed frozen confirmation cohort for this protocol, not
an established globally unseen test set. The user permits treating the high-quality
annotations as approximately complete within their declared category/instance
scope; group/part ambiguity and out-of-scope outputs still need explicit handling.

## Completed worker package: CPU preparation

1. Verify frozen file identities, images, object schema and the refined5 lineage.
   Preserve all original data. Produce separate visible-label and evaluator-only
   hidden-label views with per-object provenance. If refined5 additions cannot be
   resolved, return that exact gap; do not invent IDs or a random replacement mask.
2. Keep hidden coordinates, categories, descriptions, IDs, counts and GT-derived
   crop choices out of discovery and automatic admission. Existing visible labels
   may identify already-known instances. Acquisition must run without access to
   the hidden bank; the offline evaluator alone reads full truth. Freeze outputs
   before scoring. Add a meaningful CPU leakage/sensitivity check: changing only
   hidden truth must not change acquisition requests or admission inputs.
3. Locate the current native inference/evaluation entrypoints and the mature untied+axis001
   step2444 composition in `research/assets.md`. Verify known
   human-refinement exposure and payload/config identity. Use the user-selected untied 2B anchor; do not use a Human13-specialized fitted checkpoint. Report gaps instead
   of silently switching family, checkpoint, adapter, vocabulary or decode policy.
4. Prepare the smallest caller using existing maintained primitives. Proposed
   acquisition contrast: one shared natural-greedy baseline; four ordinary
   full-image samples versus four fixed overlapping region views, with the same
   frozen model and bounded verification policy. Regions must not depend on hidden
   truth. Report actual pixels/visual tokens, generated tokens, model calls and
   time separately; equal call counts alone are not equal compute. Freeze exact
   prompts, sampling/stop policies, crop mapping and verification selection before
   the confirmation cohort. Do not build a generic pipeline or a new verifier
   model. If these choices require a consequential scientific ruling, return a
   compact proposal and strongest alternative to the lead.
5. Prepare readback evaluation of hidden-owner recovery, newly recovered owners
   absent from baseline greedy, retained known owners, duplicate/invalid/uncertain
   candidates, geometry quality and verification cost. One-to-one annotation
   matching is a stated proxy, not automatic physical truth; unmatched and
   same-owner extent ambiguity need separate accounting. Do not make arbitrary
   IoU clusters ground-truth identity or equate detector absence with a negative.

Owned writes: this unit directory; one minimal task-specific probe under `probes/`
and its necessary tests if existing entries cannot directly serve the task;
task outputs under
`/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/`.
No unrelated edits, dataset modifications, global configuration, training,
publication, broad refactors, architecture additions or runtime cleanup. No Git
commit yet. If routine research catalog/index registration is necessary, make one
minimal current-unit entry and frontier pointer; do not rewrite prior conclusions.

## Checkpoint, acceptance and stop

Return a stable `candidate` to the lead containing the data manifest and hashes,
refined5 mapping and exposure caveats, exact proposed command/model/policies,
owned diff, fresh CPU checks and a bounded two-image smoke plan. Show that the
leakage check fails under a deliberate wrong dependency. Include job state and
unresolved decisions. No GPU job is authorized in this preparation package.

The next intended release is a two-image real smoke, followed by the 18-image
single-round pilot if technically qualified. No runtime or GPU-hour cap is imposed;
measure actual cost and keep the finite first-round schedule. Lead will issue that release after inspecting this checkpoint, without
reopening settled user choices. No automatic extra sampling, teacher-size growth,
training or second iteration. Stop after the preparation report; only the lead
marks acceptance and schedules the next package.

## CPU preparation checkpoint

See [preparation report](preparation.md) and [state](state.json). The following user rulings settle the grill-me frontier; only technical acceptance
and the lead's explicit GPU release remain pending.

## User rulings — 2026-09-26

- Scope is COCO 80 classes, retaining existing instance/group semantics and the
  user's approximately complete annotation assumption within that scope.
- Use the same mature 2B teacher. Enlarged crops and repeated queries are allowed;
  no external detector or larger teacher arm is added.
- No preset precision/ratio target (including 95%) governs admission. Retain every
  candidate, screening/verification score and reason; report attainable recovery,
  quality/error types and recall/error tradeoffs. Candidates are not automatically
  trusted positives. Training remains unauthorized.
- No runtime or GPU-hour cap. Prefer eight-GPU parallel throughput using supported
  sharding, without waiting for random stress occupancy. Actual pixels, visual
  tokens, generated tokens, calls and generation/verification time remain separate.
- The 18-image cohort and one initial acquisition round stay fixed. No endless
  sweep; finite sampled support is not a perception upper bound. Reuse saved K1–K4
  prefixes for curves. Any additional levels need residual evidence and a new lead
  schedule, never hidden-truth-selected queries.
- CPU preparation remains authorized. Lead inspects the stable candidate before
  releasing the real two-image vertical slice, then the technically qualified
  18-image pilot. No GPU job has been released to this worker.

## Initial finite schedule

Freeze before model outputs or hidden-truth scoring: one shared original-image,
empty-history greedy call; four full-image samples; four fixed overlapping
5/8-region samples in top-left, top-right, bottom-left, bottom-right order.
Snap crop boundaries outward to 32px; preserve native scale in this first contrast.
All arms use the same mature 2B composition and exact COCO-80 prompt; sampling
T=.7/top-p=1/top-k=0, repetition penalty 1, seeds 92601..92604, <=3084 tokens per
call, EOS stop, no inherited generation defaults. K1–K4 uses cumulative saved
queries, not further generation. Every candidate survives in the raw bank.

Initial bounded verification is saved-query self-consistency: for each candidate,
record maximum same-category IoU in each other query of its own arm and the count
at IoU>=.5. These are correlated teacher-screening scores, not owner identities or
a trusted-label criterion. Show the few discrete support levels available in each
K prefix against offline recovery and errors; never choose a winning threshold
from confirmation truth or automatically admit it. Preserve invalid rows, literal
repeats, unmatched candidates and all reasons. No additional verifier calls are
in this first schedule. Enlarged candidate crops or repeated targeted verification
remain allowed follow-up proposals if residual evidence warrants them; they are
not silently added to this round.

The source caller supports existing eight-rank request sharding through torchrun;
the two-image smoke uses Human13 IDs 1584 and 2299 (18 requests) to protect refined5
confirmation outcomes. Policy is frozen for preparation but marked
`schedule_frozen_awaiting_lead_release`, so acquisition fails closed until released.

## Final anchor and future learning boundary — 2026-09-26

The user corrected the intervening tied-Source choice and froze the original
**untied+axis001 step2444** for acquisition and, originally, the later learning starting point:
`/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444`.
Base:
`/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
Load its DoRA adapter plus independent input/output special-token deltas. Preserve
untied semantics through loading and later warm start. All acquisition routes
share this exact composition. The user's performance impression is a starting
preference, not a causal finding about untying; its objective/history differ from
tied Source. The tied instruction and `preparation-v4-tied` evidence are
**superseded**, not another arm. Reuse `preparation-v3` and already-valid untied
CPU processor evidence where identity matches; do not rerun preparation merely
because the tied proposal intervened.

The user's subsequent 2026-09-26 correction supersedes the future-loss requirement
to retain `raw_axis_validity_hinge`. Keep the three roles: `base_ce`,
`token_type_gate`, and prefix-conditioned `conditional_order_gate` geometry.
The old expectation hinge remains part of the frozen teacher's training history,
not a correct implementation of strict emitted-coordinate validity. Do not rewrite
historical scores or switch checkpoint mid-probe. A separately prepared fresh
untied illegal-mass run belongs to task `01a0b28f-ddec-74f0-89bd-7d3f094059bd`;
it is not launched by this unit. Compare a qualified new checkpoint on the same
fixed probe before choosing the later learning anchor. The lead owns learning
rates, weights, optimizer groups, scheduler, effective batch and exposure; equal
numerical weights do not mean equal scales across the two geometry objectives.
Unknown-target and untrusted-terminal-EOS masking must be
consistent across CE, type gate and eligible geometry; acquisition cannot consult
hidden GT. Warm-starting model weights is distinct from resuming optimizer or
scheduler state. This requirement is recorded only: acquisition preparation
contains no training implementation or launch.

## Lead acceptance and completed package: two-image GPU smoke — 2026-09-26

This section supersedes the preparation package's no-commit/no-GPU restriction
only for the following release. The lead accepts corrected CPU preparation after
independent hash verification, source review, hidden-truth mutation replay, eight
passing tests, and direct replay of the norm1000 crop-mapping counterexample.
The original correction RED evidence failed both mapping assertions; corrected
`evaluate()` recovers the one hidden annotation at IoU .5. This is technical CPU
acceptance, with scientific outcomes and GPU qualification still pending.

The lead may make one local commit of the seven task-owned source/test/record
files to satisfy the existing clean-source runtime contract. No publication.
The launch receipt under the task output root, `lead-release-01.json`, binds that
commit, the corrected preparation report/receipt, and `smoke-policy-01.json`.
Only the policy release status changes; model, prompts, sampling, crops and
budgets remain frozen. The worker uses the committed checkout and must preserve
its clean state throughout acquisition and finalization; runtime records go to
outputs until the lead accepts or rejects this package.

Worker `926-worker` (same UUID, Astra / low) is released to execute:

```bash
torchrun --standalone --nproc_per_node=8 -m probes.hidden_human_recovery acquire \
  --visible /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/preparation-v3/acquisition/visible.json \
  --policy /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/smoke-policy-01.json \
  --image-ids 1584 2299 \
  --output /data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/smoke-01
```

Reconcile matching live work before starting exactly one invocation. Do not wait
for random stress occupancy; react to concrete launch failure. Redirect the full
log and retain exit status, rank/PID ownership, source identity and actual wall
time. The finite schedule has 18 calls and at most 55,512 new tokens; there is no
wall-time/GPU-hour veto. Do not change numerical policy to work around failure.

After success, use fresh-process frozen-shard readback, the existing `review`
entry for saved-query screening, and `evaluate` with an explicit empty review
list. No human/automatic admission is claimed. Check 18 unique requests across
eight ranks, exact prepared prompt/media/grid identities, empty extensions,
DoRA plus untied input/output delta composition receipts, token bounds and stop
reasons, norm1000 crop mapping, and durable outputs readable by the evaluator.
Report per-route calls/pixels/visual and generated tokens/time, load/wall cost,
raw hidden-recovery proxy and category disagreement separately. Neither correlated
agreement nor this two-image slice qualifies trusted positives or a perception
upper bound. Preserve refined5 confirmation outputs unopened and ungenerated.

Return a stable candidate report and hashed receipt directly to the lead; stop
after the smoke or a concrete failure. Do not alter source, restart a failed
run, launch the 18-image pilot, add queries, train, or publish without a new lead
assignment. The next decision is technical smoke acceptance, then release of the
already intended fixed pilot if qualified. Existing user choices stay settled.

## Completed package: fixed 18-image pilot — 2026-09-26

Smoke acceptance evidence: `smoke-01.report.md`, `smoke-01.receipt.json` and
`smoke-01.lead-replay.json` in the task output root. The lead revalidated all
52 receipt artifacts, 18 unique requests, eight source receipts, exact CPU-plan
prompt/media/grid/crop/seed parity, and identical fresh screening/evaluation.
The run exited 0 in 78.512 seconds with all 18 EOS stops and 2,629 new tokens.
Successful committed loader receipts plus bound payload keys establish the
DoRA and independent untied delta loading path; they are not live tensor dumps.
Legacy singular metadata `tensor_key` is not evidence of tied execution.

Raw two-image hidden coverage is greedy 16/36, full K4 19/36 (five new beyond
greedy), region K4 24/36 (ten new). These are development-image annotation proxies,
not admitted supervision or a population result. No threshold was selected.
The model, data views, query policy, matching and screening stay unchanged.

Release exactly one eight-rank invocation of the same `acquire` command above,
omit `--image-ids`, and set output to
`/data/CoordExp/outputs/research/hidden-human-annotation-recovery/2026-09-26/pilot-01`.
Use the identical `smoke-policy-01.json`; `lead-release-02.json` separately grants
this cohort expansion and binds the clean local documentation commit. All 136
runtime source files must equal the accepted smoke bytes. The pilot contains
162 calls, at most 499,608 new tokens; expected input visual tokens are 17,660
greedy, 70,640 full and 29,028 region. Record actual cost with no elapsed veto.
Rerun both smoke images as part of this single complete invocation, keeping their
smoke artifacts separate and checking repeat token parity; do not count them twice.
This uses the existing complete-shard reader without adding a merge mechanism.

Worker ownership remains outputs and ignored transport files only. Keep source
clean, reconcile prior processes, start one invocation, preserve full logs and exit
status, and return promptly on concrete failure without changing settings or
restarting. Apply the smoke's composition, input parity, token/stop and durable
readback checks to all 162 requests. Freeze raw outputs before offline scoring.
Run existing screening/evaluation with explicit empty reviews; retain every raw
candidate, invalid report and saved prefix/support curve, with no auto-admission.

Report Human13 and refined5 separately, then the combined annotation proxy:
hidden and known coverage, new IDs beyond greedy, greedy-covered IDs absent from
each arm, category disagreements, unmatched/invalid/boundary cases, K1-K4/support
curves and actual costs. Refined5 is confirmation for this frozen protocol, not
globally unseen. Its 105 added region IDs are not proven distinct new physical
owners. Image 7116 has no hidden denominator and remains a preservation control.
Do not tune policy or select a threshold from confirmation truth. The lead may
commit only these status/protocol updates locally; no publication is released.

Return a stable candidate report and hashed receipt directly to the lead after
the finite round; stop. No additional model queries, admission, training, source
edits, restart or next round is authorized by this package.

## Completed package: CPU complementarity audit preparation — 2026-09-26

Accepted interpretation and evidence are in [pilot-results.md](pilot-results.md)
and output `lead-acceptance-03.json`. The next question is whether the 32 regional
witnesses absent from full K4 describe additional physical units or alternative
extents/assignments of units already represented by full-query candidates.
This audit uses evaluation-selected IDs; it is not a deployable admission rule.

The same Astra/low worker prepares exactly 32 blinded cases from saved outputs.
Each presents the target candidate, original-image context, readable local view,
and nearby same-category full-query candidates. Show anonymous case/box labels
and candidate descriptions; hide arm names, scores, annotation IDs, GT boxes and
recovery status. Freeze the display-neighbor rule using prediction geometry only
before rendering; neighbors are context, not inferred owner equivalences.
Keep the hidden-ID/arm crosswalk and exact prediction provenance in a separate
evaluator-only key. All 32 targets must survive without selecting nicer examples.

Use existing maintained visualization operations, or one-off inline CPU drawing
where appropriate. Owned writes are task outputs under `complementarity-audit-01`
and ignored transport files. No maintained source change, model call, external
model, training, candidate admission, threshold selection or dataset edit.
Verify image/box mapping, input hashes, 32-case coverage and blinded fields.
Return the stable packet, hashes and exact limits directly to the lead, then stop.
The lead reviews entity/category, extent, instance/group/part, cross-candidate
identity and uncertainty before deciding a candidate-local verification test.

## Completed package: candidate-local re-detection implementation and smoke

From the same frozen untied teacher and pilot candidate bank, does class-blind
local re-detection at native versus enlarged visual scale supply localization
and ambiguity evidence beyond saved-query agreement? The immediate package
qualifies mechanics, not a precision operating point. User ruling: beverage cans
belong to `bottle` in this annotation scope; preserve the original references.

Select candidates from the saved 2,111 literal-unique full/region records using
predictions and visible labels only. Within each image, stratify by route,
visible-known IoU>=.5 flag, zero versus nonzero saved-query support, and crop-boundary
flag (full route always false). Choose one candidate per nonempty cell by minimum
SHA256 of `926-local-v1:` plus prediction ID. This gives at most 12 cells/image,
216 candidates and 432 future queries. Keep every original candidate in the bank;
these strata are a diagnostic sample, not a population precision sample.

For each selected candidate, make a centered original-pixel ROI with each side
`max(192, 2 * predicted_side_length)`, snap boundaries outward to multiples of 32
and clip to the image. Run the same ROI at native scale and exactly 2x bicubic
resize, with identical trained COCO-80 prompt, no proposed category, candidate box
overlay or hidden labels in the prompt. Both are empty-history greedy queries
at the original frozen model/FP32/SDPA, EOS, repetition penalty 1 and max 3084
new tokens. Cropping is proposal-conditioned; resizing supplies no new pixel
information. Preserve actual model-input pixels/tokens and original crop geometry
separately; both outputs map to original norm1000 coordinates without rounding.

Implement the smallest change to the existing task probe/tests using maintained
native generation and sharding. Preserve the original acquisition/readback behavior.
Prepare CPU request identities and counts for the whole selected bank, including
proof that hidden truth mutations cannot alter selection or queries. Test real
caller geometry for offset, edge-clipped ROI and 2x view, record cost bounds, and
replay prior probe tests. No new generic runner or external verifier.

In the same CPU preparation, retain a separate certain-invalid evidence bank
from all frozen raw requests. Reuse parser drops and exact saved token IDs: bind
each complete geometry-invalid row to its request/crop, generated-order row,
coordinate slots, own x1/y1 prefix and first illegal x2/y2 token. Preserve all
342 occurrences and literal-repeat multiplicity; do not treat the 299 greedy geometry
failures on image351017 as independent specimens. Keep the two capped incomplete
rows separate. Reject wrong row/token alignment, and distinguish valid 0/999
boundary contact from zero area. Hidden truth cannot affect this bank either.
This is extraction/validation only, with no backward pass or optimizer update.

The future negative-supervision rule is local to the actual supplied prefix:
penalize x2<=x1 or y2<=y1 probability mass, without treating an entire invalid row,
unmatched valid box, group, duplicate, or budget-truncated suffix as a negative
target. A prior corner at999 creates an empty legal completion set; flag that
earlier dead-end decision rather than taking log of an empty set. Generated
invalid rows cannot be passed unchanged through a valid-GT target constructor.
Own-prefix replay and its reduction/gradient qualification belong to a later
learning package. No inference mask is added.

The lead authorizes the worker to make one scoped local commit of probe/tests only
after checks pass and the source is otherwise clean. Then run one torchrun8 smoke:
one selected candidate in each of Human13 images1584,2299,2685,4134, chosen by the
same stable hash, both views per candidate. Eight queries, <=24,672 new tokens;
no time/GPU-hour veto. Reconcile live jobs first; retain existing runtime/identity
and fresh-process frozen-readback requirements. If a check or actual run fails,
return evidence without restart or numerical-policy changes. No unrelated edits.

Save all re-detections, same-category target IoUs, competing matches, and cross-view
geometry; retain ambiguity rather than promoting a positive. Ground truth enters
offline diagnostics only. Do not fit a threshold or overwrite pilot metrics.
Return a stable CPU plan, exact source diff/commit, eight-query smoke receipt and
readback candidate directly to the lead. Stop. The full selected-bank run, more
sampling, automatic admission and training need a subsequent lead release.

## Completed package: full frozen candidate-local verification experiment

The user authorized autonomous mainline research on 2026-09-26 until a useful
stage result, convergence, or evidence that further attempts are not worthwhile.
The active goal is scalable automatic supervision, without per-item visual review.
Production training remains held for separate explicit user authorization.

Question: does target-local evidence separate annotation-supported proposals
from neighbor agreement and localization ambiguity better than saved-query
agreement, while retaining hidden-reference coverage? The strongest alternative
is that local queries consistently find a nearby object rather than validate the
proposal. Smoke1584 already realizes this alternative: same-category cross-view
IoU0.527338, but both target IoUs0. Repetition is also present with valid geometry.

Freeze the existing selection: 189 candidates from2,111,18 images,378 paired
native/2x requests in `candidate-local-01/selection.json` and `cpu-plan/`.
No image/candidate omission, score-based reselection, prompt/category hint,
numerical policy, model, crop or seed change. Reuse the existing CPU identities.
Exactly one torchrun8 invocation, without an image filter, is released after the
CPU evaluator checks and scoped source commit below. Max1,165,752 new tokens;
120,145 planned visual tokens. No wall/GPU-hour veto. The eight smoke requests
recur as identity controls and are never counted twice in scientific totals.
No restart after an actual GPU failure; retain capped and invalid outcomes.

Before generation, implement prediction-only compact scalar readback in the
existing probe/test files. For target box/category t,c and literal-unique
same-category detections D1,D2 from its two views, retain:

- B: existing within-route other-query IoU>=.5 support count.
- L1,L2: max IoU(t,d) separately for the two views, empty set=>0.
- U: max IoU(a,b) across D1,D2, empty pair set=>0; diagnostic only.
- A: max over pairs of min(IoU(t,a),IoU(t,b),IoU(a,b)), empty pair set=>0.
- The A-maximizing witness pair, ties by prediction IDs, and each witness's
  target IoU minus maximum IoU to any other same-category saved-bank proposal.
  Preserve strongest competitor IDs; competitors can duplicate the same owner.

Do not fit score weights or choose an operating threshold. Literal repeats never
add support votes; retain their counts and every raw output. Save compact per-target
scores/witnesses and raw references instead of serializing every Cartesian edge.
Process one ROI pair at a time; qualify CPU time/RSS/artifact size on both the
saved smoke and a full-cap repeated/unique-row counterexample. No new framework.
Missing or corrupt requests are technical failures, not zero scores. Valid empty
outputs have zero support; cap/EOS and invalid burdens remain separate fields.

Offline only, define G(box) as original same-category annotation IDs with IoU>=.5.
Retain G(t),G(a),G(b), their hidden/visible/cohort identities and empty/multiple
flags. Report same-singleton target agreement, same-singleton neighbor agreement,
witness disagreement and ambiguous/unsupported outcomes. Distinguish an unsupported
target with supported local outputs as a possible repair/addition, not validation
of the original proposal. These are annotation-localization proxies, never human
adjudication or physical precision. Keep refined5 redraw uncertainty explicit.

For every attainable threshold of B,L1,L2,A, retain complete tied-score groups.
Report selected/retained counts, the above proxy outcomes, and hidden/visible
coverage of retained ORIGINAL candidates, using the existing class-agnostic
one-to-one matcher atIoU.5 plus its post-assignment category-agreement counts.
Do not silently replace that matcher with category-constrained rematching.
Keep coverage of newly generated local outputs separate (native,2x,union).
Preserve full hidden302/visible268 denominators, image/cohort/selection-stratum
breakdowns, and all18 images including the zero-hidden control. Comparisons are
within this fixed diagnostic sample; no population precision or transfer claim.

CPU acceptance must falsify neighbor-only agreement (1584), duplicate-vote
inflation, wrong target/view pairing and truth-dependent scores. Show expected
target agreement versus neighbor/ambiguous reference outcomes on small fixtures.
Replay the existing11 tests when source changes. Preserve acquisition behavior;
verify all378 prompt/media/crop/grid/seed identities against the existing plan.
Only probe/tests may be edited and committed by the worker; records belong to
the lead. Routine bounded CPU repairs may iterate; semantic/numerical conflicts
return to the lead. One clean scoped commit precedes the one GPU invocation.

Return the full run, compact readback/curves, costs, limits and bound receipts.
Stop with zero admitted positives and no training. The decision is whether A or
per-view localization adds annotation-specific separation beyond B, or local
outputs are more useful as repaired proposals. If gains only reflect U, repeats
or neighbors, automatic verification remains unsupported. Mixed results remain
inconclusive; further queries require a new finite lead assignment, not a sweep.

## Stage decision and closure — 2026-09-26

The autonomous goal reached a reproducible stage result; no active package remains.
See the final sections of [pilot-results.md](pilot-results.md) and output
`lead-acceptance-06.json`. Technical execution and the frozen annotation-proxy
results are accepted; physical precision, trusted positive admission and learning
benefit remain unestablished. No score/threshold was selected for production.

Enlarged-view localization provides useful ranking evidence in the fixed sample.
Requiring both views through A creates a weak-view bottleneck and loses hidden
coverage at the saved-support count controls. Local outputs supply31 additional
hidden IDs beyond the complete historical pilot's independent matched-ID union
(30 by the category-agreeing counterpart), largely from2x. Most local support is
already available; repeating this schedule without a new discriminator is not
worthwhile. Preserve the current teacher as historical development evidence.

Next research, if reopened, should target enlarged-view proposal repair and a
matched qualification on the separately retrained teacher, or a small own-prefix
negative-supervision gradient/learning test with the corrected geometry term.
These are reopening conditions, not scheduled jobs. Do not require per-item
assistant visual approval, silently treat unknown candidates as negatives, or
resume production training without the user's separate explicit authorization.
