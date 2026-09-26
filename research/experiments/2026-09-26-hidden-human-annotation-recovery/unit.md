# Hidden human-annotation recovery

Status: corrected CPU preparation lead-accepted; two-image GPU smoke released below. No training released.

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
**untied+axis001 step2444** for acquisition and the later learning starting point:
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

For a later separately frozen learning contrast, the lead owns learning rates,
weights, optimizer groups, scheduler, effective batch and exposure. Retain existing
`base_ce`, `token_type_gate`, and `raw_axis_validity_hinge`. The geometry term
constrains expected x2-x1 and y2-y1 by a margin; it is not GT regression or a
promise of greedy-valid boxes. Do not invent GIoU, Gaussian/RPS or another loss
under that name. Unknown-target and untrusted-terminal-EOS masking must be
consistent across CE, type gate and eligible geometry; acquisition cannot consult
hidden GT. Warm-starting model weights is distinct from resuming optimizer or
scheduler state. This requirement is recorded only: acquisition preparation
contains no training implementation or launch.

## Lead acceptance and current package: two-image GPU smoke — 2026-09-26

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
