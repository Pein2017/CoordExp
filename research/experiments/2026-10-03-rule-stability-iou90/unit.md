# Learned geometry and strong-overlap suppression on the full-label18 cohort

## Question, authority and predecessor boundary

From the mature untied-axis001 step2444 anchor, does adding sampled-trajectory
strong-overlap credit to full-label positives and fresh greedy geometry training
reduce ordinary free-generation errors on the same18 images over16 updates?

The user selected model learning, with rule guards only diagnostic; all570 labels
may enter positive training. The user fixed a class-agnostic, strict IoU>0.9
duplicate proxy, chose18-image training-internal learnability, and explicitly
rejected a fixed coverage-loss threshold. Coverage/gained/lost/retained remain
observations: no quality early stop, perfect-F1 gate, or post-hoc clean checkpoint
selection. All8 GPUs are available with no user compute cap; efficiency matters.
The user requested a persistent GPT-6.1/xhigh implementation worker on2026-10-03.

Lead owns this protocol, state, source/input bindings, research interpretation
and final consumer acceptance. Worker owns implementation, package checks and
ordinary repair inside this contract. Initial dispatch authorizes CPU preparation
only. Lead may release a source-bound native qualification and primary16 package
after preparation; no further user approval is needed for the already agreed
finite study. Worker cannot self-release a missing native packet, add arms, alter
meaning/thresholds or schedule another unit.

Predecessors establish that errors can be learned away, but selected-prefix
repair can coexist with new natural errors, owner exchange and early stopping:
[online credit](../2026-09-27-online-row-credit/results.md),
[short-dose ranking](../2026-10-03-short-dose-ranking/results.md), and
[full-label fitting](../2026-10-02-full-label-self-rollout-fit/results.md).
The small whole-row-probability duplicate term in the older
[row-credit contrast](../2026-09-27-rollout-row-credit/results.md) did not establish
that effective duplicate supervision fails. This new unit tests additional
sampled rule credit from the stronger val200 anchor; it is not a continuation of
the recently closed auxiliary-head run.

## Inputs and policy

- Checkpoint: `/data/CoordExp/outputs/shared/checkpoints/untied-axis001-step2444/payload`.
  Base+adapter+independent input/output deltas are all required, as in
  [assets](../../assets.md). Stored inference manifest aggregate digest:
  `3b168b98f23f5e42b00b6aa7ad8ca5438767bcb8c05f4cc4ce97087d800e0403`.
- Full-label source: `../2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json`,
  bound by that unit's `inputs/manifest.json`; full-label SHA256
  `1cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792`.
  Preserve all18 images,570 annotation IDs, class strings and norm1000 xyxy boxes.
  Use the actual image/prompt/media identities, not old retained513 metadata.
- Keep the anchor's median coordinate-output normalization in both arms, with
  the same current-weight rule at sampling, replay and greedy evaluation. This
  is a fixed policy choice, not evidence for norm-OFF behavior. The historical
  best val200 result was48.0328% mAP under median; it is not a new baseline.
- Ordinary greedy: same bound detection prompt/template, temperature0,
  repetition_penalty1, max_new_tokens3084, normal model EOS. No geometry mask,
  runtime deduplication, special minimum output length or first-error stop.
- Stochastic trajectory: same image/prompt and cap, temperature1, full-vocabulary
  support, no top-p/top-k truncation, repetition_penalty1, normal EOS. One sample
  per image/update in each arm. Bind a deterministic seed schedule based on
  base seed92711, update and image identity, shared in form across the two arms.
  Save actual tokens and both raw/policy likelihood channels with request IDs.

These18 images are an exposed training laboratory. Thirteen overlap the retained
val200 image list; neither full val200 nor the18 are a new independent generalization
test. Full val200 is not part of this first finite native package. Unmatched
predictions remain annotation-relative unknown, not automatic physical negatives.

## Frozen training contrast

Both arms restart from the same anchor with independent fresh AdamW state and
run16 continuous updates. Each update observes its own latest model on all18
images with one greedy and one stochastic trajectory per image, then updates
once after the prescribed image contributions. Later histories naturally differ
between arms; do not claim matched later prefixes. Sampling/greedy acquisition
counts are matched, but B has additional replay/backward work that must be reported.

For image i, P_i averages the positive loss first within each GT-owned row, then
across every GT row. The row loss is the maintained CE+0.1 type gate+0.01
conditional-order gate. Supervise all570 complete annotated rows in their bound
order; do not silently omit the two owners absent in the former redirect bank.
Exclude terminal EOS from positive targets: annotation-list completion is not
an assertion that the physical scene has no remaining objects. There is no old
M/CHAIN/GT-selected duplicate redirect branch in this experiment.

G_i is the mean over certified illegal coordinate decisions in that image's
latest greedy trajectory of `softplus(1 + max_illegal(z) - max_legal(z))`.
The legal set uses the actual causal coordinate prefix: starts0..998, x2>x1,
y2>y1, and proper coordinate-token family. The complement spans the full
vocabulary, including non-coordinate escape. No-error images contribute zero;
the outer denominator remains18. Use actual causal positions, and preserve
emission-versus-replay disagreement rather than silently relabeling the history.
Certification does not require a completed box or row. After an unambiguous
literal object description and box-start prefix, score an illegal coordinate-slot
action that was actually emitted, even if generation later ends or is capped.
For example, `<|object_ref_start|>person<|object_ref_end|><|box_start|><|coord_999|>`
contains an illegal x1 action and one G site without a box-end token. Later
malformation does not erase an already certified causal decision. Do not invent
missing actions, complete or repair the prefix, or guess slots after alignment
becomes ambiguous; record the unavailable certification. This clarification
preserves the actual-decision contract; the maintained completed-wrapper helper
is not its eligibility boundary. Duplicate events still require complete valid
boxes as defined below.
If an already-illegal start makes an end-slot legal set empty, record that dead
end and correct the offending start; do not take a maximum over an empty set or
invent a repaired history to make the end loss finite.

`L_A = mean_18(P_i + 0.1 G_i)`.
`L_B = mean_18(P_i + 0.1 G_i) + L_dup`.

Duplicate events are defined chronologically on complete, positive-area boxes.
The later row contributes one event iff its IoU with any earlier valid row is
strictly greater than0.9, regardless of category. Multiple qualifying historical
partners do not multiply that row's cost. No GT matching exemption. Literal and
other overlap diagnostics are retained separately. Invalid/malformed rows do not
become duplicate events; geometry/schema burdens remain visible.

Assign an event at the generated token that completes its object row. For each
generated token position t, let D_sample(i,>=t) and D_greedy(i,>=t) count remaining
duplicate events in the two trajectories, using the same absolute generated-token
index and zero beyond the respective end. Define fixed H=3084 and

`A_it = (-D_sample(i,>=t) + D_greedy(i,>=t)) / H`;
`L_dup = -mean_18(sum_t stopgrad(A_it) * log pi_theta(sample_token_it | prefix_it))`.

Use all actual sampled actions, including sampled EOS; capped output is censored
at the fixed horizon, not assigned invented EOS. The greedy baseline depends on
image/current model/token index, not the stochastic action, and receives no
gradient. It is not a same-prefix counterfactual value. Do not normalize by actual
response length/row count, standardize a singleton reward, clip advantages, add
GT coverage rewards, or substitute the absolute complete-row probability. The
duplicate coefficient is1, an explicit starting dose rather than tuned strength.
Record event supply, nonzero advantages, branch losses and gradient contributions;
zero duplicate signal is not evidence of algorithmic ineffectiveness.

The differentiable likelihood must implement the same normalized score policy
used to sample. Median factors depend on effective current output weights;
preserve that dependency in the gradient rather than silently detaching factors
or replaying raw logits. Bind the actual normalization precision/cast semantics
and validate the transform and causal likelihood path. Cached generation versus
full-history replay can have numerical differences; measure them, do not assume
bitwise backend parity or rewrite the stored behavior probabilities. Escalate a
policy-definition conflict before native work rather than silently changing norm.

Keep existing trainables: language DoRA and independent input/output special-token
deltas, no auxiliary head or KV/controller architecture. Initial AdamW proposal:
language lr1e-5, each delta lr5e-6, betas(0.9,0.999), eps1e-8, weight_decay0,
warmup0, global clip1, seed92711. Continuous optimizer state within each arm;
no parameter-block ablation, LR search or adaptive loss reweighting.

## Observation, finite stop and evidence limits

Greedy versions0..15 are already acquired for training; add one terminal greedy
evaluation at version16. Report all17 versions and the declared endpoint16,
including every image. Save checkpoints0/1/4/8/16 with actual optimizer/job
continuity recorded; do not rerun saved successful measurements merely for readback.
Reuse the maintained full570 evaluation contract, including its original matching
and category-credit rules. Duplication IoU>0.9 is separate from detection matching.

Report per-image and aggregate duplicate events, longest consecutive event burst,
literal repeats, overlap distribution/pairs, invalid/malformed outputs, valid and
unique rows, generated length, EOS/cap/empty cases, and annotation-relative
coverage/FN plus gained/lost/retained IDs against each arm's fresh baseline.
Show B-versus-A endpoint and trajectories without converting annotation matches
into physical truth. Earlier EOS, box jitter, malformed escape, reduced coverage
or pushing repetition past the horizon can lower rule cost; these are possible
mixed/negative results, not automatically implementation faults.

End each primary arm after16 updates regardless of quality. No user GPU-hours cap
or5%/other retention gate exists. Native packets still specify logical work,
normal completion, failure/cleanup ownership and an operational observation
deadline based on qualified execution; a timeout is not permission to relaunch.
Nonfinite values, corrupted identities/positions, unsupported policy probabilities,
distributed failure or OOM are technical failures requiring repair/rebinding,
not scientific nulls. No automatic dose extension, retuning or new unit.

## Worker package, efficiency and acceptance

Canonical cwd is `/data/CoordExp/.worktrees/research-probes`. New module owner:
`probes/rule_stability/` with minimal responsibility-based files and thin CLI;
focused tests under `tests/probes/test_rule_stability*.py`. Worker may make narrow
necessary changes to maintained Qwen generation/replay/loading and existing owner
or probe helper boundaries, with explicit opt-in behavior and affected caller
checks. Preserve unrelated source/dirty work and historical evidence. No source
execution from outputs, other worktrees, scratch or /external. No push or shared
asset mutation.

Reuse current full-label rendering/object-owned atom selection, row/image loss
reduction, seeded generation likelihood channels, resident inference, source-bound
export/readback and18-image workload balancing. Reuse mechanisms, not legacy513
provenance or hardcoded greedy assumptions. Same-version processor input reuse is
available; cached vision features or backward graphs must not be assumed reusable.
Prefer measured end-to-end efficiency to a new framework. Account separately for
startup, acquisition, replay/backward, refresh/export, finalizer and cleanup, with
actual work counts, rank skew and relevant memory peaks.

Initial deliverable is a CPU candidate and exact native qualification/primary
proposals. Follow full-pipeline-smoke: first traverse the actual rank-entry ->
artifact write -> readback -> offline-consumer sequence with only model compute
substituted. Falsify wrong > versus >=, class filtering, pairwise double counting,
history/position shifts, EOS masking, detached/mismatched score transforms,
uneven-rank weighting, output collisions and stale self-including manifests.
Use existing frameworks/fixtures and meaningful counterexamples. Routine package
repair/recheck is worker-owned; do not wait for lead approval at each technical fix.

Output root: worktree `outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/`.
Disposable transport messages: worktree `.local/scratch/rule-stability-iou90/`.
Worker writes implementation/tests, narrow required helpers, `probes/README.md`
entry and this unit's `results.md` technical log. Lead alone writes `unit.md`,
`state.json`, research index/catalog and rulings. Explicit owned-path local commits
are allowed after checks; no broad stage/reset/clean/unlock or publication.

At CPU candidate return actual exits, source/diff identity, input/anchor bindings,
inventory/cost bounds, proposed exact commands, consumer checks and unresolved
native risks. Hash source once at the clean execution boundary, not every repair.
Stop before model load/GPU/model forwards until the lead's exact packet. After a
release, worker owns that invocation through checks, in-scope repairs permitted
by the packet, cleanup and direct return. No successful test/model/readback repeats
without a changed source/input or a concrete unresolved concern. Lead accepts the
final consumer and scientific result, not transport status or worker self-report.
