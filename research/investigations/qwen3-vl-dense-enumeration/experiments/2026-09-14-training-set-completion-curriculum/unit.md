---
title: Cumulative training-set completion through self-rollout and verified route learning
description: Autonomous bounded iterations toward complete trusted-owner coverage on each cumulative training stage before expansion.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-14-training-set-completion-curriculum
topic: qwen3-vl-dense-enumeration
status: paused_by_user_after_fourth_fit
evidence_status: fourth_fit_accepted_stage_incomplete
updated: 2026-09-15
---

# Decision and authority

User-owned objective, quoted from this task:

> 把探索得到的可信对象支持与有用后续路径，学习进普通模型参数，使模型从原图、空 assistant 前缀出发，通过一次自然 greedy，完成更多真实对象，同时不把问题转移到旧对象丢失、重复、错误几何或停止异常上。

The user selected **GT plus verified old and new physical owners** as the
training target population, and **every cumulative stage must pass before
adding training images**. Early validation/generalization measurements are
deferred and cannot veto this training-first investigation. The user then
offered autonomous multi-round research, eight 80-GB GPUs, no aggregate compute
ceiling, and training jobs shorter than five hours. Self-rollout and low-temperature
sampling are the principal discovery sources. Failure-driven attention/KV
diagnosis and relevant inference-backend debugging are within scope.

The user subsequently accepted both remaining rulings:

1. Confirmed pre-existing false predictions may remain temporarily, but new
   confirmed errors must be repaired before stage growth. This is permission
   to tolerate residual debt, not permission to reinforce errors as positives.
2. Owner coverage uses physical instance identity and reasonable geometry;
   category errors are recorded separately, not automatically as missing owners.

The semantic authorization is complete. Before each model run, bind its exact
input, model, sampling/training recipe, and finite budget in the stage manifest.
Implementation details and bounded recipe adjustments within these objectives
are delegated to the lead; they do not require repeated user confirmation.

The user additionally requested practical execution primarily with Luna,
Terra, and Sol. Owner review must inspect one sample at a time using the
original image, plotted bbox overlay, and the corresponding crop, and persist
per-sample decisions. A compressed multi-image contact sheet cannot substitute
for this review. A plot/crop is a viewing aid, not new physical evidence by itself.

Latest user waiting rule: wait for subagents with long event-driven native
waits, without polling. Reserve wake-me-up for actual launched long GPU jobs,
not native-worker settlement or a CPU/package completion log.

# Primary question and contrast

Current execution checkpoint: the first 44 native trajectories are
**lead-accepted acquisition evidence only**. Root independently replayed saved
tokens through decoding and parsing, checked 62 artifact bindings and nine
successful worker terminals, and reran the five focused tests. The bound
receipt is `stage01-acquisition-v1-retry1-config-batch2/root-acquisition-acceptance.json`
under the artifact root below. All 44 ended naturally; the raw outputs contain
924 valid boxes and eight invalid-geometry rows. Those eight remain debts.
All 11 single-image owner reviews are complete. Root corrected a duplicate
new-person identity, excluded five unresolved shelf/reflection candidates from
admission, and reviewed full reference boxes for three partial raw predictions.
The fixed population is 220 atomic owners (169 GT + one prior non-GT + 50 new),
with six crowd annotations separate. `target-owners-complete-v2.json` binds all
220 reference geometries. No training-stage pass or physical-zero result is
claimed.

The current real-entry technical slice is recorded in
`runtime-preparation-v1/manifest.json` and `runtime-smoke-v1/launch.json`.
It uses the individually reviewed 210457 temperature-0.7 route (48 unchanged
tokens, five trusted owners), two single-GPU updates, and a 600-second hard
training bound. Its contrast is uninterrupted step 2 versus cold resume from
step 1, with exact adapter/optimizer and natural-token readback checks. It
qualifies execution only; the scientific training stage remains all 11 images.
The low-weight raw-axis hinge now uses the verified tokenizer coordinate table
and normalized coordinates, without box canonicalization. No raw-token
geometry guarantee is inferred from this auxiliary expectation loss.

From a bound native anchor, can ordinary parameter learning of credible
self-rollout repairs jointly complete every trusted owner on the current
cumulative training set, with zero repeated owners, malformed output, invalid
geometry, and cap debt, under original-image, empty-assistant-prefix greedy?

The primary comparison in each iteration is its immutable parent versus the
candidate on exactly the same training images and target-set version. It tests
the whole recipe unless a separate matched treatment contrast is frozen.
Conditional scores, individual corrected rows, and mechanism probes are
diagnostics; they do not replace natural completion. No validation panel owns
stage acceptance.

# Evidence anchors

- [Current North Star](https://app.notion.com/p/3db9d9ce3f59813ca946f3df64d627c4)
  and [evidence appendix](https://app.notion.com/p/3db9d9ce3f5981f7b874cfffc3f64270),
  fetched on 2026-09-14; the current user's training-first and stage-growth
  decisions supersede earlier proposed validation gates.
- [Human13 pure CE](../2026-09-05-human13-pure-ce-replay/results.md): natural
  392/392 annotated owners at IoU50/60/80, zero duplicate/malformed/cap debt,
  on its exact Source/magnitude-only recipe. Do not repeat this merely to
  prove tiny-set fit or transplant its learning rate into another parameter surface.
- [N16](../2026-09-12-native-owner-scale-and-state/results.md): useful natural
  branch compilation, with real retention and new-loop debt.
- [Latest conditional/physical results](../2026-09-14-label-vs-compilation/results.md):
  successful c and immediate w can coexist with later loss or reselection.

# Cumulative task and acceptance

Each image has a versioned trusted owner set T: original GT plus reviewed
native and exploratory additions, deduplicated by physical identity. Disputed
GT and unknown proposals remain explicit; neither silent deletion nor matching
convenience may shrink the declared denominator. Newly verified support creates
a new task version rather than retroactively changing old results.

Every stage includes all earlier training images and owners. A prediction can
credit at most one atomic owner; group boxes are separate. Different legitimate
orders, descriptions, and boxes are allowed. The stage manifest will bind the
existing geometric matching rule and case-specific reviewed identity/extent
decisions; exact token identity is not the deployment target.

Stage completion requires zero missing trusted owners on every image, zero
physical repeats, zero malformed raw rows, zero invalid generated boxes, and
natural EOS without cap. Strict geometric repeat counts are only a selector;
same-owner reboxing below its threshold still counts. Parser dropping, output
deduplication, retries, multi-sample unions, and forced histories cannot produce
a passing natural output. Pre-existing confirmed false predictions and class
errors may temporarily remain; newly introduced confirmed errors must be
repaired before growth. Freeze their baseline identities/quality per image,
not merely an aggregate FP count. Old repeated/malformed/invalid/capped outputs
receive no exception from their mandatory zero-debt conditions.
Unknown output stays separate and receives neither automatic positive nor
negative labels; uncertainty limits the physical-zero claim.

## FP treatment and direct CE masks

GT-unmatched is a matching status, not a truth label. Apply these rules:

- Verified real unique owner: positive support, including unannotated owners;
  supervise only verified fields and add the owner to a versioned target set.
- Unknown physical support: no direct CE target on the unverified row; retain
  its actual token IDs and positions when using a downstream suffix witnessed
  in that context. No automatic negative, EOS, deletion, or pseudo-label.
- Verified physical owner with incorrect/uncertain category: retain correct
  geometry support; correct the description when independently known, otherwise
  mask its unverified description targets. Do not train a known wrong class.
- Confirmed false object or clearly wrong extent: do not positively imitate
  that erroneous content. Prefer a verified alternative at the earlier harmful
  branch. A false row cannot be removed from a stored trajectory while keeping
  its old suffix certificate; generate/revalidate the new condition instead.
- A targeted negative objective is a possible later treatment only for
  verified bad behavior with a frozen scope/control, not a blanket penalty on
  annotation-relative FP or an implicit prerequisite for the initial CE fit.

Masking means excluding direct target positions from the CE numerator and
its active-target normalization; it does not delete tokens or detach their
hidden states. Other supervised losses can still backpropagate through the
attention path processing those tokens, and softmax/shared-parameter updates
can lower other useful alternatives. Thus neutral is a label policy, not a
guarantee of zero gradient or preserved output probability. The primary
natural-readback ledger measures such side effects.

Intermediate training regressions are allowed. A stage's completion must
survive cumulative-task readback and saved-model cold readback; repeating one
deterministic checkpoint does not independently establish training stability.
When a stage fails, retain its unsolved images and pause growth rather than
substituting easier images or averaging away old-owner losses.

# Learning loop and CE safeguards

1. Bind the anchor, input preprocessing, native prompt, tokenizer, adapter,
   exact trainable surface, target-set version, output budget, and initial
   complete natural outputs. Use only training-eligible images; previous
   confirmation panels do not become label or discovery sources.
2. Discover credible objects and useful continuations principally from current
   self-rollout and sampling at temperatures 0.1, 0.3, and 0.7. Bind sampling
   count and maximum corrections before each acquisition batch. The assistant
   budget remains 3084 including any supplied history unless the user changes it.
3. Accept verified useful objects/segments without demanding positive margin,
   current greedy success, or an already perfect full trajectory. Execute and
   inspect complete consequences; remaining defects become explicit repair
   obligations. Cross-sample object support is not itself a joint route witness.
4. Learn coherent verified rows/routes under their exact histories using a
   simple CE-based starting recipe and cumulative successful-task replay.
   Identical conditions cannot have incompatible unique-greedy demands. Masked
   unknown spans remain in the actual conditioning history; never delete them
   and reuse old suffix distributions or capability certificates.
5. CE masking does not guarantee preservation: softmax competition and shared
   updates can suppress unselected useful actions. Do not blindly reinforce
   unknown content, already-completed-owner repetition, or old premature EOS.
   Balance old and new examples with one declared normalization rather than
   silently adding another independently weighted bank at every growth step.
6. Save intermediate model and optimizer state where supported; bind whether
   continuation is exact optimizer resume or an explicitly fresh-optimizer
   iteration. Inspect natural outputs at scheduled checkpoints, then repair
   the first harmful decision rather than any harmless literal divergence.
   Repair an irreversible output error before its occurrence, not by pretending
   a later suffix can erase it.
7. Full-row margins and conditional CE locate failure. Shared task realization
   and complete natural outputs own acceptance. A changed objective, target-set
   version, checkpoint family, or claimed mechanism gets an explicit new
   iteration/phase record before execution, without rewriting old evidence.

Low-weight bbox validity support is desired. The currently located historical
bbox_geo term expectation-decodes and canonicalizes boxes before regression;
it is not a guarantee of raw greedy x2>x1 and y2>y1. Confirm the intended
implementation before reuse. Any new experiment-local validity term requires
a minimal real-gradient check and actual generated-box readback; positive
signed area alone is insufficient because both axes could be reversed.

# Mechanism exploration

The lead may insert a small diagnostic when it could change the next repair:
which token/site routes to the repeated object, what historical/visual evidence
is consumed, and whether an attention or KV intervention selectively changes
the harmful decision and its full consequences. Start from a saved native
failure and one discriminating prediction. Use the relevant native/self and
wrong-target or magnitude control rather than an automatic large matrix.

Attention maps and decodability are descriptive; a causal intervention must
show its actual output consequence. Easy escape after a perturbation does not
prove instance-specific state or stable trainability. Distinguish sampling
variation, checkpoint/history dependence, and backend numerical drift. A
diagnostic failure closes that diagnostic, not the primary training question.
KV/attention instrumentation can be an instrument or teacher; final acceptance
still uses ordinary parameters without an inference-time oracle. Any proposed
new deployment architecture is a separate user-owned decision.

# Runtime, scope, and stopping

- Fixed checkout: /data/CoordExp/.worktrees/research-probes. Preserve the
  existing untracked post-Pro handoff and all immutable old artifacts.
- Runtime artifacts: /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum,
  with a distinct immutable run directory per attempt. Interpretation stays here.
- Start with the shortest accepted HF path. Use available GPUs according to
  measured throughput; do not require all eight ranks for a tiny task or wait
  for random stress occupancy to disappear.
- Initial training chunks should be short enough to inspect several doses;
  every job has a declared wall-time budget below five hours and checkpoint/
  terminal handling. No unlimited extension of one training process.
- If inference/refresh becomes a measured bottleneck, qualify the smallest
  vLLM collection path independently. Preserve model materialization and
  precision identity; BF16 throughput is not exact dynamic-HF parity. Native HF
  replay/learning remains the reference until another deployment contract is
  explicitly qualified. Do not build generic online-RL machinery just to collect trajectories.
- All long model jobs run in named tmux sessions with persistent logs and
  terminal receipts. For a launched long GPU producer, arm and verify its
  durable wake monitor before ending for that wait. Subagents use long native
  event-driven waits without a wake monitor. A queued notification, idle task,
  or exited process is not scientific completion.
- Before each round, bind a finite acquisition/training/repair budget and the
  decision its result changes. A failed dose is not an impossibility result.
  If progress stalls, stop growth, diagnose the observed bottleneck, and record
  the next bounded recipe change. Do not let unrelated mechanism work or
  backend repairs indefinitely replace the training objective.
- No early validation veto, production promotion, broad architecture rewrite,
  or automatic external publication. Later validation begins as a separately
  declared generalization phase after meaningful cumulative training success.

# Current state

**User stop boundary executed:** fourth-fit64→256 training and current11-image
native evaluation are complete and root-accepted. Stop for discussion. No fifth
fit, data expansion, new recipe/mechanism probe, held-out evaluation or GPU work.

Latest authoritative review rule: inherit existing class-agnostic one-to-one
IoU>=0.5 matches; `view_image` only adjudicates unmatched rows. Prior stricter
matched-row visual judgments remain historical evidence and do not veto matches.
Fixed training232: parent64 177 covered→final256 212;41 gained,6 lost,20 missing.
Current all-known248 after this review:183→213 covered,35 final missing.
Final has11 physical repeat rows,3 confirmed false,4 unknown,18 parser-invalid,
and11/11 natural EOS. Stage has not passed. All858 raw rows are accounted for.

Latest local annotation export is`annotations-with-unlabeled-v4/annotations.jsonl`:
169 original GT unchanged plus79 valid unlabeled,19 classes unknown/null. Two
new unmatched owner admissions are saved with actual local review and visual
provenance. Training input remains frozen232. See
[final-round report](2026-09-15-fourth-fit-final-review.md) for per-image counts,
matching semantics, execution verification, training curve and stopped scope.

The records below retain earlier phase context.

Latest checkpoint: second-fit training, all 33 cold native readbacks and the
full step16 single-image physical review are root-accepted as evidence. On the
fixed v3 population, corrected parent coverage is 153/228 and second-fit step16
coverage is 159/228: nine newly covered fixed owners and three lost old owners.
The original parent155 count is superseded by a symmetric correction of two
ambiguous image25274 extents; no denominator was removed. Step16 has 125 repeat
rows, 175 confirmed false rows, 612 parser drops and four caps. Step32/64 have
six/eight caps; their full physical owner populations were not reviewed.
No stage promotion or image growth is justified.

The four-cell 210457 stopping diagnostic is complete: the two coordinate-token
history changes do not flip the next token within either tested checkpoint.
Root has admitted four further physical owners and disambiguated two old
reference boxes. The next fixed task is `target-owners-complete-v4.json`, 232
owners on the same 11 images, with six crowds separate. Complete corrected
teacher bank v2 is root-accepted. Third fit completed64 updates/704 forwards
in1160.6 seconds and all33 durable cold greedy readbacks. Root accepted the
training/optimizer evidence; readback audit and full step32 physical review
are in progress. Step16/32/64 have9/11/10 natural EOS outputs and2/0/1 caps.
Step32 still has5 parser-dropped rows; all-EOS is not a stage pass. Objective
continues declining from1.8239 to0.6916, without an observed plateau. Prepare
a strictly unchanged-target/optimizer continuation through total256 updates,
preserving exact AdamW/RNG state and frozen earlier source bindings; launch
is now executing after root acceptance of that execution path.

The first all-11 fit and all single-image physical reviews are complete.
Against the frozen first-fit 220-owner target, step 16 covers 151 owners and
step 32 covers 148 with reasonable geometry. These review-derived counts are
bound by `first-fit-review-extraction-v1/root-acceptance.json` and
`summary-v2.json`; the earlier summary counted image keys instead of owner-set
members in its aggregate union and is superseded only for that summary.
Step 32 completes image 210457, but the cumulative stage has not passed.

Eight newly verified owners now enter `target-owners-complete-v3.json`, making
the next task 228 atomic owners on the same 11 images, with six crowd regions
separate. The first-fit comparison remains 220. Root admission and visual
evidence are in `first-fit-new-owner-admissions-v1/admissions.json`.

The accepted stage-02 refresh collected 44 original-image, empty-prefix
trajectories from first-fit step 16. All 11 greedy routes exactly reproduce
the saved readback. All 44 end naturally, with 808 valid raw boxes, no invalid
rows, and 7,729 generated tokens. This is acquisition evidence, not physical
completion. No GPU producer or wake monitor from that phase remains active.

Next is the bounded single-owner repair acquisition specified below. The lead
owns meaning, catalog admission, masks and acceptance. Terra
`/root/masked_route_train` owns its producer and GPU execution; Luna
`/root/stage01_support` owns a disjoint projection of already reviewed prefix
decisions. Native workers settle through long native waits, not wake monitors.

Historical monitor `1c4bc563-fd66-45b3-a3c8-95bf5a0b6d02` was armed for
worker terminal logs, then **cancelled before any delivery attempt** following
the user's corrected waiting rule. The workers continue unaffected and the
lead awaits them through native waits. The original armed receipt is retained:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/supervision/monitor-armed.json`.
Current cancellation receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/supervision/monitor-cancelled.json`.
No monitor currently observes these package logs. The initial native-worker
condition was rejected before registration because
the current App lacks `thread/agent/observe`; the accepted log condition is a
documented alternative, with no daemon restart or plugin change. These are
historical attempts, not a reason to use worker monitors in future. A terminal
marker is candidate settlement only; inspect exact acquisition/support
artifacts before accepting any result. For a future GPU-job wake, call
wake_me_up_status once before inspecting the producer's actual artifacts.

## Native readiness accepted on 2026-09-14

The lead inspected the underlying generation and trainer ranges, rather than
accepting a worker summary alone. The candidate initial lineage is N16, whose
adapter exists at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-12-native-owner-scale-and-state/scale/training/full-fixedP-N16-v2/adapter`.
Its eleven original positive training images are the candidate first cohort:
25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415,
and 528944 (COCO 2017 train image IDs). Their full trusted owner support still
needs a bound ledger; sixteen old positive packages are not that ledger.

Reuse candidates are `src/qwen/generation.py` for seeded native sampling and
literal histories, `probes/parallel_owner_research/training.py` for native
DoRA learning mechanics, and `probes/native_owner_scale/evaluation.py` for
saved-adapter natural readback. Existing N16 acquisition hardcodes greedy,
its packet producer fixes the historical schedule, and its trainer saves an
adapter but no reloadable optimizer. The new loop needs bounded sampling,
schedule, and intermediate-state wiring; the old sealed files/packets must
not be rewritten to give historical runs new capabilities.

The old N16 packet also includes 54 reference images. They are not silently
reused as an uncounted training/retention side set: if any additional image
supplies a learning loss in this curriculum, it is explicitly part of the
declared cumulative training population and its obligations are recorded.
No vLLM dependency is required to reach the first scientific iteration.

## First acquisition boundary

Use the eleven bound N16 training image IDs above and the accepted N16 adapter.
For each image collect one empty-prefix native greedy output and one seeded
full-output sample at each temperature 0.1, 0.3, and 0.7: 44 unique requests,
each with a total assistant budget of 3084 tokens, repetition penalty 1.0,
top_p 1.0, top_k 0. Bind the exact seeds, config, media, and model before launch.
This first pass has no forced corrections and does not claim that all desired
owners must already be found. It supplies source-bound native failures and
exploratory support for the first repair decision.

The first two requests form the real persistence/readback smoke and are reused
in the 44-request result if their execution remains valid; do not duplicate
their inference for convenience. Acquisition processes have a one-hour budget
and use named tmux plus terminal receipts. If the bounded pass cannot finish,
preserve completed requests and classify the failure before any new attempt.
Training and new owner labels follow lead acceptance and visual support review
of the resulting artifacts. No old validation or confirmation image is used.

# First masked-route fit: frozen execution decision

The two-update single-image technical slice passed exact cold optimizer/adapter
resume (588 tensors) and natural token readback. Its natural output shortened
from 95 raw rows to seven, but still repeated the left cup and emitted one
uncertain background feature. The first scientific fit starts afresh from N16,
not from that technical checkpoint.

Use all 11 images, one coherent observed trajectory per image, with trusted
fields supervised and all original tokens retained. The first bank selects
maximum reviewed reasonable-geometry owner coverage, breaking ties by raw
error debt and then lower temperature. The route bank is partial; a complete
observed trajectory is not a prerequisite to learning useful support. Only
210457 currently has a trusted EOS target. Unknowns and confirmed bad rows are
not positive targets and remain raw-output obligations.

Root found a concrete mask bug before launch: image 99937's book/media-case
class was overridden to unknown, but the first bank consulted its older
verified class. Preserve bank v1 as rejected training input; v2 must consume
effective class authority. Geometry stays trusted, class CE is masked.

The frozen first fit is 32 updates of all 588 language DoRA tensors from N16,
AdamW learning rate 1e-5, equal mean over 11 per-image active-token means,
raw-axis hinge weight 0.01, checkpoint steps 8/16/32, and a 3,600-second hard
training bound (352 image forwards). No undeclared normal bank or validation
examples. GPU0 performs the fit; GPUs1/2/3 perform cold original-image empty-
assistant-prefix greedy readbacks of the three saved checkpoints in parallel
afterward, each with a 900-second outer bound. This deliberately uses the
already-qualified single-GPU gradient path before changing training topology.

Contrast: each checkpoint versus the same N16 anchor on the fixed 220-owner
task. Decision-bearing outcomes are per-image physical coverage and retained
old owners, raw duplication, invalid/malformed output, EOS/caps, and new
confirmed FP/class debt. IoU50/80 matching is a review selector, not physical
truth or stage acceptance. A low CE value alone cannot promote the stage.
The strongest alternative is teacher-forced route fit without natural branch
access or retention; three cold readbacks distinguish that from initial
parameter-learning benefit. Stop the individual job at the frozen bound;
regression pauses growth and directs a new bounded repair/refresh iteration.

The independent eight-array stop-logit diagnostic is lead accepted in
`stage01-210457-stop-logits-v1/retry1/root-acceptance.json`. At each of four
exact five-owner histories, row-open stays argmax. Two technical updates
raise EOS versus row-open by about 0.67 logits, without making EOS greedy.
The successful sampled stop is therefore consistent with selecting a
non-argmax EOS; this is not evidence of a distinct KV-cache mechanism.

## First-fit GPU checkpoint

The first all11 training job completed 32 updates / 352 image forwards in
587.434 seconds. Input bank v2 is lead accepted in
`first-masked-route-bank-v2/root-acceptance.json`; active CE is 1,420 tokens,
including 604 coordinate positions. The manifest is
`first-fit-preparation-v1/manifest.json`; tmux session `tsc-first-fit-v1`;
producer PID 3444754. The phase controller is now completing parallel cold
greedy readbacks for steps8/16/32. This is a completed training runtime, not
a scientific benefit or stage-pass result. GPU phase terminal and logs live
in `first-fit-v1/`. Wake-me-up, if armed, monitors only that actual GPU phase.
All native subagent preparation tasks have settled via native long waits.

Next: inspect `first-fit-v1/terminal.json` plus all3 readback/exit receipts,
score their original token streams against `target-owners-complete-v2.json`
with `probes.training_set_completion.readback_selectors`, and compare the
fixed220 references with the accepted N16 anchor selectors at
`anchor-comparison-v1/scored.json`. Baseline geometric IoU50 matches139/220;
this is diagnostic only and overcredits known extent errors in59571/417044.
Readback visual renderer: `B/first-fit-visualization-preparation-v1/render_readback.py`
under the artifact root (the literal B directory is part of its actual path).
It is lead accepted for preserving rawrows/crops at the checked native image
dimensions. Reuse same-image reviewers for actual first-fit output, with
bbox overlay + crop; matched/unmatched status is never an FP truth label.
Keep all11 images. Continue bounded repair/refresh iterations without early
validation or stage expansion until the user-owned completion contract passes.

## First-fit cold results and refresh decision

The entire GPU phase completed in801.393seconds (32-update training587.434sec).
All33 saved readbacks are verified original-image, empty-assistant-prefix
**greedy**; the inherited route_id labels refer to training provenance and do
not describe the readback sampling policy. Root independently checked all
three saved adapter identities,588tensor surfaces,32update receipts,352forwards,
and33 prompt/media/token/termination records. `first-fit-v1/root-runtime-acceptance.json`
binds that acceptance. Wake registration raced already-completed GPU output
and returned PID-not-found; no monitor armed.

Geometric selectors (not physical owner truth): N16 has139/220 IoU50 matches
and107 IoU80; step8 has148/103; step16 has152/105; step32 has146/104. Raw
invalid drops are3/5/0/0 respectively. All outputs naturalEOS. StrictIoU>.95
repeat pairs are133/184/0/0, but physical repeats remain belowthatthreshold.
Loss falls1.236->0.631 while native geometry/retention is nonmonotone. This
exposes a CE side effect to repair; it does not justify an early validation
veto. All per-image owner reviews for steps 16 and 32 subsequently completed
using plotted bboxes and individual crops; see the current-state section for
their root-accepted geometry-conditioned counts.

Image210457 is root-confirmed complete atstep32: five fixed owners, reasonable
geometry, zero repeat/unknown/false/class/invalid/malformed debt and naturalEOS.
Step16 still has one repeated leftcup and one unknown background feature.
This singlecheckpoint image result does not establish all11completion or
training stability. Image99937 illustrates the geometry condition: class-
agnostic matchingcredits7/8, but a chair extent is uncertain and a monitor
stand/base is omitted, leaving5/8 qualifiedfixedowners. Its printed-item
owner was already admitted and must not be mislabeled pendingnew.

Freeze a second44trajectory acquisition fromstep16, using all11 images and
greedy plus temperatures.1/.3/.7, new seedroot2026091402, cap3084 and identical
FP32HFSDPA inputs. The checkpoint choice is a current training diagnostic
for discovery, not promotion. First compare one actual nativegreedy route
exactly to savedstep16, then acquire43remaining requests and compare all11
greedy token streams. Artifacts `stage02-refresh-v1/`; tmux
`coordexp-stage02-refresh-v1`; worker hardbound1800seconds. Root authorized
this bounded refresh before launch; no heldout/validation data or target
version change. KeepT220 fixed for the first-fit comparison. Newly verified
objects may enter a separately versioned next target catalog after root review.

## Single-owner repair: frozen acquisition decision

Low-temperature refresh still omits known GT owners. The next bounded question
is whether one verified missing owner, appended to the student's actual
history, provides a learnable continuation without losing existing owners.
This is label-assisted self-route repair; it is not spontaneous discovery or
a claim that the forced route is naturally reachable.

The frozen `stage03-single-owner-repair-v1/owner-plan.json` binds target v3
(228 owners on the same 11 images) and generator parent first-fit step 16.
For nine images, retain that parent's literal greedy token history except its
terminal EOS, append one physically absent atomic GT owner with its reference
geometry and category, then generate a fresh suffix. For image 323322 retain
the first six literal rows from the cleaner step-32 route and replace the
last plant row with the full GT reference; generation still uses step 16 and
records this cross-checkpoint condition. Retain the clean five-owner step-32
route for image 210457, including its reviewed EOS, without another decode.

Generate four fresh suffixes per repaired image: greedy and seeded sampling
at 0.1, 0.3 and 0.7, for 40 requests. The total assistant cap of 3084 includes
every forced prefix token. Use the same original image, prompt, FP32 HF SDPA,
coordinate tokenizer and embedding delta. Each worker has an 1800-second
bound; eight independent workers use the available GPUs. Preserve raw suffix
IDs, parser drops, forced/generated boundaries, stop reasons and model/source
identity. The declared cap, not a cleanup transformation, bounds output.

Only reviewed credible prefix fields and the verified correction can initially
receive positive CE. Newly generated suffix rows require their own physical
review before positive admission. Unknown/bad original tokens remain in the
teacher history with direct CE masked. No original suffix certificate survives
the altered prefix. Positive EOS requires a complete, clean per-image route;
otherwise it stays masked. Subsequent training must bind its new route bank,
parent, optimizer initialization, finite dose and cold native readback before
launch. Natural original-image empty-prefix greedy against the same v3 parent
comparison owns the learning result; conditional releases are diagnostic only.

## Second fit: bounded learning decision

After root acceptance of the repair bank, start from first-fit step 16 with
an explicitly fresh AdamW optimizer. The new bank and optimizer initialization
make this a whole-recipe comparison with that parent, not an isolated causal
estimate for CE or the appended row. Use all 11 images and target v3 (228),
one unchanged selected repair trajectory per image. Selection minimizes fresh
suffix length, breaking ties by lower temperature. It is a one-owner teaching
bank: unreviewed suffix rows are retained but entirely masked, and EOS remains
masked except on fully verified clean routes. Partial trusted support does not
need a complete successful trajectory to be learned.

Keep the already qualified 588-tensor language DoRA surface, FP32 SDPA,
per-image active-token CE normalization, AdamW learning rate 1e-5, gradient
clip 1.0, and raw-axis hinge weight 0.01. Run 64 updates (704 image forwards),
save at steps 16/32/64, and enforce a 3600-second training bound. GPU0 trains;
GPUs1/2/3 independently reload the three saved adapters for cold, original-
image, empty-assistant-prefix greedy readbacks afterward. Each readback has
a 900-second outer bound. The prepared driver is `second-fit-v1/run.py`;
this paragraph is authorization of a finite next dose, not a launch receipt.

The decision is whether the nine previously absent GT owners and repaired
plant geometry become naturally available while retained owners and clean
completion survive. Inspect every cumulative image and count physical repeats,
raw validity, caps, and confirmed FP/class debt. No new image or validation
panel is introduced. If this dose regresses, preserve the checkpoint evidence,
pause growth, and use its actual failure to select the next bounded repair.

## Second-fit launch checkpoint

The accepted stage-03 acquisition is in
`stage03-single-owner-repair-v1-retry2/`, not the earlier failed launch roots.
It produced 40 conditional releases, 1544 generated suffix tokens/forwards,
952 valid rows and two raw parser drops. All suffixes ended naturally within
the total 3084-token assistant cap. Root redecoded and reparsed all 40 outputs,
checked forced boundaries and eight loaded FP32 SDPA parent-16 receipts, and
published `root-acceptance.json`. Its consumer-policy correction explicitly
sets `empty_assistant_prefix=false`; these are teaching conditions.

The original manifest failed cold JSON hashing because numeric dictionary keys
changed type on serialization. Retry1 preserved that reproducible failure;
retry2 normalizes keys before hashing. This changed no request, target or
prefix tokens. Initial launch records had no model evidence. The successful
retry uses an explicit selected runtime, logging from the first command, and
an actual controller PID receipt. No earlier failed launch counts as a run.

Root accepted `stage03-mask-preparation-v2` after correcting the crowd row
25274:p16 to fully masked (195 source rows, 988 checked bindings). The final
`stage03-repair-route-bank-v1/root-acceptance.json` binds all 11 literal routes
and 217 trace cards: 195 reviewed prefix rows, 10 verified forced GT rows,
and 12 unreviewed suffix rows. There are 164 trusted boxes and 1556 active CE
tokens. Every suffix object is masked; only image 210457 has positive EOS.
The repaired 323322 path still emits an extra row, so it has no positive EOS.

Second-fit manifest: `second-fit-preparation-v1/manifest.json`, content hash
`9fad38363125b0148290eba533a49281eb71143e495cf26293f33ba290a55707`.
The live parent fingerprint is
`885b07a07990f5ee171f8364007302ff2af7d5cbcceb3f33f9c665ba5de644b6`.
The phase is launched in tmux `coordexp-second-fit-v1`, actual controller PID
3590508 and training child PID 3590512. Root verified `/proc` command identity
and the producer/training launch receipts. This is launch acceptance only;
no second-fit quality result is claimed yet. Phase log and terminal are under
`second-fit-v1/`. The wake monitor, when armed, observes only this actual GPU
phase; all native agent packages have settled.

After wake: inspect the monitor decision once, then phase/training terminal,
exit receipts, model/adapter identities, 64 updates / 704 image forwards, and
all three cold readbacks. Use target v3 (228) for this whole-recipe comparison;
the parent native token streams are saved first-fit step16 and its v3 geometry
selectors already exist in `stage02-refresh-selectors-v1/scored-t0.json`.
Do not compare new v3 counts directly to old v2 counts. Project the accepted
physical parent reviews onto the same v3 owner identities before reporting a
physical coverage delta; no unchanged GT-unmatched row becomes false by default.
Render every actual readback image with bbox overlay and crops, reuse the
single-image Luna/Terra/Sol reviewers, and keep all 11 images. Determine whether
the nine added GT owners and full plant geometry become naturally available,
then select the next bounded repair from the actual retention/error pattern.
No early validation panel or image expansion is authorized by this launch.

GPU phase wake is **armed**: monitor
`0200d078-8ecc-48e6-a1b4-0637cf811fed`, key
`tsc-second-fit-v1-gpu-phase-20260914`, expiry 5400 seconds after registration.
It observes the phase success/failure log or actual controller PID 3590508 exit.
Receipt: `second-fit-v1/wake-armed.json`. On delivery, read its decision once;
delivery itself is not training completion or scientific acceptance.

## Second-fit wake: training accepted, readbacks need recovery

Monitor `0200d078-8ecc-48e6-a1b4-0637cf811fed` fired on the phase failure
marker and actual controller exit. Root read the monitor decision exactly
once; `second-fit-v1/wake-decision.json` preserves it. The monitor is consumed
and must not be reused for another producer.

Training itself completed normally: 64 updates, 704 image forwards, 1148.802
seconds, fresh optimizer, all 588 language DoRA tensors. Root validated the
64 update records, source manifest, saved adapters and all three optimizer
states, including their 588 per-parameter step counters. Objective means at
steps 1/16/32/64 are 0.9999/0.6453/0.5070/0.3478. This is learning-runtime
evidence only, bound by `second-fit-v1/root-training-acceptance.json`.

All three cold readback processes loaded their model but exceeded the
900-second outer bound and exited -15. Their all-images-at-end publication
left no usable readback JSON. Thus the phase failed during readback, with no
second-fit natural quality result. The logs do not establish whether long
generated loops or runtime throughput caused the timeout. Do not infer either
without per-image evidence, and do not retrain the accepted checkpoints.

Recover the exact same 33 checkpoint-image requests under
`second-fit-readback-recovery-v1/`, using eight GPU workers and no more than
five images per worker. Preserve FP32 SDPA, literal original prompts/media,
empty assistant prefixes, greedy policy and the 3084 per-image cap. Each
worker has a 3600-second bound, with checkpoint grouping to limit repeated
loads. Bind every image result to its actual loaded adapter fingerprint;
mixed-checkpoint workers reload before switching groups. Save each complete
raw image output immediately, including length stops, parser drops and timing,
then assemble compatible per-checkpoint envelopes. No output filter or lower
cap may hide a repetition/stopping failure. The original terminated attempts
remain preserved. Root additionally assigned an independent CPU projection of
the actual native parent16 physical reviews onto the same v3 target identities.

Recovery launch is verified: tmux `coordexp-second-fit-readback-recovery-v1`,
actual controller PID 3609423. The final partition uses nine checkpoint loads
over eight workers; GPU2 switches from step16 to step64, with an explicit
reload and per-row checkpoint fingerprint checks. Root inspected that switch
and reran both focused recovery tests. The first six durable per-image rows
all ended naturally, taking about 20–27 seconds each at 13–14 tokens/second;
these partial observations do not yet explain the earlier whole-readback
timeouts or establish physical coverage. No new training was performed.

The accepted parent comparison is
`parent16-v3-physical-ledger-v2/root-acceptance.json`: 201 actual native step16
rows, 155 qualified owners out of 228, and 73 missing. Its old fixed220 subset
exactly reproduces the previously accepted 151 owners; four newly admitted
owners qualify. The rejected v1's 161 count was identity-only presence and
must not be used as coverage. Root also preserved the later glass-vessel
geometry and class override. Effective parent class counts are 173 verified,
17 unknown and 11 wrong, bound in `root-effective-class-decisions.json`;
that file supersedes the ledger's older canonical-only class counters.

Next wake is for this recovery GPU producer only. Inspect its status once,
then validate all 33 durable rows, nine actual model identities, native media/
prompt bindings, per-image token caps and stop reasons. Assemble/read
`readback-step-16.json`, `readback-step-32.json`, and `readback-step-64.json`
under the recovery root. Compare their v3 geometry selectors with the saved
parent selectors, then conduct the single-image bbox-and-crop physical review.
Keep raw repeated/capped output visible; a shorter filtered display is never
a shorter model output. The source checkpoints already passed training-state
acceptance, so another inference failure should be repaired at that boundary
without restarting training or declaring the learning recipe a scientific null.

Recovery GPU wake **armed**: `d6edef41-b093-42f6-8c4b-78ef43b3f3d4`,
key `tsc-second-fit-readback-recovery-v1-gpu-20260914`, expiry 4200 seconds.
It observes the recovery success/failure terminal or actual controller PID
3609423 exit. Receipt: `second-fit-readback-recovery-v1/wake-armed.json`.
Read this monitor's decision exactly once on delivery, then inspect artifacts.

## Recovery completed: second-fit natural stopping regression

Monitor `d6edef41-b093-42f6-8c4b-78ef43b3f3d4` delivered and root read its
decision exactly once (`second-fit-readback-recovery-v1/wake-decision.json`).
The recovery completed in 1154.616 seconds. Root independently validated all
33 exact original prompts/media, saved-adapter identities, raw token decodes,
parser results and envelope projections, with nine FP32 SDPA model loads,
33 image forwards and 58929 model forwards. Acceptance is bound by
`second-fit-readback-recovery-v1/root-acceptance.json`; neither monitor remains
armed for future work.

Observed native stopping: second-fit step16 has 7 natural EOS and 4 caps
(14194 tokens); step32 has 5 EOS and 6 caps (19619 tokens); step64 has 3 EOS
and 8 caps (25116 tokens). Raw parser drops occur even in some non-capped
outputs. No filtered/deduplicated output receives credit. This is a real
stopping/validity regression on the declared training task, not a validation
veto or evidence that owner improvements are impossible. Physical coverage
and retained/new owners still need per-image bbox/crop review.

Image210457 supplies a bounded stopping diagnostic. Its second-fit step32
and step64 natural token streams exactly equal the previously reviewed clean
48-token five-owner route including EOS. At step16 it emits a five-owner
prefix with two different frisbee coordinate tokens, then continues to cap.
Freeze four cells: second-fit checkpoints16/32 crossed with that actual
step16 five-row prefix and the clean five-row prefix without EOS. Measure
the raw full-vocabulary next-token logits, EOS/row-open ranks and margin,
and one naturally generated next token for each cell. Preserve exact image,
prompt, prefix token differences and model bindings. At most two GPU loads,
600 seconds per worker, no long suffix or further KV/attention variants.
A within-checkpoint history effect could motivate stop learning across
equivalent complete histories; it does not isolate an independent KV-cache
cause. This diagnostic cannot veto the main training-repair loop.

## Second-fit physical acceptance and reference corrections

`second-fit-review-extraction-v1/root-acceptance.json` independently binds
1178 raw rows, 1151 unique source/evidence files, all 11 owner partitions and
the unchanged original-image empty-prefix step16 readbacks. The physical
counts are 159 qualified owners, 69 missing, 125 repeated rows, 175 confirmed
false rows, 79 unresolved physical rows and 612 parser-invalid rows. Every
raw row survives; invalid spans do not automatically become physical FP.
There are nine newly covered fixed owners and three lost parent owners.

Root inspected the adjacent green-coat GT1323904, brown-jacket L08 and skirted
L09 people in image25274 at enlarged original resolution. Two existing raw
boxes straddle neighbors or clip the relevant owner. Apply the geometry ruling
symmetrically to parent p9/p10 and second-fit16 p8/p9. The previously accepted
parent155 becomes153; the candidate remains159 after the corresponding two
candidate exclusions. Historical receipts are preserved. The next catalog
explicitly assigns L08/L09 full references for the two distinct visible people;
all 228 prior owner IDs remain, with no merge or deletion. Evidence and patches:
`second-fit-root-rulings-v1/rulings.json`.

Root additionally admitted a white-coat person (25274), pig-chef figurine
(59571), blue glazed vessel (219546; class remains unknown), and an audience
chair (477415). The figurine's narrow generated box does not qualify as full
geometry; a newly plotted full reference includes its attached placard/base.
The new `target-owners-complete-v4.json` has SHA256
`b5c6534259a46583466dc5b92ae0b4b7854cc5adc4de8bc84801f3d96bc69f64` and
232 owners. `second-fit-root-rulings-v1/admissions.json` preserves the actual
proposal, crop and full-reference evidence. Other candidates remain unresolved.
The actual parent already covers the admitted white-coat person and chair;
root checked those exact crops. Its v4 baseline is155/232, with77 missing,
bound by `parent-v4-additions.json`. This version transition does not rewrite
prior fixed228 comparisons.

## Stop diagnosis completed

`second-fit-210457-stop-history-v1/root-acceptance.json` recomputes all four
full-vocabulary arrays and exact one-token native generation agreement.
Checkpoint16 chooses row-open for both complete histories: EOS-minus-row-open
margins -1.516781 and -1.521830, EOS rank2. Checkpoint32 chooses EOS for both:
margins +0.708155 and +0.700100, EOS rank1. Two model loads, four replay forwards
and four native one-token forwards completed in15.70 seconds. The launch's
initial missing CUDA visibility caused zero model loads and is preserved in
its separate failure receipt; retry2 completed. The observed stopping switch
is checkpoint-associated in this narrow contrast. Neither an independent KV
cause nor a full-completion intervention effect follows from this diagnostic.

`second-fit-stop-patterns-v1/analyze.py --validate` also replayed successfully:
all18 capped outputs have exact repeated-row runs or eventual token loops;
all15 naturally ended outputs lack an exact repeated raw span. The latter
statement does not exclude different-box physical repeats. The nine appended
GT geometries appear at IoU50 in2/6/9 images at16/32/64; physical verification
of their selected step64 rows is a separate nine-owner table, never full-step64
coverage or preservation evidence. The old bank has only one positive EOS
among1556 active CE targets, on one of11 images. That is an observation, not
identification of the cause of collapse.

## Third iteration: complete corrected teachers, frozen bounded recipe

Question: can a complete verified owner sequence with an explicit correct
terminal target compile broader natural completion while repairing the
partial-history recipe's omissions and raw continuation degeneration?
Compare the candidate to the actual first-fit16 parent on the same11 images
and the fixed v4 population232. This tests the whole corrected-teacher recipe;
it is not an EOS-only causal experiment. The strongest alternative is that CE
only memorizes this serialization or shifts EOS toward premature stopping,
with old-owner loss or geometry/repetition debt surviving natural readback.

Construct exactly one teacher row per fixed owner. Retain the parent's first
observed owner order where available, then append missing owners deterministically.
Replace fields with bound reasonable full reference geometry and independently
verified categories. When a category remains unresolved, preserve a bound
observed literal with direct description CE masked. Supervise object structure
and coordinates, and one final EOS after the full fixed owner set on every
image. False rows, repeated owners and unresolved physical proposals do not
enter this newly constructed teacher. The resulting sequence is explicitly
synthetic/corrected: no old suffix probability or natural-route certificate
survives the edits. It must pass fresh tokenization, native parsing, real image/
prompt binding, complete-owner and supervision-position checks before use.
A teacher's correct terminal label does not require the parent already to
choose EOS greedily. Self-rollout remains the discovery and subsequent refresh
route; successful natural readback remains the sole stage criterion.

Start from first-fit16 with an explicitly fresh AdamW optimizer. Retain the
588 language DoRA tensors, FP32 SDPA, learning rate1e-5 and raw-axis expected-
coordinate hinge weight0.01. Use the existing equal mean of per-image
active-token CE means; report EOS supervision under that normalization rather
than treating the raw count increase as its exact effective weight. Freeze
64 updates/704 image forwards, checkpoints16/32/64, hard training bound3600s.
No training job can exceed the user's five-hour bound. A new route/input hash
and source binding are required; do not mutate frozen older producers.

Read all33 checkpoint-image requests through eight workers with at most five
images each and nine actual model loads, preserving the accepted recovery's
per-image atomic outputs, checkpoint reload guards, original prompt/media,
empty assistant prefix, greedy policy and cap3084. Preserve all bad raw output.
Training and readback are separate terminal states. Inspect intermediate
natural consequences before further recipe changes; do not add images,
held-out evaluations, optimizer axes or another mechanism grid in this run.
The active stop rule remains a bounded chunk followed by repair of observed
failures; positive partial learning keeps the research open but cannot pass
an image or stage with remaining missing/repeat/malformed/geometry/stop debt.

### Third-fit concrete launch, 2026-09-15

`third-complete-bank-preparation-v2/root-acceptance.json` binds the accepted
232-row corrected bank: 2203 active CE tokens,928 active coordinate targets,
13 masked description fields,11 positive terminal EOS targets. Root replayed
all literal-description provenance, verified-class sources, token positions,
coordinate IDs, per-image owner sets and the native production parser; all232
rows parsed without malformed or dropped output. Ten focused bank/recovery
tests passed, including the actual candidate's ownership/geometry mutations
and root-unknown class precedence. V1 was superseded because five v3 admissions
already present in parent16 had not retained their native positions. V2 remaps
those five source rows and preserves all stage32-only additions as appended;
69 fixed owners are appended, while other geometrically incomplete owners
receive corrected references at existing positions. V1 and its producer
snapshot remain preserved; no GPU ever trained that candidate.

Under the unchanged equal-image mean of active-token CE means, the sum of
terminal-EOS target weights is0.00798603 for the complete bank, compared with
0.00189394 in the previous partial bank. The raw count11 versus1 is not the
exact normalization-weight ratio. This observation does not isolate EOS as
the causal treatment: owner completeness, geometry, histories and endpoint
supervision all change together.

Final manifest: `third-fit-preparation-v1/manifest.json`, file SHA256
`c3a7358c9325b89f69e3fb787f46f8c28b04b240240cc062a590b36bd33dcf40`,
content SHA256
`c1e2f0e117a5752f276a2cd1ee35de4de5eb221a4b45dfec7025d2838dc34127`.
It binds first-fit16 parent fingerprint
`885b07a07990f5ee171f8364007302ff2af7d5cbcceb3f33f9c665ba5de644b6`,
an explicitly fresh optimizer,64 updates, checkpoints16/32/64,704 planned
model forwards and a3600-second training wall bound. The preliminary CPU
consumer manifest used32 updates only to test serialization; it is not the
launched training contract.

The root-accepted `third-fit-v1/run.py` wrapper keeps the earlier recovery
source unchanged, supplies new artifact roots to every child, and binds both
source files. Root checked all adapter paths, all eight child command/CUDA
bindings and the33-job/nine-model-load partition, including the mixed16-to64
reload. The launcher verified no matching existing controller or training
output before starting `coordexp-third-fit-v1`, PID3786194. Launch receipt:
`third-fit-v1/launch.json`. A live controller is not model or quality success.

On the next GPU wake, read its decision once, then inspect the separate
training terminal/checkpoint/optimizer counters and the33 atomically persisted
readback rows under `third-fit-v1/readback-recovery/`. Check fresh actual model
identities, prompts/media, raw tokens/parser results, EOS/caps and all rows
before any physical claim. Compare against v4 parent155/232, then review
original+bbox+crop one image at a time. Keep missing owners and old losses
visible and preserve unknown/FP/class distinctions. No image expansion or
early validation panel is authorized by this launch.

The separate step64 targeted physical table is now accepted:
`second-fit-review-extraction-v1/root-step64-selected-owner-acceptance.json`.
All nine selected introduced GT owners are physically present with reasonable
geometry; eight appear before the first anomaly and all nine before sustained
collapse. Image528944's oven follows one isolated malformed row. Seven of
these nine complete outputs later cap. This confirms a limited positive
learning signal and does not establish full-step64 retention or completion.
The accepted full step16 ledger and flat decisions stayed byte-identical.

Third-fit live launch check confirms controller3786194 and owned training
process3786198, CUDA_VISIBLE_DEVICES=0, NVIDIA device descriptors and checkpoint
loading. NVML did not report a matching process ID at that observation and no
per-step result was available. Preserve this as initialization evidence only;
model-forward/training completion still needs its actual terminal artifacts.
The outer3660-second training timeout remains in force.

Third-fit GPU wake is armed: monitor4b0f9c80-2ce6-43c9-ab35-f68cc82a57fe,
key tsc-third-fit-v1-gpu-20260915, expiry7800seconds. It observes the actual
controller3786194 exit or completed/failed terminal, and stays quiet while
unchanged. Receipt: third-fit-v1/wake-armed.json. On delivery call status with
view=decision exactly once, inspect actual artifacts, and continue the frozen
training/readback acceptance and next repair. No native subagent is monitored.

### Third-fit completion and acceptance checkpoint, 2026-09-15

The third-fit wake decision was consumed exactly once and preserved in
`third-fit-v1/wake-decision.json`; the actual controller terminal reports
completed_unscored in1464.024 seconds. Training completed64 updates and704
logical image forwards in1160.609 seconds. Root replayed the training verifier:
all588 adapter tensors and588 AdamW states are finite at16/32/64, counters
match their steps, and fresh initialization binds first-fit16. Acceptance:
`third-fit-training-acceptance-v1/root-acceptance.json`. Objective values at
1/16/32/64 are1.823892/1.307717/1.063720/0.691585. Continued descent is neither
a convergence result nor evidence that native completion has been learned.

The current fixed v4 native parent ledger is
`parent16-v4-physical-ledger-v1/ledger.json`, SHA256
`f0536ed6a5a158f8bf296f1e736a5b0bdbee9d52349179f4b0ef1064bfcfce5e`.
It preserves all201 raw parent rows and applies the symmetric25274 geometry
rulings and two parent-present new admissions. Parent coverage is155/232.
The next full physical review targets third-fit32 on all11 training images
using this same denominator and original+bbox+crop evidence per image.

### Fourth-fit fixed-teacher continuation contract, 2026-09-15

Question: does continued optimization of the same complete corrected teacher,
with exact AdamW/RNG continuation, consolidate more fixed-v4 owners into native
greedy while eliminating repeated, malformed, geometry and stopping debt?
The contrast is third-fit64 versus continuation128/192/256 on the same11
images,232 owners, teacher order/literals/masks, image/prompt, CE normalization,
raw-axis hinge0.01, learning rate1e-5 and588 DoRA tensors. The strongest
alternative is exposure/serialization instability: teacher CE may improve
while the natural decoder enters different histories. This continuation tests
training dose and convergence, not an EOS-only or KV mechanism treatment.

Bind `fourth-fit-preparation-v1/manifest.json`, file SHA256
`16bca752c4b0b240b9580e20f876e9e4b3121c0afd5dd27a3ad5d74cb6cb71fa`.
Resume third-fit64 through total256:192 additional updates/2112 actual image
forwards, checkpoints128/192/256,4500-second training limit (4560 outer).
At the observed18.135 seconds/update, the segment is estimated at58minutes.
The existing runtime validator expresses its forward budget cumulatively as
2816; the segment receipt and controller require2112 actual forwards.

The separate `continue_training.py` wrapper preserves frozen training.py and
all previous manifest/source identities. Only the four declared runtime
fields change. Every other manifest value is compared exactly. Original
_restore loads all588 AdamW states, parameter layout and RNG against the
bound predecessor manifest; new checkpoints bind the new manifest. Root's
nine focused tests passed, including the actual64 optimizer/RNG restore,
cleanup on a simulated post-restore exception, masked-label/optimizer/source/
EOS mutations and checkpoint reload guards. CPU preparation is root-accepted;
actual GPU execution remains a separate acceptance question.

The wrapper reuses unchanged durable native readback code, remapping its
three checkpoints to128/192/256 with33unique requests,8workers,9model loads,
and max5images/worker. Preserve original-image empty-prefix greedy with
cap3084 and every raw row. Do not expand training images or use a held-out
panel. Stop this chunk at256 or its bound, accept actual execution separately,
then choose full single-image physical review and subsequent repair from the
observed native results. No early checkpoint is an impossibility veto.

Fourth-fit launch is active in tmux `coordexp-fourth-fit-v1`, controller
PID3875032. Receipt: `fourth-fit-v1/launch.json`. No matching prior invocation
or output existed at launch. Native physical review workers remain independent
and use native event waits. Arm the GPU-only wake monitor after useful review
integration is complete; inspect actual first-update/terminal evidence before
claiming execution success.

### User stop boundary, 2026-09-15

The user now explicitly requests: "训练和eval完这一轮,停下来跟我交流.无需继续推进."
This supersedes autonomous multi-round continuation. Finish the already-running
fourth fit and its current11-image native greedy evaluation, preserve local
per-view judgments and confirmed-unlabeled annotation exports, then stop and
discuss the results. Do not launch a fifth fit, new recipe, sample growth,
held-out panel or new mechanism/GPU investigation. Complete final256 physical
review and the immediate-parent third-fit64 comparison on the frozen232
population;128/192 readbacks provide dose diagnostics. Any additional valid
unlabeled discovered during this already-authorized evaluation must be saved
as annotations, without changing in-flight inputs or retroactive denominators.

User also requires every visual interpretation to be reusable on disk and
confirmed valid-unlabeled owners exported into annotation JSONL. Current
annotations-with-unlabeled-v3 contains11original records, unchanged169 GT
objects, and77 accepted unlabeled owners (63 previous plus14 new;18 class
unknown,59 verified). Every owner has recorded judgment and original+bbox+
crop provenance. Seven reviewed unresolved candidates remain separately
masked in third-fit-root-rulings-v1/reviewed-pending.jsonl. The complete327
third-fit32 raw decisions and per-view outcomes are persisted under
third-fit-review-extraction-v1. Root physical coverage is159/232, parent155:
13 newly covered fixed owners,9 old owners lost,69 repeat rows,11 false,
60 unknown,5 invalid and zero caps. No cumulative stage pass is claimed.
