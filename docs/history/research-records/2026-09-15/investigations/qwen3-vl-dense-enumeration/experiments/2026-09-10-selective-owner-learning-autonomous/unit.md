---
title: Autonomous selective owner learning with preserved useful continuations
description: Outcome-led Source-started learning, beginning with entrance CE and soft reference-function preservation.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-10-selective-owner-learning-autonomous
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-10
---

# Current authorization and question

**Closed on 2026-09-10:** the lead accepts `positive7-support50-81` as a
bounded, annotation-relative IoU50 development candidate after six completed
learning arms and full384 cold verification. This is not a held-out,
strict-IoU, universal preservation or deployment promotion. The first five
candidates are not promoted. No seventh arm, confirmation512 read, global
configuration change, Git publication or cleanup is authorized by this closed
unit. See [the final result](results.md#stable50-closeout-a-bounded-iou50-positive-with-explicit-debts)
and [the handoff](handoff.md). The historical stage packets below preserve
their original launch terms; they do not reopen completed work.

On 2026-09-10 the user explicitly removed candidate-count, data-growth, GPU
and elapsed-time ceilings, encouraged eight-GPU and long training when useful,
and delegated scientific acceptance and convergence judgment to the lead.
The earlier proposed three-candidate/two-GPU-hour limits are SUPERSEDED, not
active stop conditions. Start from existing scale and expand when promising.
The goal is useful natural-greedy detection, not proving a preferred mechanism.

From original Source step2444, can an update recover useful owners while
preserving overall natural-output quality, rather than only fitting two entry
logits? Entry learnability and natural recovery already have positive evidence
in the closed [point-CE unit](../2026-09-10-native-entrance-ce-feasibility/results.md).
Its extensive person repetition on the boat image is not proof that the trained
boat was forgotten. Supervision support is not parameter-influence locality.

Root owns scientific scope, iteration packets, acceptance, and closure. Prior
artifacts and unrelated dirty work remain immutable. No Git publication,
Notion update, credential/configuration change, or GT relabeling is authorized.
Model-family/architecture, ordering/native-greedy deployment changes and use of
confirmation512 remain major decisions for the user. Data/objective/support/
dose expansion inside selective learning is delegated to the lead, with an
explicit new iteration packet before execution. Unlimited resources do not
remove finite per-invocation bounds, mechanical checks or meaningful stops.

# First frozen iteration: soft-preservation-10

## Contrast and alternative

Compare original Source, retained point-CE23, and one fresh Source-started
candidate trained with the same two entry labels plus soft preservation.
This tests bounded selective feasibility. A reduced effective update dose is
a retained alternative explanation; no extra dose-matching arm is required
before measuring actual outcome benefit. Source probabilities are a reference,
not new truth. This is not an owner-ledger or representation-identification test.

## Inputs, model and objective

- Reuse exactly image368/person2022537 (x1=455, action index97) and
  image7116/boat181378 (x1=291, action index22), including their prior moderate-
  confidence annotation-backed admission and occlusion/extent qualifications.
- Original Source step2444, existing selected embeddings, FP32/SDPA and patch
  linearization; the same588 language DoRA A/B/m tensors (18006016 scalars).
  Base, vision, projector, embeddings and output head stay frozen and unmerged.
- Original entry/token/model/media identities come from the closed point-CE
  training inputs. Reference trajectories are the exact `arm=A` rows in
  `2026-09-10-owner-row-continuation-robustness/execution/rows.jsonl`.
- For each image replay the complete healthy Source P+A+suffix trajectory.
  Compute target CE at only the original x1 index. Preserve full-vocabulary
  Source distributions at action target positions BEFORE that index and
  AFTER the full supplied A row, including natural EOS. Exclude the corrected
  x1 and the remaining supplied A coordinates/boundary from preservation.
  The shared row header before x1 remains eligible. Verify exact partition,
  prefix compatibility, EOS inclusion and causal target alignment.
- Image loss is `CE(entry) + 10 * mean_state KL(Source || candidate)`;
  average the two image losses equally. Reference distributions are computed
  once before any update and detached; no gradient reaches the reference.
  No hard all-token replay constraint, new positive suffix labels, unmatched
  penalty, reward or GT is introduced. Soft preservation can affect unknown
  outputs without declaring them false or true.
- Fresh AdamW lr1e-5, betas(.9,.999), eps1e-8, weight_decay0, foreachFalse;
  joint gradient clipping1.0; deterministic eval-mode forward with autograd.
- Exactly23 updates, no margin-based extension or early scientific selection.
  Save one terminal candidate regardless of whether fixed-state margins cross.
  Nonfinite gradients or checkpoint/source identity failure stop the run.

## Mechanical resource envelope

Training owns GPU0 for one model load. Two reference full-trajectory forwards,
46 differentiable forwards and at most48 entry score forwards; at most96 total
model forwards. Each full action trajectory has139 or46 tokens. Initial true
KL must be approximately zero; after learning it must be finite with a genuine
detached-reference gradient. Cache reference distributions once, measure bytes,
RSS and peak GPU allocation. A3600-second run alarm is a recoverable invocation
ceiling, NOT a user-imposed total research-budget limit. The first real update
is the production-shaped mask/gradient/saving-path smoke. Preserve exit status.

Independent evaluation owns GPU1 after the sealed checkpoint receipt arrives.
Use one cold load, two entry-score parity reads,18 complete natural generations,
RP1/T0/top_p1 and cap3084 (at most55512 new tokens). Reuse the preexisting
train2/guard16 evaluation manifest and Source controls by exact identity; do
not claim they are new or untouched evaluation data. A3600-second invocation
alarm is operational only. Verify every output with the native consumer.

# Natural-output judgment and next iteration

Report train2 and guard16 separately: target recovery, annotated owner sets and
paired gained/lost owners at IoU50/60/80, F1, valid predictions, annotation-
relative unmatched counts, strict repeats, malformed drops, tokens and EOS/caps.
No KL/loss/margin substitutes for those outcomes; exact teacher-route reproduction
is not acceptance. Unknown unmatched predictions are not automatically false.

For the first screen, call the candidate promising only with useful new owner
recovery and joint F1/burden improvement over the defective point-CE23, without
hiding deterioration relative to original Source. A clean selective win requires
both targets, preserved or improved Source aggregate coverage/F1, and no new
repetition or runaway burden on the tiny training panel, plus a favorable guard
tradeoff. Individual owner exchanges are allowed and must be visible. Marginal
tradeoffs remain qualified; they do not receive an automatic promotion label.

Guard16 is exposed development/selection data. A locked candidate may later get
one broader check on dev128 excluding guard16; selection must be frozen before
reading that result. Those remaining images are also historical development,
not pristine heldout. Further training support may expand under the user's
authority, but each changed training/evaluation boundary must be recorded.

After each coherent iteration, the lead chooses the cheapest decision-bearing
next action from actual natural outcomes. Do not repeat entrance feasibility,
retry unchanged recipes indefinitely, or auto-launch a preset portfolio. Stop
at a substantive positive or negative convergence, low expected information
from further within-scope changes, or a major user-owned decision. Failed runs
remain technical failures, not scientific nulls. Protect completed evidence.

# Locked broader read: dev112

The lead freshly replayed training/evaluation verification after the first
screen. Both targets recover naturally, train TP50/60 increases16 to18 without
owner loss or new repeats, and F1@50 increases.864865 to.923077. Guard16 retains
56 TP50 with2 gained/2 lost owners, FP146 to135, F1.384880 to.400000,
repeats44 to43 and parser drops7 to8. This is promising bounded selectivity,
not proof of generalized improvement or a preservation-specific mechanism.

Before any remaining-dev outcome is generated, lock the sole candidate:
training receipt SHA256
`99be264f3b19a7ab84722d4b36832490c0efa7b3f5225ff4941241903a6d0607`;
adapter tensor-file SHA256
`712815e80d1323e74deda1b0aa94a8aeb1a6041c79d7ad9068819931f4b33002`.
No further dose, lambda or checkpoint selection occurs in this read.

Evaluate all112 Source dev128 images excluding exactly the frozen guard16;
assert no training-image overlap and no omissions, replacements or outcome
filtering. Freeze the exact IDs, Source outputs, GT/media/prompt/config hashes
and deterministic four-shard assignment before loading. Use the same native
consumer and FP32/SDPA/patch-linearized T0/RP1/top-p1 empty-prefix generation,
cap3084. Four independent inference processes own GPUs1,3,4,5, each one load
and28 generations (86352 tokens per shard;345408 aggregate maximum).
Each process has a3600-second recoverable invocation alarm. No distributed
training, architecture change or confirmation512 access is involved.

This is one expansion of the locked candidate, not four candidates. Verify
saved model identity in every process; the existing two cold-entry checks
already bind the checkpoint, so do not repeat them per shard. Preserve terminal
status, per-shard model/image/token counts, RSS/GPU peaks, raw trajectories and
all112 exact native-consumer reductions. Before accepting merge, require each
expected ID exactly once and all shard receipts complete. Report paired owner
gains/losses, TP/FN/annotation-relative FP and F1 at50/60/80, repeats, parser
drops, tokens and caps for dev112; keep train2 and selected guard16 separate.
Combined dev128 is descriptive and includes the selected guard, not a pristine
test. If the broader result supports coverage/F1 without runaway burden,
consider selective feasibility positively converged rather than spending the
unlimited allowance automatically. If it fails, use the actual failure pattern
to choose a new support/objective packet; never retune silently on this read.

# Second frozen iteration: soft-preservation-wide31

The locked first candidate improves local targets but degrades dev112 F1@50
(.655314 to.624729), with TP+18, FP+123 and repeats+41. The broader panel is now
exposed development. A CPU decomposition shows the largest parser repair is
not the main source of added FP; broader preservation support is a meaningful
next intervention. No claim that KL10's isolated local success generalizes is
retained, and no additional training of that saved candidate is authorized.

## Contrast, reference admission and objective

Restart from original Source with the same23-step optimizer, two entry labels,
two healthy-branch references and original `mean2 CE + 10 * mean2 KL` objective.
Add `10 * mean31 KL`, where each added KL is a full-vocabulary Source-to-
candidate KL averaged over all action states of that image's original Source
natural trajectory. Equal image weighting is retained; do not concatenate
token positions into a length-weighted average. All real target positions,
including real EOS, are preserved. No new hard suffix/GT labels, sampled
actions, rewards or unmatched-negative labels are introduced. Source is a
reference function, not a claim that each Source prediction is correct.

Reference support comes from Source train256, excluding the two targets,
ranked by SHA256 of `selective-preservation-support32-v1:<numeric_image_id>`.
The initial32 are retained in `support32-preflight.json`. Exactly one is
explicitly ineligible: rank8 image417044 ends at the3084-token cap, with307
predictions and2 parser drops. Reproducing that unfinished trajectory is not
admitted as useful behavior preservation. Keep the other31, with **no backfill**;
all31 end naturally and have zero parser drops. They are disjoint from dev128.
This Source-only admission is now frozen before candidate generation, not
candidate-outcome filtering or hidden removal of a failed training case.

The31 add1568 action states, maximum155 per trajectory and1472 prompt-plus-
action tokens. Combined static detached FP32 full-vocabulary reference cache
is1064415240 bytes. Cache each reference once before the first update. Accumulate
all33 image losses sequentially, apply one global clip1.0, then one AdamW step.
Do not divide the original two losses again, divide KL by vocabulary size, or
weight images by their action length. All33 cached references must have
approximately zero initial KL; no reference receives gradients. The same588
language DoRA A/B/m tensors change; base/vision/projector/embeddings/head remain
byte-identical. No train-mode/dropout/gradient-checkpointing change is included.

The question is whether this package preserves broad useful behavior while
retaining natural owner gains. Added regularization mass and effective-dose
change are alternatives to a support-specific mechanism; causal separation
is not required for this outcome-led iteration. It is not a31-image increase
in supervised owner labels: there remain exactly two positive entry labels.

## Execution and acceptance

Training owns GPU0, one load, exactly23 updates and one terminal checkpoint,
with33 reference forwards,759 differentiable forwards and48 entry scores:
840 total forwards. The first update includes all33 trajectories and the
longest-sequence real gradient/memory smoke. Capture finite gradients, adapter
movement, frozen hashes, reference bytes and measured RSS/CUDA peaks. A3600-
second invocation alarm is operational, not a global research limit. On OOM,
nonfinite or identity failure stop and preserve the receipt; do not silently
drop support, alter precision, modify batches or extend/restart the arm.

Independent evaluation freezes the same130 images: train2, guard16 and dev112.
All are now exposed development panels and remain separately reduced. Reuse
their exact Source GT/media/prompt/output identities; do not add or replace IDs.
Compare Source, original KL10 and wide31 with the same native consumer and
T0/RP1/top-p1 cap3084 empty-prefix FP32/SDPA generation. Four isolated cold
processes on GPUs1,3,4,5 get33/33/32/32 images, deterministic manifest order
round-robin, one load each. Only the first shard performs the two terminal
entry-vector parity reads; every process validates saved adapter identity.
The per-shard token ceilings are101772/101772/98688/98688; aggregate400920.
Each process has a3600-second invocation alarm. Merge only complete exact-ID
shards and preserve all130 native raw outputs and independent reductions.

The lead judges train2 target recovery and Source-owner retention separately
from dev128 owner gains/losses, IoU50/60/80 F1, annotation-relative unmatched
counts, strict repeats, parsing drops, length and caps. No checkpoint is chosen
from losses or natural outcomes. Further iteration requires actual outcome
evidence and another packet; confirmation512 and architecture remain protected.

# Third frozen iteration: soft-preservation-dense48

Wide31 keeps both local targets but fails the combined development tradeoff:
Source TP50/FP50=614/489 versus wide31=611/545; repeats112 to163. Two tail images
contribute98/133 gross-positive FP increases and71/80 gross-positive repeat
increases. Their Source trajectories have525/685 tokens, while all31 reference
trajectories have at most155. The next contrast targets reference coverage,
not an automatic lambda grid or an unchanged-dose extension.

Keep the original two CE/10-KL terms,23 fresh-Source AdamW updates and all
numerics/parameters unchanged. Replace the support `10 * mean31 KL` by
`10 * mean48 KL`: retain all31 and add exactly the17 non-target Source-train
images with natural EOS, zero parser drops and at least20 valid predictions
that are not already among the31. IDs:
25274,49327,64010,90862,101636,114340,152252,158044,203986,422969,474979,
511251,532132,540107,548337,568311,575627.
This Source-only density/serialization admission is explicit; no candidate
outcome, annotation-relative FP or duplicate filter is applied. All48 are
disjoint from dev128. They contain4 strict repeats and245 annotation-relative
FP; preservation does not assert those outputs are truth. The preflight's
246 clean-EOS pool is inventory, **not** an authorized full-pool training arm.

The48 have5848 action states, maximum325 actions/1627 total prompt-plus-action
tokens. Full-state reference KL includes each actual EOS. Total static FP32
reference cache including the original two is3678125640 bytes across ranks.
Mean-image weighting keeps the additional KL mass10; it redistributes weight
over reference images and does not uniquely isolate sequence length as cause.
No longer-than325-token healthy Source reference is available in this admitted
pool, so this addresses a relative coverage gap, not exact dev-tail support.

## Eight-GPU execution with the existing DDP pattern

Use eight replicas on GPUs0–7, one cold Source load per rank, no model-parallel
or architectural change. Reuse the existing trainer's DDP pattern:
`broadcast_buffers=False`, verified identical initial tensors, complete
forward/backward pairs inside `no_sync()` except the last local trajectory,
and world-size compensation8 of already globally normalized losses before
DDP's gradient averaging. Clip the globally reduced gradient once, then all
ranks apply the same AdamW step. Never average a rank's images again or clip
rank-local gradients. Unequal local forward counts must not create unequal
collective order. Model eval mode and frozen selected embeddings stay intact.

Sort support48 by numeric image ID, assign round-robin six per rank. Train
image368 on rank0 and image7116 on rank1. Only rank0 performs both original
entry-score reads at step0 and after each step, outside DDP with an agreed
barrier phase. All other ranks perform no entry scores. Global counts are50
reference +1150 differentiable +48 entry forwards=1248, with23 optimizer steps
and46 supervised token occurrences. Per-rank forward ceilings are216,168 and
144 for each remaining rank. Reference caching is local and computed once
before learning. Retain per-rank reference identities/counts, forward/image/
collective/step counts, times, CUDA/RSS peaks, code and checkpoint identity.

Before launch, require a load-bearing deterministic distributed-loss fixture
that detects omitted world-size compensation, rank-local averaging and
pre-reduction clipping. The first **two actual updates** are the production-
shaped acceptance slice, not a separate selected checkpoint or training arm:
initial KL approximately zero; step1 full-vocabulary entry vectors match the
previous identical first update within1e-5; step2 exercises nonzero preservation
KL; all ranks have identical reduced gradients and post-step adapter/optimizer
state, finite gradients and unchanged frozen bytes. Persist the two-step
receipt before automatically continuing to fixed23 when it passes. A failed
slice stops the invocation and remains technical-invalid, not a research null.

One torchrun launcher owns all ranks and their run-specific logs. A3600-second
per-rank alarm plus bounded distributed timeout protects failure paths. No
silent rank/support drop, precision/topology fallback or automatic relaunch.
Rank0 saves only after all ranks' training checks agree; publish a sealed
receipt only with exact completed rank coverage and a successful launcher exit.
Eight replicas have independent memory; measure the longest real trajectory
rather than inferring fit from parameter size. No unrelated GPU process is
stopped to make room. The user's uncapped compute authorization covers this stage;
the final closeout above governs whether any further stage may launch.

## Natural read and stop for this coverage contrast

Freeze the same train2/guard16/dev112 manifest before outcomes. After the sealed
checkpoint, reuse the four-process130 read on GPUs1,3,4,5, one load per shard,
two cold entry-vector checks only on the first shard, unchanged native policy
and400920-token aggregate ceiling. Keep Source, KL10, wide31 and dense48 metrics
separate; require exact unique130 IDs and all complete receipts before merge.
There is no new heldout claim or confirmation512 access.

The lead judges local recovery and development owner gains/losses, F1 at50/60/80
and repetition/excess/parsing/length together. The two dev-tail examples are
descriptive, never excluded from acceptance. If this coverage addition merely
shifts damage or sacrifices recovery, close this coverage contrast and choose
the next question from evidence; do not automatically train48→246 or sweep KL.

## Dense48 technical stop and bounded numerical discriminator

The first eight-rank invocation stops after update1: both original Source
entry vectors match exactly, but the full-vector comparison to **wide31**
differs by1.08838e-4/3.71933e-5, exceeding the frozen1e-5 gate. All8 processes
settle failed, launcher exit1;104 forwards and no saved candidate are retained.
No quality evaluation occurs. This is technical-invalid, not a scientific null.

CPU evidence shows all33 shared reference caches are bit-identical and entry
top1/ranks unchanged. Real FP32 Source fixtures demonstrate exactly zero KL
with nonzero backward residuals (FP64 residuals are much smaller). Consequently,
the mathematical zero-KL argument did not justify assuming1e-5 equality across
different reference populations. This is a plausible explanation, not an
established cause through the model and Adam. The live cross-rank hash checks
passed but were not persisted before the failing score gate; do not invent
those snapshots. Output proximity alone cannot exclude gradient scaling errors.

Authorize one **mechanical serial first-update control**, not another evaluated
candidate: original Source, exact dense48 objective and saved50 reference
caches from the failed invocation, one GPU0 load, no DDP and no world-size
compensation. Validate every cache/input/model hash; each first differentiable
forward must verify approximately zero initial KL. Reuse rather than recompute
the already sealed Source caches. Two initial entry scores,50 differentiable
forwards and two post-update scores give54 model forwards, with no generation.
Keep23-step scientific arm stopped. The control has one step and a1200-second
invocation bound. Persist raw global gradient **before clipping**, clipped
gradient, Adam state and adapter tensors before any vector comparison, plus
full-vector/entry statistics against the retained failed DDP step1. All are
mechanical artifacts under `dense48-parity-serial`, never a promotion candidate.

Root decides the next numerical admission from this evidence. No threshold
change, DDP retry or natural evaluation is authorized by this control alone.

### Subsequent root ruling: one gradient-bound retry

The serial control completes54 forwards in89.54 seconds and is freshly
CPU-verified by root. All50 initial KL values are0; raw gradient L2 is
154.664641918 and the saved adapter movement is.0422637603675. The wide31
comparison discrepancy reproduces without DDP (max1.37568e-4/4.38690e-5).
The **same-objective** serial versus failed-DDP comparison is smaller but still
above1e-5 (max5.14984e-5/3.43323e-5), with unchanged entry top1/ranks. Therefore
the original cross-population1e-5 output gate is not a valid numerical-equivalence
requirement for this FP32 recipe. This does not itself prove correct DDP scaling.

Replace that gate with a more direct, frozen numerical contract for **one**
fresh Source retry, with identical dense48 objective/data/23 steps/8-rank
topology. At step1 compare the complete selected-parameter raw gradient and
clipped gradient to the serial-control tensors, using identical layout:
each relative L2 error must be at most1e-5, computed in FP64. This directly
tests the global objective's weighting before clipping/Adam can hide errors.
Persist gradients, optimizer/adapter state and parity measurements BEFORE
raising any comparison failure. All rank gradients/adapter/optimizer hashes
must still agree exactly and frozen bytes remain unchanged. Step1 full-vocab
entry logits compare to the **same dense48 serial control**, with maximum
absolute error at most1e-4 and unchanged entry top1/ranks; the old wide31
comparison is descriptive only. This output tolerance is frozen from the
mechanical observation, not from any natural-quality result. Step2 must still
exercise nonzero KL and cross-rank state equality before continuing to23.

Preserve the failed `training` directory, all original code bytes and the
mechanical control. Write the new attempt to `training-retry1`; use a separate
`evaluation-retry1` with the same130 image/control payload and no new selection.
Scoped edits to the dense trainer/evaluator and their tests are authorized
after archiving the pre-revision bytes; completed earlier-arm code is immutable.
The new sealed receipt must bind this numerical-admission revision, serial
control hashes, passed gradient/two-step receipts, all8 rank terminals and
launcher exit0. No checkpoint from the mechanical control is eligible.

Training then independent130 evaluation may proceed automatically through
these gates, with the same1248-forward training and400920-token evaluation
bounds. If the new direct-gradient gate fails, stop again; this ruling does
not authorize further tolerance increases or another retry. Scientific meaning,
candidate selection and confirmation512 boundaries are unchanged.

# Fourth frozen iteration: soft-preservation-dense48-strong100

Dense48's completed retry preserves both train targets but fails the coverage
contrast: combined dev128 TP613 versus Source614, FP617 versus489, repeats214
versus112 and F1.578029 versus.615848. The added17 reference histories do not
resolve joint preservation; do not automatically expand48 to246. This fourth
iteration tests the remaining weak-constraint explanation with one substantial
weight change, not a parameter scan or a new family of models.

Freeze exactly the same Source start,48 support images and5848 action states,
two positive entry labels, original `mean2 CE + 10 * mean2 KL`, optimizer and
23 updates. Change ONLY the additional support coefficient:
`10 * mean48 KL` becomes `100 * mean48 KL`. The static teacher distributions,
full-state/EOS masks and equal image weighting are unchanged. Preserve all
prior candidates; this is a fresh arm, not continued optimization of dense48.
Reduced effective learning or surrendered target recovery is a retained
alternative; smaller loss/movement alone cannot establish a selective win.

Training uses the now production-verified eight-rank DDP path with the same
six supports per rank and the two training entries on ranks0/1. Scale globally
normalized losses by8 before DDP averaging and clip only after reduction.
Use one load per rank,50 reference +1150 differentiable +48 entry forwards,
23 steps,3678125640 cached bytes and3600-second per-rank alarms. The first two
actual updates must again show zero initial KL, finite gradients, nonzero
step2 KL, exact cross-rank gradient/adapter/optimizer identity and unchanged
frozen bytes. Save evidence before checking numerical failures.

Do not repeat the invalid cross-objective first-update output gate: changing
the KL coefficient can change FP32 zero-KL backward residuals. The admitted
dense48 serial/raw-gradient comparison establishes the unchanged DDP execution
path, not equality between coefficients10 and100. A new load-bearing analytic
fixture must verify the additional support gradient is exactly10 times the
coefficient10 fixture while the original two terms are unchanged, and still
detect missing DDP compensation/rank-local averaging/pre-reduction clipping.
No extra serial model control or new tolerance experiment is authorized.
The new receipt must identify `ddp_path_reuse_support100_v1`, link the prior
accepted serial-gradient path proof and current coefficient fixture, and bind
the current two-step smoke, all8 complete ranks and launcher exit0. A failed
run stops without automatic retry. The stale unused support-use counter found
in dense48 may be corrected for this new arm; require actual48×23=1104 uses
and leave the older sealed receipt unchanged.

Use separate `soft-preservation-dense48-strong100/training` and `/evaluation`
roots. Freeze the same130 Source/control payload before outcomes; independently
evaluate using four cold replicas on GPUs1,3,4,5 after training exits, exactly
as in dense48. Preserve train2/guard16/dev112 separately and dev128 descriptively;
retain Source, KL10, wide31 and dense48 references. Same400920-token aggregate
ceiling, unchanged native decode/metric semantics and two cold entry checks.
No architecture, positive-label support or confirmation512 change is included.

Acceptance still requires useful owner recovery without concealing broad F1,
coverage or repeat tradeoffs. If stronger preservation only returns to Source
by abandoning recovery, or fails to remove/contain the broader damage, close
this single strength contrast. Do not automatically proceed100→1000 or alter
the dose. The next question, if useful, must be chosen from that outcome rather
than treating unlimited compute as a reason to continue an unchanged recipe.

# Frozen expansion read: unchanged strong100 on the remaining train254

Strong100 is the first joint development-positive candidate: dev128 TP50
614 to618, FP489 to443, F1.615848 to.633197 and repeats112 to96, retaining
both trained targets. It still exchanges21 gained/17 lost owners, loses3
TP80 and has a guard16 parser-drop regression7 to22. A bounded read-only
diagnosis checks the latter; it is not permission to repair the parser or
discard outputs. No new positive labels, training steps or weight scan follow
this result automatically.

The lead freezes one larger natural read before calling this result converged:
the exact original Source-train256 population minus the two already evaluated
positive images368/7116, with NO outcome, EOS, parser, category or density
filter. Divide the254 new outputs into the48 fixed preservation-reference
images and206 images outside the current update's positive/reference support.
The primary additional contrast is unchanged strong100 versus immutable Source
on those206; reference48 is a separately reported preservation diagnostic.
Combine existing train2 and dev128 only descriptively to form the384-image
union, without generating them again. This is the existing acquisition pool,
not a new held-out dataset; original Source and all earlier baseline access
remain disclosed. No confirmation512 data are read.

Bind the checkpoint to sealed receipt SHA256
`037f1af3561e5522837364c41266af135e20c3d80579767092c63b761d6821ba`.
Freeze all254 Source raw trajectories, GT/media/prompt/token identities and
group/shard membership BEFORE candidate generation. Reuse existing native
decode and scorer semantics: fp32/SDPA, no sampling, RP1, max3084, empty forced
prefix. All Source capped/dropped examples remain included. There are no new
entry score forwards, because the same saved candidate's two cold checks have
already passed. Use8 cold replicas, GPUs0–7, numeric-image-ID round-robin
shards32/32/32/32/32/32/31/31, exactly254 continuations, at most783336 new
tokens, one model load per rank and3600 seconds per rank. One owner retains
the launcher and logs; technical failure stops without automatic relaunch.
Use a task-local runner under
`soft-preservation-dense48-strong100/expanded-train254`; completed arm code and
artifacts are immutable. CPU tests must expose missing/duplicate images,
reference/outside-support overlap, forced context, changed candidate identity
and incomplete shard acceptance; raw consumer replay closes final acceptance.

Keep gained/retained/lost owners, TP/FP/FN/F1 at50/60/80, repeats, parser drops,
output tokens and caps for each stratum and the descriptive union. A meaningful
positive convergence needs the joint owner/F1/burden direction to persist
outside the fitted reference support, rather than only on the already-exposed
development set or through hidden invalid-output suppression. Report smaller
strict-IoU/owner-exchange tradeoffs instead of concealing them. If this read
reverses the joint result, close this proposed promotion and choose the next
question from the observed failure; no extra candidate or data sweep is
authorized by this packet. If it supports the result without a material new
failure, close at a bounded development-positive candidate, not a general or
publication-grade claim. The six prospective extra positives in the inventory
remain unadmitted and untrained.

# Frozen positive-supply entrance read: eight variants, no training yet

The expanded strong100 read fails the joint direction on outside206:
TP869 to863, FP817 to879, F1.569836 to.555699 and repeats463 to501. The
descriptive384 union gains only2 net owners while F1/repeats worsen. Close
strong100 promotion. Instead of automatically raising KL again or expanding
reference-only coverage, test whether existing sampled evidence can supply
additional useful entrance labels beyond the same two used by all four arms.
The strongest alternative is that the extra candidates are not accessible
through a single entrance action or damage existing successors. No training
benefit is inferred from their sampled completions.

Use the already-retained inventory SHA256
`7bddf95ea074085b951ca478584be4323e01a7305d8ea51e2a5c37b29ac7d817`
under `positive-support-inventory/inventory.json`. Freeze all8 variants of
exactly6 owners:252411:1676022 hair drier(seed2026090602),
465695:1612233 bed(seed2026090601),496747:1614837 dining table(seeds
2026090602/03/04),529411:2147505 teddy bear(seed2026090603),
538814:1121312 oven(seed2026090602),540567:1491504 bottle(seed2026090601).
The lead has directly viewed all six original images. The hair drier and
bottle are small visible objects; the plush animal is tiger-shaped but the
immutable annotation category is `teddy bear`, not a new semantic relabel.
The bed is heavily occluded by luggage with an almost full-frame annotation;
table and oven candidates may be localization repair rather than previously
unaddressed instances. Retain those distinctions rather than claiming a
uniform missing-instance mechanism. These observations admit a bounded
conditional probe, not positive training labels or new ground truth.

For each variant use original Source, its exact native action prefix before
the first divergent token, append only the one frozen target token, and
greedily generate the remainder to EOS/cap. Prefix lengths/target indices are
1 for hair/bed/plush/bottle,7 for each table variant and4 for oven. The target
tokens are respectively `hair`, `bed`, coord610/613/631, `ted`, coord263 and
`b`; subtokens are not unique instance addresses. Require exact prefix
equality with Source and the sampled row before loading; use the actual
greedy realized row/owner, not the sampled row's IoU, to judge the result.
Do not force complete sampled rows or add fallback continuations: the single-
token release is the cheapest sufficient evidence for the proposed label.

Use one original Source load on GPU0,8 continuations/8 image forwards,
fp32/SDPA/native greedy/RP1 and max total action length3084 including forced
prefix. The aggregate maximum new tokens is24635; allow at most24643 model
forwards including prefill overhead,3600 seconds for the single invocation.
No new training, natural checkpoint quality read, full-row arm, resampling or
automatic retry is authorized. Freeze the Source/GT/media/prompt/row-token
identities and all8 variants before outcomes. Reuse existing native generation
and canonical parser/global matching through a task-local runner under
`positive-branch-expansion`; preserve all previous artifacts. Persist forced
and free token boundaries explicitly: none of these are autonomous outputs.

Prospective positive eligibility requires that the actual first generated row
containing the supplied entry matches the intended owner atIoU50, all Source
IoU50 owners remain, FP/repeats/parser drops do not increase and completion
reaches EOS. Report60/80 and all owner exchanges/burden regardless of passage.
If multiple table variants pass, select the smallest retained seed, not the
largest favorable margin or IoU. Preserve all failed variants. A passing
control supplies a candidate label and a concrete complete Source trajectory
for future preservation; it does not establish autonomous retrieval or
trainability. Root must first verify/admit the candidates and freeze a new
training objective, mask and dose. If none passes, stop this supply branch
without substituting other sampled histories or scanning further candidates.

# Fifth frozen iteration: seven positive entrances with support47

Root freshly verifies all8 conditional outputs and8 CPU boundary/eligibility
tests. Five additional owners pass the frozen rule and are now **lead-admitted
as training entrances for this one arm**:252411:1676022,465695:1612233,
529411:2147505,538814:1121312 and540567:1491504, using the exact variants
named above. All3 table variants fail actual first-row target matching with
IoUs.391/.394/.375 and remain excluded; no fallback or substitute is used.
The5 actual first-row IoUs are.607201/.987554/.926241/.911548/.581158.
All Source owners at50/60/80 remain in their conditional outputs; five new
targets match at50, four at60 and three at80. Bottle additionally recovers
owner1488726 without direct CE. These are single-token conditional controls,
not autonomous retrieval or proof of trainability. The whole read costs one
Source load,8 image forwards/continuations,334 generated tokens and35.36s.
Bind its input SHA256
`710c21601156a6e04a3ad195d71f824203c0a2f7eb3cf080dd0b7da79180eb4f`
and reduction SHA256
`4bc95c3d736dab4d2614d087c67c24a36477f621ea12a14605fb180d6c712728`.

The fifth learning package adds these5 to original368/7116, giving exactly7
positive images/labels and7 categories. For original2 retain exactly their
previous healthy Source full-row-release trajectories and masks. For new5
use their actual single-token-release complete Source trajectories, NOT the
sampled witness coordinates. Supervise only the frozen entrance token; Source
KL preserves positions before that entrance and after the completed actual
owner row, including EOS. Exclude the entire interval from entrance through
that row's `box_end` from KL, while CE still uses the entrance position.

| Image | Entry index | Actual action length | Excluded KL interval | Preserved states |
|---|---:|---:|---|---:|
|368|97|139|[97,102)|134|
|7116|22|46|[22,27)|41|
|252411|1|31|[1,11)|21|
|465695|1|29|[1,9)|21|
|529411|1|21|[1,11)|11|
|538814|4|10|[4,9)|5|
|540567|1|49|[1,10)|40|

There are273 local KL states. Remove529411's old full-Source trajectory from
the separate preservation set: retaining that conflicting first-row `dog`
distribution while training a `ted` entrance is not authorized. The remaining
47 references are otherwise identical, unfiltered and unmodified;5838 full
action states, no backfill. Define the new objective exactly as
`mean7 CE + 10 * mean7 local_KL + 100 * mean47 support_KL`, where each KL
averages states and sums vocabulary before equally averaging images. This
keeps component masses, not each old image's coefficient, fixed. References
are detached static original-Source full-vocabulary distributions.

Start fresh from the same Source, same588 language DoRA A/B/m tensors,
frozen base/vision/projector/selected embedding/head, FP32/SDPA and AdamW
recipe. Train exactly81 updates, with no margin/argmax early stop and only
one terminal candidate. The dose is `ceil(23*7/2)`: roughly preserve historical
per-positive coefficient-time(81/7 versus23/2) under the larger mean, not
claiming Adam/clip equivalence. This is an outcome-led positive-pool/dose
package, NOT a pure diversity-at-fixed-dose causal contrast. Underfitting,
overfitting and broader first-category biases remain alternatives. Do not
automatically add dose checkpoints, continue beyond81 or scan coefficients.

Use the accepted8-rank DDP path with global-loss times8 before DDP averaging
and global clip after reduction. Numeric-ID-sorted7 positives go one each to
ranks0–6; sorted47 references go round-robin to ranks0–7(6 each on0–6,5
on7). Thus54 reference trajectories and54 train forwards per step, with7
items on0–6 and5 on7. Score the7 isolated entrances only at initial Source
and terminal update81 on rank0:14 extra forwards, no per-step score loops.
Bound exactly54 reference +4374 differentiable +14 isolated =4442 total
forwards,8 model loads,567 positive-token uses,3807 support-trajectory uses,
6111 cached states/3731865480 bytes. Per-rank forward counts are588 onrank0,
574 onranks1–6 and410 onrank7. Keep3600 seconds per rank and600-second
distributed operation timeout. Each rank has at least5 losses; uneven counts
must not introduce rank-local averaging. Persist frozen code/input identities.

Before launch use real-token mask fixtures for all7, with causal entrance
alignment/EOS inclusion/excluded-nonentry invariance; analytic CE/local/support
gradient weights must detect accidental reuse of the old half-weight helper,
state-count pooling, omitted8 compensation and pre-reduction clipping. The
actual first2 updates again require50-plus4=54 exactly-zero initial references,
finite gradients, nonzero step2 KL, exact8-rank gradient/adapter/Adam identity
and unchanged frozen bytes. Save rank identities and state before failures;
then automatically continue81 if passed. Reuse the accepted DDP path proof,
not invalid cross-pool first-update equality or another serial model control.
Seal only after all8 complete terminals and launcher exit0. Technical failure
stops without automatic retry. Producer owns new
`probes/dora_owner_learning/selective_preservation_seven.py`, its focused test
and `positive7-support47-81/training`; completed prior code is immutable.

Freeze ONE384-image natural read before outcomes using the already-bound
Source and strong100 controls. Keep four disjoint strata:positive7,
reference47,remaining train202 and exposed dev128. Primary broader evidence
is the latter202+128=330 images outside the new update's positive/reference
support, with each subset still reported separately; train256/union384 are
descriptive. No new cohort/GT/confirmation512 or hidden quality filter.
Use8 cold replicas on0–7 after training exits, numeric-ID round-robin48 each,
384 continuations,7 cold isolated target checks onshard0 only and at most
1184256 new tokens;3600 seconds per shard, same native T0/RP1/cap3084 with
empty forced prefix. Bind exact fixed81/seven positives/47 references/masks,
current mechanics proof and saved adapter before loading. Source/strong100
controls on the same384 are immutable, not regenerated. Consumer owns only
new `selective_preservation_seven_eval.py`, its focused test and this arm's
`evaluation` root. One owner/launcher per long run; no nested workers/retries.

Accept only through natural owner/F1/burden outcomes, not conditional passes,
training loss or fixed-state margins. Report all7 target matches, extra owner
gains/losses,50/60/80 quality, repeats/drops/tokens/caps and strict-IoU tradeoffs.
A broader benefit must not be solely an in-sample or invalid-output effect.
The result decides whether increased positive supply earns promotion or
closes this package; no sixth training arm is automatically authorized.

# Sixth frozen iteration: targeted Source coverage, support47 to50

Seven81 makes all7 trained targets natural and preserves their old owners,
but the complete384 read fails broader preservation. Outside330 TP1477 to1476,
FP1304 to1395, F1.588095 to.577352, repeats575 to674 and caps4 to7. Root
replays native384 reduction and all81 training identities. The323 outside
images uncapped under both models improve jointly(TP1459 to1466, FP787 to746,
F1.665450 to.673868, repeats124 to100, drops64 to22), but this is a post-
outcome diagnostic and cannot replace the complete panel. Three new capped
train images,73843/360573/545632, had clean-EOS Source trajectories of only
10/95/81 tokens. This motivates one targeted preservation-coverage addition
rather than more labels, more steps, a larger coefficient or a filtered score.

Freeze exactly the same positive7 cases, actual trajectories,273 local states,
entrance-only CE, fixed81 steps and all optimizer/model/decode semantics.
Add Source images73843,360573 and545632 to reference47, with their original
complete native Source action sequences INCLUDING EOS. No candidate-generated
tokens, alternative boxes, GT changes or extra positives are used. These3
are in original train256, outside positive7/reference47/dev128, and selected
adaptively from the exposed failure; disclose that selection. Their full
sequence lengths are1372/1415/1365. New reference50 has6024 action states,
all clean EOS. Keep the component mass fixed:
`mean7 CE + 10mean7 localKL + 100mean50 supportKL`. The old47 per-image support
weight is redistributed from100/47 to100/50. This is complete-trajectory
coverage, not an isolated EOS-only intervention. Source's four original
capped trajectories are NOT newly trained or removed from evaluation.

Start fresh Source, not continued Seven81. Eight-rank DDP follows the same
proven path with current two-step identity/frozen-byte checks and a new
objective/reference fixture; no serial control or cross-pool equality gate.
Sorted7 positives go one each to0–6; sorted50 references round-robin give7
to0–1 and6 to2–7. Local items are8/8/7/7/7/7/7/6. Exactly57 reference +4617
train +14 initial/terminal isolated score forwards =4688 overall;8 loads,
567 CE uses,4050 support uses,6297 cached states/3845451960 bytes. Rank
forwards are670/656/574/574/574/574/574/492. Keep3600s per rank/600s collective
timeout, one saved terminal candidate, no per-step score loop or early stop.
The current first2 actual updates must show57 initial KLs exactly0, finite
gradients, nonzero step2 KL, exact cross-rank gradient/adapter/Adam identities
and unchanged frozen bytes; save evidence before failures. The analytic
fixture must verify unchanged positive/local contributions, support mass100
with50 equal image means, actual3 Source EOS paths, correct DDP compensation
and post-reduction clipping. It must reject a retained47 denominator or a
missing/modified/non-Source new reference. No backfill or fourth case.

Producer owns only new `selective_preservation_stable.py`, its focused test,
and `positive7-support50-81/training`. Preserve all previous code and receipts.
Seal only after all8 complete rank terminals and launcher exit0. CPU preflight
and the actual two-step slice permit automatic continuation to81; technical
failure stops without automatic retry. No seventh candidate is authorized by
this packet.

Before outcomes freeze the exact SAME384 Source/strong100/Seven81 controls.
Use positive7/reference50/remaining train199/dev128 as disjoint strata;
primary outside327 is199+128, with both subsets also reported separately.
Keep previous outside330 and all384 comparisons descriptively available so
moving the3 failures into training cannot conceal their outcomes. Explicitly
report the3 targeted stop failures plus all original/new cap identities, not
only aggregate cap counts. Same8 GPUs after training release,48 images each,
384 native generations,7 cold isolated score checks only onshard0, at most
1184256 new tokens,3600s/shard. No forced prefix, policy change, output repair,
post-filtering or Source/control regeneration. Consumer owns only new
`selective_preservation_stable_eval.py`, its focused test and this arm's
`evaluation` root. Bind exact81/7/50/masks/6297states/4688forwards/current
proof/all completion before loading; no positive-margin gate.

Decision remains complete natural owner/F1/burden evidence. Targeted cases
returning to normal is insufficient if damage migrates elsewhere or the
broader owner/F1 direction remains negative. Keep repetition, invalid-output,
cap, strict-IoU and owner-exchange tradeoffs explicit; no claim of universal
preservation or held-out generalization is authorized. This one addition
tests containment of the observed failure, not an automatic adversarial-data
growth loop. Any further candidate requires a fresh outcome-led ruling.

# Deferred diagnostics and historical correction

The lead activates one parallel explanatory sidecar, not a prerequisite for
the primary learning contrast: exactly two crossed boat suffix continuations.
H_good is Source's retained `arm=A` prefix plus complete supplied A row;
H_new is the old point-CE23 natural prefix through its generated target-boat
row (prediction index4, bbox bins[298,460,371,556]), before the person-repeat
burst. Derive and freeze the exact complete token-row boundary before loading.
Compare old point-CE23 at H_good and Source at H_new. The two diagonal native
suffixes are already retained. Identical full prefix tokens are used within
each checkpoint contrast; comparisons across histories are not a pure state-
memory effect because the completed owners, geometry and workloads differ.
Only new suffix contributions/repetition are attributed to continuation.
Reuse original deterministic FP32/SDPA/RP1 native execution, cap3084 minus
prefix length, and the native consumer. At most2 model loads on GPU2,2 new
continuations,6168 new tokens and1800 seconds. No adaptive prefix/arm expansion.
These are Source versus PREVIOUS point-CE23, not the new soft-preservation arm.

July painting Step0 already compared center points and tight boundaries.
Step2 anti-copy training changed corrupted-mark use (selected val100 2x-box F1
.032 to.563; closer-mark .971 to.136; tight-mark F1 .748 to.693). These are a
different painted checkpoint and teacher-prefix conditional-localization study,
not current-Source autonomous addressing. No repeated point-prompt probe follows.

# Bounded DeepSeek trial

Use the existing `ds-flash-max` CLI profile and shared CODEX_HOME for a read-only
scientific/code-boundary task. Root independently evaluates it alongside the
Astra work already useful to this study. Record exact provider/model/effort,
start/end, useful findings, correction burden, result/error and session identity;
one task cannot establish a general intelligence or speed gap. The user funded
only10 RMB: on quota/auth/transport failure stop DeepSeek, preserve sanitized
evidence, and continue with GPT as explicitly authorized. No recharge, global
provider change, API-key disclosure or repeated error retry.

# Artifact root

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-selective-owner-learning-autonomous`

Each arm has an immutable input packet, checkpoints, raw outputs and one
acceptance receipt; this unit owns interpretation and the current next action.
