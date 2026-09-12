# Stable50: GT-free later-row negative learning

**Closed:** the fixed32 candidate and natural384 evaluation completed.
Technical execution is verified; scientific joint criteria fail (TP50−3,
parser drops+100 despite fewer strict repeats). Adapter not promoted; no more
model calls pending. See [results.md](results.md). The protocol below is the
unchanged historical contract, not a new launch instruction.

## Frozen question and authorization

From the accepted Stable50 composition, does adding prediction-only
`IoU>0.95` later-row coordinate-confidence unlikelihood, with selective
Stable50 behavior preservation, reduce native duplicate output without
reducing aggregate annotated-owner recovery or creating new stop failures?

The user authorized the lead to execute the recommended next experiment
without further confirmation. This unit owns **one32-update candidate** and
its natural exposed384 comparison against its unchanged Stable50 anchor.
No dose/weight sweep, new positive CE labels, architecture/KV/special tokens,
GT edits, default checkpoint promotion, confirmation512, or automatic extension.
The previous [conditional admission unit](../2026-09-11-recursive-owner-composition/results.md)
is closed; its oracle trajectories are not this experiment's training data.

## Anchor and controls

The anchor is `positive7-support50-81/training/adapter`, with its original
unchanged Source special-token embeddings and Qwen3-VL2B base. The machine
packet binds the exact adapter, embedding, tokenizer, input, media and code
hashes. This is **not** a fresh original-Source run or continuation of Rweak64.

All seven non-protected train256 images with any strict repeat are included:
158044,248167,274509,351017,417044,477415,502725. Their saved Stable50 outputs
contain495 qualifying later rows. Four are capped at3084 tokens; usable complete
rows within those streams remain eligible despite dropped spans elsewhere.
The deterministic zero-repeat control is9813, the smallest numeric clean-EOS
remaining199 image. Online8 assignment is numeric order, one image per rank.
No visual/GT matcher determines negative eligibility.

The static preservation set is the existing support50 plus positive7,
excluding the overlapping online image158044:56 unique disjoint images,
seven per rank. Every reference uses its **current Stable50 natural action
sequence**, not the earlier original-Source trajectory. These sequences have
6056 total tokens before masking. No dev image provides a training signal.

The unchanged anchor is sufficient control here: there is no continuing
positive/Rweak objective whose extra training needs a matched continuation
arm. KL is minimized at the anchor; initial same-trajectory KL and mask
checks are required. Prior [Rweak64 repair evidence](../2026-09-08-coco-owner-recovery-dedup/results.md)
already found lower repetition with this surrogate, without established owner
recovery. The present experiment tests a different accepted anchor and
preservation regime; the primitive is not claimed as new or previously untried.

## Fixed loss and refresh

Parse each detached actual student trajectory with the native parser. For
each geometry-valid **pixel** box, flag a later row once iff its IoU is
strictly greater than0.95 against any earlier valid row, including earlier
flagged rows. Categories, descriptions, GT and global annotation assignment
are absent from this predicate. Invalid/dropped rows are not negative labels;
they do not prevent processing valid complete rows around them.

For each flagged row j, use its four original action-token coordinate positions:

`q_j = exp(mean_4 log p_theta(sampled_coordinate | actual_prefix))`

`u_j = -log((1-q_j+1e-8)/(1+1e-8))`

`UL(image) = sum_j u_j / number_of_valid_rows`.

This is geometric-mean coordinate confidence, **not** joint box probability or
a penalty integrated over the high-IoU box region. Zero-valid/zero-eligible
images contribute differentiable zero and remain in the fixed image mean.
The source parser geometry and original token positions must agree; never
retokenize a generated trajectory and silently change its conditioning.

The total objective is fixed:

`mean_online8 [ UL + 10 * KL_online_nonduplicate ]`

`+ 100 * mean_reference56 [ KL_reference_nonduplicate ]`.

KL is full-vocabulary forward KL `reference || student`, averaged over selected
positions within each image. Its positions are all tokens of valid complete
**nonduplicate** rows plus native terminal EOS when present. Entire qualifying
duplicate rows and invalid/dropped spans are excluded. In particular, KL does
not directly protect the actions this loss is trying to suppress. No EOS is
fabricated for capped output. Preserving a nonduplicate unknown row is explicit
behavior preservation, not declaring it semantically correct or probability-
invariant neutrality. There are no positive CE labels or unmatched-FP penalties.

At optimizer steps0,8,16,24, generate a fresh natural greedy trajectory for
each online image, recompute detached eligibility/masks, and score the exact
new trajectory with a **frozen Stable50 teacher**. Reuse those exact actions
for the next eight updates. Thus acquisition is current-policy at each refresh,
with lagged replay between refreshes—not fresh sampling at every update.

## Execution and mechanical gate

Eight A10080GB GPUs, one shared language DoRA adapter through DDP. Each GPU
holds one student and one frozen teacher;16 model loads total. FP32, SDPA,
patch-embedding linearization, model eval mode with student autograd. Train
only the accepted588 language DoRA tensors/18,006,016 scalars; freeze base,
vision, projector, embeddings and LM head. No runtime architecture intervention.

Fresh AdamW: lr1e-5, betas(.9,.999), eps1e-8, weight_decay0, foreachFalse;
global gradient clip1. Online image coefficient is1/8; reference is100/56.
Multiply each local contribution by8 before DDP averaging. Accumulate under
`no_sync` except the final local item; clip only after synchronized reduction.
All ranks have one online and seven static-reference replays per update.

The first **two real updates count within32**. Before update3, verify causal
positions/masks, finite loss/gradients, unchanged frozen tensors, nonzero
update, identical reduced gradient/adapter/Adam state across ranks and initial
same-trajectory KL. This first slice includes real3084-token capped streams:
the prior short-reference trainer does not by itself prove that memory scale.
No cross-population first-step-logit equality gate is permitted.

Limits:32 updates;32 refresh continuations globally with total action cap3084
and no forced prefix, temperature0, repetition penalty1;256 training replays
and11 teacher replays per rank; one natural evaluation over384 images on
eight independent48-image shards. Train rank deadline3600s; evaluation shard
deadline3600s. Record actual model/image forwards, generation lengths, payload,
cache/RSS/CUDA memory, optimizer deltas and rank/launcher exits. Technical
failure closes no scientific question; do not silently reuse a partial run.

## Decision and stop

The primary fixed denominator is all exposed384 images. Report online8,
preservation56, remaining-train192 and dev128 separately, as descriptive
strata rather than independent confirmations. Natural raw output is scored
without NMS/dedup postprocessing. Preserve old/gained/lost annotated-owner
sets, TP/FP/FN/F1 at IoU50/60/80, repeats, invalids, parser drops, lengths,
EOS and exact cap identities. Verify the seven previously learned target
owners individually. GT is used for evaluation only, never negative selection.

A bounded joint-quality positive requires fewer strict repeats, no lower
aggregate TP50 or F1, all seven protected targets retained, no new cap image,
and no aggregate invalid/drop increase. Any individual old-owner losses remain
explicit debts even if gains offset them; this is not uniform preservation.
IoU80 and the loss/gain ledger remain visible rather than silently optimizing
only IoU50. No deployment/held-out generalization claim follows.

The strongest alternatives are coordinate jitter and blanket suppression.
Compute class-blind later-overlap counts at0.90 and0.80 as diagnostics, and
view a fixed bounded panel of changed online outputs plus decision-bearing
new failures. A0.951→0.949 shift is not evidence of semantic deduplication.
Loss reduction alone never passes. If primary joint criteria fail, report the
fixed-dose tradeoff/negative and stop; do not extend dose or add adaptive
preservation cases. If criteria pass but changed-row inspection shows mere
jitter, do not promote a true-deduplication claim.

Raw root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup`.
Root owns science, input packet, evaluation and acceptance. Sol-high owns the
new trainer; Luna-max owns the pure loss/mask helpers and counterexamples.
Existing dirty work is preserved.

## Launch packet and pre-model checks

The [original packet, now archived](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/technical-invalid-attempt-01/inputs.json)
has SHA-256 `b7c839bc80da55f3b424bbee6beaa74f68b8c01f8fc029f8ec0374e683d7cc4c`.
Real saved-token layouts on all64 training/reference images reproduce native
repeat counts and distinguish EOS from caps. Online KL has1280 initial states;
reference KL has6047 after excluding its one malformed row. All495 later
repeat rows are selected, and the zero-image remains in the mean.

The lead reran25 focused CPU tests. A real-tokenizer EOS omission in the
initial Luna-max helper was reproduced and fixed before freeze; fake-tokenizer
tests alone had missed it. The trainer's empty-KL rejection and premature
nonzero-gradient requirement were also fixed before launch. The UL calculation
uses one vectorized gather, avoiding hundreds of full-logit backward scatters
on long repetitive streams. No model call used the rejected versions.

The task-local pipeline invokes the existing new trainer and then the new
eight-shard native evaluator, with success/failure receipts for both. It does
not select another scientific arm. The first invocation, PID2638082, stopped
with a real SDPA OOM on the capped3084-token student replay before its first
completed global optimizer update. It is preserved under
`technical-invalid-attempt-01`, including the frozen inputs and code snapshots.
No parameter-update or model-quality result is accepted from that invocation.

First-invocation repair (now completed) retained all images, tokens, precision,
objective and32-step dose. The repair uses standard non-reentrant activation
checkpointing on language decoder blocks during gradient-enabled replay only;
model/module eval mode and generation remain unchanged. A short real-model
loss/gradient parity control will compare checkpoint off/on before any update,
then the capped first-two-step slice must pass. This is a memory/computation
schedule change, not a new learned architecture or a shortened trajectory.
All28 language blocks use non-reentrant recomputation only with gradients
enabled. Rank1 compares the same246-token online158044 trajectory without and
with checkpointing, using its already computed frozen-teacher reference.
Two extra student loss/backward forwards perform **no optimizer update**;
relative gradient L2 and maximum absolute gradient difference must each be
at most1e-5, with loss parity and unchanged parameters. These two forwards
raise only rank1's model/image bounds from12607/271 to12609/273. The other
ranks wait at the same control boundary before DDP construction. The32-update
scientific dose and all56 reference trajectories remain unchanged.
The
[continuation note](continuation.md) owns the current invocation pointer;
do not start a second producer.

The runtime-only retry completed as detached pipeline PID2646225. Its
[current packet](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-11-stable50-geometric-dedup/inputs.json)
SHA is `749fee8e60bee1a8a06e5deb6f018ac6270df2dbd0c2feac7b2eb667c9f83fc4`.
The lead compared all scientific fields against the archived packet: online
and reference populations, every action/mask, objective, optimizer and eval
records are exactly unchanged.28 focused CPU tests pass. A PEFT-wrapper owner
path was corrected before retry; checkpointing now binds the unique actual
named language module and has a PEFT-shaped hierarchy regression. No model
call used the intermediate prelaunch packet02a.
