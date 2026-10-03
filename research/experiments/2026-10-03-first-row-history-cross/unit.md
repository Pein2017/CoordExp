# Early first-row history cross

## Question and authority

Does the severe early localization failure on image351017 depend on the three
coordinate-token differences between its native first person row and the same
annotated person's teacher row, and does that dependence differ between saved
A0 and A16? The user explicitly authorized an autonomous valuable next experiment
before sleeping. The lead selects this finite diagnostic after closing the
rule-stability A/B unit and absorbing/retiring the two named temporary worktrees.
It is a new unit, not an extension or relaunch of the completed16-update study.

Observation: in A the first person row is unchanged, the second category changes
from person to bottle after update1, and both A16/B16 end at1/49matches and the
3084-action cap. This first category change precedes repeated history. Teacher
ordering also asks for bottle next, but its preceding person box differs.
Hypothesis: learned category/order behavior transfers into a nearby generated
history where localization fails. Strongest alternative: conditional bottle
localization remains unlearned or damaged under both first-row histories.
A checkpoint-by-history interaction would distinguish these better than more
training or a geometry-only intervention. Later repetition may amplify either.

## Frozen contrast and identity

Exactly four singleton deterministic HF requests, serial by checkpoint:
A0/native, A0/GT, A16/native, A16/GT. Use saved checkpoints0 and16 from
`outputs/research/physical-fn-recovery/2026-10-03/rule-stability-iou90/native-primary-A-01/`.
These are prescribed anchor/endpoint choices, not a best-checkpoint search.
No B checkpoint, optimizer, weight update, sampling, category forcing or new image.
The previous execution source was `249b9b09762457d2f786ab2fbc0c7de677cb5d3f`;
the new implementation receives its own clean revision and input packet.

Input image351017 and all49labels come from the unchanged full18/570 label file
`research/experiments/2026-10-02-full-label-self-rollout-fit/inputs/full-labels.json`
(SHA2561cfdeb3bba14bb26034dd5dc4245a5c10ab240e1e780acdfc1edbdf6c32d4792).
Reuse its original manifest/prompt/grid/media binding. The selected person has
annotation ID-1276804344338180. The next x1,y1,ID-sorted target is bottle
ID-4947389372712316 with box[186,30,207,106]. These are annotation identities,
not newly verified physical-owner identities.

Native prefix (both saved versions), person[0,13,536,990]:
`[151646,8987,151647,151648,151670,151683,152206,152660,151649]`.
GT prefix, same person[0,24,517,999]:
`[151646,8987,151647,151648,151670,151694,152187,152669,151649]`.
Only zero-based positions5,6,7 differ. Verify against the actual tokenizer and
maintained full-label rendering, saved native actions and original input identity.

## Concrete computation and controls

Reuse `native_components(checkpoint)` without NativeEngine/AdamW, maintained
native request/batch preparation, `MedianPolicy` and `generate_continuations`.
Restore BF16 base, FP32 DoRA and independent input/output deltas as originally;
use eval(), inference_mode(), BF16 CUDA autocast, neutral greedy GenerationConfig,
actual EOS/PAD, empty extension and a73-action budget. Force the first9actions
sequentially with a small probe-local logits processor AFTER median normalization.
This keeps ordinary prompt prefill and cached one-token transitions. Do not insert
the whole9-action prefix into a new prefill, which changes the numerical path.
From step9 onward return incoming logits unchanged:64 genuinely free actions.
No extra forward, replay, diagnostic or warmup is allowed.

Before forcing, preserve each requested prefix token's unforced median logprob
and pre-force greedy choice; raw selected-action likelihoods remain separately
retained. Forcing must return fresh scores, never mutate the raw tensor. Label
post-force prefix likelihoods as forced selection, not ordinary policy likelihood.
Free-action likelihoods retain ordinary raw/median meaning.

For each native-history condition, every pre-force argmax for steps0..8 must
equal the saved action AND all73result IDs must exactly equal the corresponding
saved greedy first73IDs. Both saved continuations exceed73actions; expected
short-run stop is length. This pair's native condition is its real execution
qualification. If it fails, publish technical HOLD, preserve the mismatch, stop
before its dependent GT condition and remaining requests, and do not retry or
relax fidelity. A0/GT may already exist if A16/native later fails; label it partial.

## Evidence and decision

Report the four literal continuations and every completed free row, their boxes,
categories and action positions. Primary comparison: first free category and
localization of the intended next bottle owner, using explicit IoU>=.5 and exact
description as the previous annotation-relative detection proxy. Also report the
first generated bottle's box/IoU to that target, even when another row came first,
and all free rows' best same-category overlap/owner candidates. Report invalid
geometry, strict class-agnostic IoU>.9 duplicate events (including overlap with
the supplied first row), EOS/cap and incomplete boundary rows separately.
Do not count the forced first person as recovered coverage or treat the64-action
budget stop as natural repetition/termination debt comparable with3084.

If A16/GT localizes the intended bottle while A16/native does not, supplied-history
conditional ability is demonstrated; A0's same cross distinguishes an existing
sensitivity from a training-dependent change. Failure under both weakens the
small-prefix-mismatch explanation for this local target; it does not exclude
other prefixes, localization elsewhere, cross-image interference or later-history
amplification. Category or geometry shifts without target localization are partial
observations. No outcome establishes sustained native coverage, physical-FN
recovery, a causal training-source attribution, or a general method verdict.

## Work, implementation and stop

At most4requests ×(9forced+64free)=292actions, two checkpoint loads, one GPU,
zero optimizer/backward/replay, zero exports. Reuse saved immutable base evidence;
qualify selected checkpoint and input contents once at the execution boundary,
not whole-source/model trees per request. Expected cold runtime is minutes;
600seconds is an observation estimate, never a kill/retry grant. GPU stress is
expected: launch directly, react to concrete OOM/conflict. Preserve actual counts,
phase timings, peak RSS/CUDA memory, errors and terminal process exit/settlement.

Implementation owner: `probes/first_row_history.py`, focused CPU tests,
`probes/README.md` entry and this unit's `results.md`. Reuse existing serialization,
parser and geometry helpers; no generic runner, permanent monitor or shared API
change without a concrete blocker. Fake-compute end-to-end entry/artifact/consumer
check plus meaningful negative prefix/fidelity/likelihood checks precede release.
No research checkpoint/model/GPU load or real tiny-model forward during CPU phase.
The lead owns unit/state/index/catalog, interpretation and exact native release.

Outputs belong to canonical worktree
`outputs/research/physical-fn-recovery/2026-10-03/first-row-history-cross/`.
After CPU candidate, the lead binds clean source and inputs and releases one
native invocation. The package owner runs it once, checks terminal evidence,
settles the process and returns. Routine CPU implementation repair is authorized;
replacement native execution requires a new decision. Stop after four conditions
or declared technical HOLD/failure. No automatic training intervention or next unit.
