# Single-prefix versus nearby-prefix geometry ranking

## Decision and authority

The user's autonomous research grant remains active. A new GPT-6.1-Sol/high
worker executes this finite unit; the lead owns its meaning, release and
acceptance. The question returns to the user's train/decode gap: does exposure
to nearby histories improve transfer of an actual learned ranking correction,
beyond fitting the originally observed decision?

[Round01](../2026-10-03-greedy-prefix-branching/results.md) found high structural
valid mass alongside an illegal greedy winner. This does not prove that mass
training is inferior. [Round03](../2026-10-03-completed-row-crossover/lead-ruling-02.md)
shows that72 is unnecessary for one fresh conditional exchange, but neither the
forced coordinate nor its large row is a validated positive learning target.
This unit compares prefix exposure under one ranking objective, not ranking
versus mass optimization, and does not train the72 branch.

## Fixed input and objective

Both arms independently start from the same endpoint16 original-LR balanced
constant checkpoint used in rounds01–03, weight identity
`07ea98e90220a9126042a27b3a76f60e9955320fed4bf77c2a8803ac9d52001a`.
Norm stays OFF. Reuse the original18 image/prompt/media inputs and evaluator-only
18/570 annotation snapshot. The checkpoint is already fitted on this diagnostic
cohort; natural readout here is not held-out population generalization.

Use the two prediction-selected historical structural-error sites7511-626 and
351017-1507. Their native baseline emissions may now be legal; keep the sites
without replacement. No GT selects a prefix, support set, perturbation or action.
At each site's x1 decision, allowed tokens are exactly coordinate bins0..998,
which admit a strict-valid box completion. The complement is the full remaining
vocabulary, including non-coordinate tokens. This is an existential structural
condition, not evidence of a correct physical owner or later valid completion.

For logits z and allowed set L, use the maintained Gmax computation:

`softplus(1 + max(z[not L]) - max(z[L]))`.

This is the standalone equal mean of six unweighted Gmax terms, not the prior
mixed objective or its0.1-weighted geometry component. There is no semantic/GT
target, replay CE, remaining-owner target or online target refresh. Both arms use
the same loss, reductions, parameter surfaces and optimizer.

## Frozen neighborhood

Find complete strict-valid earlier rows nearest first, with coordinate priority
y2,x2,y1,x1. Select the first coordinate whose plus/minus2 bins remain in0..999
and preserve that row's strict validity. Change just that saved token; retain
all subsequent history and the current header exactly. Never decode/re-encode.
The previously inspected tokens imply this table; CPU preparation must rederive
it from accepted raw identities and stop on disagreement, not choose a replacement.

| Site | Earlier valid row | Changed generated position | Original | Training values | Held-out values |
|---|---|---|---|---|---|
| 7511-626 | [982,601,999,638] | 620,y2 | 638 | 637,638,639 | 636,640 |
| 351017-1507 | [966,915,999,999] | 1499,y1 | 915 | 914,915,916 | 913,917 |

These are structurally valid neighboring histories, not certified same-owner
perturbations. The target x1 support is unchanged. Freeze all ten contexts
(two sites times offsets0,-1,+1,-2,+2), exact token/media identities and full
native processed inputs before training.

## Matched learning arms

- **R-single:** each original prefix repeated three times per update.
- **R-multiple:** original,-1,+1 prefixes once each per update.

Use site order7511-626 then351017-1507 and the listed view order, six equally
weighted losses per update, one mean reduction, one global clip and one optimizer
step. Each arm takes exactly16 updates; no interim native evaluations, adaptive
extension or best-checkpoint selection. Equal updates/compute do not guarantee
equal functional repair; retain measured original-margin differences.

Reuse the established FP32 language DoRA and special-token delta training
surfaces over the BF16 FA2 base:588 DoRA tensors(rank16,alpha32,dropout0), plus
input/output delta tensors of shape[1004,2048]. Fresh AdamW state per arm:
DoRA LR1e-5, delta LR5e-6,betas(.9,.999),epsilon1e-8,weight decay0,clip1,seed92711,
constant LR, zero attention dropout and activation checkpointing OFF. Verify
these bindings against maintained composition/optimizer
helpers and checkpoint metadata before release; a material mismatch returns
to the lead. Use serial model phases and independent anchor reloads, never carry
the first arm's weights or optimizer into the second.

## Fixed observations

At the unchanged anchor and both fixed endpoints, acquire ten native full-vocab
one-token scores/emissions for the frozen contexts, followed by ordinary
empty-history greedy outputs for all18 inputs in original order. One observation
per context/image/checkpoint: no repeatability or global determinism claim.
The fresh anchor is the comparator; old generated trajectories are not a substitute.

Report native emitted-token legality, full-vocabulary legal mass and best-legal
minus best-complement margin. Ties stay explicit; nonnegative rounded margin
does not guarantee legal emission. Measure corresponding HF margins for all ten
contexts at the anchor and endpoints, and training loss/gradient/update telemetry.
Native/HF disagreement is an execution gap, not deployed correction success.
Exact differentiable replay excludes the target token: the last returned logit
predicts that token, with rebuilt position IDs and cache disabled.

Primary conditional denominators distinguish original versus training-neighbor
versus held-out-neighbor contexts, baseline-illegal repair versus baseline-legal
retention, and each site. Do not call an already legal baseline a repaired error.
If no baseline original is illegal, state that the intended original-error repair
contrast is absent rather than changing sites. Numeric margin improvement remains
a separate measured outcome.

For the54 natural outputs report per-image and aggregate category-correct and
geometry-only known-owner gains/losses/retention, invalidity, exact/near repeats,
malformed output, length and stopping. Keep full owner identities and both arms'
tradeoffs. Unknown/unmatched stays neutral; no annotation-unmatched row becomes
a physical negative. Conditional supplied histories earn no natural recovery
credit; empty-history outputs are separate observations.

## Interpretation and stop

If both arms repair the baseline-illegal original native decisions and R-multiple
repairs more held-out contexts, broader exposure improves conditional transfer at
this fixed dose. Greater original-margin change remains an alternative to a
specific robustness mechanism. If only one arm repairs originals, compare repair
effectiveness without claiming a matched robustness experiment. If neither does,
distinguish under-repair from HF/native disagreement; the intended post-repair
robustness comparison is not established.

Held-out conditional gains without natural gains leave natural prefix access or
other decisions unresolved. Improved natural geometry with owner losses is a
structural benefit with a preservation cost. No outcome alone proves physical-FN
recovery, semantic prefix invariance, population efficacy or ranking's superiority
over mass training. Stop after the fixed dose and readout; no threshold relaxation,
new sites, selected extra checkpoints, retry of successful requests or recipe tuning.

## Execution package

Reuse `iterative_positive.compose/save_checkpoint`,
`online_row_credit.max_geometry_margin/verify_start_export/native_batch/vllm_requests`,
`src.qwen.native.exact_history_inputs`, `VllmDoraRollout.generate_exact` for
scores and ordinary `generate` for natural output, and the maintained evaluator.
Do not invoke the old eight-rank `online.start/run` or old-recipe `witness_inputs`.
Add only the unit's small runner and focused
checks; do not alter earlier frozen runners or shared runtime by default. CPU
preparation must prove the literal splice, unchanged support and input identity,
equal weighting/optimizer counts, independent reset, evaluator-only GT boundary,
and final-consumer rejection of false credit/counters/source or swapped arms.

CPU preparation authorizes no model/GPU calls. The lead then selects and
clean-qualifies the execution source and releases exact commands. The first
scheduled native requests and first real update are the production seams, not
extra smoke calls. Source/input/checkpoint/budget or non-finite failures stop the
package with partial evidence. Scientific non-repair does not cancel later fixed
observations or authorize extension.

Finite work:1GPU/rank at a time,1 native sequence,32 optimizer steps,192 training
replays plus at most30 HF diagnostic forwards;30 native score requests and54
natural continuations, each natural output capped at3084 tokens, at most166566
new native tokens total. Context4456 binds the maximum processed prompt1372
(image1584) plus3084; CPU preparation must verify all18 prompts and the ten
contexts. Reuse2GiB KV if qualified. Each of three
native phases and two training phases has1800s plus30s cleanup, at most9000s
active package time. Measure wall/RSS/CUDA allocation and artifact bytes; retain
explicit unmeasured resource fields. No parallel engine duplication or hidden
qualification/warmup calls outside these counters.

Canonical preparation/transport outputs:
`outputs/research/physical-fn-recovery/2026-10-03/prefix-exposure-ranking-04/`;
scratch messages: `.local/scratch/prefix-exposure-ranking-04/`.
Native/training outputs belong locally to the selected clean execution checkout.
The worker owns one new probe/test and this unit's state/results. The lead owns
protocol/index/catalog, release, interpretation and next-unit selection.
