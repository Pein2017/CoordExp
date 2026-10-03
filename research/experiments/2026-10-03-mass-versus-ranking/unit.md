# One-update legal mass versus greedy ranking

## Decision

At the frozen structural sites, does one aggregate-legal-mass update repair
native greedy emissions as effectively as the accepted one-update ranking
checkpoint, and what natural known-owner preservation and repetition costs
accompany each result?

The autonomous research grant remains active. A new GPT-6.1-Sol/high worker owns
execution; the lead owns meaning, release and acceptance. The accepted
[short-dose contrast](../2026-10-03-short-dose-ranking/lead-ruling-02.md) found
that one ranking update already repaired6/6 illegal contexts and retained4/4
legal contexts, while exchanging15/22 category-correct owners. Later updates
added costs without further conditional repair. This unit directly tests the
user's aggregate-mass versus argmax question at that early dose.

The strongest alternative to a useful ranking-specific effect is that either
objective suffices at this update rule, with general parameter/history changes
causing preservation costs. Equal optimizer steps and LRs do not match gradient
shape, parameter displacement or attained functional margins. This is an
operational fixed-update comparison, not scale-independent objective superiority.

## Frozen computation and inputs

Inherit the round06 accepted release at
`/data/CoordExp/.worktrees/greedy-prefix-native-01/outputs/research/physical-fn-recovery/2026-10-03/short-dose-ranking-06/released-contract-01.json`,
SHA256 `a61d29fc25951e50df483f34dfd41ddd13e1e3c6271ce5bba8e07961b9edccd7`.
Its lead acceptance is `lead-acceptance-01.json` in that directory, SHA256
`e5b46c940c420af18bfb2bb5ddcb00b77f1d33e06296ad88529834c02d3f2367`.
Reuse its anchor, exact ten contexts, original18 images/570 evaluation labels,
media frontend, target-excluding replay, native settings and unchanged evaluator.
Anchor weight identity is
`07ea98e90220a9126042a27b3a76f60e9955320fed4bf77c2a8803ac9d52001a`.

Train only Gmass for one update from this original anchor. In FP32 logits,
`Gmass(z,L) = logsumexp(z_all) - logsumexp(z_L) = -log P(L)`.
Legal x1 support is the same coordinate bins0..998; its full-vocabulary
complement includes999 and every non-coordinate token. Reuse the maintained
formula from `online_row_credit.legal_objective` directly at the one frozen
predicting logit; do not import its row selection or any mixed-objective weight.
An inline two-logsumexp expression is sufficient. No CE, semantic/GT target,
preservation term, duplicate penalty or adaptive support is added.

The exact original offset0 prefixes for7511-626 and351017-1507 are each repeated
three times, in that site order, with six equal1/6 terms. Neighbors at-1,+1,-2,+2
remain diagnostic only. Use the same588 FP32 language DoRA tensors(rank16,
alpha32,dropout0) and two FP32[1004,2048] delta tables; freeze BF16 FA2 base,
vision and projector. Fresh AdamW once: LRs1e-5/5e-6,betas(.9,.999),epsilon1e-8,
weight decay0, one global clip1 and one step. Seed92711,attention dropout0,
checkpointing OFF,norm OFF. Verify the composed checkpoint0 export matches the
anchor. Collect ten no-grad HF diagnostics at anchor and after the mass update,
without changing the dose or selecting a checkpoint.

Use round06's accepted `package-01/train/checkpoint-1` as fixed Gmax1; do not
retrain it. Weight identity:
`b2fd7427a6f698a0b64a47aa0643cbd6b0dea0fbf3eb0029f11b1d352c8bf1c9`;
parameter identity:
`966fadca07e44c1934efb3c2ac2b4953362a0dd9ccc4db2723debecb7701fc07`.
Bind its exact five payload identities through the accepted receipt. Its loss
was `softplus(1+max(full complement)-max(legal))` under the same six-term update.
Reuse its already accepted HF diagnostics and update/gradient record as
historical evidence, explicitly distinguished from fresh round07 measurements.

Record Gmass per-term losses, full trainable gradient norms, global preclip norm,
clip application and finite trainable state. From saved checkpoint tensors,
compute CPU-only trainable parameter displacement from the common checkpoint0
for Gmass1 and Gmax1. Align component plus exact serialized key; bind equal key
sets, shapes and starting values. Subtract and accumulate in FP64; report combined
and separate DoRA/input-delta/output-delta L2 displacement, relative L2 and element
count. This measures the common serialized parameterization, not merged-weight
or functional distance, and never rescales training. AdamW's first update can
normalize non-negligible gradient scale; a small
mass loss alone does not imply a small parameter change. No gradient matching,
extra optimizer step or extra forward is authorized.

## Fixed observations and interpretation

Four serial model phases: `train`, `native-anchor`, `native-gmax1`,
`native-gmass1`. Finish and release HF resources before native execution.
Each fresh native engine scores/emits the same ten literal contexts in inherited
order, then generates the original18 images from empty history in image order.
Fresh Gmax1 readout is required; old natural outcomes are descriptive context.
Gmax1 remains one previously realized training outcome, not a fresh paired
training replicate; this unit does not estimate training variability.
A changed fresh Gmax1 or anchor outcome is execution variation, not caused by
the new mass objective. One observation per input does not establish determinism.

For Gmax1 and Gmass1 against the shared fresh anchor, report:

- Native emitted legality: repaired/baseline-illegal and retained/baseline-legal,
  by site and inherited context split. Preserve literal tie outcomes and zero
  denominators. Legal mass, native max margin and HF margin are separate readouts.
- Natural category-correct and geometry-only known-owner gains, losses and
  retained IDs per image and aggregate. Also report direct Gmax1-to-Gmass1 owner
  transitions, without calling the transition a sequential training trajectory.
- Invalid rows, literal complete/valid repeats, near-repeat occurrence pairs,
  malformed output, unmatched rows, category disagreements, length and stop.
  Unknown/unmatched remains neutral; repeat pairs are not physical entity counts.

Use unchanged class-agnostic cardinality-first IoU>=.5 assignment and exact
category comparison. The already-fitted18/570 cohort is not population-held-out.
Evaluator GT is never a training target, acquisition selector or dose selector.
Saved-token first divergence and exact-context visitation may be reused as
descriptive readouts only; no extra scoring or causal claim follows.

If Gmass raises legal mass but repairs fewer native errors, that demonstrates
an aggregate-mass/argmax gap under this fixed update rule. If both repair, compare
preservation and recurrence alongside their unequal attained margins. Fewer
repairs with better preservation is a tradeoff, not dominance. Similar outcomes
give no practical ranking advantage at this point. Neither objective contains
owner coverage or repetition credit; the experiment does not prove the cause
of natural owner losses, general SFT inadequacy or physical false-negative
recovery. Do not force a perfect-retention, monotonicity or superiority gate.

## Execution and stop

CPU preparation authorizes no model or GPU execution. Reuse pure maintained
replay/frontend/measurement/evaluator and tested process-group cleanup helpers
from prior probes. Add only `probes/mass_versus_ranking.py` and a focused check
under `tests/probes/`; do not mutate frozen globals, edit predecessor probes or
refactor shared runtime into a new framework. Reuse unchanged earlier checks.

The new actual caller/consumer checks must distinguish the exact Gmass loss and
gradient from Gmax, execute one six-term optimizer update, preserve twenty
no-grad diagnostics, bind the immutable Gmax checkpoint and reject a swapped
arm, changed support, false counters/source/credit or mismatched displacement
starting state. CPU checks may use doubles; the first real update and first
scheduled native readout are the production seams after exact lead release.

The lead selects and qualifies clean execution source, then releases one exact
immutable package. The same worker owns its four serial phases, final readback,
owned-group cleanup and direct return; successful phases continue without a new
gate. Scientific non-repair still completes all fixed observations.

Bounds: one GPU/rank/sequence;1 update,6 training forwards,20 HF diagnostic
forwards,30 native one-token scores and54 natural generations capped at3084:
84 native requests and at most166566 new tokens. Context4456,2GiB KV. Each of
four phases has1800 active seconds plus one30-second owned-group cleanup window,
7200 total active seconds. Preserve native startup/capture defaults; internal
startup forward count remains explicitly unmeasured and its cost stays inside
the phase windows. Record wall/RSS/CUDA/artifact resources and unmeasured fields.
No extra warmup, qualification request, dose, context, label or threshold.

Identity, nonfinite, resource or execution failure stops with partial evidence
and owned-group cleanup; no automatic relaunch or retry of completed requests.
Stop after the fixed comparisons and readback. The worker cannot schedule a
following unit or self-accept scientific results.

Canonical preparation/transport output is
`outputs/research/physical-fn-recovery/2026-10-03/mass-versus-ranking-07/`.
Real outputs belong to the lead-selected execution checkout. The worker owns
the new runner/check, this unit's state/results and task output/scratch. The
lead owns protocol/index/catalog, release, interpretation and acceptance.
