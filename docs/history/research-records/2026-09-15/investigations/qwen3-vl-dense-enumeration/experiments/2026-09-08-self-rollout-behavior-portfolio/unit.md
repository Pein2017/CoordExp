---
title: Bounded self-rollout behavior portfolio and routing comparison
type: investigation
role: research-portfolio
status: completed
evidence_status: bounded_inference_lead_accepted
architecture_promotion_status: not_promoted
updated: 2026-09-08
---

# Authority and stop

The user explicitly authorized immediate parallel execution without another
confirmation on 2026-09-08, plus a Sol-versus-Astra comparison, then permitted
L2 subagents. This supersedes the preceding discussion-only boundary. Root owns
scientific acceptance, portfolio integration, resource allocation and routing.
It does not authorize an unbounded training sweep or a Codex config change.

First wave: complete the bounded inference/measurement probes below and draft
the corresponding training decision. No optimizer update, new architecture,
promotion, publication, Git commit, or mutation of old payloads in this wave.
Each lane has its own question; one lane's ambiguity does not block the others.

New direction worktree: `/data/CoordExp/.worktrees/self-rollout-behavior`, branch
`probe/self-rollout-behavior`, from research-probes commit
`73b8b3cc2614db052055c7c49fc33d6207b0f80a`. The protected source was locked and
had only shared-guidance deletions at admission; no source dirt was copied or
modified. Existing executable dependencies in
`/data/CoordExp/.worktrees/dora-prox-linear-n2` remain read-only and hash-bound.

Output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-08-self-rollout-behavior`.

## Shared scientific invariants

- Original Source step-2444, frozen native architecture/DoRA and selected
  embeddings. Feedback-off alone instead uses the terminal feedback-trained
  DoRA and ablates its inference projection contribution.
- Primary behavior: one native greedy trajectory, RP1.10, maximum 3084 generated
  tokens for the WHOLE trajectory, including a retained prefix. Preserve real
  image/template/token/position semantics. No prediction postprocessing.
- Category-consistent pixel-space IoU>=0.50 global cardinality-first one-to-one
  matching; tie-break by total IoU. Recompute assignments after extensions.
  This remains the frozen official A/B/F metric. The later user clarification
  below adds a separate category-agnostic P diagnostic without rewriting it.
- Owner count is not a number of emitted rows. Retain gained/lost IDs, full
  denominators, parsing/length/EOS outcomes and annotation-relative unmatcheds.
- Geometry-repeat diagnostic: any earlier prediction box at pixel IoU strictly
  >0.95, category agnostic. This is not equivalent to GT-owner duplication.
- Unknown/unmatched is not automatically hallucination or a negative label.
  Reviewer labels are auxiliary evidence, not automatically human gold.
- No best-of-K union or forced-prefix outcome is a deployed-model improvement.
- A/B use 16 seed-selected images from the existing feedback TRAIN128 pool,
  selected without generated-output inspection. This pool has prior selection
  history; it is not representative confirmation. Dev64 remains development.
- `input-v1/source16.jsonl` preserves resolved absolute media paths as audit
  serialization, but the native loader rejects absolute source-image references.
  A attempt1 exposed this before model loading. Native consumers therefore load
  the manifest's original `raw_train256` and select the same frozen16 IDs, checking
  each original row and resolved media hash. The sealed audit input is unchanged;
  this root-owned serialization error is not a model/worker defect or a new cohort.
- Existing train/dev/confirmation boundaries remain intact. Do not read the
  held-out512 or legacy-heldout128 outputs to choose events or hypotheses.

## F: feedback-off, also the model-routing work sample

Question: conditional on the already trained feedback DoRA, does removing the
inference projection contribution recover native dev64 behavior?

One full cold evaluation, same dev64/order/RP1.10/cap as the prior pilot. Preserve
and validate the original feedback payload before disabling its runtime
contribution. Compare to existing feedback-on, control and Source artifacts.
Co-adaptation limits causal attribution; this is not a new training control.

Sol-high and Astra-low independently implement the same small evaluator in
disjoint candidate directories, with the identical `routing-brief.md`. Neither
may spawn helpers or inspect the other candidate. Both get one bounded real
qualification; only the accepted selected candidate runs full dev64 once.
Root compares contract correctness, real checks, rework, elapsed time and
available task-boundary usage. One work sample cannot establish universal
model superiority or justify silently deleting Sol from configuration.

F budget: GPU0 for the selected full evaluation, at most 40 minutes. Candidate
qualification uses GPU4 (Astra) or GPU5 (Sol), <=10 minutes each. No retraining,
payload rewrite, additional RP sweep or checkpoint search.

## P: coordinate-proxy evidence and minimal reviewer fallback

Question: is existing coordinate likelihood sufficient for a reliable
unmatched-object proxy, rather than merely a ranking signal?

The July29 reports show a binary signal but not independent five-way semantic
labels. Their recorded raw output root is absent after a bounded recovery-path
check. Missing artifacts are not a scientific proxy failure.

Score the retained Source-RP1.10 dev64 trajectories at their exact natural
prefixes with the original Source. Record raw chosen coordinate-token logprobs,
coord_mean/coord_min, description, geometry, row position, GT matching and
geometric-repeat flags. No new generation. Reuse the existing canonical
teacher-forced chosen-token scorer when possible, not transformed RP logits.
Qualify on one real trajectory before all64; preserve EOS/action identity.

No five-class p-value or deployment threshold may be claimed from GT-unmatched
labels alone. First return a calibrated-boundary verdict (usable / triage-only /
insufficient independent evidence), and a bounded blinded candidate review set.
Root may then invoke one independent native vision-capable reviewer on crops,
with tight plus context views, entity/category, single-instance, geometry and
uncertain judgments. Do not build an external service, invoke an unconfigured
paid endpoint, or treat reviewer self-confidence as human truth. Retain the
official-GT metric separately from any candidate supplemental owner evidence.

P budget: GPU1, <=30 minutes scoring, at most two technical attempts including
qualification. Initial review proposal <=24 candidate boxes; no automatic
review of every prediction or feature/threshold sweep.

## A: same-prefix downstream-value probe

Question: do alternative next actions with equal immediate owner gain have
different eventual coverage under the frozen native greedy continuation?

For each of the selected16 Source images, generate one native baseline. Choose
the boundary after floor(number_of_valid_complete_rows/2) rows, with boundary0
allowed. Freeze the exact token prefix before selecting any sampled actions.
Sample K4 next actions independently at T1/top-p1/RP1, without forced opener.
An action ends at the first box-end marker, EOS, or the remaining global cap;
malformed actions remain outcomes. Then resume frozen native greedy RP1.10.
The sampled action and suffix share the remaining 3084-token global allowance.
The original greedy branch is a separate reference, not an iid RLOO member.

Use C(prefix+action)-C(prefix) for immediate count and
C(full_branch)-C(prefix) for final incremental count. Keep paired owner IDs and
diagnostics. Report equal-immediate/different-final cases and ranking reversals;
absence of sampled support closes only this finite sampler/panel, not the
existence of better actions. No oracle GT row silently enters the policy bank.

Deliver a training proposal, not a trained model: row-only likelihood credit,
the precise sampling law, fresh-bank update boundary, weighting and unknown/
legal-sibling semantics must be explicit. No claim that local surrogate gains
guarantee deployment gains. Reuse native mechanisms; do not create a trainer
framework. A owns the sole source16 baseline producer and publishes its sealed
baseline artifact for B; do not independently regenerate those baselines.

A budget: GPU2, <=2 hours, at most two technical attempts total. One real
baseline/branch/parser/matcher/serialization qualification precedes the full
fixed panel. At most16 baselines plus64 sampled branch continuations.

## B: same-covered-set history sensitivity/recoverability

Question: does a small legal history-order change, with the same covered
prediction multiset, materially alter access to remaining annotated owners?

Reuse A's sealed source16 native baseline and midpoint token boundary. For
eligible prefixes with >=2 complete valid rows and exact segment boundaries,
swap the final two complete object rows, preserving every token inside each
row, total prefix length and the multiset of category/coordinate rows. Keep
everything else fixed, then perform native greedy RP1.10 continuation under
the same whole-trajectory budget. Ineligible images remain explicit; no
reselection. Compare to the exact native unperturbed continuation/reference.

This is one predefined off-path stress test, not a claim that the sorted policy
must be order invariant or that history sensitivity proves missing memory.
Report final coverage/gains/losses, remaining-owner realization and repetition/
EOS/caps. No learned state module, layer sweep, iterative repair or training.

B budget: GPU3, <=1 hour, <=16 swapped continuations plus the minimum identity
qualification. At most two technical attempts. B does not write A's baseline
or shared helpers. It may prepare its implementation while waiting for the
baseline completion notification, without polling the producer.

## Execution and integration

Use run-specific durable logs, exact PID/command/exit receipts and atomic final
artifacts. No shell sleep loops or short agent polls. Forward worker completion
or semantic blockers; root replays the decision-bearing evidence before
acceptance. L2 is allowed only for cheap disjoint work within the same package
budget; no descendants in the paired routing sample and no L3.

At the first-wave stop, root produces one synthesis linking each lane's result,
its scope/negative findings, and the Sol/Astra acceptance-cost comparison. Do
not auto-promote a proxy, expand sample sizes, or continue training because
an attractive hypothesis remains unresolved.

## Later user ruling and closeout

The user clarified: `类别错误` is acceptable if a box really corresponds to an
`owner`; `desc` may be masked. The active P candidate criterion is therefore
entity existence, single-instance binding and acceptable localization, with
description correctness diagnostic only. The previous all-four-positive veto
is superseded, not evidence against the owner-binding proxy. Reaggregation of
the same blind labels gives4/16 unmatched candidates instead of1/16. A CPU-only
category-agnostic diagnostic on the existing64 outputs is recorded separately.

Masking description loss is a permissible candidate design direction, not an
executed optimizer change. It neither removes description tokens from the
autoregressive context nor establishes box groundedness. If the proposed A
full-action RLOO is changed to coordinate-only credit, it must be described as
a conditional/partial-score surrogate rather than silently retaining a full
joint-action policy-gradient claim. No new loss weighting or owner matcher is
automatically selected for training by this closeout.

All four bounded inference lanes and the paired engineering work sample are
complete and root-verified. Scientific conclusions and routing recommendation
are in `results.md`; exact evidence bindings are in `lead-acceptance.json`.
No optimizer update, new sample, model promotion, config change or Git commit
was performed. The run-specific GPU processes have exited. Stop reached.
