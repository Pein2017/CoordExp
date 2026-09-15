---
title: Historical Experiment Handoff Intake Audit and Current Routes
description: Audited classification of the recovered CoordExp experiment synthesis, its evidence limits, and its destinations in the current research graph.
type: investigation
role: intake-audit
authority: non_normative_research
status: complete
updated: 2026-07-18
---

# Historical Experiment Handoff Intake Audit and Current Routes

## Audit scope

This audit read the router and all thirteen numbered Markdown documents in the
source package. It also parsed every row in the two tab-separated ledgers:

- 74 claim rows in `claims.tsv`;
- 4,058 source rows in `source_coverage.tsv`, comprising 684 historical
  `progress/` contents and 3,374 output Markdown contents.

The source package validator completed successfully before import:

```text
OK: 4058 source rows (684 progress, 3374 output)
OK: 74 bidirectionally linked claims
```

The imported ledgers have no duplicate claim or source identifiers, no missing
required fields, and no missing destination documents. Four identical content
hashes occur once in each source domain; these are intentional cross-domain
copies, not duplicate source identifiers.

## Import decision

All thirteen synthesis documents were retained because each has a distinct
future use, but they were assigned different reading roles. This is not a
promotion of every historical claim.

| Document | Intake decision | Reason for retention | Authority after import |
|---|---|---|---|
| `01_cross_branch_landscape.md` | Keep as a routing snapshot | Prevents a new probe from overlooking an already-tested family | Historical router; current owners must be re-read |
| `02_coverage_ledger_mechanism.md` | Keep as bounded mechanism evidence | Separates hidden-state accessibility from actual output causality | Selected-case evidence only; no explicit-ledger claim |
| `03_gaussian_rps_mechanistic_round.md` | Keep as negative and validity evidence | Records that the compared runs had incompatible token-health and validity states | No objective-superiority or production claim |
| `04_autoregressive_duplication_capsule.md` | Keep as bounded mechanism evidence | Preserves the gap between attention-route movement and output recovery | Selected cases only; no universal head or mask |
| `05_binding_and_formation_capsule.md` | Keep as bounded mechanism evidence | Shows that layer, site, onset, and geometry effects are heterogeneous | Mechanism-generating evidence, not population evidence |
| `06_prefix_history_and_route_capsule.md` | Keep with a supersession warning | Preserves the usable prefix-route result and records why the earlier `v7` arm is invalid | `v9`-bounded historical evidence only |
| `07_stage2_rollout_failure_registry.md` | Keep on the active historical reading path | Gives reusable failure classes and corrected-artifact boundaries | Historical results; current Stage 2 runtime remains authoritative |
| `08_coordinate_objective_and_decode_negatives.md` | Keep on the active historical reading path | Prevents repeated objective and decode confounds | Negative-result registry, not a current objective ranking |
| `09_training_and_runtime_lessons.md` | Keep on the active historical reading path | Converts repeated execution mistakes into cheap checks | Reusable research lessons; current docs own behavior |
| `10_inference_evaluation_and_artifact_lessons.md` | Keep on the active historical reading path | Preserves denominator, parser, and artifact-identity rules | Reusable evidence lessons; current evaluation docs own contracts |
| `11_historical_result_registry.md` | Keep as lookup only | Retains exact historical metric handles that may seed matched replication | No cross-family leaderboard authority |
| `12_legacy_design_lineage.md` | Keep as vocabulary and lineage lookup | Maps old names to planning, execution, and result states | Historical provenance only |
| `13_evidence_atlas.md` and the ledgers | Keep as provenance | Makes every retained claim traceable to content hashes and source paths | Identity and coverage authority, not scientific truth |

The raw byte-preserving unions were not imported. They contain roughly 326
megabytes across more than 8,000 files and would make the active worktree a raw
archive rather than a research reading path. Their Git commits remain the
retrieval root; this package keeps the content hashes and source paths needed to
recover a particular record.

## Findings worth carrying forward

### 1. Rollout evidence cannot be inferred from teacher forcing

Similar coordinate-token loss, coordinate mass, or local readout can coexist
with very different free-rollout detection behavior. Every training experiment
that matters to enumeration should therefore include a small free decode with
object count, termination, truncation, invalid-row, and duplicate observations.

### 2. Decode policy is an experimental factor

Historical repetition-penalty changes materially altered prediction count,
duplication, validity, and Average Precision. An objective comparison is not
causal when repetition penalty, temperature, top-p sampling, token budget,
prompt, template, or stop handling also changed.

### 3. Hidden accessibility is weaker than causal use

Large direct projections through the language-model output head, attention
mass shifts, or route-state movement often produced small or inconsistent
changes in actual final logits and generated output. Future work must keep
direct-hidden, final-normalized-hidden, final-logit, and free-continuation
effects separate.

### 4. No durable object ledger has been established

History-conditioned coordinate and continuation signals exist in selected
states. That is compatible with a coverage-like signal, but it does not prove a
stable, explicit, object-indexed memory. The current question remains whether
emitting one coherent row redistributes probability toward a valid uncovered
object, rather than merely continuing a learned serialization pattern.

### 5. Binding is not localized to one universal layer or operation

The historical panels separate descriptor choice, first-coordinate basin,
later geometry, continuation, and stopping. Effects vary by checkpoint,
prefix, coordinate slot, intervention site, and layer. A successful selected
patch does not justify a universal layer, head, or fixed residual direction.

### 6. Failure labels must remain distinct

Exact duplicate, near duplicate, fragmented geometry, category mismatch, real
but unlabeled object, and unsupported hallucination are different outcomes.
Collapsing all unmatched predictions into false positives hides the behavior
that a treatment is supposed to change.

## Historical claims that must not be promoted

- The Gaussian coordinate soft-target plus Ranked Probability Score runs do
  not establish superiority; their validity and token-health states differed.
- Repetition penalty reducing exact duplicates does not prove better coverage.
- Oracle-K recoverability does not define a deployable inference policy.
- A high historical Average Precision row does not outrank another family
  unless data, prompt, template, serialization, checkpoint, decode, parser,
  evaluator, and denominator are matched.
- A decodable or patchable hidden direction does not prove the model naturally
  uses that direction during rollout.
- A selected attention head or decoder layer is not a universal mechanism.
- An OpenSpec plan, model card, smoke run, or visualization gallery is not proof
  that a population-level model claim holds.

## Current research routes

Use these current owners before acting on a historical claim:

- [Qwen3-VL dense enumeration compass](../qwen3-vl-dense-enumeration/compass.md)
  for the active belief state and next discriminator.
- [Autoregressive binding template study](../../archive/autoregressive-binding-template-study/index.md)
  for the existing 223-record binding synthesis.
- [Painted ground-truth transcription probe](../../ideas/qwen3-vl-painted-gt-transcription-probe/overview.md)
  for privileged visual designation evidence.
- [Prefix denoising supervised fine-tuning](../../ideas/prefix-denoising-sft/overview.md)
  for the tested denoising recipe and its limits.
- [Require target-specific causal consumption](../../decisions/require-target-specific-causal-consumption.md)
  for the current decision boundary on learned bridges.
- [Stage 1 objective](../../../docs/training/STAGE1_OBJECTIVE.md),
  [Stage 2 runbook](../../../docs/training/STAGE2_RUNBOOK.md), and
  [evaluation workflow](../../../docs/eval/WORKFLOW.md) for current runtime
  and evaluation contracts.

## Use rule

A historical result may do one of three things: prevent an identical failed
probe, supply a matched control, or motivate a new discriminator. It may not
silently set a current default. Before reuse, bind the current checkpoint,
data slice, prompt, output format, decode policy, parser, evaluator, and
artifact root, then state which historical limitation the new experiment
changes.
