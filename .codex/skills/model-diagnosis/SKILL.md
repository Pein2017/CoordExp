---
name: model-diagnosis
description: Use when diagnosing model behavior before or after a training, decoding, data, objective, architecture, or optimization innovation, especially when a production run needs tiny-dataset probes, metrics-based trend checks, failure localization, or evidence-backed corrective strategies.
---

# Model Diagnosis

## Overview

Use this skill to turn a training innovation into an evidence-backed diagnosis before or after scaling:

1. What symptoms did the model acquire?
2. Which root causes best explain those symptoms?
3. What fixes or experiments should be tried next?
4. Is the innovation healthy enough to launch a larger production run?

Stay causal. Do not jump from a metric drop to a fix until you have separated artifact problems, implementation bugs, objective mismatch, optimization instability, and genuine model behavior changes.

## Diagnostic Workflow

### 1. Pin The Innovation

State the exact change being evaluated:

- Objective, loss mask, sampling, data, optimizer, schedule, model architecture, decoding, parser, reward, or metric change.
- Intended mathematical effect.
- Expected observable behavior if the change works.
- Comparison baseline and scope: checkpoint, dataset slice, decoding settings, seed, eval contract, and step count.

If scope differs between runs, mark the comparison as non-isomorphic before interpreting quality.

### 2. Build A Symptom Table

Compare before and after along at least these axes when available:

- Main metric: AP, accuracy, reward, loss, perplexity, F1, win rate, or task metric.
- Decomposed metric: precision vs recall, false positives vs false negatives, valid vs invalid outputs, subgroup buckets.
- Generation/rollout health: parse validity, length, stop rate, repetition, empty output, malformed output, schema start form.
- Training dynamics: objective loss, auxiliary losses, gradient norm, learning rate, warmup, optimizer state, train/eval divergence.
- Data path: sample counts, selection distribution, label completeness, weighting, sharding, packing, truncation.

Prefer concrete deltas over adjectives:

```text
AP: 0.40 -> 0.21
parse_valid_rate: 0.96 -> 0.54
GT>=11 recall: 0.52 -> 0.05
```

### 3. Classify Symptoms Before Explaining Them

Use this symptom map as a first pass:

| Symptom | Likely Meaning |
|---|---|
| Train loss improves while eval drops | Objective/eval mismatch, metric hacking, or overfitting |
| Teacher-forced metrics improve while free rollout worsens | Off-policy state mismatch; model behaves badly on self-generated prefixes |
| Parse validity or output validity collapses | Serialization, constrained decoding, token type, or boundary-state failure |
| Recall drops with high precision | Under-generation, premature stop, over-filtering, candidate suppression, missing tail supervision |
| Precision drops with recall stable | Over-generation, duplicate basin, weak filtering, incomplete labels, score calibration |
| Both precision and recall drop | Optimization instability, data corruption, wrong mask positions, checkpoint/eval bug |
| Failure only in crowded/high-count samples | Subset-selection bias, annotation incompleteness, late-object undertraining, exposure bias |
| New non-domain tokens appear in constrained slots | Train-time constrained normalization vs full-vocabulary eval mismatch |
| Sudden change after a schedule point | LR/warmup/optimizer instability, unfreezing, clipping, accumulation, mixed precision |
| Eval artifact reports impossible provenance | Stale checkpoint, wrong adapter, callback artifact bug, merge/shard issue |

### 4. Test Root-Cause Hypotheses In Order

Investigate in this order unless evidence makes a later layer obviously primary:

1. **Artifact and eval validity**
   - Verify the evaluated checkpoint/adapters are the intended ones.
   - Check eval scope, strictness, parser, decoding params, seeds, distributed merge order, and metric source.
   - Compare raw predictions, not only aggregate metrics.

2. **Implementation correctness**
   - Inspect labels, masks, shift positions, boundary tokens, padding, truncation, special tokens, and gradient-contributing positions.
   - Check tokenizer/chat-template alignment with exact token-level examples.
   - Confirm optimizer param groups and frozen/trainable modules match the intended design.

3. **Objective/eval alignment**
   - Ask whether the trained conditional distribution is the same one used at eval time.
   - Look for teacher-forced vs autoregressive mismatch, local branch vs global rollout mismatch, constrained-train vs unconstrained-decode mismatch, or surrogate metric vs target metric mismatch.

4. **Data and sampling distribution**
   - Compare selected training examples with eval failure buckets.
   - Check subset sampling, candidate selection, tail cases, class/length/count distributions, label completeness, and weighting.

5. **Optimization and regularization**
   - Check LR at failure step, warmup equivalence, gradient norms, loss scale, auxiliary-loss ratios, KL/base anchoring, and catastrophic drift.

6. **Genuine model limitation**
   - Only conclude this after simpler bugs and objective mismatches are unlikely.
   - Identify which capability is missing: perception, localization, counting, stopping, ranking, schema control, or robustness under self-generated state.

### 5. Localize The Failure Surface

When outputs are sequential or structured, inspect actual rollouts:

- Start form: does the model enter the expected output mode?
- Slot type: does each field contain the correct token type?
- Boundary transitions: append vs close, separator tokens, terminal state.
- Stop behavior: too early, too late, or post-terminal continuation.
- Repetition: exact duplicate loops, nearby duplicate basin, max-token truncation.
- Per-sample bucket: low-count, high-count, long-tail class, small/medium/large object, hard negatives.

For each failure cluster, give an example raw output and the smallest state transition that goes wrong.

### 6. Separate Fix Types

Propose fixes by mechanism, not by hope:

- **Implementation fix**: repair mask, shift, token span, optimizer group, checkpoint loading, parser, data order.
- **Objective fix**: add missing term, remove misaligned term, reweight terms, change normalization, restore full-vocab competition, add KL/base anchor.
- **Distribution fix**: change subset selection, tail protection, count-bucket sampling, label-completeness weighting, hard-case replay.
- **Decoding fix**: constrain token types, deterministic serializer, grammar-constrained decoding, calibrated stop rule.
- **Optimization fix**: lower LR, restore warmup comparability, freeze sensitive modules, reduce aux weight, add early health gates.
- **Evaluation fix**: improve provenance, add hygiene metrics, preserve strict benchmark contract while adding diagnostics.

State which fixes are diagnostic ablations and which are intended production changes.

## Pre-Production Tiny Probe

Use a tiny probe before expensive training whenever the innovation changes objective math, target construction, tokenizer/template behavior, sampling, packing, precision, optimizer groups, decoding, or evaluation semantics.

The goal is not to prove final quality. The goal is to catch wrong training signals, collapsed objectives, malformed output modes, and unhealthy optimization dynamics while the run is still cheap.

### Probe Design

Build a probe that is small, fast, and adversarial enough to exercise the new contract:

- **Dataset slice**: use tens to a few hundred training examples plus a small held-out slice. Include ordinary examples and known stress buckets such as long outputs, high object count, rare classes, long contexts, boundary cases, or hard negatives.
- **Repetition budget**: train for enough steps to see directional movement, not just one optimizer step. The model should revisit the same tiny slice multiple times so objective-specific metrics can move.
- **Baseline anchor**: compare against the closest existing baseline with the same model, adapter, tokenizer, preprocessing, decoding, batch semantics, seed policy, and eval scope whenever possible.
- **Scope label**: name every result as tiny, smoke, subset, valN, full-val, proxy, teacher-forced, free-rollout, or same-prefix. Never promote tiny results as final validation.
- **Artifact capture**: preserve resolved config, sample counts, tokenizer ids, trainable parameter groups, loss-term weights, raw predictions, parse/drop counters, and metric logs.

### Metrics-Based Trend Pack

Design metrics from the innovation contract rather than from what is already logged. For each new training method, ask: which probability mass should increase, which failure mass should decrease, and which distribution should remain non-collapsed?

Use this metric pack as a reusable checklist:

| Metric Family | What To Track | Healthy Tiny-Probe Signal |
|---|---|---|
| Objective progress | primary loss, decomposed loss terms, auxiliary terms, weighted total, unweighted raw terms | intended terms improve without one term silently dominating |
| Target composition | target counts per sample, target-type fractions, effective weights, support size, mask density | the tiny run actually exercises the new objective instead of degenerating into the old objective |
| Probability mass | valid-target mass, invalid-target mass, allowed-token/type mass, stop-vs-continue margin, entropy or KL to the intended target distribution | valid mass rises, invalid/schema-violating mass falls, and entropy matches the intended sharp or flat distribution |
| Structured output health | parse validity, malformed rate, empty rate, length/count, stop rate, duplicate/repetition rate, truncation rate | format stays valid while task behavior changes |
| Task behavior | precision, recall, FP/FN buckets, count error, subgroup metrics, hard-case buckets | movement matches the intended behavioral hypothesis, not just aggregate metric noise |
| Optimization | gradient norm, LR, loss scale, NaN/Inf counters, parameter update norms, train/eval divergence | stable updates with no precision or accumulation pathology |
| Runtime contract | padding/packing keep rate, logits positions, label-mask positions, effective batch, shard/merge counts | metrics prove the intended rows and token positions contributed to loss |

For multi-target or soft-label objectives, add distribution-health metrics. Useful generic examples:

- mass assigned to all valid positives vs invalid vocabulary;
- entropy or KL inside the valid-positive set;
- top-1 target identity changes across steps or prefixes;
- fraction of examples with more than one active positive;
- stop-token mass vs continue-token mass when early termination is risky;
- allowed token-type mass when structured schemas must remain parseable.

For autoregressive structured generation, always pair teacher-forced metrics with at least one tiny free-rollout diagnostic. Teacher-forced improvement without free-rollout stability is an exposure-bias warning, not a launch signal.

### Trend Interpretation

Read tiny-probe trends mechanically before interpreting task quality:

- **Healthy**: objective loss decreases, valid probability mass rises, malformed/schema mass stays low, gradient norms remain bounded, and free rollouts preserve the intended output mode.
- **Collapse to baseline**: main loss improves but target-composition metrics show the new target family is rare, zero-weighted, or dominated by canonical one-hot positions.
- **Objective miswire**: loss is finite but invariant metrics fail, teacher token is outside support, effective weights are zero, logits are shifted to the wrong positions, or padding/truncation tokens contribute to loss.
- **Distribution collapse**: valid mass rises by concentrating on one easy/canonical target when the intended target distribution should stay broad or balanced.
- **Schema drift**: task metrics move but malformed outputs, invalid token types, or parse drops increase. Treat this as a contract failure until proven acceptable.
- **Stop-policy failure**: stop probability rises or falls contrary to the intended count/continuation policy, especially in high-count or incomplete-label buckets.
- **Precision pathology**: probability metrics are noisy, saturated, or NaN/Inf in low precision. Recompute log-softmax, logsumexp, and probability diagnostics in fp32 where the objective depends on small mass differences.

### Launch Gate

A tiny probe supports larger training only when all of these are true:

- no artifact, tokenizer, label-mask, logits-position, precision, or eval-provenance contradiction is visible;
- objective-specific composition metrics prove the innovation is active on enough training positions;
- trend metrics move in the intended direction for the exact probability masses the innovation claims to change;
- structured-output health does not regress beyond an explicitly accepted tolerance;
- at least one free-rollout or task-facing diagnostic agrees with the teacher-forced trend;
- residual risks are named with the smallest next smoke or ablation that can falsify them.

If any gate fails, report the probe as a diagnostic failure, not a weak-quality result.

## Output Format

Use this compact structure unless the user asks otherwise:

```text
Diagnosis:
  One-sentence causal summary.

Evidence:
  Metric deltas, failure buckets, raw-output examples, and training dynamics.

Probe Scope:
  Tiny/full, train/eval slice, step count, checkpoint/adapters, decoding settings, and artifact roots.

Symptom:
  What changed in model behavior, not just what metric changed.

Likely Root Cause:
  Ranked hypotheses with why each is supported or weakened.

Corrective Strategies:
  Concrete fixes grouped by implementation, objective, data, decoding, optimization, and eval gates.

Verification:
  Minimal tests/smokes/artifact checks that would confirm or falsify the diagnosis.
```

## Guardrails

- Do not hide malformed outputs by relaxing the parser unless the user explicitly wants a different benchmark contract.
- Do not call a failure “optimization-generalization mismatch” until masks, token shifts, data selection, and eval provenance have been checked.
- Do not trust aggregate metrics alone when the task has structured outputs; inspect raw predictions or rollouts.
- Do not propose more loss weight blindly. First identify whether the missing signal is absent, misaligned, too weak, or trained under the wrong token distribution.
- Always distinguish “fixes the measured symptom” from “improves the underlying task.”
