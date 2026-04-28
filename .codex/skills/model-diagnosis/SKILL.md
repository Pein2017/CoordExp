---
name: model-diagnosis
description: Diagnose model behavior after a specific training, decoding, data, objective, architecture, or optimization innovation. Use when Codex needs to identify resulting model symptoms, compare artifacts or metrics before and after a change, infer likely implementation or objective-mismatch root causes, and propose corrective strategies or verification gates for ML training runs.
---

# Model Diagnosis

## Overview

Use this skill to turn a training innovation into an evidence-backed diagnosis:

1. What symptoms did the model acquire?
2. Which root causes best explain those symptoms?
3. What fixes or experiments should be tried next?

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

## Output Format

Use this compact structure unless the user asks otherwise:

```text
Diagnosis:
  One-sentence causal summary.

Evidence:
  Metric deltas, failure buckets, raw-output examples, and training dynamics.

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
