---
name: model-diagnosis
description: "Use when CoordExp model behavior is already abnormal or uncertain: metric drops, FP/FN shifts, invalid or malformed outputs, duplication bursts, length/stop/repetition changes, train/eval divergence, optimization instability, or launch-health symptoms."
---

# Model Diagnosis

Stay causal. This skill is a **symptom-to-root-cause debugger**, not a pre-launch contract audit. Separate artifact/eval validity, implementation bugs, objective mismatch, data distribution, optimization instability, and genuine model limitation before recommending fixes.

## Role Boundary

Use this skill when the question is:

- "Why did this metric, rollout behavior, loss, parse rate, prediction count, or qualitative output change?"
- "Is this real model degradation, eval/artifact mismatch, objective mismatch, data shift, optimization instability, or a model limitation?"
- "What small probe or visualization will separate competing behavioral hypotheses?"

Do **not** use this skill as the first pass when the user is asking whether a planned or newly wired mechanism is contract-safe before results exist. Use `model-innovation-risk-audit` first for pre-launch/new-integration risk checks.

Switch to `model-innovation-risk-audit` when symptom evidence suggests silent train/eval/config/runtime mismatch, stale artifacts, wrong adapter, prompt/template drift, schema drift, or metric-contract ambiguity.

## Diagnostic Order

1. **Pin the innovation**: exact change, intended math, expected observable effect, baseline, checkpoint, dataset slice, decoding, seed, and step count.
2. **Build symptom deltas**: main metric, precision/recall or FP/FN, parse validity, length/stop/repetition, loss terms, gradient/LR, data counts/weights/packing.
3. **Classify symptoms** before explaining:
   - train improves, eval drops -> objective/eval mismatch or overfit;
   - teacher-forced improves, free rollout worsens -> exposure/off-policy mismatch;
   - validity collapses -> serialization, constrained decoding, token type, or boundary-state failure;
   - recall down, precision high -> under-generation, stop/filtering, missing tail supervision;
   - precision down, recall stable -> over-generation, duplicates, incomplete labels, calibration;
   - both down -> optimization, corruption, mask/shift, checkpoint/eval bug;
   - crowded-only failure -> sampling, annotation incompleteness, exposure bias;
   - impossible provenance -> stale checkpoint, wrong adapter, merge/shard bug.
4. **Inspect raw rollouts**: start form, field token type, boundary transition, stop, repetition, per-sample bucket.
5. **Propose mechanism-specific fixes**: implementation, objective, distribution, decoding, optimization, or evaluation.

## Tiny Probe Gate

Use before expensive training when abnormal behavior needs a causal read and the change affects objective math, targets, tokenizer/template, sampling, packing, precision, optimizer groups, decoding, or eval semantics.

Probe requirements:

- tens to a few hundred examples plus held-out slice;
- enough steps for repeated exposure, not one optimizer step;
- closest baseline with same model/tokenizer/preprocessing/decode/batch semantics;
- scope label: tiny, smoke, subset, valN, full-val, proxy, teacher-forced, free-rollout;
- artifacts: resolved config, sample counts, tokenizer ids, trainable groups, loss weights, raw predictions, parse/drop counters, metrics.

Track innovation-specific probability mass:

- valid vs invalid target mass;
- entropy/KL inside valid set;
- stop vs continue margin;
- allowed token-type mass;
- malformed/empty/duplicate/truncation rates;
- effective target counts, mask density, support size;
- gradient norm, LR, NaN/Inf, update norms.

Healthy launch signal: intended terms move, valid mass rises, schema health holds, free rollout agrees with teacher-forced trend, and no provenance/mask/precision contradiction is visible.

## Output

```text
Diagnosis
Evidence
Probe Scope
Symptom
Likely Root Cause
Corrective Strategies
Verification
```

Always distinguish "fixes the measured symptom" from "improves the underlying task." Never relax parsers to hide malformed outputs unless the user explicitly changes the benchmark contract.
