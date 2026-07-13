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

For a reproducible code, config, CLI, runtime, integration, flake, or
performance failure without a model-behavior symptom, use
`debug-feedback-loop`. Return here when the observed failure is a change in
model outputs, metrics, rollouts, optimization, or train/eval behavior.

Switch to `model-innovation-risk-audit` when symptom evidence suggests silent train/eval/config/runtime mismatch, stale artifacts, wrong adapter, prompt/template drift, schema drift, or metric-contract ambiguity.

For large or independent diagnosis lanes, delegate to the `model_diagnostician` custom agent. Add `upstream_relation_tracer` only when the symptom depends on upstream library behavior, and add `contract_auditor` only when artifact/eval validity or launch approval is the question.

## Artifact Triage First

Start from the exact artifact root named by the user before interpreting metrics.

- Identify run root, checkpoint, dataset slice, config, decode surface, bbox format, and scope label.
- Open durable summaries before logs when present: `resolved_config.json`, `summary.json`, `metrics*.json`, `run_metadata.json`, and relevant manifests.
- For infer/eval, check raw and scored prediction JSONL, parser/drop counters, token traces, confidence/scoring sidecars, and duplicate guard reports when present.
- For invalid outputs, aggregate failure families before sampling examples: wrong arity, missing fields, unexpected keys, bad coordinate slots, empty objects, truncation, repetition tails, and max-token saturation.
- Decide whether the evidence is `artifact invalid`, `implementation bug likely`, `objective mismatch`, `decoding/parser mismatch`, `data distribution issue`, `optimization issue`, `model limitation`, or `inconclusive-needs-probe`.

## Run Receipt Boundary Check

Use a compact run receipt to rule out surface and provenance traps before explaining a score:

- authored config and resolved/effective config or runtime when available;
- checkpoint/adapter identity, step count, and training budget;
- data slice, image/sample count, bbox/coord format, and evaluation scope;
- decode kwargs, tokenizer/template surface, max tokens, stop handling, and parser/scorer path;
- metric artifact, raw prediction artifact, and intended baseline/comparison target.

This receipt is a boundary check, not the diagnosis. It catches wrong-run, stale-artifact, decode/config, budget, and scorer mismatches, but low performance often comes from hidden behavioral roots. If the receipt is consistent, continue to symptom deltas, raw rollouts, failure-family counts, and mechanism-specific probes instead of concluding "runtime/config is fine, so the model is bad." If receipt facts are missing or ambiguous, label artifact validity as unresolved and name the smallest check that would settle it.

## Diagnostic Order

1. **Pin the observed run**: artifact root, checkpoint, config, dataset slice, decoding, seed, step count, baseline, and intended change if any.
2. **Build the run receipt**: verify the observed run is comparable enough to interpret; treat mismatches as artifact/eval validity blockers, but do not overfit the diagnosis to surface receipt fields.
3. **Build symptom deltas**: AP/AP50/AP75/AR, precision/recall or FP/FN, prediction counts, parse/drop validity, duplicate tails, length/stop/repetition, loss terms, gradient/LR, data counts/weights/packing, and raw-vs-guarded metrics.
4. **Classify symptoms** before explaining:
   - train improves, eval drops -> objective/eval mismatch or overfit;
   - teacher-forced improves, free rollout worsens -> exposure/off-policy mismatch;
   - validity collapses -> serialization, constrained decoding, token type, or boundary-state failure;
   - recall down, precision high -> under-generation, stop/filtering, missing tail supervision;
   - precision down, recall stable -> over-generation, duplicates, incomplete labels, calibration;
   - both down -> optimization, corruption, mask/shift, checkpoint/eval bug;
   - crowded-only failure -> sampling, annotation incompleteness, exposure bias;
   - impossible provenance -> stale checkpoint, wrong adapter, merge/shard bug.
5. **Inspect raw rollouts**: start form, field token type, boundary transition, stop, repetition, per-sample bucket.
6. **Propose mechanism-specific fixes or probes**: implementation, objective, distribution, decoding, optimization, or evaluation. Prefer a discriminating probe when multiple hidden roots remain plausible.

## Tiny Probe Gate

When abnormal behavior needs a causal read before expensive training, read
[probe-gates.md](references/probe-gates.md). It defines tiny-run comparability,
mechanism-conditioned negative-result checks, and Stage-1 coordinate-locality
evidence.

## Mechanism-Conditioned Probe Gate

Before interpreting weak or negative mechanism results, use the corresponding
gate in [probe-gates.md](references/probe-gates.md). A negative claim is complete
only when baseline health, probe handle, sensitive rows, slot-wise effects, and
relevant controls have been checked.

## Stage-1 Coordinate Locality

For SoftCE, Gaussian, or hard-CE coordinate decisions, read the slot-wise
coordinate-locality branch in [probe-gates.md](references/probe-gates.md).

## Output

```text
Diagnosis
Evidence
Run Receipt / Artifact Validity
Probe Scope
Symptom Taxonomy
Likely Root Cause
Corrective Strategies
Verification
Confidence / Verdict
```

Always distinguish "fixes the measured symptom" from "improves the underlying task." Never relax parsers to hide malformed outputs unless the user explicitly changes the benchmark contract.
