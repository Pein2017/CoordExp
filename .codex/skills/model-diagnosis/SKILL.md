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

## Artifact Triage First

Start from the exact artifact root named by the user before interpreting metrics.

- Identify run root, checkpoint, dataset slice, config, decode surface, bbox format, and scope label.
- Open durable summaries before logs: `resolved_config.json`, `summary.json`, `metrics*.json`, `run_metadata.json`, and relevant manifests.
- For infer/eval, check raw and scored prediction JSONL, parser/drop counters, token traces, confidence/scoring sidecars, and duplicate guard reports when present.
- For invalid outputs, aggregate failure families before sampling examples: wrong arity, missing fields, unexpected keys, bad coordinate slots, empty objects, truncation, repetition tails, and max-token saturation.
- Decide whether the evidence is `artifact invalid`, `implementation bug likely`, `objective mismatch`, `decoding/parser mismatch`, `data distribution issue`, `optimization issue`, `model limitation`, or `inconclusive-needs-probe`.

## Diagnostic Order

1. **Pin the observed run**: artifact root, checkpoint, config, dataset slice, decoding, seed, step count, baseline, and intended change if any.
2. **Build symptom deltas**: AP/AP50/AP75/AR, precision/recall or FP/FN, prediction counts, parse/drop validity, duplicate tails, length/stop/repetition, loss terms, gradient/LR, data counts/weights/packing, and raw-vs-guarded metrics.
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

## Mechanism-Conditioned Probe Gate

Use this before interpreting weak or negative results from denoising, robustness, prefix, hidden-state, coordinate-basin, binding, or duplicate-control experiments.

Do not infer "the mechanism is irrelevant" from low average CE/KL deltas, mild metric movement, or a valid-looking perturbation until you verify the probe actually targets the documented mechanism surface.

Check:

- known healthy baseline quality gate, such as the user's expected mAP/AP/F1 range;
- authored config, resolved runtime, and exact artifact root;
- whether the perturbation can move the state basin or only stays inside a local valid-bbox neighborhood;
- slot-wise effects for `x1`, `y1`, `x2`, `y2`, boundary/control tokens, and stop/continue sites;
- mechanism-sensitive rows, not only aggregate averages;
- wrong-control or same-desc competitor prefixes when prefix/binding is the claimed handle;
- hidden-state patch, visual-region mask, or rollout-generated bad-prefix evidence when prior notes identify those as causal handles;
- sparse sampling traps such as `num_objects_per_image=1`, identical-prefix cases, or mostly insensitive rows.

If prior mechanism notes conflict with the new aggregate result, reconcile them explicitly. Prefer the verdict "probe handle mismatch" or "inconclusive-needs-mechanism-panel" over broad causal claims when the perturbation family does not match the documented failure mode.

## Stage-1 Coordinate Locality

For SoftCE, Gaussian, or hard-CE coordinate objective decisions, keep the readout slot-wise and rollout-aware:

- compare `x1`, `y1`, `x2`, and `y2` separately;
- include teacher-forced logits and self-prefix logits;
- pair distribution tables with plots when judging shape;
- keep A5/A6 or other variants separate instead of flattening them into one "SoftCE" verdict;
- use guarded rollout metrics only with explicit scope labels such as `val200`, checkpoint id, and decode settings.

Existing harness:

```bash
PYTHONPATH=/data/CoordExp python scripts/analysis/run_hard_ce_coord_logit_locality.py --help
python -m pytest tests/test_hard_ce_coord_logit_locality.py -q
```

If a direct script launch cannot import `src`, set `PYTHONPATH=/data/CoordExp` explicitly.

## Output

```text
Diagnosis
Evidence
Probe Scope
Symptom Taxonomy
Likely Root Cause
Corrective Strategies
Verification
Confidence / Verdict
```

Always distinguish "fixes the measured symptom" from "improves the underlying task." Never relax parsers to hide malformed outputs unless the user explicitly changes the benchmark contract.
