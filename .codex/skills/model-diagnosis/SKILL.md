---
name: model-diagnosis
description: Diagnose abnormal or uncertain CoordExp model behavior such as metric, FP/FN, validity, duplication, length, stop, repetition, or optimization shifts.
---

# Model Diagnosis

Run a **causal diagnosis** from the exact artifact root. Separate invalid
evidence, implementation or contract defects, objective mismatch, data
distribution, optimization, decoding, and genuine model limitation before
recommending a change.

Use the owning code/config/runtime path for engineering failures without a
model-behavior symptom and `model-innovation-risk-audit` when the main risk is a
silent newly wired contract mismatch.

## Diagnose

1. **Bind the observed run.**
   - Resolve checkpoint and adapter, authored and effective config, data slice,
     prompt/template, decode, parser/scorer, coordinate surface, budget, seed,
     artifact root, and comparison target.
   - Complete when the run is identifiable and comparison confounds are labeled.

2. **Validate the evidence boundary.**
   - Inspect durable summaries, raw outputs, parser/drop counters, manifests,
     and metric artifacts before logs.
   - Decide `artifact invalid`, `comparable`, or `unresolved`.
   - Complete when wrong-run, stale-artifact, and eval-scope traps are ruled out
     or promoted to the primary finding.

3. **Describe the symptom before its cause.**
   - Build deltas for the metrics and primitives that changed: prediction
     counts, FP/FN, validity, duplicates, length/stop, loss terms, gradients,
     learning rate, data exposure, and raw versus guarded behavior as relevant.
   - Aggregate failure families before sampling examples.
   - Complete when the visible failure has a bounded unit and onset.

4. **Separate hypotheses with the smallest probe.**
   - Inspect raw behavior and use matched baselines, teacher-forced/free-rollout
     contrast, slot or boundary readouts, visual review, tiny controlled runs, or
     mechanism-specific interventions only when they discriminate leaders.
   - Verify that a negative mechanism probe actually touches the claimed
     surface.
   - Complete when one explanation dominates or the next discriminator is
     explicit.

5. **Choose the action at the cause.**
   - Distinguish a symptom patch from an intervention that improves the
     decision-owning behavior.
   - Route silent contract drift to `model-innovation-risk-audit` and a
     long-running hidden-state investigation to
     `coordexp-vllm-mechanistic-loop`.
   - Complete when the recommendation, evidence scope, and claim boundary align.

## Report

State diagnosis, artifact validity, symptom taxonomy, competing explanations,
probe evidence, likely root cause, corrective strategies, verification,
confidence, and what remains unproven.
