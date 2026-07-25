---
name: coordexp-vllm-mechanistic-loop
description: Lead a long-running CoordExp vision-language model mechanism investigation from checkpoints and artifacts when surface metrics cannot decide among hidden causal explanations.
---

# CoordExp Vision-Language Mechanistic Loop

Run an **experiment-first mechanism loop**. Use `model-diagnosis` for immediate
symptom triage; use this skill when the user authorizes recursive probes across
multiple turns or runs.

## Start

1. Bind the exact worktree, checkpoints, configs, artifact roots, available
   compute authority, decision-owning outcome, and stop condition.
2. Read current routers and the owning research unit before historical notes.
3. Write or refine one falsifiable unit: question, strongest alternative,
   contrast, primary observation, meaning-bearing invariants, reused
   infrastructure, non-goals, rough cost, and reopening condition.

Start is complete when a unit—not the conversation—owns the executable question.

## Investigate

1. **Select evidence-rich cases.**
   - Use aggregate metrics to find divergent or representative samples, then
     inspect raw rollouts, parser state, traces, and visual evidence.
   - Complete when selection is justified against the question rather than
     convenience.

2. **Obtain the smallest real observation.**
   - Reuse model loading, batching, parsing, scoring, and artifact writers.
   - Keep the first intervention experiment-local and run a representative
     end-to-end smoke through every conclusion-bearing consumer.
   - Complete when the changed factor executed and the primary observation is
     attributable.

3. **Separate competing mechanisms.**
   - Move from behavior and boundary/slot readouts to prefix or control
     contrasts, representation traces, routing evidence, and finally causal
     intervention only as needed.
   - Keep token, span, row, trajectory, module, and free-rollout evidence at
     their actual units. Let the owning unit choose sample scale and probe family.
   - Complete when the leading explanations make different testable predictions
     or the evidence is explicitly inconclusive.

4. **Interpret conservatively.**
   - Separate symptom, candidate mechanism, causal handle, prevalence, useful
     treatment, and scaling evidence.
   - Reconcile aggregate and sample-level evidence; do not treat attention,
     forced continuation, teacher forcing, or local token rescue as the
     decision-owning outcome without a transfer argument.
   - Complete when the claim boundary and strongest unresolved alternative are
     explicit.

5. **Adapt or stop.**
   - Continue with the cheapest discriminating probe while it can change the
     mechanism picture. Pause for user direction on research-meaning changes or
     material new critical-path cost.
   - Stop when the unit's criterion is met, paths are exhausted, evidence
     invalidates the framing, or the user asks to pause.
   - Complete when the result, limitation, and next discriminator are durable.

## Evidence And Artifacts

Record exact checkpoint/config/case identities, request conditions, raw output,
parser or failure status, probe artifacts, and representative result. Preserve
semantic alignment across image, prompt, history, token slots, geometry,
ordering, decode, and evaluation.

Put durable interpretation in the owning research record and run artifacts in
the owning output root. Keep scratch disposable. Promote shared code only after
a stable second consumer or compatibility boundary appears.

After an invalid run, reproduce the failure and pass the full representative
smoke before interpreting a fresh immutable run.

## Report

State objective and owner, selected evidence, executed probes and artifact
roots, mechanism picture with claim boundary, strongest alternative, changed
files, verification, and next gate.
