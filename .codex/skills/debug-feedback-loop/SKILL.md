---
name: debug-feedback-loop
description: Diagnose or fix a reproducible CoordExp code, config, CLI, runtime, integration, flake, or performance failure with a tight red-to-green feedback loop.
---

# Debug Feedback Loop

Build a **tight loop** that lets reality reject the explanation. Route model
quality, rollout, metric, or optimization symptoms to `model-diagnosis` and
pre-launch mechanism-contract risk to `model-innovation-risk-audit`.

## Choose The Mode

- `diagnose`: reproduce, minimize, identify cause, and specify correction
  without modifying production surfaces.
- `fix`: retain a regression check, implement the bounded correction, and close
  the original symptom.
- `performance`: establish a repeatable measurement and profiler signal before
  changing code.

## Loop

1. **Make the exact symptom red.**
   - Use the narrowest real seam: focused test, config load, CLI fixture,
     artifact replay, known-good/bad differential, disposable harness, seeded
     flake loop, bisect, or profiler.
   - Complete only when the signal detects the reported failure, is repeatable,
     and can be run by the agent.

2. **Minimize without leaving the path.**
   - Reduce one dimension at a time and rerun after each cut. Preserve both the
     minimal signal and original reproducer.
   - Complete when remaining elements are load-bearing or another cut would
     stop exercising the real failure.

3. **Discriminate causes.**
   - Rank a few hypotheses from code and runtime evidence. Each must predict one
     observable change.
   - Instrument only the boundary that separates the leaders; change one
     variable per probe.
   - Complete when one cause explains the evidence or the missing discriminator
     is explicit.

4. **Classify the correction.**
   - Distinguish an implementation defect, missing seam, contract mismatch, and
     a user-owned algorithm/data/objective/statistical decision.
   - Pause before changing user-owned semantics; explain the alternatives and
     evidence target.
   - Complete when the fix boundary and owner are explicit.

5. **Turn red green.**
   - In `fix` mode, retain the regression check at the owning seam, apply the
     smallest correction, rerun it, then rerun the original reproducer and
     adjacent checks selected by impact.
   - Remove temporary instrumentation.
   - Complete only when both minimal and original signals close.

If no credible loop can be built, stop with attempted evidence and the smallest
capture or instrumentation needed. Do not promote a hunch to root cause.

## Report

State mode and symptom, loop command/result, minimized reproducer, hypotheses
and falsification evidence, root cause, correction, regression/original-loop
verification, cleanup, and residual risk.
