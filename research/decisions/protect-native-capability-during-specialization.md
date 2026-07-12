---
id: decision.protect-native-capability-during-specialization
type: decision
status: active
updated: 2026-07-12
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-09-pvci-identity-conflict/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-causal-proposal-bridge/own-prefix-causal-behavior-results-2026-07-12.md
relations:
  supports:
    - decision.let-architecture-emerge-from-hypothesis-gates
  narrows: []
  supersedes: []
---
# Protect Native Capability During Specialization

## Decision

Every training experiment that strengthens designation, proposal, commit, or
stopping must run a matched capability battery. A row-specialized checkpoint
is not a detector improvement unless it preserves enumeration, precision,
valid closure, and language-grounding behavior.

## Evidence

- The anti-copy row checkpoint exhibited strong visual designation while its
  standard one-shot detection prediction count, recall, and mAP collapsed.
- The proposal-bridge rollout produced more predictions and slightly more
  labeled recall, but mostly as duplication, malformed output, and precision
  loss rather than target-specific selection.

## Belief Update

The desired capability is compositional: better row control can coexist with a
worse detector. Training success must therefore be judged on both the intended
mechanism and the pretrained/native behaviors it is meant to retain.

## Next Discriminator

For the next bounded training unit, evaluate before and after on the same
one-shot enumeration slice, controlled row-designation panel, duplicate and
invalid-output counters, prediction cardinality, STOP/closure behavior, and
phrase-geometry binding. If one adapter cannot hold both capabilities, test a
separate routed or composed adapter before changing the architecture.
