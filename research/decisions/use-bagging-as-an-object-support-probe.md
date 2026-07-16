---
id: decision.use-bagging-as-an-object-support-probe
type: decision
status: active
updated: 2026-07-14
topic: qwen3-vl-dense-enumeration
evidence:
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md
relations:
  supports:
    - decision.separate-selection-transcription-commit-stop
  narrows: []
  supersedes: []
---
# Use Bagging as an Object-Support Probe

## Decision

Use repeated full-image stochastic rollouts to study distributed object support,
trajectory diversity, and candidate generation. Do not promote raw bagging or
its current merge policy as the final enumeration method.

Treat repeated boxes as correlated samples from object, category, geometry,
and traversal uncertainty. Count unique audited object modes separately from
repeat-hit events, duplicates, category or localization disagreement,
unsupported hallucination, and invalid rows.

## Evidence

- Equal-call full-image bagging matched or exceeded the final post-merge utility
  of the masked policy despite a reproducible one-opportunity masked advantage.
- Bagging recovered additional audited objects, but generated many repeated
  hits per covered object and failed the absolute manual-precision and
  prediction-count-expansion gates.
- Manual review found a mixture of real unlabeled objects, correlated duplicate
  or fragmented boxes, category disagreement, and true hallucination.
- At exact prefix state 56 on image `12576`, 32 one-row samples split between
  a target pizza and competing left cup while greedy selected the cup. The
  target disappeared at the paired greedy-terminal state. An independent chair
  case reproduced rescue-entry availability and terminal-state disappearance.

## Belief Update

Bagging can expose both fixed-state competing object modes and objects unlocked
by earlier trajectory state. It does not establish that visual capacity is
complete, that greedy decoding is the only bottleneck, or that the sampled
union is a safe detector output. Its immediate value remains diagnostic and as
a human-audited data candidate source.

## Next Discriminator

Use the bounded Prefix-State Phrase-Geometry Factorial named by the completed
causal-replay unit. Independently vary complete-row description and geometry at
the same exact prefix and equal token length, with no-row, covered-duplicate,
and irrelevant controls. Training remains unauthorized until this identifies a
nontrivial endogenous state target and a safe labeled cohort exists.
