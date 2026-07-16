---
id: decision.use-bagging-as-an-object-support-probe
type: decision
status: active
updated: 2026-07-16
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

Use the bounded [Human-Audited Rare-Object Trajectory Genealogy and Causal
Branch Replay](../investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/unit.md)
unit. Freeze a candidate-conditioned physical-entity ledger, distinguish
fixed-state stochastic selection from earlier natural-row unlocking or
killing, and measure future unique-object utility on fresh suffix seeds.
Training remains unauthorized until a natural branch has a source-specific
causal effect and a preservation-safe target can be stated.
