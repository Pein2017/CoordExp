---
id: decision.use-bagging-as-an-object-support-probe
type: decision
status: active
updated: 2026-07-19
topic: qwen3-vl-dense-enumeration
evidence:
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-13-spatial-scope-history-disentanglement/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-fixed-prefix-complete-box-coherence-and-coordinate-release-factorial/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/results.md
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
- Human review found real Common Objects in Context 80-category (`COCO-80`)
  entity support behind all `119` purposive
  unmatched bagging candidates, while only `28` had acceptable geometry and
  no unique physical-entity references were frozen.
- A fixed-prefix successor found strong `x1`-to-`x2` geometry transport in one
  dense-chair state, but a visibly complete fork remained part-like after its
  whole-object `x1,y1` were forced. The chair geometry also changed the next-
  row category without proving correct uncovered-object redistribution.

## Belief Update

Bagging can expose both fixed-state competing object modes and objects unlocked
by earlier trajectory state. It does not establish that visual capacity is
complete, that greedy decoding is the only bottleneck, or that the sampled
union is a safe detector output. Its immediate value remains diagnostic and as
a human-audited data candidate source. A retrieved row must not be treated as
one committed physical object solely because its phrase is correct: geometry
may represent a discriminative part, a hybrid, or several same-category
instances. Early-coordinate coupling is evidence of autoregressive spatial
transport, not automatically evidence of instance ownership.

The 2026-07-18 human-resolved branch-value screen adds a further boundary: no
sampled-only branch produced a positive, safety-preserving four-row value gain
over the greedy branch on the relabeled image `2299`, so a bagging-revealed
row is a candidate for audit, not automatically a preferable training target.

## Next Discriminator

Do not increase same-arm bagging or coordinate-release sample count. If this
branch resumes, freeze a candidate-conditioned unique physical-entity ledger
and compare a real-object `x1` cue with a matched synthetic or object-free `x1`
cue that has the same displacement and box-width prior. Adjudicate both the
released current box and whether the next row is new, previously committed, or
a geometry-sorted recovery. Training remains unauthorized until a physical-
object-specific causal effect and a preservation-safe target can be stated.
