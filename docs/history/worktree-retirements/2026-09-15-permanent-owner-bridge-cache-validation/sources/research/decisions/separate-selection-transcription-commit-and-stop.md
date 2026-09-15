---
id: decision.separate-selection-transcription-commit-stop
type: decision
status: active
updated: 2026-07-12
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-10-pvci-native-commit-to-uncovered-redistribution/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-spatial-commit-field/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-identity-slot-counterbalance/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-canonical-commit-depth/unit.md
relations:
  supports:
    - decision.let-architecture-emerge-from-hypothesis-gates
  narrows: []
  supersedes: []
---
# Separate Selection, Transcription, Commit, and Stop

## Decision

Treat selection, row transcription, commit/coverage, and stopping as distinct
capabilities in experiments and metrics. Do not infer a stable object ledger
from ordinary pure-CE serialization or from a reduction in exact repetition.

## Evidence

- A valid canonical row immediately suppresses itself and promotes later rows
  in candidate scores, largely through coordinate spans.
- The suppression is asymmetric under identity/slot counterbalances and remains
  compatible with learned raster order, row position, and coordinate history.
- Canonical teacher-forced transitions persist across several depths, while
  endogenous continuation remains noisy and is not an order-free ledger.

## Belief Update

The native prefix/KV state is a real short-horizon control substrate, but the
current evidence identifies an order-conditioned transition rather than stable
instance commitment or complete coverage. Sorted order is therefore useful
evidence about an implicit traversal prior, not a solution to enumeration.

## Next Discriminator

At one controlled transition, write the same committed visual identity through
canonical, parser-clean endogenous, and counterfactual prefixes. Measure whether
the next choice suppresses that identity and redistributes specifically to
annotated-uncovered candidates, while STOP is scored separately. Only a
successful identity-specific write-read test authorizes an explicit commit
carrier experiment.
