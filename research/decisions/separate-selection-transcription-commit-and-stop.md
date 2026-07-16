---
id: decision.separate-selection-transcription-commit-stop
type: decision
status: active
updated: 2026-07-15
topic: qwen3-vl-painted-gt-transcription-probe
evidence:
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-10-pvci-native-commit-to-uncovered-redistribution/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-spatial-commit-field/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-identity-slot-counterbalance/unit.md
  - research/ideas/qwen3-vl-painted-gt-transcription-probe/experiments/2026-07-11-pvci-native-canonical-commit-depth/unit.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-14-sampled-rescue-object-transition-causal-replay/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-15-prefix-state-phrase-geometry-factorial/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-15-visual-support-counterfactual-commit/results.md
  - research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-15-pre-vision-raw-bounding-box-visual-support-counterfactual-commit/results.md
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
from ordinary pure cross-entropy (`CE`) serialization or from a reduction in
exact repetition.

## Evidence

- A valid canonical row immediately suppresses itself and promotes later rows
  in candidate scores, largely through coordinate spans.
- The suppression is asymmetric under identity/slot counterbalances and remains
  compatible with learned raster order, row position, and coordinate history.
- Canonical teacher-forced transitions persist across several depths, while
  endogenous continuation remains noisy and is not an order-free ledger.
- At one exact prefix, completing the target pizza, an already covered pizza,
  or a syntax- and length-matched unsupported chair all predominantly advances
  to the same left cup. Completing that left cup advances to a right cup rather
  than recovering the omitted target pizza. This fails the declared
  object-specific commit-to-uncovered-redistribution crossover.
- At the same exact prefix, independently crossing the target-pizza and
  left-cup descriptions and geometries rejects a phrase-only effect, a
  geometry-only effect, and uniform complete-row advancement. Only the
  coherent `cup` plus left-cup geometry row advances to the right cup; either
  factor alone returns the left cup.
- Replacing the selected local post-vision left-cup support, and then replacing
  the complete raw left-cup bounding-box pixels before visual re-encoding, did
  not change the right-cup successor in greedy decoding or any of eight paired
  samples. The pre-vision panel passed its trust gate, and the control-minus-
  target right-cup fraction was `0.0`.

## Belief Update

The native prefix and key-value (`KV`) cache state is a real short-horizon
control substrate, but the current evidence identifies an order-conditioned
transition rather than stable instance commitment or complete coverage. Sorted
order is therefore useful evidence about an implicit traversal prior, not a
solution to enumeration.
The newest factorial narrows the leading explanation to a
phrase-and-geometry-compatibility-gated serialization transition. Both local
post-vision support replacement and complete raw target-box donor replacement
preserve the successor in this exact state. A text- or prefix-mediated transaction
followed by visual selection of the still-visible successor is therefore the
leading bounded explanation, but visual independence and a canonical
geometry-sorted text transducer remain unproven. This still does not establish
an order-free object ledger or uncovered-set redistribution.

## Next Discriminator

The bounded **Visual-Support Counterfactual Commit Test** and the stronger
**Pre-Vision Raw-Bounding-Box Visual-Support Counterfactual Commit Test** are
complete. Their strong-null result closes this committed-object-support
question for the frozen state and operator family. Do not authorize an
explicit commit carrier or training screen from this result. The next locus is
the bounded own-rollout question recorded in the pre-vision result: whether
causally correcting an approximate or incoherent emitted row changes the next
valid uncovered successor, while separating event construction from broader
next-object state.
