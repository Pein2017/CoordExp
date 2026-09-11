## Why

The completed Human-13 static successors can move K-hit owners into greedy
decode, but newly reached prefixes create untrained duplicate attractors and
the gain repeatedly exchanges previously visible owners.  The next smallest
useful experiment is a closed-loop same-panel controller that harvests the
current natural decode, repairs only its first decision-blocking site, and
accepts the update only when a fresh clean-greedy decode preserves the protected
owner set.

## What Changes

- Add an experiment-local dynamic frontier ledger.  At each accepted iteration
  it binds the current HF clean-greedy trajectory, current covered owners,
  metric-valid K-native aliases, duplicate states, and a candidate shortlist.
- Select candidates on the decision-owning HF fp32/SDPA surface.  Packed
  BF16/FA2 logits may prefilter, but cannot decide the scientific shortlist or
  margin until a candidate-specific cross-surface receipt is available.
- Compare two fresh-Source safe arms only: `O-Full-Safe` trains the selected
  full native row; `O-First-Safe` trains only the first token position where
  that row is not greedy-feasible.  Both retain the rectangle-valid gate and
  dynamic duplicate redirection.
- Make every update transactional.  One fresh original-prompt clean-greedy
  decode both accepts or rejects the preceding update and, if accepted,
  materializes the next iteration state.  Rejection restores model parameters,
  AdamW moments, scheduler state, and the prior accepted checkpoint exactly.
- Run one real vertical iteration, including a deliberate rollback drill,
  before a bounded panel pilot.  Final evaluation remains unconstrained HF
  clean greedy at repetition penalty `1.0`.
- Keep the work narrow: no K-miss supervision, positive exhaustiveness target,
  external bridge, full sequence RL, unconstrained same-batch repeated updates,
  validation claim, or production API change.  Do not over-audit or over-design
  this same-panel probe.

## Capabilities

### New Capabilities

- `coordexp-infras-human13-on-policy-bottleneck-successor`: Experiment-local
  contracts for HF-surface dynamic frontier selection, full-row versus
  first-bottleneck training, transactional clean-greedy acceptance/rollback,
  and bounded same-panel execution evidence.

### Modified Capabilities

None.  Existing production packing, model assembly, optimizer, checkpoint,
inference, parser, matcher, and evaluator contracts remain unchanged and are
reused through experiment-local adapters.

## Impact

- New Human-13 research scripts, typed receipts, two leaf configs, focused
  tests, an immutable vertical artifact, and a bounded two-arm artifact tree.
- Reuse of the sealed Human-13 manifest, selected K-native rows, language-only
  DoRA/AdamW training surface, no-padding FA2 path, HF fp32/SDPA scorer and
  evaluator, and one-to-one owner matcher.
- The owning research unit defines scientific meaning, protected owners,
  acceptance rules, stop rules, and claim scope.  OpenSpec owns the bounded
  implementation and execution contract.
- No dependency, architecture, stable production behavior, checkpoint
  promotion, validation, or transfer claim.
