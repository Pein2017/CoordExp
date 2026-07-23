## Why

The current rollout-calibration path can imitate useful rows, but it cannot
isolate the incremental training effect of a confirmed repeated physical
instance or represent a duplicate-deleted counterfactual prefix without
falsifying candidate-generation provenance. A bounded extension is needed to
test whether rejecting physical-owner duplication and supervising recovery into
a later valid owner improves greedy unique-object coverage.

## What Changes

- Add strict reviewed physical-owner duplication and recovery event semantics,
  including separate source-prefix and replay-prefix provenance.
- Add a field-balanced complete-row pairwise loss that raises a verified new
  owner and lowers a confirmed covered-owner duplicate under one replay prefix.
- Add duplicate-cleaned complete-row imitation events whose prefixes are
  explicitly declared counterfactual rewrites rather than observed rollouts.
- Add matched training profiles for local rejection and recovery,
  duplicate-cleaned imitation, their combination, positive-only control, and
  Source-preservation-only control.
- Add burst- and image-balanced credit, family-stratified planned-step
  scheduling, diagnostics, focused tests, and one experiment-local StateBank
  assembler.
- Preserve unmatched official false positives as neutral unless review confirms
  a category error or entity hallucination.
- Preserve the existing model, DoRA adapter, selected-token embedding delta,
  packed forward path, Accelerate runtime, checkpoint format, and ordinary
  inference path.

## Capabilities

### New Capabilities

- `coordexp-swift-physical-owner-duplication-training`: Reviewed duplicate and
  recovery provenance, pairwise complete-row treatment, counterfactual cleaned
  trajectory imitation, matched controls, normalization, and artifact
  requirements for the bounded training screen.

### Modified Capabilities

None.

## Impact

The change extends strict rollout-calibration config and StateBank validation,
replay metadata, loss assembly, planned-step scheduling, training diagnostics,
and research data assembly. It adds no runtime dependency and does not change
ordinary supervised training or inference behavior unless a new duplication
training profile is explicitly selected.
