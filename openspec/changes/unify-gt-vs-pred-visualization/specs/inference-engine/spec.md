# inference-engine Specification (Delta)

## ADDED Requirements

### Requirement: Inference artifacts preserve prediction order for downstream visualization
When the inference engine emits `gt_vs_pred.jsonl`, it SHALL preserve prediction
order so downstream visualization can trace model interpretation order faithfully.

Normative behavior:

- the emitted `pred` array MUST preserve the original relative order of the
  surviving parsed predictions for each sample,
- the engine MUST NOT sort predictions for visualization or evaluation
  convenience,
- if parsed predictions are dropped for validity reasons, the remaining
  predictions MUST preserve their original relative order in the emitted record,
- additive debug metadata MAY preserve dropped-object diagnostics, but MUST NOT
  reorder the surviving predictions.

#### Scenario: Emitted `pred` order matches parsed rollout order
- **GIVEN** a parsed prediction sequence whose surviving valid objects appear in
  source order `A`, then `B`, then `C`
- **WHEN** inference writes the corresponding `gt_vs_pred.jsonl` record
- **THEN** the emitted `pred` array preserves that same `A`, `B`, `C` order
- **AND** it does not sort the predictions by score, geometry, or description.
