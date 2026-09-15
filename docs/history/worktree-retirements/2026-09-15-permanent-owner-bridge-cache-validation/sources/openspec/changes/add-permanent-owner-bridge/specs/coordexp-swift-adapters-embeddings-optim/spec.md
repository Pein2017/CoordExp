## ADDED Requirements

### Requirement: Permanent bridge trainable surface

Stage 1 SHALL freeze the vision tower and multimodal merger and jointly train the existing language DoRA parameters, selected special-token embedding deltas, and all permanent-owner-bridge parameters. The bridge group SHALL include atom inventory, routing key/query projections, learned null state, availability admission, and row-write parameters. Every trainable parameter MUST belong to exactly one explicit optimizer group; any missing, duplicated, frozen-when-required, or unexpectedly trainable parameter MUST fail before the first optimizer step.

The production profile SHALL use learning rates `5e-5` for language DoRA, `2.5e-5` for selected embeddings, and `2.5e-4` for bridge parameters under one fresh AdamW/cosine schedule. The row-write output projection MUST be zero initialized, and configured RMS caps MUST bound forward contributions without detaching their gradients.

#### Scenario: Merger parameter becomes trainable
- **WHEN** trainable-surface validation finds a vision-tower or merger parameter requiring gradients
- **THEN** Stage 1 setup MUST fail before optimizer construction completes

#### Scenario: Bridge parameter is not grouped
- **WHEN** a router, atom, admission, null, or row-write parameter is trainable but unmatched by an optimizer group
- **THEN** setup MUST fail and name the unmatched parameter

