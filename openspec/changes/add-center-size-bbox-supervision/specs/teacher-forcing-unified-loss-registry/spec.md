## MODIFIED Requirements

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The system SHALL define geometry loss (`geo`) on decoded continuous boxes in a way that is:
- stable under near-degenerate boxes,
- compatible with packing (segment-local indices),
- compatible with Channel-B FP-neutral masking.

Normative behavior:
- The system MUST decode 4 coords per box from coord-subspace logits using the
  fixed expectation decode path.
- The system MUST canonicalize decoded boxes before applying IoU-based losses:
  - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
  - enforce non-zero size with an `eps` floor where required for CIoU-like terms.
- The system MUST implement `geo` as a weighted sum of:
  - one regression term on decoded boxes, and
  - CIoU on the same canonicalized box representation.
- The default regression term MUST be SmoothL1 (Huber) on `(x_lo,y_lo,x_hi,y_hi)`.
- The system MAY support an opt-in internal bbox loss-space mode such as `center_size`, provided that:
  - the decoded/public bbox representation remains canonicalized `xyxy`,
  - CIoU remains computed on canonicalized `xyxy`,
  - downstream bbox state shared with other loss modules remains canonical `xyxy`,
  - the additional loss-space terms are derived from the canonical decoded box rather than from a new outward bbox expression.
- When the same optional bbox loss-space mode is exposed through both Stage-1 and Stage-2 surfaces, the regression decomposition and weighted reduction MUST remain semantically aligned for equivalent canonical decoded-box inputs so the two paths do not drift.
- The system MUST aggregate `geo` as a mean over the supervised object set for the current context:
  - Stage-2 Channel-A `gt`: identity-aligned GT objects,
  - Stage-2 Channel-B `rollout`: `matched_clean` + `fn` objects (`duplicate` and `unmatched_clean` excluded).

#### Scenario: Channel-A geometry loss aggregates over GT objects only
- **WHEN** Channel-A geometry loss is computed under `context=gt`
- **THEN** it averages over the GT-supervised object set for that Channel-A step
- **AND** it does not depend on any deprecated self-context/final-pass path.

#### Scenario: Center-wise loss-space mode keeps canonical decoded-box state
- **WHEN** an opt-in `center_size` loss-space mode is enabled for `geo`
- **THEN** the regression branch may operate on internal center and `log_w` / `log_h` coordinates
- **AND** CIoU and shared decoded-box state remain canonicalized `xyxy`.

#### Scenario: Shared bbox regression semantics keep Stage-1 and Stage-2 aligned
- **WHEN** Stage-1 and Stage-2 both expose the same optional bbox loss-space mode
- **THEN** they apply the same regression decomposition and reduction semantics on canonical decoded boxes
- **AND** stage-specific wrappers differ only in extraction, masking, or metric emission concerns.
