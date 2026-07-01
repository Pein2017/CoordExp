# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The system SHALL define geometry loss (`geo`) on decoded continuous boxes in a way that is:
- stable under near-degenerate boxes,
- compatible with packing (segment-local indices),
- compatible with Channel-B FP-neutral masking.

Normative behavior:
- The system MUST decode 4 coords per box from coord-subspace logits using `coord_decode_mode`:
  - `exp`: expectation decode (CoordExp / soft expectation),
  - `st`: ST decode (hard argmax forward + expectation grad).
- The system MUST canonicalize decoded boxes before applying IoU-based losses:
  - `x_lo = min(x1, x2)`, `x_hi = max(x1, x2)`
  - `y_lo = min(y1, y2)`, `y_hi = max(y1, y2)`
  - enforce non-zero size with an `eps` floor where required for CIoU-like terms.
- The system MUST implement `geo` as a weighted sum of:
  - SmoothL1 (Huber) on `(x_lo,y_lo,x_hi,y_hi)`,
  - CIoU on the same canonicalized box representation.
- The system MUST aggregate `geo` as a mean over the supervised object set for the current context:
  - Stage-2 Channel-A `self_context`: identity-aligned GT objects,
  - Stage-2 Channel-B `rollout`: `matched_clean` + `fn` objects (`duplicate` and `unmatched_clean` excluded).

#### Scenario: Duplicate and unmatched clean extras do not contribute to geometry loss
- **WHEN** Stage-2 Channel-B runs with duplicate-certified continuations and unmatched clean extras present
- **THEN** those objects contribute `0` to `geo`
- **AND** only `matched_clean` and `fn` objects contribute to `geo`.

## ADDED Requirements

### Requirement: `bbox_size_aux` is a separate optional decoded-box loss component
The unified loss registry SHALL treat bbox size supervision as a separate
optional loss component `bbox_size_aux`, not as an implicit expansion of `geo`.

Normative behavior:

- `bbox_size_aux` MUST consume canonicalized decoded boxes in the current
  `xyxy` / top-left-bottom-right contract,
- `bbox_size_aux` MUST preserve the current public bbox expression
  `bbox_2d: [x1, y1, x2, y2]` and the current coord-token `0..999` contract,
- `bbox_size_aux` MAY include:
  - matched `bbox_log_wh`,
  - matched `bbox_log_area`,
  - thresholded `bbox_oversize`,
- `bbox_size_aux` MUST be mean-like over the supervised object set for the
  current context,
- the oversize term MUST remain opt-in and MUST NOT define a default small-box
  prior,
- the supervised object subsets MUST match the existing matched bbox contract
  for the active context.

#### Scenario: Matched size auxiliaries vanish on exact match
- **WHEN** predicted and target boxes are identical after canonicalization
- **THEN** the matched `bbox_log_wh` and `bbox_log_area` auxiliaries are near
  zero
- **AND** the result does not depend on original corner ordering.

#### Scenario: Public bbox slot order remains unchanged
- **WHEN** `bbox_size_aux` is enabled
- **THEN** the public bbox expression remains `bbox_2d: [x1, y1, x2, y2]`
- **AND** internal canonicalization does not redefine the serialized slot order.
