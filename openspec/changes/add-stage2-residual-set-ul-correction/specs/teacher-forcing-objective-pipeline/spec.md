## ADDED Requirements

### Requirement: Residual-set builders compile events into TeacherForcingTargetIR

The teacher-forcing objective pipeline SHALL receive residual-set Stage-2
supervision through `TeacherForcingTargetIR` / `SupervisionAtom`, not through
Stage-2-specific loss-module logic.

Normative behavior:

- Stage-2 residual builders MAY create `CorrectionEvent` records upstream for
  diagnostics and artifact provenance.
- Objective runners MUST consume only objective-facing atom fields: roles,
  valid token ids, selected token id, loss tags, loss weight, coverage weights,
  and target/logit positions.
- Loss modules MUST NOT reimplement FN/FP/duplicate/UL mining semantics.
- Every atom produced by the residual-set builder MUST satisfy next-token
  alignment and selected-token equality validation.

#### Scenario: Correction event compiles to atom

- **WHEN** a residual-set correction event is selected
- **THEN** the builder compiles it into one or more `SupervisionAtom` records
- **AND** the objective pipeline computes loss from the IR without inspecting
  Stage-2 correction-event internals.

#### Scenario: Logit alignment is validated

- **WHEN** a residual-set atom is validated
- **THEN** validation requires `logit_position + 1 == target_position`
- **AND** the selected token id matches the corrected input sequence at
  `target_position`.

### Requirement: Coordinate spans compile to multiple aligned atoms

The teacher-forcing objective pipeline SHALL support a single residual-set
coordinate correction event compiling to a local bbox-tail span of aligned
atoms.

Normative behavior:

- Each coordinate slot in the bbox tail SHALL compile to its own
  `SupervisionAtom`.
- All atoms from the span MUST share event provenance.
- Each atom's effective loss weight SHALL be computed from that slot's committed
  object or valid-action support provenance; coordinate span atoms are not
  required to share one scalar weight when commitment/provenance changes within
  the span.
- Each atom MUST independently declare the correct token role, valid token set,
  selected token id, target position, and logit position.
- The span MUST NOT require loss modules to know whether the event came from
  matched GT, UL, transition failure, or repeated-object provenance.

#### Scenario: y1 coordinate event compiles to y1 x2 y2 atoms

- **WHEN** a coordinate correction anchors before `y1`
- **THEN** the builder emits aligned atoms for `y1`, `x2`, and `y2`
- **AND** each atom is valid under the current residual-state-machine action
  state at that slot.

#### Scenario: Coordinate span atom weights follow slot provenance

- **WHEN** a coordinate span begins with mixed labeled/UL ambiguous support and
  later commits to a UL-promoted object
- **THEN** the ambiguous atom uses the mixed-support scalar weight
- **AND** the committed UL atoms use the configured UL provenance weight.
