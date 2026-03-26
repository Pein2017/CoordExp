# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical loss component names and contexts are canonical and shared
The unified loss registry SHALL define an internal objective-module taxonomy that separates family classification from semantic role while keeping the current public module names stable.

Normative behavior:

- in addition to canonical public module names and canonical loss component names, each objective module MUST have:
  - a family classification,
  - and a semantic role,
- the same taxonomy definition MUST be reused across:
  - strict config validation,
  - runtime objective/diagnostic registry coverage validation,
  - and Stage-2 objective-atom projection,
- duplicated hardcoded module-identity tables across those surfaces are not part of the intended architecture,
- public objective module names and emitted atom names remain unchanged in this first refactor slice.

#### Scenario: Module taxonomy is reused across validation and projection
- **WHEN** the codebase reasons about a teacher-forcing objective module
- **THEN** config validation, runtime registry coverage, and Stage-2 atom projection reuse the same canonical module taxonomy
- **AND** module identity is not re-authored independently in each surface.

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The unified loss registry SHALL classify bbox-dependent objective modules and atoms coherently under a shared bbox family while keeping geometry and size auxiliaries semantically separate.

Normative behavior:

- bbox-dependent objective modules MUST be classified under a shared internal `bbox` family,
- within that family:
  - `bbox_geo` MUST have semantic role `geometry`,
  - `bbox_size_aux` MUST have semantic role `size_aux`,
- bbox-geometry atoms such as:
  - `bbox_smoothl1`
  - `bbox_ciou`
  MUST remain owned by the geometry role,
- bbox-size auxiliary atoms such as:
  - `bbox_log_wh`
  - `bbox_oversize`
  MUST remain owned by the size-aux role,
- bbox-dependent terms MUST NOT be folded into `coord_reg` taxonomy or treated as generic coord regularization just because they consume coord-decoded boxes,
- `ciou` remains a bbox-geometry term even when the decoded-box path reuses coord-token predictions.

#### Scenario: CIoU remains categorized as bbox geometry
- **WHEN** the system classifies the `bbox_ciou` objective atom
- **THEN** it is owned by the bbox geometry role under the shared bbox family
- **AND** it is not categorized as a coord-regularization term.
