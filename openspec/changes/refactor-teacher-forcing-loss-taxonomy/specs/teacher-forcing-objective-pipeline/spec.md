# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing pipeline SHALL derive strict module validation from one canonical objective-module catalog so the config surface, runtime registry, and projected Stage-2 atoms cannot drift independently.

Normative behavior:

- the canonical objective-module catalog MUST own, per public module name:
  - module family
  - semantic role
  - allowed config keys
  - optional config keys
  - allowed application presets
  - projected Stage-2 objective atoms when the module participates in Stage-2 atom emission,
- strict module `config` validation MUST be derived from that shared catalog rather than from duplicated handwritten allowlists,
- strict `application.preset` validation MUST be derived from that same catalog,
- `bbox_geo` and `bbox_size_aux` MUST remain separate public objective module names and independently removable pipeline entries,
- `bbox_geo` and `bbox_size_aux` MAY share a common internal family classification such as `bbox` without changing their public YAML names,
- `bbox_geo` continues to own bbox-geometry weights such as:
  - `smoothl1_weight`
  - `ciou_weight`,
- `bbox_size_aux` continues to own bbox-size auxiliary weights such as:
  - `log_wh_weight`
  - `oversize_penalty_weight`
  - `oversize_area_frac_threshold`
  - `oversize_log_w_threshold`
  - `oversize_log_h_threshold`
  - `eps`.

#### Scenario: Adding an objective module updates validation from one source of truth
- **WHEN** a new objective module is added to the canonical module catalog
- **THEN** strict config validation and allowed application preset validation update from that same definition
- **AND** the codebase does not require parallel handwritten allowlist updates in multiple unrelated tables.

### Requirement: Module registry is strict and validated before training starts
The strict teacher-forcing module registry SHALL validate runtime coverage against the same canonical module catalog used for config validation.

Normative behavior:

- unknown public module names MUST still fail fast before the first training step,
- objective and diagnostics runtime registries MUST be checked against the canonical module catalog during initialization,
- missing runtime handlers for catalog-declared modules MUST fail fast,
- stale runtime handlers that are not present in the canonical module catalog MUST fail fast,
- runtime registry resolution MUST remain deterministic and MUST NOT depend on runtime reflection of unrelated modules.

#### Scenario: Runtime registry drift fails fast
- **WHEN** the runtime objective registry and the canonical module catalog disagree
- **THEN** trainer initialization fails fast
- **AND** the error identifies the missing and unexpected registry entries.
