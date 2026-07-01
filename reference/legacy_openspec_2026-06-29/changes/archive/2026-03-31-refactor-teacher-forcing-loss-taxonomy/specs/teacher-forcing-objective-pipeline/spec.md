# teacher-forcing-objective-pipeline Specification (Delta)

## MODIFIED Requirements

### Requirement: Teacher-forcing objective is declared as an ordered YAML pipeline
The teacher-forcing pipeline SHALL derive strict module validation and Stage-2 objective-atom projection from companion canonical module catalogs while preserving authored YAML execution order and module-to-module state handoff.

Normative behavior:

- one canonical objective-module catalog MUST own, per public objective module name:
  - module family
  - semantic role
  - allowed config keys
  - optional config keys
  - allowed application presets
  - module-level `emission_group` when the module participates in Stage-2 atom emission
  - projected Stage-2 objective-atom definitions when the module participates in Stage-2 atom emission,
- one companion diagnostic-module catalog MUST own, per public diagnostic module name:
  - module family
  - semantic role
  - allowed config keys,
- strict objective-module `config` validation MUST be derived from the objective catalog rather than from duplicated handwritten allowlists,
- strict diagnostic-module `config` validation MUST be derived from the companion diagnostic catalog rather than from duplicated handwritten allowlists,
- strict objective `application.preset` validation MUST be derived from the objective catalog,
- objective modules MUST execute in authored YAML order,
- runtime validation and registry resolution MUST NOT reorder the authored objective list,
- later objective modules MAY consume state produced by earlier objective modules in the same pipeline pass,
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
- **WHEN** a new objective module is added to the canonical objective catalog
- **THEN** strict config validation and allowed application preset validation update from that same definition
- **AND** the codebase does not require parallel handwritten allowlist updates in multiple unrelated tables.

#### Scenario: Authored order preserves state handoff
- **WHEN** an objective pipeline declares `bbox_geo` before `bbox_size_aux` and `coord_reg`
- **THEN** later objective modules observe state emitted by `bbox_geo` within the same pipeline pass
- **AND** catalog-driven validation does not reorder execution.

### Requirement: Module registry is strict and validated before training starts
The strict teacher-forcing module registries SHALL validate runtime coverage against their corresponding canonical catalogs before the first training step.

Normative behavior:

- unknown public module names MUST still fail fast before the first training step,
- the runtime objective registry MUST be checked against the canonical objective-module catalog during initialization,
- the runtime diagnostics registry MUST be checked against the companion diagnostic-module catalog during initialization,
- missing runtime handlers for catalog-declared modules MUST fail fast,
- stale runtime handlers that are not present in the corresponding canonical catalog MUST fail fast,
- runtime registry resolution MUST remain deterministic and MUST NOT depend on runtime reflection of unrelated modules,
- registry drift failures MUST identify the missing and unexpected registry entries.

#### Scenario: Runtime objective registry drift fails fast
- **WHEN** the runtime objective registry and the canonical objective-module catalog disagree
- **THEN** trainer initialization fails fast
- **AND** the error identifies the missing and unexpected registry entries.

#### Scenario: Runtime diagnostics registry drift fails fast
- **WHEN** the runtime diagnostics registry and the companion diagnostic-module catalog disagree
- **THEN** trainer initialization fails fast
- **AND** the error identifies the missing and unexpected registry entries.
