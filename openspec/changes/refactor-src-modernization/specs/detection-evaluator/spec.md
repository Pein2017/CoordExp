# detection-evaluator Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Evaluator ingestion diagnostics are path-and-line explicit
Detection-evaluator SHALL provide path-and-line explicit diagnostics for malformed JSONL ingestion failures.
Diagnostics MUST identify source file and 1-based line number for parse failures.
Diagnostics SHOULD include a clipped payload snippet for rapid operator triage.

#### Scenario: Malformed JSONL line reports precise source context
- **GIVEN** an input artifact containing malformed JSON on one line
- **WHEN** evaluator ingestion parses the file
- **THEN** diagnostics identify the source path and 1-based line number for the malformed record
- **AND** diagnostics include a clipped snippet of the malformed payload.

### Requirement: Evaluator reuses shared coordinate and geometry helpers
Detection-evaluator SHALL reuse shared coordinate/geometry helper contracts for conversion and validation, rather than maintaining parallel helper implementations.
For overlapping active deltas, helper-level strictness/diagnostic defaults are authoritative in `src-ambiguity-cleanup-2026-02-11`; this change MUST remain consistent with that contract.
This requirement SHALL preserve existing evaluation metric intent and artifact compatibility.

#### Scenario: Shared helper reuse preserves evaluation eligibility behavior
- **GIVEN** bbox/poly mixed geometry records
- **WHEN** evaluator processes coordinates through shared helpers
- **THEN** match eligibility decisions remain consistent with canonical helper behavior.

### Requirement: Evaluation artifact and metric schema parity is preserved during refactor
Detection-evaluator SHALL preserve existing evaluation artifact schema (`metrics.json`, per-image outputs, match artifacts where enabled) and existing metric naming conventions during internal refactor.

#### Scenario: Refactored evaluator produces schema-compatible outputs
- **GIVEN** the same evaluator inputs and settings
- **WHEN** evaluation runs before and after refactor
- **THEN** output artifact schema and stable metric key names remain compatible for downstream consumers.

### Requirement: Evaluator strict-parse behavior is config-driven and bounded
Evaluator ingestion strictness SHALL be controlled by `eval.strict_parse` (default `false`).
- `eval.strict_parse=true`: fail fast on first malformed/non-object JSONL record.
- `eval.strict_parse=false`: warn+skip malformed records deterministically with bounded diagnostics (`warn_limit=5`, `max_snippet_len=200`).

For overlapping helper-consolidation deltas, this change MUST stay consistent with `src-ambiguity-cleanup-2026-02-11` as the authoritative strict-parse helper contract.

#### Scenario: Strict and non-strict parse modes remain deterministic
- **GIVEN** one malformed JSONL record in evaluator input
- **WHEN** `eval.strict_parse=true`
- **THEN** evaluation fails immediately with explicit path+line diagnostics
- **AND WHEN** `eval.strict_parse=false`
- **THEN** evaluator emits bounded warnings and skips the malformed record deterministically.
