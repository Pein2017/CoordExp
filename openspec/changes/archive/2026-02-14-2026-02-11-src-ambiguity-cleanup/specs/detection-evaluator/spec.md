# detection-evaluator Spec Delta

This is a delta spec for change `2026-02-11-src-ambiguity-cleanup`.

## ADDED Requirements

This change is the authoritative helper-contract delta for evaluator ingest strictness defaults and shared diagnostic helper behavior across active overlaps.

### Requirement: Semantic encoder implementation is shared across training and evaluator
Semantic description normalization and sentence-embedding computation used for evaluator description mapping and Stage-2 semantic gating/monitoring SHALL use the same canonical implementation (normalization rules, mean pooling, and L2 normalization).
The evaluator MUST NOT carry a separate parallel encoder implementation that could drift.

#### Scenario: Normalization rules are consistent across surfaces
- **GIVEN** two descriptions that differ only by punctuation/whitespace (e.g., `Armchair/Chair (Wood)` and `armchair chair wood`)
- **WHEN** training gating and evaluation normalize descriptions
- **THEN** they produce the same normalized description string.

### Requirement: Evaluation JSONL ingestion diagnostics are centralized
When the evaluator loads `gt_vs_pred.jsonl`, JSON parsing diagnostics (path + 1-based line number + clipped snippet) SHALL be implemented via a shared helper so parsing/warning behavior is consistent.
Strict mode MUST fail fast on malformed records; non-strict mode MUST warn a bounded number of times and skip malformed records deterministically.
The governing config key is `eval.strict_parse` (default `false`).
Bounded diagnostics defaults are normative: `warn_limit=5`, `max_snippet_len=200`.

#### Scenario: Malformed JSONL line is reported with path and line number
- **GIVEN** a `gt_vs_pred.jsonl` containing a malformed JSON line at line 3
- **WHEN** the evaluator loads records in non-strict mode
- **THEN** it emits a warning containing the file path and `:3`
- **AND** it skips the malformed record.

#### Scenario: Strict parse mode fails on first malformed JSONL record
- **GIVEN** a `gt_vs_pred.jsonl` containing malformed JSON
- **WHEN** the evaluator loads records with `eval.strict_parse=true`
- **THEN** it fails immediately with explicit path+line diagnostics
- **AND** it does not continue with partial record ingestion.

### Requirement: Image-path resolution helper is shared
Evaluator surfaces that resolve image paths (e.g., overlay rendering) SHALL delegate to shared image-path resolution helpers rather than implementing ad-hoc base-dir logic.

#### Scenario: Relative image path resolves deterministically
- **GIVEN** an image field `images/foo.jpg` and an explicit base directory
- **WHEN** the evaluator resolves the image path via the shared helper
- **THEN** it deterministically resolves to `<base_dir>/images/foo.jpg` (absolute path).
