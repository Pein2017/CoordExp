# inference-engine Specification (Delta)

## MODIFIED Requirements

### Requirement: Inference preserves structured, machine-readable error observability
Inference output artifacts MUST preserve structured error reporting and machine-readable run-level counters.

Normative behavior:
- Per-sample error metadata MUST use stable codes and stage tags.
- Run summaries MUST include aggregated error counts by class/code.
- Logs alone MUST NOT be the only error signal when structured artifacts exist.

#### Scenario: Prediction parse failure is visible in artifacts
- **WHEN** a sample has malformed model output during prediction parsing
- **THEN** the emitted sample record includes structured error metadata
- **AND** run summary counters reflect the corresponding error code.

### Requirement: Operator-controlled input violations fail fast in inference/eval
Operator-controlled inference/eval input violations MUST terminate the run and MUST NOT be silently skipped.

Normative behavior:
- Input contract checks SHOULD run in preflight before generation/evaluation work.
- Violations (schema/JSONL/image/size/geometry contract failures) MUST terminate non-zero.
- Implementations MAY aggregate a bounded set of actionable diagnostics before raising.

#### Scenario: Missing required image metadata terminates inference
- **GIVEN** inference/eval input records
- **WHEN** a required field such as `width`/`height` is missing or invalid
- **THEN** inference terminates non-zero with actionable diagnostics
- **AND** processing does not continue by silently skipping that sample.

### Requirement: Model-output parse/validation failures are continue-but-observable
Prediction parse/validation failures caused by model-generated output MAY continue per sample, but MUST be observable.

Normative behavior:
- Parse/validation failures on produced `pred_text` MUST emit structured sample errors and increment run counters.
- Invalid predicted objects MAY be dropped for that sample; subsequent samples MAY continue.
- Continue-and-skip under this rule is limited to model-output consumer behavior and does not apply to operator input contracts.

#### Scenario: Invalid prediction text yields sample-scoped error and continue
- **GIVEN** generation produced `pred_text` for a sample
- **WHEN** parsing/validation of that `pred_text` fails
- **THEN** the sample record includes structured error metadata and error counters increment
- **AND** subsequent samples continue.

### Requirement: Unexpected internal exceptions and generation failures terminate the run
Unexpected internal exceptions during inference/eval, including generation failures that prevent usable `pred_text`, MUST terminate the run non-zero.

Normative behavior:
- Internal/runtime failures MUST NOT be converted into silent success outputs.
- Unexpected exceptions MAY be annotated with diagnostics before re-raise, but the run MUST terminate.

#### Scenario: Backend generation failure terminates inference
- **WHEN** generation backend fails for a sample before producing usable prediction text
- **THEN** inference terminates non-zero
- **AND** the failure is not converted into empty predictions with continued execution.
