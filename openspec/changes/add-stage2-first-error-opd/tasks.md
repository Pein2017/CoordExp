## 1. Contract

- [x] 1.1 Create OpenSpec change for first-error OPD semantics.
- [x] 1.2 Specify live-token mismatch handling and ambiguity-scoped multiple positives.
- [x] 1.3 Specify required residual-set diagnostics.

## 2. Red Tests

- [x] 2.1 Add tests showing wrong live tokens are not added to residual valid sets.
- [x] 2.2 Add tests showing explicit mismatch provenance is required by target IR validation.
- [x] 2.3 Add tests for residual-set loss metrics: ambiguous targets, strict targets, and target-token mismatches.

## 3. Implementation

- [x] 3.1 Change residual-set IR adapter so oracle valid actions define positives.
- [x] 3.2 Allow explicit first-error target-token mismatches in shared IR validation.
- [x] 3.3 Add residual-set diagnostics for ambiguity, strict CE collapse, coordinate ambiguity, and mismatch correction.
- [x] 3.4 Preserve singleton hard CE behavior and existing no-role/shape fail-fast behavior.
- [x] 3.5 Align legacy teacher-forcing objective validation with explicit first-error mismatch provenance.

## 4. Verification

- [x] 4.1 Run targeted residual-set and teacher-forcing tests.
- [x] 4.2 Run Stage-2 config contract tests covering the updated objective registry.
- [x] 4.3 Inspect the running 8-GPU infra run only for status; do not terminate it.
- [x] 4.4 Run legacy teacher-forcing objective runner mismatch-provenance test.
