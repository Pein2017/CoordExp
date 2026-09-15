## Why

Qualification source identity omits project-local semantic dependencies such as the template renderer, and its child supervisor can wait indefinitely. A renderer snapshot also remains stale after the approved package identity rename.

## What Changes

- Cover qualification's project-local semantic source dependencies with the existing source identity and guard coverage against drift; reject stale receipts through existing admission.
- Bound qualification child execution with an explicit configurable deadline, terminate owned process groups on timeout, and fail closed. Bound external GPU census commands used by that supervision path.
- Update only the stale template identity values in the golden fixture after confirming rendered content is unchanged.
- Preserve receipt schema, path binding, dynamic HF authority, numerical acceptance and training behavior. No GPU/model launch, new quality receipt, generic manifest framework, DDP degradation or unrelated cleanup.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `infra-base`: specify source-drift rejection and bounded qualification supervision within qualified inference backends.

## Impact

Qualification producer, run CLI, their tests and operating documentation; renderer fixture. Broader source coverage intentionally invalidates old qualification identities and requires requalification before production admission. This task verifies CPU behavior only and does not regenerate model qualification artifacts.
