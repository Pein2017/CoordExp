## ADDED Requirements

### Requirement: Clean-break refactors preserve retained semantics instead of historical runtime compatibility

Runtime-critical refactors under the DetectionScene clean-break program SHALL
protect retained canonical detection semantics while allowing historical runtime
compatibility to be dropped after an archive checkpoint.

Normative behavior:

- retained canonical surfaces MUST preserve geometry, coordinate, template,
  token-supervision, Stage-2 correction, eval, metric, provenance, and config
  semantics unless a separate semantic-change decision approves otherwise;
- historical execution paths classified as delete or quarantine MUST NOT require
  compatibility adapters in the new stack;
- old public names MUST NOT be kept solely to preserve historical imports,
  scripts, or configs after the archive checkpoint;
- deletion/search gates MUST classify remaining historical names rather than
  treating all old-name matches as evidence of active support.

#### Scenario: Compatibility cleanup is not blocked by retired configs

- **GIVEN** a historical config family is classified as delete after archive
  checkpoint
- **WHEN** the clean-break refactor removes it from active configs, docs, and
  tests
- **THEN** runtime-architecture review does not require a live compatibility
  path for that family
- **AND** retained canonical detection semantics still have targeted parity or
  contract checks.
