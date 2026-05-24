## ADDED Requirements

### Requirement: vLLM sync preparation failures are observable and fail-fast

CoordExp SHALL treat vLLM weight-sync preparation as a correctness-critical
path. Unexpected failures during snapshot preparation, validation, or local
broadcast setup MUST terminate the run rather than being logged and ignored.

Normative behavior:

- If materialization fails on the learner side, vLLM sync MUST abort before
  broadcasting an invalid tensor bucket.
- Under DDP, sync failures MUST be propagated rank-symmetrically so non-owner
  ranks do not hang at later collectives.
- Error messages MUST include enough context to identify the sync stage and the
  unsupported or malformed key/state.
- Best-effort filtering is not allowed for active adapter state that affects
  rollout correctness.

#### Scenario: Materialization failure terminates all learner ranks

- **GIVEN** Stage-2 training runs with multiple learner ranks
- **AND** rank 0 cannot materialize active token-row adapter state for vLLM
  full-sync
- **WHEN** sync preparation runs
- **THEN** all learner ranks terminate non-zero without hanging
- **AND** the error identifies vLLM sync materialization as the failing stage.

### Requirement: Known-invalid vLLM sync snapshots are rejected before server load

CoordExp MUST reject known-invalid native vLLM full-sync snapshots before
server-side `load_weights()` is invoked. Stronger acknowledgement for arbitrary
worker-side load failures remains a future server protocol improvement.

Normative behavior:

- Learner-side validation MUST remove known unsupported key families before
  server load.
- Until stronger server acknowledgements exist, known unsupported key families
  MUST be caught by learner-side validation before sending.
- Server-side sync/load failures SHOULD be surfaced through acknowledgement or
  health checks where practical, but Phase 1 validity does not depend on a new
  server acknowledgement protocol.

#### Scenario: Unknown key is caught before vLLM load

- **WHEN** a vLLM-bound sync snapshot still contains a key family known to be
  unsupported by native vLLM
- **THEN** learner-side validation fails before the server update request
- **AND** the run does not rely on fire-and-forget server logs to discover the
  failure later.
