## 1. Contract-First Failure Tests

- [ ] 1.1 Add strict canonical-value tests that accept an explicit nested
  receipt and reject live callbacks, tensors, paths, sets, non-string mapping
  keys, NaN, and infinities before any file or backend work.
- [ ] 1.2 Add execution-plan tests for atomic create/reload, complete identity
  and plan fingerprints, ordered unique work-item identifiers, occupied-root
  rejection, and injected failures at temporary-file sync, atomic publication,
  and containing-directory sync.
- [ ] 1.3 Add work-item tests for serialize-before-publish, independent durable
  records, unique sequence and work-item identities, injected finalizer/process
  failure, and unchanged prior evidence after a bad later payload.
- [ ] 1.4 Add process-attempt and continuation tests for unique attempt
  identities, mechanics-only failure receipts, exact-identity completed-item
  discovery, rejection after any execution-identity drift, and refusal to
  terminalize an incomplete plan.
- [ ] 1.5 Add inference artifact tests for strict opaque context preflight,
  digest-bound success/failure publication, no-context compatibility, shard
  byte-source propagation, plan-only journal references, and merge rejection on
  context bytes, file digest, value fingerprint, locator, or reference drift.
- [ ] 1.6 Add a residue test that rejects stable journal or inference owners
  from naming or interpreting probe arms, cohorts, conditioning,
  interventions, estimands, matching, unmatched outcomes, thresholds,
  scientific validity, claims, or stop rules.

## 2. Shared Strict Artifact Values

- [ ] 2.1 Implement one `src/artifacts/` owner for recursive strict JSON-value
  validation, canonical bytes, SHA-256 fingerprints, strict reload, and staged
  atomic publication with typed artifact-contract errors.
- [ ] 2.2 Replace the training run writer's private strict serializer with the
  shared owner and prove existing valid logging and atomic JSON bytes remain
  compatible.
- [ ] 2.3 Replace inference artifact `default=str` conversion with the shared
  strict owner; add explicit semantic projections for any legitimate current
  non-JSON internal values and delete the permissive conversion path.
- [ ] 2.4 Run focused training- and inference-artifact tests to verify invalid
  values fail before publication and all pre-change valid fixtures remain
  readable and semantically unchanged.

## 3. Execution Evidence Journal

- [ ] 3.1 Implement immutable journal creation, exclusive single-writer
  ownership, serialize-to-temp, temporary-file `fsync`, non-replacing atomic
  publication, containing-directory `fsync`, immediate reload validation, and
  occupied-root rejection.
- [ ] 3.2 Implement unique process-attempt receipts and atomic per-work-item
  record publication through the same crash-consistent sequence, with
  execution, plan, sequence, attempt, payload, and record-digest binding.
- [ ] 3.3 Implement full journal reload validation, exact completed-item
  discovery, temporary-file diagnostics, exact-identity continuation checks,
  and terminal completion bound to the ordered record-digest aggregate.
- [ ] 3.4 Add one CPU interruption fixture that accepts several records,
  injects a terminal-finalizer failure, starts a fresh process attempt, reloads
  prior records, completes the missing plan, and verifies the terminal receipt.
- [ ] 3.5 Measure plan creation, per-record sync, reload, and terminalization for
  representative hundreds and low thousands of records; record the wall-clock,
  byte, and inode receipt and stop for redesign if persistence is a material
  primary-observation bottleneck.

## 4. Inference Context Integration

- [ ] 4.1 Add an optional immutable inference execution-context input containing
  prepared strict context bytes/fingerprint and an optional journal plan
  reference; materialize one controller-owned context file and validate it
  before backend or worker construction.
- [ ] 4.2 Publish one digest-bound `execution_context.json` sidecar in both
  successful and terminal-failure artifact families and bind its relative path
  and SHA-256 from summary and manifest.
- [ ] 4.3 Propagate the controller-owned context locator, exact file bytes and
  SHA-256, canonical-value fingerprint, and journal plan reference through
  worker launch contracts and rank-local artifacts; require exact agreement
  during strict merge and top-level terminal publication.
- [ ] 4.4 Preserve no-context single-process and data-parallel behavior, remove
  obsolete pass-through knowledge, and verify existing evaluator consumers need
  no adaptation.

## 5. Mechanics Acceptance Gate

- [ ] 5.1 Run the exact CPU failure-shape gate: a live callback in planned
  metadata must fail before work, while its explicit receipt mapping must pass
  plan write/read and interruption recovery.
- [ ] 5.2 Run focused `tests/artifacts/` and `tests/inference/` suites plus
  formatting, lint, type, and whitespace checks for every touched source and
  test surface.
- [ ] 5.3 Run strict OpenSpec validation and the research-vocabulary residue
  check; resolve all contract contradictions and any priority-zero or
  priority-one finding before runtime admission.
- [ ] 5.4 Obtain separate independent engineering-standards and
  research-versus-infrastructure intent audits on the fixed implementation and
  test receipts.
- [ ] 5.5 After explicit runtime authorization, run one minimal real inference
  case through plan preflight, backend execution, work-item publication,
  terminal publication, context sidecar, and fresh-process readback. Label the
  receipt mechanics-only and stop before support expansion, cohort execution,
  or scientific interpretation.
