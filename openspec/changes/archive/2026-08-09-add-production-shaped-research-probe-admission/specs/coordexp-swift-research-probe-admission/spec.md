## Purpose

Provide a mechanics-only gate that binds the exact local execution surface and
closes CPU plus bounded vertical evidence before a costly research-probe launch
can be presented as production-path ready.

## ADDED Requirements

### Requirement: Admission inputs use typed immutable bindings
The capability SHALL capture one immutable binding manifest before creating an
admission root.  The manifest MUST contain an ordered unique set of named
bindings whose kinds are limited to regular file, directory tree, resolved data
file, absolute executable, or strict JSON value.  Every binding MUST carry its
kind-specific identity and the manifest MUST carry one canonical content
fingerprint.

Regular files and executables MUST be non-symlink regular files bound by
absolute resolved path, byte count, and full SHA-256.  An executable MUST also
be executable and MUST be supplied as an absolute path rather than resolved
implicitly through `PATH`.  A directory tree MUST reject symlinks and
non-regular descendants, inventory every regular file by full POSIX relative
path, byte count, and full SHA-256, sort by relative path, and bind the complete
inventory digest.  Short digest prefixes MUST NOT be identity keys.  Strict
values MUST use the accepted recursive finite canonical-JSON algebra.

Unsupported kinds, duplicate names, missing paths, wrong path kinds,
symlinks, unreadable inputs, non-executable executables, non-finite values, live
callbacks, tensors, handles, `Path` objects, bytes, sets, or other non-JSON
values MUST fail before any admission root is created.

#### Scenario: Heterogeneous binding manifest is captured
- **WHEN** a caller supplies valid file, directory, executable, resolved-data,
  and strict-value bindings with unique names
- **THEN** the capability returns one deterministic strict manifest whose full
  identities and content fingerprint revalidate against the live inputs

#### Scenario: Live callback is supplied as identity
- **WHEN** any binding or admission context contains a live callback or another
  unsupported runtime object
- **THEN** capture fails with a typed admission error and no admission root or
  partial evidence file exists

#### Scenario: Directory iteration order differs
- **WHEN** the same non-symlink directory files are encountered in different
  filesystem or creation orders
- **THEN** the sorted full-path inventory and complete directory digest are
  identical without truncating or overwriting colliding digest prefixes

### Requirement: Resolved data and executable identities match consumer use
A resolved-data-file binding SHALL preserve the caller's declared path and the
absolute base directory used by the production consumer.  Relative declarations
MUST resolve against that base directory, while absolute declarations MUST
remain explicit.  The resulting target MUST satisfy the regular-file contract.
Changing the declaration, base directory, resolved path, or target bytes MUST
change or invalidate the binding even when the caller's current working
directory would find another file.

An absolute-executable binding SHALL identify the exact file that the consumer
will invoke.  A bare command name, symlink, missing target, non-regular target,
or non-executable target MUST be rejected rather than searched or substituted.

#### Scenario: Relative data moved with its declaring artifact
- **WHEN** an unchanged relative data string is checked under a different base
  directory where its target is absent or differs
- **THEN** admission fails before production work instead of resolving it from
  the current working directory or an earlier source location

#### Scenario: Launcher basename names a shim
- **WHEN** a caller supplies only `env` or another non-absolute executable name
- **THEN** admission rejects the executable binding and requires the exact
  absolute executable path and bytes

### Requirement: Reserved output paths are absent before work
An admission plan SHALL name every root reserved for CPU evidence, vertical
execution, finalization, or downstream validation.  Before publishing the plan,
the capability MUST require each reserved path to be absolute, unique, absent,
and non-symlink.  It MUST reject an occupied, aliased, duplicate, or relative
reserved path without deleting, replacing, or cleaning it.

Reserved paths are pre-work facts rather than immutable input bindings.  Once a
stage executes, its accepted evidence MUST bind the exact output files it
created; later validation MUST NOT incorrectly require the reserved path to
remain absent.

#### Scenario: Fresh execution roots are reserved
- **WHEN** every declared output path is absolute, unique, and absent
- **THEN** the immutable admission plan records those paths before either stage
  is allowed to publish evidence

#### Scenario: Prior root is empty but present
- **WHEN** a declared output root already exists, even if it is empty
- **THEN** admission fails without reusing, deleting, or writing inside that
  root

### Requirement: Admission has exactly two ordered evidence stages
The capability SHALL use the accepted execution-evidence journal to require
exactly `cpu_preflight` followed by `vertical_smoke` under one immutable
admission and binding identity.  It MUST validate the complete binding manifest
against live inputs immediately before accepting either stage.  It MUST reject
stage reordering, duplication, foreign binding fingerprints, unplanned stage
identifiers, and continuation under changed inputs.

An accepted CPU record MUST remain independently durable if the vertical stage
or final admission publication fails.  A later process MAY continue only under
the exact admission identity and with a separately identified attempt.  The
capability MUST NOT automatically retry, continue, launch, or replace an
accepted stage.

#### Scenario: Vertical process exits after CPU admission
- **WHEN** the CPU record is durable and the vertical process exits before a
  vertical record is accepted
- **THEN** the journal remains non-terminal with the CPU record unchanged and
  exposes the missing vertical stage for explicitly authorized continuation

#### Scenario: Input changes between stages
- **WHEN** a bound source, runtime tree, config, model payload, executable, or
  strict identity changes after CPU preflight
- **THEN** the vertical stage is rejected before its evidence is accepted and a
  new admission identity is required

### Requirement: Stage evidence closes the production path mechanically
Each stage record SHALL be a strict mechanics envelope that binds its stage
identifier, admission and binding fingerprints, exact producer and validator
binding names, complete output-file identities, fixed mechanics assertions, and
an explicit claim boundary.  Every referenced producer and validator MUST exist
in the immutable binding manifest, and every output file MUST be captured and
revalidated as a regular-file identity before the record is accepted.

The CPU preflight assertions MUST establish that the exact production
entrypoint was resolved, the current consumer input/projection validator ran,
a model-free finalizer fixture ran, and that fixture's declared downstream
validator accepted it, with no model loaded and no GPU used.  The
vertical-smoke assertions MUST establish that the bound production entrypoint
executed, the declared model/runtime loaded, at least one work item became
independently durable, the terminal/finalizer path completed, and the declared
bounded mechanics validator accepted its owned output.  Both stages MUST state
that they made no scientific interpretation.  A bounded validator MUST NOT be
presented as closure of a fuller scientific merger denominator that the smoke
did not execute.

The capability SHALL validate these mechanics assertions and exact artifact
bindings only.  It MUST NOT infer whether the consumer selected the intended
cohort, intervention, arm, owner, outcome, estimand, threshold, claim, or stop
rule.

#### Scenario: Helper-only test claims production readiness
- **WHEN** stage evidence lacks the exact production producer, current validator
  bindings, required mechanics assertions, or bound output files
- **THEN** the stage is rejected even if a helper or fake-model test passed

#### Scenario: Scientific outcome is stored in an opaque output
- **WHEN** a bound output contains caller-owned unmatched, STOP, invalid, null,
  or effect fields
- **THEN** admission binds the output bytes but neither interprets those fields
  nor changes the stage's mechanics-only claim boundary

### Requirement: Final admission receipt is immutable and non-authorizing
The capability SHALL publish a canonical write-once admission receipt only
after both stage records and the journal terminal validate.  The receipt MUST
bind the admission plan, binding manifest, stage payload fingerprints, journal
terminal identity, output-file identities, and claim boundary.  A publication
failure MUST leave the terminal journal and both stage records intact for exact
reload and idempotent finalization.

The final status MUST be `mechanically_admitted`.  That status proves only the
declared production-path mechanics and MUST NOT authorize bulk execution,
training, automatic continuation, architecture or decoder changes, scientific
promotion, publication, or deployment.

#### Scenario: Final receipt publication fails late
- **WHEN** both stages are durable and journal terminal validation succeeds but
  final receipt publication fails
- **THEN** no accepted stage is lost or rewritten and an exact reload can
  publish only the same canonical receipt bytes

#### Scenario: Mechanics admission is presented as a model result
- **WHEN** a caller reads a valid `mechanically_admitted` receipt
- **THEN** the receipt explicitly excludes scientific validity, model quality,
  launch authorization, and downstream promotion

### Requirement: Two real consumers cross the same admission seam
Acceptance SHALL exercise the shared capability through both the current
natural-boundary support path and the current K10-H20 crossover path.  Each
consumer adapter MUST reuse the same typed binding and CPU-stage interface
while retaining its own plan, projection, finalizer, validator, scientific
vocabulary, and output schema.  The selected bounded natural-boundary support
adapter MUST additionally cross the vertical stage and final admission
interface.

The acceptance gate SHALL include deterministic CPU tests for every binding
kind and failure class, a production-path CPU round trip for both consumers,
and one bounded single-GPU smoke through an existing production worker.  The GPU
smoke MUST write a fresh root, bind the exact model/config/runtime/device and
consumer identities, publish at least one durable work item, reach the current
bounded terminal/finalizer and its mechanics validator, and remain
mechanics-only.  It MUST NOT claim that the natural-boundary legacy merger was
validated unless that merger's complete declared shard and context denominator
was actually supplied.

The active `research-probes` worktree and every sealed historical root MUST
remain immutable.  Consumer examples MAY read and digest-bind that worktree but
MUST NOT edit, stage, commit, merge, cherry-pick, push, or rewrite it.

#### Scenario: Both consumers pass through one interface
- **WHEN** support and crossover adapters submit their distinct CPU evidence
  and the selected support adapter also submits bounded vertical evidence
  through the shared admission capability
- **THEN** both produce contract-valid mechanics stage receipts and the support
  path produces a final admission receipt without adding either consumer's
  scientific schema to stable infrastructure

#### Scenario: Consumer semantics differ
- **WHEN** the two consumers use different plans, projections, endpoints, and
  outcome fields
- **THEN** those differences remain visible and caller-owned while their shared
  path, identity, durability, and evidence-closure mechanics are validated by
  the same admission owner
