## Context

See [proposal.md](proposal.md) for motivation. The active source already has a
strict `resume` config, candidate training-state publication/admission modules,
checkpoint callback choreography, inference-payload manifests, run lineage,
pre-model cache admission, and execution-provenance collection. Unit and
control-plane tests cover substantial parts of those surfaces. Stable specs and
canonical docs, however, still state categorically that training checkpoints do
not carry exact state, while the superseded broad change left its final
interruption and compatibility gates unchecked.

This change therefore starts from live source, current tests, and newly executed
receipts. The archived change is provenance for locating risks, not an authority
whose requirements or checked tasks can be copied forward. Source presence does
not make the candidate path an accepted exact-resume capability. The four delta
specs in this change are the intended contract only after their corresponding
tasks and gates pass and the deltas are synchronized.

## Goals / Non-Goals

**Goals:**

- Produce one evidence matrix mapping every proposed requirement to current
  source ownership, focused tests, executed evidence, and remaining gaps.
- Close only gaps necessary to make the opt-in exact-state sibling honest,
  atomic, independently ignorable by inference, and fail-closed for supported
  same-world-size optimizer-boundary continuation.
- Reconcile stable contract deltas and operator docs without changing existing
  inference semantics or the disabled-resume default.
- Preserve a historical-reader boundary that can interpret older artifacts
  without upgrading them to current exact-resume eligibility.
- Classify provider authority without expanding this change: stable
  `coordexp-swift-packing-forward` supports only explicit `synchronous` and
  `overlapped`; `legacy_fused` and the environment override are unsupported
  implementation residue whose later deletion belongs to the decomposition
  change.

**Non-Goals:**

- No changed-order packing-policy promotion, source-order replacement, or new
  cache materialization policy.
- No performance claim, throughput gate, speculative optimization, cache
  publication campaign, or production training launch.
- No telemetry/reporting enhancement, loss-objective change, RL composition,
  training-orchestration decomposition, dependency upgrade, or broad cleanup.
- No cross-world-size resume, mid-accumulation save/resume, automatic checkpoint
  pruning, or bitwise-equivalence promise across independent CUDA launches.

## Decisions

### 1. Reconcile from an explicit evidence matrix, not from the archive

The first implementation artifact will enumerate each delta requirement and
record: current source owner, current test owner, executable verification,
receipt or absence, and disposition (`already accepted`, `gap to close`, or
`remove from delta`). A requirement with neither live implementation nor a
bounded path to qualification is removed from this change rather than inferred
from the archived design.

The same Wave-0 authority pass records that the stable packing-forward contract
names only `synchronous` and `overlapped` provider modes. It MUST classify
`legacy_fused` and `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` as unsupported
live residue, MUST NOT copy either into a delta spec, and MUST hand that exact
classification plus the resulting implementation commit to the later
decomposition change.

Alternative considered: synchronize all six archived delta specs and finish
their old task list. Rejected because it would inherit unfinished packing,
efficiency, interruption, and final-gate claims into one authority surface.

#### Qualified contract correction: exact mode may be publish-only

Task 3 initially encoded exact mode without a checkpoint path as an invalid
configuration. Fixed-target review showed that this shape is the live and
required publish-only control/parent branch: exact mode enables training-state
publication, while a non-null checkpoint path independently selects restore.
The uninterrupted control and interrupted parent must create committed exact
boundaries before a child checkpoint exists.

Therefore no production-source change is authorized. The config delta is
narrowed to reject the incompatible shape the live validator owns (disabled
mode with a path), and a positive test MUST preserve exact mode plus a null
path. This is not a silent downgrade: the resolved config and active-profile
identity retain exact mode, and runtime publishes exact state without entering
the restore branch. Introducing a distinct publish-only mode or authored-intent
signal is a separate user-owned compatibility design and is outside this
change.

### 2. Keep two payload types with one ordered checkpoint transaction

The inference payload remains the first independently authenticated payload in
the step directory. When exact state is enabled, the training-state publisher
runs as the typed second phase; aliases and the completed checkpoint event are
committed only after it succeeds. When disabled, the second phase is absent,
not an empty manifest or compatibility placeholder. Inference readers consume
the explicit inference manifest and configured adapter/delta paths and ignore
the sibling directory.

Alternative considered: extend the inference manifest with optimizer and
cursor state. Rejected because it couples inference compatibility to mutable
training machinery and makes disabled mode indistinguishable from incomplete
exact state.

### 3. Define exactness at the optimizer-step boundary only

The proposed supported save boundary, conditional on qualification, is a completed optimizer step after all planned
micro-steps and before the next pack is consumed. Admission requires the same
world size and compatible rank mapping, strict replay policy, cache/policy
identity, topology, trainable inventory, and dependency identity. Unsupported
world-size or accumulation-position requests fail before restore.

Qualification requires a matched branch pair from one authenticated boundary:
an uninterrupted control consumes the next planned pack and applies its next
optimizer update, while a child restores the boundary, consumes that same pack,
and applies its first post-resume update. Before either branch's forward, the
receipt MUST compare next-input/pack identity and the declared trainable,
optimizer, scheduler, scaler, per-rank RNG, and cursor state. It MUST then
compare objective/loss fields and resulting trainable parameters after the
corresponding update under the declared exact policy. Publication, admission,
or restored-field inspection without this first forward and update is not an
exact-continuation qualification.

Alternative considered: serialize pending gradients for mid-accumulation
resume. Rejected because it enlarges state and distributed failure surfaces
without a current repository need or completed qualification.

### 4. Qualify atomic publication with failure-shaped execution

Leaf serialization tests are insufficient for the unresolved risk. Acceptance
will include deterministic multi-rank control-plane tests plus the smallest
production-shaped distributed probe that exercises a successful matched
uninterrupted-versus-resumed branch pair,
one-rank contribution failure, malformed/incomplete rank inventory, and an
interruption before commit. The proof must show no resumable manifest/event or
selector aliases are published on failure and that surviving ranks converge
without hanging. The independently committed inference payload may remain, but
the reader must classify it as inference-only.

Alternative considered: reason from atomic rename helpers and unit tests alone.
Rejected because the prior unchecked gate was specifically about distributed
failure and interruption behavior.

This probe is acceptance evidence, not implicit launch authority. Immediately
before execution, the implementer MUST present a fresh launch packet and obtain
fresh user authorization bound to the exact commit, config, artifact root, and
commands. The packet fixes `world_size=2`, uses at most two GPUs, permits at most
three bounded semantic arms: one success arm comprising the matched control and
resumed branches, one rank-failure arm, and one interruption arm. Each success
branch executes exactly one next forward and at most one applied optimizer
update; each failure-shaped arm executes no more than one forward and one
applied update per rank. The packet declares numeric, config-derived ceilings
for model forwards and collectives per rank and per branch/arm, plus per-arm and
total wall time, per-rank CPU RSS and GPU-memory high-water marks, new artifact
bytes across both success branches and the failure-shaped arms, and required
free disk. Missing bounds, changed commands/commit, an occupied artifact
target, or any exceeded bound stops the launch and requires a new packet rather
than an automatic retry.

### 5. Make historical interpretation conservative and explicit

One reader/admission matrix will cover: current committed exact state, current
inference-only payload, older adapter-plus-delta payload with extra metadata,
partial/current staging state, and unknown historical resume-like files.
Inference accepts compatible explicit payloads without reading the sibling;
exact admission accepts only the current committed schema. No filename or
directory-shape heuristic upgrades a historical artifact.

Alternative considered: add a migration reader for archived state formats.
Rejected because no such format is a current accepted contract and migration
would create behavior rather than reconcile it.

### 6. Treat cache and provenance deltas as audit-constrained reconciliation

Cache identity/admission and executed-environment provenance enter the final
stable contract only after the evidence matrix confirms current source,
focused tests, and an executable receipt. This change may repair a demonstrated
contract gap but will not redesign cache ownership or introduce new identity
dimensions. Cache verification remains pre-model and immutable-target;
provenance remains bounded and non-secret.

Alternative considered: move these surfaces to the later orchestration
decomposition change. Rejected for the exact-resume compatibility fields that
are already mandatory, but unrelated cache refactoring remains deferred.

### 7. Reconcile docs only after behavior gates pass

Canonical docs will describe the inference payload and optional exact-state
sibling as separate surfaces, list the supported boundary and non-goals, and
link to the owning stable specs. They will not embed receipt-specific details
or repeat full schemas. A final stable-vs-delta and docs-vs-source scan must
show no remaining categorical claim that exact state is never written.

Alternative considered: update docs first to match current source. Rejected
because doing so would promote the capability before the unresolved atomic
publication gate is closed.

## Risks / Trade-offs

- **[Risk] Current source covers success but not a real distributed failure
  sequence.** → Keep the delta conditional until the failure-shaped probe
  passes; remove or narrow the requirement if a safe bounded repair cannot be
  made.
- **[Risk] A committed inference payload remains after exact-state failure and
  is mistaken for resumable state.** → Require typed admission and negative
  historical-reader tests; selectors and completed exact events remain gated
  on phase two.
- **[Risk] Strict replay language overclaims numerical identity.** → Require
  the matched first post-resume update, compare the exact fields named by the
  policy with its declared comparison rule, and make no broader cross-launch
  bitwise claim.
- **[Risk] Cache/provenance requirements import unfinished archive scope.** →
  Require per-requirement source, test, and receipt evidence and remove
  unsupported language before sync.
- **[Risk] A planning approval is mistaken for permission to consume GPUs or
  publish probe artifacts.** → Require the fresh, commit-bound quantitative
  launch packet in Decision 4; no inherited or blanket approval satisfies it.
- **[Risk] Concurrent user-owned documentation edits overlap reconciliation.**
  → Inspect ownership and diff at apply time, patch only exact stale claims,
  and never reset or overwrite unrelated changes.

## Migration Plan

1. Freeze the evidence matrix and identify any delta language unsupported by
   current source/tests.
2. Add or tighten focused tests first, then make the minimum source repair for
   demonstrated gaps.
3. Freeze the exact probe commands and quantitative bounds, pass the independent
   pre-cost/distributed-qualification audit, obtain fresh commit-bound launch
   authorization, then execute the bounded matched-success/failure/interruption
   probe and publish its immutable scope-labeled receipt under this change.
4. Reconcile canonical docs and run the stable-vs-delta conflict scan.
5. Run focused, full relevant, and strict OpenSpec validation, then obtain the
   single independent final audit with both code-quality and contract lenses.
6. Sync/archive only after every task and gate is complete. If a gate fails,
   keep the change active or archive it explicitly incomplete; do not edit
   stable specs or claim exact-resume support.

Rollback before archive is deletion/reversion of only this change's source,
test, and docs edits; the compatibility behavior remains `resume.mode:
disabled` with inference-only checkpoint publication. Published evidence is
preserved with its failure disposition rather than rewritten.
