## Context

See `proposal.md` for motivation and
`specs/coordexp-infras-research-probe-admission/spec.md` for the observable
contract.  The stable `ExecutionEvidenceJournal` already owns strict durable
plan, attempt, record, continuation, and terminal mechanics.  The current
support adapter adds consumer-specific model/source bindings and a supervised
worker, while the sibling crossover path separately implements file and
directory inventories, projection checks, reserved roots, runtime identity,
write-once finalization, and a post-execution successor receipt.

The new owner must deepen only the shared admission concept.  It must not absorb
support scheduling, crossover endpoint semantics, model-forward behavior,
actuator validity, process supervision, finalization algorithms, or research
authority.  The sibling consumer is dirty and separately owned, so adoption is
proved through exact read-only bindings and adapters in this worktree rather
than in-place edits.

## Goals / Non-Goals

**Goals:**

- Give typed byte/path identities and two-stage admission one stable owner.
- Fail all strict-value and immutable-input defects before costly work or
  admission-root creation.
- Make a late vertical or final-receipt failure preserve the accepted CPU and
  vertical evidence that already exists.
- Force real consumer validators and finalizer surfaces into acceptance rather
  than accepting helper-only green tests.
- Keep the public surface small enough that both real consumers can cross it
  without exposing their scientific schemas.

**Non-Goals:**

- Running subprocesses, allocating GPUs, supervising PIDs, retrying, scheduling,
  or authorizing continuation or launch.
- Deciding whether a probe's operator, cohort, outcome, denominator, estimand,
  claim, or stop rule is scientifically valid.
- Generalizing the support LPT scheduler or crossover successor policy.
- Replacing execution-model, adapter, embedding-delta, config, or data semantic
  validators.  Admission binds their current outputs and producer identities.
- Modifying the active research unit or treating the mechanics smoke as a model
  result.

## Decisions

### 1. Own one admission dossier, not a general probe runner

Three interfaces were compared.

**Minimal binding utility.**  A handful of `bind_file` and `bind_directory`
functions would remove hashing duplication, but callers would still implement
stage order, late-failure preservation, final closure, and production-path
evidence independently.  It is too shallow to solve the observed failure
cluster.

**Flexible callback runner.**  A framework that invokes arbitrary preflight,
model, finalizer, and validator callbacks could enforce order directly.  It
would also become a scheduler/control plane, expose framework objects, make
callback identity difficult to attest, and let a helper callback masquerade as
the production path.  It is rejected as speculative and too authoritative.

**Contract-first admission dossier (selected).**  One deep artifact module
captures typed immutable inputs, owns a nested evidence journal with two fixed
stage identifiers, validates exact producer/validator/output evidence, and
materializes one compact final receipt.  Consumer adapters execute their own
real paths and submit evidence through this interface.  This keeps execution
and science visible while concentrating identity, failure, and closure rules.

Deleting the selected module would spread typed path rules, inventory order,
stage closure, continuation identity, and final admission publication back into
both consumers, so the seam has real leverage.

### 2. Represent caller intent with typed requests and persist only strict values

The public module will expose immutable request/value types plus one admission
object:

```python
manifest = capture_binding_manifest(
    [
        RegularFileBinding("consumer", consumer_path),
        DirectoryTreeBinding("runtime", runtime_root),
        ResolvedDataFileBinding("image", declared_image, data_base),
        AbsoluteExecutableBinding("python", python_path),
        StrictValueBinding("runtime_policy", runtime_policy),
    ]
)

with ResearchProbeAdmission.create(
    root=fresh_admission_root,
    admission_id=admission_id,
    bindings=manifest,
    reserved_output_paths=reserved_paths,
    context=caller_owned_mechanics_context,
) as admission:
    attempt = admission.start_attempt()
    admission.append_stage(
        stage="cpu_preflight",
        evidence=cpu_evidence,
        attempt_id=attempt,
    )
    admission.append_stage(
        stage="vertical_smoke",
        evidence=vertical_evidence,
        attempt_id=attempt,
    )
    receipt_path = admission.finalize()
```

Request types may contain local `Path` values while being constructed, but the
capture boundary immediately validates and converts them to strict canonical
mappings before creating a root.  Persistent plans and stage evidence never
contain Python paths, callbacks, tensors, model objects, handles, exceptions,
or implicit stringification.

Bindings are immutable inputs.  Reserved output paths are a separate pre-work
section because they are expected to become present during a stage and cannot
be live-revalidated as absent afterward.

### 3. Use full file identities and one deterministic tree algorithm

Regular-file identity is `(absolute resolved path, byte_count, sha256)` after
rejecting symlinks and non-files.  Directory identity inventories every
non-symlink regular descendant as `(relative_path, byte_count, sha256)`, sorts
by full POSIX relative path, and hashes the complete list.  No short hash prefix
is an identity or dictionary key.

Resolved data identity records the declaration, absolute base directory, and
resulting file identity.  Relative paths are interpreted against the supplied
base, not cwd; absolute declarations remain explicit.  Admission does not add a
containment rule because existing data contracts may intentionally refer to a
sibling directory.  The consumer's data validator remains authoritative for
whether such a declaration is allowed.

Executable identity requires an absolute non-symlink regular executable and
binds its bytes.  The admission layer never calls `which` and never stores a
basename as executable identity.

**Alternative considered: reuse each consumer's existing inventory.** Rejected
because the two implementations already disagreed on ordering and would retain
two owners for the same mechanics.

### 4. Reuse a nested execution journal for stage durability

The admission root contains:

```text
bindings.json
journal/plan.json
journal/records/...
journal/attempts/...
journal/terminal.json
admission.json
```

All request normalization, strict-value checks, live input capture, and
reserved-path checks complete in memory before the admission root is created.
The root then publishes `bindings.json` exclusively and creates a nested
`ExecutionEvidenceJournal` whose exact expected work items are
`cpu_preflight` and `vertical_smoke`.

Opening an unfinished admission revalidates `bindings.json`, every live input,
reserved-path plan identity, and the journal continuation identity.  Reserved
paths are not required to remain absent after creation.  Each stage append
revalidates immutable inputs, validates its fixed mechanics envelope, captures
the stage's output files, and delegates durable append to the journal.

The existing journal owns attempts and exact continuation.  The new module does
not change journal schema version 1 or expose private record filenames.

### 5. Make evidence closure explicit but keep truth ownership with adapters

The stage envelope has a fixed mechanics schema:

- stage identifier and binding/admission fingerprints;
- producer and validator binding names;
- output regular-file identities;
- required boolean/count assertions for that stage;
- strict caller detail for diagnostics;
- `scientific_interpretation=false` and an explicit mechanics-only statement.

Admission verifies that names resolve to the immutable manifest, output files
exist and match their captured bytes, all required assertions have exact types
and accepted values, and no undeclared assertion replaces a required one.  It
cannot prove from first principles that a consumer adapter invoked the intended
scientific condition.  The adapter must therefore call the existing production
validator and bind both its source identity and output receipt.  Acceptance
tests inspect those adapters directly and run them against the two current
consumers.

This is deliberately stronger than accepting a free-form `passed: true` and
weaker than importing consumer semantics into stable infrastructure.

### 6. Materialize final admission as an idempotent post-terminal projection

After both stage payloads validate, the module finalizes the nested journal and
builds `admission.json` from the binding manifest fingerprint, journal plan and
terminal identities, immutable stage payload fingerprints, captured outputs,
and claim boundary.  The compact receipt excludes attempt IDs and other
path-dependent diagnostics except through the separately bound journal
terminal identity.

Publication uses the existing crash-consistent exclusive helper.  If the final
publish is uncertain or fails after journal completion, a fresh process inspects
the terminal journal and may publish only the same canonical receipt.  An
existing byte-identical receipt is accepted as idempotent closure; differing
bytes fail closed.

The receipt status is `mechanically_admitted`, not `approved`, `qualified`, or
`ready_to_launch`.

### 7. Keep adapters thin and validate two real consumer shapes

The natural-boundary example will translate its existing source/runtime/model
bindings and sealed-plan validator into the shared requests.  Its CPU fixture
will exercise the legacy receipt materializer plus unchanged merger validator;
its bounded vertical stage will instead use the existing bounded terminal and
a mechanics-only validator for that terminal.  A one-record smoke cannot
honestly satisfy the legacy merger's eight-shard, 200-context denominator.  The
crossover example will translate its planner/sealer CPU preflight, runtime
identity, finalizer, and current evidence validator without copying endpoint
semantics.

CPU tests cover both examples.  The real GPU gate uses only the already bounded
support worker because one real vertical consumer is sufficient to prove model
load, durable work-item publication, terminal materialization, and downstream
closure; the second consumer proves interface variation and production-shaped
CPU closure.  Neither example edits its source worktree.

## Risks / Trade-offs

- **[Capturing a large runtime or model directory is expensive]** → Run it only
  at admission boundaries, retain the canonical inventory in `bindings.json`,
  and revalidate before each of only two stages.  Do not add a mutable cache in
  V1.
- **[A caller can fabricate mechanics assertions]** → Require exact producer
  and validator source bindings plus output receipts, test adapters against the
  real consumer functions, and keep admission explicitly non-authorizing.
- **[Reserved output paths become present after creation]** → Separate their
  pre-work state from immutable inputs and bind produced files in stage records.
- **[Late final receipt failure leaves a terminal journal]** → Treat final
  admission as an idempotent projection recoverable from fully validated
  terminal records.
- **[The active sibling changes during implementation]** → Re-capture exact
  bytes and fail compatibility gates on drift; never patch or normalize that
  worktree implicitly.
- **[The change drifts toward actuator or scientific validation]** → Keep all
  consumer-specific assertions and payload meanings in adapters; a later
  actuator-consumption change owns requested/resolved/consumed semantics.

## Migration Plan

1. Add red tests for typed bindings, reserved roots, two-stage order, input
   drift, strict evidence, late publication, and absence compatibility.
2. Implement the admission owner over unchanged JSON and journal primitives.
3. Add support and crossover adapters with CPU production-path fixtures; retain
   all scientific schemas in those adapters.
4. Run focused and downstream CPU regressions, then create a fixed-tree source
   manifest for the bounded GPU gate.
5. Run one fresh-root single-GPU support smoke and close the vertical admission
   from its durable terminal and bounded mechanics validator, explicitly
   excluding legacy merger closure.
6. Freeze the diff and receipts for independent engineering and
   intent-contract review.  Do not archive, merge, cherry-pick, or push without
   a separate user request.

Rollback is deletion of the new admission module, adapters, tests, and active
change.  Existing journal roots, stable schemas, consumers, and sealed research
artifacts remain unaffected because no existing on-disk contract is changed.
