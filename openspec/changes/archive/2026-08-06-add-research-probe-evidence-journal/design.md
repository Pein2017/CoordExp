## Context

The current infrastructure has three useful but disconnected owners:

- backend results can carry immutable per-request decode receipts;
- training `RunWriter` serializes a logging row before one append write and
  stages atomic JSON replacement; and
- inference artifacts stage and validate a complete final artifact family.

They do not give a long-running caller a pre-execution plan receipt or an
independently durable boundary for each resumable work item. Inference artifact
serialization also routes values through a `default=str` conversion, while its
terminal manifest admits only a fixed set of inference fields. A caller can
therefore lose type identity or caller-owned execution context even when the
terminal status itself is written successfully.

The capability must be reusable by research callers without moving scientific
meaning into `src/artifacts/` or `src/inference/`. The current request-scoped
decode receipt remains the owner of one decode result. The exact-history seam
remains the owner of exact HF token-history evidence. This change sits outside
both as an execution-evidence lifecycle.

## Goals / Non-Goals

**Goals:**

- Make every CPU-discoverable evidence-value, identity, serialization, and
  write-read error fail before costly work.
- Preserve each completed caller-selected work item independently of later
  work, process failure, or terminal finalization.
- Distinguish one immutable execution plan from the process attempts that may
  contribute records to it.
- Expose exact-identity continuation discovery without deciding whether the
  caller is allowed to continue.
- Carry compact opaque execution context through inference success, failure,
  sharding, and merge without interpretation.
- Replace permissive artifact stringification with one strict canonical-JSON
  owner.

**Non-Goals:**

- Define `Probe`, `Arm`, `Condition`, `Intervention`, `Observation`, cohort,
  estimand, metric, unmatched, scientific-validity, claim, or stop semantics.
- Add a research runner, declarative schema, plugin registry, hook framework,
  reducer, scheduler, automatic retry policy, or GPU resource broker.
- Extend HF exact-history into arbitrary forward hooks or attention actuators.
- Provide concurrent multi-writer publication into one journal. Independent
  shards use independent journals or one controller-owned writer.
- Recover arbitrary in-memory state, provide exact training-state resume, or
  rewrite historical sealed artifacts.
- Migrate sibling-worktree research scripts as part of this planning change.

## Deferred Independent Improvements

This change is the first infrastructure wave, not a claim that every observed
probe friction belongs in one module. Keep the remaining candidates as
separate changes with their own consumers and acceptance:

1. Extend `HFExactHistory` only with supported exact-forward input
   materialization and declared attention-kernel policy after one current
   causal caller can be migrated without arbitrary hook ownership. Require
   full-vocabulary no-op parity, exact position/mask evidence, and actuator
   consumption; keep intervention meaning caller-owned.
2. Promote a token-native row and boundary grammar only after a second real
   consumer needs the same closed/commit, pre-opener/post-opener, stop, and
   token-offset semantics. Keep owner matching and outcome interpretation out.
3. Build costed and resumable shard admission by composing the existing shard
   planner with this journal after the journal passes its real-case gate.
   Estimate work-item and forward counts before accelerator admission and
   resume only exact identities; do not add a scheduler or device broker.

Do not start these follow-ups merely because this proposal is apply-ready.
Their named consumer, smallest acceptance test, and user-owned compatibility or
cost decision must be present first.

## Decisions

### 1. Publish one strict artifact-value owner

Add a small owner under `src/artifacts/` for recursive JSON-value validation,
canonical bytes, SHA-256 fingerprints, strict load, and atomic write helpers.
The accepted value algebra is explicit; callers must project a live object to
an authored receipt before crossing the boundary. The owner never invokes
`str`, `repr`, `dataclasses.asdict`, a custom JSON encoder, or a discovered
`.receipt()` method automatically.

The canonical byte encoding used for fingerprints and journal records matches
existing repository fingerprints: UTF-8, ASCII escaping, sorted mapping keys,
compact separators, and `allow_nan=False`. The shared owner also exposes strict
validation separately from presentation formatting, so existing pretty JSON
artifacts need not change bytes merely to share the value contract. Validation
runs before a file is opened or a backend session is created. The training
writer and inference artifacts adopt this owner instead of retaining different
private permissiveness rules.

**Alternative considered: retain `default=str` and add a known-type list.**
Rejected because a new callback or tensor type would again produce a plausible
but semantically false receipt.

**Alternative considered: automatically call `.receipt()` on arbitrary
objects.** Rejected because persistence would then depend on hidden dynamic
dispatch and callers could not review the exact projection at construction.

### 2. Use an immutable plan plus atomic per-work-item record files

The journal is one directory with four visible surfaces:

```text
<journal-root>/
  plan.json
  records/<sequence>-<work-item-id-digest>.json
  attempts/<process-attempt-id>/start.json
  attempts/<process-attempt-id>/outcome.json
  terminal.json
```

`plan.json` binds the journal schema, execution identifier, complete execution
identity mapping and fingerprint, plan fingerprint, ordered expected work-item
identifiers, and opaque caller context. Creation stages, syncs, publishes,
reloads, and validates this file before returning admission to the caller. An
occupied root is never repurposed.

A work item is the caller-selected durability and continuation unit, such as
one case-condition result or one support context. It is not necessarily one
scalar model forward. Each accepted record binds the plan, sequence,
work-item identifier, process attempt, opaque payload, and digests. The writer
fully serializes first, takes the journal's exclusive writer lock, verifies
that sequence and work-item identities are unused, writes and syncs a temporary
file, atomically publishes it, and syncs the containing directory. Temporary
files are not accepted evidence and are reported on reload without being
silently promoted.

Atomic record files are selected over one JSONL stream because a killed process
cannot leave the last accepted record ambiguous or require destructive tail
truncation before prior records are reusable. The trade-off is one inode per
resumable work item; callers choose a decision-useful boundary rather than one
file per scalar forward. Focused tests and a small filesystem benchmark must
confirm that the expected hundreds or low thousands of records are not a
material end-to-end bottleneck.

**Alternative considered: hold all work-item outputs and stage one terminal
family.** Rejected because terminal failure remains a single point of total
evidence loss.

**Alternative considered: one fsynced JSONL journal.** Rejected for the first
version because torn-tail recovery and duplicate append after lost
acknowledgment require a repair protocol that is broader than the requested
seam.

### 3. Separate execution, process attempt, and terminal state

The execution plan remains immutable. Each process invocation receives a new
attempt identifier and may append an attempt receipt describing start,
mechanical failure, or clean exit. Attempt failure does not assign scientific
meaning and does not write terminal completion.

On reload, the journal validates every accepted record and reports the exact
completed work-item identifiers. A successor process may open the journal only
when the entire execution identity and plan fingerprint match. It chooses a new
attempt identifier and may work only on missing planned items. The library does
not launch, retry, or claim that continuation is scientifically permitted.

Any source, config, model, tokenizer, runtime-policy, intervention-code, or
other caller-bound identity change rejects continuation. An infrastructure
repair therefore starts a new execution journal. A mere process interruption
with identical evidence semantics may be continued if the research owner
permits it.

`terminal.json` is published only after the exact planned set exists once and
all record digests validate. It binds the ordered record-digest aggregate. A
finalizer exception leaves prior records readable and the journal visibly
non-terminal.

**Alternative considered: make every process failure terminal.** Rejected
because it conflates a failed invocation with a failed immutable execution and
prevents exact-identity continuation after ordinary interruption.

**Alternative considered: automatically resume missing items.** Rejected
because continuation may change research meaning, cost, or a frozen stop rule;
the infrastructure can report state but cannot own that decision.

### 4. Carry inference execution context as a digest-bound sidecar

Add an optional immutable execution-context input at inference assembly. It
contains an opaque context mapping and an optional journal plan reference that
binds only the pre-existing journal schema, execution identifier, plan
fingerprint, and plan-file SHA-256. It never contains a future terminal
fingerprint. Assembly canonicalizes and validates the input, then materializes
one immutable controller-owned `execution_context.json` before backend or
worker construction.

The data-parallel launch contract carries the context file's verified locator,
file SHA-256, canonical-value fingerprint, and journal plan reference. Each
worker reads and verifies the exact bytes before backend construction, copies
those bytes into its successful or terminal-failure artifact family, and binds
the same evidence from its manifest. A digest alone is never treated as a
transport for reconstructing the context.

When present, both successful and terminal-failure publication retain one
`execution_context.json` sidecar. Summary and manifest reference its relative
path, file SHA-256, and canonical-value fingerprint instead of duplicating
caller context. Merge verifies every rank-local copy against the controller
source and publishes those exact controller-owned bytes as the top-level
sidecar. The sidecar has no required arm, cohort, or claim keys.

The inference sidecar remains an immutable preflight record. If the execution
journal later completes, `terminal.json` binds the work-item record that points
back to the inference artifacts; the sidecar is not mutated to add a terminal
fingerprint. A separate consumer may form a post-terminal reference from the
plan and terminal digests after both exist.

Inference artifact writers replace permissive `_json_safe(default=str)` use
with the shared strict serializer. Migration tests first enumerate currently
valid artifact payload types and add explicit projections for any legitimate
non-JSON internal value. Unsupported values then fail before model work rather
than at finalization.

**Alternative considered: put caller fields directly into
`DecodeExecutionReceipt`.** Rejected because one decode receipt must remain
backend-owned and independent of experiment orchestration.

**Alternative considered: copy caller context into every summary, manifest,
and shard.** Rejected because copies can drift; one digest-bound sidecar keeps a
single byte owner.

### 5. Gate adoption with mechanics evidence, then stop

The first gate is CPU-only and uses the exact failure shape: a planned callback
object is rejected while its explicit receipt mapping completes plan
write/read. An interruption fixture appends several records, injects a
finalizer failure, reloads the journal, and verifies all earlier records.

The second gate runs one bounded real inference case through context preflight,
model execution, work-item publication, terminal publication, and readback.
It proves only the mechanics path. Support expansion, cohort execution, and
scientific interpretation remain separate research decisions and are not
implementation acceptance for this change.

## Risks / Trade-offs

- **[The journal becomes a generic research framework]** → Keep stable fields
  limited to identity, plan membership, attempts, strict payload bytes,
  durability, integrity, and completion; enforce a residue test against
  research vocabulary.
- **[Per-record sync or inode cost delays primary observations]** → Measure the
  expected hundreds or low thousands of resumable records and keep work-item
  granularity caller-owned; do not journal every scalar forward by default.
- **[Strict serialization breaks a currently tolerated artifact value]** →
  inventory live payload types in tests and add explicit semantic projections;
  never restore implicit stringification.
- **[A caller resumes after a semantic change]** → Bind complete caller-owned
  execution identity and reject any fingerprint drift. Infrastructure reports
  compatibility but does not authorize continuation.
- **[Journal paths are mistaken for proof]** → A preflight reference binds the
  immutable plan fingerprint and file SHA; a post-terminal reference separately
  binds the terminal fingerprint after it exists. Consumers validate content
  before use and never infer completion from a plan path.
- **[A successful mechanics smoke is reported as a model result]** → Label
  smoke receipts mechanics-only and keep all scientific fields outside the
  capability.

## Migration Plan

1. Add failing strict-value, plan, record, interruption, terminal, and
   continuation tests before publishing the new module.
2. Implement the shared canonical artifact-value owner and migrate existing
   valid training/inference serializer callers with byte-compatibility tests.
3. Implement the single-writer journal and CPU production-shaped interruption
   fixture.
4. Add optional inference execution context, sidecar publication, shard
   propagation, and merge validation while preserving no-context behavior.
5. Run focused artifact and inference suites, strict OpenSpec validation, and
   an independent intent/boundary audit.
6. After explicit runtime authorization, run one minimal real mechanics smoke.
   Only then hand the seam to a research-owned consumer for a separately
   authorized scientific pilot.

Rollback removes the optional context path and journal consumer imports, while
leaving already written journal directories readable through their schema
version. Do not rewrite or delete existing evidence during rollback.
