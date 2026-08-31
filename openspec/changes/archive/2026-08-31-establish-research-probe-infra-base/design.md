## Context

See [proposal.md](proposal.md) for motivation and
[the infra-base delta spec](specs/coordexp-swift-research-probe-infra-base/spec.md)
for the observable contract.

The current shared mechanics are already deep, independently useful modules:

- `src/artifacts/json_values.py` owns strict JSON, canonical identity, and
  crash-consistent exclusive publication;
- `src/artifacts/evidence_journal.py` owns immutable plans, distinct process
  attempts, independently durable work items, and exact-identity continuation;
- `src/artifacts/research_probe_admission.py` owns typed bindings, clean target
  revalidation, and the fixed mechanics-only `cpu_preflight` to
  `vertical_smoke` gate;
- `src/inference/runtime.py` and `src/inference/backend.py` own the public
  processor-only frontend, serializable launch, decode request/result, session,
  receipt, and cleanup contracts.

The missing boundary is discoverability and composition, not another executor.
The public `src.artifacts` facade already exposes much of the artifact surface,
but not the complete minimal set a new probe needs.  There is no current guide
that tells a direction when to stop at exclusive publication, when to add a
journal, when admission is justified, or when the inference session does not
fit.

The two requested specimens constrain the design differently:

- The live Image2299 direction owns distributed rank groups, experiment stages,
  direct differentiable HF access, manual or optimizer-backed updates,
  checkpoint/evaluation policy, and direction-specific atomic output helpers.
- Human13 is represented by records and `probe-final/human13-*` tags, not a
  live worktree.  Its standalone N/K route used independently scheduled
  acquisition, rebase, and cell producers, wrote no model checkpoint, and
  restored each cell to Source.  Its all-HF route stopped before an update and
  cannot be treated as completed execution evidence.

Their common contract is identity and artifact durability.  Their runtime,
rollback, checkpoint, scheduling, and scientific contracts are not identical.

## Goals / Non-Goals

**Goals:**

- Give a new probe one short decision path to the smallest existing mechanics
  owner and a stable import path for that owner.
- Keep one-shot, journaled, admitted, and inference-backed behavior independently
  composable per producer.
- Make it cheap to prove that the base preserves caller topology and treats
  scientific payloads as opaque.
- Leave a precise evidence gate for promoting the next genuinely repeated
  cross-direction mechanic.

**Non-Goals:**

- Migrating or rewriting the active Image2299 direction or reconstructing a
  Human13 worktree.
- Defining a probe class, lifecycle object, phase enum, callback registry,
  configuration DSL, scheduler, or generated launcher.
- Generalizing Image2299 checkpoint policy or Human13 transaction/rollback
  semantics.
- Adding persistent training-state resume; it remains outside the current
  project promise.
- Changing journal, admission, inference, branch, worktree, or scientific
  contracts.

## Decisions

### Use capability profiles over a run abstraction

The selected design is a documented selection table whose rows are ordinary
producer needs:

| Producer need | Existing owner | Caller still owns |
| --- | --- | --- |
| one strict immutable result | strict JSON plus exclusive publication | result schema, meaning, output path |
| several durable work items or process continuation | evidence journal | work plan meaning, scheduling, retry authorization |
| CPU evidence before a bounded model launch | research-probe admission | scientific plan, launcher, result, stop rule |
| production-aligned deterministic decode | inference runtime/backend session | requested contrast, parsing/evaluation choice, claims |

A producer imports the selected owners directly.  The profiles have no runtime
representation and therefore cannot acquire a hidden lifecycle or serialize
unrelated producers.

**Alternative A: `src/artifacts/probe_execution.py` wrapper.**  A dynamic
wrapper could accept optional journal and admission arguments.  It is rejected
because it would be a pass-through layer over current deep owners, would have
to invent generic `run`, `stage`, and `result` vocabulary, and would still not
fit Image2299 and Human13 topology without modes.

**Alternative B: generic runner or phase DSL.**  A runner could own workers,
callbacks, checkpoints, retries, and terminal status.  It is rejected because
the specimens disagree on distributed topology, checkpoint existence,
rollback scope, and stage meaning.  The shared blast radius would be larger
than the duplicated glue it removes.

**Alternative C: promote a trainable-model transaction now.**  Human13 has a
well-tested parameter/optimizer/scheduler/RNG transaction, while Image2299 has
both manual tensor updates and optimizer-backed distributed updates plus
direction-specific restoration checks.  Their overlap is not yet a single
caller-visible contract.  Promotion is deferred until a second live direction
uses the same state surface and failure semantics.

### Deepen the existing artifact facade, not the implementation graph

`src.artifacts` remains the convenience import boundary and its lazy-loading
pattern remains unchanged.  The change adds only the stable symbols required
by the profiles that are currently available solely from their owner modules:

- `publish_json_exclusive`;
- `BindingManifest`, `TargetTreeBinding`, `TargetTreeIdentity`, and
  `AdmissionInspection`;
- `capture_target_tree_binding`, `revalidate_binding_manifest`, and
  `revalidate_target_tree_binding`.

The existing public exports for canonical JSON, binding request types,
`capture_binding_manifest`, `ExecutionEvidenceJournal`, and
`ResearchProbeAdmission` remain.  Serialization, locking, filesystem
publication, Git inspection, journal validation, and admission logic do not
move into `__init__.py`.

Adding a new package is rejected: it would make the infra-base appear to be a
separate execution framework and create another place to search for owners.
Re-exporting inference objects through `src.artifacts` is also rejected;
inference keeps its own direct public module paths.

### Document exact selection and failure meaning once

Add `docs/RESEARCH_PROBE_INFRA_BASE.md` as the operator-facing decision page.
It will contain:

1. the fixed route
   `research-probe-infras -> research-probes -> probe/<direction>` with a link
   to, not a copy of, the branch/worktree policy;
2. the four-row capability table above and minimal import snippets;
3. output-root and per-producer rules, including why an experiment-global
   journal is not implied;
4. mechanics-only status language and the boundary around scientific payloads;
5. a promotion checklist requiring two live cross-direction consumers, exact
   semantic match, and one cheap falsifying comparison;
6. explicit non-owners: worktree lifecycle, scheduling, training mutation,
   checkpoint semantics, monitoring, metrics, claims, and stop rules.

`docs/AGENT_INDEX.md` will route reusable probe-mechanics questions to this
page.  `docs/BRANCH_AND_WORKTREE_POLICY.md` will add one link at the existing
infra-lane description; its lifecycle rules remain authoritative and are not
restated.

### Verify composition at consumer-visible boundaries

One focused CPU test module will exercise only new composition promises:

- importing the complete public artifact surface remains lazy and CPU-only;
- exclusive publication rejects an occupied target and preserves its bytes;
- two journal roots created from a common immutable input can progress in
  different orders without a global lock, shared sequence, or cross-mutation;
- opaque payload fields that look scientific round-trip without acquiring a
  shared interpretation.

The test reuses existing journal and publication implementations.  Existing
owner tests remain the authority for corruption, fsync uncertainty, attempt
integrity, binding drift, and admission stage behavior; this change does not
duplicate their fault matrices.

No GPU test is needed because the new behavior is import selection,
composition, and documentation.  Existing GPU admission task 4.2 remains a
separate authorization-bound gate and is not completed by this change.

The broad research-vocabulary residue suite is baseline-red on unchanged deep
owners in the implementation seed, so it is not a hard gate for this change.
The task-specific documentation residue audit remains required.

### Treat specimens as evidence, not migration targets

The implementation does not edit the dirty live Image2299 worktree and does
not restore Human13 source.  Acceptance instead checks that the guide can map
both observed shapes without claiming they are interchangeable:

- Image-like immutable leaves may use exclusive publication; distributed
  barriers, checkpoints, mutable HF access, and overwrite-permitted local
  intermediates remain outside the base.
- Human-like independent producers may each use an exclusive leaf or their own
  journal; no experiment-global order or checkpoint requirement is introduced.

This avoids turning a historical route into a fake second live consumer while
still using its failure history to bound the interface.

## Risks / Trade-offs

- [A guide can become stale as public owners evolve] -> Keep it a routing page,
  link normative specs, and test only the named public imports and composition
  examples.
- [A convenience facade can become a grab bag] -> Export only existing stable
  owner symbols required by the four profiles; no aliases, callbacks, or
  experiment types.
- [Callers may use exclusive publication where overwrite is intended] -> The
  guide makes occupied-path failure explicit and offers no silent overwrite
  fallback; overwrite-permitted intermediates stay caller-owned.
- [A per-producer journal can be mistaken for a global experiment journal] ->
  The independent-roots test and guide state that producer topology remains
  caller-owned.
- [Deferring rollback/runtime sharing leaves some repeated code] -> Record the
  promotion discriminator; add shared code only after a second live direction
  proves identical state and failure semantics.
- [Existing unrelated checkout dirt is mistaken for change work] -> Scope every
  diff and verification command to this OpenSpec and its declared source/docs
  surfaces; do not stage or alter unrelated `.codex` deletions.

## Migration Plan

1. Characterize the existing `src.artifacts` public surface and add the minimal
   missing lazy exports with one import-level regression.
2. Add the one-shot and independent-journal composition checks without changing
   any deep owner behavior.
3. Add the canonical infra-base guide and the two routing links; run a residue
   check that it does not duplicate lifecycle or scientific contracts.
4. Run focused artifact tests, the existing journal/admission suites, strict
   OpenSpec validation, compile/format checks, and one bounded intent-contract
   review of the frozen diff.
5. Submit the verified change for user review.  Merge into `research-probes`,
   tag a new research baseline, or update a live direction only under a later
   explicit authorization.

Rollback is ordinary source rollback of the new exports, test, guide, and
links.  No persisted schema or external artifact migration is introduced.
