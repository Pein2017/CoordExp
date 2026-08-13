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

This probe is acceptance evidence, not implicit launch authority. The executor
MUST freeze a launch packet bound to the exact commit, config, artifact root,
and commands and pass an independent pre-cost review. If the external signed
review receipt for that exact frozen packet is valid and `READY`, the lead-only
executor proceeds under the current goal-level authority without asking the
user again. The manifest cannot authorize itself. Any implementation,
manifest, packet, review path, config, command, target, or bound mutation
invalidates `READY` and requires a new freeze plus review; it does not require
a repeated authorization prompt.
The packet fixes `world_size=2`, uses at most two GPUs, permits at most
three bounded semantic arms: one success arm comprising the matched control and
resumed branches, one rank-failure arm, and one interruption arm. Each success
branch executes exactly one next forward and at most one applied optimizer
update; each failure-shaped arm executes no more than one forward and one
applied update per rank. The packet declares numeric, config-derived ceilings
for model forwards and collectives per rank and per branch/arm, plus per-arm and
total wall time, command-specific CPU RSS measurement modes and bounds,
per-rank GPU-memory high-water marks, new artifact bytes across both success
branches and the failure-shaped arms, and required free disk. The two success
commands retain per-rank CPU RSS maxima. Setup, rank failure, interruption, and
verification instead use the maximum concurrent sum of RSS across the exact
owned command tree in any one sampler snapshot; this command-tree aggregate is
not the sum of independent per-PID high-water marks and requires at least one
owned sample. Missing bounds or samples, changed commands/commit, an occupied
artifact target, or any exceeded bound stops the launch and requires a new
packet rather than an automatic retry.

Wave-3 attempts 1-3 and all of their receipts are immutable historical
evidence. Attempt 3 is frozen at implementation commit
`037ab6683f9eeeb99157960f9fcf5bb3176a7044`, manifest SHA-256
`c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`,
and packet SHA-256
`70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02`.
Its pre-cost disposition is `HOLD` before launch because two P1 gaps remain:
the external parent observer has a check-to-signal window in which the trainer
can enter step 2 after committing step 1, and the six direct commands have no
outer executor guaranteed to publish an attempt-level receipt on either
success or failure. No attempt-3 command, GPU/model work, cache preparation, or
artifact-target mutation has executed. The manifest and packet are immutable
evidence and MUST NOT be edited to repair these gaps.

Attempt 4 is also immutable historical evidence. It is frozen at
implementation commit `5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e`,
manifest SHA-256
`b3538fb6167186cd5063f343a447ef1ed8f3024dc07f96284c7628c1ab08d3a9`,
and packet SHA-256
`45afe6475e2aa3cae6e106bc446725de4b60197b77ba3d0a3a8ff45263242a87`.
Setup and `success.uninterrupted_control` both returned zero and cleanup was
confirmed after each command, but the signed outer receipt stopped at
`packet_executor.missing_gpu_rank`; no later command or retry ran. The failure
is evidentiary, not a training failure: NVML/`nvidia-smi` returned host PIDs
while the executor observed `/proc` in a container PID namespace, and there is
no verifiable mapping between those identities. Consequently a selected UUID
plus an NVML PID/starttime process row cannot own semantic GPU-rank coverage.
The attempt-4 manifest, packet, marker, review, outer receipt, and produced
artifacts remain immutable and MUST NOT be edited or reinterpreted as a
successful qualification.

Attempt 5 is likewise immutable historical evidence and executed nothing. It is
frozen at implementation commit `9ef726085d2118dca90b373ccde4a88d068bf516`,
manifest SHA-256
`ee9227fdb310f7e9a2d629000df137ac5dbc2b472461a83f729df56909d58c24`,
and packet SHA-256
`48b7bb924f960bb5f65a87dfa4be4b9f5289b79b9e02b5375b253dd7b4caf897`,
with target root `reconcile_exact_resume_2026-08-13-r5`. Its independent
pre-cost review
(`receipts/wave-3-attempt-5-pre-cost-review.json`, reviewer
`cc-opus5-xhigh-attempt5-20260813`) returned `HOLD` on two P0s, so no attempt-5
command, marker, cache preparation, model, `torchrun`, GPU allocation, or
artifact-target mutation ran. The first P0 is stale role-config identity: all
three `config_files` digests
(`08eae66e...` control, `7e24e6bf...` parent, `2daf76ee...` child) were carried
forward from the Attempt-4 target, but the rendered role configs embed
`run.artifact_root` (and, for the child, `resume.checkpoint_dir`), so every
digest changes with the target root and setup would have failed closed at
`packet_executor.setup_config_mismatch` after the marker and target were
already consumed. The second P0 is the resumed-parent run-state binding
corrected below. The attempt-5 manifest, packet, and signed `HOLD` review keep
their exact recorded hashes and MUST NOT be edited, re-signed, or reused; they
are the durable record of a pre-cost stop, not a failed launch.

A successor Attempt 6 is authorized only after the resumed-parent run-state
correction is committed and reviewed. It MUST use a new absent `-r6` target
root with its own marker and outer-receipt paths, and it MUST deterministically
render all three role configs against that `-r6` root and compare the rendered
bytes and SHA-256 against the manifest before freezing. No role-config digest
may be copied forward from any earlier attempt. Attempts 1-5 and every one of
their receipts remain readable evidence and MUST NOT be edited or deleted to
make room for Attempt 6.

The only authorized parent repair is probe-local and test-first in
`scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`: the resumed
parent alone uses a synchronous held-parent entry route that wraps the real
pipeline checkpoint handler, calls that real handler first, and then blocks at
the committed step-1 boundary so the trainer cannot enter step 2. The
uninterrupted control and resumed child continue to launch `src.train`.
Production `src/` behavior, exact-admission strictness, max-step/checkpoint
compatibility, and provider support remain unchanged.

The only authorized execution repair is a separate experiment-local
`reconcile_exact_resume_packet_executor.py`. Its manifest schema v2 carries a
machine-readable `execution_contract` fixing the command order to setup,
control, resumed, rank failure, interruption, verification, and declaring each
command's CPU measurement mode, required rank rows, and corresponding numeric
bound. The same `execution_contract` binds the exact immutable pre-cost review
receipt path; it does not bind that receipt's hash, because the review in turn
binds the manifest hash. The independently signed review uses schema
`coordexp-swift-reconcile-resume-probe-pre-cost-review-v1` and contains
`status: READY`, the exact implementation commit, manifest SHA-256, packet
SHA-256, independent reviewer identity, and `receipt_payload_sha256`. The
executor validates that signature, self-digest, schema, independence, status,
and all three frozen identities before marker creation. A manifest-side
`READY`, missing review, `HOLD`, stale identity, mutable/replaced review path,
or invalid signature never authorizes execution.

After that validation, the executor claims an absent attempt marker with
`O_EXCL`, stops on the first failure with no retry, and launches exactly one
Popen-like process for one command at a time. That returned process is the
sole command/process-group owner. Every post-launch exit path -- including
process/NVML sampler failure, artifact summarization failure, timeout, bound
failure, launch-observation error, nonzero exit, or executor exception -- runs
the same bounded process-group cleanup: identify the group by leader PID and
Linux process starttime, send `TERM`, escalate to `KILL` if needed, reap the
returned process, and check group absence without accepting PID reuse. A
cleanup or absence-check failure is recorded and forces a failed terminal
outcome; no successful terminal receipt may precede confirmed group absence.

The successor contract retains `nvidia-smi` only for the frozen exact
`physical_index` to UUID map and initial occupancy/headroom preflight. The
executor revalidates that map before marker creation and immediately before
the first GPU command. NVML process rows may be retained as optional
observations, but host/container PID-namespace ambiguity means no such row can
satisfy or fail semantic GPU-rank coverage.

Success GPU coverage instead comes from a target-bound three-way artifact
join. The signed success receipt binds the implementation commit, exact config,
run directory, and role. That run's `run.json` binds `runtime.world_size`,
completed progress, and immutable
`policy_identities.runtime_determinism.launcher_attestations` for rank,
local rank, logical CUDA device, and `cuda_visible_devices`. Canonical train
rows in `logging.jsonl` bind each rank's
`per_rank_measurement.<rank>["resource/gpu_max_memory_allocated_bytes"]` and
`per_rank_measurement.<rank>["resource/gpu_max_memory_reserved_bytes"]`. The accepted
rank inventory is exactly integer ranks `0` and `1`; each value MUST be a JSON
number that is finite, nonnegative, integer-valued, and not a boolean. The
conservative bounded value for each rank is the maximum of allocated and
reserved high-water marks. The outer receipt declares
`gpu_measurement_source: torch_allocator_high_water` and retains the raw
allocated and reserved maxima, accepted row/step inventory, each source file
path and SHA-256, and the complete
rank -> local rank -> logical CUDA device -> physical index -> UUID mapping.

For `success.uninterrupted_control`, the signed control receipt MUST bind a
completed `world_size=2` run with `completed_steps=2`; `logging.jsonl` MUST
contain exactly one train row for each of steps 1 and 2 and no other accepted
train step, and maxima cover both rows. For `success.resumed_child`, the signed
resumed receipt spans both lifetimes: the parent MUST have an authenticated,
controlled exit at exactly step 1 rather than normal completion and exactly
one parent train row at step 1; because a `SIGTERM`-terminated held parent
never reaches `RunWriter.finalize()`, its durable `run.json` `status` MUST be
the creation-time `initialized` with a null `completed_at`, and `completed`,
`failed`, or any other status fails closed while the exact completed-progress,
checkpoint-event, and topology checks stay unchanged; the child MUST be completed with
`world_size=2`, `completed_steps=2`, and exactly one child train row at step 2.
Its maxima merge parent and child rows. Control, parent, and child MUST each
attest exact topology `cuda_visible_devices: ["6", "7"]`, ranks/local
ranks/logical devices `0 -> 0 -> 0` and `1 -> 1 -> 1`, joined through the
manifest's physical-index-to-UUID map.

Missing or invalid receipt digest, commit, config, run-directory, or role
binding; invalid run state, progress, or topology; duplicate, wrong-step, or
missing train rows; missing ranks or fields; non-finite, boolean, negative, or
non-integer-valued measurements; or a conservative maximum above its frozen
bound fails closed. Fake or otherwise admissible-looking NVML process rows
cannot replace any missing receipt, `run.json`, `logging.jsonl`, or topology
binding.

Required CPU coverage remains command-specific.
`success.uninterrupted_control` and `success.resumed_child` use CPU mode
`per_rank` with `required_cpu_ranks: [0, 1]`. `setup`,
`rank_failure`, `interruption`, and `verification` use CPU mode
`command_tree_aggregate` with `required_cpu_ranks: []`; their accepted CPU
evidence is an owned command-tree sample and
`cpu_rss_command_tree_max_bytes`, the maximum across sampler snapshots of the
concurrent RSS sum for all exact PID/starttime-owned processes in that
snapshot, bounded by the frozen `max_cpu_rss_command_tree_bytes`. The
executor MUST NOT infer semantic rank inventory from unranked model-free OS
process rows or turn the failure-shaped arms into `torchrun` commands. Missing
required success CPU rows, a missing owned aggregate sample, or either kind of
CPU bound exceedance is a terminal evidence failure.

Rank meaning for the two model-free arms belongs to their signed semantic arm
receipts. Their existing experiment-local schemas remain
`coordexp-swift-reconcile-resume-probe-rank-failure-receipt-v2` and
`coordexp-swift-reconcile-resume-probe-interruption-receipt-v2`. Every
rank-failure injection records exact `expected_ranks: [0, 1]`,
`serialized_ranks: [0, 1]`, and injection-derived `published_ranks`; the frozen
representative `rank=1, kind=missing` therefore records `[0]` and exact
`error_code: training_state.incomplete_rank_set`. Every interruption boundary
records exact `expected_ranks: [0, 1]`, serialized ranks, and published ranks;
at the frozen `stop_after=1`, serialized and published ranks are both `[]` and
an explicit `rank_state_boundary_reached` field is false. The verifier checks
every field exactly and rejects missing, stale-schema, re-signed-tampered, or
digest-tampered arm receipts.

Only after a command returns zero, its process-group cleanup is confirmed, and
its bounded artifact summary is complete does the executor validate the
command's artifact GPU metrics; resource success is impossible before that
ordering completes.

The executor always attempts one signed outer terminal receipt. It binds the
implementation, manifest, packet, and review identities; exact argv
observations; marker and command process-group identities; launcher callable
and Python runtime identity; per-command CPU coverage and optional process/NVML
observations; artifact-derived GPU source, bindings, topology, inventories,
hashes, raw allocated/reserved maxima, and conservative per-rank maxima; and
aggregate
`cpu_rss_command_tree_max_bytes` values; bounded artifact-tree summaries under
only the declared roots and numeric entry/depth/path/byte limits; cleanup
outcome; stop outcome; and the inner verifier receipt when verification is
reached.
Artifact-summary overflow fails instead of silently truncating. These two
seams are qualification tooling, not production orchestration.

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
- **[Risk] A stale review is mistaken for authority to consume GPUs or publish
  probe artifacts.** → Require exact frozen hashes and a `READY` pre-cost
  review; any mutation requires re-freeze/re-review. Under the current goal,
  `READY` is the lead-only execution gate and no repeated prompt is required.
- **[Risk] Concurrent user-owned documentation edits overlap reconciliation.**
  → Inspect ownership and diff at apply time, patch only exact stale claims,
  and never reset or overwrite unrelated changes.

## Migration Plan

1. Freeze the evidence matrix and identify any delta language unsupported by
   current source/tests.
2. Add or tighten focused tests first, then make the minimum source repair for
   demonstrated gaps.
3. Freeze the exact probe commands and quantitative bounds, pass the independent
   pre-cost/distributed-qualification audit, then let the lead-only executor
   proceed under current goal authority without another prompt. Any mutation
   requires re-freeze/re-review before the bounded probe may execute and publish
   its immutable scope-labeled receipt under this change.
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
