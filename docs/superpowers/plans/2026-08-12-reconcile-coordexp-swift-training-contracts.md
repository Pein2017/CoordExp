# Reconcile CoordExp-Swift Training Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the live opt-in exact-training-state path and reconcile stable contracts and canonical documentation only to behavior proven by target-bound evidence.

**Architecture:** OpenSpec is the sole authority for scope, compatibility, and completion; this plan is an execution index over its proposal, design, tasks, and four delta specs. Work starts from a clean, pinned commit and a requirement-by-requirement evidence matrix; any demonstrated production gap stops execution for a test-first OpenSpec and plan revision rather than inviting an inferred repair.

**Tech Stack:** Python 3.12, PyTorch distributed/Accelerate replicated DDP, Pydantic/YAML configuration, safetensors/PEFT checkpoint payloads, pytest, OpenSpec CLI, Git, and the `ms` Conda environment.

## Global Constraints

- Work only in `/data/CoordExp/.worktrees/CoordExp-swift`; run Python and pytest through `conda run -n ms`.
- The authoritative change is [`reconcile-coordexp-swift-training-contracts`](../../../openspec/changes/reconcile-coordexp-swift-training-contracts/): [proposal](../../../openspec/changes/reconcile-coordexp-swift-training-contracts/proposal.md), [design](../../../openspec/changes/reconcile-coordexp-swift-training-contracts/design.md), [tasks](../../../openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md), and [delta specs](../../../openspec/changes/reconcile-coordexp-swift-training-contracts/specs/). If this plan and those artifacts differ, stop and repair the plan; do not reinterpret the OpenSpec contract here.
- Exact training state is disabled by default. The supported candidate is limited to `resume.mode: exact_same_world_size`, `runtime.determinism.mode: strict_cuda_replay_v1`, completed optimizer-step boundaries, the same world size, and compatible rank/device mapping.
- The minimal inference payload remains independently loadable; inference ignores `training_state/`. Disabled mode writes no exact-state sibling or identity.
- Only explicit `synchronous` and `overlapped` input-provider modes are supported. This change records `legacy_fused` and `COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE` as unsupported residue but does not delete them.
- Do not add changed-order packing, a cache materialization campaign, an efficiency claim, logging, loss/RL behavior, orchestration decomposition, dependency upgrades, historical migration, cross-world-size resume, or mid-accumulation resume.
- Preserve unrelated dirty work. Never reset, clean, bulk-stage, overwrite user edits, repair an occupied immutable cache target, or rewrite historical evidence.
- Tests precede any production repair. A failing acceptance test may demonstrate a gap; it does not authorize a source change until Task 2's stop/re-plan checkpoint has produced an amended OpenSpec task and an amended execution task with exact interfaces and a red/green command.
- The distributed probe requires an exact frozen packet and an external signed independent pre-cost review receipt whose `READY` binds the implementation commit, manifest and packet hashes, configs, artifact roots, commands, comparison policy, and quantitative bounds. The manifest cannot authorize itself. Once that exact review is valid and `READY`, the lead-only executor proceeds under the current goal-level authority without asking the user again. Any mutation invalidates `READY` and requires re-freeze/re-review, not a repeated authorization prompt. Its success arm is a matched pair: one uninterrupted control branch and one parent-to-resumed-child branch, each executing the corresponding next forward and at most one optimizer update.
- Commits below are future execution checkpoints. Stage only the paths named by the step, inspect `git diff --cached`, and never include unrelated work.

---

## File and Ownership Map

The following map constrains navigation; it does not pre-authorize edits to production owners.

- `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`: one row per delta requirement/scenario, with exact source owner, test owner, command, receipt, and `accepted|gap|remove` disposition.
- `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/`: command manifests and immutable, scope-labeled verification receipts created by this change.
- `src/config/models.py`, `src/config/loader.py`, `src/config/writer.py`, `src/runtime/seeding.py`: strict resume configuration and pre-CUDA deterministic admission owners; read first and modify only after the stop/re-plan gate.
- `src/artifacts/checkpoints.py`, `src/artifacts/training_state.py`, `src/artifacts/run_writer.py`, `src/artifacts/provenance.py`: inference payload, exact-state publication/admission, lineage/event, and provenance owners; read first and modify only after the stop/re-plan gate.
- `src/training/exact_resume.py`, `src/training/pipeline.py`: distributed exact-resume control plane and production composition owners; read first and modify only after the stop/re-plan gate.
- `src/training/pack_cache.py` and the live cache owner located by `rg -n "determinant|verification_level|payloads|manifest" src/training`: cache identity/admission owners; the evidence matrix records the exact current symbols before any conclusion.
- `tests/config/test_train_config.py`, `tests/artifacts/test_checkpoint_payload_identity.py`, `tests/artifacts/test_checkpoint_writer.py`, `tests/artifacts/test_training_state.py`, `tests/artifacts/test_run_artifacts.py`, `tests/artifacts/test_provenance.py`: config and artifact acceptance owners.
- `tests/training/test_exact_resume.py`, `tests/training/test_pipeline_exact_resume.py`, `tests/training/test_pipeline_cache_preflight.py`, `tests/training/test_pack_cache.py`, `tests/training/test_pack_cache_determinant_registry.py`, `tests/training/test_prepare_train_cache_cli.py`: exact-resume and cache acceptance owners.
- `tests/inference/test_pipeline.py`, `tests/inference/test_artifacts.py`, `tests/adapters/test_inference_reload_status.py`: inference-reader compatibility owners.
- `docs/COORDEXP_SWIFT.md`, `docs/SYSTEM_OVERVIEW.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/ARTIFACTS.md`: canonical documentation owners, edited only after qualification.

### Task 1: Pin a Clean Base and Build the Authority Matrix

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-0-baseline.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`

**Interfaces:**
- Consumes: the proposal/design/tasks/spec links in Global Constraints and the exact checked-out Git tree.
- Produces: a pinned implementation commit, frozen baseline command list, and one disposition row for every requirement and scenario that all later tasks consume.

- [ ] **Step 1: Verify the execution checkout is the requested clean base**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
```

Expected: `git status --short` and `git diff --check` produce no output, the branch is the intended implementation branch, and `git rev-parse HEAD` prints one 40-character commit. If the tree is dirty, stop and ask the current owner to commit or isolate it; do not stash, reset, clean, or continue from an unpinned tree.

- [ ] **Step 2: Freeze the executable baseline command manifest before evaluating behavior**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms python -VV
conda run -n ms python -c 'import accelerate, torch; print(torch.__version__); print(accelerate.__version__)'
conda run -n ms python -m pytest --collect-only -q tests/config/test_train_config.py tests/artifacts/test_checkpoint_payload_identity.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_training_state.py tests/artifacts/test_run_artifacts.py tests/artifacts/test_provenance.py tests/runtime/test_train_runtime.py tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py tests/training/test_pipeline_cache_preflight.py tests/training/test_pack_cache.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_prepare_train_cache_cli.py tests/inference/test_pipeline.py tests/inference/test_artifacts.py tests/adapters/test_inference_reload_status.py
```

Expected: Python 3.12 from the `ms` environment, importable Torch and Accelerate versions, and a nonzero collected test count with exit code 0. Record the exact outputs, commit, cwd, command strings, and UTC timestamp in `receipts/wave-0-baseline.md` using `apply_patch`.

- [ ] **Step 3: Create the evidence matrix with an exact, reviewable schema**

Use `apply_patch` to create `evidence-matrix.md` with these columns in this order:

```markdown
| Delta spec | Requirement / scenario | Live source owner | Focused test owner | Verification command | Receipt | Disposition | Claim boundary |
|---|---|---|---|---|---|---|---|
```

Populate one row for every requirement and every `#### Scenario:` in all four delta specs. Every source/test cell names an existing path plus symbol or pytest node; every command is runnable from the exact cwd; every disposition is exactly `accepted`, `gap`, or `remove`. An absent owner is recorded as `gap`, never as an inferred production behavior.

- [ ] **Step 4: Record provider authority and stable/doc conflicts**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
rg -n "legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE|synchronous|overlapped" src tests configs docs openspec/specs
rg -n "never.*resume|no.*training.state|inference.only|training_state|exact_same_world_size|strict_cuda_replay_v1|mutable.*cache|repair.*cache" docs openspec/specs src tests
```

Expected: the matrix records stable support only for `synchronous|overlapped`, classifies the legacy mode/environment override as later-decomposition residue, and lists every live contract conflict without editing source or docs.

- [ ] **Step 5: Run the Wave-0 contract gate**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
openspec validate reconcile-coordexp-swift-training-contracts --strict
rg -n "\| *(accepted|gap|remove) *\|" openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md
```

Expected: strict OpenSpec validation passes, the number of disposition rows equals the number of requirement/scenario rows, and no row has an unnamed source/test/command disposition. Obtain a read-only contract audit and resolve every P0/P1 finding before checking OpenSpec tasks 1.1–1.6.

- [ ] **Step 6: Commit the pinned evidence baseline**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-0-baseline.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git diff --cached --check
git diff --cached
git commit -m "docs(openspec): pin training contract evidence baseline"
```

Expected: the commit contains only the matrix, baseline receipt, and evidence-backed task checkboxes.

### Task 2: Execute the Baseline and Enforce the Stop/Re-plan Gate

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-1-focused-baseline.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Modify only after a gap is demonstrated and the current task stops: the owning OpenSpec artifacts and this plan.

**Interfaces:**
- Consumes: Task 1's pinned commit, baseline command manifest, and evidence rows.
- Produces: an executed focused baseline and exactly one decision: continue with qualification, remove unsupported delta language, or stop for a test-first re-plan.

- [ ] **Step 1: Run the complete focused baseline without changing production code**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q \
  tests/config/test_train_config.py \
  tests/artifacts/test_checkpoint_payload_identity.py \
  tests/artifacts/test_checkpoint_writer.py \
  tests/artifacts/test_training_state.py \
  tests/artifacts/test_run_artifacts.py \
  tests/artifacts/test_provenance.py \
  tests/runtime/test_train_runtime.py \
  tests/training/test_exact_resume.py \
  tests/training/test_pipeline_exact_resume.py \
  tests/training/test_pipeline_cache_preflight.py \
  tests/training/test_pack_cache.py \
  tests/training/test_pack_cache_determinant_registry.py \
  tests/training/test_prepare_train_cache_cli.py \
  tests/inference/test_pipeline.py \
  tests/inference/test_artifacts.py \
  tests/adapters/test_inference_reload_status.py
```

Expected: exit code 0, with exact pass/fail/skip counts recorded in `wave-1-focused-baseline.md`. Every unexpected skip or failure is classified against a matrix row before any edit.

- [ ] **Step 2: Apply the decision rule row by row**

Use this exact rule:

```text
accepted = current source owner exists AND a focused test/probe executes the scenario AND the receipt supports only the stated claim boundary
remove   = proposed normative language lacks a current capability or a bounded qualification path
gap      = the proposed contract has a named owner and bounded need, but current executable evidence fails or is absent
```

Expected: no requirement is marked accepted from source inspection alone, archived receipts, mocked-only coverage for a distributed claim, or exit status without artifact inspection.

- [ ] **Step 3: Stop or continue at the production-gap checkpoint**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
rg -n "\| *gap *\|" openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md
rg -n "\| *remove *\|" openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md
```

Expected decision:

- If any `gap` row requires a production-source change, stop this plan. Update the owning OpenSpec artifact first, then amend this plan with one task that names the failing pytest node, exact source symbol/signature, minimal red/green implementation, regression command, and rollback. Obtain contract review before writing production code.
- If a row is `remove`, narrow the delta spec and corresponding tasks with `apply_patch`, run strict validation, and rerun the focused executable gate before continuing.
- Only if all retained rows are `accepted` may execution continue to Task 3 without production edits.

This checkpoint forbids speculative edits to `src/`; it is the plan's explicit defense against inventing a production gap.

- [ ] **Step 4: Commit the executed baseline and disposition**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-1-focused-baseline.md openspec/changes/reconcile-coordexp-swift-training-contracts/design.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md openspec/changes/reconcile-coordexp-swift-training-contracts/specs
git diff --cached --check
git diff --cached
git commit -m "test(training): classify contract qualification baseline"
```

Expected: either a no-source accepted baseline/narrowed delta commit, or no commit because execution stopped for re-planning. Do not stage an unchanged or unrelated path.

### Task 3: Qualify Config, Payload, Admission, and Historical Reading

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-2-contract-qualification.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`
- Test: `tests/config/test_train_config.py`
- Test: `tests/artifacts/test_checkpoint_payload_identity.py`
- Test: `tests/artifacts/test_checkpoint_writer.py`
- Test: `tests/artifacts/test_run_artifacts.py`
- Modify: `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py:FROZEN_REQUEST_PRODUCER_SOURCE_SHA256`
- Test owners: the config, artifact, exact-resume, pipeline-exact-resume, and inference files listed in the File and Ownership Map.

**Interfaces:**
- Consumes: a matrix with no unresolved production gaps and the current public interfaces `load_train_config`, `CheckpointWriter.write_checkpoint`, `publish_distributed_exact_resume`, `admit_and_restore_current_rank`, and `admit_exact_resume_checkpoint_publication`.
- Produces: target-bound evidence for disabled/enabled config and payload behavior, fail-closed admission, restored state/cursors, and conservative historical reading.

- [ ] **Step 1: Run the strict config matrix as an isolated acceptance gate**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q \
  tests/config/test_train_config.py::test_exact_resume_defaults_disabled_and_is_persisted \
  tests/config/test_train_config.py::test_exact_resume_requires_strict_cuda_replay_determinism \
  tests/config/test_train_config.py::test_exact_resume_accepts_same_world_mode_and_resolves_checkpoint_path \
  tests/config/test_train_config.py::test_exact_resume_rejects_unknown_or_incompatible_controls
```

Expected: all nodes pass and demonstrate omission/default persistence, YAML-relative exact path resolution, strict replay requirement, and unknown/incompatible rejection. If a node is missing or fails semantically, return to Task 2 and re-plan test-first.

- [ ] **Step 1A: Preserve the exact publish-only control/parent configuration**

Withdraw the Task-3 discovery RED because fixed-target review proved it
contradicts the exact-state control/parent path. Replace it with a positive
test:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q \
  tests/config/test_train_config.py::test_exact_publish_only_mode_preserves_null_checkpoint_path
```

Expected: exit code `0`; the resolved config retains
`{"mode": "exact_same_world_size", "checkpoint_dir": null}`. Do not modify
`src/config/models.py` or `src/training/pipeline.py`. The live split is:
`resume.mode` enables exact-state publication, while a non-null
`resume.checkpoint_dir` selects restore.

Run all affected config consumers, not only the leaf config file:

```bash
conda run -n ms pytest -q \
  tests/config/test_train_config.py \
  tests/training/test_wave7_exact_resume_config_bundle.py \
  tests/training/test_input_attestation.py \
  tests/training/test_wave7_exact_resume_sequence.py
```

Expected: every config consumer passes except an independently frozen
historical-source identity assertion already classified with exact observed and
expected hashes. Repair such a failure only through its owning historical
receipt/disposition, never by changing publish-only semantics.

The archive-path cleanup changed only the request producer's OpenSpec authority
path, turning its SHA-256 from
`540101ead19eeb82a7a2821954371a01e57e936f1484d8a0cf5130b330af71ac` to
`7c2e3d4688977c8b60ca08c24325e1497dd562792cd0bf8286e5af93bfc04b3f`.
Update only `FROZEN_REQUEST_PRODUCER_SOURCE_SHA256` to that observed current
hash; retain the exact source-hash assertion and do not relax or remove it.

- [ ] **Step 2: Run inference-payload and exact-sibling acceptance tests**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/artifacts/test_checkpoint_payload_identity.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_training_state.py
```

Expected: the receipt cites passing nodes for independent inference commit, no exact sibling in disabled mode, typed sibling in enabled mode, delayed aliases/events, incomplete/corrupt state rejection, and absence of base/full-embedding/LM-head tensors. A broad green file is not enough; cite the exact nodes that own each matrix row.

- [ ] **Step 3: Run exact admission, restoration, and lineage tests**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py tests/artifacts/test_run_artifacts.py
```

Expected: cited nodes prove same-world-size/rank-map checks, optimizer-boundary enforcement, pre-mutation rejection, next-step/pack cursor restoration, optimizer/scheduler/scaler/RNG restoration, and append-only child lineage. Numeric comparison claims remain dtype-appropriate and do not become cross-launch bitwise claims.

- [ ] **Step 4: Run the historical-reader fixture matrix**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/adapters/test_inference_reload_status.py tests/inference/test_pipeline.py tests/inference/test_artifacts.py tests/artifacts/test_training_state.py -k 'checkpoint or payload or training_state or historical or resume'
```

Expected: compatible explicit adapter/selected-token payloads remain loadable with extra historical metadata, inference never requires the sibling, and exact admission accepts only the current committed typed schema. If the filter collects zero tests, stop and re-plan a named fixture test instead of treating it as evidence.

- [ ] **Step 5: Publish and execute the Wave-2 gate**

Use `apply_patch` to record exact commands, pinned commit, test counts, cited pytest nodes, matrix rows closed, failures/skips, and claim boundary in `wave-2-contract-qualification.md`. The focused executable matrix is the Wave-2 gate; any failure, unexpected skip, or zero-test selection returns to Task 2. Do not add another independent audit layer.

- [ ] **Step 6: Commit the qualified interface evidence**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-2-contract-qualification.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git add openspec/changes/reconcile-coordexp-swift-training-contracts/design.md openspec/changes/reconcile-coordexp-swift-training-contracts/specs docs/superpowers/plans/2026-08-12-reconcile-coordexp-swift-training-contracts.md
git add scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py
git add tests/config/test_train_config.py tests/artifacts/test_checkpoint_payload_identity.py tests/artifacts/test_checkpoint_writer.py tests/artifacts/test_run_artifacts.py
git diff --cached --check
git diff --cached
git commit -m "test(training): qualify bounded resume interfaces"
```

Expected: the commit contains the positive publish-only config test, qualified
scenario tests, corrected delta/design/tasks/plan, and evidence-backed Wave-2
artifacts. No source, pipeline, checkpoint format, historical migration, or
inference-runtime implementation change is included. Rollback removes only the
new tests/evidence and restores affected matrix rows to `gap`; it does not
re-introduce the invalid exact-without-path clause.

### Task 4: Close the Attempt-4 GPU-Evidence Stop and Qualify Attempt 8

**Files:**
- Read only: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-3-command-manifest.json`
- Read only: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-3-launch-packet.md`
- Read: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-3-pre-cost-review.md`
- Read only: every `wave-3-attempt-4-*` manifest, packet, marker, review,
  terminal receipt, and produced artifact.
- Read only: every `wave-3-attempt-5-*` manifest, packet, and signed review;
  Attempt 5 executed nothing and produced no marker, receipt, or artifact.
- Read only after this evidence commit: the exact immutable Attempt-6 manifest,
  packet, and signed `READY` review; Attempt 6 executed nothing and produced no
  marker, command, target, cache, model, `torchrun`, GPU, or output artifact.
- Modify: `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py`
- Test: `tests/training/test_reconcile_exact_resume_probe.py`
- Modify: `scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py`
- Test: `tests/training/test_reconcile_exact_resume_packet_executor.py`
- Create after this planning/evidence commit: new Attempt-8 schema-v2 command
  manifest, packet, independent pre-cost review, attempt marker, and outer
  terminal receipt under new change-local receipt and `-r8` artifact roots.
- Modify after executed evidence: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md` and `tasks.md`.

**Attempt-3 disposition:** implementation commit `037ab6683f9eeeb99157960f9fcf5bb3176a7044`, manifest SHA-256 `c3e4e93d997c87ad26379b0246f5536aec4f96afbc9a59be16985572a718cf42`, and packet SHA-256 `70b4f4cc7b235db0f21dcf5e3ade68d06b5db923a4876ceee6ba92ceed07ca02` are frozen evidence. Pre-cost review is `HOLD` on two P1s: the external observation-to-signal controller has a parent step-2 TOCTOU, and direct execution has no outer attempt receipt. No attempt-3 command, GPU/model work, cache preparation, or artifact-target mutation executed. Do not edit the manifest or packet.

**Attempt-4 disposition:** implementation commit
`5d68a41081aecc3282cb44ce70bcb5bcdcc7c19e`, manifest SHA-256
`b3538fb6167186cd5063f343a447ef1ed8f3024dc07f96284c7628c1ab08d3a9`,
and packet SHA-256
`45afe6475e2aa3cae6e106bc446725de4b60197b77ba3d0a3a8ff45263242a87`
are frozen evidence. Setup and `success.uninterrupted_control` both returned
zero, each cleanup was confirmed, and the signed outer receipt stopped at
`packet_executor.missing_gpu_rank`; no later command or retry ran. The selected
UUID plus PID/starttime gate is unverifiable because NVML/`nvidia-smi` reports
host PIDs while executor `/proc` observes a container PID namespace and no
authenticated mapping exists. Preserve every Attempt-4 file and output.

**Attempt-5 disposition:** implementation commit
`9ef726085d2118dca90b373ccde4a88d068bf516`, manifest SHA-256
`ee9227fdb310f7e9a2d629000df137ac5dbc2b472461a83f729df56909d58c24`,
packet SHA-256
`48b7bb924f960bb5f65a87dfa4be4b9f5289b79b9e02b5375b253dd7b4caf897`,
and target root `reconcile_exact_resume_2026-08-13-r5` are frozen evidence.
The independent signed pre-cost review
`receipts/wave-3-attempt-5-pre-cost-review.json` (reviewer
`cc-opus5-xhigh-attempt5-20260813`) is `HOLD` on two P0s, and **no attempt-5
command, marker, cache preparation, model, `torchrun`, GPU allocation, or
artifact-target mutation executed**. P0-1: all three `config_files`
`expected_sha256` values (`08eae66e...` control, `7e24e6bf...` parent,
`2daf76ee...` child) are Attempt-4 renders, while the rendered role configs
embed `run.artifact_root` and, for the child, `resume.checkpoint_dir`; setup
would have failed closed at `packet_executor.setup_config_mismatch` after the
`O_EXCL` marker, the `-r5` target, and the private cache were already consumed.
P0-2: the executor bound the interrupted `resumed_parent` `run.json` to status
`running`, which production `RunWriter` never writes, because a
`SIGTERM`-terminated held parent never reaches `finalize()`; real interrupted
parents under `outputs/probes/coordexp_swift/wave7_exact_resume/2026-08-12-core-3`
and `-core-4` record `initialized` with `completed_at: null`. Only P0-2 is an
implementation defect and it is fixed test-first in this task; P0-1 is closed
only by a successor freeze. Do not edit, re-sign, delete, or reuse the
Attempt-5 manifest, packet, or review: their exact hashes and the signed `HOLD`
are the durable record of a pre-cost stop. Preserve every attempt 1-5 file.

**Attempt-6 disposition:** implementation commit
`1fbd9d68be4cf5dee1be1b09886f7c8a7b84205e`, manifest SHA-256
`4dff3f0bc81ca0e55cf614f3ef676430bddb1a236f24b36d97e948d5eafbe1ae`,
packet SHA-256
`28e79a1d850a2308980e172089b4f03823703be0c004bb25e373527818afb0b2`,
and independent `READY` review SHA-256
`8b23ae8838ad453ea5fb430f17e244759c67f382e67473810e2a0f680dd76a43`
with payload digest
`1f8705c595bc5771cb34e791fb6859483dd8ddbc1a344c169d61f8567d06c78f`
are frozen evidence. The review passed for those exact identities. Before the
marker, any command, the `-r6` target, cache preparation, model load,
`torchrun`, GPU allocation, or output mutation, concurrent tests-only commit
`27abc8087e6c8591257463b19c8a83a63fd554e1` changed tracked HEAD and therefore
invalidated Attempt 6's exact-commit and tracked-clean binding. Attempt 6
executed nothing. Preserve its three files byte-for-byte; do not launch, edit,
re-sign, delete, or reuse them. Attempt 8 is an append-only successor, not a
repair of Attempt 6.

**Interfaces:**
- Held parent: only `success-resumed` routes the parent through a probe-local synchronous entry that wraps the real pipeline checkpoint handler, calls the real handler first, and blocks after committed step 1. Control and child continue to use `-m src.train`. No `src/` edit or compatibility weakening is allowed.
- Packet executor callable: `execute(*, manifest_path: Path, packet_path: Path, expected_manifest_sha256: str, expected_packet_sha256: str, attempt_marker_path: Path, terminal_receipt_path: Path, launch=..., gpu_sampler=..., process_sampler=...) -> dict`.
- Packet executor CLI: `execute --manifest ... --packet ... --manifest-sha256 ... --packet-sha256 ... --attempt-marker ... --terminal-receipt ...`.
- Preserve that public callable/CLI and the exact six-command order. Do not add
  a runner framework or another launch abstraction. `launch` returns exactly
  one Popen-like process; the executor owns that process and its process group
  through terminal cleanup.
- Schema v2 adds only `execution_contract`: exact six-command order, `O_EXCL`
  marker creation, stop-on-failure, retry count zero, required resource
  observations with a CPU measurement mode and matching numeric bound for each
  command, required outer-receipt/inner-receipt bindings, and the exact
  immutable signed pre-cost review receipt path. It MUST NOT include the review
  receipt hash: the review binds the manifest hash, so adding the reverse hash
  would make the freeze circular.
- The review schema is
  `coordexp-swift-reconcile-resume-probe-pre-cost-review-v1` and requires
  `status: READY`, exact implementation commit, manifest SHA-256, packet
  SHA-256, independent reviewer identity, and `receipt_payload_sha256` over
  the canonical review payload excluding that digest and its signature
  envelope. The executor validates the signed receipt and all bindings before
  marker creation; manifest self-`READY` alone has no authority.
- Retain `nvidia-smi` for the frozen exact `physical_index` to GPU UUID map and
  initial occupancy/headroom preflight, validating the map before marker
  creation and immediately before the first GPU command. NVML process rows are
  optional observations only and never a GPU-rank gate.
- Resource coverage is exact per command. The true two-rank torchrun commands
  `success.uninterrupted_control` and `success.resumed_child` use CPU mode
  `per_rank` and `required_cpu_ranks: [0, 1]`. Their GPU coverage uses
  `gpu_measurement_source: torch_allocator_high_water` and exact ranks
  `[0, 1]` from a target-bound signed-success-receipt + `run.json` + canonical
  `logging.jsonl` join. The model-free `setup`, `rank_failure`,
  `interruption`, and `verification` commands use CPU mode
  `command_tree_aggregate`, `required_cpu_ranks: []`, and
  no GPU artifact requirement. Their CPU bound and observed maximum are
  `max_cpu_rss_command_tree_bytes` and `cpu_rss_command_tree_max_bytes`,
  respectively. The observed value is the maximum, across sampler snapshots,
  of the concurrent RSS sum for all exact PID/starttime-owned processes in that
  snapshot. It is never the sum of independent per-PID high-water marks and it
  requires at least one owned sample. OS sampler rows do not own semantic rank
  meaning and the model-free arms MUST NOT be changed to torchrun.
- The signed success receipt binds exact implementation commit, config,
  `run_dir`, and role. `run.json` binds `runtime.world_size`, completed
  progress, and immutable
  `policy_identities.runtime_determinism.launcher_attestations`. Every accepted
  run has `cuda_visible_devices: ["6", "7"]`, rank/local/logical tuples
  `0/0/0` and `1/1/1`, joined through the manifest physical-index-to-UUID map.
  Canonical train rows provide exact
  `per_rank_measurement.<rank>["resource/gpu_max_memory_allocated_bytes"]` and
  `per_rank_measurement.<rank>["resource/gpu_max_memory_reserved_bytes"]`;
  values must be finite,
  nonnegative, integer-valued JSON numbers and not booleans. The bounded value
  is `max(allocated, reserved)`.
- Control requires a signed control receipt, completed `world_size=2` run with
  `completed_steps=2`, and exactly one train row at steps 1 and 2; maxima span
  both. Resumed coverage requires a signed resumed receipt spanning parent and
  child: the parent has an authenticated controlled exit at exactly step 1
  (not normal completion) and exactly one train row at step 1; the child is
  completed at `world_size=2`, `completed_steps=2`, with exactly one train row
  at step 2; maxima merge both lifetimes. The parent's durable `run.json`
  `status` is the creation-time `initialized` with a null `completed_at`,
  because the terminated held parent never reaches `RunWriter.finalize()`;
  `completed`, `failed`, and `running` fail closed, and the exact
  completed-progress, checkpoint-event, and topology checks are unchanged.
- Accumulation invariant: the accepted top-level `consumed_packs` of `1` and
  `2` and each checkpoint event's `committed_progress.consumed_packs` equal to
  its step are correct only at exactly one pack per optimizer step. That ratio
  is not a config field: production computes it in
  `src/config/resolve.py:resolve_effective_batch_runtime` as
  `resolved_grad_accum_steps = effective_batch_size // world_size`, taking
  `effective_batch_size` from the loaded config and `world_size` from the
  caller. Attempt 8 retains and re-freezes the same three separate authorities.
  (1) Config identity: the
  manifest and packet bind the absolute base-config path
  `/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
  with SHA-256
  `44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
  plus each rendered role config's bytes and SHA-256, and every rendered role
  config loaded through production `load_train_config` MUST resolve
  `effective_batch_size` to `2`. (2) Launch topology: `world_size` is never
  read from a role config; the frozen launch argv and manifest MUST bind
  `world_size = 2` (setup `--world-size 2` and the two-rank `torchrun` success
  commands), verified against that argv. (3) Derived accumulation: production
  `resolve_effective_batch_runtime(config, world_size=2)` MUST return
  `resolved_grad_accum_steps: 1` for `uninterrupted_control`,
  `resumed_parent`, and `resumed_child`, hence expected consumed progress `1`
  at step 1 and `2` at step 2. The executor revalidates all three before marker
  creation and in setup validation and stops without retry on drift. The packet
  stays experiment-local
  and fixed at one pack per step; do not generalize the executor to arbitrary
  accumulation in this round.
- Missing or bad receipt digest/commit/config/run-dir/role, run state/progress
  or topology, duplicate/wrong/missing rows or steps, missing rank/field,
  non-finite/boolean/negative/non-integer measurement, or bound excess fails
  closed. Fake NVML rows cannot satisfy missing artifacts.
- Signed arm receipts own model-free rank semantics. Retain the existing
  experiment-local schemas
  `coordexp-swift-reconcile-resume-probe-rank-failure-receipt-v2`
  and `coordexp-swift-reconcile-resume-probe-interruption-receipt-v2`.
  Rank-failure receipts record exact expected `[0, 1]`, serialized `[0, 1]`,
  injection-derived published ranks, and exact production error code; the
  frozen missing-rank-1 representative records published `[0]` and
  `training_state.incomplete_rank_set`. Interruption receipts record exact
  expected `[0, 1]`, serialized/published ranks, and
  `rank_state_boundary_reached`; frozen `stop_after=1` records
  serialized/published `[]` and boundary false. The verifier checks each field
  exactly.
- The outer receipt records resolved launcher module/qualname/file digest,
  Python executable realpath/version, each command leader PID/starttime and
  process-group cleanup outcome, and bounded artifact-tree summaries under
  declared roots and numeric entry/depth/path/byte ceilings. Only after command
  return zero, confirmed cleanup, and artifact summary does the executor
  validate artifact GPU metrics and permit resource success. The receipt
  records `gpu_measurement_source`, source paths and SHA-256, exact row/step
  inventory, raw allocated/reserved maxima, conservative per-rank maxima, and
  rank-to-local/logical/physical/UUID mapping. It also records aggregate-mode
  `cpu_rss_command_tree_max_bytes`. Exceeding a summary or resource ceiling, or
  failing to obtain one owned aggregate sample, fails closed.

- [ ] **Step 1: Preserve the completed held-parent regression seam**

Run exactly, without changing production `src/`:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q \
  tests/training/test_reconcile_exact_resume_probe.py::test_resumed_parent_uses_held_parent_route_while_control_and_child_use_src_train \
  tests/training/test_reconcile_exact_resume_probe.py::test_held_parent_calls_real_handler_then_blocks_before_step_two \
  tests/training/test_reconcile_exact_resume_probe.py::test_held_parent_does_not_hold_when_step_one_publication_fails \
  tests/training/test_exact_resume.py::test_resume_compatibility_keeps_max_steps_and_checkpoint_cadence_strict
```

Expected GREEN: the already implemented route assertions and compatibility node
pass. They remain a prerequisite for Attempt 8: the real handler returns before
the step-1 hold, handler failure propagates without holding, and the trainer
cannot enter step 2 while held.

- [ ] **Step 2: Confirm the immutable Attempt-4 stop boundary**

Read the immutable Attempt-4 manifest, packet, marker, signed review, signed
outer terminal receipt, control receipt, control `run.json`, and control
`logging.jsonl`. Verify their recorded SHA-256 identities, both command return
codes, both confirmed cleanup outcomes, stop code, and absent later-command
inventory. Compare NVML host-PID observations with executor container-namespace
`/proc` identities only to establish that no authenticated mapping exists; do
not edit, re-sign, or reinterpret any Attempt-4 file.

Expected: the durable artifacts reproduce `packet_executor.missing_gpu_rank`
after successful control and show that the control logging rows already carry
ranked torch-allocator allocated/reserved high-water marks. This is root-cause
evidence and a fixture source, not qualification success.

- [ ] **Step 3: Write the packet-executor RED tests**

Extend `tests/training/test_reconcile_exact_resume_packet_executor.py` and keep
the existing `tests/training/test_reconcile_exact_resume_probe.py` semantic-arm
regressions in the focused selection. Cover the unchanged callable/CLI,
hash/schema/packet validation, exclusive
absent-marker creation, exact six-command order, stop/no-retry, resource-mode
accounting, semantic arm receipts, and outer/inner receipt bindings. Add these
exact nodes:

- `test_manifest_self_ready_without_signed_review_rejects_before_marker`
- `test_hold_or_stale_review_rejects_before_marker`
- `test_launch_returns_one_popen_like_process`
- `test_post_launch_sampler_artifact_timeout_and_error_paths_leave_no_process_group`
- `test_swapped_physical_index_uuid_mapping_rejects_before_marker`
- `test_nvml_host_pid_rows_are_observational_not_gpu_rank_gate`
- `test_unranked_model_free_cpu_subtree_needs_no_rank_environment`
- `test_command_tree_cpu_peak_is_max_concurrent_snapshot_sum`
- `test_command_tree_cpu_bound_and_missing_owned_sample_fail_closed`
- `test_success_gpu_metrics_require_signed_receipt_run_and_logging_join`
- `test_control_gpu_metrics_require_exact_steps_topology_and_values`
- `test_resumed_gpu_metrics_merge_parent_and_child_lifetimes`
- `test_success_gpu_metrics_reject_bad_bindings_rows_and_bound_excess`
- `test_fake_nvml_rows_cannot_satisfy_missing_gpu_artifacts`
- `test_gpu_artifacts_validate_after_return_cleanup_and_summary`
- `test_rank_failure_v2_records_exact_semantic_ranks_for_every_injection`
- `test_interruption_v2_records_exact_semantic_ranks_for_every_boundary`
- `test_verify_rejects_tampered_semantic_arm_rank_fields`

The post-launch cleanup test is parameterized over process sampler, optional
NVML sampler, artifact summarizer, timeout, nonzero exit, and executor-error
paths. It launches only a harmless sleeping CPU subprocess in a fresh process
group and proves TERM, bounded KILL escalation, reap, and PID/starttime-safe
group absence before inspecting the terminal receipt; it does not touch a GPU.
The CPU-mode tests launch an unranked real subprocess tree with both `RANK` and
`LOCAL_RANK` absent, prove aggregate peak is the maximum concurrent per-snapshot
sum, reject both a missing owned sample and an aggregate bound exceedance, and
preserve success CPU rank requirements `[0, 1]`. GPU tests use fixture receipts,
`run.json`, and `logging.jsonl` only: they require the exact bindings, states,
steps, topology, ranks, fields, numeric domain, conservative bound, hashes,
inventory, and parent+child merge above; prove NVML rows cannot substitute;
and assert validation ordering after return zero, confirmed cleanup, and
artifact summary. The semantic tests cover
every injection rank/kind and every interruption boundary, then prove exact
field and signature tampering is rejected.
The review tests prove that manifest self-`READY`, a signed `HOLD`, or a review
whose commit/manifest/packet identity is stale executes nothing and creates no
marker. Run the exact RED selection:

```bash
conda run -n ms pytest -q \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_manifest_self_ready_without_signed_review_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_hold_or_stale_review_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_launch_returns_one_popen_like_process \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_post_launch_sampler_artifact_timeout_and_error_paths_leave_no_process_group \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_swapped_physical_index_uuid_mapping_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_nvml_host_pid_rows_are_observational_not_gpu_rank_gate \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_unranked_model_free_cpu_subtree_needs_no_rank_environment \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_command_tree_cpu_peak_is_max_concurrent_snapshot_sum \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_command_tree_cpu_bound_and_missing_owned_sample_fail_closed \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_success_gpu_metrics_require_signed_receipt_run_and_logging_join \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_control_gpu_metrics_require_exact_steps_topology_and_values \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_resumed_gpu_metrics_merge_parent_and_child_lifetimes \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_success_gpu_metrics_reject_bad_bindings_rows_and_bound_excess \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_fake_nvml_rows_cannot_satisfy_missing_gpu_artifacts \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_gpu_artifacts_validate_after_return_cleanup_and_summary \
  tests/training/test_reconcile_exact_resume_probe.py::test_rank_failure_v2_records_exact_semantic_ranks_for_every_injection \
  tests/training/test_reconcile_exact_resume_probe.py::test_interruption_v2_records_exact_semantic_ranks_for_every_boundary \
  tests/training/test_reconcile_exact_resume_probe.py::test_verify_rejects_tampered_semantic_arm_rank_fields
```

Expected RED: collection succeeds; the new artifact-GPU nodes fail against the
pre-correction executor while existing arm-schema and exact-resume behavior
remain untouched, and no packet, artifact root, cache, GPU, or model is
executed.

- [ ] **Step 4: Implement only the artifact-derived GPU resource contract**

Modify only the experiment-local packet executor and, only if the existing
signed success-receipt verifier needs a bounded helper, the experiment-local
probe script. Preserve the executor callable/CLI, held-parent behavior, arm
receipt schema v2, and accept only the v2 manifest whose machine-readable
`execution_contract` fixes:

```text
setup
success.uninterrupted_control
success.resumed_child
rank_failure
interruption
verification
```

Validate both expected SHA-256 bindings, packet binding, commit/cwd/targets,
and contract before creating the marker. Open the contract-bound review path as
one non-symlink regular file, validate its signature,
`receipt_payload_sha256`, exact schema, `READY`, independent reviewer, and
commit/manifest/packet bindings from that same open file, and retain its
device/inode identity through marker acquisition. A missing, `HOLD`, stale,
replaced, malformed, or invalid review and any manifest self-authorization
execute nothing. This intentionally omits a review-file hash from the
manifest; the signed review binds the already frozen manifest hash.

Claim `attempt_marker_path` atomically with `O_CREAT|O_EXCL`; an existing
marker is terminal and executes nothing. Revalidate the frozen exact
`physical_index` to UUID map both before marker creation and before the first
GPU command. After ownership, call `launch` once per attempted command and
require exactly one Popen-like return value. Record its leader PID and Linux
starttime; form the observed descendant tree from PID/starttime pairs so PID
reuse cannot satisfy CPU ownership. Sample CPU through the injected process
sampler. Keep the injected NVML sampler only for physical-index-to-UUID and
initial occupancy observations; process rows are optional and never satisfy or
fail GPU-rank coverage. Enforce the CPU mode and exact required-rank matrix
from the interface section. For `per_rank`, retain rank `[0, 1]` CPU maxima. For
`command_tree_aggregate`, accept real owned PID/starttime rows without `RANK`
or `LOCAL_RANK`, require at least one owned sample, sum the concurrent RSS of
the exact owned tree separately in each snapshot, and retain the maximum of
those snapshot sums as `cpu_rss_command_tree_max_bytes`. Never sum independent
per-PID high-water marks. Apply the command's matching numeric bound and fail
closed on a missing sample or exceedance. Do not add torchrun to setup, rank
failure, interruption, or verification.

After a success command returns zero, process-group cleanup is confirmed, and
the bounded artifact summary is complete, validate its GPU metrics from the
target-bound three-way join only. Authenticate the signed success receipt and
its exact commit/config/run-dir/role bindings; hash and parse the bound
`run.json` and canonical `logging.jsonl`; validate run state/progress and exact
launcher topology; then validate exact rows, steps, ranks, field presence,
finite nonnegative integer-valued non-boolean values, and the frozen bound.
Control accepts exactly steps 1 and 2 once each from a completed world-size-2,
completed-step-2 run. Resumed accepts exactly parent step 1 after authenticated
controlled exit and child step 2 after completed world-size-2,
completed-step-2 state, and merges both lifetimes. Record
`gpu_measurement_source: torch_allocator_high_water`, source file paths and
SHA-256, row/step inventory, raw allocated/reserved maxima, conservative
per-rank maxima, and rank-to-local/logical/physical/UUID mapping. Any missing
or malformed binding/artifact/value, topology or step mismatch, duplicate row,
or bound excess fails closed; NVML process rows cannot repair it.

Preserve the existing experiment-local rank-failure and interruption receipt
schema-v2 semantics and verifier checks exactly; this correction does not
change arm meaning or production training artifacts.

Put all logic after a successful launch behind one terminal cleanup path. On
normal completion, sampler failure, artifact-summary failure, timeout, bound
failure, nonzero exit, or exception, send process-group `TERM`, escalate to
`KILL` after the declared grace, reap the returned process, and confirm group
absence keyed by leader PID/starttime. Record cleanup actions and any cleanup
or absence-check failure; such a failure forces a failed outer receipt. Write
the signed outer receipt to an absent `terminal_receipt_path` only after this
check. Bind all frozen identities including the review payload digest, exact
argv observations, resolved launcher and Python runtime identity,
PID/starttime/process-group observations, optional NVML observations,
artifact-derived GPU bindings/inventory/hashes/raw and conservative maxima,
per-rank CPU maxima, aggregate-mode `cpu_rss_command_tree_max_bytes`, bounded
per-root artifact-tree summaries, cleanup and stop reason, and the validated inner
verifier receipt if verification completed. Do not duplicate probe comparison
logic or add general orchestration.

Run:

```bash
conda run -n ms pytest -q \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_manifest_self_ready_without_signed_review_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_hold_or_stale_review_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_launch_returns_one_popen_like_process \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_post_launch_sampler_artifact_timeout_and_error_paths_leave_no_process_group \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_swapped_physical_index_uuid_mapping_rejects_before_marker \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_nvml_host_pid_rows_are_observational_not_gpu_rank_gate \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_unranked_model_free_cpu_subtree_needs_no_rank_environment \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_command_tree_cpu_peak_is_max_concurrent_snapshot_sum \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_command_tree_cpu_bound_and_missing_owned_sample_fail_closed \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_success_gpu_metrics_require_signed_receipt_run_and_logging_join \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_control_gpu_metrics_require_exact_steps_topology_and_values \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_resumed_gpu_metrics_merge_parent_and_child_lifetimes \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_success_gpu_metrics_reject_bad_bindings_rows_and_bound_excess \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_fake_nvml_rows_cannot_satisfy_missing_gpu_artifacts \
  tests/training/test_reconcile_exact_resume_packet_executor.py::test_gpu_artifacts_validate_after_return_cleanup_and_summary \
  tests/training/test_reconcile_exact_resume_probe.py::test_rank_failure_v2_records_exact_semantic_ranks_for_every_injection \
  tests/training/test_reconcile_exact_resume_probe.py::test_interruption_v2_records_exact_semantic_ranks_for_every_boundary \
  tests/training/test_reconcile_exact_resume_probe.py::test_verify_rejects_tampered_semantic_arm_rank_fields
conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_packet_executor.py
conda run -n ms pytest -q tests/training/test_reconcile_exact_resume_probe.py tests/training/test_exact_resume.py
```

Expected GREEN: every exact node from Step 3 and the full packet-executor file
pass; the harmless CPU subprocess tests leave no live process group; the
artifact fixtures prove exact GPU fail-closed behavior without NVML PID
ownership; the model-free arms retain ordinary single-process execution; and
the existing
probe and exact-resume files pass without packet, cache, GPU, or model
execution.

- [x] **Step 5: Freeze Attempt 8 and pass pre-cost review**

Note (2026-08-19): Attempt 7 executed this step faithfully — fresh `-r7`
render, correct bindings, full independent review — but was invalidated
pre-marker by concurrent docs-only commit `0b0f554e4...` during that
review and carries a signed immutable `HOLD`. Its receipts under
`receipts/wave-3-attempt-7-*` are evidence only; Attempt 8 is the sole
successor. Owning authority: the OpenSpec change's `tasks.md` 3.4/3.6.

Attempt 8 is authorized only after this planning/evidence commit exists and
the tracked worktree is clean. Record the exact then-current outputs of
`git rev-parse HEAD` and `git rev-parse HEAD^{tree}`; both become frozen launch
identities, and any subsequent tracked change invalidates the packet. Create
only the new Attempt-8 packet and schema-v2 manifest at
`receipts/wave-3-attempt-8-launch-packet.md` and
`receipts/wave-3-attempt-8-command-manifest.json`, bound to the new absent
`outputs/probes/coordexp_swift/reconcile_exact_resume_2026-08-13-r8` target,
its private cache, exact configs/argv, new absent review path
`receipts/wave-3-attempt-8-pre-cost-review.json`, new absent marker path
`receipts/wave-3-attempt-8-attempt-marker.json`, and new absent outer path
`receipts/wave-3-attempt-8-outer-terminal-receipt.json`. Bind the exact clean
HEAD/tree, manifest and packet bytes/SHA-256, retained physical-index-to-UUID
map, exact per-command CPU mode, matching bound and required-rank arrays,
bounded artifact-summary limits, and the same `world_size=2`, at-most-two-GPU,
three-arm comparison and quantitative bounds. Success commands declare CPU
mode `per_rank`, CPU ranks `[0, 1]`,
`gpu_measurement_source: torch_allocator_high_water`, artifact GPU ranks
`[0, 1]`, the target-bound receipt/run/log join, and their exact control or
parent/child row policies. The other four commands declare
`command_tree_aggregate`, empty required-rank arrays, and the exact
`max_cpu_rss_command_tree_bytes` bound. The two arm receipt schemas are v2. Do
not edit attempts 1-6 or reuse any Attempt-6 `-r6` target, packet, manifest,
review, digest, or launch authority. Before freezing, deterministically render
all three role configs against the `-r8` artifact root and compare their exact
bytes and SHA-256 against the new manifest `config_files` entries; the rendered
configs embed `run.artifact_root` and the child's `resume.checkpoint_dir`, so no
role-config digest or resolved fingerprint may be copied forward from any
earlier attempt. In the same pre-freeze pass, bind the
accumulation invariant through its three separate authorities. Load each
rendered role config through production `load_train_config` and require
`effective_batch_size` to resolve to `2`, recording each role's config bytes
and SHA-256 with the resolved fingerprint. Verify that the frozen launch argv
and manifest bind `world_size = 2` -- setup `--world-size 2` and the two-rank
`torchrun` success commands -- rather than reading `world_size` from any role
config. Then invoke production
`resolve_effective_batch_runtime(config, world_size=2)` on each loaded config
and require `resolved_grad_accum_steps: 1`, and therefore expected consumed
progress `1` at step 1 and `2` at step 2. Record the bound absolute base-config
path
`/data/CoordExp/.worktrees/CoordExp-swift/configs/coordexp_swift/smoke/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate2_ebs2_1step.yaml`
with SHA-256
`44c2cd2a6442917595e6426061e021e2989542114b7ce7cdb97cd37edb4f609f`
plus the newly rendered resolved fingerprint and its digest in the manifest
and packet. Validate the v2 manifest with the executor tests and
read-only hash/path/GPU/disk preflight. Obtain one independent signed
`coordexp-swift-reconcile-resume-probe-pre-cost-review-v1` receipt at the bound
Attempt-8 review path and resolve every P0/P1. The signed review
MUST independently verify each accumulation authority against its own owner:
the absolute base-config path/SHA-256 and role config bytes/digest with
`load_train_config` resolving `effective_batch_size` to `2`, the frozen
argv/manifest binding of `world_size = 2`, and
`resolve_effective_batch_runtime(config, world_size=2)` returning
`resolved_grad_accum_steps: 1`. The review binds
the manifest hash; the manifest does not bind a review hash.

Expected: if and only if that external signed review is valid, independent,
exactly bound to the new clean HEAD/tree/manifest/packet/config/target
identities, and `READY`, the lead-only executor may proceed under the
current goal-level authority without another user prompt. Manifest
self-`READY` never authorizes. Any implementation, manifest, packet, review
path, config, command, target, map, tree, or bound mutation invalidates `READY` and
returns to re-freeze/re-review; it does not create a repeated prompt.

- [x] **Step 6: Execute Attempt 8 exactly once through the packet executor**

After Step 5 records the exact hashes, invoke only this frozen Attempt-8 CLI,
substituting the two recorded full SHA-256 values without changing any other
argument:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms python scripts/probes/coordexp_swift/reconcile_exact_resume_packet_executor.py execute \
  --manifest openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-command-manifest.json \
  --packet openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-launch-packet.md \
  --manifest-sha256 <ATTEMPT7_MANIFEST_SHA256> \
  --packet-sha256 <ATTEMPT7_PACKET_SHA256> \
  --attempt-marker openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-attempt-marker.json \
  --terminal-receipt openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-attempt-8-outer-terminal-receipt.json
```

Before marker acquisition it validates the exact clean HEAD/tree, signed review,
packet/manifest/config identities, target absence, and physical-index-to-UUID
map. The executor
then claims the `O_EXCL` marker and runs the exact six commands in order, at
most once each, owning one returned Popen-like process and process group at a
time. Immediately after setup it must validate the prepare receipt,
implementation commit, private-cache receipt, all generated config hashes, the
bound absolute base-config path/SHA-256 with each role config resolving
`effective_batch_size: 2` through `load_train_config`, the frozen argv binding
of `world_size = 2`, the resulting
`resolve_effective_batch_runtime(config, world_size=2)` value
`resolved_grad_accum_steps: 1`, and
the GPU map again before the first model command. For each success command it
waits for return zero, confirms cleanup, completes the artifact summary, and
only then validates the signed receipt + `run.json` + `logging.jsonl` GPU join
before resource success; the resumed parent is accepted only in the durable
controlled-exit state `initialized` with a null `completed_at`. It stops without retry on
drift, missing required CPU or artifact-GPU rank rows,
missing owned aggregate samples, occupied paths, insufficient headroom,
timeout/hang/OOM, nonzero exit, artifact-summary overflow, cleanup failure,
bad receipt/run/log binding, state, topology, step, field, or numeric domain,
or any per-rank or command-tree aggregate bound failure. Optional NVML process
rows never satisfy missing artifacts.

Expected: the command is launched only by the lead executor, once, with no
automatic retry. Before the signed outer receipt is finalized, every launched process
group has been TERM/KILL/reaped and its absence checked by PID/starttime. The
receipt accounts for every attempted command, launcher/runtime identity,
each CPU mode and bound, optional NVML observations, artifact source paths and
hashes, exact row/step inventory, raw allocated/reserved maxima, conservative
per-rank GPU maxima, exact topology mapping, aggregate
`cpu_rss_command_tree_max_bytes`, bounded
artifact-tree summary, cleanup outcome, and the inner verified receipt when
reached. The inner evidence compares the required boundary/first-update state
and exactly verifies the schema-v2 rank-failure/interruption semantic fields.
Exit zero or an inner receipt without the outer receipt is insufficient.

- [x] **Step 7: Verify durable Attempt-8 artifacts and commit bounded evidence**

Use the Attempt-8 outer receipt and bound inner verifier receipt, not console
status, to prove exact six-command order, valid external review authorization, every
required success CPU and artifact-GPU rank and every semantic arm branch, an owned aggregate
sample and bounded concurrent command-tree RSS for each model-free command,
the target-bound receipt/run/log join and exact launcher-to-UUID topology,
the bound absolute base-config identity, the per-role
`effective_batch_size: 2` resolution, the argv-bound `world_size = 2`, and the
derived `resolved_grad_accum_steps: 1` that together make the recorded
consumed-pack progress exact,
launcher/runtime and target identity, exact arm schema/rank/error/boundary fields, bounded artifact-tree
summaries, process-group cleanup, comparison results, and the same-world-size
optimizer-step claim boundary. The outer receipt MUST bind the exact clean
HEAD/tree and Attempt-8 manifest/packet/review/config/target identities recorded
in Step 5. Stage only change-local Attempt-8 evidence with explicit paths,
leaving every attempt 1-6 receipt byte-for-byte untouched; inspect the staged
diff before committing. Large probe payloads remain in their bound `-r8`
artifact root.

### Task 5: Qualify Cache Admission and Non-secret Provenance

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-4-cache-provenance.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`
- Verify: `src/artifacts/provenance.py`, the current cache determinant/admission owners, and their focused tests.

**Interfaces:**
- Consumes: current cache fingerprints/manifests and `collect_execution_provenance(repository_root=...)`.
- Produces: model-free proof of current-version immutable cache admission and bounded, non-secret execution provenance; no production cache and no efficiency claim.

- [ ] **Step 1: Run cache identity and immutable-admission tests**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/training/test_pack_cache.py tests/training/test_pack_cache_determinant_registry.py tests/training/test_pipeline_cache_preflight.py tests/training/test_prepare_train_cache_cli.py
```

Expected: cited nodes prove transitive source/content invalidation, post-build determinant revalidation, full `payloads` verification in preparation, manifest-plus-eager-required-chunk validation in distributed admission, immutable occupied-target failure, and failure before model/optimizer/runtime construction. Any missing retained scenario returns to Task 2.

- [ ] **Step 2: Execute the model-free cache admission probe**

Use only fixture-scale inputs under a new temporary target outside production cache roots. Exercise absent, valid, and invalid-target outcomes through the current single-process preparation/admission interfaces; record fingerprints, manifest identities, verification levels, and model-loader call count.

Expected: the receipt proves zero model-weight loads, read-only reuse of a valid target, fail-closed rejection of an invalid occupied target, and no rewrite/repair/delete. Remove the temporary fixture target only if it was created by this task and the command records its exact path; do not touch user or production caches.

- [ ] **Step 3: Run provenance and privacy acceptance tests**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/artifacts/test_provenance.py tests/artifacts/test_run_artifacts.py -k 'provenance or dirty or dependency or secret or unavailable'
```

Expected: clean/dirty/untracked state, stable local-state digest, dependency version/source-or-binary identity, explicit unavailable reasons, deterministic strict JSON, and secret exclusion are all covered. The receipt records only non-secret values and the exact declared identity projection consumed by resume admission.

- [ ] **Step 4: Publish and execute the bounded gate**

Use `apply_patch` to create `wave-4-cache-provenance.md` with exact commands, fixture paths, fingerprints, manifest identities, verification levels, model-load count, provenance identity projection, test counts, and claim boundary. Treat the focused executable cache and privacy checks as the gate; remove unsupported normative language rather than adding another independent audit layer or importing archive scope.

- [ ] **Step 5: Commit cache/provenance qualification evidence**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-4-cache-provenance.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git diff --cached --check
git diff --cached
git commit -m "test(training): qualify cache and provenance contracts"
```

Expected: no cache payload, credential, environment secret, or unrelated file is staged.

### Task 6: Reconcile Canonical Documentation After Qualification

**Files:**
- Modify: `docs/COORDEXP_SWIFT.md`
- Modify: `docs/SYSTEM_OVERVIEW.md`
- Modify: `docs/IMPLEMENTATION_MAP.md`
- Modify: `docs/ARTIFACTS.md`
- Modify only if the conflict scan identifies a current contradiction: the smallest current router under `docs/`
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-5-doc-conflict-scan.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`

**Interfaces:**
- Consumes: fully qualified retained matrix rows and the resulting delta specs; archived changes and receipts are provenance, not documentation authority.
- Produces: canonical operator docs that distinguish minimal inference payloads from the optional exact-state sibling without duplicating schemas or promoting excluded work.

- [ ] **Step 1: Capture the failing pre-edit conflict scan**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
rg -n "never.*resume|no.*training.state|inference.only|training_state|exact_same_world_size|checkpoint.*optimizer|mutable.*cache|repair.*cache" docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md docs/AGENT_INDEX.md docs/README.md
```

Expected: `wave-5-doc-conflict-scan.md` records every stale categorical statement and its owning qualified requirement. If the scan finds no conflict, do not create cosmetic edits; record that outcome and proceed to Step 4.

- [ ] **Step 2: Patch only stale canonical statements**

Use `apply_patch` to make the smallest edits that link to the stable/delta owner and state the accepted disabled default, two-payload separation, supported same-world-size optimizer-boundary, fail-closed publication/admission, inference-ignore behavior, and non-goals. Do not copy receipt paths, full schemas, speculative performance statements, or historical narrative into evergreen docs.

- [ ] **Step 3: Preserve concurrent user edits**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
git diff -- docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md docs/AGENT_INDEX.md docs/README.md
git diff --check
```

Expected: each hunk changes only a stale contract statement found in Step 1. If a file contains concurrent user edits, stop and coordinate ownership instead of overwriting or reformatting the file.

- [ ] **Step 4: Run the passing post-edit authority scan**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
rg -n "legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE|cross.world.size|mid.accumulation|logging|RL|changed.order|throughput|MFU" docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md
openspec validate reconcile-coordexp-swift-training-contracts --strict
```

Expected: no text promotes excluded work, provider residue, cross-world-size/mid-accumulation resume, or performance claims; strict OpenSpec validation passes. Record exact results in `wave-5-doc-conflict-scan.md`; this focused executable scan is the Wave-5 gate, without another independent audit layer.

- [ ] **Step 5: Commit canonical documentation reconciliation**

```bash
git add docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-5-doc-conflict-scan.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git diff --cached --check
git diff --cached
git commit -m "docs(training): reconcile bounded resume contract"
```

Expected: only actually modified canonical docs, the conflict receipt, and evidence-backed task marks are staged. Omit unchanged paths from `git add`.

### Task 7: Final Verification, Contract Sync, and Honest Disposition

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/final-verification.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`
- Sync/archive only through the OpenSpec skills after every gate passes.

**Interfaces:**
- Consumes: final implementation commit, all target-bound receipts, the complete matrix, canonical docs, and all delta specs.
- Produces: a reproducible final verification record and either a fully supported sync/archive decision or an explicitly incomplete active/archive disposition.

- [ ] **Step 1: Run the full relevant suite from the final tree**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/config tests/artifacts tests/runtime tests/training tests/inference tests/adapters
```

Expected: exit code 0 with exact pass/fail/skip counts recorded in `final-verification.md`; investigate and disposition every unexpected skip. This broad result supplements rather than replaces the cited focused nodes.

- [ ] **Step 2: Re-run the target-bound matched-success and one failure/interruption verifier**

Run only the already-reviewed frozen verifier argv against the final artifact tree. Do not relaunch training if the implementation or execution inputs changed; re-freeze the exact successor and obtain a new pre-cost `READY`, after which the lead-only executor proceeds under the current goal authority without another prompt.

Expected: next input/pack identity, declared pre-forward state, first-update objective/loss fields and resulting trainable parameters, manifest/event/selector identities, rank/branch inventory, and historical-reader classification remain bound to the final commit and durable artifacts. If implementation changed after Task 4, the old probe is stale and this step stops for re-freeze/re-review of both success branches; no repeated user authorization prompt is required after `READY`.

- [ ] **Step 3: Run strict OpenSpec and residue gates**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
openspec validate reconcile-coordexp-swift-training-contracts --strict
rg -n "legacy_fused|COORDEXP_SWIFT_FORWARD_INPUT_PROVIDER_MODE|changed.order|throughput|MFU|tensorboard|RL|cross.world.size|mid.accumulation" openspec/changes/reconcile-coordexp-swift-training-contracts docs/COORDEXP_SWIFT.md docs/SYSTEM_OVERVIEW.md docs/IMPLEMENTATION_MAP.md docs/ARTIFACTS.md
git diff --check
git status --short
```

Expected: strict validation passes; matches are limited to explicit exclusions/provider classification; no unrelated change is attributed to this change; the worktree status is fully explained path by path.

- [ ] **Step 4: Reconcile the matrix and obtain the final audit**

Use `apply_patch` to bind every retained matrix row to final source, cited tests, and a final receipt. Obtain one independent final audit against the exact commit and receipts, with both standards/code-quality and intent/contract verdicts. Resolve every P0/P1 finding and record lower-priority dispositions in `final-verification.md`.

- [ ] **Step 5: Commit final verification evidence**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/final-verification.md openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git diff --cached --check
git diff --cached
git commit -m "chore(openspec): finalize training contract evidence"
```

Expected: every checked task is supported by executed evidence; no source, generated payload, or unrelated documentation is accidentally staged.

- [ ] **Step 6: Sync and archive only on complete evidence**

If every task and gate passes, use `openspec-sync-specs` to synchronize the four qualified deltas, verify the merged stable blocks, then use `openspec-archive-change`. If any gate fails, keep the change active or archive it explicitly incomplete with `disposition.md`; do not alter stable specs and do not claim exact-resume support.

Expected after successful sync:

```bash
openspec validate --all --strict
```

Expected: all stable specs and active changes pass strict validation, and the archive contains complete evidence rather than unchecked claims.
