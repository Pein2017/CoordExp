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
- The distributed probe is not authorized by approval of this plan. It requires fresh user authorization immediately before launch, bound to the exact commit, configs, artifact roots, frozen commands, declared comparison policy, and quantitative bounds. Its success arm is a matched pair: one uninterrupted control branch and one parent-to-resumed-child branch, each executing the corresponding next forward and at most one optimizer update.
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

### Task 4: Qualify Distributed Atomic Publication Under Fresh Authorization

**Files:**
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-command-manifest.json`
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-launch-packet.md`
- Create: `openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-terminal-receipt.json`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md`
- Modify: `openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md`
- Read/verify: `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py`, `scripts/probes/coordexp_swift/wave7_exact_resume_interrupt.py`, and their tests; do not reuse them unless their current grammar exactly satisfies this change's bounds.

**Preparatory tooling (commit `test(training): add two-rank exact-resume qualifier`, corrected by `fix(training): make resume qualifier fail closed`):** the Wave-7 controller's grammar is fixed to eight ranks and cannot express `world_size=2`, so Step 2's command manifest should freeze `scripts/probes/coordexp_swift/reconcile_exact_resume_probe.py` (`prepare`/`success-control`/`success-resumed`/`rank-failure`/`interruption`/`verify`, tested by `tests/training/test_reconcile_exact_resume_probe.py`) rather than the Wave-7 scripts. It reuses `src.artifacts.training_state`/`src.config.loader`/`src.prepare_train_cache`; `prepare` now runs real model-free cache preparation (`load_model=False`) behind an injectable seam; `verify` fails closed via `admit_training_state`-based checkpoint comparison plus a required `logging.jsonl` step-2 row on both branches; `success-control`/`success-resumed` require `--commit` bound to the prepare receipt and current `HEAD` and still need a real launch to exercise the injectable `launch` seam for a genuine forward/update.

**Interfaces:**
- Consumes: the exact post-Task-3 commit and already passing deterministic multi-rank control-plane tests.
- Produces: one immutable manifest of exact argv arrays, one freshly authorized quantitative launch packet, and target-bound receipts for the matched success branch pair plus rank-failure and interruption arms.

- [ ] **Step 1: Prove the model-free multi-rank control plane first**

Run:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
conda run -n ms pytest -q tests/training/test_exact_resume.py tests/training/test_pipeline_exact_resume.py -k 'publish or contribution or rank or interrupt or atomic or alias or event'
```

Expected: cited nodes cover one complete unique contribution per rank, missing/duplicate/malformed/corrupt contributions, one-rank failure convergence without hang, and interruption boundaries. A zero-test selection or missing scenario returns to Task 2 for a test-first re-plan.

- [ ] **Step 2: Freeze the command manifest without launching**

Inspect the current probe CLI and use `apply_patch` to create strict JSON whose fields are `schema`, `implementation_commit`, `cwd`, `world_size`, `arms`, `setup_command`, `commands`, `config_files`, `artifact_root`, `comparison_policy`, `verification_command`, and `authorization_status`. Record the exact observed commit, absolute paths, and argv tokens; `schema` is exactly `coordexp-swift-reconcile-resume-probe-command-manifest-v1`, `cwd` is exactly `/data/CoordExp/.worktrees/CoordExp-swift`, `world_size` is exactly `2`, `arms` is exactly `success, rank_failure, interruption` in that order, and `authorization_status` is exactly `not_requested`. `setup_command` is the single model-free `prepare` argv and `verification_command` is the single durable-artifact `verify` argv; both are authorized and executed at most once with the arm commands. `commands["success"]` and `config_files["success"]` each have exactly `uninterrupted_control` and `resumed_child` branch keys. The `resumed_child` command owns both its one-step parent setup and one-step resumed child launch; `config_files["success"]["resumed_child"]` names both generated files as `setup_parent` and `resumed_child`. The failure-shaped commands are single representative model-free argv entries and their config-file values are `null`; exhaustive injected failure and interruption-boundary coverage remains the already-executed Step-1 test gate rather than extra launch commands. Each success branch resolves exactly one corresponding next forward and at most one applied optimizer update after its boundary, each failure-shaped arm resolves zero model forwards and zero applied optimizer updates, and the artifact root is an absent absolute path with separate arm descendants. `comparison_policy` names the exact fields and comparison rules for input/pack identity, pre-forward state, objective/loss fields, and resulting trainable parameters. Do not execute any command in the manifest during this step.

Because `prepare` creates the config files and cache only after fresh authorization, compute the three expected generated config-file SHA256 values read-only from the deterministic renderer, the exact absolute base-config path, and the absent artifact root. Record those expected values in `config_files`; after `prepare`, execution MUST compare each generated file against the pre-authorized value before any success launch. A mismatch stops without retry.

Validate the authored file without launching:

```bash
conda run -n ms python - <<'PY'
import json
from pathlib import Path

path = Path("openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-command-manifest.json")
payload = json.loads(path.read_text(encoding="utf-8"))
assert set(payload) == {
    "schema", "implementation_commit", "cwd", "world_size", "arms",
    "setup_command", "commands", "config_files", "artifact_root",
    "comparison_policy", "verification_command", "authorization_status",
}
assert payload["schema"] == "coordexp-swift-reconcile-resume-probe-command-manifest-v1"
assert len(payload["implementation_commit"]) == 40
assert all(char in "0123456789abcdef" for char in payload["implementation_commit"])
assert payload["cwd"] == "/data/CoordExp/.worktrees/CoordExp-swift"
assert payload["world_size"] == 2
assert payload["arms"] == ["success", "rank_failure", "interruption"]
assert set(payload["commands"]) == set(payload["arms"])
assert set(payload["config_files"]) == set(payload["arms"])
branches = {"uninterrupted_control", "resumed_child"}
assert set(payload["commands"]["success"]) == branches
assert set(payload["config_files"]["success"]) == branches
assert isinstance(payload["setup_command"], list) and payload["setup_command"]
assert isinstance(payload["verification_command"], list) and payload["verification_command"]
assert all(isinstance(payload["commands"]["success"][branch], list) and payload["commands"]["success"][branch] for branch in branches)
assert Path(payload["config_files"]["success"]["uninterrupted_control"]["path"]).is_absolute()
assert set(payload["config_files"]["success"]["resumed_child"]) == {"setup_parent", "resumed_child"}
assert all(Path(item["path"]).is_absolute() for item in payload["config_files"]["success"]["resumed_child"].values())
assert all(len(item["expected_sha256"]) == 64 for item in (
    payload["config_files"]["success"]["uninterrupted_control"],
    *payload["config_files"]["success"]["resumed_child"].values(),
))
assert all(isinstance(payload["commands"][arm], list) and payload["commands"][arm] for arm in ("rank_failure", "interruption"))
assert payload["config_files"]["rank_failure"] is None
assert payload["config_files"]["interruption"] is None
assert Path(payload["artifact_root"]).is_absolute()
assert not Path(payload["artifact_root"]).exists()
assert isinstance(payload["comparison_policy"], dict) and payload["comparison_policy"]
assert payload["authorization_status"] == "not_requested"
PY
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Prepare the quantitative launch packet**

Use `apply_patch` to record in `wave-3-launch-packet.md`: manifest SHA256, implementation commit, config SHA256 values, artifact-root absence proof, exact GPU identities/occupancy, required free disk, model-forward and collective ceilings per rank and branch/arm derived from the resolved configs, per-branch/arm and total wall-time limits, per-rank CPU RSS/GPU-memory high-water limits, new-artifact-byte limit across both success branches and all arms, the declared exact comparison policy, stop conditions, and no-retry rule.

Run the read-only preflight:

```bash
cd /data/CoordExp/.worktrees/CoordExp-swift
git status --short
git rev-parse HEAD
sha256sum openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-command-manifest.json
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader
df -B1 /data/CoordExp/.worktrees/CoordExp-swift
```

Expected: commit and manifest bindings match, the artifact root remains absent, and numeric headroom satisfies every declared ceiling. Any drift, occupied target, or insufficient headroom stops before authorization.

- [ ] **Step 4: Pass pre-cost qualification and obtain fresh authorization**

Obtain the single independent pre-cost/distributed-qualification audit against the frozen commands, bounds, comparison policy, and model-free control-plane evidence; resolve every P0/P1 finding. Then present the unchanged packet and request authorization immediately before execution. Expected: explicit approval bound to the recorded commit, manifest digest, configs, artifact root, exactly two GPUs, `world_size=2`, the success pair plus two failure-shaped arms, exactly one next forward and at most one applied optimizer update per success branch, and the recorded failure-arm bounds. Earlier approvals and this plan do not satisfy this step.

- [ ] **Step 5: Execute exactly the authorized argv arrays once**

Run each argv array exactly as frozen, through the `ms` environment, in the order `setup_command`, `uninterrupted_control`, `resumed_child`, `rank_failure`, `interruption`, `verification_command`. Immediately after `setup_command`, verify the prepare receipt signature, implementation commit, private cache receipt, and all three generated config SHA256 values against the packet before any model launch. Stop without retry if the commit/command/config changes, the target becomes occupied, a timeout/hang/OOM occurs, or any declared resource/forward/collective/artifact bound is exceeded.

Expected: `wave-3-terminal-receipt.json` binds every rank and branch, command/commit/config/artifact identity, terminal status, resource maxima, and stop outcome. Before the corresponding next forward it compares input/pack identity and trainable/optimizer/scheduler/scaler/RNG/cursor state; after both branches' first corresponding optimizer update it compares objective/loss fields and resulting trainable parameters under the frozen policy. It also proves aliases/events appear only after the authenticated exact-state manifest; failure and interruption leave no resumable alias/event, and any committed inference payload remains inference-only.

- [ ] **Step 6: Verify durable artifacts, not console status**

Run the frozen verification command from the manifest against the final artifact tree. Expected: every required rank and success branch is accounted for, all matched state/sequence/objective/result comparisons are target-bound, the failure-shaped artifacts satisfy publication semantics, and the claim is limited to same-world-size optimizer-step-boundary continuation. This is a focused executable gate, not another independent audit. Exit zero, admission, or payload presence alone is insufficient.

- [ ] **Step 7: Commit the immutable distributed qualification packet**

```bash
git add openspec/changes/reconcile-coordexp-swift-training-contracts/evidence-matrix.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-command-manifest.json openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-launch-packet.md openspec/changes/reconcile-coordexp-swift-training-contracts/receipts/wave-3-terminal-receipt.json openspec/changes/reconcile-coordexp-swift-training-contracts/tasks.md
git diff --cached --check
git diff --cached
git commit -m "test(training): qualify atomic exact-state publication"
```

Expected: the commit contains only the authorization/command/terminal evidence, matrix updates, and completed task marks; large probe payloads stay in their bound artifact root.

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

Run only the already authorized, frozen verifier argv arrays against the final artifact tree; do not relaunch training without a new authorization packet.

Expected: next input/pack identity, declared pre-forward state, first-update objective/loss fields and resulting trainable parameters, manifest/event/selector identities, rank/branch inventory, and historical-reader classification remain bound to the final commit and durable artifacts. If implementation changed after Task 4, the old probe is stale and this step stops for a new packet and fresh authorization for both success branches.

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
