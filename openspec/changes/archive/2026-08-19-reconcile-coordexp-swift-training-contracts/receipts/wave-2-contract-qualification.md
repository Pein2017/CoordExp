# Wave 2 contract qualification — Task 3

- UTC recorded: `2026-08-12`; CWD: `/data/CoordExp/.worktrees/CoordExp-swift`;
  branch: `coordexp-swift`.
- Pinned commit at execution: `be720f58d82bca79c526bbb3ec1e6dbfa4c95240`
  (docs/evidence-only descendant of the pinned implementation source
  `71dab9772983a4680dfcea742aee949c79960560`; `src/` is unchanged at both).
- No `src/` file was created, edited, or reverted for this receipt.  The
  committed change set is the four Task 3 test owners listed under
  "Test-owner changes", the corrected config delta plus design/plan/tasks and
  this matrix/receipt evidence, and one frozen-identity repair in
  `scripts/probes/coordexp_swift/wave7_exact_resume_sequence.py`
  (`FROZEN_REQUEST_PRODUCER_SOURCE_SHA256`, `540101ea...` -> `7c2e3d46...`).
  That constant went stale at `71dab9772`, which edited the request producer's
  archived OpenSpec authority path without refreshing its frozen hash; the
  strict `!=` identity assertion is preserved and the observed hash was
  recomputed independently.
- Each command below ran in its own isolated process group with no overlapping
  launch.
- Scope: CPU/unit/artifact-fixture behavior only.  Nothing here is a GPU,
  distributed-interruption, production-launch, or exact-continuation receipt.

## Executed commands and results

### Step 1 — strict config matrix (isolated acceptance gate)

```bash
conda run -n ms pytest -q \
  tests/config/test_train_config.py::test_exact_resume_defaults_disabled_and_is_persisted \
  tests/config/test_train_config.py::test_exact_resume_requires_strict_cuda_replay_determinism \
  tests/config/test_train_config.py::test_exact_resume_accepts_same_world_mode_and_resolves_checkpoint_path \
  tests/config/test_train_config.py::test_exact_resume_rejects_unknown_or_incompatible_controls
```

Result: exit `0`; `7 passed in 5.56s`; no failures, no skips.  The four
selected names collect seven nodes because
`test_exact_resume_rejects_unknown_or_incompatible_controls` is parametrized
over `resume0..resume3` (`{"mode": "restart"}`,
`{"mode": "disabled", "checkpoint_dir": "checkpoint"}`, `{"mode": "exact"}`,
`{"mode": "EXACT_SAME_WORLD_SIZE"}`).

This command covers the corrected incompatible-control contract. Fixed-target
review later established that exact mode with a null path is not an invalid
control: it is the publish-only control/parent branch and is covered by the
positive follow-up gate recorded below.

### Step 2 — inference payload and exact sibling

```bash
conda run -n ms pytest -q \
  tests/artifacts/test_checkpoint_payload_identity.py \
  tests/artifacts/test_checkpoint_writer.py \
  tests/artifacts/test_training_state.py
```

Result: exit `0`; `127 passed in 9.40s`; no failures, no skips.  The baseline at
`be720f58d` was `124 passed`; the three added nodes are the disabled-mode
publication test and the two historical/sibling reader tests listed below.

### Step 3 — exact admission, restoration, and lineage

```bash
conda run -n ms pytest -q \
  tests/training/test_exact_resume.py \
  tests/training/test_pipeline_exact_resume.py \
  tests/artifacts/test_run_artifacts.py
```

Result: exit `0`; `105 passed, 6 warnings in 6.40s`; no failures, no skips.  The
baseline at `be720f58d` was `104 passed`; the added node is the inference-only
publication-event guard test.  The six warnings are the known
`multiprocessing.popen_fork` deprecations from the named Gloo failure-control
tests in `tests/training/test_exact_resume.py`.

### Step 4 — historical-reader fixture matrix

```bash
conda run -n ms pytest -q \
  tests/adapters/test_inference_reload_status.py \
  tests/inference/test_pipeline.py \
  tests/inference/test_artifacts.py \
  tests/artifacts/test_training_state.py \
  -k 'checkpoint or payload or training_state or historical or resume'
```

Result: exit `0`; `94 passed, 78 deselected in 9.04s`; no failures, no skips.
The filter selected a non-empty set, so the zero-selection stop condition did
not fire.

### Withdrawn RED and corrected publish-only contract

```bash
conda run -n ms pytest -q \
  tests/config/test_train_config.py::test_exact_publish_only_mode_preserves_null_checkpoint_path
```

The original node expected `ConfigContractError` and failed because exact mode
with a null path is intentionally accepted. Fixed-target review demonstrated
that rejecting it breaks the uninterrupted-control and interrupted-parent
config bundles and input-attestation consumers. The node is replaced with a
positive test that preserves exact mode and a null path. No production-source
change is authorized.

## Test-owner changes (test-only; no `src/` change)

- `tests/config/test_train_config.py::test_exact_publish_only_mode_preserves_null_checkpoint_path`
  — positive control/parent config contract.
- `tests/artifacts/test_checkpoint_writer.py::test_disabled_exact_state_publishes_only_the_inference_payload_and_aliases`
  — green.
- `tests/artifacts/test_run_artifacts.py::test_inference_only_publication_event_cannot_carry_exact_state_identity`
  — green.
- `tests/artifacts/test_checkpoint_payload_identity.py::test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata`
  — green.
- `tests/artifacts/test_checkpoint_payload_identity.py::test_historical_payload_without_a_current_manifest_stays_inference_loadable`
  — green.

## Matrix rows closed by cited nodes

| Delta spec | Scenario | Cited passing nodes |
|---|---|---|
| training-artifacts | Exact state is disabled | `tests/artifacts/test_checkpoint_writer.py::test_disabled_exact_state_publishes_only_the_inference_payload_and_aliases` |
| training-resume | Exact training state is disabled | the same node plus `tests/artifacts/test_run_artifacts.py::test_inference_only_publication_event_cannot_carry_exact_state_identity` and `tests/artifacts/test_checkpoint_payload_identity.py::test_completed_publication_event_atomically_binds_payload_and_progress` |
| training-resume | Exact training state is enabled | `tests/artifacts/test_checkpoint_writer.py::test_exact_state_callback_runs_after_durable_payload_and_before_aliases`; `tests/artifacts/test_training_state.py::test_distributed_rank_contributions_commit_one_complete_atomic_state`; `tests/artifacts/test_training_state.py::test_publication_and_admission_authenticate_every_rank_payload`; `tests/training/test_pipeline_exact_resume.py::test_checkpoint_handler_persists_step3_and_final_exact_publication_events`; `tests/artifacts/test_checkpoint_payload_identity.py::test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata` |
| training-resume | Inference reads a checkpoint with exact state | `tests/artifacts/test_checkpoint_payload_identity.py::test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata` |
| training-artifacts | Existing checkpoint is used | `tests/artifacts/test_checkpoint_payload_identity.py::test_historical_payload_without_a_current_manifest_stays_inference_loadable`; `tests/artifacts/test_checkpoint_payload_identity.py::test_payload_reading_ignores_exact_sibling_and_extra_historical_metadata` |
| training-resume | Historical artifacts contain extra metadata | `tests/artifacts/test_checkpoint_payload_identity.py::test_historical_payload_without_a_current_manifest_stays_inference_loadable`; `tests/artifacts/test_training_state.py::test_model_only_and_uncommitted_checkpoints_are_rejected`; `tests/artifacts/test_training_state.py::test_schema_v1_manifest_is_explicitly_unsupported`; `tests/artifacts/test_training_state.py::test_undeclared_file_is_rejected_before_callback`; `tests/artifacts/test_training_state.py::test_inference_minimal_artifact_type_is_not_exact_state` |

Rows explicitly **not** closed here: `First post-resume update matches the
uninterrupted branch`, `Exact-State Publication Is Atomic Across Ranks`,
`Publication is interrupted before commit`, and the requirement-level `Exact
Resume Is Same-World-Size And Optimizer-Boundary Only`.  Their remaining
evidence is Wave 3 distributed/GPU work, which this task did not run.  The
Step 3 admission and restoration nodes are interface evidence only; they do not
qualify exact continuation.

## Wave 2 gate outcome: **PASS**

The four planned commands pass. The first gate attempt stopped before source
edits on an over-broad RED. Fixed-target review proved the live shape is
intentional: `resume.mode` enables exact-state publication, while a non-null
`checkpoint_dir` selects restore. Exact mode with a null path is required for
the uninterrupted control and interrupted parent. The delta, test, matrix, and
plan are corrected. The positive config and affected consumer command passed
`311` tests after updating only the sequence controller's exact frozen
request-producer source hash for the already-committed OpenSpec archive-path
move. The complete Task-3 interface regression then passed `311` tests. No
source edit was permitted or required; no failure, skip, or zero-selection
remains in the Wave-2 gate.

```bash
conda run --no-capture-output -n ms pytest -q \
  tests/config/test_train_config.py \
  tests/training/test_wave7_exact_resume_config_bundle.py \
  tests/training/test_input_attestation.py \
  tests/training/test_wave7_exact_resume_sequence.py
# 311 passed

conda run --no-capture-output -n ms pytest -q \
  tests/artifacts/test_checkpoint_payload_identity.py \
  tests/artifacts/test_checkpoint_writer.py \
  tests/artifacts/test_training_state.py \
  tests/training/test_exact_resume.py \
  tests/training/test_pipeline_exact_resume.py \
  tests/artifacts/test_run_artifacts.py \
  tests/adapters/test_inference_reload_status.py \
  tests/inference/test_pipeline.py \
  tests/inference/test_artifacts.py
# 311 passed
```

## Claim boundary

Everything above is CPU-only unit and artifact-fixture evidence at
`be720f58d`.  It supports disabled/enabled publication shape, fail-closed
admission, restored cursors and runtime owners at the interface level, and
conservative historical reading through the payload and training-state
readers.  It does not support exact continuation, first-post-resume-update
matching, distributed interruption behavior, live model loading, or any
production-launch claim.
