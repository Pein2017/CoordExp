# Implementation Receipt

## Scope and commits

- Change: `add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot`
- Worktree: `/data/CoordExp/.worktrees/app-own-prefix-calibration-pilot`
- Starting commit: `2d3ea09ee988c9babb02ba5ae75fc640a2a80417`
- Implementation commit: `HEAD` containing this receipt. Its exact immutable SHA is produced after the commit and reported in the final handoff; a commit cannot embed its own SHA.
- Mandatory repair base: `485e5a4b8c2d1d9c498bc94e5b75d716830abbeb`.
- Mandatory repair commit: `HEAD` containing this updated receipt; its exact SHA is reported in the repair handoff.
- Formal 256-image and 12-job screen: not launched.

## Implemented owner surfaces

- `src/rollout_calibration/`: immutable reviewed state-bank assembly and loading, strict checkpoint/token/image/split/count validation, compact validation receipt, exact-token replay, and atomic event planning.
- `src/losses/rollout_calibration.py`: 32-bit floating-point grouped entity-transition, first-wrong-coordinate, and intended token-type-gate math.
- `src/training/rollout_calibration.py`: event-balanced streaming loss integration, global planned-step denominators, joint-arm atomic windows, diagnostics, and existing-trainer loss contexts.
- `src/config/`: strict rollout-only profiles, forbidden-surface checks, manifest-only checkpoint binding, and resolved-config bank identity evidence.
- `src/training/pipeline.py` and `src/training/supervised_trainer.py`: one explicit calibration branch reusing the existing Qwen forward, `SupervisedTrainer`, Weight-Decomposed Low-Rank Adaptation warm start, selected-token embedding delta, optimizer, Accelerate runtime, gradient clipping, scheduler, and checkpoint writer.
- `src/artifacts/run_writer.py`: immutable rank-zero run binding with the validated bank receipt; no new run-tree file inventory.
- Existing inference code was not changed and has no state-bank or calibration-controller dependency.

## Mandatory repair qualification

- Replaced arbitrary candidate provenance with typed, exact generation policy,
  checkpoint, prompt-token, and prefix-token identities. Every accepted event now
  has exactly one producer-declared greedy candidate, which is exactly the sole
  harmful branch; positives must be producer-declared same-prefix samples.
- Added ordered per-prefix-row physical-owner proofs with typed reviewer
  provenance. Positive owners must be absent from the covered set and duplicate
  harmful owners must be present before aliases can share owner weight.
- Replaced the single claimed wrong boundary with ordered `x1`, `y1`, `x2`, `y2`
  observations through the first wrong boundary. Earlier values must be accepted,
  the selected value must be rejected, all token identities are resolved through
  the active coordinate-token order, and the correction owner must equal the
  trusted candidate owner.
- Owner-resolution intervals now begin at candidate offset zero. Single-objective
  token-type gates include only the sites selected by the active objective.
- Removed the calibration scheduler's hidden family injection and modulo replay.
  The resolved schedule must expose every profile-admitted frozen-bank event
  exactly once; missing joint-family evidence remains visible at the planned-step
  loss gate.
- Exact replay now validates one contiguous active Qwen image-placeholder run and
  requires its count to equal the active processor's merged visual-token count.
- Step-zero warm-start parity now rejects initialized target tensors, ignored
  source tensors, or failed post-copy equality. The loaded selected-token
  embedding delta is frozen; configuration and runtime receipts require the
  optimizer to own language-tower Weight-Decomposed Low-Rank Adaptation parameters
  only.
- Rank-zero `run.json` now persists source and parity evidence, exact trainable and
  frozen surface groups, explicit not-run real-smoke fields, and the finite nonzero
  post-backward gradient norm plus optimizer-update status when a calibration step
  executes. Redundant profiling synchronizations around each forward and backward
  were collapsed.

## Verification evidence

The repaired targeted contract suite passed with `373 passed`:

```text
conda run -n ms pytest -q tests/config tests/rollout_calibration tests/losses tests/optim/test_trainable_surface_receipts.py tests/training/test_rollout_calibration_integration.py tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/training/test_train_entry.py tests/artifacts/test_run_artifacts.py tests/artifacts/test_checkpoint_writer.py tests/adapters/test_inference_reload_status.py tests/inference/test_execution_model.py tests/inference/test_execution_model_composition.py tests/inference/test_pipeline.py tests/qwen/test_special_token_embeddings.py
```

Additional checks passed:

```text
conda run -n ms ruff check src/artifacts/run_writer.py src/config/loader.py src/config/models.py src/config/paths.py src/losses/__init__.py src/losses/rollout_calibration.py src/rollout_calibration src/training/pipeline.py src/training/rollout_calibration.py src/training/supervised_trainer.py tests/artifacts/test_run_artifacts.py tests/config/test_rollout_calibration_config.py tests/losses/test_rollout_calibration.py tests/rollout_calibration tests/training/test_rollout_calibration_integration.py tests/training/test_supervised_trainer.py
conda run -n ms ruff format --check src/artifacts/run_writer.py src/config/loader.py src/config/models.py src/config/paths.py src/losses/__init__.py src/losses/rollout_calibration.py src/rollout_calibration src/training/pipeline.py src/training/rollout_calibration.py src/training/supervised_trainer.py tests/artifacts/test_run_artifacts.py tests/config/test_rollout_calibration_config.py tests/losses/test_rollout_calibration.py tests/rollout_calibration tests/training/test_rollout_calibration_integration.py tests/training/test_supervised_trainer.py
openspec validate add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot --strict
git diff --check
```

Residue inspection found one training-loop owner, `SupervisedTrainer`, and no inference-side state-bank import, rollout collector, alternate trainer, custom inference controller, or online refresh path. Independent contract and runtime reviewers both returned `APPROVE` after bounded fixes.

## Smoke and runtime observations

- The required lead-owned Smoke A/B fixture manifest is absent. No substitute checkpoint, state bank, event, optimizer value, loss value, seed, or blind-cohort image was used.
- Therefore tasks `0.1`, `1.1`, `2.1`, and `5.1` through `5.3` remain open. Real exact-prefix source-logit parity, margin movement after a tiny update, produced-checkpoint inference, runtime duration, GPU peak memory, and multi-rank GPU execution were not claimed.
- Synthetic CPU tests establish schema, replay construction, compact-logit mapping, loss gradients, event/global normalization, source identity rejection, artifact evidence, and unchanged inference composition. They are not reported as real smoke evidence.
- Runtime and peak-memory observation: not measured because no authorized real-model smoke could be run without the immutable shared fixture.
- Real step-zero source-logit parity, pre/post-update target-margin movement, and
  checkpoint reload through ordinary inference remain explicitly `not_run` in the
  durable qualification receipt until the shared fixture is available. The repair
  therefore qualifies the code path for the shared Smoke A attempt but does not
  claim that the full frozen pre-smoke gate or Smoke A has passed.

## Limitations and deviations

- No deviation from the frozen scientific objective or inference architecture was introduced.
- Calibration configs disable scheduled training-time evaluation; ordinary free-row inference remains the separate post-checkpoint smoke required by the research unit.
- The implementation deliberately uses one complete calibration event per isolated packed micro-step. This is stronger than the required same-planned-step atomicity and fails visibly when the complete candidate group or enabled joint-family window cannot fit.
- The formal screen remains blocked pending lead-agent worktree comparison, publication of the shared fixture, successful Smoke A/B evidence, and separate user authorization.
