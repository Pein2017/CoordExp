# Implementation Receipt

## Scope and commits

- Change: `add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot`
- Worktree: `/data/CoordExp/.worktrees/app-own-prefix-calibration-pilot`
- Starting commit: `2d3ea09ee988c9babb02ba5ae75fc640a2a80417`
- Implementation commit: `HEAD` containing this receipt. Its exact immutable SHA is produced after the commit and reported in the final handoff; a commit cannot embed its own SHA.
- Formal 256-image and 12-job screen: not launched.

## Implemented owner surfaces

- `src/rollout_calibration/`: immutable reviewed state-bank assembly and loading, strict checkpoint/token/image/split/count validation, compact validation receipt, exact-token replay, and atomic event planning.
- `src/losses/rollout_calibration.py`: 32-bit floating-point grouped entity-transition, first-wrong-coordinate, and intended token-type-gate math.
- `src/training/rollout_calibration.py`: event-balanced streaming loss integration, global planned-step denominators, joint-arm atomic windows, diagnostics, and existing-trainer loss contexts.
- `src/config/`: strict rollout-only profiles, forbidden-surface checks, manifest-only checkpoint binding, and resolved-config bank identity evidence.
- `src/training/pipeline.py` and `src/training/supervised_trainer.py`: one explicit calibration branch reusing the existing Qwen forward, `SupervisedTrainer`, Weight-Decomposed Low-Rank Adaptation warm start, selected-token embedding delta, optimizer, Accelerate runtime, gradient clipping, scheduler, and checkpoint writer.
- `src/artifacts/run_writer.py`: immutable rank-zero run binding with the validated bank receipt; no new run-tree file inventory.
- Existing inference code was not changed and has no state-bank or calibration-controller dependency.

## Verification evidence

The final targeted contract suite passed with `352 passed`:

```text
conda run -n ms pytest -q tests/config tests/rollout_calibration tests/losses tests/training/test_rollout_calibration_integration.py tests/training/test_supervised_trainer.py tests/training/test_pipeline_assembly.py tests/training/test_train_entry.py tests/artifacts/test_run_artifacts.py tests/artifacts/test_checkpoint_writer.py tests/adapters/test_inference_reload_status.py tests/inference/test_execution_model.py tests/inference/test_execution_model_composition.py tests/inference/test_pipeline.py tests/qwen/test_special_token_embeddings.py
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

## Limitations and deviations

- No deviation from the frozen scientific objective or inference architecture was introduced.
- Calibration configs disable scheduled training-time evaluation; ordinary free-row inference remains the separate post-checkpoint smoke required by the research unit.
- The implementation deliberately uses one complete calibration event per isolated packed micro-step. This is stronger than the required same-planned-step atomicity and fails visibly when the complete candidate group or enabled joint-family window cannot fit.
- The formal screen remains blocked pending lead-agent worktree comparison, publication of the shared fixture, successful Smoke A/B evidence, and separate user authorization.
