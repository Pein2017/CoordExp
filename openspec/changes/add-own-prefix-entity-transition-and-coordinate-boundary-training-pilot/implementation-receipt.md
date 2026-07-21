# Implementation Receipt

This receipt originally recorded the independent implementation race before
the lead-run research screens. Its historical statements about missing shared
pre-fork fixtures remain true for that race. The later addendum below is the
current authority for tasks 7 through 10 and must not be read as retroactively
satisfying procedural tasks 0.1 or 1.1.

## Scope and commits

- Change: `add-own-prefix-entity-transition-and-coordinate-boundary-training-pilot`
- Worktree: `/data/CoordExp/.worktrees/app-own-prefix-calibration-pilot`
- Starting commit: `2d3ea09ee988c9babb02ba5ae75fc640a2a80417`
- Implementation commit: `HEAD` containing this receipt. Its exact immutable SHA is produced after the commit and reported in the final handoff; a commit cannot embed its own SHA.
- Mandatory repair base: `485e5a4b8c2d1d9c498bc94e5b75d716830abbeb`.
- Mandatory repair commit: `HEAD` containing this updated receipt; its exact SHA is reported in the repair handoff.
- Formal 256-image and 12-job screen at the time of this original receipt: not
  launched. Later lead-run evidence is recorded in the addendum below.

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

## Historical Smoke and Runtime Observations at Implementation-Race Close

- The required lead-owned Smoke A/B fixture manifest is absent. No substitute checkpoint, state bank, event, optimizer value, loss value, seed, or blind-cohort image was used.
- At that time, tasks `0.1`, `1.1`, `2.1`, and `5.1` through `5.3` remained
  open. Real exact-prefix source-logit parity, margin movement after a tiny
  update, produced-checkpoint inference, runtime duration, Graphics-Processing-
  Unit peak memory, and multi-rank execution were not claimed by the race
  receipt.
- Synthetic CPU tests establish schema, replay construction, compact-logit mapping, loss gradients, event/global normalization, source identity rejection, artifact evidence, and unchanged inference composition. They are not reported as real smoke evidence.
- Runtime and peak-memory observation: not measured because no authorized real-model smoke could be run without the immutable shared fixture.
- Real step-zero source-logit parity, pre/post-update target-margin movement, and
  checkpoint reload through ordinary inference remain explicitly `not_run` in the
  durable qualification receipt until the shared fixture is available. The repair
  therefore qualifies the code path for the shared Smoke A attempt but does not
  claim that the full frozen pre-smoke gate or Smoke A has passed.

## Subsequent Lead-Run Research Addendum

After the implementation race, the lead research loop exercised this path with
hash-bound real artifacts. The latest completed successor is the Best Sampled
Trajectory Positive Row Imitation Screen:

- immutable bank: 512 events from 118 images, bank identifier
  `c496a3653f6f46539d6bce2e110729c7ee77668fab751a5957bd60fe7829fde3`;
- one-event real-model smoke: exact replay, finite gradient, checkpoint reload,
  and ordinary inference completed;
- full training: one eight-Graphics-Processing-Unit epoch, learning rate
  `1e-5`, sixteen optimizer steps, saved at steps 5, 10, 15, and 16;
- evaluation: clean greedy train-256 at steps 10, 15, and 16, plus the twelve
  human-refined development images at steps 5, 10, 15, and 16;
- route-level shift: matches on the fixed route-added owner set rise from 93
  for the separate clean Source rollout to 107, 109, and 110, while ordinary
  owner retention falls from 712 to 700, 697, and 697. Only 118 of the 238
  route-added owners are direct positive-row event targets; direct-target
  matches move from 61 to 62, 63, and 63, while most of the shift occurs on
  other owners in the same selected-route family;
- decision: do not run the identical 1,024-image replication; retain the
  positive-credit implementation for a matched-arm, preservation-aware
  256-image successor.

Evidence:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/
```

The durable interpretation is in:

```text
research/investigations/qwen3-vl-dense-enumeration/experiments/
2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/results.md
```

The original unchecked pre-fork fixture tasks remain visible in `tasks.md`.
They are historical process gaps, not claims that the later hash-bound screen
or real-model smoke did not run.

## Limitations and deviations

- No deviation from the frozen scientific objective or inference architecture was introduced.
- Calibration configs disable scheduled training-time evaluation; ordinary free-row inference remains the separate post-checkpoint smoke required by the research unit.
- The implementation deliberately uses one complete calibration event per isolated packed micro-step. This is stronger than the required same-planned-step atomicity and fails visibly when the complete candidate group or enabled joint-family window cannot fit.
- The original implementation-race launch was blocked at closeout. The later
  lead-run addendum records the authorized real smoke and training successor;
  only the explicitly historical pre-fork fixture tasks remain unresolved.
