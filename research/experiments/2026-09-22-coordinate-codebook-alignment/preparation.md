# Worker preparation — coordinate-codebook alignment

Authority: unit.md, new package; prior address-readout package remains CLOSED.
Primary estimand is fitted-image empty-prefix completion. Monitor outcomes are
informative only. No model call or package clock has started.

## Implementation allowlist, frozen before shared edits

Parent integration owner:
- src/training/pipeline.py
- src/artifacts/checkpoints.py
- src/qwen/runtime_loading.py
- src/qwen/loading.py (only if actual model reconstruction requires it)
- src/inference/hf_backend.py (only opt-in dynamic reconstruction)
- tests/training/test_pipeline_assembly.py
- tests/training/test_checkpoint_handler_identity.py
- tests/artifacts/test_checkpoint_writer.py
- tests/inference/test_coordinate_codebook_reload.py (new)
- probes/training_set_completion/coordinate_codebook_alignment/ except child-owned admission.py
- configs/research/coordinate_codebook_alignment/

Address implementation child:
- src/qwen/coordinate_codebook.py (new concrete module)
- tests/qwen/test_coordinate_codebook.py (new)

Config/optimizer child:
- src/config/models.py
- src/optim/parameter_groups.py
- src/optim/trainable_surface.py
- src/optim/factory.py
- tests/optim/test_parameter_groups.py
- tests/optim/test_optimizer_factory.py
- tests/optim/test_trainable_surface_receipts.py
- tests/config/test_coordinate_codebook_config.py (new)
- src/training/pack_cache.py (seeded schedule extension)
- tests/training/test_pack_cache.py (seeded schedule coverage)
- src/config/paths.py (relative codebook/resume path resolution)
- tests/config/test_coordinate_codebook_config.py (relative codebook/resume path coverage)
- probes/training_set_completion/coordinate_codebook_alignment/qualify.py (bounded maintained-entry qualification wrapper)

Admission child:
- probes/training_set_completion/coordinate_codebook_alignment/admission.py (new)
- new package selection JSON/JSONL evidence only; no model execution

All children read unit.md and the relevant maintained owners. Parent supervises
integration and real-model execution. Any necessary allowlist addition is recorded
before editing, with its concrete caller need. No shared adapter edits are currently
planned: reuse warm_start_expand_dora. No copied sidecar training loop, generic
module registry, historical-result edits, annotation edits, lead record edits,
commit or publication. Existing deleted CLAUDE.md/GEMINI.md and all unrelated
changes are preserved. Source captures use the maintained provenance helper.

Resume implementation child (bounded to reusable supervised training resume):
- src/training/pipeline.py (wire config-driven resume, seeded stream skip, and checkpoint state into the existing entry)
- src/training/supervised_trainer.py (start-step/stream-offset continuation while preserving existing constructor callers)
- src/artifacts/checkpoints.py (atomic optimizer/scheduler/RNG/schedule/source-state payload beside existing adapter payload)
- src/training/resume.py (optional concrete state validation/serialization helper only if existing owners cannot remain minimal)
- tests/training/test_training_resume.py (split-run parity and fail-closed incompatibility checks)
- tests/artifacts/test_checkpoint_writer.py (only if existing checkpoint atomicity coverage needs the new state payload)

This addition is required because the existing checkpoint writer only publishes
adapter/embedding/codebook payloads and the existing trainer always starts at
step one; the current pipeline has no reusable optimizer/scheduler/RNG or
deterministic stream continuation path.

## CPU integration repair before model entry

The first maintained cache preparation rejected the auxiliary `_admission` field
in selection-v4 rows. The immutable selection is unchanged. Runtime JSONL files
under `runtime-inputs/` remove only that metadata field; `projection.json` binds
both files and verifies all remaining fields, labels, images and IDs are equal.
The failed cache-preparation.log remains retained. No model call or GPU allocation
occurred. Qualification uses admitted IDs 2299, 13004 and 417044.

Numeric mode is BF16/FlashAttention2 (the maintained packed trainer requires it);
source parity and all native comparison conditions use the same mode. Effective
batch is expressed in maintained packed sequences, with segment-balanced full
response CE across every eligible image segment. Full-data 1/2/4-epoch configs
are prepared only and are not authorized to launch.

### Source-expansion initialization repair allowlist

Lead mechanical feedback authorizes `src/adapters/dora.py` and
`tests/adapters/test_dora_setup.py`: recompute only source-absent zero-B DoRA
magnitudes with the promoted adapter runtime norm precision. Preserve copied
source tensors and the frozen source-parity tolerance. Prior receipts remain
historical; corrected initialization requires fresh model qualification.

### Live reference contract v1 (lead ruling 02)

The previous source-path conflict report incorrectly attributed the reference to
merging. Actual `attach_dora_adapter` uses Transformers Mixin injection; the
separate merge helper was not executed. That explanation is withdrawn.

The independent reference now builds only the payload's mature target set with
`get_peft_model(..., autocast_adapter_dtype=True)` and copies exact saved adapter
values after promotion, matching training's copy policy. It contains no address
module or expansion targets. Scientific evaluation selects this explicit live
runtime for both source and trained conditions. Existing default Mixin callers
remain unchanged. No source tensors are renormalized. Corrected training setup
and optimizer execution are unchanged; prior corrected update receipts are reused.

### Production compatibility check repair

`tests/config/test_train_config.py` was changed under the assigned focused-test
surface to project only four new legacy-neutral serialized defaults before the
frozen historical semantic digest. Nondefault tie=false remains detectable;
the historical fixture and infrastructure allowlist are unchanged. This exact
path addition is recorded after the delegated edit (the prior allowlist named
the config integration tests, not this file). Four focused digest tests pass;
the unrelated 26-profile inventory discrepancy remains outside this package.
