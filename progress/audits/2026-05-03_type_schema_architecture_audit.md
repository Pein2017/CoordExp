---
doc_id: progress.audits.type-schema-architecture-2026-05-03
layer: progress
doc_type: audit
status: active-audit
domain: architecture
summary: Type-system and schema-boundary audit for raw dict/list/tuple domain payloads across CoordExp.
updated: 2026-05-03
---

# Type-System And Schema-Boundary Audit (2026-05-03)

Scope: `/data/CoordExp/.worktrees/refactor-type-schema` on branch `codex/refactor-type-schema`.

This initial architecture audit looked for high-signal raw `dict`, `list`, `tuple`, `Mapping[str, Any]`, and positional container surfaces that carry meaningful CoordExp concepts: config schemas, dataset/cache artifacts, training runtime state, rollout payloads, prediction records, metrics, and reproducibility artifacts.

Method/scope note: this is not yet an exhaustive repository-wide hit classification. The global inventory and per-hit classification remain tracked in Task 0 of `docs/superpowers/plans/2026-05-03-type-schema-refactor.md`.

Upstream baseline note: after this audit was first drafted, committed branch `codex/compact-detection-sequence` was fast-forward merged into `codex/refactor-type-schema` through `1ed47b3` (`docs(plan): record latest recursive detection launch`). For this refactor, `1ed47b3` is the production-training codebase baseline. The compact detection stack, latest recursive detection launch configs, detection dataset preparation, config loader changes, SFT wiring, and upstream recursive detection launch plan are now baseline scope for the remaining global inventory rather than untracked imported work.

## Current Baseline

- The repo already has a clear style target: stable contracts should use typed structures, and positional tuples should become explicit outputs when the fields have domain meaning. See `docs/standards/CODE_STYLE.md`.
- The healthiest typed pockets are config dataclasses in `src/config/schema.py`, JSONL `TypedDict`s in `src/common/schemas.py`, Stage-1 set-continuation full-suffix dataclasses in `src/trainers/stage1_set_continuation/full_suffix.py`, rollout parse/match contracts in `src/trainers/rollout_matching/contracts.py`, and Stage-2 two-channel `TypedDict`s in `src/trainers/stage2_two_channel/types.py`.
- The compact detection baseline added through `1ed47b3` introduces new schema-bearing surfaces in `configs/stage1/recursive_detection_ce_latest/`, `src/detection/`, `src/detection/dataset.py`, latest detection training config dataclasses in `src/config/schema.py`, `src/config/loader.py`, trainer/loss/objective wiring, packing helpers, SFT production-training orchestration, the upstream recursive detection launch plan, and contract tests. These must be classified by Task 7 before later detection-stack refactor slices choose the next code target.
- The remaining ambiguity is concentrated at orchestration and artifact seams: runtime state, cache manifests, pipeline summaries, prediction records, and wide pass-through config payloads.

## Post-Merge Classification Policy

A raw container is refactor-worthy only when it carries a meaningful CoordExp concept across a module, config/schema, artifact/manifest, trainer/loss, metrics/evaluation, reproducibility, or run-metadata boundary. Local construction dictionaries, dynamic metric/logging maps, and stable serialized JSON/YAML payloads may remain mappings when their openness is intentional and documented.

Classification labels:

- `already typed`
- `serialized mapping, intentionally stable`
- `dynamic metric/logging map`
- `local scratch container`
- `cross-module domain object needing refactor`
- `artifact/provenance object needing refactor`
- `needs separate contract/spec before code change`
- `historical-only reference`

## Post-Merge Inventory Classification

This table is intentionally partial until Task 0 is executed against the merged branch. It records current known rows and the exact table shape required for the exhaustive classification pass.

| Area | File/Symbol | Domain Concept | Current Shape | Boundary Type | Risk | Decision | Follow-Up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| encoded-cache | `src/datasets/encoded_sample_cache.py::EncodedSampleCacheRequest` | cache request | frozen dataclass plus mapping serializer | runtime/artifact | low for internals, medium at producer/run-metadata seams | already typed, extend producer/run-metadata coverage | Tasks 2-6 |
| encoded-cache | `src/datasets/encoded_sample_cache.py::EncodedSampleCacheManifest` | cache manifest | frozen dataclass plus stable JSON mapping serializer; malformed complete manifests reject before reuse | artifact/manifest | low for manifest parsing | already typed | none for manifest internals |
| compact-detection | `src/detection/objective.py` and `src/detection/loss.py` | recursive detection objective/loss payloads | Task 7 routing row, not a completed classification | trainer/loss | classification required before code changes | classify before code changes | Task 7 |
| compact-detection | `src/detection/dataset.py`, `src/config/loader.py`, `configs/stage1/recursive_detection_ce_latest/` | latest recursive detection dataset and launch config preparation | Task 7 routing row, not a completed classification | training input/config | classification required before code changes | classify before code changes | Task 7 |
| compact-detection | `src/sft.py` | latest detection runtime orchestration, prompt shim, dataset selection, packing/cache rejection, static-packing fingerprint interaction | Task 7 routing row, not a completed classification | training orchestration/runtime support | classification required before code changes | classify before code changes | Task 7 |
| compact-detection | `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md` | production recursive detection launch constraints | upstream production-training baseline document | launch/provenance document | medium if overwritten by unrelated refactor docs | inspect and preserve unless launch-status docs are intentionally updated | Task 7 |

## Post-Merge Priority Order

1. Finish encoded-cache producer, consumer, run-metadata, docs, and spec consistency because the initial typed internals already exist and the serialized contract is focused.
2. Classify compact detection baseline surfaces before any detection-stack code refactor.
3. Decide static packing, prediction/eval, and Stage-2 state through separate gates so broad architecture risks do not get hidden inside the cache slice.

## Severity-Ranked Findings

### P1: Stage-2 Runtime State Is A Versionless Raw Dict

Evidence:
- `src/trainers/stage2_two_channel.py:802`
- `src/trainers/stage2_two_channel.py:848`

Current shape:
- Nested `Dict[str, Any]` payloads for checkpoint/runtime state, pending logs, post-rollout segments, raw step buffers, and train-monitor candidates.

Recommended representation:
- A versioned `Stage2RuntimeState` dataclass tree with explicit serializers for pending train rollout logs, post-rollout buffers, raw step buffers, and train-monitor candidates.

Why it matters:
- Resume compatibility and silent schema drift are both high-risk here. A typed state boundary gives migration points and fail-fast roundtrip tests.

Suggested checks:
- Runtime-state serialize/restore roundtrip.
- Legacy payload compatibility fixture.
- Negative tests for malformed state versions and missing required state fields.

### P1: Stage-2 Prepared Segments And Executor Payloads Still Depend On Tuple/Dict Aliases

Evidence:
- `src/trainers/stage2_two_channel/types.py:91`
- `src/trainers/stage2_two_channel/executors.py:147`
- `src/trainers/stage2_two_channel.py:3566`

Current shape:
- `Stage2EncodedSample = Dict[str, Any]`
- `Stage2PreparedSegment = tuple[...]`
- `Stage2BatchMetrics = Dict[str, float]`

Recommended representation:
- `Stage2EncodedSample` protocol or dataclass, `PreparedSegment` dataclass, and metric `TypedDict`s grouped by metric domain.

Why it matters:
- This is the executor/scheduling seam the runtime refactor spec wants isolated. Positional segment tuples make downstream code depend on field order rather than contract names.

Suggested checks:
- Prepared-segment constructor/validation tests.
- Packing-buffer tests that assert fields instead of tuple positions.
- `_segments_only` parity tests for Channel-A and Channel-B.

### P1: Detection Prediction Records Are Not Canonicalized Once

Evidence:
- `src/common/prediction_parsing.py:81`
- `src/infer/engine.py:12`
- `src/eval/detection.py:1140`
- `src/eval/detection.py:2491`
- `src/eval/confidence_postop.py:591`
- `docs/eval/CONTRACT.md:25`

Current shape:
- Raw parse payloads use `objects`, inference artifacts use canonical `pred`, and evaluation still accepts legacy `predictions` in some places.

Recommended representation:
- A single `DetectionRecord` / `PredictionRecord` schema at ingest with canonical `pred`, plus explicit alias rejection or one-time migration.

Why it matters:
- A legacy artifact can score in one path but be dropped or interpreted differently by confidence post-op or duplicate control.

Suggested checks:
- Same-row smoke test through evaluator, confidence post-op, and duplicate control.
- Alias handling test that proves either canonicalization or rejection is consistent.

### P1: Config Pipeline Module Specs Carry Open Mappings Inside A Strict Config Tree

Evidence:
- `src/config/schema.py:2905`
- `src/config/schema.py:3005`
- `configs/stage2_two_channel/base.yaml:68`

Current shape:
- Ordered `tuple[Stage2PipelineModuleSpec, ...]`, but each module still carries `application` and `config` as open mappings.

Recommended representation:
- Per-module dataclasses or `TypedDict`s for known module configs and `Literal`/enum choices for application presets.

Why it matters:
- Stage-2 behavior depends on module order and allowlisted knobs. Open module config mappings weaken the strict-unknown-key policy at a high-impact surface.

Suggested checks:
- Extend `tests/test_stage2_ab_config_contract.py` with module-specific unknown-key negatives and preset validation.

### P1: Static Packing Plans And Cache Artifacts Are Schema-Bearing Lists/Dicts

Evidence:
- `src/datasets/wrappers/packed_caption.py:544`
- `src/datasets/wrappers/packed_caption.py:702`
- `src/datasets/wrappers/packed_caption.py:883`
- `src/datasets/wrappers/packed_caption.py:1070`

Current shape:
- `raw_plan` / `aligned_plan` as list-of-lists of ints, untyped fingerprints, and ad hoc JSON manifest/index payloads.

Recommended representation:
- `PackPlan`, `PackingFingerprint`, `PackingManifest`, and `PackedBatch` dataclasses or `TypedDict`s.

Why it matters:
- Static packing artifacts are reproducibility truth and are already versioned/checksummed. Typed wrappers make plan compatibility explicit.

Suggested checks:
- Existing `tests/test_packing_wrapper.py` plan-cache tests plus manifest roundtrip assertions.

### P2: Encoded-Sample Cache Producer/Run-Metadata Canonicalization Remains Partial

Status: partially resolved by the initial encoded-cache internal-typing slice.

Evidence:
- `src/datasets/encoded_sample_cache.py:140`
- `src/datasets/encoded_sample_cache.py:191`
- `src/datasets/encoded_sample_cache.py:386`
- `src/datasets/encoded_sample_cache.py:473`

Resolved in the initial slice:
- `EncodedSampleCacheRequest`, `EncodedSampleShard`, and `EncodedSampleCacheManifest` now exist as internal dataclasses.
- `EncodedSampleCacheStore.__init__()` normalizes incoming request mappings through `EncodedSampleCacheRequest`.
- Cache shard metadata is written through `EncodedSampleShard.to_mapping()`.
- Manifest validation and shard indexing parse through `EncodedSampleCacheManifest`.
- Complete manifests now require explicit `fingerprint_sha256`, `num_samples`,
  `shard_size`, `payload_keys`, and `shards`; malformed complete manifests are
  rejected before cache reuse.
- Existing JSON artifact keys and `EncodedSampleCacheStore.info()` keys are preserved.

Remaining shape ambiguity:
- `src/sft.py` still produces encoded-cache request dictionaries directly.
- `src/bootstrap/run_metadata.py` still records train/eval encoded-cache info through raw mappings.
- `openspec/specs/encoded-training-cache/spec.md` has not yet been updated for the completed typed internals or current residency field coverage.
- Cached sample payload records remain dictionaries and should be considered only after request/manifest/run-metadata boundaries are globally consistent.

Recommended representation:
- Extend the completed internal wrappers into the producer and run-metadata boundaries, then consider `EncodedSampleRecord` only if sample payload dictionaries become cross-module contracts rather than serialized cache internals.

Why it matters:
- Cache reuse and cross-rank correctness depend on stable request and manifest comparisons. The first slice preserved artifact compatibility, but the producer and run-metadata seams still allow parallel dictionary shapes to drift.

Suggested checks:
- Producer roundtrip tests in `tests/test_encoded_sample_cache_runtime_config.py`.
- Run-metadata wrapper tests in `tests/test_run_metadata_file.py`.
- Global encoded-cache hit classification after Task 0.

### P2: Stage-2 Rollout Meta Is A Wide TypedDict With Nested Raw Concepts

Evidence:
- `src/trainers/stage2_two_channel.py:1735`
- `src/trainers/stage2_two_channel.py:2301`
- `src/trainers/stage2_two_channel.py:3274`

Current shape:
- `Stage2RolloutMeta` is better than raw dicts, but it still stores nested raw lists and string discriminators.

Recommended representation:
- Keep the discriminated-union shape, but promote nested concepts: `BBoxGroup`, `DuplicateClusterStats`, `TriageStats`, `SemanticStopMeta`, `DecodeMode`, and `Stage2Channel`.

Suggested checks:
- Channel-A / Channel-B meta-builder tests and enum coverage for decode mode/channel parsing.

### P2: Eval Summary And Persisted Metrics Shapes Diverge

Evidence:
- `src/eval/artifacts.py:79`
- `src/eval/orchestration.py:120`
- `src/eval/detection.py:2777`
- `src/callbacks/detection_eval.py:58`

Current shape:
- In-memory summary dicts, persisted `metrics.json`, and trainer-flattened metrics use different nested shapes without a first-class provenance marker.

Recommended representation:
- `EvalSummary` and `MetricsArtifact` schemas with fields for score source, duplicate control, per-class metrics, counters, and artifact paths.

Why it matters:
- Downstream consumers can lose whether a number came from raw, guarded, official, or F1-ish evaluation.

Suggested checks:
- Schema assertions for `metrics.json`.
- Callback smoke checks proving raw and guarded metric keys do not collide.

### P2: Batch Extras And Stage-1 Set-Continuation Sample Metadata Use Magic Keys

Evidence:
- `src/data_collators/stage1_set_continuation_collator.py:64`
- `src/data_collators/batch_extras_collator.py:80`
- `src/trainers/batch_extras.py:31`

Current shape:
- Generic batch dicts with string keys such as `set_continuation_meta`, `dataset_labels`, `dataset_segments`, `pack_num_samples`, and `instability_meta_json`.

Recommended representation:
- `Stage1SetContinuationSampleMeta` and `BatchExtras` field types, with a narrow `TrainerBatch` protocol for dict-like trainer input.

Suggested checks:
- Collator contract tests and stash/pop roundtrip tests.

### P2: Compact Detection Baseline Needs Post-Merge Schema Classification

Status: newly in scope after merging upstream through `1ed47b3`.

Evidence:
- `configs/stage1/recursive_detection_ce_latest/`
- `src/config/loader.py`
- `src/detection/data.py`
- `src/detection/dataset.py`
- `src/detection/evaluation.py`
- `src/detection/loss.py`
- `src/detection/objective.py`
- `src/detection/packing.py`
- `src/detection/template.py`
- `src/detection/tokenization.py`
- `src/config/schema.py`
- `src/sft.py`
- `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`
- `tests/test_detection_training_dataset.py`

Current shape:
- The compact detection and latest recursive detection stack appears to include many intentional schema objects and contract tests, but it was merged after the first high-signal audit pass. It therefore has not yet been separated into already-typed contracts, serialized artifact mappings, trainer/runtime payloads, dynamic metric maps, dataset preparation payloads, config-loader payloads, and raw containers that deserve refactoring.
- Batch extras and metric-flattening risk increased after the compact merge because `src/data_collators/batch_extras_collator.py`, `src/trainers/batch_extras.py`, and `src/trainers/metrics/mixins.py` are now part of the compact detection contract surface.
- Dataset/config-loader risk is now in scope because `src/detection/dataset.py`, `src/config/loader.py`, and `configs/stage1/recursive_detection_ce_latest/` define the latest recursive detection training input contract.
- Production orchestration risk is now in scope because `src/sft.py` owns latest detection runtime orchestration, prompt shims, dataset selection, packing/cache rejection, and static-packing fingerprint interaction.
- The upstream launch plan records production constraints that must stay baseline-owned unless intentionally updated: direct `torchrun -m src.sft`, `packing: false`, `training.encoded_sample_cache.enabled: false`, no bidirectional gating, bf16 with targeted fp32, `per_device_train_batch_size=16`, `gradient_accumulation_steps=1`, `effective_batch_size=128`, no latest-schema mAP callback yet, and production launch still running at the time of that note.

Recommended representation:
- Preserve existing compact detection dataclasses, config objects, and tests where they already express the contract. Promote only cross-module payloads, artifact records, loss/objective state, packing manifests, or metric summaries that remain ambiguous after Task 0 classification.

Suggested checks:
- Add compact detection, latest recursive detection dataset, config-loader, SFT orchestration, upstream launch-plan constraints, and launch-config modules to the global inventory table.
- Run compact detection contract tests alongside encoded-cache tests after each overlapping refactor slice.
- Avoid changing compact detection serialized JSON/YAML surfaces without a dedicated contract update.

### P3: Analysis Scripts Have Repeated Ad Hoc Row Shapes

Evidence:
- `src/analysis/qwen3_vl_instance_binding.py:56`
- `scripts/analysis/run_raw_text_coordinate_preburst_margin_probe.py:39`
- `scripts/analysis/run_raw_text_coordinate_fn_suppression_probe.py:37`
- `scripts/analysis/run_raw_text_coordinate_duplicate_burst_probe.py:39`

Current shape:
- Repeated config/row payload patterns expressed as raw dictionaries across analysis scripts.

Recommended representation:
- Shared probe config/result dataclasses and JSONL row `TypedDict`s only after the core runtime surfaces above are stabilized.

## Confirmed OK / Lower Priority

- Core JSONL conversation rows already have shared `TypedDict`s in `src/common/schemas.py`.
- Stage-1 full-suffix internals already use domain dataclasses and can serve as the target style for adjacent refactors.
- Rollout parsing/matching contracts are healthier than the trainer/runtime seams around them.
- Confidence post-op is internally consistent today; targeted tests already cover the expected `pred_confidence.jsonl` and `gt_vs_pred_scored.jsonl` shapes.

## Refactor Plan

1. **Encoded-sample cache request/manifest typed wrappers** (implemented first)
   - Add `EncodedSampleCacheRequest`, `EncodedSampleShard`, and `EncodedSampleCacheManifest` as internal dataclasses.
   - Preserve existing JSON artifact keys and `store.info()` shape.
   - Add request normalization, manifest roundtrip, and malformed-manifest tests.
2. **Post-merge global inventory and classification**
   - Add a classification table with one row per high-signal boundary.
   - Use the labels in `Post-Merge Classification Policy`.
   - Treat local scratch dictionaries, dynamic metric maps, and historical-only references as explicit rule-outs.
3. **Encoded-cache producer/run-metadata consistency**
   - Refresh `src/sft.py` and `src/config/schema.py` snippets against merged compact code before implementation.
   - Extend request/run-metadata wrappers only after focused red tests are written.
   - Preserve `training.encoded_sample_cache.*`, `manifest.json`, and run-metadata keys.
4. **Compact detection and latest recursive detection baseline classification**
   - Classify new `configs/stage1/recursive_detection_ce_latest/`, `src/detection/`, `src/detection/dataset.py`, latest training config, config-loader, packing, loss/objective, SFT, trainer, and metric surfaces introduced through `1ed47b3`.
   - Keep already-typed compact detection contracts as baseline.
   - Record any cross-boundary raw payloads as follow-on tasks rather than silently folding them into unrelated cache work.
5. **Static packing artifact decision**
   - Add explicit plan/fingerprint/manifest wrappers around existing `packed_caption.py` JSON payloads.
   - Preserve cache compatibility and checksums.
   - Extend packing cache tests with roundtrip validation.
6. **Prediction/eval canonical record**
   - Introduce canonical `DetectionRecord` / `PredictionRecord` at ingest.
   - Decide whether legacy `predictions` is rejected or migrated once.
   - Add cross-path eval/confidence consistency tests.
7. **Stage-2 prepared segment and runtime state**
   - Add `PreparedSegment` dataclass and convert executor/packing seams first.
   - Add a versioned `Stage2RuntimeState` serializer after segment contracts are explicit.
   - Add legacy state fixture and roundtrip tests before changing checkpoint payloads.
8. **Stage-2 pipeline module config typing**
   - Add per-module config schemas for known objective/diagnostic modules.
   - Keep a narrow explicit passthrough only where upstream compatibility requires it.
9. **Eval summary and metrics artifact schema**
   - Add `EvalSummary` and `MetricsArtifact` wrappers.
   - Preserve persisted `metrics.json` keys until a separate artifact-contract change is approved.

## Implemented Initial Slice

Current worktree status after the first implementation slice:

- `src/datasets/encoded_sample_cache.py`
  - Added `EncodedSampleCacheManifestStatus`.
  - Added `EncodedSampleCacheRequest`.
  - Added `EncodedSampleShard`.
  - Added `EncodedSampleCacheManifest`.
  - `EncodedSampleCacheStore.__init__()` now normalizes incoming request mappings through `EncodedSampleCacheRequest`.
  - Cache shard metadata is written through `EncodedSampleShard.to_mapping()`.
  - Manifest validation and shard indexing now parse through `EncodedSampleCacheManifest`.
  - Existing cache artifact keys and `EncodedSampleCacheStore.info()` keys are preserved.
- `tests/test_encoded_sample_cache.py`
  - Added `test_encoded_sample_cache_request_normalizes_typed_fields`.
  - Added `test_encoded_sample_cache_manifest_roundtrips_serialized_payload`.
- `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`
  - Backfilled the missing super-power design/spec for this refactor.
- `docs/superpowers/plans/2026-05-03-type-schema-refactor.md`
  - Backfilled the implementation plan and marked already-implemented work complete.

Verified:

```bash
conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py -q
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml')]; print('YAML_OK docs/catalog.yaml progress/index.yaml')"
rtk git diff --check
```

Observed:

- Encoded-cache test cluster: `19 passed`.
- `py_compile`: passed.
- YAML parse check: passed.
- Whitespace check: passed.

Post-merge verification after fast-forwarding `codex/refactor-type-schema` to compact upstream `5876ba5`:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/detection/__init__.py src/detection/data.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml','configs/stage1/recursive_detection_ce.yaml')]; print('YAML_OK')"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed:

- `py_compile`: passed.
- YAML parse check: `YAML_OK`.
- Whitespace check: passed.
- Focused encoded-cache plus compact-detection contract tests: `275 passed`, with four pre-existing multiprocessing fork deprecation warnings from encoded-cache static-packing tests.

Latest post-merge verification after fast-forwarding `codex/refactor-type-schema` through compact upstream `1ed47b3`:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/config/loader.py src/detection/__init__.py src/detection/data.py src/detection/dataset.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py src/sft.py
conda run -n ms python -c "from pathlib import Path; import yaml; paths=[Path('docs/catalog.yaml'), Path('progress/index.yaml')]+sorted(Path('configs/stage1/recursive_detection_ce_latest').rglob('*.yaml')); [yaml.safe_load(p.read_text(encoding='utf-8')) for p in paths]; print('YAML_OK', len(paths))"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_training_dataset.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed:

- `py_compile`: passed.
- YAML parse check: `YAML_OK 12`.
- Whitespace check: passed.
- Focused encoded-cache plus latest recursive detection contract tests: `294 passed`, with four pre-existing multiprocessing fork deprecation warnings from encoded-cache static-packing tests.

Not yet implemented:

- `src/sft.py` does not yet produce cache requests through `EncodedSampleCacheRequest`.
- `src/bootstrap/run_metadata.py` does not yet expose a typed encoded-cache run-metadata wrapper.
- `openspec/specs/encoded-training-cache/spec.md` has not yet been updated for this refactor.
- Static packing, prediction/eval records, and Stage-2 runtime-state refactors remain planned follow-on slices.

## Initial Implementation Acceptance

Status: accepted for the initial encoded-cache internal-typing slice.

Acceptance evidence:

- `tests/test_encoded_sample_cache.py` includes red-green coverage for typed request normalization, manifest serialization/parsing, and malformed complete-manifest rejection before cache reuse.
- Existing encoded-cache reuse, bypass, shard-eviction, and concurrent-load behavior still passes.
- No existing cache artifact key is renamed.
