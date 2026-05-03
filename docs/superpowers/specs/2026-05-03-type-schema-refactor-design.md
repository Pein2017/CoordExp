# Type Schema Refactor Design

Status: active super-power design, backfilled after initial implementation.

Date: 2026-05-03

Owner: CoordExp architecture and training infrastructure

Target worktree: `/data/CoordExp/.worktrees/refactor-type-schema`

Branch: `codex/refactor-type-schema`

## Purpose

CoordExp has several mature typed islands, but many important boundaries still
carry domain concepts as raw `dict`, `list`, `tuple`, or broad
`Mapping[str, Any]` payloads. This makes reproducibility artifacts, training
state, cache metadata, prediction records, and runtime contracts harder to
reason about and easier to drift silently.

This refactor makes schema-bearing boundaries explicit while preserving current
artifact compatibility. The design is deliberately incremental: introduce typed
objects where a container represents a meaningful contract, keep dynamic logging
maps as maps, and avoid changing serialized JSON/YAML keys unless the relevant
stable contract and tests are updated in the same slice.

## Backfill Note

The first implementation slice landed before this design document existed. This
spec records the intended scope and current code state after that slice:

- Completed: encoded-sample cache request, shard, and manifest typed internals.
- Completed: initial architecture audit report plus progress/catalog routing for the audit.
- Completed: fast-forward merge of committed upstream `codex/compact-detection-sequence`
  through commit `1ed47b3` into `codex/refactor-type-schema`.
- Not completed: whole-codebase canonicalization of cache producers, run
  metadata, static packing, compact detection runtime/artifact payloads,
  prediction/eval records, or Stage-2 runtime state.

The matching execution plan is
`docs/superpowers/plans/2026-05-03-type-schema-refactor.md`.

## Scope

### In Scope

- Repository-wide inventory of schema-bearing containers in `src/`, `tests/`,
  `scripts/`, `configs/`, `docs/`, `progress/`, and current `openspec/specs/`.
- Dataset and training cache contracts, starting with
  `src/datasets/encoded_sample_cache.py`.
- Strict config schema alignment when runtime types correspond to YAML
  contracts.
- Run metadata and experiment/artifact summaries when they carry stable domain
  concepts rather than local scratch data.
- Static packing plan and manifest artifacts.
- Prediction/eval ingestion records, especially `pred`, `predictions`, and
  `objects` aliases.
- Stage-2 prepared segments, checkpoint/runtime state, and module config
  payloads.
- Tests and docs needed to keep each refactor slice globally consistent.
- The committed compact detection and latest recursive detection launch baseline
  through `1ed47b3`, which is the production-training codebase baseline for
  this refactor. This includes `src/detection/`, latest detection training
  config schema, config loader wiring, dataset preparation, loss/objective
  wiring, packing helpers, trainer wiring, launch configs, the upstream
  recursive detection launch plan, and the corresponding tests.

### Post-Merge Scope Delta

Merging compact upstream through `1ed47b3` changes this from a narrow
encoded-cache cleanup into an umbrella architecture/type-system refactor against
a new detection-training baseline. The compact detection and latest recursive
detection launch work is committed upstream input, not refactor output. The plan
must therefore classify these merged schema-bearing surfaces
before selecting additional code targets:

- launch YAMLs under `configs/stage1/recursive_detection_ce_latest/`,
- latest detection training config objects in `src/config/schema.py`,
- training config loading behavior in `src/config/loader.py`,
- compact detection data and raw-schema normalization in `src/detection/data.py`,
- latest recursive detection dataset preparation in `src/detection/dataset.py`,
- prompt/template/token-span contracts in `src/detection/template.py` and
  `src/detection/tokenization.py`,
- recursive detection objective and loss payloads in `src/detection/objective.py`
  and `src/detection/loss.py`,
- packing and cache-fingerprint helpers in `src/detection/packing.py`,
- detection evaluation and metrics flattening in `src/detection/evaluation.py`
  and `src/trainers/metrics/mixins.py`,
- SFT/trainer/batch-extras wiring touched by the compact branch.
- upstream production-launch constraints in
  `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`.

The compact detection classification gate may choose no code changes, a
dedicated follow-up plan, or a named overlap with the encoded-cache slice. It
must not silently expand the encoded-cache task into a detection-stack redesign.

Baseline launch constraints from the upstream production-training plan are
preserved as inspection inputs: direct `torchrun -m src.sft`, `packing: false`,
`training.encoded_sample_cache.enabled: false`, no bidirectional gating, bf16
with targeted fp32, `per_device_train_batch_size=16`,
`gradient_accumulation_steps=1`, `effective_batch_size=128`, no latest-schema
mAP callback yet, and production launch still running at the time of the
upstream note.

### Out Of Scope For The Completed Initial Slice

- Changing `training.encoded_sample_cache.*` YAML keys.
- Changing encoded-cache `manifest.json` keys.
- Changing `EncodedSampleCacheStore.info()` keys.
- Rewriting `src/sft.py` cache request production.
- Rewriting `src/bootstrap/run_metadata.py`.
- Changing static-packing cache compatibility.
- Changing inference/eval artifact schemas.
- Refactoring Stage-2 trainer runtime state.

These are still in scope for later tasks in the plan, but they are not complete
in the current code state.

### Non-Goals

- Do not refactor every local dictionary simply because it is a dictionary.
  Dynamic metric/logging maps may remain mappings when their keys are naturally
  open-ended.
- Do not edit archived OpenSpec changes or historical `progress/` notes to make
  old snapshots look current.
- Do not introduce Pydantic or a new validation framework unless a future slice
  proves the existing dataclass/strict-schema pattern is insufficient.
- Do not rename artifact keys without updating the stable contract and adding
  roundtrip/migration tests.
- Do not edit upstream HF model files.
- Do not update OpenSpec for ordinary audit findings. Update OpenSpec only when
  serialized config, artifact keys, loss semantics, training behavior,
  evaluation behavior, or normative metric semantics change.

### Boundary Classification Rules

A raw container is refactor-worthy only when it carries a meaningful CoordExp
concept across a module, config/schema, artifact/manifest, trainer/loss,
metrics/evaluation, reproducibility, or run-metadata boundary.

Use these decision labels in the audit and plan:

- `already typed`
- `serialized mapping, intentionally stable`
- `dynamic metric/logging map`
- `local scratch container`
- `cross-module domain object needing refactor`
- `artifact/provenance object needing refactor`
- `needs separate contract/spec before code change`
- `historical-only reference`

This keeps the refactor focused on ambiguity that affects correctness,
maintainability, reproducibility, or extension points rather than replacing
every dictionary for its own sake.

## Rationale

The encoded-sample cache is the safest first slice because it has clear domain
objects and focused tests:

- A cache request is not an arbitrary mapping; it is the resolved request used
  to decide eligibility, fingerprint identity, shard residency, and split
  provenance.
- A manifest is not an arbitrary JSON blob; it is a versioned artifact proving
  whether the cache is building, complete, or failed.
- Shard metadata is a repeated structured record with stable fields.
- Existing cache JSON keys can be preserved while internal code becomes easier
  to validate and extend.

This first slice also establishes the refactor style for later work:

- typed value object at the boundary,
- `from_mapping()` for existing serialized payloads,
- `to_mapping()` for stable artifact compatibility,
- focused negative validation at ingestion,
- tests that prove roundtrip compatibility.

## Current Implementation State

### Upstream Baseline Merge

`codex/refactor-type-schema` now includes committed upstream compact detection
and latest recursive detection launch work from `codex/compact-detection-sequence`:

```text
1ed47b3 docs(plan): record latest recursive detection launch
1668530 feat(sft): wire latest detection dataset into training
d8e6f2c feat(config): add latest recursive detection launch configs
246e31e fix(detection): reject non-canonical coord tokens
5876ba5 feat(detection): add compact detection sequence stack
```

That commit is baseline input for this type-schema refactor, not part of the
encoded-cache checkpoint. The global inventory must therefore include the
new compact detection schema and runtime surfaces:

- `configs/stage1/recursive_detection_ce.yaml`
- `configs/stage1/recursive_detection_ce_latest/`
- `src/config/__init__.py`
- `src/config/loader.py`
- `src/config/schema.py`
- `src/bootstrap/trainer_setup.py`
- `src/data_collators/batch_extras_collator.py`
- `src/data_collators/enrichers.py`
- `src/detection/data.py`
- `src/detection/dataset.py`
- `src/detection/evaluation.py`
- `src/detection/loss.py`
- `src/detection/objective.py`
- `src/detection/packing.py`
- `src/detection/template.py`
- `src/detection/tokenization.py`
- `src/metrics/dataset_metrics.py`
- `src/sft.py`
- `src/trainers/batch_extras.py`
- `src/trainers/metrics/mixins.py`
- compact detection, recursive detection, detection dataset, packing,
  random-order, and SFT preparation contract tests under `tests/`.

The design stance is conservative: keep compact detection dataclasses, strict
config objects, and contract tests as upstream truth unless the global inventory
finds raw containers that cross module, artifact, trainer, loss, or metric
boundaries.

### Completed Files

- `src/datasets/encoded_sample_cache.py`
  - Added `EncodedSampleCacheManifestStatus`.
  - Added `EncodedSampleCacheRequest`.
  - Added `EncodedSampleShard`.
  - Added `EncodedSampleCacheManifest`.
  - Routed `EncodedSampleCacheStore.__init__()` through
    `EncodedSampleCacheRequest.from_mapping()`.
  - Routed shard metadata writes through `EncodedSampleShard.to_mapping()`.
  - Routed manifest validation through `EncodedSampleCacheManifest.from_mapping()`.
  - Routed shard indexing through typed manifest/shard objects.
  - Routed `setup_encoded_sample_cache_for_dataset()` fingerprint parsing through
    `EncodedSampleCacheRequest.from_mapping()`.
- `tests/test_encoded_sample_cache.py`
  - Added request-normalization coverage.
  - Added manifest roundtrip coverage against a real built cache manifest.
- `progress/audits/2026-05-03_type_schema_architecture_audit.md`
  - Added the architecture/type-system audit and first-slice plan.
- `progress/audits/README.md`
  - Linked the audit report.
- `progress/index.yaml`
  - Added the audit report to the progress index.
- `docs/catalog.yaml`
  - Added the audit report to the docs catalog.
- `docs/superpowers/plans/2026-05-03-type-schema-refactor.md`
  - Backfilled the execution plan and marked the completed initial slice.
- `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`
  - This backfilled design/spec document.

### Current Encoded-Cache Request Contract

`EncodedSampleCacheRequest` currently owns:

```python
enabled: bool
root_dir: Path
ineligible_policy: str
wait_timeout_s: float
max_resident_shards: int
dataset_split: str
dataset_jsonl: Any
fingerprint: dict[str, Any]
fingerprint_sha256: str
cache_dir: Path
manifest_path: Path
```

`from_mapping()` behavior:

- Converts `wait_timeout_s` to `float`.
- Rejects non-finite timeout values.
- Resolves `root_dir` to an absolute `Path`.
- Defaults `ineligible_policy` to `error`.
- Defaults `max_resident_shards` to `4` and clamps it to at least `1`.
- Defaults `dataset_split` to `train`.
- Preserves `dataset_jsonl`.
- Canonicalizes the fingerprint mapping by stable JSON representation.
- Rejects enabled requests without `root_dir`; disabled requests may still omit
  it because setup returns no cache.
- Derives `fingerprint_sha256` from the canonical fingerprint and rejects any
  supplied value that does not match.
- Derives `cache_dir` as `root_dir / fingerprint_sha256` and rejects any
  supplied path that does not resolve to that path.
- Derives `manifest_path` as `cache_dir / "manifest.json"` and rejects any
  supplied path that does not resolve to that path.

`to_mapping()` behavior:

- Emits the same request keys consumed by the cache store.
- Includes producer-side derived fields: `fingerprint_sha256`, `cache_dir`, and
  `manifest_path`.

`src/sft.py::_build_encoded_sample_cache_request()` routes producer payloads
through `EncodedSampleCacheRequest.from_mapping(...).to_mapping()`, and bypass
metadata is also emitted from the canonical request object. Dataset setup accepts
either a mapping payload or an `EncodedSampleCacheRequest` instance.

### Current Manifest Contract

`EncodedSampleCacheManifest` currently owns:

```python
version: int
status: Literal["building", "complete", "error"]
fingerprint: dict[str, Any]
fingerprint_sha256: str
dataset_split: str
dataset_jsonl: Any
num_samples: int | None
shard_size: int | None
payload_keys: tuple[str, ...]
shards: tuple[EncodedSampleShard, ...]
build_started_at: str | None
build_completed_at: str | None
build_failed_at: str | None
error: str | None
path: Path | None
```

`from_mapping()` validation:

- Rejects unknown manifest status values.
- Requires `fingerprint` to be a mapping.
- Normalizes missing or falsey `payload_keys` to an empty list.
- Normalizes missing or falsey `shards` to an empty list.
- Raises `TypeError` for truthy non-list `payload_keys`.
- Raises `TypeError` for truthy non-list `shards`.
- Requires each shard entry to be a mapping.
- Converts shard metadata into `EncodedSampleShard` objects.

`to_mapping()` compatibility:

- Preserves the current serialized manifest shape.
- Emits complete-manifest fields only when `status == "complete"`.
- Emits building/error fields according to status.

### Current Shard Contract

`EncodedSampleShard` owns:

```python
shard_index: int
file: str
start: int
end: int
count: int
```

Shard records are still serialized as dictionaries in `manifest.json`, but
construction and indexing now go through the typed object.

## Data Flow

Current encoded-cache data flow after the completed slice:

```text
src/sft.py
  builds encoded_sample_cache request as a dict
  ->
src/datasets/dense_caption.py
  passes request mapping into BaseCaptionDataset
  ->
src/datasets/encoded_sample_cache.py
  setup_encoded_sample_cache_for_dataset()
  -> EncodedSampleCacheRequest.from_mapping()
  -> EncodedSampleCacheStore
  -> manifest read/write
  -> EncodedSampleCacheManifest.from_mapping()
  -> EncodedSampleShard records
```

Future tasks should move the producer side from raw dict construction toward the
same canonical request object, then extend run metadata to a typed wrapper.

## Boundary Cases

- Disabled or missing request: `setup_encoded_sample_cache_for_dataset()` returns
  `(None, None)`.
- Enabled request with missing `root_dir`: setup still fails fast before cache
  construction.
- Non-finite wait timeout: request parsing raises `ValueError`.
- Malformed manifest status: manifest parsing raises `ValueError`.
- Manifest fingerprint that is not a mapping: manifest parsing raises
  `TypeError`.
- Truthy manifest `payload_keys` or `shards` values that are not lists:
  manifest parsing raises `TypeError`. Missing or falsey values are normalized
  to empty lists for non-complete manifests only.
- Malformed complete manifests: missing required complete fields, blank or
  mismatched `fingerprint_sha256`, non-positive `shard_size`, negative
  `num_samples`, empty shard lists for positive sample counts, invalid shard
  ranges, and shard count/range mismatches raise before cache reuse.
- Non-mapping shard metadata: manifest parsing raises `TypeError`.
- Complete manifest roundtrip: `to_mapping()` must equal the JSON payload read
  from `manifest.json`.
- Existing cache artifact keys must remain stable through this initial slice.

## Verification Already Run

The initial implementation was verified with:

```bash
conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py -q
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml')]; print('YAML_OK docs/catalog.yaml progress/index.yaml')"
rtk git diff --check
```

Observed result:

- `tests/test_encoded_sample_cache.py` and
  `tests/test_encoded_sample_cache_runtime_config.py`: `19 passed`.
- `py_compile`: passed.
- YAML parse check: passed.
- `git diff --check`: passed.

After merging committed compact detection upstream `5876ba5`, the combined
baseline plus type-schema slice was verified with:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/detection/__init__.py src/detection/data.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml','configs/stage1/recursive_detection_ce.yaml')]; print('YAML_OK')"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed result:

- `py_compile`: passed.
- YAML parse check: `YAML_OK`.
- `git diff --check`: passed.
- Focused encoded-cache plus compact-detection contract tests: `275 passed`,
  with four pre-existing multiprocessing fork deprecation warnings from
  encoded-cache static-packing tests.

The branch was later fast-forwarded through upstream `1ed47b3`. That adds latest
recursive detection launch configs, `src/detection/dataset.py`, `src/config/loader.py`,
and additional SFT/config tests. The combined baseline plus type-schema slice was
then verified with:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/config/loader.py src/detection/__init__.py src/detection/data.py src/detection/dataset.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py src/sft.py
conda run -n ms python -c "from pathlib import Path; import yaml; paths=[Path('docs/catalog.yaml'), Path('progress/index.yaml')]+sorted(Path('configs/stage1/recursive_detection_ce_latest').rglob('*.yaml')); [yaml.safe_load(p.read_text(encoding='utf-8')) for p in paths]; print('YAML_OK', len(paths))"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_training_dataset.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed result:

- `py_compile`: passed.
- YAML parse check: `YAML_OK 12`.
- `git diff --check`: passed.
- Focused encoded-cache plus latest recursive detection contract tests:
  `294 passed`, with four pre-existing multiprocessing fork deprecation
  warnings from encoded-cache static-packing tests.

## Completion Status

Completed:

- Initial architecture/type-system audit report.
- Audit routing through `progress/audits/README.md`, `progress/index.yaml`, and
  `docs/catalog.yaml`.
- Fast-forward merge of committed compact detection and latest recursive
  detection upstream baseline through `1ed47b3`.
- Post-merge syntax, YAML, whitespace, encoded-cache, and compact-detection
  contract verification.
- Encoded-sample cache request dataclass.
- Encoded-sample cache shard dataclass.
- Encoded-sample cache manifest dataclass.
- Store initialization, manifest validation, and shard indexing through typed
  wrappers.
- Tests for request normalization, manifest roundtrip, and malformed
  complete-manifest rejection.
- Backfilled super-power design/spec and plan.

Not completed:

- Global encoded-cache producer/consumer canonicalization in `src/sft.py`,
  `src/datasets/dense_caption.py`, and `src/bootstrap/run_metadata.py`.
- `openspec/specs/encoded-training-cache/spec.md` update for all current typed
  internals and `max_resident_shards`.
- Static packing typed artifact wrappers.
- Compact detection runtime/artifact payload classification after the upstream
  merge.
- Prediction/eval canonical record adapter.
- Stage-2 runtime-state and prepared-segment typing.
- Module-specific Stage-2 config typing.
- Eval summary and metrics artifact typing.

## Next Design Direction

The user has explicitly allowed whole-codebase hierarchy redesign. The next
execution step should still be sliced:

1. Finish post-merge global inventory and classification so `1ed47b3` compact
   detection and latest recursive detection launch surfaces are explicitly
   routed.
2. Finish encoded-cache global consistency across producer, consumer, config,
   run metadata, docs, and specs.
3. Run the compact detection schema classification gate; create a separate plan
   if it finds cross-boundary detection payloads that need refactoring.
4. Decide whether static packing needs an immediate typed manifest/plan slice or
   only a documented rule-out.
5. Introduce one canonical prediction/eval adapter before touching metrics.
6. Promote Stage-2 runtime state only after the stable prepared-segment boundary
   is explicit.

This preserves the whole-codebase ownership goal while keeping each commit
small enough to verify.
