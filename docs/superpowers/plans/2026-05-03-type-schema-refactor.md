# Type Schema Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CoordExp's schema-bearing runtime boundaries globally consistent by replacing ambiguous domain `dict`/`list`/`tuple` payloads with explicit typed value objects or documented typed mappings while preserving serialized artifact contracts.

**Architecture:** Treat this as an umbrella architecture refactor with mandatory classification gates before implementation slices. Each slice introduces types only at meaningful contracts, keeps JSON/YAML artifact keys stable unless the stable contract is intentionally updated, and follows the existing CoordExp pattern of frozen dataclasses plus strict config parsing.

**Tech Stack:** Python dataclasses, `typing`/`TypedDict` where a serialized mapping is the actual contract, existing strict config schema helpers, Serena for Python symbol exploration, `rg`/`rtk` for broad searches, and `rtk conda run -n ms python -m pytest` for noisy test verification.

**Backfill State:** The matching design/spec is `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`. The first encoded-sample cache slice was implemented before this plan/spec pair existed; this plan records that work as completed and separates it from future global-consistency tasks.

---

## Execution Rules

- Use the worktree `/data/CoordExp/.worktrees/refactor-type-schema`.
- Do not edit the parent checkout `/data/CoordExp`.
- Before editing Python files, use Serena symbol or pattern search to identify the symbol boundary.
- Use `rg`/`rtk` for broad repository scans, excluding `output/`, `temp/`, and `.git/`.
- Preserve current JSON/YAML artifact keys unless a task explicitly updates `openspec/specs/` and tests for the contract change.
- Stage and commit only after a task's focused tests pass; keep unrelated dirty files untouched.
- Do not implement broad subsystem refactors from an inventory gate. If a gate finds a cross-boundary raw payload that needs code changes, either route it into the encoded-cache slice when it directly overlaps or create a separate follow-up super-power plan with exact files, exact tests, and exact code snippets.

## Post-Merge Planning Model

This plan has two roles after merging compact upstream through `1ed47b3`, which
is the production-training codebase baseline for this refactor:

1. **Umbrella audit and routing plan.** Task 0 and later decision gates classify schema-bearing containers across the merged branch and record decisions in `progress/audits/2026-05-03_type_schema_architecture_audit.md`.
2. **Executable encoded-cache implementation plan.** Tasks 2-6 are the only concrete code-change slice currently authorized by this plan. They must be refreshed against the merged `src/sft.py`, `src/config/schema.py`, and compact-detection runtime tests before implementation.

Compact detection is baseline input, not dirty refactor work. The compact-detection gate may produce no code change, a dedicated follow-up plan, or a small overlap item for the encoded-cache slice. It must not silently expand the encoded-cache task into a detection-stack redesign.

## Boundary Classification Rules

A raw container is refactor-worthy only when it carries a meaningful CoordExp concept across a boundary:

- module boundary,
- config or strict schema boundary,
- artifact or manifest boundary,
- trainer/loss/objective boundary,
- metrics or evaluation-summary boundary,
- reproducibility or run-metadata boundary.

Use these classification labels in the audit table:

- `already typed`
- `serialized mapping, intentionally stable`
- `dynamic metric/logging map`
- `local scratch container`
- `cross-module domain object needing refactor`
- `artifact/provenance object needing refactor`
- `needs separate contract/spec before code change`
- `historical-only reference`

Do not refactor local construction dictionaries, dynamic logging maps, or serialized JSON/YAML payloads unless the audit identifies a real ambiguity, drift risk, or maintenance burden at one of the boundaries above.

## Current Worktree State

Status: compact upstream merged; encoded-cache contract checkpoint is implemented in this branch and covered by this plan/spec/audit.

`codex/refactor-type-schema` was fast-forwarded from `0cb1a5a` through committed upstream branch `codex/compact-detection-sequence` at `1ed47b3` (`docs(plan): record latest recursive detection launch`). The compact detection stack and latest recursive detection launch support are now part of the refactor baseline rather than local worktree-only content. Treat `1ed47b3` as the source-of-truth upstream production-training commit for config/schema, detection, packing, loss/objective, dataset/trainer wiring, latest launch configs, and compact-detection test surfaces.

Refactor checkpoint scope after the compact upstream merge and selective restoration of type-schema work:

- Included: `docs/catalog.yaml`
- Included: `progress/audits/README.md`
- Included: `progress/index.yaml`
- Included: `src/datasets/encoded_sample_cache.py`
- Included: `tests/test_encoded_sample_cache.py`
- Included: `docs/superpowers/plans/2026-05-03-type-schema-refactor.md`
- Included: `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`
- Included: `progress/audits/2026-05-03_type_schema_architecture_audit.md`

## Verification Profiles

Use named profiles so future tasks run the narrowest realistic check without losing coverage clarity.

### `encoded-cache-contract`

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  -q
```

### `encoded-cache-global`

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_run_metadata_file.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  -q
```

### `compact-detection-contract`

```bash
rtk conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_detection_normalization_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_stage1_json_pretty_template.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_token_span_masks_from_templates.py \
  tests/test_compact_et_rmp_span_contract.py \
  tests/test_random_order_sft_contract.py \
  tests/test_random_permutation_et_rmp_ce_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_sft_preparation_contract.py \
  tests/test_batch_extras_contract.py \
  tests/test_length_insensitive_loss_normalization.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_trainer_mixin.py \
  tests/test_recursive_detection_state_weighting.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_runtime_profile.py \
  -q
```

### `merged-baseline-smoke`

Run `encoded-cache-contract` plus `compact-detection-contract`. Latest observed `1ed47b3` merged-baseline result after complete-manifest hardening: `294 passed`, with four pre-existing multiprocessing fork deprecation warnings from encoded-cache static-packing tests.

## Commit Boundaries

Use this checkpoint commit after verification:

1. `refactor(datasets): type encoded cache contracts`
   - Include `src/datasets/encoded_sample_cache.py`, `tests/test_encoded_sample_cache.py`, and the audit/spec/index/docs files that accurately describe this already-implemented slice.

Future commits after this checkpoint:

2. `refactor(cache): canonicalize encoded cache producer metadata`
   - Include only the later Tasks 2-6 implementation files after they are refreshed, implemented, and verified.

Do not include compact detection or latest recursive detection launch files from `1ed47b3` in these commits. They are already upstream baseline.

## File Ownership Map

### Current Audit And Planning Artifacts

- Modify: `progress/audits/2026-05-03_type_schema_architecture_audit.md`
  - Keep the architecture/type-system audit current as implementation discoveries refine severity, scope, and rule-outs.
- Modify: `progress/audits/README.md`
  - Keep the audit linked from the audit index.
- Modify: `progress/index.yaml`
  - Keep the audit discoverable from the progress index.
- Modify: `docs/catalog.yaml`
  - Keep the audit discoverable from the docs catalog.
- Modify: `docs/superpowers/plans/2026-05-03-type-schema-refactor.md`
  - This file owns the execution sequence and task status.
- Create: `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`
  - Backfilled design/spec covering intended scope, rationale, current implementation, affected files, boundary cases, and completion status.

### Encoded-Sample Cache Contract

- Modify: `src/datasets/encoded_sample_cache.py`
  - Own canonical request, shard, manifest, and runtime info representations.
- Modify: `tests/test_encoded_sample_cache.py`
  - Unit tests for canonical request, manifest, cache-store behavior, and malformed complete-manifest rejection.
- Inspect: `src/datasets/dense_caption.py`
  - No current code change is planned unless Task 0 finds a real encoded-cache boundary in this module.
- Inspect now / modify in future Task 3: `src/sft.py`
  - Produce canonical encoded-cache request payloads and bypass metadata.
- Inspect now / modify in future Task 4: `src/bootstrap/run_metadata.py`
  - Own the run-metadata shape for train/eval encoded-cache info.
- Inspect now / modify only if strict-config tests fail in future Task 5: `src/config/schema.py`
  - Keep strict YAML schema aligned with runtime request/config types.
- Modify: `tests/test_encoded_sample_cache_runtime_config.py`
  - Unit tests for `src/sft.py` request production and bypass metadata.
- Modify: `tests/test_run_metadata_file.py`
  - Unit tests for run-metadata encoded-cache block shape.
- Modify: `tests/test_training_config_strict_unknown_keys.py`
  - Strict config normalization and unknown-key coverage.
- Modify: `tests/test_stage1_static_packing_runtime_config.py`
  - Existing residency-bound config coverage when config/schema semantics change.
- Modify: `openspec/specs/encoded-training-cache/spec.md`
  - Stable contract doc for cache YAML fields, manifest fields, provenance, and residency limits.
- Modify: `docs/data/PACKING.md`
  - Operator guidance for cache eligibility, bypass, and resident shard cap.

### Static Packing Contract

- Inspect: `src/datasets/wrappers/packed_caption.py`
- Inspect: `src/sft.py`
- Inspect: `tests/test_packing_wrapper.py`
- Inspect: `tests/test_stage1_static_packing_runtime_config.py`
- Inspect: `docs/data/PACKING.md`
- Decision: after the encoded-cache slice is globally consistent, either implement a focused `StaticPackingPlan` / `StaticPackingManifest` type slice or record a rule-out in the audit if the current wrappers already expose clear dataclasses at the contract boundary.

### Prediction And Evaluation Record Contract

- Inspect: `src/infer/pipeline.py`
- Inspect: `src/infer/engine.py`
- Inspect: `src/infer/artifacts.py`
- Inspect: `src/eval/detection.py`
- Inspect: `src/eval/artifacts.py`
- Inspect: `src/eval/parsing.py`
- Inspect: `src/eval/confidence_postop.py`
- Inspect: `scripts/evaluate_detection.py`
- Inspect: `scripts/postop_confidence.py`
- Inspect: `scripts/export_coco_submission.py`
- Inspect: `tests/test_unified_infer_pipeline.py`
- Inspect: `tests/test_detection_eval_ingestion_diagnostics.py`
- Inspect: `tests/test_confidence_postop.py`
- Decision: standardize `pred` / `predictions` / `objects` interpretation through one typed adapter before changing downstream metric code.

### Stage-2 And Teacher-Forcing Runtime Contract

- Inspect: `src/trainers/rollout_matching/`
- Inspect: `src/trainers/teacher_forcing/`
- Inspect: `src/trainers/stage1_set_continuation/`
- Inspect: `src/trainers/batch_extras.py`
- Inspect: `tests/test_stage2_ab_training.py`
- Inspect: `tests/test_stage2_two_channel_training.py`
- Inspect: `tests/test_batch_extras_contract.py`
- Inspect: `tests/test_teacher_forcing_token_ce.py`
- Inspect: `tests/test_stage1_set_continuation_branch_runtime.py`
- Decision: convert only stable batch/result/state payloads first; do not refactor metric dictionaries that are intentionally dynamic logging payloads unless they represent a domain state object.

### Compact Detection Upstream Baseline

The compact detection stack and latest recursive detection launch support landed upstream in `codex/compact-detection-sequence` through `1ed47b3` and are now committed baseline for this refactor. Do not re-implement or restage them as refactor work. Do include them in global type-schema inventory and follow-on design decisions.

- Inspect: `configs/stage1/recursive_detection_ce.yaml`
- Inspect: `configs/stage1/recursive_detection_ce_latest/`
- Inspect: `src/config/__init__.py`
- Inspect: `src/config/loader.py`
- Inspect: `src/config/schema.py`
- Inspect: `src/bootstrap/trainer_setup.py`
- Inspect: `src/data_collators/batch_extras_collator.py`
- Inspect: `src/data_collators/enrichers.py`
- Inspect: `src/detection/data.py`
- Inspect: `src/detection/dataset.py`
- Inspect: `src/detection/evaluation.py`
- Inspect: `src/detection/loss.py`
- Inspect: `src/detection/objective.py`
- Inspect: `src/detection/packing.py`
- Inspect: `src/detection/template.py`
- Inspect: `src/detection/tokenization.py`
- Inspect: `src/metrics/dataset_metrics.py`
- Inspect: `src/sft.py`
- Inspect: `src/trainers/batch_extras.py`
- Inspect: `src/trainers/metrics/mixins.py`
- Inspect: `tests/test_latest_training_config_contract.py`
- Inspect: `tests/test_detection_*`
- Inspect: `tests/test_detection_training_dataset.py`
- Inspect: `tests/test_recursive_detection_*`
- Inspect: `tests/test_packing_*`
- Inspect: `tests/test_random_*`
- Inspect: `tests/test_sft_preparation_contract.py`
- Decision: keep compact detection dataclasses and tests as the current upstream contract unless Task 0 identifies ambiguous raw containers that cross module or artifact boundaries.

## Backfilled Completed Work

- [x] **Step 1: Produce initial architecture/type-system audit report**

Created `progress/audits/2026-05-03_type_schema_architecture_audit.md` with high-signal severity-ranked findings for Stage-2 runtime state, prepared segments, prediction records, pipeline module configs, static packing artifacts, encoded-cache artifacts, rollout metadata, eval summaries, batch extras, and analysis-script row shapes. Exhaustive repository-wide hit classification remains pending under Task 0.

- [x] **Step 2: Route audit into repo indexes**

Updated:

```text
progress/audits/README.md
progress/index.yaml
docs/catalog.yaml
```

The current checkpoint has the audit routed through the progress audit index, progress index, and docs catalog.

- [x] **Step 3: Backfill super-power design/spec**

Created:

```text
docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md
```

The spec records:

- intended whole-codebase scope,
- rationale for encoded-cache as the first safe slice,
- affected files,
- current implementation details,
- boundary cases,
- verification already run,
- completed and incomplete work.

- [x] **Step 4: Backfill super-power implementation plan**

Created and expanded:

```text
docs/superpowers/plans/2026-05-03-type-schema-refactor.md
```

This plan now treats already-implemented changes as completed tasks and keeps later whole-codebase redesign work as explicit pending slices.

- [x] **Step 5: Implement encoded-sample cache typed internals**

Modified `src/datasets/encoded_sample_cache.py` to add and use:

```python
EncodedSampleCacheManifestStatus
EncodedSampleCacheRequest
EncodedSampleShard
EncodedSampleCacheManifest
```

The implementation currently normalizes cache requests internally, writes shard metadata through typed records, validates manifests through a typed manifest wrapper, and indexes shards through typed shard records.

- [x] **Step 6: Add encoded-cache tests**

Modified `tests/test_encoded_sample_cache.py` to add:

```python
test_encoded_sample_cache_request_normalizes_typed_fields
test_encoded_sample_cache_manifest_roundtrips_serialized_payload
```

- [x] **Step 7: Verify completed initial slice**

Already run:

```bash
conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py -q
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml')]; print('YAML_OK docs/catalog.yaml progress/index.yaml')"
rtk git diff --check
```

Observed:

- encoded-cache tests: `19 passed`,
- `py_compile`: passed,
- YAML parse check: passed,
- diff whitespace check: passed.

- [x] **Step 8: Merge committed compact detection upstream baseline**

Merged local branch `codex/compact-detection-sequence` into `codex/refactor-type-schema` after the compact worktree was committed.

Observed:

- compact upstream commit: `5876ba5 feat(detection): add compact detection sequence stack`,
- merge mode: fast-forward from `0cb1a5a` to `5876ba5`,
- conflict status: no merge conflicts,
- compact implementation files are now committed baseline, not untracked refactor work,
- refactor-specific audit/spec/cache changes were restored afterward as the only remaining dirty work.

- [x] **Step 9: Verify merged compact baseline plus refactor dirty slice**

Run after the merge:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/detection/__init__.py src/detection/data.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml','configs/stage1/recursive_detection_ce.yaml')]; print('YAML_OK')"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed:

- `py_compile`: passed,
- YAML parse check: `YAML_OK`,
- diff whitespace check: passed,
- focused encoded-cache plus compact-detection contract tests: `275 passed`, with four pre-existing multiprocessing fork deprecation warnings from encoded-cache static-packing tests.

- [x] **Step 10: Merge latest committed recursive detection upstream baseline**

Merged local branch `codex/compact-detection-sequence` into `codex/refactor-type-schema` after the upstream worktree advanced beyond `5876ba5`.

Observed:

- latest upstream commit: `1ed47b3 docs(plan): record latest recursive detection launch`,
- included commits after `5876ba5`:
  - `246e31e fix(detection): reject non-canonical coord tokens`,
  - `d8e6f2c feat(config): add latest recursive detection launch configs`,
  - `1668530 feat(sft): wire latest detection dataset into training`,
  - `1ed47b3 docs(plan): record latest recursive detection launch`,
- merge mode: fast-forward from `5876ba5` to `1ed47b3`,
- conflict status: no merge conflicts,
- latest recursive detection configs, `src/detection/dataset.py`, `src/config/loader.py`, and SFT wiring are now committed baseline.

- [x] **Step 11: Verify latest recursive detection upstream baseline plus refactor dirty slice**

Run after the `1ed47b3` merge:

```bash
conda run -n ms python -m py_compile src/datasets/encoded_sample_cache.py src/config/__init__.py src/config/schema.py src/config/loader.py src/detection/__init__.py src/detection/data.py src/detection/dataset.py src/detection/evaluation.py src/detection/loss.py src/detection/objective.py src/detection/packing.py src/detection/template.py src/detection/tokenization.py src/bootstrap/trainer_setup.py src/data_collators/batch_extras_collator.py src/data_collators/enrichers.py src/trainers/batch_extras.py src/trainers/metrics/mixins.py src/sft.py
conda run -n ms python -c "from pathlib import Path; import yaml; paths=[Path('docs/catalog.yaml'), Path('progress/index.yaml')]+sorted(Path('configs/stage1/recursive_detection_ce_latest').rglob('*.yaml')); [yaml.safe_load(p.read_text(encoding='utf-8')) for p in paths]; print('YAML_OK', len(paths))"
git diff --check
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_raw_schema_contract.py tests/test_detection_training_dataset.py tests/test_detection_normalization_contract.py tests/test_detection_template_registry.py tests/test_detection_stage1_json_pretty_template.py tests/test_detection_compact_full_template.py tests/test_detection_template_span_alignment.py tests/test_detection_template_parsing_eval.py tests/test_token_span_masks_from_templates.py tests/test_compact_et_rmp_span_contract.py tests/test_random_order_sft_contract.py tests/test_random_permutation_et_rmp_ce_contract.py tests/test_recursive_detection_ce_target_builder.py tests/test_sft_preparation_contract.py tests/test_batch_extras_contract.py tests/test_length_insensitive_loss_normalization.py tests/test_packing_cache_fingerprints.py tests/test_packing_template_contracts.py tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_sft_wiring.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_recursive_detection_state_weighting.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_runtime_profile.py -q
```

Observed:

- `py_compile`: passed,
- YAML parse check: `YAML_OK 12`,
- diff whitespace check: passed,
- focused encoded-cache plus latest recursive detection contract tests: `294 passed`, with four pre-existing multiprocessing fork deprecation warnings from encoded-cache static-packing tests.

- [x] **Step 12: Spawn subagents for plan/spec review**

Spawned four review agents against the merged `1ed47b3` baseline and current
type-schema plan/spec/audit.

Accepted review results:

- Treat `1ed47b3` explicitly as the production-training baseline.
- Convert provisional compact rows in Task 0 into Task 7 routing, because
  exhaustive compact classification remains pending.
- Add `src/sft.py` production-training orchestration and
  `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`
  to the compact baseline inspection scope.
- Keep the current checkpoint as one combined code/tests/docs commit, because
  the docs backfill describes implemented cache internals.
- Harden complete-manifest validation and add malformed-manifest tests before
  committing.

## Task 0: Global Post-Merge Inventory And Classification

- [ ] **Step 1: Refresh changed-file scope**

Run:

```bash
git status --short --branch
git diff --name-only
git ls-files --others --exclude-standard
```

Expected: only this refactor's audit/spec/index and encoded-cache files appear as modified or untracked. Compact detection and latest recursive detection launch files must not appear as dirty work because they are committed through `1ed47b3`.

- [ ] **Step 2: Search the whole repository for encoded-cache producers and consumers**

Run:

```bash
rg -n "from src\.datasets\.encoded_sample_cache|import .*encoded_sample_cache|setup_encoded_sample_cache_for_dataset|_encoded_sample_cache|get_encoded_sample_cache_info|encoded_sample_cache" \
  src tests scripts configs docs progress openspec \
  --glob '!output/**' --glob '!temp/**' --glob '!.git/**'
```

Expected: every hit is classified into one of these categories in the audit report: producer, consumer, serialized artifact, config/schema, docs/spec, historical-only progress, archived OpenSpec, or follow-up code-change candidate.

- [ ] **Step 3: Search for adjacent raw schema containers**

Run:

```bash
rg -n "dict\[str, Any\]|Dict\[str, Any\]|list\[dict|List\[dict|tuple\[|Mapping\[str, Any\]|metrics|predictions|objects|losses|runtime_state|prepared_segment|manifest|Detection|packing|objective" \
  src tests scripts configs docs openspec \
  --glob '!output/**' --glob '!temp/**' --glob '!.git/**'
```

Expected: the audit report gets a concise table of high-signal candidates. Intentional logging maps and low-risk local scratch dictionaries are marked as rule-outs.

- [ ] **Step 4: Classify compact detection baseline schema surfaces**

Run:

```bash
rg -n "dict\[str, Any\]|Dict\[str, Any\]|Mapping\[str, Any\]|list\[|tuple\[|TypedDict|dataclass|Enum|Protocol|payload|record|state|loss|metrics|manifest|config|dataset|loader|latest" \
  configs/stage1/recursive_detection_ce_latest src/detection src/config/schema.py src/config/loader.py src/sft.py src/bootstrap/trainer_setup.py src/data_collators src/trainers tests/test_detection_* tests/test_recursive_detection_* tests/test_packing_* tests/test_detection_training_dataset.py \
  --glob '!output/**' --glob '!temp/**'
```

Expected: Task 0 records compact detection as routed to Task 7. Do not record
provisional compact classification rows here unless the exact files, symbols,
decision labels, and verification commands have already been inspected.

- [ ] **Step 5: Add the classification table to the audit**

Modify `progress/audits/2026-05-03_type_schema_architecture_audit.md` so Task 0 results use this table shape:

```markdown
## Post-Merge Inventory Classification

| Area | File/Symbol | Domain Concept | Current Shape | Boundary Type | Risk | Decision | Follow-Up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| encoded-cache | `src/datasets/encoded_sample_cache.py::EncodedSampleCacheRequest` | cache request | frozen dataclass plus mapping serializer | runtime/artifact | low | keep and extend producer/run-metadata coverage | Task 3 |
```

Every high-signal hit must use one `Decision` value from the Boundary Classification Rules section. If a file has no cross-boundary ambiguity, add a single rule-out row for that file group rather than one row per local dictionary. Compact detection rows belong in Task 7 unless Task 0 has already completed the exact Task 7 inspection.

- [ ] **Step 6: Record post-merge priority order**

Modify `progress/audits/2026-05-03_type_schema_architecture_audit.md` so it contains:

```markdown
## Post-Merge Priority Order

1. Finish encoded-cache producer, consumer, run-metadata, docs, and spec consistency because the initial typed internals already exist and the serialized contract is focused.
2. Classify compact detection baseline surfaces before any detection-stack code refactor.
3. Decide static packing, prediction/eval, and Stage-2 state through separate gates so broad architecture risks do not get hidden inside the cache slice.
```

- [ ] **Step 7: Review inventory-only artifacts, including untracked docs**

Run:

```bash
git status --short --branch
git diff -- progress/audits/README.md progress/index.yaml docs/catalog.yaml
git diff --no-index /dev/null progress/audits/2026-05-03_type_schema_architecture_audit.md || true
git diff --no-index /dev/null docs/superpowers/plans/2026-05-03-type-schema-refactor.md || true
git diff --no-index /dev/null docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md || true
```

Expected: tracked diffs and untracked planning artifacts contain only audit/spec/plan/index wording updates for this inventory step.

- [ ] **Step 8: Commit the implemented cache-contract checkpoint**

Run:

```bash
git diff --check
git add \
  src/datasets/encoded_sample_cache.py \
  tests/test_encoded_sample_cache.py \
  docs/catalog.yaml \
  progress/audits/README.md \
  progress/index.yaml \
  progress/audits/2026-05-03_type_schema_architecture_audit.md \
  docs/superpowers/plans/2026-05-03-type-schema-refactor.md \
  docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md
git commit -m "refactor(datasets): type encoded cache contracts"
```

Expected: commit includes only the encoded-cache contract code/tests and the audit/spec/index/docs files that describe that exact checkpoint. Compact detection and latest recursive detection launch files from `1ed47b3` remain untouched.

## Task 1: Encoded-Sample Cache Internal Typed Contracts

Status: implemented before the global expansion request. Keep this task as the first implemented slice, but re-open it if later global checks find inconsistency.

- [x] **Step 1: Write failing request and manifest tests**

Added tests in `tests/test_encoded_sample_cache.py` for:

```python
def test_encoded_sample_cache_request_normalizes_typed_fields(tmp_path) -> None:
    from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest

    request = EncodedSampleCacheRequest.from_mapping(_cache_request(tmp_path))

    assert request.enabled is True
    assert request.root_dir == (tmp_path / "encoded-cache").resolve()
    assert request.ineligible_policy == "error"
    assert request.wait_timeout_s == pytest.approx(5.0)
    assert request.max_resident_shards == 4
    assert request.dataset_split == "train"
    assert request.dataset_jsonl == "train.jsonl"
    assert request.fingerprint["cache_schema_version"] == 1
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
conda run -n ms python -m pytest tests/test_encoded_sample_cache.py -q -k "request_normalizes_typed_fields or manifest_roundtrips_serialized_payload"
```

Observed: failed because `EncodedSampleCacheRequest` and `EncodedSampleCacheManifest` did not exist.

- [x] **Step 3: Implement minimal dataclasses**

Implemented in `src/datasets/encoded_sample_cache.py`:

```python
EncodedSampleCacheManifestStatus = Literal["building", "complete", "error"]

@dataclass(frozen=True)
class EncodedSampleCacheRequest:
    enabled: bool
    root_dir: Path
    ineligible_policy: str
    wait_timeout_s: float
    max_resident_shards: int
    dataset_split: str
    dataset_jsonl: Any
    fingerprint: dict[str, Any]
```

- [x] **Step 4: Route production code through wrappers**

Production code now normalizes request fields through `EncodedSampleCacheRequest` and validates/indexes manifest shards through `EncodedSampleCacheManifest` / `EncodedSampleShard`.

- [x] **Step 5: Run targeted encoded-cache tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py -q
```

Observed: `19 passed`, with four pre-existing multiprocessing fork deprecation warnings from packing helpers.

## Task 2: Refresh Encoded-Cache Producer Plan Against Merged Code

Status: blocking refresh before editing `src/sft.py`, `src/config/schema.py`, or tests that compact upstream changed.

- [x] **Step 1: Locate current merged encoded-cache and compact-detection overlaps**

Run:

```bash
rg -n "_build_encoded_sample_cache_request|_build_encoded_sample_cache_bypass_info|encoded_sample_cache|LatestDetectionTrainingConfig|recursive_detection|DetectionTraining|packing|DetectionDataset|detection_dataset|recursive_detection_ce_latest" \
  src/sft.py src/config/schema.py src/config/loader.py src/detection/dataset.py tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_training_runtime_profile.py tests/test_detection_training_dataset.py
```

Expected: the executor records current function signatures and any compact-detection config or training-profile assumptions that overlap with encoded-cache request production.

Observed 2026-05-03 on `e7773e3`: `rtk rg -n ...` over the named files found the current encoded-cache producer signatures in `src/sft.py` (`_parse_encoded_sample_cache_config(training_cfg, train_args)`, `_build_encoded_sample_cache_request(..., dataset_mode, sample_limit=None, system_prompt_dense=None, system_prompt_summary=None)`, and `_build_encoded_sample_cache_bypass_info(request, *, reason)`) plus compact-detection overlap at `LatestDetectionTrainingConfig`, latest `packing`, `DetectionTrainingDataset.from_jsonl`, and SFT latest-detection dataset/cache wiring. No compact-detection hit changed the encoded-cache request parameter contract.

- [x] **Step 2: Use Serena for Python symbol inspection**

Inspect these current symbols before editing:

```text
src/sft.py::_build_encoded_sample_cache_request
src/sft.py::_build_encoded_sample_cache_bypass_info
src/sft.py::_parse_encoded_sample_cache_config
src/config/schema.py::TrainingConfig
src/config/schema.py::LatestDetectionTrainingConfig
src/config/loader.py
src/detection/dataset.py
```

Expected: plan snippets in Task 3 are checked against current merged signatures. If any snippet no longer matches the current function signature, update this plan before writing tests or code.

Observed 2026-05-03: Serena was activated on `/data/CoordExp/.worktrees/refactor-type-schema` and inspected `src/sft.py::_build_encoded_sample_cache_request`, `src/sft.py::_build_encoded_sample_cache_bypass_info`, `src/sft.py::_parse_encoded_sample_cache_config`, `src/config/schema.py::TrainingConfig`, `src/config/schema.py::LatestDetectionTrainingConfig`, `src/config/loader.py::ConfigLoader/build_train_arguments`, `src/config/loader.py::ConfigLoader/_materialize_training_config`, `src/config/loader.py::ConfigLoader/_is_latest_detection_config_payload`, and `src/detection/dataset.py::DetectionTrainingDataset.from_jsonl` / `__getitem__`. Task 3 snippets still match the current merged function signatures and the current `EncodedSampleCacheRequest` shape, where `to_mapping()` does not yet preserve `fingerprint_sha256`, `cache_dir`, or `manifest_path`.

- [x] **Step 3: Run pre-change overlap tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_latest_training_config_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_training_runtime_profile.py \
  -q
```

Expected: PASS before implementation. If this fails, diagnose the merged baseline before changing encoded-cache code.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache_runtime_config.py tests/test_latest_training_config_contract.py tests/test_detection_training_dataset.py tests/test_training_runtime_profile.py -q` passed with `76 passed in 0.97s`.

- [x] **Step 4: Update the audit with refresh result**

Modify `progress/audits/2026-05-03_type_schema_architecture_audit.md`:

```markdown
- Encoded-cache producer refresh: current merged `src/sft.py` and `src/config/schema.py` signatures were inspected before implementation. Compact detection config surfaces do not change the intended encoded-cache request contract.
```

If compact detection changes the cache request assumptions, replace the second sentence with the exact overlapping symbol and update Task 3 snippets before proceeding.

Observed 2026-05-03: audit updated with the encoded-cache producer refresh bullet. Compact detection config surfaces do not change the intended encoded-cache request contract, so Task 3 snippets were left unchanged.

## Task 3: Canonicalize Encoded-Cache Request Producer And Consumer Boundaries

- [x] **Step 1: Write failing producer roundtrip test**

Modify `tests/test_encoded_sample_cache_runtime_config.py`:

```python
from pathlib import Path

from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest
from src.sft import _build_encoded_sample_cache_request
```

Add:

```python
def test_build_encoded_sample_cache_request_returns_canonical_payload(tmp_path) -> None:
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text('{"id": 1}\n', encoding="utf-8")
    runtime_cfg = _parse_encoded_sample_cache_config(
        {
            "encoded_sample_cache": {
                "enabled": True,
                "root_dir": str(tmp_path / "cache"),
                "ineligible_policy": "bypass",
                "wait_timeout_s": 5,
                "max_resident_shards": 2,
            }
        },
        SimpleNamespace(output_dir=str(tmp_path / "out")),
    )

    payload = _build_encoded_sample_cache_request(
        runtime_cfg=runtime_cfg,
        training_config=SimpleNamespace(
            global_max_length=1024,
            template={"system": "sys", "truncation_strategy": "raise"},
        ),
        custom_config=_custom_config(),
        template=_Template(max_length=128),
        train_args=SimpleNamespace(max_model_len=512),
        dataset_seed=7,
        dataset_jsonl=str(train_jsonl),
        dataset_split="train",
        dataset_mode="dense",
        sample_limit=64,
        system_prompt_dense="sys",
        system_prompt_summary=None,
    )

    assert payload is not None
    request = EncodedSampleCacheRequest.from_mapping(payload)
    assert payload == request.to_mapping()
    assert payload["fingerprint_sha256"]
    assert payload["cache_dir"] == str(Path(payload["root_dir"]) / payload["fingerprint_sha256"])
    assert payload["manifest_path"] == str(Path(payload["cache_dir"]) / "manifest.json")
```

Observed 2026-05-03: added this test to `tests/test_encoded_sample_cache_runtime_config.py` with the canonical `EncodedSampleCacheRequest` roundtrip assertion and derived path checks.

- [x] **Step 2: Run test and verify red**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache_runtime_config.py::test_build_encoded_sample_cache_request_returns_canonical_payload -q
```

Expected before implementation: FAIL because `EncodedSampleCacheRequest.to_mapping()` does not preserve `fingerprint_sha256`, `cache_dir`, or `manifest_path`.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache_runtime_config.py::test_build_encoded_sample_cache_request_returns_canonical_payload -q` failed as expected at `assert payload == request.to_mapping()` because the roundtrip dropped `fingerprint_sha256` and `cache_dir`.

- [x] **Step 3: Extend `EncodedSampleCacheRequest` to own derived artifact paths**

Modify `src/datasets/encoded_sample_cache.py`:

```python
@dataclass(frozen=True)
class EncodedSampleCacheRequest:
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

Update `from_mapping()` to canonicalize the three derived fields:

```python
fingerprint = _canonicalize_fingerprint(dict(payload.get("fingerprint") or {}))
fingerprint_sha256 = str(payload.get("fingerprint_sha256") or _fingerprint_digest(fingerprint))
root_dir = Path(str(payload.get("root_dir") or ".")).resolve()
cache_dir = Path(str(payload.get("cache_dir") or (root_dir / fingerprint_sha256)))
manifest_path = Path(str(payload.get("manifest_path") or (cache_dir / "manifest.json")))
```

Update `to_mapping()` to include:

```python
"fingerprint_sha256": self.fingerprint_sha256,
"cache_dir": str(self.cache_dir),
"manifest_path": str(self.manifest_path),
```

Observed 2026-05-03: `EncodedSampleCacheRequest` now canonicalizes and emits `fingerprint_sha256`, `cache_dir`, and `manifest_path`; `EncodedSampleCacheStore` reads those canonical fields from the request.

- [x] **Step 4: Use the canonical request in `src/sft.py`**

Modify `src/sft.py` imports:

```python
from src.datasets.encoded_sample_cache import EncodedSampleCacheRequest
```

At the end of `_build_encoded_sample_cache_request()`, replace direct raw-dict return with:

```python
payload = {
    "enabled": True,
    "root_dir": str(runtime_cfg.root_dir),
    "ineligible_policy": runtime_cfg.ineligible_policy,
    "wait_timeout_s": float(runtime_cfg.wait_timeout_s),
    "max_resident_shards": int(runtime_cfg.max_resident_shards),
    "dataset_split": str(dataset_split),
    "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
    "fingerprint": fingerprint,
    "fingerprint_sha256": fingerprint_sha256,
    "cache_dir": str(cache_dir),
    "manifest_path": str(cache_dir / "manifest.json"),
}
return EncodedSampleCacheRequest.from_mapping(payload).to_mapping()
```

Observed 2026-05-03: `_build_encoded_sample_cache_request()` now returns `EncodedSampleCacheRequest.from_mapping(payload).to_mapping()` with `manifest_path` included.

- [x] **Step 5: Use the canonical request in bypass metadata**

Modify `_build_encoded_sample_cache_bypass_info()` in `src/sft.py`:

```python
canonical = EncodedSampleCacheRequest.from_mapping(request)
return {
    "enabled": True,
    "status": "bypassed",
    "reason": str(reason),
    "policy": canonical.ineligible_policy,
    "wait_timeout_s": canonical.wait_timeout_s,
    "dataset_split": canonical.dataset_split,
    "dataset_jsonl": canonical.dataset_jsonl,
    "fingerprint": dict(canonical.fingerprint),
    "fingerprint_sha256": canonical.fingerprint_sha256,
    "root_dir": str(canonical.root_dir),
    "cache_dir": str(canonical.cache_dir),
    "manifest_path": str(canonical.manifest_path),
}
```

Observed 2026-05-03: `_build_encoded_sample_cache_bypass_info()` now builds bypass metadata from `EncodedSampleCacheRequest.from_mapping(request)`.

- [x] **Step 6: Widen the dataset boundary type without forcing callers to import dataclasses**

Modify `src/datasets/encoded_sample_cache.py`:

```python
EncodedSampleCacheRequestInput = Mapping[str, Any] | EncodedSampleCacheRequest

def _coerce_cache_request(
    request: EncodedSampleCacheRequestInput,
) -> EncodedSampleCacheRequest:
    if isinstance(request, EncodedSampleCacheRequest):
        return request
    return EncodedSampleCacheRequest.from_mapping(request)
```

Update `setup_encoded_sample_cache_for_dataset()` signature:

```python
def setup_encoded_sample_cache_for_dataset(
    dataset: Any,
    request: EncodedSampleCacheRequestInput | None,
) -> tuple[EncodedSampleCacheStore | None, dict[str, Any] | None]:
```

Update the first enabled check:

```python
if request is None:
    return None, None
cache_request = _coerce_cache_request(request)
if not cache_request.enabled:
    return None, None
```

Observed 2026-05-03: added `EncodedSampleCacheRequestInput` and `_coerce_cache_request()`; `setup_encoded_sample_cache_for_dataset()` now accepts mappings or `EncodedSampleCacheRequest`.

- [x] **Step 7: Run focused cache producer/consumer tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  -q
```

Expected: PASS.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest tests/test_encoded_sample_cache.py tests/test_encoded_sample_cache_runtime_config.py tests/test_stage1_set_continuation_cache_policy.py -q` passed with `28 passed, 4 warnings` (warnings are multiprocessing fork deprecation warnings from existing static-packing cache tests).

## Task 4: Type Encoded-Cache Run Metadata

- [x] **Step 1: Write failing run-metadata wrapper test**

Modify `tests/test_run_metadata_file.py`:

```python
from src.bootstrap.run_metadata import EncodedSampleCacheRunMetadata
```

Add:

```python
def test_encoded_sample_cache_run_metadata_omits_empty_splits() -> None:
    metadata = EncodedSampleCacheRunMetadata(
        train={"status": "ready", "root_dir": "/tmp/train"},
        eval=None,
    )

    assert metadata.to_mapping() == {
        "train": {"status": "ready", "root_dir": "/tmp/train"}
    }
```

- [x] **Step 2: Run test and verify red**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_run_metadata_file.py::test_encoded_sample_cache_run_metadata_omits_empty_splits -q
```

Expected before implementation: FAIL because `EncodedSampleCacheRunMetadata` does not exist.

Observed red:

- `rtk conda run -n ms python -m pytest tests/test_run_metadata_file.py::test_encoded_sample_cache_run_metadata_omits_empty_splits -q`
- Result: failed during collection with `ImportError: cannot import name 'EncodedSampleCacheRunMetadata' from 'src.bootstrap.run_metadata'`.

- [x] **Step 3: Implement wrapper in `src/bootstrap/run_metadata.py`**

Add:

```python
from dataclasses import dataclass
```

Add near `attach_encoded_sample_cache_run_metadata()`:

```python
@dataclass(frozen=True)
class EncodedSampleCacheRunMetadata:
    train: Mapping[str, Any] | None = None
    eval: Mapping[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.train is not None:
            payload["train"] = copy.deepcopy(dict(self.train))
        if self.eval is not None:
            payload["eval"] = copy.deepcopy(dict(self.eval))
        return payload
```

Update `attach_encoded_sample_cache_run_metadata()`:

```python
encoded_sample_cache = EncodedSampleCacheRunMetadata(
    train=train_cache_info,
    eval=eval_cache_info,
).to_mapping()
if encoded_sample_cache:
    meta["encoded_sample_cache"] = encoded_sample_cache
```

- [x] **Step 4: Run metadata tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_run_metadata_file.py \
  tests/test_encoded_sample_cache_runtime_config.py::test_attach_encoded_sample_cache_run_metadata_scopes_train_and_eval \
  -q
```

Expected: PASS.

Observed green:

- `rtk conda run -n ms python -m pytest tests/test_run_metadata_file.py tests/test_encoded_sample_cache_runtime_config.py::test_attach_encoded_sample_cache_run_metadata_scopes_train_and_eval -q`
- Result: `3 passed in 0.76s`.

Additional bookkeeping:

- Updated `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md`
  to record the implemented run-metadata wrapper without changing the
  serialized `encoded_sample_cache` artifact contract.
- Updated `progress/audits/2026-05-03_type_schema_architecture_audit.md` to
  remove run metadata from the remaining raw-boundary list and keep producer
  dictionaries as the remaining encoded-cache seam.

## Task 5: Align Strict Config Schema, Runtime Config, Specs, And Docs

- [x] **Step 1: Strengthen strict-config test for residency bound**

Modify `tests/test_training_config_strict_unknown_keys.py`:

```python
def test_training_encoded_sample_cache_keys_are_allowed_and_normalized() -> None:
    payload = _base_training_payload()
    payload["training"] = {
        "encoded_sample_cache": {
            "enabled": True,
            "wait_timeout_s": 42,
            "max_resident_shards": 3,
        }
    }

    cfg = TrainingConfig.from_mapping(payload, PromptOverrides())
    cache_cfg = cfg.training["encoded_sample_cache"]
    assert cache_cfg["enabled"] is True
    assert cache_cfg["ineligible_policy"] == "error"
    assert cache_cfg["wait_timeout_s"] == 42
    assert cache_cfg["max_resident_shards"] == 3
```

- [x] **Step 2: Run strict-config tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_training_config_strict_unknown_keys.py::test_training_encoded_sample_cache_keys_are_allowed_and_normalized \
  tests/test_training_config_strict_unknown_keys.py::test_training_encoded_sample_cache_unknown_nested_key_fails_fast \
  tests/test_stage1_static_packing_runtime_config.py::test_parse_encoded_sample_cache_config_accepts_residency_bound \
  tests/test_stage1_static_packing_runtime_config.py::test_parse_encoded_sample_cache_config_rejects_nonpositive_residency_bound \
  -q
```

Expected: PASS. If this fails, fix `src/config/schema.py` and `_parse_encoded_sample_cache_config()` so `max_resident_shards` remains a positive integer in both schema and runtime config.

- [x] **Step 3: Update current OpenSpec encoded-cache spec**

Modify `openspec/specs/encoded-training-cache/spec.md` canonical v1 fields:

```markdown
  - `training.encoded_sample_cache.max_resident_shards: int`
```

Add normative behavior:

```markdown
- `training.encoded_sample_cache.max_resident_shards` MUST default to `4` and
  MUST be greater than zero.
- Runtime request, manifest, and run-metadata internals SHOULD use typed
  representations, but serialized artifact keys MUST stay compatible with v1
  unless this specification is revised.
```

- [x] **Step 4: Update packing guide operator text**

Modify `docs/data/PACKING.md` encoded-cache bullets to mention:

```markdown
- `training.encoded_sample_cache.max_resident_shards` bounds the number of shard
  files kept resident by the cache store. The default is `4`; raise it only when
  repeated shard reloads dominate dataset fetch time.
```

- [x] **Step 5: Verify docs and YAML remain parseable**

Run:

```bash
conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml')]; print('YAML_OK')"
```

Expected: `YAML_OK`.

Task 5 evidence (2026-05-03):

- `tests/test_training_config_strict_unknown_keys.py::test_training_encoded_sample_cache_keys_are_allowed_and_normalized`
  now asserts authored `max_resident_shards: 3` survives strict config
  normalization.
- `openspec/specs/encoded-training-cache/spec.md` now includes
  `training.encoded_sample_cache.max_resident_shards: int` in canonical v1
  fields, requires default `4`, requires values greater than zero, and records
  the typed-internal/stable-serialized-artifact compatibility rule.
- `docs/data/PACKING.md` now documents the operator meaning and default for
  `training.encoded_sample_cache.max_resident_shards`.
- `progress/audits/2026-05-03_type_schema_architecture_audit.md` and
  `docs/superpowers/specs/2026-05-03-type-schema-refactor-design.md` now
  reflect that strict config, runtime config, current spec, and operator docs
  are aligned for the residency bound while preserving serialized v1 artifact
  keys.
- `rtk conda run -n ms python -m pytest tests/test_training_config_strict_unknown_keys.py::test_training_encoded_sample_cache_keys_are_allowed_and_normalized tests/test_training_config_strict_unknown_keys.py::test_training_encoded_sample_cache_unknown_nested_key_fails_fast tests/test_stage1_static_packing_runtime_config.py::test_parse_encoded_sample_cache_config_accepts_residency_bound tests/test_stage1_static_packing_runtime_config.py::test_parse_encoded_sample_cache_config_rejects_nonpositive_residency_bound -q`
  passed with `4 passed in 0.79s`.
- `conda run -n ms python -c "from pathlib import Path; import yaml; [yaml.safe_load(Path(p).read_text(encoding='utf-8')) for p in ('docs/catalog.yaml','progress/index.yaml')]; print('YAML_OK')"`
  printed `YAML_OK`.

## Task 6: Global Encoded-Cache Regression Gate

- [x] **Step 1: Run all encoded-cache and adjacent config tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_run_metadata_file.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  -q
```

Expected: PASS, allowing only pre-existing multiprocessing fork deprecation warnings.

Observed 2026-05-03 on `34ed54a`: passed with `179 passed, 4 warnings in
2.44s`. The only warnings were the pre-existing multiprocessing fork
deprecation warnings from
`tests/test_encoded_sample_cache.py::test_static_packing_reuses_cache_backed_dataset_after_full_length_probe`
and
`tests/test_encoded_sample_cache.py::test_static_packing_prefers_cache_backed_length_helper_over_dataset_getitem`.

- [x] **Step 2: Re-run repository-wide encoded-cache search**

Run:

```bash
rg -n "encoded_sample_cache|EncodedSampleCache|setup_encoded_sample_cache_for_dataset|get_encoded_sample_cache_info" \
  src tests scripts configs docs progress openspec \
  --glob '!output/**' --glob '!temp/**' --glob '!.git/**'
```

Expected: every non-historical hit is either updated, intentionally left serialized as JSON/YAML, or documented in the audit rule-out table.

Observed 2026-05-03 on `34ed54a`: `566` hits across `45` files, classified as:

- production code: `151` hits in `7` files; typed runtime boundaries or
  compatibility-preserving serialized payload keys in `src/bootstrap`,
  `src/config`, `src/datasets`, and `src/sft.py`.
- tests: `120` hits in `10` files; encoded-cache contract, strict-config,
  run-metadata, stage1 set-continuation, stage2 config/profile, and recursive
  detection wiring coverage.
- configs: `7` hits in `7` YAML files; intentionally serialized
  `training.encoded_sample_cache` config keys.
- current docs/spec: `212` hits in `8` files; operator docs, current
  super-power plan/spec material, and stable encoded-cache OpenSpec contract.
- progress historical references: `37` hits in `3` files.
- active OpenSpec changes: `3` hits in `3` files; historical/current planning
  context only, not code-contract drift.
- archived OpenSpec: `36` hits in `7` files.

No inconsistency was found that required code changes. The remaining
non-historical hits are updated typed boundaries, compatibility-preserving
serialized JSON/YAML keys, test coverage, or current docs/spec references.

- [x] **Step 3: Commit encoded-cache global consistency slice**

Run:

```bash
git diff --check
git add \
  progress/audits/2026-05-03_type_schema_architecture_audit.md \
  docs/superpowers/plans/2026-05-03-type-schema-refactor.md
git commit -m "docs(audit): record encoded cache regression gate"
```

Expected: commit includes only Task 6 plan/audit evidence updates unless the
gate finds a real encoded-cache inconsistency.

## Task 7: Compact Detection Schema Classification Gate

Status: decision gate only. Do not implement compact-detection code changes from this task.

- [x] **Step 1: Inspect compact detection schema-bearing surfaces**

Run:

```bash
rg -n "dataclass|TypedDict|Enum|Protocol|Mapping\\[str, Any\\]|dict\\[str, Any\\]|Dict\\[str, Any\\]|list\\[|tuple\\[|payload|record|state|manifest|metrics|loss|objective|packing|template|config|dataset|loader|latest" \
  configs/stage1/recursive_detection_ce_latest src/detection src/config/schema.py src/config/loader.py src/sft.py src/bootstrap/trainer_setup.py src/data_collators src/trainers/metrics/mixins.py src/trainers/batch_extras.py tests/test_detection_* tests/test_recursive_detection_* tests/test_packing_* tests/test_random_* tests/test_sft_preparation_contract.py tests/test_detection_training_dataset.py docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md \
  --glob '!output/**' --glob '!temp/**'
```

Expected: audit report identifies compact-detection surfaces as one of: already typed dataclass/config object, serialized mapping, dynamic metric/logging map, local scratch container, or cross-boundary payload needing follow-up. The upstream production-launch plan is inspected as a baseline constraint document, not as a refactor target.

Observed 2026-05-03: the required `rg` scan returned 4381 matching lines. Targeted exact reads then classified the relevant groups: `src/detection/data.py` uses frozen dataclasses after raw JSONL `Mapping[str, Any]` ingress; `src/config/schema.py` owns latest detection config dataclasses; `src/detection/template.py` / `tokenization.py` expose dataclass/protocol span contracts; `src/detection/objective.py` / `loss.py` expose dataclass/enumerated loss state; `src/detection/packing.py` exposes packing/fingerprint dataclasses while preserving static-packing fingerprint mappings; `src/detection/evaluation.py` uses a manifest dataclass and canonical parser payload mappings; `src/trainers/metrics/mixins.py` keeps recursive CE logging as a dynamic metric map; `src/sft.py` only assembles local runtime/provenance dictionaries and rejects latest recursive detection packing/encoded-cache use until sidecar fingerprints exist. The upstream launch plan was inspected as a baseline launch/provenance document only.

- [x] **Step 2: Add compact detection rows to the inventory table**

Modify `progress/audits/2026-05-03_type_schema_architecture_audit.md` under `Post-Merge Inventory Classification`. Add one row for each exact compact-detection group below, filling every table column with the concrete result from Step 1 and using decision labels from the Boundary Classification Rules section:

- `src/detection/data.py`: normalized raw sample and recursive target payloads, boundary type `data/preparation`.
- `src/detection/dataset.py`, `src/config/loader.py`, and `configs/stage1/recursive_detection_ce_latest/`: latest recursive detection dataset/config loading, boundary type `training input/config`.
- `src/detection/template.py` and `src/detection/tokenization.py`: template and token-span contracts, boundary type `prompt/tokenization`.
- `src/detection/objective.py` and `src/detection/loss.py`: recursive detection objective/loss state, boundary type `trainer/loss`.
- `src/detection/packing.py`: compact packing artifacts and cache fingerprints, boundary type `packing/artifact`.
- `src/detection/evaluation.py` and `src/trainers/metrics/mixins.py`: detection metrics and evaluation summary payloads, boundary type `metrics/eval`.
- `src/sft.py`: latest detection runtime orchestration, prompt shim, dataset selection, packing/cache rejection, and static-packing fingerprint interaction, boundary type `training orchestration/runtime support`.
- `docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md`: recursive detection production-launch constraints, boundary type `launch/provenance document`.

Observed 2026-05-03: `progress/audits/2026-05-03_type_schema_architecture_audit.md` now contains one completed inventory row for each exact compact-detection group above, using the Boundary Classification Rules labels.

- [x] **Step 3: Record the compact detection gate decision**

Record exactly one of these outcomes in `progress/audits/2026-05-03_type_schema_architecture_audit.md`:

```markdown
- Compact detection decision: no code refactor in this branch; merged compact detection contracts are already sufficiently typed or intentionally serialized, and remaining raw containers are local scratch or dynamic metrics.
```

or:

```markdown
- Compact detection decision: create a dedicated follow-up plan because the Task 7 inventory found cross-boundary compact detection payloads that need named typed contracts before further refactor work.
```

or:

```markdown
- Compact detection decision: fold only the named encoded-cache overlap into Tasks 3-6 because the compact detection inventory found an overlap at `src/sft.py::_build_encoded_sample_cache_request` that directly affects encoded-cache request or run-metadata contracts.
```

The third outcome is allowed only if the exact overlapping symbol is named in the decision and Tasks 3-6 are updated with exact tests and implementation steps for that symbol.

Observed 2026-05-03: the audit records the first decision exactly: no code refactor in this branch. No Tasks 3-6 encoded-cache overlap was found, and no dedicated compact-detection follow-up plan is required by this gate.

- [x] **Step 4: Run compact detection contract profile**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_detection_normalization_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_stage1_json_pretty_template.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_token_span_masks_from_templates.py \
  tests/test_compact_et_rmp_span_contract.py \
  tests/test_random_order_sft_contract.py \
  tests/test_random_permutation_et_rmp_ce_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_sft_preparation_contract.py \
  tests/test_batch_extras_contract.py \
  tests/test_length_insensitive_loss_normalization.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_trainer_mixin.py \
  tests/test_recursive_detection_state_weighting.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_runtime_profile.py \
  -q
```

Expected: PASS, allowing only previously observed warnings. If it fails, diagnose the compact baseline before using the gate decision.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest ... -q` passed with `270 passed in 3.05s`.

- [x] **Step 5: Verify upstream production-launch plan remains baseline-owned**

Run:

```bash
git diff -- docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md
```

Expected: empty diff unless this task intentionally updates launch-status docs.
The inspected constraints from that upstream plan are: direct
`torchrun -m src.sft`, `packing: false`,
`training.encoded_sample_cache.enabled: false`, no bidirectional gating, bf16
with targeted fp32, `per_device_train_batch_size=16`,
`gradient_accumulation_steps=1`, `effective_batch_size=128`, no
latest-schema mAP callback yet, and production launch still running at the time
of the upstream note.

Observed 2026-05-03: `git diff -- docs/superpowers/plans/2026-05-02-training-infra-template-mode-refactor.md` was empty, so the upstream production-launch plan remained baseline-owned and untouched.

- [x] **Step 6: Create follow-up plan if needed**

If Step 3 selects a dedicated follow-up plan, create `docs/superpowers/plans/2026-05-03-compact-detection-schema-refactor.md` with exact target files, exact failing tests, implementation snippets, and verification commands. Do not implement compact-detection code in this umbrella plan.

Observed 2026-05-03: not needed because the selected decision was no compact-detection code refactor in this branch.

## Task 8: Static Packing Schema Decision Gate

- [x] **Step 1: Inspect static-packing symbols**

Run:

```bash
rg -n "Packed|packing_plan|pack_plan|static_packing|manifest|INDEX.json|raw_plan|aligned_plan" \
  src/datasets/wrappers src/sft.py tests/test_packing_wrapper.py tests/test_stage1_static_packing_runtime_config.py docs/data/PACKING.md \
  --glob '!output/**' --glob '!temp/**'
```

Expected: audit report lists the static-packing domain concepts and current representation.

Observed 2026-05-03:

- Required scan was run with `rtk rg -n "Packed|packing_plan|pack_plan|static_packing|manifest|INDEX.json|raw_plan|aligned_plan" ...`.
- Static-packing domain concepts found: raw plan, DDP-aligned plan, plan checksums, fingerprinted plan cache, setup `INDEX.json`, length cache, `StaticPackedCaptionDataset`, train/eval static cache roots, and SFT logging/metadata consumption of plan fields.
- Current representation is list-of-list plan payloads plus ad hoc JSON cache/index mappings in `src/datasets/wrappers/packed_caption.py`, exposed as dataset attributes and consumed by `src/sft.py`.

- [x] **Step 2: Record the decision gate outcome**

Record exactly one of these outcomes in `progress/audits/2026-05-03_type_schema_architecture_audit.md`:

```markdown
- Static packing decision: create/use the follow-up plan for typed `StaticPackingPlan` / `StaticPackingManifest`; no static-packing code is implemented by this branch.
```

or:

```markdown
- Static packing decision: defer; current raw containers are local scratch data or already guarded by tests, and the global encoded-cache contract is the only affected cache refactor in this branch.
```

Observed 2026-05-03:

```markdown
- Static packing decision: create/use the follow-up plan for typed `StaticPackingPlan` / `StaticPackingManifest`; no static-packing code is implemented by this branch.
```

- [x] **Step 3: Create a separate implementation plan if needed**

If the decision is to implement static-packing typed wrappers, stop this plan at the decision gate and create a follow-up super-power plan that names exact files, exact test bodies, implementation snippets, and verification commands. Do not implement static-packing code from this decision gate.

Observed 2026-05-03: created follow-up plan `docs/superpowers/plans/2026-05-03-static-packing-schema-refactor.md`. No static-packing code, tests, or operator docs were modified by this decision gate.

- [x] **Step 4: Run existing tests when deferring or before writing the follow-up plan**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_packing_wrapper.py \
  tests/test_stage1_static_packing_runtime_config.py \
  -q
```

Expected: PASS.

Observed 2026-05-03: `78 passed, 10 warnings in 2.44s`. The warnings were multiprocessing fork deprecation warnings from `tests/test_packing_wrapper.py`.

## Task 9: Prediction And Evaluation Record Schema Decision Gate

- [x] **Step 1: Inspect prediction ingestion aliases**

Run:

```bash
rg -n "\"pred\"|\"predictions\"|\"objects\"|gt_vs_pred|Prediction|Detection|metrics.json|scored" \
  src/infer src/eval scripts tests \
  --glob '!output/**' --glob '!temp/**'
```

Expected: audit report identifies all readers/writers of `pred`, `predictions`, and `objects`.

Observed 2026-05-03:

- Required scan was run exactly with `rg -n "\"pred\"|\"predictions\"|\"objects\"|gt_vs_pred|Prediction|Detection|metrics.json|scored" src/infer src/eval scripts tests --glob '!output/**' --glob '!temp/**'`.
- Inspected representation groups: canonical inference writer in `src/infer/engine.py`, eval readers in `src/eval/detection.py`, confidence post-op in `src/eval/confidence_postop.py`, constant-score materialization in `src/eval/artifacts.py`, proxy GT filtering in `src/eval/proxy_views.py`, proxy bundle orchestration in `src/eval/proxy_eval_bundle.py`, focused eval/infer tests, and analysis scripts that use local scratch dictionaries.
- Current representation classification: inference serialized artifacts intentionally use canonical `pred`; evaluator and duplicate-control paths accept `pred` then legacy `predictions`; confidence post-op and constant-score materialization read only `pred`; GT ingestion accepts `gt` or `objects`; `pred_confidence.jsonl` local `objects` and `metrics.json` metrics/counters are separate serialized/dynamic artifact surfaces; analysis-script `pred` and `objects` hits are local scratch or diagnostics.

- [x] **Step 2: Record the adapter decision**

Record one concrete decision in `progress/audits/2026-05-03_type_schema_architecture_audit.md`:

```markdown
- Prediction/eval decision: implement a canonical read adapter before metric changes; the adapter will preserve serialized record mappings while standardizing read access for `pred`, `predictions`, and `objects`.
```

or:

```markdown
- Prediction/eval decision: defer code changes; current alias behavior remains documented as a risk until a dedicated eval-artifact contract plan owns migration or rejection semantics.
```

Observed 2026-05-03:

```markdown
- Prediction/eval decision: implement a canonical read adapter before metric changes; the adapter will preserve serialized record mappings while standardizing read access for `pred`, `predictions`, and `objects`.
```

Rationale: current alias behavior is not uniformly centralized. Eval and duplicate-control already accept legacy `predictions`, while confidence post-op and constant-score scoring read only canonical `pred`. A read adapter is the smallest follow-up before metric changes because it can preserve serialized artifact compatibility while making read semantics shared.

- [x] **Step 3: Create a separate implementation plan if needed**

If the decision is to implement a canonical prediction/eval adapter, stop this plan at the decision gate and create a follow-up super-power plan that names exact files, exact test bodies, implementation snippets, and verification commands. Do not implement eval adapter code from this decision gate.

Observed 2026-05-03: created follow-up plan `docs/superpowers/plans/2026-05-03-prediction-eval-record-adapter.md`. This Task 9 gate did not implement adapter code, did not modify production Python code, and did not change metric semantics or artifact key names.

- [x] **Step 4: Run eval/infer baseline tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_unified_infer_pipeline.py \
  tests/test_confidence_postop.py \
  tests/test_proxy_eval_bundle.py \
  -q
```

Expected: PASS.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest tests/test_detection_eval_ingestion_diagnostics.py tests/test_unified_infer_pipeline.py tests/test_confidence_postop.py tests/test_proxy_eval_bundle.py -q` passed with `57 passed in 0.58s`.

## Task 10: Stage-2 And Teacher-Forcing Runtime State Decision Gate

- [x] **Step 1: Inspect runtime state boundaries**

Run:

```bash
rg -n "PreparedSegment|prepared_segment|runtime_state|batch_extras|ModuleResult|PipelineResult|state: dict|dict\[str, Any\]|losses|metrics" \
  src/trainers tests \
  --glob '!output/**' --glob '!temp/**'
```

Expected: audit report separates stable state/result objects from dynamic metric maps.

Observed 2026-05-03:

- Required scan was run exactly with `rg -n "PreparedSegment|prepared_segment|runtime_state|batch_extras|ModuleResult|PipelineResult|state: dict|dict\[str, Any\]|losses|metrics" src/trainers tests --glob '!output/**' --glob '!temp/**'`.
- Inspected runtime-state concepts: `Stage2ABTrainingTrainer._coordexp_checkpoint_runtime_state()`, `_coordexp_restore_checkpoint_runtime_state()`, `_PendingStage2Log`, `_stage2_metric_snapshots`, `_stage2_post_rollout_segments`, `_stage2_b_step_raw`, `_stage2_a_step_raw`, `_stage2_train_monitor_candidates`, `_rollout_matching_batch_metrics`, `BatchExtras`, `ModuleResult`, `PipelineResult`, and `Stage2PreparedSegment`.
- Stable state/result objects: `_PendingStage2Log` is already a dataclass accumulator; `BatchExtras` is already a dataclass for collator extras; `ModuleResult` and `PipelineResult` are already teacher-forcing result dataclasses.
- Dynamic metrics classification: `ModuleResult.metrics`, `PipelineResult.metrics`, `_stage2_metric_snapshots`, `_rollout_matching_batch_metrics`, `Stage2BatchMetrics`, and the many `loss/*`, `stage2/*`, `rollout/*`, `gradmon/*`, and `packing/*` keys remain dynamic metric/logging maps.
- Concrete runtime-state risk selected for follow-up: the Stage-2 checkpoint runtime-state payload serialized by `_coordexp_checkpoint_runtime_state()` and restored by `_coordexp_restore_checkpoint_runtime_state()`.

- [x] **Step 2: Protect existing typed contracts**

Confirm current typed teacher-forcing contracts before adding new types:

```bash
sed -n '1,130p' src/trainers/teacher_forcing/contracts.py
sed -n '1,180p' src/trainers/teacher_forcing/objective_pipeline.py
```

Expected: `ModuleResult` and `PipelineResult` remain the canonical teacher-forcing result types; do not replace them with new overlapping containers.

Observed 2026-05-03:

- `src/trainers/teacher_forcing/contracts.py` still defines `ModuleResult(loss, metrics, state)` and `PipelineResult(total_loss, module_losses, metrics, state)` as frozen dataclasses.
- `src/trainers/teacher_forcing/objective_pipeline.py::run_teacher_forcing_pipeline()` still returns `PipelineResult` and accumulates module losses, metrics, and state through those existing contracts.
- Task 10 did not edit teacher-forcing Python code and did not add overlapping teacher-forcing result containers.

- [x] **Step 3: Decide first Stage-2 state object**

Record exactly one decision in `progress/audits/2026-05-03_type_schema_architecture_audit.md`:

```markdown
- Stage-2 decision: create a dedicated follow-up plan for the first concrete runtime state object selected by the Task 10 inspection; do not reshape logging keys in this branch without a separate metric-contract spec.
```

or:

```markdown
- Stage-2 decision: defer code changes; current high-risk raw dictionaries are dynamic metric payloads, and the branch should avoid reshaping logging keys without a separate metric-contract spec.
```

Observed 2026-05-03:

```markdown
- Stage-2 decision: create a dedicated follow-up plan for the first concrete runtime state object selected by the Task 10 inspection; do not reshape logging keys in this branch without a separate metric-contract spec.
```

Selected first runtime-state object: `Stage2CheckpointRuntimeState`, scoped to `Stage2ABTrainingTrainer._coordexp_checkpoint_runtime_state()` and `_coordexp_restore_checkpoint_runtime_state()`.

- [x] **Step 4: Create a separate implementation plan if needed**

If the decision is to implement a Stage-2 runtime state object, stop this plan at the decision gate and create a follow-up super-power plan that names exact symbols, files, test bodies, implementation snippets, and verification commands. Do not implement Stage-2 runtime-state code from this decision gate.

Observed 2026-05-03: created follow-up plan `docs/superpowers/plans/2026-05-03-stage2-runtime-state-schema-refactor.md`. The follow-up plan is limited to a future `Stage2CheckpointRuntimeState` typed wrapper around the existing checkpoint runtime payload. This Task 10 gate did not implement Stage-2 runtime-state code, did not change `Stage2PreparedSegment`, did not modify teacher-forcing result types, and did not reshape metric/logging keys.

- [x] **Step 5: Run Stage-2 focused tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_stage2_ab_training.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_batch_extras_contract.py \
  tests/test_teacher_forcing_token_ce.py \
  tests/test_stage1_set_continuation_branch_runtime.py \
  -q
```

Expected: PASS.

Observed 2026-05-03: `rtk conda run -n ms python -m pytest tests/test_stage2_ab_training.py tests/test_stage2_two_channel_training.py tests/test_batch_extras_contract.py tests/test_teacher_forcing_token_ce.py tests/test_stage1_set_continuation_branch_runtime.py -q` passed with `205 passed in 2.46s`.

## Task 11: Final Audit Closure And Verification

- [x] **Step 1: Update audit report with implemented changes and rule-outs**

Modify `progress/audits/2026-05-03_type_schema_architecture_audit.md` so it contains:

```markdown
## Implementation Closure

- Implemented:
  - Encoded-sample cache request, shard, manifest, and run metadata typed boundaries.
- Verified unchanged serialized contracts:
  - `training.encoded_sample_cache.*` YAML keys.
  - cache `manifest.json` keys.
  - `run_metadata.json["encoded_sample_cache"]` train/eval split keys.
- Rule-outs:
  - Historical `progress/` and archived `openspec/changes/archive/` references were left unchanged.
```

Observed 2026-05-03: added `## Implementation Closure` to `progress/audits/2026-05-03_type_schema_architecture_audit.md`. The closure records the implemented encoded-sample cache request, shard, manifest, and run-metadata typed boundaries; unchanged serialized YAML, manifest, and run-metadata keys; compact detection no-code-refactor classification; explicit static packing, prediction/eval adapter, and Stage-2 checkpoint runtime-state follow-up plan paths; final affected-file scan counts/classification; and historical `progress/` / archived OpenSpec rule-outs.

- [x] **Step 2: Run final targeted verification**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_run_metadata_file.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_unified_infer_pipeline.py \
  tests/test_confidence_postop.py \
  tests/test_stage2_ab_training.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_batch_extras_contract.py \
  tests/test_teacher_forcing_token_ce.py \
  -q
```

Expected: PASS, or a documented pre-existing skip/warning that does not affect this refactor.

Observed 2026-05-03:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_encoded_sample_cache.py \
  tests/test_encoded_sample_cache_runtime_config.py \
  tests/test_run_metadata_file.py \
  tests/test_training_config_strict_unknown_keys.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_stage1_set_continuation_cache_policy.py \
  tests/test_stage1_set_continuation_benchmark_profiles.py \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_unified_infer_pipeline.py \
  tests/test_confidence_postop.py \
  tests/test_stage2_ab_training.py \
  tests/test_stage2_two_channel_training.py \
  tests/test_batch_extras_contract.py \
  tests/test_teacher_forcing_token_ce.py \
  -q
```

Result after the final complete-manifest hardening commits: `442 passed, 4 warnings in 3.77s`. The warnings were the previously observed multiprocessing fork deprecation warnings from encoded-cache static-packing tests and do not affect this refactor.

- [x] **Step 3: Run compact detection contract verification**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_detection_normalization_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_stage1_json_pretty_template.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_token_span_masks_from_templates.py \
  tests/test_compact_et_rmp_span_contract.py \
  tests/test_random_order_sft_contract.py \
  tests/test_random_permutation_et_rmp_ce_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_sft_preparation_contract.py \
  tests/test_batch_extras_contract.py \
  tests/test_length_insensitive_loss_normalization.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_trainer_mixin.py \
  tests/test_recursive_detection_state_weighting.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_runtime_profile.py \
  -q
```

Expected: PASS, allowing only previously observed warnings.

Observed 2026-05-03:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_latest_training_config_contract.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_detection_normalization_contract.py \
  tests/test_detection_template_registry.py \
  tests/test_detection_stage1_json_pretty_template.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_span_alignment.py \
  tests/test_detection_template_parsing_eval.py \
  tests/test_token_span_masks_from_templates.py \
  tests/test_compact_et_rmp_span_contract.py \
  tests/test_random_order_sft_contract.py \
  tests/test_random_permutation_et_rmp_ce_contract.py \
  tests/test_recursive_detection_ce_target_builder.py \
  tests/test_sft_preparation_contract.py \
  tests/test_batch_extras_contract.py \
  tests/test_length_insensitive_loss_normalization.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py \
  tests/test_recursive_detection_ce_loss_adapter.py \
  tests/test_recursive_detection_ce_sft_wiring.py \
  tests/test_recursive_detection_ce_trainer_mixin.py \
  tests/test_recursive_detection_state_weighting.py \
  tests/test_stage1_static_packing_runtime_config.py \
  tests/test_training_runtime_profile.py \
  -q
```

Result: `270 passed in 3.07s`.

- [x] **Step 4: Run static checks**

Run:

```bash
conda run -n ms python -m py_compile \
  src/datasets/encoded_sample_cache.py \
  src/datasets/dense_caption.py \
  src/sft.py \
  src/bootstrap/run_metadata.py \
  src/config/schema.py
git diff --check
```

Expected: both commands pass.

Observed 2026-05-03:

```bash
conda run -n ms python -m py_compile \
  src/datasets/encoded_sample_cache.py \
  src/datasets/dense_caption.py \
  src/sft.py \
  src/bootstrap/run_metadata.py \
  src/config/schema.py
git diff --check
```

Result: both commands exited 0.

- [x] **Step 5: Final repository-wide affected-file check**

Run:

```bash
rg -n "encoded_sample_cache|EncodedSampleCache|predictions|PreparedSegment|runtime_state|dict\[str, Any\]" \
  src tests scripts configs docs progress openspec \
  --glob '!output/**' --glob '!temp/**' --glob '!.git/**'
```

Expected: remaining hits are either typed, serialized contract mappings, intentional metric/logging maps, or documented rule-outs in the audit.

Observed 2026-05-03:

```bash
rg -n "encoded_sample_cache|EncodedSampleCache|predictions|PreparedSegment|runtime_state|dict\[str, Any\]" \
  src tests scripts configs docs progress openspec \
  --glob '!output/**' --glob '!temp/**' --glob '!.git/**'
```

Result after removing ambiguous per-pattern closure counts: `1,778` hits across `248` files. Top-level counts were `src` `843`, `tests` `175`, `scripts` `12`, `configs` `12`, `docs` `345`, `progress` `132`, and `openspec` `259`.

Per-pattern subcounts are intentionally omitted from the closure record because they require a separate counting convention from the exact combined `rg -n` line-hit scan. Quality-review verification found the direct line-hit count for `encoded_sample_cache` is `475`, while occurrence counting with `rg -o` gives `537`; the earlier recorded `encoded_sample_cache` subcount came from grepping the saved combined scan output and is not a valid repository line-hit count.

Classification: encoded-cache hits are typed boundaries, serialized compatibility mappings, tests, configs, or current docs/specs; compact detection hits remain classified as no code refactor in this branch; prediction/eval, static packing, and Stage-2 runtime-state hits are owned by explicit follow-up plans; metric/logging and manifest maps remain intentionally dynamic where no stable contract wrapper was selected; analysis-script hits are local diagnostics; historical `progress/` and archived `openspec/changes/archive/` references were left unchanged.

- [x] **Step 6: Final status summary**

Run:

```bash
git status --short --branch
git log --oneline --decorate -5
```

Expected: branch contains logically scoped commits for the audit/plan and implemented refactor slices.

Task 11 closure commit under review:

```text
4a52e8e docs(audit): close type schema refactor audit
```

Post-review and final-code-review correction history:

- `4a52e8e` is the Task 11 closure commit whose stale scan counts and pre-commit status/log were reviewed.
- `b5cfd4c` is the first docs-only follow-up that refreshed the total scan counts but still carried an ambiguous per-pattern count and an unsatisfied-looking post-fix verification note.
- `134d630` is the docs-only follow-up that removed ambiguous per-pattern closure counts, kept static packing follow-up-only, and recorded the then-current status/log shape.
- `e34c49f`, `08daba9`, and `00a5c70` are the final whole-branch-review code/test fixes that hardened complete-manifest shard inventory validation.
- This closure refresh records verification at `00a5c70`; the final response records the hash of the docs-only closure refresh commit because embedding a commit's own final hash inside that same commit would change the hash.

Observed before this closure refresh commit on 2026-05-03:

```bash
git status --short --branch
git log --oneline --decorate -5
```

Result:

```text
## codex/refactor-type-schema
00a5c70 (HEAD -> codex/refactor-type-schema) fix(cache): reject empty shard tails
08daba9 fix(cache): align encoded shard indexes
e34c49f fix(cache): validate encoded shard inventory
134d630 docs(audit): clarify closure scan provenance
b5cfd4c docs(audit): refresh final closure provenance
```

Final post-refresh status/log shape for this docs-only closure refresh:

```text
## codex/refactor-type-schema
final-docs-refresh (HEAD -> codex/refactor-type-schema) docs(audit): refresh final manifest closure
00a5c70 fix(cache): reject empty shard tails
08daba9 fix(cache): align encoded shard indexes
e34c49f fix(cache): validate encoded shard inventory
134d630 docs(audit): clarify closure scan provenance
```

The exact closure refresh hash is recorded in the final task response.

Residual follow-up plans:

- Static packing typed plan/manifest wrapper: `docs/superpowers/plans/2026-05-03-static-packing-schema-refactor.md`.
- Prediction/eval canonical read adapter: `docs/superpowers/plans/2026-05-03-prediction-eval-record-adapter.md`.
- Stage-2 checkpoint runtime-state typed wrapper: `docs/superpowers/plans/2026-05-03-stage2-runtime-state-schema-refactor.md`.

Task 11 closure summary:

- Implemented scope remains encoded-cache typed boundaries only: request, shard, manifest, complete-manifest inventory validation, producer/consumer normalization, and run-metadata wrapper boundaries.
- Verified serialized contracts remain unchanged for `training.encoded_sample_cache.*`, cache `manifest.json`, and `run_metadata.json["encoded_sample_cache"]`.
- Decision gates created executable follow-up plans for static packing, prediction/eval alias access, and Stage-2 checkpoint runtime state.
- Broad scans classified remaining hits instead of converting them into unscoped production refactors.
- No full repository-wide type cleanup is claimed by this branch.

## Self-Review

- The plan now treats the repository-wide search as a required input to implementation, not an optional follow-up.
- The encoded-cache refactor is expanded from one local module to producer, consumer, config, run metadata, tests, specs, and docs.
- Compact detection is now an explicit post-merge classification gate instead of an implicit expansion of the cache task.
- Static packing, prediction/eval, and Stage-2 runtime state are owned by explicit decision slices so they cannot silently fall out of scope.
- Serialized artifact compatibility is protected by roundtrip tests and OpenSpec/docs updates where the stable contract is touched.
- Historical `progress/` and archived OpenSpec references are not edited unless they are current routing/index files.
