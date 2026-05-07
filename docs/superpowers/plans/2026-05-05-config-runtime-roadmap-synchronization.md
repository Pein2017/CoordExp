# Config Runtime Roadmap Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize CoordExp configs, scripts, docs, and schema guardrails with the current runtime contracts so compact latest detection, legacy Stage-1 SFT, set-continuation ET-RMP-CE, and Stage-2 remain clearly separated and reproducible.

**Architecture:** Make source guardrails safer before moving YAML, then split config/documentation surfaces by runtime family, then migrate scripts to config-first entrypoints. Do not bulk-convert legacy `custom.*` configs into latest detection: latest compact recursive detection, Stage-1 baseline SFT, set-continuation, raw-text/geometry ablations, and Stage-2 are separate contracts until source support unifies them.

**Tech Stack:** Python, CoordExp strict dataclass config, YAML config overlays, shell entrypoint wrappers, pytest, repo docs, `rtk conda run -n ms python -m pytest`.

---

## Scope Boundary

This plan implements the roadmap from the compact/latest detection config audit. It does not start new training runs, does not change model objective semantics, does not enable recursive-detection packing, and does not add a new OpenSpec change unless a later reviewer identifies a stable compatibility contract that must be normatively pinned.

Audit caveat: the roadmap evidence was gathered across the active main checkout and `.worktrees/compact-detection-sequence`. Implementation must first establish which checkout owns the latest compact detection source/config family. Do not write tests, docs, configs, or scripts that reference `LatestDetectionTrainingConfig`, `configs/stage1/recursive_detection_ce_latest/`, `configs/stage1/compact_detection_sequence/`, or `src/detection/runtime.py` in main until Task 0 proves those paths and symbols exist in the current implementation checkout or explicitly materializes them from a named source.

## Current Execution Base

Implementation is approved for the clean integration worktree, not the dirty main checkout:

```text
worktree: /data/CoordExp/.worktrees/refactor-latest-integration
branch: codex/refactor-latest-integration
merge commit: d507c2e Merge compact detection sequence into refactor integration
parents: baddd3f39ea56a5b6550fde25c5b6e56155f4ccd 57bf25e64051f737d1bb69b8f7a319a98f67d227
```

The safe integration procedure used to establish this base was:

1. Inspect `main` and `.worktrees/compact-detection-sequence` without assuming merge status.
2. Confirm `main` did not contain `LatestDetectionTrainingConfig` or `configs/stage1/recursive_detection_ce_latest/`.
3. Confirm `codex/compact-detection-sequence` carried the latest detection source/config family.
4. Avoid merging directly into `/data/CoordExp` because `main` had unrelated dirty `.codex/skills/gitnexus-*` deletions.
5. Create `/data/CoordExp/.worktrees/refactor-latest-integration` from `main`.
6. Merge `codex/compact-detection-sequence` into `codex/refactor-latest-integration`.
7. Confirm the integration branch contains both `main` and `codex/compact-detection-sequence`, and that `src/config/schema.py::LatestDetectionTrainingConfig`, `src/detection/runtime.py`, and `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml` exist.

Do not implement this plan from `/data/CoordExp` until the unrelated dirty `.codex/skills/gitnexus-*` deletions are intentionally handled. Do not commit from implementation workers unless the user separately asks for commits.

The implementation must preserve these active runtime families:

- Latest compact recursive detection uses `LatestDetectionTrainingConfig` with top-level `data`, `prompt`, `detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, and `validation`.
- Legacy/general Stage-1 SFT still uses `TrainingConfig` and `custom.*`.
- Stage-1 set-continuation ET-RMP-CE stays separate from latest recursive detection and keeps `configs/stage1/set_continuation/production.yaml` as the `bsz16` / `16/128` production contract.
- Stage-2 two-channel and rollout-aligned stay on their strict `stage2_ab` / `rollout_matching` contracts.
- Raw-text and alternate bbox-geometry profiles stay out of latest compact detection until source schema support exists.

## File Structure

- Modify: `src/config/schema.py`
  - Type or validate latest `debug` keys.
  - Reject or loudly warn on legacy `custom.coord_loss`.
  - Cross-check latest-detection packing owners.
- Modify: `src/sft.py`
  - Make latest `debug.output_dir` handling mapping-aware, or remove runtime support and fail fast when set.
- Modify: `tests/test_latest_training_config_contract.py`
  - Cover latest `debug`, obsolete latest keys, packing owner consistency, and `custom` rejection.
- Create: `tests/test_legacy_config_contract.py`
  - Cover legacy `custom.coord_loss` behavior and `custom.extra.rollout_matching` rejection.
- Remove: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml`
  - Do not keep a comment-only `.yaml` under the positive smoke directory.
- Create: `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`
  - Preserve the unsupported-packing contract under an explicitly negative location.
- Modify: `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`
  - Either archive/rename in a later task or add a legacy bridge warning if a move would break existing references.
- Create: `configs/_shared/stage1_sft/README.md`
  - Document that these overlays use legacy `custom.*`.
- Create: `configs/_shared/latest_detection/README.md`
  - Document latest top-level section ownership and forbid `custom.*`.
- Create: `configs/_shared/latest_detection/datasets/coco_1024_bbox_max60.yaml`
  - Latest-detection dataset overlay using top-level `data.*`.
- Create: `configs/_shared/latest_detection/surfaces/compact_full_coord_token_xyxy.yaml`
  - Latest compact surface overlay containing `prompt`, `detection_template`, `token_rows`, `evaluation`, and `validation`.
- Create: `configs/_shared/latest_detection/objectives/recursive_detection_ce_support2.yaml`
  - Latest recursive CE objective overlay.
- Create: `configs/_shared/latest_detection/objectives/random_order_sft.yaml`
  - Latest compact random-order SFT ablation overlay.
- Modify: `docs/training/README.md`
  - Mark `recursive_detection_ce_latest` as canonical latest compact detection and `compact_detection_sequence` as legacy bridge.
- Modify: `docs/data/PACKING.md`
  - Clarify latest-detection top-level `packing` versus runtime adapter `training.packing`.
- Modify: `docs/eval/WORKFLOW.md`
  - Clarify compact infer/eval artifact flow and scope-preserving benchmark labels.
- Modify: `docs/AGENT_INDEX.md`
  - Add routing note for latest compact detection versus legacy compact bridge.
- Modify: `docs/catalog.yaml`
  - Keep the machine-readable routing layer consistent with `docs/AGENT_INDEX.md`.
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
  - Preserve raw-text/geometry boundaries and correct stale set-continuation production batch references.
- Modify: `docs/training/STAGE1_ET_RMP_CE.md`
  - Correct stale set-continuation production batch references or point to `production.yaml` as the sole source of truth.
- Modify: `scripts/README.md`
  - Separate stable entrypoints, compatibility/debug wrappers, and historical diagnostics.
- Modify: `scripts/run_infer_eval.sh`
  - Either migrate to YAML pipeline mode or mark as legacy/debug with an explicit warning.
- Modify: `scripts/run_vis.sh`
  - Mark as manual/debug unless updated to consume resolved pipeline artifacts.
- Modify: `scripts/pipelines/run_rollout_stability_probe.sh`
  - Repair hardcoded environment/script references or move out of the stable pipeline path.
- Test: `tests/test_latest_detection_config_contract.py`
- Test: `tests/test_legacy_config_contract.py`

## Task 0: Latest Detection Provenance And Checkout Preflight

**Files:**
- Inspect only: `src/config/schema.py`
- Inspect only: `src/sft.py`
- Inspect only: `configs/stage1/`
- Inspect only: `.worktrees/compact-detection-sequence/`
- Modify later tasks only after this task chooses a path.

Current execution status: completed for `/data/CoordExp/.worktrees/refactor-latest-integration`. This worktree already materializes the latest compact detection branch into a clean integration branch. Future workers should still preserve the checks below as guardrails and should stop if they are not true in their working directory.

- [ ] **Step 1: Locate the latest parser and config family in the current checkout**

Run exact read-only searches:

```bash
rg -n "LatestDetectionTrainingConfig|DetectionTemplateConfig|recursive_detection_ce_latest|compact_full_support2|compact_detection_sequence" src configs docs
```

Expected if latest detection is already in the current checkout: matches include `src/config/schema.py::LatestDetectionTrainingConfig`, a canonical `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`, and any runtime files referenced by docs.

Expected if latest detection is not in the current checkout: no current-main matches for the latest parser/config family, and this task must choose Step 2B or stop for user approval.

- [ ] **Step 2A: If latest detection exists in the current checkout, pin its owners**

Record the exact current-main paths in this plan before implementation:

- Latest parser owner: `src/config/schema.py::LatestDetectionTrainingConfig`.
- Latest loader owner: `src/config/loader.py`.
- Latest runtime adapter owner: `src/detection/runtime.py`.
- Training entrypoint owner: `src/sft.py`.
- Canonical latest production YAML: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`.
- Canonical latest SFT ablation YAML: `configs/stage1/recursive_detection_ce_latest/prod/compact_full_random_sft.yaml`.
- Latest smoke YAMLs: `configs/stage1/recursive_detection_ce_latest/smoke/*.yaml`.
- Legacy compact bridge YAML: `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`.

Then continue to Task 1.

- [ ] **Step 2B: If latest detection exists only under `.worktrees/compact-detection-sequence`, stop for import approval**

Do not silently copy from `.worktrees`.

Ask the user to choose:

1. Materialize the latest detection source/config family into main from `.worktrees/compact-detection-sequence`, preserving exact paths and provenance.
2. Re-scope this plan to current-main legacy `TrainingConfig` cleanup and docs corrections only.

Tasks 1, 2, 4, 5, 6, and 7 are blocked until this choice is approved.

- [ ] **Step 3: Establish latest packing ownership for this checkout**

If latest top-level `packing.*` exists, document whether it is:

- A schema/config-authoring facade normalized into `training.*` before `src/sft.py` reads packing; or
- Not present in this checkout, in which case all packing tests must use the existing `training.*` owner.

Do not imply `src/sft.py` reads latest `objective.id` or top-level `packing.*` unless Task 0 proves that is true in the current checkout.

## Task 1: Add Source Guardrail Tests For Config Contracts

**Files:**
- Create: `tests/test_latest_detection_config_contract.py`
- Create: `tests/test_legacy_config_contract.py`

- [ ] **Step 1: Add latest detection contract tests**

Create `tests/test_latest_detection_config_contract.py` with focused parser tests. The tests must build minimal in-memory YAML mappings rather than launching training.

The first test must be a source-contract assertion from Task 0:

- The current checkout exposes the chosen latest parser class/function.
- The chosen latest parser accepts the documented top-level latest sections.
- The materialized object exposes a runtime `training` payload that `src/sft.py` can consume, or fails before runtime with the intended schema error.

The file should cover these cases:

- Latest configs reject top-level `custom`.
- Latest configs reject unknown `debug` keys after the debug schema is tightened.
- Latest configs consume or reject `debug.output_dir` consistently with the runtime decision made in Task 2.
- Latest configs fail when `training.packing` and `packing.static_packing` disagree.
- Latest configs reject recursive-CE packing when `objective.id: recursive_detection_ce`.
- Latest configs keep `compact_full` requiring omitted `detection_template.object_field_order`.

Use this command for the first red run:

```bash
rtk conda run -n ms python -m pytest tests/test_latest_detection_config_contract.py -q
```

Expected before implementation: at least the new unknown-debug-key test and packing-owner consistency test fail.

- [ ] **Step 2: Add legacy config contract tests**

Create `tests/test_legacy_config_contract.py` with focused parser tests for legacy/general `TrainingConfig`.

The file should cover these cases:

- Legacy `custom.coord_loss` no longer silently disappears.
- Active Stage-2 configs do not rely on `custom.coord_loss` remaining non-fatal before the parser flips to hard error.
- Legacy `custom.extra.rollout_matching` remains rejected.
- Legacy top-level `extra` remains rejected.
- Legacy `training.packing_length` remains rejected with guidance toward `global_max_length` or `template.max_length`.

Use this command for the first red run:

```bash
rtk conda run -n ms python -m pytest tests/test_legacy_config_contract.py -q
```

Expected before implementation: the `custom.coord_loss` test fails if the current parser still pops and ignores that key.

## Task 2: Tighten Latest Debug And Packing Ownership

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/sft.py`
- Test: `tests/test_latest_detection_config_contract.py`

- [ ] **Step 1: Type or allowlist latest `debug`**

In `src/config/schema.py`, latest debug must parse through the existing `DebugConfig.from_mapping`.

The implementation must make the tests from Task 1 pass and must not loosen unknown-key handling for any other latest section.

- [ ] **Step 2: Fix latest `debug.output_dir` behavior**

Keep `debug.output_dir` supported through typed `DebugConfig` only. Assert the materialized latest training config exposes `debug` as `DebugConfig`; do not add a generic mapping fallback that would let unknown or removed debug keys bypass `DebugConfig.from_mapping`.

If current `src/sft.py` still needs a focused test seam, extract a pure helper such as `_debug_output_override(debug_config: DebugConfig) -> str | None` and test it without launching training.

- [ ] **Step 3: Cross-check latest packing owners**

In `src/config/schema.py` or the latest materialization layer selected in Task 0, add a latest-detection consistency check:

- If top-level `packing.static_packing` is true, `training.packing` must be true.
- If top-level `packing.static_packing` is false, `training.packing` must be false for latest recursive CE.
- If `packing.padding_free_packed` is true, recursive CE must still fail until sidecar offset rewriting is implemented.

If Task 0 proves the current runtime only consumes `training.*`, document top-level `packing.*` as a schema/config-authoring facade and require it to normalize into `training.packing`, `training.packing_mode`, `training.eval_packing`, and cache/runtime fields before `_parse_packing_config` is called.

Do not enable new packing behavior in this task.

- [ ] **Step 4: Run targeted latest contract tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_latest_detection_config_contract.py -q
```

Expected: all latest detection contract tests pass.

## Task 3: Remove The Legacy `custom.coord_loss` Silent No-Op

**Files:**
- Modify: `src/config/schema.py`
- Test: `tests/test_legacy_config_contract.py`

- [ ] **Step 1: Convert `custom.coord_loss` into a hard error**

In `CustomConfig.from_mapping`, stop popping and ignoring `coord_loss`.

Replace the silent compatibility path with a `ValueError` whose message names the active surfaces:

```text
custom.coord_loss is no longer supported; use custom.coord_soft_ce_w1 for legacy Stage-1 SFT losses or latest objective.* for LatestDetectionTrainingConfig.
```

If a hard error breaks a required old profile during implementation, pause and escalate before switching to warning-only behavior.

Before changing the parser, inventory or test active Stage-2 AB and rollout-aligned config surfaces to prove none still rely on `custom.coord_loss` being non-fatal. If any active config still carries `custom.coord_loss`, migrate that config to the current objective namespace before flipping the parser behavior.

- [ ] **Step 2: Run targeted legacy contract tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_legacy_config_contract.py -q
```

Expected: all legacy config contract tests pass.

## Task 4: Rehome Unsupported Packing Smoke As A Negative Contract

**Files:**
- Create: `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`
- Remove: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml`
- Test: `tests/test_latest_training_config_contract.py`

- [ ] **Step 1: Create the negative profile directory and file**

Create `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml` with the same semantic intent as the current unsupported packing smoke:

- It must extend or mirror the latest compact recursive detection smoke/prod base.
- It must set `training.packing: true`.
- It must set `packing.static_packing: true`.
- It must include a top comment stating this is a negative parser/runtime-support contract, not a launchable smoke.

- [ ] **Step 2: Remove or demote the positive smoke profile**

Run a reference sweep first:

```bash
rg -n "compact_full_packing_unsupported|compact_full_static_packing_should_fail" docs configs scripts tests
```

Update active tests, docs, and scripts to reference
`configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`,
then remove `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml`.
Do not leave a comment-only or launchable unsupported-packing `.yaml` in the
positive smoke directory.

- [ ] **Step 3: Add the negative profile parser assertion**

Extend `tests/test_latest_detection_config_contract.py` so the negative profile fails for the intended packing reason after config materialization.

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_latest_detection_config_contract.py -q
```

Expected: the negative profile assertion passes by observing the intended failure.

## Task 5: Split Shared Config Overlays By Schema Family

**Files:**
- Create: `configs/_shared/stage1_sft/README.md`
- Create: `configs/_shared/latest_detection/README.md`
- Create: `configs/_shared/latest_detection/datasets/coco_1024_bbox_max60.yaml`
- Create: `configs/_shared/latest_detection/surfaces/compact_full_coord_token_xyxy.yaml`
- Create: `configs/_shared/latest_detection/objectives/recursive_detection_ce_support2.yaml`
- Create: `configs/_shared/latest_detection/objectives/random_order_sft.yaml`

- [ ] **Step 1: Document legacy shared overlays**

Create `configs/_shared/stage1_sft/README.md` explaining:

- Existing `configs/_shared/datasets/*.yaml` and `configs/_shared/prompts/*.yaml` are legacy Stage-1 SFT overlays.
- They use `custom.train_jsonl`, `custom.val_jsonl`, `custom.object_ordering`, `custom.object_field_order`, and `custom.extra.prompt_variant`.
- Latest detection configs must not extend these overlays.

- [ ] **Step 2: Document latest detection overlays**

Create `configs/_shared/latest_detection/README.md` explaining:

- Latest detection uses top-level `data`, `prompt`, `detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, and `validation`.
- `custom.*` is rejected for latest detection.
- `data.object_ordering` uses `random_permutation`, not legacy `random`.
- Top-level `packing` is the semantic owner; `training.packing` is a runtime adapter until a resolver derives it.

- [ ] **Step 3: Add latest dataset overlay**

Create `configs/_shared/latest_detection/datasets/coco_1024_bbox_max60.yaml` by copying the exact `data.*` block from the materialized canonical latest production profile selected in Task 0.

Do not use guessed `data/images` or `data/processed/...` paths. If Task 0 does not materialize a canonical latest production profile into the current checkout, do not create this overlay.

- [ ] **Step 4: Add compact latest surface overlay**

Create `configs/_shared/latest_detection/surfaces/compact_full_coord_token_xyxy.yaml` containing the canonical latest compact surface:

- `prompt.system_variant: stage1_detection`
- `prompt.user_variant: compact_detection`
- `prompt.prompt_variant_enabled: true`
- `detection_template.id: compact_full`
- `detection_template.coordinate_surface: coord_token`
- `detection_template.bbox_format: xyxy`
- `detection_template.strict_parse: true`
- `token_rows` equivalent to the canonical latest production token-row groups.
- `evaluation.expected_template: compact_full`
- `validation.validate_span_alignment: true`
- `validation.validate_template_capabilities: true`
- `validation.fail_fast: true`

Do not include `detection_template.object_field_order` for `compact_full`.

- [ ] **Step 5: Add objective overlays**

Create `configs/_shared/latest_detection/objectives/recursive_detection_ce_support2.yaml` containing:

```yaml
objective:
  id: recursive_detection_ce
  variant: random_permutation_et_rmp_ce
  trie_support_weight: 2.0
  trie_balance_weight: 1.0
```

Create `configs/_shared/latest_detection/objectives/random_order_sft.yaml` containing:

```yaml
objective:
  id: sft
  variant: random_order_sft
  trie_support_weight: 0.0
  trie_balance_weight: 0.0
```

Name or document this overlay as coord-token latest compact SFT only. It explicitly excludes raw-text norm1000 profiles and `custom.bbox_geo` geometry-loss ablations.

If the canonical latest production selected in Task 0 uses additional objective keys such as `state_weighting` or `normalization`, paste the exact keys into this plan before implementation approval.

## Task 6: Mark The Legacy Compact Bridge Clearly

**Files:**
- Modify: `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`
- Modify: `docs/training/README.md`
- Modify: `docs/AGENT_INDEX.md`

- [ ] **Step 1: Add a legacy bridge warning to the YAML**

First decide whether the bridge exists in current main.

If the bridge exists in current main, add a top-of-file comment to `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`:

Add a top-of-file comment to `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml`:

```yaml
# Legacy bridge smoke: this file exercises legacy TrainingConfig + custom.detection_sequence_format.
# Canonical latest compact detection lives under configs/stage1/recursive_detection_ce_latest/.
# Do not use this file as a LatestDetectionTrainingConfig example.
```

Do not move or delete the file in this task unless a reference sweep proves no active scripts/docs depend on the path.

If the bridge only exists under `.worktrees/compact-detection-sequence`, do not create a warning-only fake bridge. Either materialize the full semantic YAML body from the named source after user approval, or remove the file-modification task and document that current main has no legacy compact bridge.

- [ ] **Step 2: Update training docs**

In `docs/training/README.md`, make the distinction explicit:

- `configs/stage1/recursive_detection_ce_latest/` is canonical latest compact recursive detection.
- `configs/stage1/compact_detection_sequence/` is a legacy bridge using `custom.*`.
- Set-continuation ET-RMP-CE is not latest recursive detection.

- [ ] **Step 3: Update the agent router**

In `docs/AGENT_INDEX.md`, route Stage-1 compact detection questions only to paths that Task 0 proves exist in the current checkout.

If latest compact detection was materialized, route to:

```text
configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml
src/config/schema.py::LatestDetectionTrainingConfig
src/detection/runtime.py
```

Mention `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml` only as a legacy bridge example.

- [ ] **Step 4: Update the machine-readable catalog**

Update `docs/catalog.yaml` to match `docs/AGENT_INDEX.md`, `docs/training/README.md`, `docs/training/STAGE1_OBJECTIVE.md`, and `docs/training/STAGE1_ET_RMP_CE.md`.

Only add catalog entries for latest compact detection and legacy compact bridge after Task 0 has materialized those paths into the current checkout.

## Task 7: Update Packing And Infer/Eval Documentation

**Files:**
- Modify: `docs/data/PACKING.md`
- Modify: `docs/eval/WORKFLOW.md`

- [ ] **Step 1: Clarify latest packing ownership**

In `docs/data/PACKING.md`, add a latest-detection subsection stating:

- Latest compact recursive detection currently requires `training.packing: false`, `training.eval_packing: false`, `packing.static_packing: false`, and `packing.padding_free_packed: false`.
- Top-level `packing` is the semantic latest-detection owner.
- `training.packing` is a runtime adapter field until the loader derives it.
- Unsupported packing examples belong under `negative/` or `contract_failures/`, not positive `smoke/`.

- [ ] **Step 2: Clarify compact infer/eval surface naming**

In `docs/eval/WORKFLOW.md`, add a compact detection note stating:

- Training currently uses `detection_template.id: compact_full` and `data.object_ordering: random_permutation`.
- Inference configs may still expose `infer.detection_sequence_format: compact_full` and `infer.object_ordering: random`.
- New author-facing docs should call the semantic operation `random_permutation`.
- Benchmark summaries must preserve scope labels such as `val200`, `limit=200`, checkpoint id, confidence policy, grammar, and launch shape, and must link the exact artifact root plus `resolved_config.json` or `resolved_config.path`, `summary.json`, and the metric file used for the claim.
- Partial, proxy, and tiny labels are interpretation labels, not substitutes for artifact provenance.

## Task 7B: Correct Set-Continuation And Legacy Boundary Docs

**Files:**
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/STAGE1_ET_RMP_CE.md`

- [ ] **Step 1: Correct stale set-continuation batch references**

Update `docs/training/STAGE1_OBJECTIVE.md` and `docs/training/STAGE1_ET_RMP_CE.md` so they no longer state stale `support2_bsz32`, `per_device_train_batch_size: 32`, or `effective_batch_size: 256` as the current production contract for `configs/stage1/set_continuation/production.yaml`.

Either replace duplicated numbers with:

```text
artifact_subdir: coco1024_sota1332_setcont_et_rmp_ce_support2_bsz16_v1
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
effective_batch_size: 128
```

or remove duplicated numbers and point to `configs/stage1/set_continuation/production.yaml` as the sole authoritative source.

- [ ] **Step 2: Preserve raw-text and geometry boundaries**

Keep raw-text norm1000 and bbox-geometry ablations documented as legacy Stage-1 SFT surfaces, not latest compact detection overlays.

Add a materialization verification item for:

- `configs/stage1/profiles/2b/raw_text_xyxy_pure_ce_coco80_desc_first_1024_lvis_proxy.yaml`
- `configs/stage1/profiles/2b/bbox_geo_center_size_coco80_desc_first_1024_lvis_proxy.yaml`

## Task 8: Reclassify Script Entrypoints

**Files:**
- Modify: `scripts/README.md`
- Modify: `scripts/run_infer_eval.sh`
- Modify: `scripts/run_vis.sh`
- Modify: `scripts/pipelines/run_rollout_stability_probe.sh`

- [ ] **Step 1: Update scripts README taxonomy**

In `scripts/README.md`, create sections:

- Stable training entrypoints: `scripts/train.sh`, `scripts/train_stage2.sh`.
- Stable infer/eval entrypoints: `scripts/run_infer.py --config`, `scripts/evaluate_detection.py --config`, `scripts/postop_confidence.py --config`, `scripts/evaluate_oracle_k.py --config`, `scripts/export_coco_submission.py --config`, and any proxy/scored-bundle utility only when it preserves a documented scored-artifact contract.
- Compatibility/debug wrappers: shell wrappers that bypass resolved config provenance.
- Historical diagnostics: stale probes that are not maintained launch paths.

- [ ] **Step 2: Demote `run_infer_eval.sh` and add hard metric guardrails**

For this roadmap-sync plan, do not perform a broad behavioral rewrite of `scripts/run_infer_eval.sh` into a new official pipeline.

Add an early warning that says the wrapper is legacy/debug and not the official benchmark path.

Make it impossible for the legacy raw path to produce official-looking COCO/LVIS/`both` benchmark evidence:

- If `EVAL_METRICS` is `coco`, `lvis`, or `both`, the wrapper must refuse to evaluate base `gt_vs_pred.jsonl` unless the input is explicitly a scored artifact or a confidence/constant-score stage is configured.
- The legacy wrapper may default to f1ish-only for raw predictions.
- The README and script warning must say outputs from the legacy raw path are not benchmark evidence unless a scored-artifact requirement is satisfied.

If a future approved migration makes this wrapper stable, it must take a YAML config path as the primary input and must not recreate checkpoint, dataset, limit, metric, confidence, grammar, or output paths from shell env vars. If env-var overrides remain, the wrapper stays compatibility/debug and `resolved_config.json` remains the provenance authority.

- [ ] **Step 3: Demote `run_vis.sh` unless it consumes pipeline artifacts**

Add a warning or README note that `scripts/run_vis.sh` is manual/debug unless it is changed to consume resolved pipeline artifacts and manifests.

- [ ] **Step 4: Repair or archive rollout stability probe**

For `scripts/pipelines/run_rollout_stability_probe.sh`, remove hardcoded machine-specific Python paths and stale script references, or move the wrapper out of the stable pipeline path.

If `run_infer_eval.sh` is demoted to compatibility/debug, `run_rollout_stability_probe.sh` must not remain a stable `scripts/pipelines/` wrapper that depends on it.

If repaired:

- Source `scripts/_lib/backbone.sh` and use `COORDEXP_PYTHON` instead of hardcoding `PYTHON_BIN`.
- Call the current report path `scripts/analysis/report_rollout_stability.py`.
- Consume a YAML pipeline config or explicit existing artifact root.
- Report the artifact root and loaded `summary_json` / `eval_metrics_json`.

## Task 9: Verification Gate

**Files:**
- Test: `tests/test_latest_detection_config_contract.py`
- Test: `tests/test_legacy_config_contract.py`
- Configs/docs/scripts modified in earlier tasks.

- [ ] **Step 1: Run focused parser contract tests**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_latest_detection_config_contract.py \
  tests/test_legacy_config_contract.py
```

Expected: all focused contract tests pass.

- [ ] **Step 2: Run representative existing config tests if present**

Search for existing config loader tests using:

```bash
rg -n "ConfigLoader|LatestDetectionTrainingConfig|TrainingConfig|recursive_detection_ce_latest|set_continuation" tests
```

Then run only the targeted files that cover config loading and runtime policy. Use `rtk conda run -n ms python -m pytest <targeted-test-files>`.

Expected: existing config/runtime-policy tests pass.

- [ ] **Step 3: Perform a no-training materialization check**

For representative stable YAMLs, run the repo's narrowest side-effect-free materialization path if one exists. Cover:

- Latest compact recursive production.
- Latest compact SFT ablation.
- Legacy Stage-1 SFT profile.
- Stage-1 set-continuation production.
- Stage-2 two-channel base.
- Raw-text xyxy norm1000 legacy profile.
- Bbox-geometry legacy profile.
- Negative latest static-packing profile, expecting failure for the intended reason.

Do not launch training or inference in this verification task.

- [ ] **Step 4: Scan for stale set-continuation production numbers**

Run:

```bash
rg -n "support2_bsz32|per_device_train_batch_size: 32|effective_batch_size: 256" docs configs
```

Expected: no stale current-production references remain. Historical mentions must be explicitly labeled historical or superseded.

- [ ] **Step 5: Commit in logical slices, only after implementation approval and verification**

Use small commits:

```bash
git add src/config/schema.py src/sft.py tests/test_latest_detection_config_contract.py tests/test_legacy_config_contract.py
git commit -m "fix: tighten config contract guardrails"

git add configs/_shared configs/stage1/recursive_detection_ce_latest configs/stage1/compact_detection_sequence
git commit -m "chore: separate latest detection config overlays"

git add docs/AGENT_INDEX.md docs/training/README.md docs/data/PACKING.md docs/eval/WORKFLOW.md scripts/README.md scripts/run_infer_eval.sh scripts/run_vis.sh scripts/pipelines/run_rollout_stability_probe.sh
git commit -m "docs: clarify config runtime families"
```

Adjust staging if unrelated dirty changes are present. Do not stage unrelated files or nested repositories.

## Self-Review Checklist

- Latest compact recursive detection remains canonical under `configs/stage1/recursive_detection_ce_latest/`.
- Legacy Stage-1 SFT and raw-text/geometry ablations remain legacy unless source schema support is added.
- Stage-1 set-continuation ET-RMP-CE remains separate from latest recursive detection.
- Stage-2 two-channel and rollout-aligned contracts remain unchanged.
- No training, inference, or benchmark result is claimed by this plan.
- Every verification command is parser/materialization/test-only.
- Unsupported packing is represented as a negative contract, not a positive smoke profile.
- The implementation stops and asks before changing production semantics such as enabling packing, changing objective weights, or moving canonical config paths.

## Execution Handoff

Do not implement this plan until the user explicitly approves an execution mode.

Plan review is complete only when all P0/P1 audit findings have either been incorporated into this document or deliberately deferred with a written reason.

After approval, offer exactly these implementation modes:

1. Subagent-Driven, recommended: use `superpowers:subagent-driven-development`, dispatch bounded workers by task, and review between tasks.
2. Inline Execution: use `superpowers:executing-plans`, implement in this session, and pause at checkpoints before code/config/doc edits with hidden risk.

If approval is not given, stop after saving and reviewing the plan. Do not edit production code, configs, docs, scripts, tests, or git state beyond this planning artifact.
