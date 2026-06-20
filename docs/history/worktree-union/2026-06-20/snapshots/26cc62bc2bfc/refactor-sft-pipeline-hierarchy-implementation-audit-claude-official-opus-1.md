# Refactor `refactor-sft-pipeline-hierarchy` — Implementation Audit

- **Auditor:** claude-opus (read-only implementation review)
- **Date:** 2026-06-19
- **Scope:** worktree `/data/CoordExp/.worktrees/codebase-refactoring-program`,
  branch `codex/codebase-refactoring-program` (refactor work is **uncommitted**:
  72 changed/added/deleted paths in the working tree).
- **Mode:** implementation-vs-contract audit (OpenSpec + plan vs code/config/tests/artifacts).
- **Guardrails:** no files staged/committed/stashed; no worktrees pruned; no training/inference jobs run.

---

## Verdict

**HOLD — not ready for the next human appro
val step.** The semantic intent of the
refactor is largely realized and the high-risk research surfaces (geometry,
Stage-2 separation, provenance, fail-fast guards) are genuinely implemented and
tested. **No P0 correctness/eval-validity defect was found.** However, the
delivered worktree contains **two refactor-introduced failures in pre-existing,
committed contract tests** that directly contradict the "tests pass" readiness
claim, plus an unreconciled config/test/design disagreement. These must be fixed
or explicitly waived by the user before approval.

The implementing agent's "Broader targeted suite: 383 passed, 96 skipped" claim
is **selectively scoped**: it covers the test files the refactor touched (I
reproduce 376 passed / 96 skipped / **1 failed** over that set) but excludes the
two import-safety unit tests that the refactor breaks. The broader claim of
readiness is therefore not supported by the actual test surface.

---

## Blocking Findings

### P1-1 — Import-safety contract regression in `src/training_runtime/preflight.py`

**Evidence:**
- `src/training_runtime/preflight.py:9` (added by this refactor) now does:
  ```python
  from src.config.schema import TEACHER_FORCING_OBJECTIVE_ID
  ```
  Confirmed added via `git diff` (the line is a `+` hunk; the sibling rename
  `teacher_forcing_epoch_varying_rollin` → `research_teacher_forcing_…` is in the
  same hunk).
- `src/training_runtime/__init__.py` imports `.preflight`, so importing the
  package (and therefore `src.training_runtime.plan`, whose parent package must
  load first) now transitively imports `src.config.schema`, which itself imports
  `src.trainers.teacher_forcing.module_registry` (`schema.py:60`) and
  `src.training.stage2.rollout_codec` (`schema.py:68`) → pulls in
  `torch`, `transformers`, `swift`, `datasets`, `src.config`, `src.trainers`.
- Failing committed (unmodified) tests:
  - `tests/test_training_runtime_plan.py::test_plan_module_is_import_safe`
  - `tests/test_training_runtime_plan.py::test_training_runtime_package_root_is_import_safe`
  - `tests/test_training_runtime_profile.py::test_profile_module_is_import_safe`
  - Reproduced failure: `AssertionError: ['torch', 'transformers', 'swift', 'datasets', 'src.config', 'src.trainers']`.
- `src/training_runtime/plan.py:12-13` advertises this module as the
  "Import-safe trainer-variant setup ownership contract" — the contract these
  tests guard.

**Impact:** Violates a stated architectural guarantee (lightweight runtime
plan/profile importable without the heavy ML stack). Manifest/preflight/CLI
contexts that import `training_runtime.profile`/`plan` expecting no torch will
now pay the full import. It does not corrupt training correctness (training
imports torch regardless), so it is **P1, not P0** — but it is a real regression
in committed test coverage that the readiness claim did not surface.

**Fix direction:** Do not import from `src.config.schema` in `preflight.py`.
Either inline the literal `"research_teacher_forcing"`, or hoist
`TEACHER_FORCING_OBJECTIVE_ID` into an import-light constants module (e.g.
alongside `src/training_runtime/plan.py`) and have `schema.py` import it from
there, preserving the single source of truth without dragging the heavy chain
into `training_runtime`.

**Verification:** `python -m pytest tests/test_training_runtime_plan.py
tests/test_training_runtime_profile.py -q` (must pass without the banned modules
appearing in `sys.modules`).

### P1-2 — Config hierarchy contract broken; config/test/design disagree

**Evidence:**
- `configs/stage2/rollout_correction/base.yaml` **removed** its first line
  `extends: ../../base.yaml` (confirmed via `git diff`; the old committed base
  inherited the universal `configs/base.yaml`).
- Failing committed (unmodified) test:
  `tests/test_training_config_hierarchy_contract.py::test_representative_migrated_leaves_stay_within_semantic_depth_budget`
  → `AssertionError: configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml
  is missing one of the required semantic layers: ['universal_base']`.
  The test (`tests/test_training_config_hierarchy_contract.py:143`) treats
  `universal_base` (`configs/base.yaml`) as a **required** ancestor for
  representative migrated leaves, including the Stage-2 prod leaf.
- Concrete effective-value divergence introduced by dropping the universal base
  (materialized via `ConfigLoader.load_yaml_with_extends`):
  - `data.dataloader_num_workers`: `16` (universal base) → **absent** in resolved
    Stage-2 config (falls back to ms-swift default).
  - `training.max_grad_norm`, `warmup_ratio`, `deepspeed` are now explicitly
    re-specified in the self-contained Stage-2 base (so those are intentional,
    not silently wrong), but the dataloader defaults are silently dropped.

**Impact:** A committed contract test fails, and the migration leaves
**config, test, and design out of agreement**: the design doc never authorized
dropping universal-base inheritance, the test still requires it, and the config
removed it. This is exactly the "config migration, docs, tests, and artifacts
agree" property the audit must check — and it does not hold. Reproducibility
risk is moderate (Stage-2 no longer auto-tracks universal defaults; dataloader
worker count changes), not severe (key hyperparameters are re-specified).

**Fix direction:** Decide explicitly and reconcile all three:
(a) restore `extends: ../../base.yaml` on the Stage-2 base (cheapest; keeps the
contract green and inheritance intact), or
(b) if self-contained Stage-2 bases are intended, update
`test_training_config_hierarchy_contract.py`'s required-roles contract and record
the decision in the design/lifecycle docs, and re-add any dropped universal
defaults the Stage-2 runtime actually depends on.

**Verification:** `python -m pytest tests/test_training_config_hierarchy_contract.py -q`.

---

## Non-Blocking Findings

### P2-1 — `TrainingPipelineRegistry` is shadow-only; two parallel pipeline-policy sources

**Evidence:**
- `grep -rn "pipeline_registry\|TrainingPipelineRegistry" src/ --include=*.py`
  returns **only** `src/training/pipeline_registry.py` itself; the descriptor
  classes (`Stage1JsonCEPipeline`, `Stage1CompactTrieCEPipeline`,
  `Stage2RolloutCorrectionPipeline`) are imported only by that module and tests.
- The actual runtime selection path is independent:
  `DetectionPipelineConfig` (`src/config/schema.py:3280`) +
  `_detection_validate_pipeline_objective_pairing` (`schema.py:4516`) +
  `ConfigLoader._runtime_trainer_variant_for_config` (`loader.py:606`) →
  `resolve_training_runtime_plan` → `resolve_trainer_cls` (`sft.py:139`).
- The registry's required top-level domains
  (`run/pipeline/data/template/supervision/objectives/observability/artifacts/runtime`,
  `pipeline_registry.py:22`) match **no** real config under `configs/`.

**Impact:** The design's stated resolution chain
("authored YAML → typed config → pipeline registry → …") is not realized; the
registry is a parallel, test-exercised governance/vocabulary artifact. Pipeline→
objective policy now has two sources of truth (`PIPELINE_OBJECTIVE_POLICIES` in
`pipeline_registry.py:238` vs the pairing logic in `schema.py:4516`) that can
drift. This is consistent with the design's "shadow resolver" framing (Decision:
"Rename The Shadow Resolver Toward Pipeline Vocabulary"), so it is **not a
blocker** — the load-bearing split (Stage-1 standard / research-TF / Stage-2) is
real and verified end-to-end (see Confirmed-OK). Flagged for clarity/drift risk.

**Fix direction:** Either wire the registry into the real resolution path so it
becomes the single seam, or document explicitly that it is a non-load-bearing
shadow/vocabulary resolver and add a cross-check test asserting its policies
match `schema.py`'s pairing rules, to prevent silent drift.

### P2-2 — `cfg_only` debug summary still emits a `trainer_variant` field

**Evidence:** `src/sft.py:2741-2743` always emits
`"trainer_variant": str(getattr(custom_config, "trainer_variant", "") or "")`.
For target-hierarchy (detection) runs, `custom_config` is the runtime shim
(`build_detection_runtime_custom_shim`, `src/detection/runtime.py:126`) which has
no `trainer_variant` attribute, so this resolves to `""` for **all** detection
configs, including Stage-2.

**Impact:** Cosmetic legacy-vocabulary emission in a `--cfg-only` stdout summary
(not a durable artifact, not in `effective_runtime.json` /
`pipeline_manifest.json` / `experiment_manifest.json`). Inconsistent with the
pipeline-based public identity goal but low risk.

**Fix direction:** Emit `pipeline.id` (or drop the field) in the cfg-only summary
for detection configs instead of an empty `trainer_variant`.

### P2-3 — Environment/baseline test debt obscures CI signal

**Evidence:** Full suite in `ms` env (excluding 2 swift-import collection errors
in `test_stage2_ab_packing_mask_gradients.py` / `test_swift_rollout_endpoints_contract.py`,
and `tests/analysis/`): **174 failed, 3380 passed, 147 skipped**. Sampling shows
the large majority are **pre-existing, not refactor-introduced**:
- Missing data: `FileNotFoundError: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
  (e.g. all 5 `test_detection_normalization_contract.py` failures).
- Missing doc referenced by an unmodified test:
  `docs/training/STAGE1_ET_RMP_CE.md` (absent at HEAD too) →
  `test_legacy_surface_absence.py`.
- Stale path in an unmodified test: `configs/stage2_rollout_correction`
  (real path is `configs/stage2/rollout_correction`) →
  `test_removed_training_mechanisms_absent.py:462/476`.
- torch/swift subprocess-import-heavy suites (`test_stage2_ab_training.py`,
  `test_stage2_two_channel_training.py`, 72 failures combined).

**Impact:** Makes "tests pass" claims hard to verify and lets genuine
regressions (P1-1, P1-2) hide in the noise. Not caused by this refactor, but it
means the readiness gate cannot rely on a green full suite.

**Fix direction:** Out of scope for this change, but the implementer should run
the import-safe and hierarchy-contract tests explicitly as part of the readiness
checklist, since they are clean (no data/hardware dependency) and currently red.

---

## Evidence Reviewed

**OpenSpec / plan / lifecycle:**
- `openspec/changes/refactor-sft-pipeline-hierarchy/design.md` (full),
  `tasks.md`, `proposal.md` (present), `specs/`.
- `docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md` (full).
- `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`,
  `repo_lifecycle/`.

**Code:**
- `src/training/pipeline_registry.py` (full), `src/training/pipelines/*`.
- `src/config/schema.py` (pipeline/sample_factory/objective/Stage-2 sections,
  rejection paths `_detection_reject_removed_target_hierarchy_paths` @4452,
  pairing @4516, legacy `teacher_forcing`/`token_ce` rejection @4068/@4358).
- `src/config/loader.py` (`_runtime_trainer_variant_for_config` @606,
  `_materialize_training_config` @946, `resolve_prompts` migration guards @445).
- `src/detection/runtime.py` (Stage-2 guard in `detection_mode` @186, shim @126).
- `src/sft.py` (`resolve_trainer_cls` @139, `_uses_stage1_detection_dataset_builder`
  @972, fingerprint @1702, effective-runtime payload @982, authored-experiment
  cleaning @759).
- `src/training_runtime/{plan,profile,preflight,stage2_projection}.py`.
- `src/bootstrap/{stage2_policy_provenance,pipeline_manifest,experiment_manifest}.py`,
  `src/utils/run_manifest.py`.
- `src/detection/packing.py` (static-packing eligibility/fingerprint).

**Configs:** `configs/base.yaml`, `configs/stage1/sft_base.yaml`,
`configs/stage1/profiles/2b/pure_ce_…`, `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`,
`configs/stage2/rollout_correction/{base,prod/…}.yaml`.

**Tests:** the refactor-touched set (18 files) plus the broken
`test_training_runtime_plan.py` / `test_training_runtime_profile.py` /
`test_training_config_hierarchy_contract.py`.

---

## Verification Run

| Check | Command | Result |
|---|---|---|
| Refactor-touched test files | `pytest <18 modified/added test files> + hierarchy contract` | **376 passed, 96 skipped, 1 failed** (hierarchy depth) |
| Import-safety | `pytest test_training_runtime_plan.py test_training_runtime_profile.py` | **3 failed** (torch/swift pulled in) |
| Packing/cache fingerprints | `pytest test_packing_cache_fingerprints.py test_stage1_static_packing_runtime_config.py` | 77 passed |
| Stage-2 provenance | `pytest test_stage2_policy_provenance.py` | passed; asserts `"trainer_variant" not in provenance` |
| Lifecycle registry | `pytest test_lifecycle_registry_report.py` | passed |
| OpenSpec | `openspec validate refactor-sft-pipeline-hierarchy --type change --strict` | **valid** |
| Whitespace | `git diff --check` | clean (exit 0) |
| Config resolution | materialize 3 prod configs via `ConfigLoader.load_materialized_training_config` | all resolve; pipeline/objective/runtime-variant correct (see below) |

Resolved identities (end-to-end through the real loader):

```
stage1 standard SFT  → pipeline=stage1_standard_sft            objective=standard_ce              runtime_variant=None
stage1 research TF    → pipeline=stage1_research_teacher_forcing objective=research_teacher_forcing runtime_variant=None
stage2 rollout corr.  → pipeline=stage2_rollout_correction      objective=None                     runtime_variant='stage2_rollout_correction'
```

**Confirmed OK / ruled out (prevents backtracking):**
- **Geometry/bbox/image alignment:** untouched by the refactor; migrated configs
  preserve `coordinate_surface: coord_token`, `bbox_format: xyxy`,
  `do_resize=false`; no coordinate drop/reorder. Shim
  (`runtime.py:126`) forwards `object_ordering`/`object_field_order`/`bbox_format`
  from `sample_factory.target_sequence` faithfully.
- **Reviewer issue #1 (Stage-2 entering Stage-1 detection builder):** fixed with
  two layers — `_uses_stage1_detection_dataset_builder` (`sft.py:972`) returns
  False when `runtime_plan.post_rollout_packing_owner is not None` (Stage-2), and
  `detection_mode` (`runtime.py:186`) hard-raises for Stage-2 configs.
- **Reviewer issue #2 (trainer_variant leak in public artifacts):** addressed —
  `effective_runtime` pops `trainer_variant` (`run_manifest.py:175`);
  `experiment_manifest`/`pipeline_manifest` carry `pipeline.id`
  (`experiment_manifest.py:132-138`, `pipeline_manifest.py`) not the selector;
  Stage-2 provenance omits it (test-asserted).
- **Reviewer issue #3 (empty default tuple/list authored fields):** fixed —
  `_clean_authored_experiment_payload` (`sft.py:759`) drops empty sequences;
  `DetectionExperimentConfig` defaults `key_deviations`/`runtime_settings` to `()`.
- **Reviewer issue #4 (`rollout_matching.pipeline` guidance):** fixed — both
  `schema.py:4649` and `stage2_projection.py:166` now direct users to
  `pipeline.id=stage2_rollout_correction`, not `custom.trainer_variant`.
- **Public selector:** `custom.trainer_variant` and `surface.id` rejected in
  target-hierarchy configs (`schema.py:4452`, test
  `test_stage2_rollout_correction_rejects_legacy_custom_selector`); `pipeline.id`
  is the sole public selector and is load-bearing (resolves runtime variant).
- **Objective vocabulary:** `objective.id: teacher_forcing` and `token_ce`
  rejected with migration guidance (`schema.py:4068/4358`); `standard_ce` /
  `research_teacher_forcing` are the public ids.
- **Packed research-TF fail-fast:** preserved — `research_teacher_forcing` rejects
  `training.packing=true` (`schema.py:3039-3059`); `compact_support2.yaml`
  authors `exact_packing_mapping.enabled: false` and `packing: false`.
- **Stage-2 separation:** distinct trainer (`Stage2RolloutCorrectionTrainer`),
  distinct runtime plan (identity collator, no dataset static packing, post-rollout
  trainer packing), residual-set-only objective; Stage-1 `objective.id` rejected
  for Stage-2 (`schema.py:4537`).
- **Cache/provenance additive identity:** `training_hierarchy` payload
  (`sft.py:1520`) added to fingerprints/manifests without replacing existing
  dataset/tokenizer/prompt/packing discriminators (regression tests pass).

---

## Residual Risks

1. **No production-scale or full-pipeline smoke run was performed** (per
   guardrails). Verified at the unit/config-resolution/manifest level only. A
   real Stage-2 rollout-correction launch and a Stage-1 packed launch should be
   smoke-tested before promotion to confirm runtime parity, since the refactor
   touched packing/cache fingerprints, the Stage-2 selection path, and the
   Stage-2 config inheritance graph.
2. **Shadow vs real selection duality (P2-1):** if future work assumes
   `TrainingPipelineRegistry` is the live seam, it may edit the wrong place.
   Drift between the registry's objective policy and `schema.py` pairing is
   currently uncaught by any cross-check test.
3. **Baseline test debt (P2-3):** the environment cannot produce a green full
   suite, so regressions are easy to miss; readiness should be gated on the
   specific clean tests, not the whole suite.
4. **Stage-2 dataloader defaults** silently changed by the `extends` removal
   (P1-2); confirm Stage-2 throughput/behavior is unaffected if option (b) is
   chosen.

---

## Recommended Next Step

**HOLD for one short fix-and-recheck cycle, then re-gate.** Specifically:

1. Fix **P1-1** by removing the `src.config.schema` import from
   `src/training_runtime/preflight.py` (inline the constant or relocate it to an
   import-light module). Re-run the three import-safe tests.
2. Resolve **P1-2** by deciding (with the user, since it is a config-hierarchy
   contract decision) whether to restore `extends: ../../base.yaml` on the
   Stage-2 base or to formally change the hierarchy contract + test + docs, then
   make config/test/design agree.
3. Re-run, at minimum: the refactor-touched suite **plus**
   `test_training_runtime_plan.py`, `test_training_runtime_profile.py`,
   `test_training_config_hierarchy_contract.py`, and report them green.
4. Address P2-1/P2-2 or record them as accepted residual debt.

Once P1-1 and P1-2 are green and the hierarchy decision is reconciled, the change
is in good shape for approval: the core research contracts (geometry, Stage-2
separation, provenance, fail-fast, public `pipeline.id` selection) are correctly
implemented and the four prior reviewer issues are genuinely fixed.
