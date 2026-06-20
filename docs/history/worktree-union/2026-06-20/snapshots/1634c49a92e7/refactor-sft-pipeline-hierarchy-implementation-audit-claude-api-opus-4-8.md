# Implementation Audit — refactor-sft-pipeline-hierarchy

- **Auditor:** Claude (Opus 4.8), read-only review
- **Date:** 2026-06-19
- **Worktree:** `/data/CoordExp/.worktrees/codebase-refactoring-program`
- **Branch:** `codex/codebase-refactoring-program`
- **Change under review:** `openspec/changes/refactor-sft-pipeline-hierarchy/`

> **Topology note (important):** The branch tip `HEAD` (`fde1eac9`) equals the
> merge-base with `main`. The **entire refactor is uncommitted working-tree
> change** (64 files, +4192/−2006) layered on `fde1eac9`. `src/training/surfaces.py`
> (−690) and `tests/test_training_surface_resolver.py` (−360) are deleted in the
> working tree; `src/training/pipeline_registry.py`, `repo_lifecycle/`, and
> lifecycle tests are new/untracked. All findings below are against that
> working-tree state.

---

## Verdict

**Ready for the next human approval step — with non-blocking follow-ups. No
blocking correctness, contract, or launch defect was found in any *wired*
surface.**

I verified the substance, not just the test tally. The three-way split
(standard SFT / research teacher-forcing / Stage-2 rollout correction) is **real,
not nominal** — it is enforced at the config schema, the loader's runtime-variant
derivation, and the dataset-builder routing, and I confirmed each boundary by
loading the actual migrated configs and exercising the runtime resolvers. All
four previously-reported reviewer issues are genuinely fixed, with negative
assertions in tests to match. Fail-fast for deferred packed research-TF is
correct. Cache/packing/provenance identity additively includes the normalized
hierarchy. Artifacts are coherent (no `trainer_variant`/`token_rows`/`custom`
leakage) and round-trip idempotent.

The reasons this is not an unqualified "clean" are two honest gaps, neither of
which breaks a wired path:

1. One **orphaned, unmigrated active config** that hard-fails to load (fails
   *safely*, fail-fast), contradicting the "active configs migrated" completeness
   claim.
2. **No test asserts active configs materialize** under the new schema — the gap
   that let #1 slip through.

Both are "should-fix before declaring the migration complete," not launch
blockers. Details and exact paths below.

---

## Blocking Findings

**None.** Every config reachable through the catalog / `extends` graph / test
graph loads and routes correctly, the public selector and objective contracts
behave exactly as the OpenSpec deltas require, and the Stage-1/Stage-2 split is
enforced in production code (not only in a parallel test schema).

To be explicit about what I checked for and did *not* find:
- No path where a Stage-2 config enters the Stage-1 detection dataset builder.
- No `trainer_variant` leak in any manifest carrier or policy provenance.
- No active config that *recommends* a retired selector.
- No silent dual-source for object ordering / strict-parse.
- No geometry/bbox/field-order reordering or drop.
- No packing path that bypasses the research-TF exact-remapping guard.

---

## Non-Blocking Findings

### NB-1 (should-fix) — Orphaned unmigrated active config hard-fails to load

`configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml` is a
complete, git-tracked, standalone **smoke** config still authored entirely in the
retired dialect:

- `token_rows:` (not `token_embeddings_adapter`)
- `prompt.prompt_variant_enabled: true` (not `prompt.variant`)
- `objective.id: teacher_forcing` (not `research_teacher_forcing`)
- `detection_template.{coordinate_surface,bbox_format,strict_parse}` (not under
  `sample_factory.target_sequence`)
- no `pipeline:`, no `sample_factory:`

Loading it fails:

```
ValueError: flat token_rows is retired; use top-level token_embeddings_adapter
```

Severity rationale (why non-blocking):
- It is **orphaned** — not referenced by `docs/catalog.yaml`, `docs/AGENT_INDEX.md`,
  any `extends` chain, or any test. The lifecycle registry names the *migrated*
  directory `configs/stage1/detection_teacher_forcing` as the active research-TF
  location (`lifecycle_registry.yaml:158`).
- A **migrated equivalent loads cleanly**:
  `configs/stage1/detection_teacher_forcing/smoke/compact_tiny.yaml` →
  `pipeline.id=stage1_research_teacher_forcing`, `objective=TeacherForcingObjectiveConfig`.
- It fails **fast and safely** (clear migration error, no silent wrong behavior),
  which is actually the refactor's fail-fast contract working.

Why it still matters: it is a single leftover directory (`teacher_forcing/`, vs
the migrated `detection_teacher_forcing/`) that the residue-elimination goal of
this program was meant to remove, and it directly dents the "active configs
migrated" claim. Recommend deleting or migrating it.

### NB-2 (should-fix) — No config-materialization completeness test

The only suite that sweeps all configs,
`tests/test_legacy_config_contract.py::test_active_legacy_configs_do_not_rely_on_custom_coord_loss`
(lines 44–58), globs `configs/**/*.yaml` but only checks for a single key
(`custom.coord_loss`). **Nothing asserts that active Stage-1/Stage-2 training
configs actually load under the new schema.** This is precisely the gap that let
NB-1 ship undetected.

My own sweep (`ConfigLoader.load_materialized_training_config` over
`configs/stage1` + `configs/stage2`, archive excluded): **22 load OK, 7 fail.** Of
the 7, six are legitimate `extends`/`_shared` base fragments that are not meant to
load standalone (`_shared/coord_soft_ce_gate_*.yaml`, `sft_base.yaml`,
`*/smoke/common_prodlike.yaml`); the seventh is NB-1. A real completeness test
should enumerate **leaf** training configs (resolving `extends`) and assert each
materializes, so the next unmigrated leaf is caught automatically.

### NB-3 (code hygiene, pre-existing) — `recursive_detection_ce` is dead-but-live

`recursive_detection_ce` remains a first-class objective id in:
- `DetectionObjectiveConfig.id: Literal[...]` (`schema.py:4155`, allowed in
  `__post_init__` at `4176`),
- active runtime handling (`detection/runtime.py:244`, and
  `resolve_recursive_detection_ce_runtime_cfg`, exported at `runtime.py:583`),

…but it is **unconstructible**: `DetectionObjectiveConfig.from_mapping`
(`schema.py:4346–4377`) has no dispatch branch for it, so any
`objective.id: recursive_detection_ce` payload falls through to
`raise ValueError("objective.id must be 'standard_ce' or 'research_teacher_forcing'…")`.
I confirmed empirically:

```
FAIL id=recursive_detection_ce variant=prefix_rollin_et_rmp_ce -> ValueError: objective.id must be 'standard_ce' or 'research_teacher_forcing'…
```

The only tests that would exercise the runtime resolver
(`test_detection_training_config_contract.py:1096/1152/1185`) are among **56
skipped** tests (`reason="legacy recursive_detection_ce config contract retired by
teacher_forcing objective"`).

Severity rationale (why non-blocking, and not a spec violation):
- **Pre-existing**, not introduced here — `from_mapping` at `HEAD` also lacked a
  `recursive_detection_ce` branch, and the 56 skips are identical in count at
  `HEAD`.
- The spec only requires ET-RMP/recursive lineage to remain **"inspectable or
  runnable as preserved comparator lineage"**
  (`specs/stage1-detection-objectives/spec.md:55–62`), which is satisfied by the
  committed `configs/infer/recursive_detection_ce/` inference/eval routes (these
  use a separate infer schema, not `DetectionTrainingConfig`). No **active
  training** config authors `recursive_detection_ce` (verified: zero non-archive
  hits).

Why flag it anyway: this is exactly the "looks like infrastructure but is dead"
residue the refactoring program exists to reduce. A first-class id with a
validated dataclass and an exported runtime resolver that can never be
constructed, guarded only by skipped tests, is a coherence smell worth a
follow-up (either restore a dispatch branch behind the research lane, or remove
the id/Literal/runtime handlers/exports).

### NB-4 (clarity) — `pipeline_registry.py` is a renamed shadow resolver, test-only

`src/training/pipeline_registry.py` is the rename of the deleted `surfaces.py`
(`TrainingSurfaceResolver` → `TrainingPipelineRegistry`). It has **zero
production importers** — only `tests/test_training_pipeline_registry.py`,
`tests/test_objective_profile_resolution.py`, and
`tests/helpers/training_architecture_fixture_builder.py` use it. It also uses a
**different "shadow schema"** (`run/data/template/supervision/objectives/
observability/artifacts/runtime/experimental`) than the production config
(`pipeline`/`sample_factory.target_sequence`/`detection_template`/`objective`).

Production training-family selection actually flows through
`DetectionPipelineConfig` (`schema.py:3281`) →
`ConfigLoader._runtime_trainer_variant_for_config` (`loader.py:607–615`) →
`resolve_training_runtime_plan` (`training_runtime/plan.py:41`). The OpenSpec
design frames "pipeline registry as the public selection seam"
(`design.md:119–133`), which a reader could over-interpret as the runtime path.

This is **not a regression** — `surfaces.py` was equally test-only — and the
production seam is correct and coherent. Flagged only so reviewers understand the
registry is governance/aspirational scaffolding, not the wired selector. The
deletion of `surfaces.py` and absence of a compat shim are correctly enforced by
`test_surfaces_module_is_not_a_long_lived_compatibility_shim`
(`test_training_pipeline_registry.py:173`).

### NB-5 (doc nit) — Stale "alias candidate" wording

`docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml:77`
still describes `teacher_forcing` as a *"Narrow migration alias candidate for
research_teacher_forcing"*, whereas the locked decision (and the code) is outright
**rejection**, not aliasing (`schema.py:4358–4363`,
`tasks.md:2.3`). Cosmetic; update wording to "rejected (not an alias)".

### Pre-existing, out-of-scope (not introduced by this change)
- **15 test-collection errors** in the full suite, all in **unchanged**,
  out-of-scope files: `tests/analysis/**` (analysis subtree is an explicit
  Non-Goal, `design.md:89`), `tests/test_stage2_ab_packing_mask_gradients.py`
  (`ImportError: get_model_tokenizer from swift.llm` — environment), and
  `tests/test_swift_rollout_endpoints_contract.py`
  (`ImportError: apply_coord_row_patch_for_rollout_server` — symbol absent at
  `HEAD` too). None touch the refactor surface; all refactor-touched test files
  (455 tests) collect cleanly.

---

## Evidence Reviewed

**Intent artifacts (read in full):** `proposal.md`, `design.md`, `tasks.md`, the
nine relevant delta specs (`sft-pipeline-hierarchy`, `stage2-rollout-correction`,
`rollout-matching-sft`, `teacher-forcing-unified-loss-registry`,
`stage1-detection-objectives`, …), the superpowers plan
(`docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md`), and
`lifecycle_registry.yaml`.

**Core code (read + traced):**
- `src/config/schema.py` — `DetectionPipelineConfig` (3281), `DetectionTargetSequenceConfig`/`DetectionSampleFactoryConfig` (3305/3352), `DetectionObjectiveConfig` (4153) + `TeacherForcingObjectiveConfig` (3991), `_detection_validate_pipeline_objective_pairing` (4516), `_detection_reject_removed_target_hierarchy_paths` (4452), `DetectionTrainingConfig.from_mapping`/`to_mapping` (4572/4766).
- `src/config/loader.py` — `_runtime_trainer_variant_for_config` (607), `_is_detection_training_config_payload` (959).
- `src/sft.py` — `_uses_stage1_detection_dataset_builder` (972), `_build_normalized_training_hierarchy_identity` (1520), `_build_static_packing_fingerprint` (1702), encoded-cache fingerprint (1988), `_clean_authored_experiment_payload` (759), Stage-2 routing (3089–3169).
- `src/detection/runtime.py` — `build_detection_runtime_custom_shim` (126), `detection_mode` (183), `resolve_detection_runtime_support` (235), `assert_detection_runtime_supported` (252).
- `src/training_runtime/plan.py` — `resolve_training_runtime_plan` / `_stage2_plan`.
- `src/bootstrap/{stage2_policy_provenance,experiment_manifest,pipeline_manifest}.py`, `src/utils/run_manifest.py`, `src/detection/packing.py`, `src/training/pipeline_registry.py`.

**Tests inspected:** `test_training_pipeline_registry.py`, `test_detection_training_config_contract.py`, `test_stage2_policy_provenance.py`, `test_experiment_manifest_file.py`, `test_legacy_config_contract.py`, and others.

**Reviewer-fix verification (all four CONFIRMED with evidence):**
1. *Stage-2 entering Stage-1 builder* — `_uses_stage1_detection_dataset_builder` returns `False` when `runtime_plan.post_rollout_packing_owner` is set; `detection_mode()` additionally raises if a Stage-2 config reaches the Stage-1 runtime (`runtime.py:186–191`). Empirically: Stage-2 base config → `uses_stage1_detection_builder=False`.
2. *`trainer_variant` artifact leak* — `stage2_policy_provenance` now gates on and emits `pipeline.id`, no `trainer_variant` (diff confirmed); `run_manifest.py` pops `trainer_variant` from `effective_runtime`; `experiment_manifest.py` dropped it from carried keys. Empirically: provenance built from `pipeline.id` only → `{"pipeline":{"id":"stage2_rollout_correction"}}`, `"trainer_variant" not in provenance`. Tests assert the negative (`test_stage2_policy_provenance.py:32,95`; `test_experiment_manifest_file.py:69,119`).
3. *Empty default tuple/list emission* — `_clean_authored_experiment_payload` (`sft.py:759`) strips empty sequences; `test_teacher_forcing_authored_experiment_preserves_claim_scope` asserts `authored == {"surface":"smoke","claim_scope":"smoke"}` exactly.
4. *Fail-fast guidance text* — `docs/training/STAGE2_RUNBOOK.md:166` lists `custom.trainer_variant: stage2_rollout_correction` under "deprecated authored knobs fail fast" with "(use top-level `pipeline.id: …`)". The `schema.py:4979` reference is inside the legacy `TrainingConfig` path describing its own contract, not a recommendation for target-hierarchy configs.

---

## Verification Run

Environment: `ms` conda env (`source activate ms`). No training/inference jobs run.

| Check | Result |
|---|---|
| `openspec validate refactor-sft-pipeline-hierarchy --type change --strict` | **valid** (exit 0) |
| `git diff --check` | **clean** (exit 0) |
| `python -m repo_lifecycle.report_lifecycle_registry` | ran; only the known intentional `compact_full_public_refs_need_owner_review` warning |
| Broad targeted suite (18 files: contract, objective, registry, stage2, packing, cache, manifests, prompts, runtime integration, lifecycle, artifact-contract) | **409 passed, 96 skipped, 2 warnings** |
| `test_lifecycle_registry_report.py` + packing/stage2/runtime integration | **121 passed** |
| Refactor-touched test files collection | **455 collected, 0 errors** |

**Empirical probes I ran directly (beyond the suite):**
- End-to-end load of migrated Stage-1 std-CE, Stage-1 research-TF, and Stage-2
  configs → all load; Stage-2 derives `variant='stage2_rollout_correction'`,
  `post_rollout_owner=trainer`, `uses_stage1_detection_builder=False`; Stage-1
  variants → `uses_stage1_detection_builder=True`.
- Research-TF runtime → `TeacherForcingObjectiveConfig`,
  `tf_target_ir_required=True`; flipping `training.packing=true` →
  packing guard fires: *"latest research_teacher_forcing target IR requires
  training.packing=false; exact atom-position packing mapping is not implemented
  yet."*
- 11/11 dual-authoring / migration guards reject with precise, migration-pointing
  messages (`custom.trainer_variant`, `surface.id`, `pipeline_id`,
  `custom.object_field_order`, duplicate `data.object_ordering`, duplicate
  `detection_template.strict_parse`, `target_sequence.template_id`,
  `prompt.prompt_variant_enabled`, flat `token_rows`, `objective.id: teacher_forcing`,
  `objective.id: token_ce`).
- Cache identity: `random` vs `random_permutation` preserved **distinctly** in the
  normalized fingerprint hierarchy (no stale-cache collision), even though the
  dataset's internal ordering field collapses both to `"random"`.
- Resolved-config `to_mapping()` emits coherent public identity and **omits**
  `data.object_ordering`, `token_rows`, `custom`, `trainer_variant`; load →
  to_mapping → reload is **idempotent** for all three pipeline types.
- Residue scan (plan Task-6 Step-4 pattern) over active trees: no active
  config/doc/spec **recommends** a retired public selector (remaining hits are
  rejection maps, internal module names explicitly allowed by the
  rollout-matching spec, or the plan's own "reject these" list).

---

## Residual Risks

1. **Migration completeness is unverified by tests (NB-2).** I found NB-1 by
   sweeping configs myself; CI would not. Until a leaf-config materialization test
   exists, future unmigrated leaves can ship. *Mitigation: fail-fast means they
   error loudly at load, not silently mistrain.*
2. **No production smoke / real-data training run was performed** (out of scope by
   instruction). Schema, routing, fingerprints, manifests, and packing guards are
   validated by construction and unit tests; an actual tiny end-to-end Stage-1 and
   Stage-2 launch was not exercised here.
3. **Two parallel config worlds** (production `DetectionTrainingConfig` vs the
   test-only `TrainingPipelineRegistry` shadow schema, NB-4). They are kept
   consistent by intent and tests, but divergence between them would not be caught
   by a single failing assertion.
4. **`recursive_detection_ce` dead surface (NB-3)** could mislead a future agent
   into thinking ET-RMP training is selectable via `DetectionTrainingConfig`; it is
   not (only inference/comparator + archive).
5. **Packing fail-fast is the only thing standing between research-TF and
   incorrect packed supervision.** It is correct and well-tested today
   (`runtime.py:259–289` rejects `packing`/`eval_packing`/`static_packing`/
   `padding_free_packed`/`encoded_sample_cache`). The deferred remapping remains a
   genuine future-work item, as the spec intends.

---

## Recommended Next Step

**Proceed to human approval.** The refactor is substantially correct, the
intended split is real and enforced, all four reviewer fixes are verified, and
no wired path is broken.

Before final sign-off, ask the implementer to close two cheap gaps so the
"migration complete" claim is honest and self-defending:

1. **NB-1:** delete or migrate
   `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml` (a
   migrated equivalent already exists under `detection_teacher_forcing/`).
2. **NB-2:** add a test that enumerates **leaf** active Stage-1/Stage-2 training
   configs (resolving `extends`) and asserts each materializes via
   `ConfigLoader.load_materialized_training_config`.

NB-3/NB-4/NB-5 are lower-priority hygiene/clarity items that can be scheduled as
follow-ups rather than gating this approval.
