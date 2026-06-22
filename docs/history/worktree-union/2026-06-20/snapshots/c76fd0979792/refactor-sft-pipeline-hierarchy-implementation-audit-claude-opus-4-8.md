# Implementation Audit (RE-AUDIT) — refactor-sft-pipeline-hierarchy

- Reviewer: Claude (Opus 4.8), read-only implementation audit
- Date: 2026-06-19 (re-audit after the author reported fixing the prior blocker)
- Worktree: `/data/CoordExp/.worktrees/codebase-refactoring-program`
- Branch: `codex/codebase-refactoring-program`
- Subject: the **uncommitted** working-tree diff over `HEAD` (`fde1eac9`). The SFT
  pipeline-hierarchy refactor remains uncommitted; `git diff` against `HEAD` is the
  implementation under review.

## Verdict

**Still not ready for the next human approval step — the previously-reported blocker is
genuinely fixed, but a full-suite run surfaces a real Stage-2 launch-path regression that
must be resolved first.**

Two things are true at once:

1. **The author's fix is real and complete for everything the prior audit flagged.** The
   import-safety blocker (old B1) is resolved cleanly, and *every* prior non-blocking item
   (N1–N4, the dead AB-scheduler test, the missing-doc test) is also fixed. This was careful,
   above-and-beyond work — see "Previously-Flagged Items: All Resolved."

2. **Running the entire test suite (which neither the prior audit's targeted runs nor the
   author's reported runs covered) exposes the same class of blind spot that hid the original
   B1: untested launch/integration paths.** 18 tests that pass on `HEAD` now fail on the
   working tree. 13 are stale-test/fixture debt, but **5 reflect a genuine break of the
   documented Stage-2 launch entrypoint**: migrated Stage-2 configs lose `.custom`, yet the
   vLLM launcher preflight still reads `custom.train_jsonl/val_jsonl/offline_max_pixels`. I
   reproduced the crash directly. That is a launch blocker (new **B1** below).

Methodology note for trust: I established cause by building a read-only `HEAD` baseline
(`git archive HEAD` → `/tmp`, data dirs symlinked) and diffing the failing-test sets. 173
working-tree failures vs 156 `HEAD` failures ⇒ exactly **18 newly broken by this diff**, 1
newly fixed, 155 pre-existing/environmental. Evidence below.

## Previously-Flagged Items: All Resolved (verified)

| Prior finding | Status | Evidence |
|---|---|---|
| **B1** import-safety break (`preflight.py` imported `src.config.schema`) | **FIXED** | New import-light module `src/objective_ids.py` (string constants only). `preflight.py:9` now imports from it. The 3 contract tests pass; interpreter check `import src.training_runtime.plan/preflight/profile` loads **none** of `torch/transformers/swift/datasets/src.config/src.trainers`. Bonus: preflight now resolves the runtime plan from `pipeline.id` (`_resolve_runtime_plan_for_preflight`) instead of `custom.trainer_variant`. |
| **N1** leftover legacy config + stale doc | **FIXED** | `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml` deleted; zero references remain in `configs/ docs/ tests/ src/ openspec/`. |
| **N2** Stage-2 error string used retired vocabulary | **FIXED** | `stage2_rollout_correction_impl.py:5615-5620` now says `objective.id=research_teacher_forcing with pipeline.id=stage2_rollout_correction`. |
| **N3** `--cfg-only` emitted `trainer_variant` | **FIXED** | `_build_cfg_only_summary` (`sft.py:801-836`) now emits `pipeline.id`/`objective.id`, no `trainer_variant` key. |
| **N4** `custom.trainer_variant` scope | **ADDRESSED** | New explicit helper `_runtime_trainer_variant_for_config` (`loader.py`): detection configs derive the variant from `pipeline.id`, non-detection `TrainingConfig` legitimately retains `custom.trainer_variant` (out of scope). The `schema.py:4981` message naming `custom.trainer_variant` sits inside `TrainingConfig.from_mapping` (non-detection path, reads `custom.trainer_variant` at 4927) and is therefore correct for its surface. |
| Dead AB-scheduler test (`from_dict` doesn't exist) | **FIXED** | `test_stage2_rollout_correction_removed_schedule_contract.py` now uses `TrainingConfig.from_mapping(payload, PromptOverrides())`; both params (`schedule`, `b_ratio`) execute and pass — the contract is now CI-guarded. |
| Missing-doc test (`STAGE1_ET_RMP_CE.md`) | **FIXED** | Stale path removed from `tests/test_legacy_surface_absence.py::ACTIVE_DOCS`; file passes (7/7). |

The substantive design intent verified in the prior audit still holds (re-checked): `pipeline.id`
is the real selector, the three lanes are schema-enforced, durable artifacts are off
`trainer_variant`, deferred packed-research-TF is fail-fast at multiple layers, `surfaces.py`
is retired, OpenSpec `--strict` validates, and `git diff --check` is clean.

## Blocking Findings

### B1 (NEW). Stage-2 vLLM launcher preflight crashes on migrated configs — `.custom` removed but the launcher still reads it

The refactor removed `custom` from `DetectionTrainingConfig` (migrated Stage-2 configs now
parse as `DetectionTrainingConfig` with **no** `.custom`). The main training entry in
`sft.py:2764-2771` was correctly updated to build a `_detection_runtime_custom_shim` for
detection configs. **But the Stage-2 launcher preflight was not updated and is not on that
shimmed path.**

`src/trainers/rollout_matching/preflight.py` (UNCHANGED by this diff) reads, in
`build_stage2_launcher_preflight`:

```python
preflight.py:204  train_jsonl_raw       = getattr(training_config.custom, "train_jsonl", None)
preflight.py:212  val_jsonl_raw         = getattr(training_config.custom, "val_jsonl", None)
preflight.py:217  offline_max_pixels_raw= getattr(training_config.custom, "offline_max_pixels", None)
```

`resolve_stage2_launcher_preflight` (same file) loads the config via
`ConfigLoader.load_materialized_training_config(...)` → returns `DetectionTrainingConfig` →
`build_stage2_launcher_preflight(...)` → `AttributeError: 'DetectionTrainingConfig' object has
no attribute 'custom'`.

This is the **documented, canonical Stage-2 launch entrypoint**, not a legacy corner:

- `scripts/train_stage2.sh:35` → `exec python -m src.launchers.stage2_vllm_server`
- `src/launchers/stage2_vllm_server.py:671` → `preflight = resolve_stage2_launcher_preflight(str(config_path))`
- `docs/training/STAGE2_RUNBOOK.md:344` documents `bash scripts/train_stage2.sh` as the launch path.

Direct reproduction (read-only, `ms` env):

```
$ python -c "from src.trainers.rollout_matching.preflight import resolve_stage2_launcher_preflight; \
  resolve_stage2_launcher_preflight('configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml')"
AttributeError: 'DetectionTrainingConfig' object has no attribute 'custom'
```

(The prod config `…/prod/coco1024_online_residual_correction_vllm_tail_append.yaml` trips an
earlier single-server check in my invocation only because I did not pass the runtime-injected
server URL; line 204 executes once that check passes — and the vLLM single-server preflight
contract tests below confirm the path is broken.)

Tests that **passed on `HEAD` and fail now** (refactor-caused; were green because `HEAD`'s
`DetectionTrainingConfig` still had `.custom`):

- `tests/test_stage2_launcher_server_template_flags.py::test_stage2_launcher_default_config_is_single_server_vllm_preflight`
- `tests/test_stage2_launcher_server_template_flags.py::test_stage2_launcher_round_trips_preflight_engine_kwargs`
- `tests/test_stage2_preflight_path_resolution.py::test_stage2_preflight_resolves_root_image_dir_relative_to_repo_root_not_cwd`
- `tests/test_stage2_preflight_server_knob_plumbing.py::test_stage2_preflight_extracts_server_runtime_knobs_from_yaml`
- `tests/test_stage2_launcher_preflight_contract.py::test_stage2_teacher_forcing_packing_fails_before_launcher_rollout_setup`
  (related regression: this one **`DID NOT RAISE`** — the launcher's TF-packing fail-fast guard
  no longer fires, a behavioral regression on the same preflight surface.)

Fix: route `build_stage2_launcher_preflight` through the same detection runtime custom shim used
by `sft.py` (`_detection_runtime_custom_shim`), or read the migrated surface directly
(`data.train_jsonl` / `data.val_jsonl` / the offline max-pixels location) for detection configs.
Then re-confirm the TF-packing fail-fast guard still triggers.

Why blocking: the canonical Stage-2 launch path crashes in preflight for migrated Stage-2
configs. Unlike the old import-safety B1 (which did not affect runtime training), this prevents
Stage-2 server-mode launches from starting. It also escaped "tests pass" for the same reason the
old B1 did — the contract tests for this path were not in the targeted suites that were run.

## Non-Blocking Findings

### NB1. 10 compact span-adapter/projector tests use the retired `compact_full` template id (stale-test debt; production is correct)

The diff *correctly generalized* `src/training/span_adapters/compact_projector.py` from a
hardcoded `if view.template_id != "compact_full": raise` to
`if not is_compact_template_id(view.template_id): raise`. Migrated configs resolve to **supported**
semantic ids (`compact_support2.yaml` → `compact`; the sorted/packed Stage-1 profile →
`compact_object_box_closed`), so production is fine. The fixtures in
`test_compact_span_projector.py` / `test_stage1_compact_span_adapter.py` /
`test_stage2_compact_span_adapter.py` still author `detection_template.id="compact_full"`, which
`template_contracts.py:124` rejects (supported ids: `stage1_json_pretty, compact,
compact_box_closed, compact_object_closed, compact_object_box_closed,
compact_object_box_closed_lines`). Update these fixtures to a supported compact id. (Note: the
helper `is_compact_template_id` *raises* on an unknown id rather than returning `False`; benign
for valid compact ids but worth tidying.)

### NB2. 2 test-side `.custom` / `.get()` accesses on migrated config classes (stale-test debt)

- `tests/test_coord_token_mode_invariants.py:18` — reads `.custom` on a `DetectionTrainingConfig`.
- `tests/test_ms_swift_placeholder_dataset_contract.py:8` — calls `.get()` on a `DetectionDataConfig`.

Both are test-side (first access is in the test) and were green on `HEAD`. Production readers are
already migration-aware (only the B1 launcher preflight was missed — see blast-radius check).
Update these tests to the migrated surfaces; the second one means the
stage1/stage2 anchored-config placeholder-dataset invariant is currently unguarded.

### NB3. Stage-2 golden-thread fixture still authors `runtime.trainer_variant` (fixture residue, conflicts with the new registry contract)

`tests/fixtures/training_architecture/stage2_rollout_source.json:49` (inside the
`pipeline_config.runtime` block the diff itself restructured) still has
`"trainer_variant": "stage2_rollout_correction"`. The refactor's own new contract
(`test_training_pipeline_registry.py::test_registry_rejects_trainer_variant_in_runtime_domain`)
makes `TrainingPipelineRegistry().resolve` reject `runtime.trainer_variant`, so
`build_stage2_golden_thread` raises `Unknown runtime keys: ['runtime.trainer_variant']`. This test
also fails on `HEAD` (so it is **not** a new regression and is excluded from the 18), but the diff
touched this fixture and left residue inconsistent with the contract it introduced — clean it up so
the Stage-2 golden-thread snapshot is exercised.

## Evidence Reviewed

Fix verification (all read-only, `ms` env at `/root/miniconda3/envs/ms/bin/python`):

- `src/objective_ids.py` (new, untracked): import-light string constants
  (`STANDARD_CE_OBJECTIVE_ID`, `RESEARCH_TEACHER_FORCING_OBJECTIVE_ID`,
  `LEGACY_TEACHER_FORCING_OBJECTIVE_ID`, `TEACHER_FORCING_OBJECTIVE_ID =
  RESEARCH_TEACHER_FORCING_OBJECTIVE_ID = "research_teacher_forcing"`). `schema.py:71-76` imports
  these; no local re-definition remains (no drift).
- `import src.training_runtime.{plan,preflight,profile}` → banned-module probe returns `[]`.
- Targeted contract/integration suite (plan, profile, pipeline_registry, detection contract,
  teacher_forcing contract, stage2 contract + profile-leaf + removed-schedule, stage2 provenance,
  experiment/run manifest, hierarchy contract, objective-profile, encoded-cache, static-packing,
  prompt-variants, legacy-surface-absence, artifact-contract-docs) → **386 passed, 96 skipped, 0
  failed**. Integration + detection contract → **96 passed, 0 failed**.
- `openspec validate refactor-sft-pipeline-hierarchy --strict` → valid. `git diff --check` → clean.

Launch-path break (B1):

- `sft.py:2764-2771` builds the detection runtime custom shim for detection configs (migration-aware).
- Only production non-test reader of `.custom` that can receive a migrated detection config is
  `rollout_matching/preflight.py:{204,212,217}` (the launcher preflight). The diff does not touch
  `preflight.py`, `stage2_vllm_server.py`, or `scripts/train_stage2.sh`.
- Migrated Stage-2 configs parse as `DetectionTrainingConfig` with `hasattr(cfg,'custom') == False`.
- Direct invocation of `resolve_stage2_launcher_preflight` on the HF smoke config → `AttributeError`.

## Verification Run

Full-suite causation analysis (the key new evidence):

- Worktree full suite (excluding 3 known import-erroring modules: `tests/analysis/*`,
  `test_stage2_ab_packing_mask_gradients.py`, `test_swift_rollout_endpoints_contract.py`):
  **173 failed, 3396 passed, 147 skipped** (236s).
- `HEAD` baseline: `git archive HEAD | tar -x -C /tmp/coordexp_head` (sha `fde1eac9`), gitignored
  `output/`/`temp/` symlinked in to remove the path/data confounder; same failing files re-run.
  **156 failed.**
- Set diff of failing test ids: **18 fail in worktree but pass on `HEAD` (refactor-caused)**, 1
  passes in worktree but fails on `HEAD` (refactor-fixed), 155 fail in both (pre-existing).
- The 18 refactor-caused, by root cause:
  - **5** — B1 Stage-2 launcher preflight (`.custom` ×4 + TF-packing guard `DID NOT RAISE` ×1).
  - **10** — NB1 stale `compact_full` template-id fixtures (`template_contracts.py:124`).
  - **2** — NB2 test-side `.custom`/`.get()` on migrated config classes.
  - **1** — NB1 projector rejection-message regex (`test_compact_span_projector.py:152`).

Pre-existing / environmental red (155 failures, fail identically on `HEAD` — **not** caused by
this diff; do not block this change):

- 72 in `test_stage2_ab_training.py` + `test_stage2_two_channel_training.py` — exercise removed
  AB/two-channel mechanisms and hit an unchanged `_stage2_checkpoint_identity_from_owner`
  precondition (verified byte-identical to `HEAD`; the diff touches neither `checkpoint_identity`
  nor `decode_provenance`).
- Missing data/checkpoints (`FileNotFoundError` on `public_data/...`, `output/stage1/...`),
  swift API drift (`Stage2RolloutRuntime`/`Stage2RolloutCorrectionTrainer` attribute drift,
  `swift.llm` symbols), `cpu_ddp` multiprocess tests, codebase policy/lint scans
  (`test_no_silent_except_exception_pass`, `test_silent_failure_policy`,
  `test_infer_layout_import_gates`, `test_removed_training_mechanisms_absent`), numeric drift
  (`length_insensitive_loss_normalization`), and `test_teacher_forcing_loss_catalog` (the catalog
  key `duplicate_burst_prefix_rollback` predates this diff; the diff only removed the adjacent
  retired `TEACHER_FORCING_OBJECTIVE_ID` literal).
- Plus 15 pre-existing collection errors (`tests/analysis/*`, swift API drift) — unchanged from the
  prior audit; none of the erroring files are touched by this diff.

## Residual Risks

- **No end-to-end launch executed.** B1 is proven at the preflight boundary (the launcher's first
  step) and via the contract tests, not by a full `scripts/train_stage2.sh` run (out of scope for a
  read-only audit; requires GPU/servers). A 1-step smoke launch remains the only in-situ proof once
  B1 is fixed.
- **Pre-existing red is large (155) and partly environmental.** I could not run a GPU/data-complete
  environment, so some of the 155 may be infra-only and would pass in CI; I did not attempt to
  green them (out of scope, and they predate this diff). They are flagged so the approver is not
  surprised, but they should not gate this change.
- **Baseline caveat:** the `HEAD` archive shares the worktree's gitignored data via symlink, so the
  set diff isolates code differences for the config-surface failures (the ones that matter here);
  a handful of path-sensitive tests could in principle differ, but all 18 refactor-caused failures
  have config-surface (path-independent) signatures.
- Geometry/bbox/image-alignment and cache-fingerprint additivity were reviewed structurally in the
  prior pass and are unchanged by the fix; not re-run here.

## Recommended Next Step

1. **Fix B1:** make `build_stage2_launcher_preflight` migration-aware — read JSONL/max-pixels from
   the detection runtime custom shim (as `sft.py` does) or the migrated `data.*` surface — and
   re-confirm the launcher's research-TF + packing fail-fast guard still raises. Re-green the 5
   `test_stage2_launcher_*`/`test_stage2_preflight_*` tests.
2. **Clear the stale-test/fixture debt** so "tests pass" is true: update the `compact_full`
   fixtures to a supported compact id (NB1, 11 tests), update the 2 `.custom`/`.get()` tests to the
   migrated surfaces (NB2), and remove `runtime.trainer_variant` from
   `stage2_rollout_source.json` (NB3).
3. **Run the full suite (not just targeted files)** and diff against a `HEAD` baseline before the
   next approval request — the original B1 and this new B1 both escaped because the broken paths
   were outside the suites that were run.
4. Re-request approval once B1 is green and the full-suite refactor-caused set is empty. The 155
   pre-existing failures and 15 collection errors should be acknowledged to the approver but are not
   caused by this change.

Bottom line: the author cleanly fixed everything the prior audit raised — that part is done well.
But the migration removed `DetectionTrainingConfig.custom` without updating the Stage-2 vLLM
launcher preflight that still reads it, so the documented Stage-2 launch entrypoint crashes for
migrated configs. That is a real, reproducible launch blocker and must be fixed before approval;
the remaining 13 newly-red tests are stale-test/fixture debt worth clearing in the same pass.
