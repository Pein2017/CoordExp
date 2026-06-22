# Refactor SFT Pipeline Hierarchy Implementation Audit

## Verdict

**Hold. Not ready for the next human approval step.**

The implementation passes several narrow checks and does implement important pieces of the intended break: active migrated config leaves use `pipeline.id`, Stage-2 train arguments are derived from `pipeline.id`, `custom.trainer_variant` is not emitted in the checked run/experiment manifest paths, and Stage-2 policy provenance records `pipeline.id`.

However, I found blocking contract mismatches. The largest issue is that `src/training/pipeline_registry.py::TrainingPipelineRegistry` is documented as the active resolver but is not used by active runtime code and cannot resolve the migrated active YAML shapes. There is also a non-archive Stage-1 teacher-forcing smoke config still using retired fields and failing to load, plus an artifact/provenance gap where Stage-2 normalized hierarchy records `objective.id: null` despite the OpenSpec requiring objective identity in resolved configs, manifests, provenance, and cache identity.

Audit mode: `implementation-vs-contract audit` with `change/spec audit`.

## Blocking Findings

### P1: Documented pipeline registry is not the live resolver and rejects real migrated configs

Evidence:

- `docs/catalog.yaml:22-25` declares the active Stage-2 training surface resolver as `src/training/pipeline_registry.py::TrainingPipelineRegistry`.
- `docs/AGENT_INDEX.md:62-64` sends Stage-1 agents to `TrainingPipelineRegistry`, and `docs/AGENT_INDEX.md:74` says Stage-2 is resolved through it.
- `openspec/changes/refactor-sft-pipeline-hierarchy/design.md:119-133` says future agents should look for pipeline selection in a pipeline registry and that `surfaces.py` should be replaced/deleted after active imports move.
- But `src/config/loader.py:607-615` derives the only live Stage-2 runtime selector directly from `DetectionTrainingConfig.pipeline.id`, and `src/config/loader.py:639-641` feeds that value into `TrainArguments`; it does not call `TrainingPipelineRegistry`.
- Source scan result: `rg -n "TrainingPipelineRegistry|pipeline_registry" src` returns only `src/training/pipeline_registry.py:278:class TrainingPipelineRegistry`.
- Probe result: `TrainingPipelineRegistry().resolve()` rejects both `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` and `configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml` with `ValueError Unknown top-level config domains`, because `src/training/pipeline_registry.py:403-419` expects its own `run/pipeline/data/template/supervision/objectives/...` domain schema rather than the actual migrated YAML schema.

Impact:

This makes the refactor's central registry separation mostly nominal. Active training still routes through `ConfigLoader` plus `src/sft.py` and an internal `trainer_variant` bridge for Stage-2, while the documented resolver is a parallel/test-only schema. That undermines the main design goal of making standard SFT, research teacher forcing, and Stage-2 discoverable through one public pipeline selection seam. It also creates a navigation trap: docs tell future agents to inspect a resolver that does not resolve real active configs.

Fix direction:

Either make `TrainingPipelineRegistry` the actual resolver for migrated active `DetectionTrainingConfig` objects, or rewrite docs/spec implementation claims so the first slice honestly says pipeline selection is schema-owned in `ConfigLoader` and the registry is only a future/fixture seam. If the registry remains, add tests proving it can resolve real migrated config leaves after materialization, including `compact_support2.yaml` and at least one Stage-2 smoke leaf.

Smallest verification:

```bash
python - <<'PY'
from pathlib import Path
import yaml
from src.training.pipeline_registry import TrainingPipelineRegistry
for path in [
    "configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml",
    "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml",
]:
    payload = yaml.safe_load(Path(path).read_text())
    print(path, TrainingPipelineRegistry().resolve(payload))
PY
```

Expected after fix: both active configs resolve, or docs no longer claim the registry is the active resolver.

### P1: A non-archive Stage-1 teacher-forcing config still uses retired hierarchy and now fails to load

Evidence:

- `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml:24-34` still uses `data.object_ordering` and `prompt.prompt_variant_enabled`.
- `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml:36-42` still authors flat `detection_template.coordinate_surface`, `detection_template.bbox_format`, `detection_template.strict_parse`, and flat `token_rows`.
- `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml:63-72` still authors `objective.id: teacher_forcing` plus `objective.modules`.
- The OpenSpec says active migrated configs must reject old `teacher_forcing` rather than accept it (`openspec/changes/refactor-sft-pipeline-hierarchy/specs/sft-pipeline-hierarchy/spec.md:180-205`) and must use normalized hierarchy fields (`openspec/changes/refactor-sft-pipeline-hierarchy/proposal.md:16-30`).
- Probe result: `ConfigLoader.load_materialized_training_config("configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml")` fails with `ValueError flat token_rows is retired; use top-level token_embeddings_adapter`.
- `docs/architecture/proposals/2026-06-17-refactoring-program/REFACTORING_PROGRAM_CHARTER.md:582-588` lists `configs/stage1/teacher_forcing` as a cleanup candidate "if superseded", but the implementation leaves it under `configs/stage1/` rather than moving it to an archive or converting it to an explicit rejection fixture.

Impact:

This leaves a broken launch/config path in the non-archive Stage-1 config tree. Even if the intended active route is now `configs/stage1/detection_teacher_forcing/`, this file is not in `configs/archive/` and is not a test fixture, so future users or agents can reasonably treat it as runnable. That is a config migration/documentation mismatch and weakens the "active configs migrated, historical configs archived or rejection-tested" contract.

Fix direction:

Either migrate this smoke config to `pipeline.id: stage1_research_teacher_forcing`, `objective.id: research_teacher_forcing`, `sample_factory.target_sequence.*`, top-level `token_embeddings_adapter`, and `prompt.variant`, or move it under an explicit archive/historical root and update docs/catalog/search guidance so it is not presented as an active runnable config. If it is intended as a rejection example, move it into `tests/fixtures` and add a test that names that purpose.

Smallest verification:

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
ConfigLoader.load_materialized_training_config(
    "configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml"
)
PY
```

Expected after fix: either it loads as a migrated active config, or the path no longer exists under active `configs/stage1/`.

### P1: Stage-2 normalized hierarchy records `objective.id: null`

Evidence:

- The OpenSpec requires normalized identity in resolved configs, cache fingerprints, manifests, and provenance, including `objective.id` (`openspec/changes/refactor-sft-pipeline-hierarchy/specs/sft-pipeline-hierarchy/spec.md:207-229`).
- `src/config/schema.py:4617-4627` explicitly allows Stage-2 detection configs to omit top-level `objective`; `tests/test_training_runtime_sft_integration.py:212-227` asserts this behavior and `cfg.objective is None`.
- `src/sft.py:1520-1616` builds normalized hierarchy as `"objective": {"id": _get_section_value(objective_cfg, "id")}`. For Stage-2, `objective_cfg` is `None`, so the normalized hierarchy emits `objective.id: null`.
- Probe result for `configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml`: `_build_normalized_training_hierarchy_identity(training_config=cfg)["objective"]["id"]` is `None`, while `pipeline.id` is `stage2_rollout_correction`.

Impact:

The artifact/cache/provenance identity is incomplete for Stage-2. The internal Stage-2 objective is `stage2_rollout_correction.pipeline.objective[].name: residual_set_correction`, but that identity is not represented in the normalized hierarchy field that the OpenSpec says must carry `objective.id`. This weakens reproducibility and makes Stage-2 artifact identity depend on readers knowing to merge `training_hierarchy`, `pipeline_manifest`, and `stage2_policy_provenance` rather than reading one coherent normalized hierarchy.

Fix direction:

Decide and encode the Stage-2 representation. The least disruptive fix is probably to record the Stage-2 internal objective as a normalized Stage-2 objective identity, for example `objective.id: residual_set_correction` or `objective: {"stage2_internal": ["residual_set_correction"]}`, then update tests and docs accordingly. If `objective.id: null` is intentional, the OpenSpec must be changed to exempt Stage-2 and define which artifact field owns Stage-2 objective identity.

Smallest verification:

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
from src.sft import _build_normalized_training_hierarchy_identity
cfg = ConfigLoader.load_materialized_training_config(
    "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml"
)
print(_build_normalized_training_hierarchy_identity(training_config=cfg)["objective"])
PY
```

Expected after fix: the normalized Stage-2 hierarchy contains a non-null objective identity, or the OpenSpec explicitly says Stage-2 stores objective identity elsewhere.

## Non-Blocking Findings

### P2: `--cfg-only` still prints a `trainer_variant` key

Evidence:

- `src/sft.py:2732-2760` prints a cfg-only JSON summary that includes `"trainer_variant": str(getattr(custom_config, "trainer_variant", "") or "")`.
- For migrated detection configs, the runtime shim does not set `custom_config.trainer_variant`, so this appears as an empty field rather than leaking `stage2_rollout_correction`.

Impact:

This is less serious than manifest leakage, because it is CLI stdout and currently empty for migrated detection configs. Still, it keeps the deprecated vocabulary in a user-visible summary immediately after the refactor removed `trainer_variant` from public config and artifact identity.

Fix direction:

Replace the cfg-only field with `pipeline: {"id": ...}` or `pipeline_id` only if the project intentionally accepts flattened runtime summaries. Prefer the nested `pipeline.id` shape for consistency.

Smallest verification:

Run `python scripts/run_sft.py --cfg-only ...` or the repo's equivalent config-only path on a Stage-2 smoke config and check that the stdout summary no longer contains `trainer_variant`.

## Evidence Reviewed

- OpenSpec change: `openspec/changes/refactor-sft-pipeline-hierarchy/proposal.md`, `design.md`, `tasks.md`, and targeted delta specs for `sft-pipeline-hierarchy`, `stage2-rollout-correction`, `packing-dataset`, and `encoded-training-cache`.
- Implementation plan: `docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md`.
- Active routing docs: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, `docs/ARTIFACTS.md`.
- Core implementation: `src/config/schema.py`, `src/config/loader.py`, `src/sft.py`, `src/detection/runtime.py`, `src/training/pipeline_registry.py`, `src/training_runtime/stage2_projection.py`, `src/bootstrap/pipeline_manifest.py`, `src/bootstrap/stage2_policy_provenance.py`, `src/bootstrap/experiment_manifest.py`, `src/utils/run_manifest.py`.
- Active configs: `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`, `configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml`, `configs/stage2/rollout_correction/base.yaml`, and Stage-2 smoke/pilot leaves.
- Tests: `tests/test_training_pipeline_registry.py`, `tests/test_stage2_policy_provenance.py`, `tests/test_stage2_rollout_correction_profile_leaf_contract.py`, `tests/test_training_runtime_sft_integration.py`, `tests/test_run_manifest_files.py`, `tests/test_experiment_manifest_file.py`, `tests/test_encoded_sample_cache_runtime_config.py`, `tests/test_stage1_static_packing_runtime_config.py`.
- Git state: the implementation is working-tree only; `git diff main...HEAD` is empty, while `git status --short --branch` shows the refactor as modified/untracked files in `/data/CoordExp/.worktrees/codebase-refactoring-program`.

## Verification Run

Passed:

```bash
openspec validate refactor-sft-pipeline-hierarchy --type change --strict
# Change 'refactor-sft-pipeline-hierarchy' is valid
```

```bash
python -m pytest tests/test_training_pipeline_registry.py tests/test_stage2_policy_provenance.py tests/test_stage2_rollout_correction_profile_leaf_contract.py -q
# 25 passed
```

```bash
python -m pytest \
  tests/test_training_runtime_sft_integration.py::test_detection_stage2_pipeline_id_builds_stage2_runtime_train_arguments \
  tests/test_training_runtime_sft_integration.py::test_loader_materializes_stage2_pipeline_id_as_detection_config \
  tests/test_run_manifest_files.py::test_write_run_manifest_files_resolved_config_uses_target_hierarchy \
  tests/test_experiment_manifest_file.py::test_write_experiment_manifest_file_captures_soft_and_hard_context \
  -q
# 4 passed
```

```bash
git diff --check
# passed with no output
```

```bash
python -m repo_lifecycle.report_lifecycle_registry
# completed; report includes the expected placeholder analysis-surface info and one owner-review warning for compact_full public refs
```

Additional probes:

```bash
python - <<'PY'
from pathlib import Path
import yaml
from src.training.pipeline_registry import TrainingPipelineRegistry
for path in [
    "configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml",
    "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml",
]:
    payload = yaml.safe_load(Path(path).read_text())
    try:
        TrainingPipelineRegistry().resolve(payload)
    except Exception as exc:
        print(path)
        print(type(exc).__name__, str(exc))
PY
```

Observed: both active YAMLs are rejected by `TrainingPipelineRegistry` with `Unknown top-level config domains`.

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
ConfigLoader.load_materialized_training_config(
    "configs/stage1/teacher_forcing/smoke/compact_full_hard_sft_tiny.yaml"
)
PY
```

Observed: fails with `ValueError flat token_rows is retired; use top-level token_embeddings_adapter`.

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
from src.sft import _build_normalized_training_hierarchy_identity
from src.bootstrap.stage2_policy_provenance import build_stage2_policy_provenance
cfg = ConfigLoader.load_materialized_training_config(
    "configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml"
)
print(_build_normalized_training_hierarchy_identity(training_config=cfg)["objective"])
print(build_stage2_policy_provenance(cfg)["pipeline"])
PY
```

Observed: normalized hierarchy objective is `{"id": None}` while Stage-2 policy provenance records `{"id": "stage2_rollout_correction"}`.

Not run:

- No expensive training, inference, rollout, or eval jobs, per request.
- I did not rerun the previously reported 223/383-test suites; the blockers above are contract/coverage gaps not broad pass/fail issues.

## Residual Risks

- I did not audit every changed test fixture or every artifact payload end to end. The manifest checks above are narrow and focused on the reported risk areas.
- I did not validate production-scale image/geometry alignment or bbox metric behavior with real training/rollout artifacts, because the request explicitly forbids expensive jobs.
- The implementation is entirely unstaged working-tree state in this worktree. Any human approval should account for the fact that `main...HEAD` does not represent the reviewed diff.
- Some legacy terms in `src/sft.py`, `src/training_runtime/plan.py`, and `src/bootstrap/*` are internal compatibility vocabulary. I did not mark all of them as findings because the strongest public leaks are covered above.

## Recommended Next Step

Do not approve yet. Ask the implementer to address the three P1 blockers, then rerun:

```bash
openspec validate refactor-sft-pipeline-hierarchy --type change --strict
python -m pytest tests/test_training_pipeline_registry.py tests/test_training_runtime_sft_integration.py tests/test_stage2_policy_provenance.py tests/test_run_manifest_files.py tests/test_experiment_manifest_file.py tests/test_stage2_rollout_correction_profile_leaf_contract.py -q
git diff --check
```

Also add two missing regression tests before re-review:

- a test proving the documented resolver can resolve real active migrated Stage-1 and Stage-2 configs, or a docs test proving the registry is not documented as the live resolver;
- a test proving Stage-2 normalized training hierarchy contains a non-null Stage-2 objective identity, or a strict spec/docs update that defines why `objective.id: null` is the intended Stage-2 representation.
