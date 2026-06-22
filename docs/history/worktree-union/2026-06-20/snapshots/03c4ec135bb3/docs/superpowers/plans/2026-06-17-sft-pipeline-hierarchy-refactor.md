# SFT Pipeline Hierarchy Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the OpenSpec change `refactor-sft-pipeline-hierarchy` so active CoordExp training configs use explicit `pipeline.id`, `sample_factory.target_sequence`, `standard_ce`, and `research_teacher_forcing` identities without starting code work before user approval.

**Architecture:** Add a pipeline registry as the public selection seam, keep `detection_template.id` top-level, move sequence-materialization controls into `sample_factory.target_sequence`, and keep Stage-2 rollout correction separate from Stage-1 SFT while preserving its internal `stage2_rollout_correction.pipeline.objective[]` namespace. Implement the migration test-first, then update active configs/docs/provenance once behavior is proven.

**Tech Stack:** Python, pytest, YAML config schema, OpenSpec, CoordExp Stage-1 SFT data builders, teacher-forcing objectives, Stage-2 rollout-correction trainer, static packing and encoded-sample cache fingerprints.

---

## Source Of Truth

- OpenSpec change: `openspec/changes/refactor-sft-pipeline-hierarchy/`
- Refactor charter: `docs/architecture/proposals/2026-06-17-refactoring-program/REFACTORING_PROGRAM_CHARTER.md`
- Lifecycle registry: `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`
- Main checkout reference boundary: `/data/CoordExp`
- Stable training router: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`
- Current config schema: `src/config/schema.py`
- Current shadow resolver: `src/training/surfaces.py`
- Current pipeline descriptors: `src/training/pipelines/`
- Standard SFT data path: `src/sft.py`, `src/datasets/builders/jsonlines.py`, `src/datasets/dense_caption.py`
- Template owner: `src/detection/template_contracts.py`, `src/detection/template.py`
- Prompt variant owner: `src/config/prompt_variants.py`
- Config inheritance/materialization owner: `src/config/loader.py`
- Packing/cache owners: `src/detection/packing.py`, `src/datasets/encoded_sample_cache.py`, `src/datasets/wrappers/packed_caption.py`
- Research teacher-forcing owners: `src/detection/teacher_forcing/`, `src/trainers/teacher_forcing/`, `src/training/objectives/teacher_forcing.py`
- Stage-2 owners: `src/trainers/stage2_rollout_correction.py`, `src/trainers/stage2_rollout_correction_impl.py`, `src/trainers/stage2_rollout_runtime.py`

## Approval Gate

Implementation is not approved by this plan. Stop after review convergence with
state `ready for user approval`.

Before any code/config implementation:

1. Re-inspect `/data/CoordExp` main for active config families.
2. Ask the user to approve implementation explicitly.
3. Treat these decisions as locked unless the user explicitly reopens them:
   `teacher_forcing` is rejected for active migrated configs,
   `src/training/surfaces.py` is migrated/deleted immediately with no long-lived
   shim, research-TF packing is fail-fast in the first slice,
   `token_embeddings_adapter` is the high-level token adapter namespace, and
   active config names are semantic rather than numeric/versioned.

## File Structure

- Create `src/training/pipeline_registry.py`
  - Owns public `pipeline.id` resolution and maps ids to pipeline descriptors.
- Modify `src/training/surfaces.py`
  - Migrate active imports away from it, then delete or rename it in this slice.
  - Do not keep a long-lived compatibility shim.
- Modify `src/training/pipelines/base.py`
  - Ensure pipeline descriptors expose public pipeline ids and stable metadata.
- Modify `src/training/pipelines/stage1_json_ce.py`
  - Public pipeline identity is `stage1_standard_sft`; `standard_ce` remains objective schema/resolution vocabulary only.
- Modify `src/training/pipelines/stage1_compact_trie_ce.py`
  - Keep as research/internal descriptor unless promoted by this migration.
- Modify `src/training/pipelines/stage2_rollout_correction.py`
  - Ensure descriptor identity is `stage2_rollout_correction`.
- Modify `src/config/schema.py`
  - Adds typed `pipeline`, `sample_factory`, target sequence, `prompt.variant`, `standard_ce`, and `research_teacher_forcing` config support and fail-fast rejection for old selector paths.
  - Adds top-level `token_embeddings_adapter` and rejects flat `token_rows`
    plus `custom.token_embeddings_adapter` after migration.
- Modify `src/config/loader.py`
  - Preserve strict unknown-key behavior and materialized resolved-config identity.
- Modify `src/sft.py`
  - Consume normalized pipeline/sample/objective config without interpreting old selector paths.
- Modify `src/datasets/builders/jsonlines.py`
  - Consume target-sequence controls from normalized config.
- Modify `src/datasets/dense_caption.py`
  - Consume object ordering and field order from normalized target-sequence controls.
- Modify `src/config/prompt_variants.py`
  - Keep the shared prompt registry as the source for Stage-1 `prompt.variant` and Stage-2 rollout prompt variants.
- Modify `src/detection/packing.py`
  - Include normalized hierarchy identity in packing fingerprints.
- Modify `src/datasets/encoded_sample_cache.py`
  - Include normalized hierarchy identity in encoded-sample cache fingerprints.
- Modify `src/training/objectives/teacher_forcing.py`
  - Accept normalized `research_teacher_forcing` config identity.
- Modify `src/training/objectives/runner.py`
  - Resolve `standard_ce` and `research_teacher_forcing` to implementation modules.
- Modify `src/trainers/teacher_forcing/`
  - Preserve fine-grained token tracing and role/value/span metadata.
- Modify `src/trainers/stage2_rollout_correction.py`
  - Select Stage-2 from `pipeline.id` while preserving internal correction pipeline.
- Modify `src/bootstrap/stage2_policy_provenance.py`
  - Build Stage-2 policy provenance from top-level `pipeline.id` and record it
    alongside existing Stage-2 policy fields.
  - Do not emit a compatibility `trainer_variant` selector field after
    migration.
- Modify active configs under `configs/stage1/` and `configs/stage2/`
  - Migrate only active repo-owned configs selected from `/data/CoordExp` main.
- Modify docs after behavior is proven:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/data/PACKING.md`
  - `docs/training/README.md`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/ARTIFACTS.md`

## Review Convergence

- Mode: docs/spec/plan until user approval.
- Allowed mutation before approval: OpenSpec and roadmap documentation only.
- Required review lanes:
  - OpenSpec governance and compatibility scope.
  - Config/schema migration and active-main boundary.
  - Packing/cache/provenance correctness.
  - Stage-2 separation and rollout/self-trajectory semantics.
  - Implementation plan sufficiency.
- Convergence state before implementation must be `ready for user approval`.

### Review Results

Five read-only reviewer lanes completed:

- OpenSpec governance and compatibility scope.
- Config/schema migration and active-main boundary.
- Packing/cache/provenance correctness.
- Stage-2 separation and rollout/self-trajectory semantics.
- Implementation roadmap sufficiency.

One P0 and multiple P1/P2 findings from the external review packet were accepted
and revised into this plan and the OpenSpec artifacts:

- The catalog-canonical flat Stage-1 research-TF config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` is now
  included in the first migration slice.
- OpenSpec `MODIFIED Requirement` headers must match exact base requirement
  names for archive safety.
- Durable specs no longer make `/data/CoordExp` a normative stable contract;
  the local main checkout remains an operational pre-implementation reference.
- `standard_ce` is mandatory for active Standard SFT configs.
- `objective.id: teacher_forcing` is rejected for active migrated configs; it is
  not an alias path.
- Active `custom.detection_template_id`, `custom.object_ordering`,
  `custom.object_field_order`, `custom.extra.prompt_variant`, and duplicate
  parser-policy authoring are explicitly covered by migration/rejection tests.
- `custom.detection_sequence_format` is subsumed by `detection_template.id` plus
  `sample_factory.id`.
- Stage-1 prompt identity uses `prompt.variant`; Stage-2 rollout prompt identity
  remains under `rollout_matching.prompt_variant` and
  `rollout_matching.eval_prompt_variant`.
- Historical `prompt.prompt_variant_enabled: true` maps to
  `prompt.variant: coco_80`; `false` maps to no prompt variant.
- Cache/provenance identity is additive to the current implementation's existing
  discriminator key sets; this change does not add new image-root/view-store
  discriminators unless already present.
- Stage-2 negative-contract coverage includes retired selectors, removed
  strategy namespaces, missing `rollout_matching`, non-residual objectives, and
  the old shadow-resolver `teacher_forcing` policy.
- Stage-2 policy provenance must be generated from
  `pipeline.id: stage2_rollout_correction` and must not emit a compatibility
  `trainer_variant` selector field.
- `src/training/surfaces.py` is immediate migration/delete scope, not a
  long-term shim.
- `token_embeddings_adapter` is the high-level token adapter namespace; flat
  `token_rows` and `custom.token_embeddings_adapter` are rejected after
  migration.
- Research teacher-forcing packing must preserve the current exact-packing guard
  and packed-sidecar rejection until exact remapping is proven.
- Recursive-detection / ET-RMP handles remain preserved comparator lineage unless
  the user explicitly approves retirement.

Remaining approval-time refresh:

- Re-inspect `/data/CoordExp` main for any active config additions since this
  plan was refined, then request explicit implementation approval.

Implementation split points after approval:

1. Schema/loader normalization lands first and owns the public hierarchy.
2. Pipeline registry migration may proceed once schema tests identify
   `pipeline.id` as the selector.
3. Sample-factory data path and prompt identity migration can proceed in
   parallel after schema normalization.
4. Objective-id migration and Stage-2 selector migration are separate lanes; do
   not share public `objective.id` semantics between Stage-1 and Stage-2.
5. Cache/provenance migration starts after normalized identity payload tests
   exist.
6. Active config/docs migration runs last, from a resolved active-config
   inventory artifact whose filenames are standardized by semantic axes rather
   than numeric/version-like labels.

## Task 1: Pre-Implementation Approval Packet

**Files:**
- Modify: `openspec/changes/refactor-sft-pipeline-hierarchy/proposal.md`
- Modify: `openspec/changes/refactor-sft-pipeline-hierarchy/design.md`
- Modify: `openspec/changes/refactor-sft-pipeline-hierarchy/tasks.md`
- Modify: `docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md`

- [ ] **Step 1: Validate OpenSpec artifacts**

Run:

```bash
openspec validate refactor-sft-pipeline-hierarchy --type change --strict
```

Expected: validation passes.

- [ ] **Step 2: Run docs-plan hygiene**

Run:

```bash
python - <<'PY'
import yaml
from pathlib import Path
for path in ["docs/catalog.yaml", "progress/index.yaml"]:
    yaml.safe_load(Path(path).read_text())
    print(f"{path}: ok")
PY
git diff --check
```

Expected: YAML parse messages and no diff-check errors.

- [ ] **Step 3: Stop for user approval**

Report state as `ready for user approval`. Do not edit implementation code.

## Task 2: Schema Rejection Tests

**Files:**
- Create or modify: `tests/test_training_pipeline_hierarchy_schema.py`
- Modify: `src/config/schema.py`

- [ ] **Step 1: Add failing tests for selector rejection**

Add tests with this shape:

```python
import pytest

from src.config.schema import DetectionTrainingConfig


def _base_config():
    return {
        "model": {"model": "dummy"},
        "training": {"output_dir": "tmp/out"},
        "data": {"train_jsonl": "train.jsonl", "val_jsonl": "val.jsonl"},
        "prompt": {},
        "detection_template": {"id": "compact_object_box_closed"},
        "token_embeddings_adapter": {"enabled": True},
        "objective": {"id": "standard_ce"},
        "sample_factory": {
            "id": "detection_sequence",
            "target_sequence": {
                "task_family": "detection",
                "object_ordering": "sorted",
                "object_field_order": "desc_first",
                "bbox_format": "xyxy",
                "coordinate_surface": "coord_token",
                "strict_parse": True,
            },
        },
        "pipeline": {"id": "stage1_standard_sft"},
    }


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"pipeline_id": "stage1_standard_sft"}, "pipeline.id"),
        ({"surface": {"id": "stage1_standard_sft"}}, "pipeline.id"),
        ({"custom": {"trainer_variant": "stage2_rollout_correction"}}, "custom.trainer_variant"),
    ],
)
def test_old_pipeline_selectors_fail_fast(patch, message):
    cfg = _base_config()
    cfg.update(patch)
    with pytest.raises((TypeError, ValueError), match=message):
        DetectionTrainingConfig.from_mapping(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_training_pipeline_hierarchy_schema.py::test_old_pipeline_selectors_fail_fast -q
```

Expected: tests fail because schema support/rejection is not implemented yet.

- [ ] **Step 3: Implement minimal schema validation**

Add typed schema support in `src/config/schema.py` for:

```python
@dataclass(frozen=True)
class PipelineConfig:
    id: Literal[
        "stage1_standard_sft",
        "stage1_research_teacher_forcing",
        "stage2_rollout_correction",
    ]
```

Reject `pipeline_id`, `surface`, and `custom.trainer_variant` in active
target-hierarchy configs with dotted-path errors.

- [ ] **Step 4: Run selector rejection tests**

Run:

```bash
python -m pytest tests/test_training_pipeline_hierarchy_schema.py::test_old_pipeline_selectors_fail_fast -q
```

Expected: tests pass.

## Task 3: Sample Factory And Sequence Controls

**Files:**
- Modify: `tests/test_training_pipeline_hierarchy_schema.py`
- Modify: `src/config/schema.py`
- Modify: `src/datasets/builders/jsonlines.py`
- Modify: `src/datasets/dense_caption.py`

- [ ] **Step 1: Add failing tests for target sequence authoring**

Add tests proving `sample_factory.target_sequence.object_field_order` is
accepted, `sample_factory.target_sequence.template_id` is rejected, and
dual-authoring with old `custom.*` paths fails. Cover:

- `custom.object_ordering`
- `custom.object_field_order`
- `custom.detection_template_id`
- `sample_factory.target_sequence.template_id`
- matching old/new duplicate values
- duplicate or conflicting parser-policy paths involving
  `sample_factory.target_sequence.strict_parse`,
  `detection_template.strict_parse`, and evaluation parser strictness

Run:

```bash
python -m pytest tests/test_training_pipeline_hierarchy_schema.py -q
```

Expected: new tests fail before schema implementation.

- [ ] **Step 2: Implement typed target sequence config**

Add dataclasses for:

```python
@dataclass(frozen=True)
class TargetSequenceConfig:
    task_family: Literal["detection"]
    object_ordering: Literal["sorted", "random", "random_permutation"]
    object_field_order: Literal["desc_first", "geometry_first"]
    bbox_format: Literal["xyxy"]
    coordinate_surface: Literal["coord_token"]
    strict_parse: bool


@dataclass(frozen=True)
class SampleFactoryConfig:
    id: Literal["detection_sequence"]
    target_sequence: TargetSequenceConfig
```

Reject `sample_factory.target_sequence.template_id`.

Also reject `custom.detection_template_id` for active target-hierarchy configs
and require top-level `detection_template.id`.
Normalize `sample_factory.target_sequence.strict_parse` into the effective
template/evaluation parser policy and reject active configs that author a second
strict parser source.

- [ ] **Step 3: Route data builders through normalized config**

Update Stage-1 data builders so object ordering and field order come from the
normalized target-sequence object instead of direct `custom.*` access.

- [ ] **Step 4: Add Stage-1 prompt variant migration tests**

Add tests proving active Stage-1 training prompt variants are authored as
`prompt.variant`, resolve through `src/config/prompt_variants.py`, and reject
`custom.extra.prompt_variant` in active target-hierarchy configs. Add a
canonical-flat migration test proving `prompt.prompt_variant_enabled: true`
maps to `prompt.variant: coco_80`, `false` maps to no prompt variant, and
active migrated configs reject `prompt.prompt_variant_enabled`.

- [ ] **Step 5: Add token adapter migration tests**

Add tests proving compact token embedding adaptation is authored through
top-level `token_embeddings_adapter`, and active migrated configs reject flat
`token_rows` plus `custom.token_embeddings_adapter`.

- [ ] **Step 6: Run schema and data-path tests**

Run:

```bash
python -m pytest tests/test_training_pipeline_hierarchy_schema.py tests/test_dense_caption_prompt_override.py tests/test_prompt_variants.py tests/test_detection_training_config_contract.py -q
```

Expected: tests pass.

## Task 4: Objective Identity Migration

**Files:**
- Create or modify: `tests/test_training_objective_public_ids.py`
- Modify: `src/config/schema.py`
- Modify: `src/training/objectives/runner.py`
- Modify: `src/training/objectives/teacher_forcing.py`
- Modify: `src/trainers/teacher_forcing/`

- [ ] **Step 1: Add failing tests for public objective ids**

Add tests proving `standard_ce` resolves to ordinary token CE implementation
and `research_teacher_forcing` resolves to fine-grained teacher-forcing
metadata.

Run:

```bash
python -m pytest tests/test_training_objective_public_ids.py -q
```

Expected: tests fail before objective-id migration.

- [ ] **Step 2: Implement `standard_ce` public id**

Map `objective.id: standard_ce` to the existing token-level CE implementation
while preserving public resolved-config identity as `standard_ce`.

- [ ] **Step 3: Implement `research_teacher_forcing` public id**

Map `objective.id: research_teacher_forcing` to the existing fine-grained
teacher-forcing objective behavior.

- [ ] **Step 4: Reject legacy `teacher_forcing` authoring**

Reject `objective.id: teacher_forcing` for active migrated configs and add the
matching negative test. Historical/archive files may keep the old value as
evidence, but active config loading must point authors to
`objective.id: research_teacher_forcing`.

- [ ] **Step 5: Run objective tests**

Run:

```bash
python -m pytest tests/test_training_objective_public_ids.py tests/test_teacher_forcing_token_ce.py tests/test_detection_training_config_contract.py -q
```

Expected: tests pass.

## Task 5: Pipeline Registry Migration

**Files:**
- Create: `src/training/pipeline_registry.py`
- Modify: `src/training/surfaces.py`
- Modify: `src/training/pipelines/base.py`
- Modify: `tests/test_training_surface_resolver.py`
- Create or modify: `tests/test_training_pipeline_registry.py`

- [ ] **Step 1: Add failing pipeline registry tests**

Add tests proving the registry resolves the three public pipeline ids and no
public test authors `surface.id`. Add a Stage-2 assertion that the old shadow
resolver policy `objective_id == "teacher_forcing"` is not a public Stage-2
objective and is removed from public behavior rather than bridged as a
compatibility path.

Run:

```bash
python -m pytest tests/test_training_pipeline_registry.py tests/test_training_surface_resolver.py -q
```

Expected: tests fail until the registry exists or current surface tests are
migrated.

- [ ] **Step 2: Create pipeline registry**

Create `src/training/pipeline_registry.py` with a closed mapping for:

```python
PIPELINE_IDS = (
    "stage1_standard_sft",
    "stage1_research_teacher_forcing",
    "stage2_rollout_correction",
)
```

- [ ] **Step 3: Migrate imports from surfaces to pipeline registry**

Move active resolver imports to the new module. Delete or rename
`src/training/surfaces.py` in this slice after consumers move; do not keep a
long-lived compatibility shim. Remove the old Stage-2 `teacher_forcing` policy
instead of bridging it as public behavior.

- [ ] **Step 4: Run registry tests**

Run:

```bash
python -m pytest tests/test_training_pipeline_registry.py tests/test_training_surface_resolver.py tests/test_objective_profile_resolution.py -q
```

Expected: tests pass or intentionally renamed tests pass with equivalent
coverage.

## Task 6: Stage-2 Selector Migration

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/bootstrap/stage2_policy_provenance.py`
- Modify: `src/trainers/stage2_rollout_correction.py`
- Modify: `tests/test_stage2_rollout_correction_contract.py`
- Modify: `tests/test_stage2_rollout_runtime.py`
- Modify: `tests/test_legacy_surface_absence.py`

- [ ] **Step 1: Add failing Stage-2 selector tests**

Add tests proving `pipeline.id: stage2_rollout_correction` selects Stage-2 and
`custom.trainer_variant: stage2_rollout_correction` fails. Cover the full
negative contract:

- `custom.trainer_variant: stage2_two_channel`
- `custom.trainer_variant: stage2_ab_training`
- top-level `stage2_ab`
- `rollout_matching_sft`
- `stage2_rollout_aligned`
- `stage2_rollout_runtime`
- `rollout_matching.pipeline`
- `stage2_rollout_correction.schedule`
- `stage2_rollout_correction.b_ratio`
- `stage2_rollout_correction.channel_b`
- `stage2_rollout_correction.pipeline.objective[].channels`
- non-empty `stage2_rollout_correction.pipeline.diagnostics`
- removed correction keys such as `pseudo_positive`
- missing top-level `rollout_matching` under `pipeline.id:
  stage2_rollout_correction`
- non-residual objective modules such as `token_ce`, `hard_sft`,
  `stage2_trie_ce`, and geometry auxiliaries

Run:

```bash
python -m pytest tests/test_stage2_rollout_correction_contract.py tests/test_legacy_surface_absence.py -q
```

Expected: new tests fail until selector migration is implemented.

- [ ] **Step 2: Implement Stage-2 top-level pipeline selection**

Update config/runtime planning so top-level `pipeline.id:
stage2_rollout_correction` activates Stage-2 only when
`stage2_rollout_correction.pipeline` is present.

- [ ] **Step 3: Preserve Stage-2 internal objective namespace**

Keep `stage2_rollout_correction.pipeline.objective[]` as the residual-set
correction objective list. Do not require a top-level `objective.id` for
Stage-2, and reject `objective.id: standard_ce` or
`objective.id: research_teacher_forcing` as the active Stage-2 correction
objective.

- [ ] **Step 4: Add legacy Stage-2 retirement discovery gate**

Run a targeted search that classifies hits as `active reject`,
`historical/archive allowed`, `research/comparator preserve`, or
`remove/update`:

```bash
rg -n "stage2_ab\\b|stage2_two_channel|stage2_ab_training|rollout_matching_sft|stage2_rollout_aligned|stage2_rollout_runtime|rollout_matching\\.pipeline|legacy_hungarian_mask_iou|loss_duplicate_burst_unlikelihood|adjacent_repulsion|rollout_decode_policy|coord_decode_mode|stage2_rollout_correction\\.(schedule|b_ratio|channel_b)|pipeline\\.diagnostics|\\.channels|correction\\.pseudo_positive" configs docs openspec/specs src tests --glob '!configs/archive/**' --glob '!docs/history/**' --glob '!progress/**'
```

Expected: active configs/docs/specs/tests/code either reject or no longer
recommend retired surfaces; archive/history/progress references are allowed as
evidence.

- [ ] **Step 5: Run Stage-2 targeted tests**

Run:

```bash
python -m pytest tests/test_stage2_rollout_correction_contract.py tests/test_stage2_rollout_runtime.py tests/test_legacy_surface_absence.py tests/test_stage2_assignment_greedy_iou.py tests/test_stage2_duplicate_filter.py tests/test_stage2_supervision_planning_smoke.py -q
```

Expected: tests pass.

## Task 7: Cache, Packing, And Provenance Identity

**Files:**
- Modify: `src/detection/packing.py`
- Modify: `src/datasets/encoded_sample_cache.py`
- Modify: `src/datasets/wrappers/packed_caption.py`
- Modify: `src/bootstrap/experiment_manifest.py`
- Modify: `src/bootstrap/stage2_policy_provenance.py`
- Modify: `tests/test_packing_cache_fingerprints.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Add failing fingerprint tests**

Add tests proving cache fingerprints differ when only `pipeline.id`,
`objective.id`, object ordering, object field order, bbox format, coordinate
surface, strict parse, `detection_template.id`, prompt variant, prompt hash,
tokenizer identity, chat-template identity, or effective packing length changes.
Cover these axes for both encoded-sample cache and static-packing cache wherever
the axis can affect the relevant identity.
Also add regression tests proving existing discriminators still affect both
encoded-sample and static-packing fingerprints:

- dataset file path/content identity
- train/eval split and sample-limit identity
- tokenizer/model identity
- system prompt text and prompt hash
- packing min-fill/drop-last/allow-single-long/shuffle behavior
- offline image-pixel budget

Do not add a new image-root or view-store discriminator in this change unless
it is already part of the current key set under test.

Run:

```bash
python -m pytest tests/test_packing_cache_fingerprints.py tests/test_stage1_static_packing_runtime_config.py -q
```

Expected: new tests fail before fingerprint identity includes the normalized
hierarchy.

- [ ] **Step 2: Add normalized identity payload helper**

Create or extend a small helper that builds the normalized training identity
payload from resolved config. Include all fields required by the OpenSpec.
The payload must include:

- `pipeline.id`
- `objective.id`
- `detection_template.id`
- `sample_factory.id`
- `sample_factory.target_sequence.*`
- resolved prompt variant
- prompt hash
- tokenizer identity
- chat-template identity
- effective packing length

The helper must be additive: do not replace existing cache discriminators for
dataset source, image root/view metadata, tokenizer/model, prompts,
preprocessing, sample limits, packing knobs, dataloader shuffle, or offline
image budget.

- [ ] **Step 3: Route fingerprints and manifests through identity payload**

Use the helper for encoded-sample cache, static packing, run manifests, and
Stage-2 policy provenance. Add tests for resolved-config serialization,
`experiment_manifest.json`, run metadata, and Stage-2 policy provenance, not
only fingerprint diffs.
For Stage-2, add a pipeline-id-only fixture without `custom.trainer_variant` and
assert `effective_runtime.json`, `pipeline_manifest.json`, `run_metadata.json`,
and `experiment_manifest.json` carry Stage-2 policy provenance with assignment
strategy/effective threshold, duplicate-filter strategy/thresholds, object
ordering policy/strategy id, sample object ordering, rollout template family,
invalid-rollout policy, and fallback loss weight. Assert those artifacts do not
emit a compatibility `trainer_variant` selector field.

- [ ] **Step 4: Run cache/provenance tests**

Run:

```bash
python -m pytest tests/test_packing_cache_fingerprints.py tests/test_stage1_static_packing_runtime_config.py tests/test_training_resolved_config_identity.py tests/test_stage2_policy_provenance.py tests/test_training_architecture_tiny_smoke.py -q
```

Expected: tests pass.

## Task 8: Research Teacher-Forcing Packing Guard

**Files:**
- Modify: `src/config/schema.py`
- Modify: `src/training/objectives/teacher_forcing.py`
- Modify: `tests/test_detection_training_config_contract.py`
- Modify: `tests/test_stage1_static_packing_runtime_config.py`

- [ ] **Step 1: Add fail-fast tests before remapping exists**

Add a test proving `pipeline.id: stage1_research_teacher_forcing` with packing
enabled fails until exact atom-position remapping is implemented.
Preserve the existing target-IR packing guard semantics while migrating names:
`objective.target_ir.exact_packing_mapping.enabled`,
`validation.validate_span_alignment`, and packed target-IR sidecar rejection
remain active until exact remapping support lands.

Run:

```bash
python -m pytest tests/test_stage1_static_packing_runtime_config.py::test_research_teacher_forcing_rejects_packing_until_remap -q
```

Expected: test fails until guard is implemented.

- [ ] **Step 2: Implement fail-fast guard**

Reject packed research teacher forcing with an error naming exact atom-position
remapping.
If exact remapping is implemented in a later approved slice, require a
multi-segment packed fixture with distinct examples/images and unequal
prompt/train lengths; assert `encoded_len`, `cu_seq_lens`, cumulative atom
position offsets, valid sets, force/weight policies, sample ids, image-grid
slices, and packed-vs-unpacked aggregated loss all match the contract.

- [ ] **Step 3: Preserve future implementation hook**

Keep the guard local and testable so a later implementation can replace it with
validated remapping support.

- [ ] **Step 4: Run packing guard tests**

Run:

```bash
python -m pytest tests/test_stage1_static_packing_runtime_config.py tests/test_detection_training_config_contract.py -q
```

Expected: tests pass.

## Task 9: Active Config Migration

**Files:**
- Modify: active configs selected from `/data/CoordExp` main under `configs/stage1/`
- Modify: active configs selected from `/data/CoordExp` main under `configs/stage2/`
- Modify: config contract tests that parse those files

- [ ] **Step 1: Re-inspect main for active configs**

Run the cheap git boundary checks and a broad hint search:

```bash
git -C /data/CoordExp rev-parse --short HEAD
git -C /data/CoordExp status --short
rg -n "custom\\.trainer_variant|custom\\.object_field_order|custom\\.object_ordering|custom\\.detection_template_id|objective\\.id|stage2_rollout_correction" /data/CoordExp/configs /data/CoordExp/docs /data/CoordExp/openspec/specs --glob '!**/archive/**' --glob '!docs/history/**'
```

Then create a resolved active-config inventory artifact under the worktree's
temporary planning area. It must record:

- `/data/CoordExp` commit SHA and dirty status
- exact active config paths selected from docs/catalog routing and active
  families
- explicit inclusion status for
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`
- explicit inclusion status for active Standard SFT sorted/random ordering
  object-closure and bbox-closure launch-prep leaves from main
- inherited effective values after `extends`
- effective pipeline/objective/template/prompt/object ordering/object field
  order/bbox/strict-parse/packing/trainer-variant/rollout settings
- parse-test coverage per config

Use `ConfigLoader.load_yaml_with_extends()` or
`ConfigLoader.load_materialized_training_config()` rather than relying on regex
matches alone.

Expected: record the main commit and resolved active config matrix before
editing worktree configs.

- [ ] **Step 2: Migrate active Standard SFT configs**

Add `pipeline.id: stage1_standard_sft`, `objective.id: standard_ce`, and
`sample_factory.target_sequence` controls. Migrate
`custom.detection_template_id` to top-level `detection_template.id`, migrate
`custom.extra.prompt_variant` to `prompt.variant`, and preserve object-ref and
box closure token-row intent for `compact_object_box_closed`.

- [ ] **Step 3: Migrate active research teacher-forcing configs**

Add `pipeline.id: stage1_research_teacher_forcing` and
`objective.id: research_teacher_forcing`. Reject `objective.id:
teacher_forcing` for active migrated configs.
Preserve recursive-detection / ET-RMP comparator handles as runnable or
inspectable comparator lineage unless the user explicitly retires them.

- [ ] **Step 4: Migrate active Stage-2 configs**

Add `pipeline.id: stage2_rollout_correction` and remove
`custom.trainer_variant` selector authoring. Preserve `rollout_matching.*`
runtime settings and verify missing `rollout_matching` fails fast.

- [ ] **Step 5: Run active config parse tests**

Run:

```bash
python -m pytest tests/test_detection_training_config_contract.py tests/test_stage2_rollout_correction_contract.py tests/test_training_config_strict_unknown_keys.py -q
```

Expected: tests pass.

## Task 10: Docs And Final Verification

**Files:**
- Modify: `docs/AGENT_INDEX.md`
- Modify: `docs/catalog.yaml`
- Modify: `docs/data/PACKING.md`
- Modify: `docs/training/README.md`
- Modify: `docs/training/STAGE1_OBJECTIVE.md`
- Modify: `docs/training/STAGE2_RUNBOOK.md`
- Modify: `docs/ARTIFACTS.md`
- Modify: `docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`

- [ ] **Step 1: Update docs after tests pass**

Update routing docs to name `pipeline.id`, `sample_factory.target_sequence`,
`prompt.variant`, `standard_ce`, `research_teacher_forcing`, and the
pipeline-selection owner.

- [ ] **Step 2: Run OpenSpec validation**

Run:

```bash
openspec validate refactor-sft-pipeline-hierarchy --type change --strict
```

Expected: validation passes.

- [ ] **Step 3: Run lifecycle registry checks**

Run:

```bash
python -m repo_lifecycle.report_lifecycle_registry
python -m pytest tests/test_lifecycle_registry_report.py -q
```

Expected: registry report completes and tests pass. Existing intentional
warnings must be reported if still present.

- [ ] **Step 4: Run targeted training hierarchy tests**

Run:

```bash
python -m pytest \
  tests/test_training_pipeline_hierarchy_schema.py \
  tests/test_training_pipeline_registry.py \
  tests/test_training_objective_public_ids.py \
  tests/test_detection_training_config_contract.py \
  tests/test_stage2_rollout_correction_contract.py \
  tests/test_packing_cache_fingerprints.py \
  tests/test_stage1_static_packing_runtime_config.py \
  -q
```

Expected: tests pass.

- [ ] **Step 5: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 6: Report residual risks**

Report any skipped hardware-heavy, production training, or full-pipeline smoke
checks. Do not claim production readiness unless those checks are explicitly
run later.
