## Why

CoordExp's active Stage-1 SFT, research teacher-forcing, and Stage-2 rollout
correction paths currently share too much vocabulary through `custom.*`,
`surface`, and generic `teacher_forcing` names. This makes active training
routes harder for agents to navigate, weakens cache/provenance identity, and
lets temporary experiment knobs look like permanent infrastructure.

This change defines the stable config and provenance hierarchy for the SFT
pipeline refactor before implementation. It is intentionally breaking for
active repo-owned configs: active routes should migrate to the new hierarchy
rather than accumulate compatibility aliases.

## What Changes

- **BREAKING**: active repo-owned training configs select training family with
  top-level `pipeline.id`; `custom.trainer_variant` is rejected as an active
  selector rather than accepted as a compatibility reader.
- **BREAKING**: the shadow `surface.id` vocabulary is not public config
  language. The implementation registry should move from
  `src/training/surfaces.py` to `src/training/pipeline_registry.py` in the
  first implementation slice, without a long-lived compatibility shim.
- **BREAKING**: sequence-materialization controls move out of generic
  `custom.*` authoring and into `sample_factory.target_sequence`.
- **BREAKING**: active configs using old `objective.id: teacher_forcing` fail
  fast after migration. Historical/archive fixtures may keep the old value only
  as evidence or explicit rejection-test input.
- **BREAKING**: compact token embedding adaptation moves to a high-level
  `token_embeddings_adapter` namespace. Flat `token_rows` and
  `custom.token_embeddings_adapter` authoring are deprecated by this program and
  rejected after migration.
- Keep `detection_template.id` as the stable template identity. Do not move
  template identity under `sample_factory.target_sequence` in this change.
- Define public pipeline ids:
  - `stage1_standard_sft`
  - `stage1_research_teacher_forcing`
  - `stage2_rollout_correction`
- Define `sample_factory.id: detection_sequence` for current object/bbox target
  sequence materialization without a broad rename away from `detection`.
- Define `objective.id: standard_ce` as the public Standard SFT objective id.
  `token_ce` remains implementation/metric vocabulary unless separately
  promoted by a later spec.
- Define `objective.id: research_teacher_forcing` for fine-grained research
  teacher forcing with token role/span/value tracing and internal weighted
  terms. The old `teacher_forcing` id is rejected for active migrated configs.
- Keep `stage2_rollout_correction.pipeline.objective[]` unchanged in this
  change except for documenting its relationship to top-level `pipeline.id`.
- Replace Stage-2 selector/provenance identity based on `trainer_variant` with
  top-level `pipeline.id`. Do not emit a compatibility `trainer_variant`
  provenance field from migrated active Stage-2 runs.
- Include the catalog-canonical flat Stage-1 research-TF config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` in the
  first migration slice alongside the active Standard SFT launch-prep leaves
  from `/data/CoordExp` main.
- Require resolved config, cache fingerprints, run metadata, and provenance to
  include normalized training identity: `pipeline.id`, `objective.id`,
  `detection_template.id`, `sample_factory.id`, target-sequence ordering and
  field order, bbox/coord surface, strict parsing mode, prompt variant, prompt
  hash, tokenizer/chat-template identity, and packing length. These fields are
  additive to the current implementation's existing cache/provenance
  discriminators. This change must not introduce a new image-root or
  view-store discriminator unless it is already part of the current key set.
- Require Stage-1 research teacher forcing to treat packing as a long-term
  first-class requirement. It may reject packing only until exact atom-position
  remapping is implemented and tested.
- Standardize active config filenames by semantic axes such as pipeline,
  template, ordering, prompt, and packing. Do not introduce ambiguous numeric
  or `version` fields as migration identity.
- Keep implementation planning, code-motion order, and review/self-audit steps
  in the associated Superpowers roadmap; this OpenSpec change owns stable
  config/schema/provenance/packing contracts only.

## Capabilities

### New Capabilities

- `sft-pipeline-hierarchy`: Defines the public pipeline selector, standard SFT
  and research teacher-forcing objective identities, sample factory hierarchy,
  provenance identity, and packing contract for the SFT pipeline refactor.

### Modified Capabilities

- `training-config-hierarchy`: active training config hierarchy becomes
  explicitly breaking for `custom.trainer_variant` selectors and moves
  sequence controls out of `custom.*` authoring.
- `stage1-detection-objectives`: Stage-1 Standard SFT and research
  teacher-forcing identities become explicit while ET-RMP remains preserved as
  research/comparator behavior.
- `stage2-rollout-correction`: Stage-2 selection moves to top-level
  `pipeline.id`, while the existing internal
  `stage2_rollout_correction.pipeline.objective[]` contract remains unchanged.
- `rollout-matching-sft`: rollout runtime settings remain supported under
  `rollout_matching.*`, but `custom.trainer_variant` is no longer the active
  Stage-2 selector.
- `teacher-forcing-unified-loss-registry`: distinguishes public
  `standard_ce`, public `research_teacher_forcing`, implementation
  `token_ce`, and Stage-2 residual-set correction loss terms.
- `object-field-ordering`: moves training/evaluation sequence authoring from
  `custom.object_field_order` and `custom.object_ordering` to
  `sample_factory.target_sequence.*` while preserving inference-specific
  ordering fields.
- `compact-template-field-order-ablation`: updates the active compact ablation
  contract to use top-level `detection_template.id` and
  `sample_factory.target_sequence.*`.
- `dataset-prompt-variants`: updates compact prompt hash/parity requirements to
  use normalized target-sequence identity and `prompt.variant` for Stage-1
  training/eval prompt generation.
- `encoded-training-cache`: requires encoded-sample cache eligibility and
  fingerprint identity to use the normalized hierarchy, including target
  sequence fields.
- `packing-dataset`: requires static packing cache identity and random-ordering
  semantics to use normalized target-sequence fields.
- `token_embeddings_adapter`: keeps compact token-row adaptation independent of
  normalized target-sequence field order.
- `stage2-ab-training`: updates the retired AB/two-channel pointer so current
  Stage-2 configs use `pipeline.id: stage2_rollout_correction`.

## Impact

- Affected config/schema surfaces during eventual implementation:
  - `src/config/schema.py`
  - `src/config/loader.py`
  - active Stage-1 config families under `configs/stage1/`
  - active Stage-2 config families under `configs/stage2/`
  - current launch-prep configs from `/data/CoordExp` main
- Affected training/runtime surfaces during eventual implementation:
  - `src/sft.py`
  - `src/training/surfaces.py`
  - `src/training/pipeline_registry.py`
  - `src/training/pipelines/`
  - `src/datasets/builders/jsonlines.py`
  - `src/datasets/dense_caption.py`
  - `src/detection/packing.py`
  - `src/trainers/teacher_forcing/`
  - `src/trainers/stage2_rollout_correction.py`
  - `src/trainers/stage2_rollout_correction_impl.py`
- Affected artifact/provenance surfaces during eventual implementation:
  - resolved config artifacts
  - encoded-sample cache fingerprints
  - static-packing cache fingerprints
  - Stage-1 and Stage-2 run manifests
  - Stage-2 policy provenance
- Affected docs/specs during eventual implementation:
  - `docs/AGENT_INDEX.md`
  - `docs/catalog.yaml`
  - `docs/data/PACKING.md`
  - `docs/training/README.md`
  - `docs/training/STAGE1_OBJECTIVE.md`
  - `docs/training/STAGE2_RUNBOOK.md`
  - `docs/ARTIFACTS.md`
- Implementation remains gated on explicit user approval after review
  convergence. This proposal does not approve production code changes.
