# Tasks

Production code MUST NOT be edited until the user explicitly approves
implementation after review convergence.

## 1. Docs/Spec/Plan Convergence

- [x] 1.1 Draft proposal, design, delta specs, and tasks for
  `refactor-sft-pipeline-hierarchy`.
- [x] 1.2 Draft the associated Superpowers implementation roadmap.
- [x] 1.3 Run independent read-only review lanes for OpenSpec scope,
  config/schema migration, packing/cache/provenance, Stage-2 separation, and
  implementation-plan sufficiency.
- [x] 1.4 Triage reviewer findings into P0/P1/P2/non-blocking/wrong/duplicate.
- [x] 1.5 Revise OpenSpec and Superpowers artifacts for accepted P0/P1
  findings.
- [x] 1.6 Run strict OpenSpec and docs-plan verification.
- [x] 1.7 Stop at `ready for user approval`; do not begin implementation.
- [x] 1.8 Resolve user blocking decisions: reject `teacher_forcing`, replace
  `trainer_variant` immediately, migrate/delete `surfaces.py` immediately,
  keep research-TF packing fail-fast first, promote `token_embeddings_adapter`,
  and standardize semantic config names.

## 2. Implementation Approval Gate

- [x] 2.1 Receive explicit user approval to implement this OpenSpec change.
- [x] 2.2 Re-inspect `/data/CoordExp` main for active configs immediately before
  implementation.
- [x] 2.3 User decision locked: `objective.id: teacher_forcing` is rejected for
  active migrated configs and is allowed only in historical/archive evidence or
  explicit rejection tests.
- [x] 2.4 User decision locked: `src/training/surfaces.py` is migrated/deleted
  immediately after import consumers move to `src/training/pipeline_registry.py`;
  do not keep a long-lived compatibility shim.
- [x] 2.5 User decision locked: first-slice research teacher-forcing packing is
  fail-fast only until exact atom-position remapping is implemented and tested.
- [x] 2.6 User decision locked: compact token embedding adaptation moves to
  high-level `token_embeddings_adapter`; flat `token_rows` and
  `custom.token_embeddings_adapter` are deprecated by this program.
- [x] 2.7 User decision locked: active config names should use semantic axes,
  not ambiguous numbers or `version`-style fields.

## 3. Characterization Tests Before Code Motion

- [x] 3.1 Add schema tests proving active target-hierarchy configs load with
  `pipeline.id` and fail for `pipeline_id`, `surface.id`, and
  `custom.trainer_variant`.
- [x] 3.2 Add schema tests proving old/new sequence-control dual authoring
  fails for `custom.object_ordering`, `custom.object_field_order`, and
  matching duplicate values.
- [x] 3.3 Add schema tests proving `detection_template.id` remains top-level and
  `custom.detection_template_id` and
  `sample_factory.target_sequence.template_id` are rejected for active configs.
- [x] 3.4 Add objective-resolution tests proving `standard_ce` maps to ordinary
  token-level CE implementation while resolved config records `standard_ce`.
- [x] 3.5 Add objective-resolution tests proving
  `research_teacher_forcing` owns role/value/span metadata and internal
  weighted terms, and `objective.id: teacher_forcing` fails fast for active
  migrated configs.
- [x] 3.6 Add prompt-identity tests proving active Stage-1 prompt variants use
  `prompt.variant`, not `custom.extra.prompt_variant`, and proving historical
  `prompt.prompt_variant_enabled: true` migrates to `prompt.variant: coco_80`.
- [x] 3.7 Add parser-policy tests proving
  `sample_factory.target_sequence.strict_parse` is the single active authored
  Stage-1 parser strictness source and conflicting legacy parser paths fail.
- [x] 3.8 Add Stage-2 tests proving `pipeline.id: stage2_rollout_correction`
  selects Stage-2 while `stage2_rollout_correction.pipeline.objective[]`
  remains the internal objective namespace.
- [x] 3.9 Add Stage-2 negative tests for retired selectors, removed strategy
  namespaces, missing `rollout_matching`, and non-residual objective modules.
- [x] 3.10 Add provenance/cache fingerprint tests proving normalized identity
  fields affect encoded-sample and static-packing fingerprints, including prompt
  variant, prompt hash, tokenizer identity, and chat-template identity.
- [x] 3.11 Add cache/provenance regression tests proving normalized hierarchy
  fields are additive to the current implementation's existing discriminator
  key sets. Do not add a new image-root or view-store discriminator unless it is
  already present in the current key set under test.
- [x] 3.12 Add research teacher-forcing packing tests proving packing rejects
  until exact atom-position remapping is implemented.
- [x] 3.13 Add migration characterization for the catalog-canonical flat config
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml`,
  including `data.object_ordering`, `prompt.prompt_variant_enabled`,
  `detection_template.coordinate_surface`, `detection_template.bbox_format`,
  `detection_template.strict_parse`, `token_rows`, and
  `objective.id: teacher_forcing`.

## 4. Config Schema And Loader Migration After Approval

- [x] 4.1 Add typed config structures for `pipeline`, `sample_factory`, and
  target-sequence controls.
- [x] 4.2 Reject `custom.trainer_variant` as an active selector with errors
  pointing to `pipeline.id`.
- [x] 4.3 Reject `pipeline_id` and public `surface.id`.
- [x] 4.4 Move active sequence-control authoring from `custom.*` to
  `sample_factory.target_sequence`.
- [x] 4.5 Keep `detection_template.id` top-level and reject any new template id
  authoring path under `sample_factory.target_sequence`.
- [x] 4.6 Migrate active Stage-1 prompt variant authoring from
  `custom.extra.prompt_variant` to `prompt.variant`.
- [x] 4.7 Migrate flat `prompt.prompt_variant_enabled: true` to
  `prompt.variant: coco_80`, migrate `false` to no prompt variant, and reject
  `prompt.prompt_variant_enabled` after migration.
- [x] 4.8 Normalize resolved config identity for pipeline, objective, sample
  factory, target sequence, template, prompt identity, tokenizer/chat-template
  identity, and packing fields.
- [x] 4.9 Preserve existing cache/provenance discriminators while adding the new
  normalized hierarchy fields.
- [x] 4.10 Migrate `custom.detection_sequence_format` into
  `detection_template.id` plus `sample_factory.id`, and reject
  `custom.detection_sequence_format` after migration.
- [x] 4.11 Migrate compact token embedding setup to top-level
  `token_embeddings_adapter`, and reject flat `token_rows` plus
  `custom.token_embeddings_adapter` after migration.

## 5. Pipeline Registry Migration After Approval

- [x] 5.1 Introduce `src/training/pipeline_registry.py` as the discoverable
  pipeline selection seam.
- [x] 5.2 Move or rename active `TrainingSurfaceResolver` consumers to pipeline
  registry vocabulary.
- [x] 5.3 Keep concrete pipeline descriptors under `src/training/pipelines/`.
- [x] 5.4 Update imports, docs, and tests so active routing no longer points to
  `src/training/surfaces.py` as the long-term owner.
- [x] 5.5 Add absence/search tests for public `surface.id` authoring.
- [x] 5.6 Delete or rename `src/training/surfaces.py` in this implementation
  slice after imports move. Do not keep a long-lived compatibility shim.
- [x] 5.7 Remove the old Stage-2 shadow-resolver `teacher_forcing` policy so it
  cannot appear as a public Stage-2 objective.

## 6. Standard SFT Objective Migration After Approval

- [x] 6.1 Add `standard_ce` as the public Standard SFT objective id.
- [x] 6.2 Map `standard_ce` to the existing token-level CE implementation.
- [x] 6.3 Preserve token-level CE metrics without making `token_ce` the public
  config id.
- [x] 6.4 Add optional auxiliary objective structure beneath `standard_ce`.
- [x] 6.5 Migrate active Standard SFT configs from `/data/CoordExp` main to
  `pipeline.id: stage1_standard_sft` and `objective.id: standard_ce`.

## 7. Research Teacher Forcing Migration After Approval

- [x] 7.1 Add `research_teacher_forcing` as the public fine-grained research
  teacher-forcing objective id.
- [x] 7.2 Preserve role/value/span tracing, valid sets, branch state, and
  force/weight policies.
- [x] 7.3 Represent research tactics as internal weighted terms instead of new
  public pipeline ids.
- [x] 7.4 Preserve recursive-detection / ET-RMP comparator lineage under the
  research lane.
- [x] 7.5 Reject the old `objective.id: teacher_forcing` id for active migrated
  configs, with errors pointing to `research_teacher_forcing`.
- [x] 7.6 Preserve recursive-detection / ET-RMP handles as comparator lineage or
  obtain explicit user approval before retiring any handle.

## 8. Stage-2 Selector Migration After Approval

- [x] 8.1 Migrate active Stage-2 configs to `pipeline.id:
  stage2_rollout_correction`.
- [x] 8.2 Keep `stage2_rollout_correction.pipeline.objective[]` unchanged for
  residual-set correction.
- [x] 8.3 Keep `rollout_matching.*` as runtime/backend/decode/eval settings.
- [x] 8.4 Reject `custom.trainer_variant: stage2_rollout_correction` with
  guidance to `pipeline.id`.
- [x] 8.5 Reject retired Stage-2 strategy surfaces, including `stage2_ab`,
  `stage2_two_channel`, `stage2_ab_training`, `rollout_matching_sft`,
  `stage2_rollout_aligned`, `stage2_rollout_runtime`,
  `rollout_matching.pipeline`, removed per-channel namespaces, and non-residual
  objective modules.
- [x] 8.6 Preserve the fail-fast requirement that
  `pipeline.id: stage2_rollout_correction` requires `rollout_matching` runtime
  settings unless a later OpenSpec defines defaults.
- [x] 8.7 Build Stage-2 policy provenance from
  `pipeline.id: stage2_rollout_correction`, preserving existing assignment,
  duplicate-filter, ordering, rollout template, invalid-rollout, and
  fallback-loss fields while omitting compatibility `trainer_variant` selector
  output.
- [x] 8.8 Update Stage-2 docs and catalog routing after tests pass.

## 9. Provenance, Cache, And Packing After Approval

- [x] 9.1 Include normalized hierarchy fields in resolved config artifacts.
- [x] 9.2 Include normalized hierarchy fields in encoded-sample cache
  fingerprints.
- [x] 9.3 Include normalized hierarchy fields in static-packing cache
  fingerprints.
- [x] 9.4 Include normalized hierarchy fields in Stage-1 and Stage-2 manifests
  and policy provenance.
- [x] 9.5 Include resolved prompt variant, prompt hash, tokenizer identity, and
  chat-template identity in cache/provenance records.
- [x] 9.6 Preserve the current implementation's existing dataset/source/model/
  tokenizer/prompt/sample-limit/packing/offline-image-budget cache
  discriminators while adding normalized hierarchy fields. Do not add a new
  image-root or view-store discriminator unless it already exists in the current
  key set under test.
- [x] 9.7 Keep research teacher-forcing packing fail-fast until exact
  atom-position remapping tests pass.
- [x] 9.8 Preserve the current target-IR exact-packing-mapping guard, span
  validation, and packed sidecar rejection until remapping support is proven.
- [x] 9.9 Defer packed research teacher-forcing remapping as a follow-up until
  tests cover label/logit positions, token roles, valid sets, force/weight
  policies, segment offsets, sample ids, image-grid slices, `encoded_len`,
  `cu_seq_lens`, and packed-vs-unpacked loss parity; preserve fail-fast guards
  in this slice.

## 10. Docs, Validation, And Handoff After Approval

- [x] 10.1 Update `docs/AGENT_INDEX.md`, `docs/catalog.yaml`,
  `docs/data/PACKING.md`, `docs/training/README.md`,
  `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE2_RUNBOOK.md`, and
  `docs/ARTIFACTS.md` after implementation tests prove behavior.
- [x] 10.2 Run targeted config/schema/objective/pipeline/cache/packing tests.
- [x] 10.3 Run `openspec validate refactor-sft-pipeline-hierarchy --type change
  --strict`.
- [x] 10.4 Run `python -m repo_lifecycle.report_lifecycle_registry`.
- [x] 10.5 Run `python -m pytest tests/test_lifecycle_registry_report.py -q`.
- [x] 10.6 Run `git diff --check`.
- [x] 10.7 Report skipped broad or hardware-heavy checks with the reason.
