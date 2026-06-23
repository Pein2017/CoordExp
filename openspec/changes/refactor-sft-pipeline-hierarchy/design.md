## Context

This change follows the refactoring-program charter at
`docs/architecture/proposals/2026-06-17-refactoring-program/REFACTORING_PROGRAM_CHARTER.md`
and the lifecycle registry at
`docs/architecture/proposals/2026-06-17-refactoring-program/lifecycle_registry.yaml`.
The user has directed the active config boundary to be read from `/data/CoordExp`
main, while all edits for this program remain in
`/data/CoordExp/.worktrees/codebase-refactoring-program`.

The current main checkout still has several stable-but-transitional facts:

- `src/config/schema.py` parses `custom.object_ordering`,
  `custom.object_field_order`, `custom.detection_template_id`, and
  `custom.trainer_variant`.
- `src/training/surfaces.py` is explicitly a shadow resolver with `surface.id`
  vocabulary.
- active stable Stage-2 specs still name `custom.trainer_variant:
  stage2_rollout_correction` as the Stage-2 selector.
- stable Stage-1 specs still use `objective.id: teacher_forcing` for
  DetectionScene-backed research teacher forcing.
- docs/catalog routing in `/data/CoordExp` main still names
  `configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml` as the
  canonical Stage-1 detection teacher-forcing config; that file uses a flat
  dialect with `data.object_ordering: random_permutation`,
  `prompt.prompt_variant_enabled: true`, `detection_template.strict_parse:
  true`, flat `token_rows`, and `objective.id: teacher_forcing`.
- active Standard SFT configs still route prompt variants through
  `custom.extra.prompt_variant` includes.
- active Standard SFT launch-prep work in main for sorted/random ordering with
  object and bbox closure is current work and belongs in the first migration
  slice with protected status, not in stale/legacy cleanup.
- `detection_template.id` is already the active template contract and should
  stay top-level for this migration.

The desired endpoint is a smaller public training vocabulary:

```text
authored YAML
  -> typed config
  -> pipeline registry
  -> sample factory / objective config
  -> dataset or rollout preparation
  -> trainer/runtime
  -> resolved config, cache, manifest, provenance
```

The refactor should reveal the fundamental sequence-learning/training
mechanism without forcing a repository-wide rename away from `detection`.
`detection` remains the concrete current object/bbox task vocabulary.

## Goals / Non-Goals

### Goals

- Make active training-family selection explicit through `pipeline.id`.
- Remove `custom.trainer_variant` as an active selector in the target
  hierarchy.
- Move sequence-materialization controls from generic `custom.*` authoring to
  `sample_factory.target_sequence`.
- Move active Stage-1 training prompt variant authoring out of
  `custom.extra.prompt_variant` and into a dedicated prompt namespace.
- Move historical flat `prompt.prompt_variant_enabled` booleans into the same
  prompt namespace during migration.
- Keep `detection_template.id` as the stable template identity.
- Make Standard SFT, research teacher forcing, and Stage-2 rollout correction
  separate lanes with explicit ownership.
- Make `standard_ce` the public Standard SFT objective id.
- Make `research_teacher_forcing` the public fine-grained research
  teacher-forcing objective id.
- Preserve recursive-detection / ET-RMP as research/comparator behavior.
- Keep Stage-2 self-trajectory rollout correction separate from GT-sequence
  SFT.
- Require provenance/cache identity to include the effective normalized
  training hierarchy.
- Promote compact token embedding adaptation to a high-level
  `token_embeddings_adapter` namespace and retire `token_rows` plus
  `custom.token_embeddings_adapter` authoring from active configs.
- Standardize active config filenames around semantic axes rather than
  ambiguous numbers or `version` fields.
- Require Stage-1 research teacher forcing to support packing after exact
  atom-position remapping is implemented and tested.

### Non-Goals

- No implementation code is approved by this change before explicit user
  approval.
- No benchmark or launch-readiness claim.
- No broad cleanup of `src/analysis`, `scripts/analysis`, or
  `configs/analysis`.
- No broad rename from `detection` to `grounding`.
- No migration of `detection_template.id` under `sample_factory.target_sequence`.
- No rewrite of Stage-2 `stage2_rollout_correction.pipeline.objective[]` in this
  change.
- No removal of recursive-detection / ET-RMP comparator lineage.
- No new stable CLI flags.

## Key Decisions

### Decision: `pipeline.id` Is The Public Selector

The public selector is a top-level mapping:

```yaml
pipeline:
  id: stage1_standard_sft
```

Supported public ids are:

- `stage1_standard_sft`
- `stage1_research_teacher_forcing`
- `stage2_rollout_correction`

Do not add `pipeline_id` as a scalar alias. Do not preserve `surface.id` as
public config vocabulary. Do not use `custom.trainer_variant` as a compatibility
reader for active target-hierarchy configs.

### Decision: Rename The Shadow Resolver Toward Pipeline Vocabulary

The current `src/training/surfaces.py` concept should move toward
`src/training/pipeline_registry.py`. Concrete implementations remain under
`src/training/pipelines/`.

This is not just a filename preference. The word `surface` is useful for
lifecycle/governance prose, but it is too abstract for authored config and
implementation discovery. Future agents should look for pipeline selection in a
pipeline registry.

The migration should replace or delete `src/training/surfaces.py` immediately
after active imports move to `src/training/pipeline_registry.py`. Do not keep a
long-lived `surfaces.py` compatibility shim, and do not preserve its old
Stage-2 `teacher_forcing` policy as public behavior.

### Decision: `detection_template.id` Remains Top-Level

`detection_template.id` already owns compact template family, closure tokens,
separator/parser compatibility, token-row requirements, and artifact metadata
through the active template work. This change does not move that identity under
`sample_factory.target_sequence`.

`sample_factory.target_sequence` owns sequence-materialization fields:

```yaml
sample_factory:
  id: detection_sequence
  target_sequence:
    task_family: detection
    object_ordering: sorted
    object_field_order: desc_first
    bbox_format: xyxy
    coordinate_surface: coord_token
    strict_parse: true
```

Old and new sequence-control paths must not be authored together, even with
matching values, outside explicit migration tests.

The first migration slice must include the catalog-canonical flat research-TF
config:

```text
configs/stage1/detection_teacher_forcing/prod/compact_support2.yaml
```

Flat fields map as follows:

- `data.object_ordering` -> `sample_factory.target_sequence.object_ordering`
- `prompt.prompt_variant_enabled: true` -> `prompt.variant: coco_80`
- `prompt.prompt_variant_enabled: false` -> no prompt variant
- `detection_template.coordinate_surface` ->
  `sample_factory.target_sequence.coordinate_surface`
- `detection_template.bbox_format` ->
  `sample_factory.target_sequence.bbox_format`
- `detection_template.strict_parse` ->
  `sample_factory.target_sequence.strict_parse`
- `token_rows` -> `token_embeddings_adapter`
- `objective.id: teacher_forcing` -> reject; migrated active configs must
  author `objective.id: research_teacher_forcing`

`custom.detection_sequence_format` is subsumed by the pair
`detection_template.id` and `sample_factory.id`. Active migrated configs must
not keep it as a generic custom field.

### Decision: Prompt Identity Is Normalized, Not Hidden In `custom.extra`

Stage-1 training prompt variant authoring should move to:

```yaml
prompt:
  variant: coco_80
```

Stage-2 rollout-correction prompt variants remain in the rollout runtime
namespace:

```yaml
rollout_matching:
  prompt_variant: coco_80
  eval_prompt_variant: coco_80
```

This keeps Standard SFT, research teacher forcing, Stage-2 train rollouts, and
Stage-2 eval rollouts explicit without creating another generic `custom` bucket.
Resolved config, prompt hash, cache fingerprints, manifests, and policy
provenance must record the resolved prompt variant, tokenizer identity, and chat
template identity.

### Decision: Standard SFT Uses `standard_ce`

Standard SFT is ordinary assistant-label teacher forcing with pure CE by
default. The public objective identity is:

```yaml
objective:
  id: standard_ce
```

Optional geometry or soft-CE auxiliaries sit under this objective rather than
becoming new public pipeline ids:

```yaml
objective:
  id: standard_ce
  auxiliaries:
    coord_soft_ce:
      enabled: false
    geometry:
      enabled: false
```

`token_ce` may remain an implementation function, module, metric component, or
internal term name. It is not the durable public objective id unless a later
OpenSpec explicitly promotes it.

### Decision: Research Teacher Forcing Uses One Public Objective Family

Research teacher forcing owns fine-grained token role/value/span tracing, valid
sets, branch state, force/weight policies, and exact label/logit positions. The
public objective identity is:

```yaml
objective:
  id: research_teacher_forcing
  terms:
    standard_ce:
      weight: 1.0
    valid_set_marginal:
      weight: 0.0
```

The internal weighted-term model prevents every research tactic from becoming a
new public pipeline id. The old `teacher_forcing` id is not a migration alias:
active migrated configs must fail fast if they author it. Rejection tests and
historical/archive evidence may still mention the old value.

### Decision: Stage-2 Keeps Its Internal Objective Namespace

Stage-2 self-trajectory correction is selected by:

```yaml
pipeline:
  id: stage2_rollout_correction
```

The existing Stage-2 internal objective contract remains:

```yaml
stage2_rollout_correction:
  pipeline:
    objective:
      - name: residual_set_correction
        enabled: true
```

This OpenSpec documents the relationship between top-level training-family
selection and Stage-2's internal correction pipeline. It does not move Stage-2
objective modules into the Stage-1 `objective.id` model.

Stage-2 provenance also migrates immediately: `pipeline.id` replaces
`trainer_variant` as the recorded selector. Migrated active Stage-2 manifests
and policy provenance must not emit a compatibility `trainer_variant` field.

### Decision: Packing Is First-Class

Standard SFT owns the high-throughput packed-forward path. Research teacher
forcing may reject packing only while exact token/atom/span remapping is
unimplemented. The implementation must add tests for packed position remapping
before declaring research teacher forcing packing-supported.

Stage-2 keeps rollout-specific post-rollout packing: rollout generation remains
unpacked, and correction segments are packed after rollout according to the
existing Stage-2 packing contract.

For research teacher forcing, the migration must preserve the current target-IR
packing guard semantics: exact packing mapping remains disabled/rejected until
metadata preservation, segment offsets, valid sets, force/weight policies,
sample ids, and image-grid slices are proven under packed execution.

### Decision: Provenance And Cache Identity Include Effective Hierarchy

Resolved config, run manifests, encoded-sample cache fingerprints, static
packing fingerprints, and relevant provenance records must include:

- `pipeline.id`
- `objective.id`
- `detection_template.id`
- `sample_factory.id`
- `sample_factory.target_sequence.object_ordering`
- `sample_factory.target_sequence.object_field_order`
- `sample_factory.target_sequence.bbox_format`
- `sample_factory.target_sequence.coordinate_surface`
- `sample_factory.target_sequence.strict_parse`
- resolved train prompt variant
- resolved prompt hash
- tokenizer identity
- chat-template identity
- effective packing length

This prevents sorted/random ordering, desc-first/geometry-first rows,
closed-marker templates, strict parser modes, and packing regimes from sharing
stale cache/provenance identity.

The normalized hierarchy fields are additive to the exact cache/provenance key
sets that already exist in the implementation. They must not replace current
dataset/source, split, sample-limit, tokenizer/model, prompt/system text,
packing, shuffle, or offline image-budget discriminators. This change does not
add a new image-root or view-store discriminator unless that discriminator is
already present in the current key set being extended.

### Decision: `strict_parse` Has One Normalized Training Source

Active Stage-1 target-hierarchy configs author parser strictness through:

```yaml
sample_factory:
  target_sequence:
    strict_parse: true
```

The loader must normalize that value into the effective template/evaluation
parser policy used by target rendering, Stage-1 eval callbacks, eval/infer
materialization, cache fingerprints, and provenance. Legacy authored
`detection_template.strict_parse` or separate evaluation parser strictness may
exist only as migration fixtures or derived resolved fields; active configs must
fail fast if multiple authored parser-policy paths disagree or duplicate each
other.

### Decision: Token Embedding Adaptation Is High-Level

Compact structural token embedding setup is configured through:

```yaml
token_embeddings_adapter:
  enabled: true
```

The adapter derives required rows from `detection_template.id`, not from
target-sequence field order. `token_rows` and
`custom.token_embeddings_adapter` are deprecated by this program and rejected
after migration. This keeps token embedding setup separate from sequence
materialization while preserving exact row-set validation for compact templates.

### Decision: Active Config Names Are Semantic

Migration should rename or compose active config leaves so filenames expose the
research meaning:

- pipeline family,
- objective family,
- template identity,
- object ordering,
- field order,
- prompt variant,
- packing mode or effective length when relevant.

Do not encode migration state with ambiguous `number`, `version`, `v1`, `v2`,
or similar fields/names. If the loader needs schema evolution later, that
should be a deliberate schema contract, not an experiment/config-family naming
shortcut.

## Risks / Trade-offs

- **Breaking active configs**: active repo-owned configs must migrate with the
  implementation slice. Mitigation: reference `/data/CoordExp` main for active
  config boundary and keep archive/historical configs out of active routing.
- **Stage-2 spec contradiction**: stable Stage-2 specs currently mention
  `custom.trainer_variant`. Mitigation: include explicit delta specs for
  `stage2-rollout-correction` and `rollout-matching-sft`.
- **Objective vocabulary churn**: `standard_ce` and `research_teacher_forcing`
  introduce new public ids while current code uses `token_ce` and
  `teacher_forcing`. Mitigation: tests must prove public config ids select the
  intended implementation modules and metrics, and tests must prove
  `teacher_forcing` is rejected for migrated active configs.
- **Packing correctness risk**: research teacher forcing requires exact
  sidecar/atom position remapping under packing. Mitigation: keep packing
  rejected until remapping tests pass.
- **Prompt/cache collision risk**: prompt variants and chat-template identity
  affect tokenized bytes. Mitigation: include prompt identity in resolved config,
  cache fingerprints, manifests, and provenance.
- **Cache discriminator regression risk**: adding normalized hierarchy fields
  could accidentally replace existing dataset/image/model/packing discriminators.
  Mitigation: state the additive contract and test both old and new axes.
- **Scope creep risk**: analysis bloat, Stage-2 runtime decomposition, and
  recursive-detection cleanup are related but not part of this OpenSpec.
  Mitigation: keep those in the refactoring-program charter and separate
  roadmap phases.

## Review Requirements

Before implementation starts:

- OpenSpec deltas must validate strictly.
- A superpower implementation roadmap must exist and remain approval-gated.
- Independent review lanes must cover:
  - OpenSpec governance and compatibility scope,
  - config/schema migration and active-main boundary,
  - packing/cache/provenance correctness,
  - Stage-2 separation and rollout/self-trajectory semantics,
  - implementation roadmap sufficiency.
- Accepted P0/P1 findings must be revised into the artifacts or converted into
  explicit user decisions.
- Final state must be `ready for user approval`, not `approved to implement`.

## Remaining Approval-Time Check

The only expected approval-time refresh is operational: re-inspect
`/data/CoordExp` main immediately before implementation in case the active
config set changed after these docs were refined. The semantic decisions above
are locked by user choice unless the user explicitly reopens them.
