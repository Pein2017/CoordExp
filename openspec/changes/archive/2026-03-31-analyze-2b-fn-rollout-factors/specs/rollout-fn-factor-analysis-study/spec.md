## ADDED Requirements

### Requirement: Fixed-checkpoint study manifest
The system SHALL require a manifest-driven study configuration for fixed-checkpoint rollout analysis.

The manifest MUST resolve, at minimum:

- one or more checkpoint aliases, absolute checkpoint paths, artifact kinds, and checkpoint fingerprints,
- dataset split,
- the split-specific offline JSONL,
- optional checkpoint provenance sidecars when provided by the study config,
- an explicit fixed image subset with image ids and image order,
- subset name and subset-selection provenance,
- bootstrap selector provenance when the subset is derived by bootstrap,
- the resolved image root,
- prompt variant,
- prompt hash,
- object field order,
- preprocessing invariants including `do_resize`,
- evaluator settings,
- generation defaults,
- prefix construction defaults,
- the authoritative backend,
- and the study seed schedule.

The study MUST record the resolved manifest before any rollout cell executes.

#### Scenario: Manifest resolves `original` and `a_only`
- **WHEN** the user launches the study with checkpoint aliases `original` and `a_only`
- **THEN** the study writes a resolved manifest that records both aliases, their resolved checkpoint paths, artifact kinds, checkpoint fingerprints, the dataset split, the split-specific offline JSONL, the subset name, the exact image ids and image order, subset-selection provenance, the image root, the prompt variant, the prompt hash, the object field order, the authoritative backend, and the seed schedule before any rollout begins.

#### Scenario: Canonical 2B pair records concrete checkpoint and dataset provenance
- **WHEN** the user launches the canonical first-version study
- **THEN** the resolved manifest records:
  - `original = output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`
  - `a_only = output/stage2_ab/2b_1024/a_only_iter1/merged_ckpt-900`
  - `a_only_config_source = output/stage2_ab/2b_1024/a_only_iter1/epoch_2-eff_size_64-n_softctx_iter_1-a_only/v0-20260309-102351/config_source.yaml`
  - `train = public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - `val = public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- **AND** each executed logical cell records which dataset split and split-specific JSONL it used.

#### Scenario: Unresolved checkpoint alias fails fast
- **WHEN** a manifest references a checkpoint alias whose path does not exist
- **THEN** the study fails before rollout execution and reports which alias could not be resolved.

#### Scenario: Reference-only artifact fails as an executable checkpoint
- **WHEN** a manifest marks an artifact as reference-only
- **AND** the user tries to execute rollout cells against that artifact
- **THEN** the study fails before rollout execution and reports that the artifact is not executable.

### Requirement: Offline fixed-checkpoint execution only
The study SHALL operate on fixed checkpoints and offline `train` / `val` data only.

The workflow MUST NOT trigger training, checkpoint mutation, or online dataset changes as part of the analysis.

#### Scenario: Study runs without training
- **WHEN** the user runs the rollout-factor analysis study
- **THEN** the workflow consumes only the resolved checkpoint paths and split-specific offline JSONL
- **AND** it does not launch any training job or modify checkpoint weights.

### Requirement: Dataset split is a first-class study factor
The study SHALL preserve dataset split as a first-class factor in manifests, logical cell ids, and final reports.

Canonical dataset splits are:

- `train`
- `val`

Each logical cell MUST record:

- `dataset_split`
- the resolved split-specific JSONL

Final reports MUST preserve separate conclusions for `train` and `val`.

#### Scenario: Train and val are not silently pooled
- **WHEN** the study writes aggregate outputs across both dataset splits
- **THEN** it preserves separate `train` and `val` sections, tables, or cohort identifiers
- **AND** it does not silently pool train and val results into one unnamed subset.

### Requirement: Canonical study runs are staged over a fixed small image subset
The study SHALL support a canonical staged experiment program over a fixed small subset of `train` or `val` images.

The study MUST preserve one explicit stage identifier for each logical analysis cell.

The canonical staged program MUST include, at minimum:

- a bootstrap subset-selection stage,
- a deterministic baseline stage,
- an image-only sampled coverage stage,
- a prefix-order stage,
- a switched/broken prefix stress stage,
- a sequence-length cross-check stage,
- and an attribution/reporting stage.

The staged workflow SHOULD emit stage-specific manifests or equivalent provenance artifacts.

Later stages MUST be able to reuse frozen earlier outputs rather than recomputing the full study from scratch.

The canonical staged program MUST support these named hard-case subsets:

- `Hard-16`
- `Hard-32`

These subsets MUST be drawn from poor-performing images inside one dataset split rather than from balanced easy/control images.
`Hard-16` MUST be a strict subset of `Hard-32` within the same dataset split.

#### Scenario: Canonical staged run records the same subset across all stages
- **WHEN** the user launches a canonical staged study on a fixed subset
- **THEN** every logical analysis cell records its stage identifier
- **AND** every stage reuses the same resolved image ids and image order from the manifest.

#### Scenario: Later stages can resume from frozen earlier outputs
- **GIVEN** frozen bootstrap, subset, or baseline outputs already exist for one study run
- **WHEN** the user reruns only a later stage such as sampling, prefix, length, or reporting
- **THEN** the later stage reuses the frozen earlier outputs
- **AND** it does not require the earlier stages to be recomputed.

#### Scenario: Hard-case subset provenance is preserved
- **WHEN** the study resolves `Hard-16` or `Hard-32`
- **THEN** the manifest records the dataset split and why each image belongs to that split-specific subset
- **AND** the study can distinguish `Hard-16` from `Hard-32` in all downstream reports.

#### Scenario: `Hard-16` is the top-16 prefix of `Hard-32`
- **WHEN** the study materializes both canonical hard subsets from one bootstrap ranking
- **THEN** `Hard-16` contains exactly the first `16` images of `Hard-32` for the same dataset split
- **AND** both subsets preserve the same relative image order.

### Requirement: Hard-case subsets are derived by a deterministic bootstrap selector
The study SHALL derive canonical hard-case subsets from one deterministic bootstrap selector pass over a frozen candidate pool for each dataset split.

The bootstrap selector MUST:

- use `image_only` deterministic baseline cells with default length only,
- operate on a frozen candidate pool recorded in a bootstrap manifest,
- preserve the dataset split in that bootstrap manifest,
- exclude rollout-health-invalid images from the final ranking,
- rank images by one deterministic score tuple,
- emit a ranked table plus frozen `Hard-32` and `Hard-16` manifests for that split.

The default deterministic ranking tuple MUST be:

1. descending mean unresolved GT count across `original` and `a_only`,
2. descending mean unmatched prediction count across `original` and `a_only`,
3. descending GT object count,
4. ascending `image_id`.

#### Scenario: Bootstrap selector is reproducible
- **WHEN** the bootstrap selector is rerun from the same candidate pool, checkpoint pair, evaluator settings, and seed schedule
- **THEN** it produces byte-identical ranked outputs
- **AND** byte-identical `Hard-32` and `Hard-16` manifests for the same dataset split.

### Requirement: Authoritative deterministic baseline layer
The study SHALL provide a deterministic baseline comparison layer for each resolved checkpoint.

For the baseline layer, the study MUST:

- use one authoritative backend configuration,
- preserve the resolved prompt and preprocessing contract,
- emit standard `gt_vs_pred`-style artifacts and run summaries,
- and write comparable metrics for each checkpoint on the same image set.

The study SHALL use `hf` for all cell types in the first version of this capability.

#### Scenario: Baseline uses one resolved contract for both checkpoints
- **WHEN** the study executes the deterministic baseline layer for `original` and `a_only`
- **THEN** both checkpoints are evaluated on the same resolved dataset split, image set, prompt contract, preprocessing contract, and authoritative backend
- **AND** each checkpoint emits comparable rollout artifacts and metrics.

### Requirement: Prefix-intervention factor matrix
The study SHALL support explicit prefix-conditioned continuation cells.

At minimum, the study MUST support these prefix modes:

- `image_only`,
- `oracle_gt_prefix_train_order`,
- `oracle_gt_prefix_random_order`,
- `self_prefix`,
- `switched_prefix`,
- `broken_prefix`.

For each prefix-conditioned cell, the study MUST record:

- prefix mode,
- prefix length,
- prefix source checkpoint or provenance,
- prefix ordering rule,
- prefix content hash,
- and any mutation applied to the prefix.

For `oracle_gt_prefix_random_order`, the study MUST use the same GT object set as the matching train-order prefix and MUST record the fixed randomization seed or permutation identifier used for that image.

For prefix-conditioned cells, the study MUST preserve enough information to score continuation-only recovery separately from objects already injected by the prefix.

Prefix-conditioned artifacts MUST serialize:

- all prefix-injected predictions first,
- all continuation-emitted predictions after them,
- `prefix_pred_count`,
- and `continuation_pred_start_index`.

`continuation_pred_start_index` MUST equal `prefix_pred_count`.

Continuation-only recovery MUST be computed by applying the normal matching rule to only the continuation tail of the prediction array, excluding prefix-injected predictions from the candidate set.

The canonical prefix-order stage MUST include comparison among:

- `oracle_gt_prefix_train_order`,
- `oracle_gt_prefix_random_order`,
- and `self_prefix`.

#### Scenario: Oracle random-order continuation records permutation provenance
- **WHEN** the study continues a checkpoint from a GT-derived random-order prefix
- **THEN** the cell artifact records `prefix_mode=oracle_gt_prefix_random_order`, the fixed randomization seed or permutation id, the prefix length, and the continued checkpoint alias.

#### Scenario: Switched-prefix continuation records provenance
- **WHEN** the study continues checkpoint `a_only` from a prefix produced by checkpoint `original`
- **THEN** the cell artifact records `prefix_mode=switched_prefix`, the prefix source checkpoint alias, the prefix length, and the continued checkpoint alias.

#### Scenario: Broken-prefix continuation records mutation
- **WHEN** the study perturbs a plausible prefix by deleting, swapping, or inserting one object before continuation
- **THEN** the resulting cell artifact records `prefix_mode=broken_prefix` and the specific mutation type.

#### Scenario: Prefix injection alone does not count as continuation recovery
- **WHEN** a prefix-conditioned cell injects one GT-aligned prefix object and the continuation emits nothing new
- **THEN** whole-scene review may show the prefix object as present
- **BUT** continuation-only recovery for that GT object is `0`.

### Requirement: Rollout-health gating precedes main causal attribution
The study SHALL evaluate rollout health before using sampled, prefix-conditioned, or length-conditioned cells in the main causal attribution tables.

At minimum, each logical cell MUST record rollout-health fields sufficient to judge whether the cell is interpretable, including:

- non-empty prediction rate,
- parse-valid rate,
- invalid-rollout count,
- duplicate-like rate,
- prediction count,
- and truncation or stop anomalies when available.

The rollout-health gate MUST define formulas, thresholds, and invalid-reason precedence in the resolved manifest.

Default invalid-reason precedence MUST be:

1. `parse_invalid`
2. `invalid_rollout`
3. `truncation_anomaly`
4. `too_few_nonempty`
5. `too_few_predictions`
6. `too_duplicate_like`

Each logical cell MUST record:

- `rollout_health_valid`,
- and `rollout_health_invalid_reason` when invalid.

Cells that fail the rollout-health gate MUST remain visible in artifacts and appendices, but MUST NOT be merged into the main causal attribution tables as if they were healthy evidence.

#### Scenario: Health-invalid prefix cell stays visible but excluded from main attribution
- **WHEN** a prefix-conditioned cell produces unhealthy rollout behavior
- **THEN** the cell remains recorded in study artifacts
- **BUT** it is excluded from the main prefix-attribution tables
- **AND** its invalid reason remains visible.

### Requirement: Union-of-K FN coverage analysis
The study SHALL support stochastic multi-sample rollout analysis for fixed checkpoints.

For each sampled cell, the study MUST record:

- number of rollouts `K`,
- decode parameters,
- logical cell id,
- execution shard id,
- per-rollout prediction counts,
- per-GT-object hit frequency across the `K` rollouts,
- and union-of-`K` recall.

The study MUST preserve enough matching detail to distinguish:

- GT objects never matched in any rollout,
- GT objects matched in some but not all rollouts,
- and GT objects matched in every rollout.

#### Scenario: Per-GT hit frequency is computed from sampled rollouts
- **WHEN** a sampled cell runs with `K=8`
- **AND** a GT object is matched in 3 of the 8 rollouts
- **THEN** the study records that GT object with hit frequency `0.375`
- **AND** it contributes to union-of-`K` recall as covered.

### Requirement: Evaluator-domain GT identity is normative for per-object analysis
The study SHALL use evaluator-domain GT identity as the normative key for per-object analysis artifacts.

The normative GT-object key MUST be:

- `record_idx`
- `gt_idx`

Per-object analysis artifacts SHOULD also preserve:

- `image_id`
- `file_name` when available

When a review or overlay artifact is materialized, the study MUST additionally preserve a derived visualization bridge:

- `canonical_gt_index`

#### Scenario: One recovery row maps to one canonical review object
- **WHEN** the study materializes a review row for one GT object
- **THEN** its `(record_idx, gt_idx)` key maps to exactly one `canonical_gt_index`
- **AND** the review artifact preserves both identities.

### Requirement: Object-level recovery taxonomy uses minimal-intervention precedence
The study SHALL produce an object-level recovery table for GT objects on the fixed study subset.

Each GT object MUST receive one mutually exclusive recovery status from:

- `deterministic_hit`,
- `decode_selection_miss`,
- `prefix_sensitive_miss`,
- `length_bias_miss`,
- `persistent_unrecovered`.

The study MUST assign these statuses with minimal-intervention precedence:

- `decode_selection_miss` takes precedence over later interventions if same-prompt sampled union-of-`K` already recovers the object,
- `prefix_sensitive_miss` applies only if deterministic and same-prompt sampled cells do not recover the object,
- `length_bias_miss` applies only if deterministic, same-prompt sampled, and prefix interventions do not recover the object,
- `persistent_unrecovered` applies only if none of the tested interventions recover the object.

The study SHOULD also preserve supporting quality flags such as `annotation_mismatch_candidate` and `semantic_confusion_candidate`.

#### Scenario: Sampled recovery outranks later interventions
- **WHEN** a GT object is missed by the deterministic baseline
- **AND** recovered by same-prompt union-of-`K`
- **AND** also recovered by a prefix-conditioned cell
- **THEN** the object is recorded as `decode_selection_miss` rather than `prefix_sensitive_miss`.

#### Scenario: Random-order-only recovery is recorded as prefix sensitivity
- **WHEN** a GT object is not recovered by deterministic baseline or image-only sampled coverage
- **AND** it is recovered only when the checkpoint continues from a train-order or random-order oracle prefix
- **THEN** the object is recorded as `prefix_sensitive_miss`
- **AND** the supporting artifact preserves which prefix ordering recovered it.

### Requirement: Sequence-length and EOS-bias analysis
The study SHALL include an explicit sequence-length analysis layer for each checkpoint.

At minimum, the study MUST record for each rollout:

- generated token count,
- emitted object count,
- stop reason,
- whether EOS or equivalent stop was reached,
- and the active `max_new_tokens` cell.

The study MUST support at least:

- a default-length cell,
- and an extended-length control cell.

The study report MUST preserve these statistics in a way that supports checkpoint-vs-checkpoint comparison on the same images.

When both dataset splits are present in one study run, the report MUST also preserve split-specific length summaries so `train` and `val` stop behavior can be compared without pooling.

#### Scenario: Extended-length control is preserved separately
- **WHEN** the study reruns a checkpoint with a larger `max_new_tokens` than the default cell
- **THEN** the run artifacts record that cell separately
- **AND** the report compares emitted object count, token count, and covered GT objects between the default and extended-length cells.

### Requirement: Evidence layers are reported separately
The study SHALL separate conclusions by evidence layer.

At minimum, the final report MUST distinguish:

- deterministic baseline findings,
- sampled union-of-`K` findings,
- prefix-sensitivity findings,
- and sequence-length findings.

The report MUST NOT collapse these layers into a single undifferentiated score when making conclusions about why objects are missed.

#### Scenario: Sampled recovery does not appear as deterministic incapacity
- **WHEN** a GT object is missed in the deterministic baseline but recovered in sampled union-of-`K` analysis
- **THEN** the final report records that object as sampling-recoverable rather than describing it as deterministic incapacity.

### Requirement: Final reports preserve stage-wise interpretation
The study SHALL report findings in stage order so later interventions are interpreted relative to earlier evidence.

At minimum, the final report MUST preserve:

- the deterministic baseline FN set,
- the subset of that FN set recovered by image-only sampling,
- the subset of the remaining FN set recovered by prefix-order interventions,
- the subset of the remaining FN set recovered by switched/broken or extended-length interventions,
- and the residual persistent-unrecovered set.

#### Scenario: Report presents a recovery waterfall instead of one flat table
- **WHEN** the study writes its final summary
- **THEN** it preserves a stage-wise recovery waterfall for GT objects
- **AND** it does not merge later-stage recoveries back into the deterministic baseline counts.

#### Scenario: Reports separate `Hard-16` and `Hard-32`
- **WHEN** the study writes final reports
- **THEN** it reports `Hard-16` and `Hard-32` separately
- **AND** it does not silently pool them into one unnamed subset.

#### Scenario: Reports separate `train` and `val`
- **WHEN** the study writes final reports across both dataset splits
- **THEN** it reports `train` and `val` separately
- **AND** it preserves split-specific recovery waterfalls before any optional cross-split comparison summary.

### Requirement: Multi-GPU study execution is cell-sharded and provenance-preserving
The study SHALL support execution across four local GPUs by sharding independent rollout cells.

The study MUST distinguish:

- one `logical_cell_id` for the analysis condition,
- and one or more `execution_shard_id` values for physical executions of that condition.

The study MUST preserve, per executed shard:

- GPU identifier,
- checkpoint alias,
- backend,
- seed,
- factor settings,
- and output artifact paths.

Merged per-GT analysis outputs MUST be written in deterministic sort order by:

1. `logical_cell_id`
2. `execution_shard_id`
3. `record_idx`
4. `gt_idx`

The study SHALL NOT require model-parallel coupling across the four GPUs for its default workflow.

#### Scenario: Four-GPU run records per-cell device provenance
- **WHEN** the study executes cells across four local GPUs
- **THEN** each completed execution shard records which GPU executed it, which checkpoint alias it used, which backend it used, and which factor settings were active
- **AND** the study preserves which execution shards belong to the same logical analysis cell.

### Requirement: Qualitative audit artifacts remain joinable to quantitative results
The study SHALL emit qualitative audit artifacts that are joinable to quantitative outputs by stable identifiers.

At minimum, the study MUST preserve stable joins using:

- checkpoint alias,
- dataset split,
- image identifier,
- study cell identifier,
- and rollout or sample index when applicable.

When a qualitative artifact is derived from a per-GT recovery row, the join MUST also preserve:

- `record_idx`
- `gt_idx`
- and `canonical_gt_index` for canonical overlays.

The study SHOULD produce paired GT-vs-Pred overlays or equivalent visual artifacts for the most informative cells.

The study SHOULD emit a review queue that highlights examples for:

- `decode_selection_miss`,
- `prefix_sensitive_miss`,
- `length_bias_miss`,
- `persistent_unrecovered`,
- and `annotation_mismatch_candidate`.

#### Scenario: Qualitative overlays can be traced back to one quantitative cell
- **WHEN** the study renders an overlay for a checkpoint/image/prefix cell
- **THEN** the overlay artifact can be joined back to the corresponding quantitative cell outputs using stable identifiers.
