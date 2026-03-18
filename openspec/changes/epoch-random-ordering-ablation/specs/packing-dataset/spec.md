# packing-dataset Spec Delta

This is a delta spec for change `epoch-random-ordering-ablation`.

## MODIFIED Requirements

### Requirement: Static pack plans are deterministic and epoch-invariant
When static pack-plan mode is selected (`training.packing_mode=static`), the system SHALL compute a pack plan deterministically such that:
- for a fixed base dataset, fixed dataset seed, and fixed packing configuration, the resulting `raw_plan` is identical across runs, and
- for a fixed `world_size` and fixed `training.dataloader_drop_last`, the resulting `aligned_plan(world_size, dataloader_drop_last)` is identical across runs, and
- both plans are identical across DDP ranks (either computed identically or shared from a single source of truth).

Determinism scope:
- Determinism MUST include which raw samples are included in the plan, their grouping into packs, and:
  - each pack’s intra-pack order, and
  - the ordering of packs in the plan.
- Tie-breaking in any heuristic (e.g., binpacking) MUST be stable and deterministic.
- The system SHALL enforce a concrete deterministic ordering:
  - Within each pack, raw sample indices SHALL be ordered ascending.
  - Packs within the plan SHALL be ordered by the smallest raw index in the pack (ascending), with deterministic tie-breaking by lexicographic comparison of the full index lists.

Epoch semantics:
- Static pack-plan mode MUST NOT regenerate `raw_plan` or `aligned_plan` per epoch for eligible datasets.
- Static pack-plan mode MAY propagate dataset-level `set_epoch` hooks into the underlying raw dataset only when epoch changes do not change base sample identity, raw-sample index membership, per-index planning length, or pack membership/order.
- For such eligible datasets, per-epoch variation MAY change only deterministic, length-invariant sample content (for example object instance order inside a sample).
- The packed dataset wrapper MUST forward epoch changes to the underlying dataset before sample fetches for the new epoch.
- Datasets that resample per epoch, depend on `set_epoch` to change the sample schedule, or change per-index planning length MUST still be rejected (fail fast) when `training.packing_mode=static`.

#### Scenario: Identical inputs produce identical pack plans
- **GIVEN** the same dataset, seed, and packing knobs across two runs
- **WHEN** pack-plan construction runs in static mode
- **THEN** the emitted sequence of packs (index sets and order) is identical between runs.

#### Scenario: Length-invariant random ordering preserves the static plan across epochs
- **GIVEN** `training.packing=true` and `training.packing_mode=static`
- **AND** the underlying dataset uses `custom.object_ordering: random`
- **AND** per-index planning length is invariant across epochs
- **WHEN** the packed dataset advances from one epoch to the next
- **THEN** `raw_plan` and `aligned_plan` remain unchanged
- **AND** the underlying dataset receives the new epoch before sample fetches
- **AND** fetched samples MAY reflect the new epoch’s deterministic object order without rebuilding the plan.

### Requirement: Static packing length measurement supports multimodal templates
When static pack-plan mode is selected, the system SHALL be able to obtain a deterministic per-sample `length` suitable for pack planning for vision-language samples.

Normative behavior:
- The length used for planning MUST correspond to the encoded token length that will be consumed by the teacher-forced forward pass under the active template.
- The system MUST support computing and storing lengths ahead of packing (e.g., via a cached length store) so the pack plan can be computed without repeatedly re-encoding the full dataset.
- Length computation MUST be deterministic for a fixed dataset record, template, and configuration.
- Any static length/plan cache fingerprint MUST include dataset source identity (for example, resolved `custom.train_jsonl` / `custom.fusion_config` identity) so stale cache reuse across different dataset sources fails fast.
- The run-scoped static cache directory under `training.output_dir` is the primary safety boundary; changing uncaptured length-affecting factors SHOULD use a fresh output directory.
- Static pack-plan mode MUST allow epoch-varying serialized content only when the per-index planning length is invariant across epochs.
- If epoch-varying content can change the planning length or invalidate `sum(length_i) <= packing_length` for an existing pack, the system MUST fail fast with actionable guidance.
- Static length precompute MAY use rank0 multiprocessing, but computed lengths MUST be written back by dataset index so results are deterministic and byte-identical to serial precompute for the same inputs.
- Non-rank0 ranks MUST continue to use the same cache synchronization path (`training.packing_wait_timeout_s`) regardless of rank0 worker count.

#### Scenario: Multimodal samples have deterministic planning lengths
- **GIVEN** a vision-language dataset sample with an image input under the active template
- **WHEN** static packing computes the sample’s planning length
- **THEN** the computed `length` is deterministic across runs for the same configuration
- **AND** the length is usable to enforce `sum(length_i) <= packing_length` without truncating or splitting samples.

#### Scenario: Random-order dataset is accepted only when planning length stays invariant
- **GIVEN** static packing is enabled for a dataset with epoch-varying object order
- **WHEN** the system validates the dataset for static pack planning
- **THEN** it MUST accept the dataset only if the per-index planning length is invariant across epochs
- **AND** otherwise it MUST fail fast with guidance to disable static packing or use a length-invariant configuration.
