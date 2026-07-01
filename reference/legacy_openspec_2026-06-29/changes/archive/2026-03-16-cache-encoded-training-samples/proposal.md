## Why

Real `stage2_two_channel` production runs are now dominated by rollout generation,
not learner forward or post-rollout packing.
Recent timing records for the rollout-heavy
`ab_mixed_coco1024_bmajority_channel_b_triage_posterior` profile show:

- Channel-B learner forward stays roughly flat at about `16s`,
- post-rollout packing stays negligible at about `0.006s`,
- rollout teacher-target encoding stays small at about `1.1-1.3s`,
- while rollout generation remains the dominant cost on B steps.

That means most direct speedups now require changing rollout behavior or decode
budgets, which is explicitly out of scope for this change.

However, one meaningful repo-side inefficiency still remains in the training hot
path:

- `BaseCaptionDataset.__getitem__` deep-copies the raw record,
- rebuilds the rendered conversation payload,
- calls `template.encode(...)`,
- and re-attaches metadata for every sample fetch, every epoch.

For deterministic production configs, this work is redundant:

- the input JSONL is fixed,
- prompts and template settings are fixed,
- object ordering is fixed,
- augmentation is disabled,
- and the encoded teacher-forced sample does not change from epoch to epoch.

We need a config-first way to materialize and reuse those encoded samples so
deterministic training runs stop paying repeated CPU/tokenization/rendering cost
inside dataloader workers.

## What Changes

- Add a deterministic encoded-sample cache for the CoordExp training dataset
  path.
- Scope the first implementation to deterministic `BaseCaptionDataset`
  training runs only:
  - plain JSONL-backed datasets with one stable base record per `base_idx`,
  - no augmentation or preprocessors that mutate sample content at fetch time,
  - no `curriculum_state`-driven mutation,
  - no `custom.object_ordering: random`,
  - no fusion/mixing schedule or hard-sample plan that changes sample
    identity/order per epoch.
- Introduce a canonical YAML-first cache surface under
  `training.encoded_sample_cache`, without new CLI flags:
  - `training.encoded_sample_cache.enabled` defaults to `false`,
  - `training.encoded_sample_cache.root_dir` is optional; when omitted, the
    cache root defaults to
    `<training.output_dir>/cache/encoded_samples/`,
  - `training.encoded_sample_cache.ineligible_policy` defaults to `error`,
    with optional explicit `bypass`,
  - `training.encoded_sample_cache.wait_timeout_s` defaults to `7200`, and `0`
    means wait indefinitely for a cache built by another rank/process.
- Materialize encoded samples to a fingerprinted cache directory using a stable
  fingerprint derived from all length/encoding-affecting inputs, including:
  - resolved train/val JSONL identity,
  - prompt/template settings,
  - `global_max_length` / template length surfaces,
  - object ordering / object field ordering,
  - coord-token configuration,
  - offline image-budget invariants,
  - dataset seed and dataset mode (`dense` vs `summary`),
  - and other deterministic dataset-rendering knobs that affect
    `template.encode`.
- Use `base_idx` as the stable lookup identity for cache reuse so epoch-local
  ordering can still shuffle while the encoded payload remains reusable.
- Build the cache eagerly with a single writer before dataloader workers enter
  the hot fetch path; on cache hit, training SHALL load the encoded sample
  payload directly instead of rebuilding messages and re-running
  `template.encode(...)` inside `BaseCaptionDataset.__getitem__`.
- Preserve existing metadata and training semantics:
  - sample ids,
  - dataset/base index provenance,
  - objects/assistant payload metadata required by downstream metrics or
    trainers,
  - packing compatibility for both Stage-1 static packing and Stage-2 raw batch
    consumption.
- Make ineligible requests explicit:
  - by default (`ineligible_policy: error`), cache initialization fails fast
    with actionable guidance,
  - `ineligible_policy: bypass` is allowed only when authored explicitly, and
    it must log the bypass reason and continue without cache reuse.
- Add operator-visible provenance without changing per-step trainer metric
  contracts:
  - startup logs must report whether the cache was built, reused, or bypassed,
  - run artifacts must record cache fingerprint, resolved root, resolved cache
    directory, policy, and any bypass reason,
  - the cache directory must contain a manifest describing the encoded artifact
    set and build provenance.

## Capabilities

### Added Capabilities
- `encoded-training-cache`: deterministic, reusable encoded-sample artifacts for
  JSONL-backed CoordExp training datasets.

## Impact

- `src/datasets/dense_caption.py` becomes the primary integration point:
  - deterministic sample rendering/encoding moves behind an optional cache
    boundary,
  - `__getitem__` becomes a cache lookup path for supported runs.
- A new cache helper/module under `src/datasets/` or
  `src/datasets/wrappers/` will likely own:
  - fingerprinting,
  - artifact layout,
  - serialization/deserialization,
  - and compatibility checks for deterministic reuse.
- `src/sft.py` and/or `src/config/loader.py` will need to thread the new YAML
  config into dataset construction without adding CLI flags.
- Production runs gain a new operational optimization path without changing:
  - model math,
  - rollout decoding policy,
  - memory ceilings,
  - or Stage-2 objective semantics.
- The change must remain contract-compatible with:
  - Stage-1 static packing consumers,
  - Stage-2 raw fetch / rollout-aware trainers,
  - and the existing training artifact layout under `training.output_dir`.
- Disk usage increases because encoded artifacts are stored explicitly, so the
  implementation must keep cache directories provenance-rich and safe for reuse.
- The safe default remains run-scoped under `training.output_dir`, but an
  explicit `root_dir` may point to a shared location for cross-run reuse of the
  same fingerprint.
- Non-deterministic dataset modes remain supported, but they do not get silent
  cache reuse; the system must disable or reject caching for those modes.
