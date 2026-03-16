## Status

Design refined after audit. This document defines the v1 cache contract tightly
enough for a separate implementer to proceed without inventing policy.

## Goals

- Move deterministic render-plus-`template.encode(...)` work out of the
  dataloader fetch hot path for supported training runs.
- Preserve the trainer-visible sample contract for both Stage-1 static packing
  and Stage-2 raw sample fetch.
- Keep the feature config-first and safe under distributed launch.

## Non-Goals

- No rollout-policy, decoding, or objective changes.
- No support in v1 for fusion/mixing, random object ordering, curriculum-driven
  mutation, or fetch-time augmentation.
- No new CLI flags.
- No change to the canonical per-step training metric contract in
  `logging.jsonl`.

## Canonical Config Contract

The cache is controlled only through `training.encoded_sample_cache`.

Typed fields for v1:
- `training.encoded_sample_cache.enabled: bool = false`
- `training.encoded_sample_cache.root_dir: str | null = null`
- `training.encoded_sample_cache.ineligible_policy: \"error\" | \"bypass\" = \"error\"`
- `training.encoded_sample_cache.wait_timeout_s: int = 7200`

Semantics:
- When `root_dir` is omitted, the resolved cache root is
  `<training.output_dir>/cache/encoded_samples`.
- `ineligible_policy: error` is the default safety contract.
- `ineligible_policy: bypass` must be authored explicitly and is the only
  allowed best-effort mode.
- `wait_timeout_s: 0` means wait indefinitely for a cache built by another
  rank/process.
- Unknown keys under `training.encoded_sample_cache` must fail fast like other
  typed config surfaces in the repo.

## Eligibility Boundary

The cache is valid only when the encoded payload is a pure function of stable
base-record identity plus fixed configuration.

V1 eligible runs:
- `BaseCaptionDataset` backed by a fixed JSONL source with one stable base
  record per `base_idx`.
- Dense or summary mode with fixed prompts/system prompts.
- Stable object ordering (`custom.object_ordering: sorted`) and stable object
  field ordering.
- No fetch-time mutation that depends on RNG, epoch, or requested index.

V1 rejected paths:
- any `augmenter`,
- any preprocessor that mutates encoded content at fetch time,
- non-empty `curriculum_state`,
- `custom.object_ordering: random`,
- fusion/mixing schedules,
- hard-sample plans or other epoch-varying sampling contracts that change the
  dataset identity seen at fetch time,
- any path where encoded output depends on `epoch`, requested `index`, or local
  RNG rather than just base-record identity and fixed config.

Policy behavior:
- `ineligible_policy: error` must stop startup with actionable guidance.
- `ineligible_policy: bypass` must continue uncached and record the bypass
  reason in logs and run artifacts.

## Fingerprint Contract

The cache fingerprint must include every authored or resolved surface that can
change the encoded payload or its usable planning length.

Minimum inputs:
- resolved dataset source identity for train/eval JSONL,
- dataset seed,
- dataset mode (`dense` vs `summary`),
- `user_prompt`, `emit_norm`, `json_format`,
- system-prompt surfaces,
- `global_max_length`, template max-length surfaces, and
  `train_args.max_model_len` fallback when used,
- coord-token configuration,
- `custom.object_ordering`,
- `custom.object_field_order`,
- `custom.use_summary`,
- offline image-budget invariants (`custom.offline_max_pixels` or equivalent),
- cache schema version / artifact format version,
- any other resolved field that changes `template.encode(...)` output.

The cache key must be based on `base_idx`, not requested dataloader index, so
epoch-local ordering can vary without invalidating stored payloads.

## Artifact Layout

Resolved cache directory:
- `<resolved_root>/<fingerprint>/`

Required contents:
- `manifest.json`
- one or more deterministic shard files holding encoded sample payloads keyed by
  `base_idx`

`manifest.json` must record at least:
- fingerprint and artifact format version,
- resolved cache root and cache dir,
- dataset source identity,
- sample count,
- shard inventory,
- preserved payload keys,
- build status (`building`, `complete`, or failed state),
- build timing/provenance,
- authoring config snapshot for the cache namespace,
- any bypass/ineligibility reason when applicable.

Cross-run reuse:
- The default root under `training.output_dir` is run-scoped and safe.
- Explicit `root_dir` may point at a shared location; reuse across runs is
  allowed only when the full fingerprint matches.

## Build And Reuse Lifecycle

Build mode for v1 is eager, single-writer materialization before normal worker
fetch begins.

Lifecycle:
1. Resolve config and eligibility.
2. Resolve root and fingerprint.
3. If a complete manifest already exists, open the cache in reuse mode.
4. Otherwise, one writer builds the cache eagerly from base records.
5. Non-writer ranks/processes wait for a complete manifest up to
   `wait_timeout_s`.
6. Writer publishes the manifest atomically only after all shard files are
   complete.

Required safety rules:
- No rank/process may read partial shards as a valid cache hit.
- Timeout must fail fast with actionable diagnostics.
- A failed or interrupted build must not masquerade as a reusable cache hit.

## Integration Boundaries

Dataset path:
- `BaseCaptionDataset.__getitem__` should continue mapping requested index to
  `base_idx` via the active epoch permutation.
- When cache reuse is active, the fetch path should load the encoded payload by
  `base_idx` and avoid `_encode_record(...)` in the hot path.
- The returned mapping must still include the same downstream keys currently
  consumed by packing and trainers.

Preserved payload contract:
- `input_ids`, `labels`, and `length`
- multimodal fields required by collators/templates
- `messages`
- `assistant_payload`
- `objects`
- `metadata`
- `sample_id`, `dataset`, and `base_idx`

Stage-1 compatibility:
- Static pack planning should be able to source deterministic per-sample lengths
  from the encoded cache when active.
- The cache must not change existing pack-plan semantics, DDP alignment, or
  effective-batch behavior.

Stage-2 compatibility:
- Raw sample fetch for `stage2_two_channel` must keep the same trainer-visible
  sample mapping and metadata.
- Rollout matching, target construction, and Channel-B metrics remain unchanged.

## Operator Observability

V1 observability lives in startup logs and run artifacts, not in new canonical
per-step metrics.

Required outputs:
- startup log line that states `built`, `reused`, or `bypassed`,
- startup log line with fingerprint, resolved root, and resolved cache dir,
- explicit bypass reason when `ineligible_policy: bypass` is used,
- run-artifact provenance under `training.output_dir` so operators can recover
  the same information after startup,
- cache-local `manifest.json` with build timing and shard inventory.

Preferred run-artifact projection:
- record an `encoded_sample_cache` block in `run_metadata.json` with status,
  fingerprint, root, cache dir, policy, and bypass reason if present.

## Verification Plan

Minimum verification surfaces:
- strict config tests for the new namespace and unknown keys,
- eligibility tests for accepted vs rejected dataset modes,
- cache reuse/invalidation tests keyed by fingerprint changes,
- single-writer / follower-wait tests,
- Stage-1 static packing compatibility checks,
- Stage-2 raw fetch compatibility checks,
- artifact/log assertions for build, reuse, and bypass provenance.
