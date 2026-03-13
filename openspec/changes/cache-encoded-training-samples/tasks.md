## 1. Proposal Alignment

- [x] 1.1 Lock the change as a standalone `encoded-training-cache` capability
  instead of implicitly modifying unrelated spec contracts.
- [x] 1.2 Lock the canonical YAML namespace
  `training.encoded_sample_cache.*`, default root, and default ineligible
  policy.
- [x] 1.3 Enumerate the exact cache-eligible and cache-rejected dataset modes
  for v1.

## 2. Design

- [x] 2.1 Define the fingerprint inputs, `base_idx` lookup identity, and cache
  invalidation rules.
- [x] 2.2 Define the artifact layout (`manifest.json` plus deterministic
  shard files) and preserved payload keys.
- [x] 2.3 Define single-writer build, follower wait, timeout, and partial-build
  safety rules.
- [x] 2.4 Define operator-visible outputs for `built`, `reused`, and
  `bypassed` startup states.

## 3. Implementation

- [x] 3.1 Add typed config support for `training.encoded_sample_cache.*` with
  strict unknown-key rejection.
- [x] 3.2 Add deterministic eligibility checks and explicit `error` vs
  `bypass` handling.
- [x] 3.3 Add eager cache build/reuse helpers with manifest + shard handling.
- [x] 3.4 Integrate cache lookup into `BaseCaptionDataset` so hot-path fetches
  load by `base_idx` instead of re-running `_encode_record(...)`.
- [x] 3.5 Keep Stage-1 static packing compatible by sourcing deterministic
  lengths/payloads from the cache when enabled, without changing packing
  semantics.
- [x] 3.6 Keep Stage-2 `stage2_two_channel` raw sample fetch and rollout-aware
  trainer behavior contract-compatible.
- [x] 3.7 Emit startup provenance and run-artifact metadata for cache build,
  reuse, and bypass states.

## 4. Validation

- [x] 4.1 Add strict-config tests for the new YAML namespace, defaults, and
  unknown-key rejection.
- [x] 4.2 Add cache eligibility tests covering accepted deterministic runs and
  rejected augmentation/random-ordering/fusion-style paths.
- [x] 4.3 Add cache reuse and invalidation tests for fingerprint-affecting
  config changes.
- [x] 4.4 Add rank-safe build/reuse tests for single-writer and follower wait
  behavior.
- [x] 4.5 Add Stage-1 static packing compatibility coverage, reusing
  `tests/test_stage1_static_packing_runtime_config.py` where appropriate.
- [x] 4.6 Add Stage-2 raw fetch compatibility coverage, reusing
  `tests/test_stage2_ab_training.py` where appropriate.
- [x] 4.7 Add artifact/log assertions proving operators can distinguish
  `built`, `reused`, and `bypassed` startup states.
- [ ] 4.8 Run targeted training-smoke validation on the intended Stage-2 prod
  path after the cache contract is implemented.
