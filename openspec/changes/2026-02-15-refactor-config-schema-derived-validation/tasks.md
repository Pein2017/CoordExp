## 1. Schema-derived strict parsing foundation

- [x] 1.1 Add a reusable strict dataclass parser utility under `src/config/*` that derives accepted keys from typed fields and reports unknown keys with full dotted paths.
- [x] 1.2 Support nested dataclass parsing for mappings/lists used by Stage-2 config trees, including clear type errors when payload shapes mismatch schema.
- [x] 1.3 Add unit coverage for parser core behavior:
  - accepted keys pass,
  - unknown keys fail with dotted path,
  - nested unknown keys fail with nested dotted path.

## 2. Migrate top-level training config sections

- [x] 2.1 Refactor `TrainingConfig.from_mapping` path to parse all top-level sections via the strict parser contract:
  - `model`, `quantization`, `template`, `data`, `tuner`, `training`, `rlhf`, `custom`, `debug`, `stage2_ab`, `rollout_matching`, `deepspeed`, `global_max_length`, `extra`.
- [x] 2.2 Preserve and reattach semantic invariants (required fields, value ranges, cross-field constraints) after strict shape parse.
- [x] 2.3 Ensure removed legacy rollout paths remain explicit fail-fast (`custom.extra.rollout_matching.*`).
- [x] 2.4 Preserve intentional extension-bucket behavior:
  - `custom.extra` remains accepted as a narrow escape-hatch mapping,
  - extension-bucket acceptance does not weaken strictness for canonical grouped sections.
- [x] 2.5 Enforce top-level `extra` policy:
  - top-level `extra` is not an author escape hatch,
  - any top-level `extra:` presence (including `{}`) fails fast with actionable guidance.
- [x] 2.6 Preserve Stage-2 trainer-variant contract:
  - `custom.trainer_variant=stage2_ab_training` still requires top-level `rollout_matching`,
  - missing top-level `rollout_matching` fails fast with actionable error.
- [x] 2.7 Keep rollout normalization/injection wiring in `src/sft.py` aligned with strict parser migration (no ownership drift).
- [x] 2.8 Remove legacy rollout server paired-list shape support (`vllm.server.base_url` + `vllm.server.group_port`) and enforce `vllm.server.servers[]` only.

## 3. De-duplicate runtime rollout shape checks

- [x] 3.1 Add parity gate checks first: prove loader-level schema catches representative static rollout unknown-key cases currently guarded in trainer runtime.
- [x] 3.2 In `src/trainers/rollout_matching_sft.py`, keep runtime-only invariants and remove duplicated schema-shape unknown-key enumeration only after 3.1 passes.
- [x] 3.3 Ensure rollout runtime still validates execution-dependent constraints (backend mode compatibility, runtime numeric bounds, etc.) with actionable errors.
- [x] 3.4 Keep `rollout_matching.decode_batch_size` source-of-truth behavior unchanged.

## 4. Verification matrix

- [x] 4.1 Positive: load canonical Stage-2 profiles under `configs/stage2_ab/prod/*.yaml` and `configs/stage2_ab/smoke/*.yaml` with `ConfigLoader.load_training_config`.
- [x] 4.2 Negative: unknown-key fail-fast checks for each top-level section plus nested rollout structures (`rollout_matching.decoding`, `rollout_matching.monitor_dump`, `rollout_matching.vllm.server.servers[]`, `rollout_matching.vllm.sync`).
- [x] 4.3 Negative: legacy `custom.extra.rollout_matching.*` keys fail fast with migration guidance.
- [x] 4.4 Runtime/preflight sanity:
  - `scripts/train_stage2.sh` preflight still resolves canonical config through Python resolver,
  - preflight JSON output keeps required schema (`rollout_backend`, `vllm_mode`, `server_base_urls`) with correct types,
  - `ROLLOUT_CONTRACT_JSON` value is verified as newline-terminated single-line JSON,
  - invalid/missing JSON contract fields fail fast before launch,
  - `bash -n scripts/train_stage2.sh` passes.
- [x] 4.5 Extension-bucket check:
  - representative `custom.extra.<minor_key>` remains accepted,
  - representative unknown non-extension key under strict sections fails fast,
  - representative top-level `extra:` presence fails fast.
- [x] 4.6 Semantic/type-shape checks:
  - representative required-field/range/cross-field invariants remain enforced,
  - representative wrong-shape payloads fail at loader level with clear path-aware errors,
  - legacy `vllm.server.base_url/group_port` shape fails fast with migration guidance.
- [x] 4.7 Negative: removed Stage-2 Channel-B knobs fail at loader level (not trainer runtime):
  - `stage2_ab.channel_b.semantic_desc_gate` fails fast with dotted-path error,
  - `stage2_ab.channel_b.reordered_gt_sft` fails fast with dotted-path error,
  - `stage2_ab.channel_b.desc_ce_weight_matched` fails fast with dotted-path error.

## 5. Documentation and cleanup

- [x] 5.1 Document the new strict parser ownership and error format in code comments near loader entrypoints.
- [x] 5.2 Remove now-redundant manual allowlist blocks that become dead after migration.
- [x] 5.3 Record a short change-note table (before/after validation ownership) in this change artifacts.

## 6. Test migration updates (explicit targets)

- [x] 6.1 Update `tests/test_custom_extra_merge.py` to reflect strict policy:
  - unknown `custom.*` no longer auto-merges into `custom.extra`,
  - `custom.extra.rollout_matching.*` is fail-fast.
- [x] 6.2 Update `tests/test_stage2_ab_vllm_server_mode_smoke.py` fixtures to author rollout knobs only under canonical top-level `rollout_matching.*`.
- [x] 6.3 Update `tests/test_stage2_ab_vllm_server_mode_ab_mixed_diag.py` fixtures to author rollout knobs only under canonical top-level `rollout_matching.*`.
- [x] 6.4 Re-verify `tests/test_stage2_ab_config_contract.py` keeps `stage2_ab_training` top-level `rollout_matching` requirement after strict-parser migration (baseline landed in commit `23a5656`).
- [x] 6.5 Update `tests/test_vllm_server_chunking.py` expectations to align with strict removal of legacy paired-list rollout server shape.

## 7. Non-test config migration targets

- [x] 7.1 Migrate `configs/dlora/sft_coord_loss.yaml` unknown `custom.*` knobs into `custom.extra.*` (or promote to first-class schema fields if intentional/long-term).
- [x] 7.2 Migrate `configs/dlora/debug-sft_coord_loss.yaml` unknown `custom.*` knobs into `custom.extra.*` (or promote to first-class schema fields if intentional/long-term).
- [x] 7.3 Add sweep task: scan `configs/**/*.yaml` for unknown `custom.*` keys outside `custom.extra.*`, migrate them, and record migration outcomes in change notes.
- [x] 7.4 Migrate `configs/dlora/stage2_rollout_matching_ckpt3106.yaml` from `custom.extra.rollout_matching.*` to canonical top-level `rollout_matching.*`.
- [x] 7.5 Migrate `configs/dlora/stage2_rollout_matching_ckpt3106_server_3v1.yaml` from `custom.extra.rollout_matching.*` to canonical top-level `rollout_matching.*`.
