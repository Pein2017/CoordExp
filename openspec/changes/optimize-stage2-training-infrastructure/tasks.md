## Workstream 0. Baseline Evidence And Contract Freeze

- [ ] 0.1 Record the current optimization baseline for the active Stage-2 path, including:
  - first-batch latency
  - step wall time
  - `rollout/gen_tokens_per_s`
  - `time/post_rollout_pack_s`
  - worker RSS
  - and `monitor_dumps/ddp_phase_trace/*` behavior on a short representative run
- Note: this remains tied to the deferred representative Stage-2 smoke run because
  the baseline requires a live runtime trace rather than unit-only evidence.
- [x] 0.2 Document the stable surfaces this change must preserve in its first slices:
  - Stage-2 clean-prefix and teacher-forcing semantics
  - geometry and object-ordering invariants
  - prompt/chat-template behavior
  - existing operator-facing output artifacts
- Note: preserved by the current change artifacts plus the regression bundle
  covering geometry, prompt, and artifact contracts.
- [x] 0.3 Collect a narrow verification baseline before implementation using:
  - `conda run -n ms python -m pytest -q tests/test_coord_geometry_invariants.py`
  - `conda run -n ms python -m pytest -q tests/test_chat_template_regression.py`
  - `conda run -n ms python -m pytest -q tests/test_prompt_variants.py`
  - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_stage2_metrics.py`
  - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_multiprocess.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_ddp_phase_monitor_disable.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_pending_metrics_aggregation.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_disable_average_tokens_across_devices.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
  - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py`
  - `conda run -n ms python -m pytest -q tests/test_encoded_sample_cache.py`
  - `conda run -n ms python -m pytest -q tests/test_common_io_jsonl.py`
  - `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`
  - `conda run -n ms python -m pytest -q tests/test_run_metadata_file.py`
  - `conda run -n ms python -m pytest -q tests/test_dependency_provenance.py`
  - `conda run -n ms python -m pytest -q tests/test_launcher_metadata_env.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_trainer_metrics_payload_contract.py`
- Note: the verification suite was collected incrementally during implementation
  and rerun on the final worktree head as part of Workstream 5.

## Workstream 1. DDP Coordination And Metric Contract

- [x] 1.1 Introduce a shared distributed-step coordination seam that owns:
  - bounded phase barriers
  - rank-symmetric failure propagation
  - readiness checks
  - and per-step metric reduction entrypoints
- [x] 1.2 Migrate Stage-2 pending-log reduction and rollout-aligned train-log reduction to that shared coordination seam.
- [x] 1.3 Replace metric-name reduction heuristics with an explicit `MetricSpec`-style contract shared by:
  - local pending aggregation
  - global DDP aggregation
  - and carry-forward snapshot policy
- [x] 1.4 Introduce an explicit snapshot namespace or equivalent documented contract so operator-facing logs distinguish current-step values from last-seen snapshots while the live `rollout/*` namespace keeps its existing sparse-emission semantics.
- [x] 1.5 Validate that the active Stage-2 path performs exactly one coherent distributed reduction per step and uses the same barrier policy across Channel-A and Channel-B.
- [x] 1.6 Update `docs/training/METRICS.md` and `tests/test_stage2_pending_metrics_aggregation.py` so snapshot behavior and live metric behavior are unambiguous.

## Workstream 2. Shared Data Path Throughput

- [x] 2.1 Preserve the current trainer-owned dynamic post-rollout packing contract for live Stage-2 runs; treat this workstream as a shared dataset/sample-fetch optimization rather than a replacement for Stage-2 packing.
- [x] 2.2 Split the dense-caption sample path into explicit internal phases for:
  - record preparation
  - prepared-record rendering payloads
  - token encoding
  - final example shaping
- [x] 2.3 Update static packing so warm plan-build or cache-backed flows can reuse prepared or encoded sample work instead of re-entering the full dataset path.
- [x] 2.4 Define explicit eligibility and fingerprinting rules for any prepared-record sidecar used before encode; these rules must be at least as strict as the encoded-sample-cache determinism boundary.
- [x] 2.5 Make length precompute deterministic and concurrency-safe:
  - use immutable helper paths where possible
  - or fall back to serial/process-isolated precompute when dataset state is mutable
- [x] 2.6 Add an explicit cache-residency bound for encoded-sample shards under persistent workers.
- [x] 2.7 Upgrade training JSONL ingestion to line-aware diagnostics with path and snippet context.
- [ ] 2.8 Validate throughput and memory changes with:
  - first-batch latency comparison
  - rank-0 static-pack plan-build wall time
  - tokens/sec
  - worker RSS across an epoch
  - and unchanged sample payload correctness
  - geometry/chat-template regression coverage
  - and an encoded-cache regression that exercises persistent-worker multi-shard eviction or bounded residency behavior
- Note: correctness, geometry/chat-template, and bounded-residency regression
  coverage are in place; the throughput and RSS comparisons remain grouped with
  the deferred representative runtime smoke evidence.

## Workstream 3. Executed Runtime Artifacts And Restartability

- [x] 3.1 Emit an `effective_runtime.json` artifact that reflects the executed runtime after bootstrap or launcher mutation rather than only the authored config.
- [x] 3.2 Persist `pipeline_manifest.json` as a first-class artifact for Stage-2 runs.
- [x] 3.3 Persist unconditional train and eval data-source provenance sidecars, including stable identity fields and optional content digests for supported sources.
- [x] 3.4 Introduce explicit checkpoint modes with `artifact_only` as the compatibility-preserving default and `restartable` as an explicit opt-in mode.
- [x] 3.5 Define the minimum `restartable` contract, including:
  - model weights
  - optimizer state
  - scheduler state
  - RNG state
  - trainer state sufficient to restore `global_step`
  - repo-owned future-affecting Stage-2 runtime state sidecars
  - and restart-sensitive callback state or a recompute-safe equivalent
- [x] 3.6 Add operator-facing resume preflight rules so incomplete or incompatible restartable checkpoints fail early and clearly.
- [ ] 3.7 Validate with:
  - artifact parity tests
  - a kill-and-resume drill from a non-empty post-rollout state
  - a hard-failure test when `restartable` mode is used on an incomplete checkpoint
  - and targeted checks for resumed schedule continuity from restored `global_step`
- Note: helper-level preflight and schedule-continuity tests are present, but the
  end-to-end Stage-2 kill-and-resume drill remains open and is grouped with the
  deferred representative runtime smoke evidence.

## Workstream 4. Observability And Train-Time Eval Efficiency

- [x] 4.1 Make train-time detection eval DDP transport scalar-first or shard-first so only rank 0 owns the final heavy aggregation and COCO execution.
- [x] 4.2 Preserve the current train-time eval summary contract while doing so:
  - keep rank-0-owned final scoring
  - keep `eval/detection/mAP` parity
  - and do not broaden this workstream into offline evaluator artifact parity
- [x] 4.3 Emit persistent health signals when best-effort diagnostics disable themselves, instead of only warning once in logs.
- [ ] 4.4 Validate with:
  - DDP eval smoke checks
  - unchanged train-time summary metric parity
  - and health-signal assertions
- Note: unit-level parity and health-signal assertions are covered; the
  representative DDP eval smoke remains open and is grouped with the deferred
  representative runtime smoke evidence.

## Workstream 5. Final Verification

- [x] 5.1 Run the targeted verification bundle after the last optimization slice:
  - `conda run -n ms python -m pytest -q tests/test_coord_geometry_invariants.py`
  - `conda run -n ms python -m pytest -q tests/test_chat_template_regression.py`
  - `conda run -n ms python -m pytest -q tests/test_prompt_variants.py`
  - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_stage2_metrics.py`
  - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_multiprocess.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_ddp_phase_monitor_disable.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_pending_metrics_aggregation.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_disable_average_tokens_across_devices.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_two_channel_training.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_rollout_aligned.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
  - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py`
  - `conda run -n ms python -m pytest -q tests/test_encoded_sample_cache.py`
  - `conda run -n ms python -m pytest -q tests/test_common_io_jsonl.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
  - `conda run -n ms python -m pytest -q tests/test_run_manifest_files.py`
  - `conda run -n ms python -m pytest -q tests/test_run_metadata_file.py`
  - `conda run -n ms python -m pytest -q tests/test_dependency_provenance.py`
  - `conda run -n ms python -m pytest -q tests/test_launcher_metadata_env.py`
  - `conda run -n ms python -m pytest -q tests/test_detection_eval_output_parity.py`
  - `conda run -n ms python -m pytest -q tests/test_trainer_metrics_payload_contract.py`
- Result on current worktree head: `455 passed, 2 skipped`.
- [ ] 5.2 Run a short representative training smoke with artifact inspection for:
  - DDP phase traces
  - executed-runtime artifacts
  - and snapshot metric readability
- Note: deferred in the current session because no GPUs are available for a representative Stage-2 smoke run.
- [x] 5.3 Confirm that any stable contract change discovered during implementation is captured by a follow-on OpenSpec delta before landing that slice.
- Note: the stable contract changes uncovered during implementation were folded
  back into this active OpenSpec change before landing, so no extra follow-on
  delta is required for the current slice set.
