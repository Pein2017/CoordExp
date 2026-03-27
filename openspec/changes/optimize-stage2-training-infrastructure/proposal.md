## Why

CoordExp's active training stack is now constrained more by infrastructure friction than by missing research features.

The strongest pressure is concentrated in the production Stage-2 path and the
shared training surfaces it depends on:

- DDP coordination and metric reduction are split across multiple trainer layers, which increases skew sensitivity, duplicate logic, and deadlock risk.
- Python-side data preparation remains expensive even when the model/runtime are healthy, especially around static packing, repeated encoding, and worker-local cache growth.
- Reproducibility artifacts are helpful but do not yet describe the exact executed runtime or a clear restartability contract.
- Operator observability is useful but still relies on heuristic metric semantics, mixed logger ownership, and heavier-than-needed DDP eval transport.

This change is not intended to reopen the full archived runtime-architecture program.
Instead, it is a narrower optimization and operability follow-on:

- keep public training semantics stable,
- improve throughput and DDP scaling efficiency,
- tighten fault tolerance and reproducibility,
- and make the active training path easier to reason about.

## What Changes

- Introduce a focused training-infrastructure optimization program for the active Stage-2 and shared training runtime surfaces.
- Unify DDP step coordination behind one explicit coordination owner for:
  - bounded phase barriers,
  - rank-symmetric failure propagation,
  - and per-step metric reduction.
- Replace Stage-2 metric reduction heuristics with an explicit metric contract so local aggregation, DDP reduction, and carry-forward snapshot behavior are defined in one place.
- Optimize the shared training data hot path by splitting sample preparation from token encoding, so static packing and cache-backed paths can reuse prepared or encoded work instead of recomputing it.
  - This change does **not** replace the current trainer-owned dynamic post-rollout packing contract used by live Stage-2 runs.
- Make threaded packing precompute deterministic and safe when datasets use mutable templates or stochastic preprocessors.
- Bound encoded-sample cache residency under persistent workers so worker RSS does not grow without limit across an epoch.
- Improve training JSONL diagnostics so malformed data fails with line-aware context rather than generic parse errors.
- Emit executed-runtime artifacts in addition to authored-config artifacts, including:
  - `effective_runtime.json`
  - `pipeline_manifest.json`
  - and stronger data-source provenance.
- Split checkpoint intent into explicit modes:
  - `artifact_only`
  - `restartable`
- Make restart-sensitive trainer and callback state explicit where it affects future steps or save behavior.
- Make training-time detection eval and related observability more DDP-efficient by preferring rank-0-owned or scalar-first aggregation patterns where full payload gathering is not required.
- Emit explicit health signals when best-effort diagnostics disable themselves, instead of relying only on one-time warnings.
- Preserve config-first operation and avoid new ad-hoc CLI flags by default.

## Recommended First Version

The first implementation slice should stay narrow and high-leverage:

- unify Stage-2 DDP coordination and metric reduction ownership,
- add an explicit `MetricSpec` contract,
- make the snapshot namespace explicit without changing live sparse-rollout semantics,
- improve rank-0 eval transport,
- and add executed-runtime plus data-source provenance artifacts for the actual launched session.

That first slice should **not** yet:

- change Stage-2 clean-prefix or teacher-forcing semantics,
- lift the current single-server rollout topology restriction,
- introduce `restartable` checkpoint mode before its artifact and preflight contract is fully specified,
- or redesign the training entrypoint around a new public CLI.

## Assumptions

- Current DDP pain is caused more by coordination ownership and rank skew than by the underlying all-reduce backend itself.
- Current data throughput is constrained more by Python-side sample preparation and repeated encoding than by raw file I/O alone.
- Explicit executed-runtime and checkpoint-mode contracts will reduce operator confusion without forcing large research-surface changes.
- The existing archived runtime refactor already covered broad modularization; this change should focus on optimization and operability gaps that remain visible in daily training.

## Non-Blocking Follow-Ups

- Re-enable or redesign Channel-B async overlap under DDP only after synchronization ownership becomes explicit and testable.
- Expand rollout launcher/preflight to full multi-server parity only in a follow-on change after coordination and runtime contracts are stable.
- Further reduce `src/sft.py` ownership through a typed `TrainingSessionPlan` or `LaunchPlan` once artifact and runtime-contract changes are settled.

## Risks To Validity

- A coordination refactor that changes reduction ordering or snapshot semantics could silently perturb dashboards if the metric contract is underspecified.
- A data-path optimization that reuses cached prepared samples could accidentally weaken prompt, geometry, or preprocessing invariants if the contract boundary is not explicit.
- A restartability mode that is only partially implemented would be more harmful than keeping the current artifact-only default.
- A logging/eval optimization that changes payload shape without explicit contract updates could break downstream analysis or review tooling.

## Required Evidence

- Proof that Stage-2 performs one coherent distributed step reduction per train step after refactoring.
- Proof that DDP phase-barrier behavior remains bounded and rank-symmetric under skew or injected failure.
- Throughput evidence for the data-path changes:
  - first-batch latency,
  - plan-build wall time,
  - tokens/sec,
  - and worker RSS.
- Artifact evidence that `effective_runtime.json`, `pipeline_manifest.json`, and data-source provenance describe the executed session accurately.
- Resume evidence showing that `restartable` mode either resumes correctly or fails early and clearly when the checkpoint is incomplete.

## Capabilities

### Modified Capabilities

- `stage2-ab-training`: modify distributed coordination ownership, metric projection ownership, and rank-aware observability behavior without changing Stage-2 learning semantics.
- `rollout-matching-sft`: modify shared runtime coordination and train-time eval transport behavior without changing rollout-learning semantics.
- `packing-dataset`: modify packing/runtime boundaries so static packing can reuse prepared or encoded sample work safely.
- `encoded-training-cache`: modify cache residency and reuse behavior under persistent workers.
- `trainer-metrics-components`: modify metric ownership so reduction semantics, snapshot behavior, and health signals are explicit and testable.
- `detection-evaluator`: modify train-time DDP aggregation and ownership patterns without changing final metric meaning.

## Impact

- Primary code surfaces are expected to include:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/executors.py`
  - `src/trainers/stage2_two_channel/coordination.py`
  - `src/trainers/stage2_rollout_aligned.py`
  - `src/trainers/rollout_aligned_evaluator.py`
  - `src/datasets/dense_caption.py`
  - `src/datasets/wrappers/packed_caption.py`
  - `src/datasets/encoded_sample_cache.py`
  - `src/common/io.py`
  - `src/utils/logger.py`
  - `src/utils/run_manifest.py`
  - `src/bootstrap/run_metadata.py`
  - `src/callbacks/save_delay_callback.py`
- Main verification surfaces are expected to include:
  - `tests/test_stage2_ab_ddp_phase_monitor_disable.py`
  - `tests/test_stage2_pending_metrics_aggregation.py`
  - `tests/test_stage2_ab_disable_average_tokens_across_devices.py`
  - `tests/test_stage2_rollout_aligned.py`
  - `tests/test_stage1_static_packing_runtime_config.py`
  - `tests/test_encoded_sample_cache.py`
  - `tests/test_training_config_strict_unknown_keys.py`
  - `tests/test_run_manifest_files.py`
  - `tests/test_run_metadata_file.py`
  - `tests/test_detection_eval_output_parity.py`
  - `tests/test_trainer_metrics_payload_contract.py`
- Operator-facing behavior should remain stable in the initial slices except where this change explicitly introduces clearer provenance artifacts, snapshot namespaces, or health-signal outputs.
