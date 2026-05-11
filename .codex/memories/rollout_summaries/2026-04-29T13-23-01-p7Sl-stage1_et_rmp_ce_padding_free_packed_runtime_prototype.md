thread_id: 019dd968-403c-7f62-a13e-21193ee3aced
updated_at: 2026-04-29T15:38:00+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/29/rollout-2026-04-29T13-23-01-019dd968-403c-7f62-a13e-21193ee3aced.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 ET-RMP-CE optimization investigation shifted from smart batching toward a new padding-free packed-row runtime, but the work was still in-progress when the rollout ended.

Rollout context: The user’s goal was to improve GPU memory utilization and throughput for the Stage-1 ET-RMP-CE support-reweighting experiment while preserving the research objective. The relevant production config was `configs/stage1/set_continuation/rmp_ce.yaml`; the memstress smoke was `configs/stage1/set_continuation/smoke/rmp_ce_memstress.yaml`. The key constraint was that the ET-RMP objective should remain central, with the key model metric `rmp/valid_child_mass_mean` and the infrastructure symptom of underfed ranks / low memory utilization. The user later clarified: “Please manage to control and reduce the forward propagation and try to pack everything into fewer forward propagation.” They then further clarified: “Try to use padding-free packing, not batching.”

## Task 1: Investigate current Stage-1 ET-RMP-CE stack and identify optimization levers
Outcome: partial

Preference signals:
- The user said the goal is to “increase training throughput and memory utilization without obscuring the ET-RMP-CE support-reweighting experiment’s interpretation” -> future work should preserve objective semantics while focusing on infrastructure/runtime efficiency.
- The user said “Please manage to control and reduce the forward propagation and try to pack everything into fewer forward propagation.” -> the user prefers reducing forward-call count, not just increasing batch size in the existing runtime.
- The user then corrected the direction to “Try to use padding-free packing, not batching.” -> future agents should prioritize padding-free packed-row execution over plain batching when asked to improve utilization for this path.

Key steps:
- Read the Stage-1 docs and configs: `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE1_ET_RMP_CE.md`, `docs/data/PACKING.md`, `configs/stage1/set_continuation/production.yaml`, `configs/stage1/set_continuation/rmp_ce.yaml`, and `configs/stage1/set_continuation/smoke/rmp_ce_memstress.yaml`.
- Confirmed the current checked-in ET-RMP runtime is `smart_batched_exact`, with `branch_batching.max_branch_rows: 8`, `ddp_sync.candidate_padding: none`, and `logits.mode: supervised_suffix`.
- Read the benchmark probe note `progress/benchmarks/2026-04-28_stage1_mp_branch_runtime_packing_probe.md`, which stated `smart_batched_exact` was faster than the packed-varlen experiments in that probe and that packed-varlen was still experimental.
- Inspected the Stage-1 branch batching and full-suffix scoring flow with Serena symbol navigation, especially `src/trainers/stage1_set_continuation/trainer.py`, `branch_batcher.py`, `full_suffix.py`, and `branch_scorer.py`.
- Found that current ET-RMP full-suffix scoring is row-based, uses `smart_batched_exact`, and explicitly rejects the ordinary Stage-1 dataset packing surface because branch/prefix sampling happens inside `compute_loss`.

Failures and how to do differently:
- The existing `smart_batched_exact` branch-batching path is not enough to satisfy the user’s latest direction; it still groups rows rather than performing padding-free packed forwarding.
- The current code path `score_full_suffix_batch_retained(...)` only supports a single trailing `logits_to_keep` crop and cannot safely represent multiple packed suffix windows; a new packing-aware scoring path was needed instead of forcing the existing retained-logit crop path.

Reusable knowledge:
- `stage1_set_continuation` currently uses one full-suffix row per sample and does not allow `training.packing: true` / `training.eval_packing: true` in v1.
- `smart_batched_exact` currently returns telemetry including `mp/smart_batched_branch_forwards`, `mp/branch_batch_count`, `mp/branch_batch_rows_mean`, `mp/branch_batch_rows_max`, `mp/branch_batch_tokens_mean`, `mp/branch_batch_tokens_max`, and `mp/branch_batch_padding_fraction`.
- The `branch_batching` config exists, but in the current implementation `min_fill_ratio` is effectively reserved for later adaptive scheduling and the realized behavior is still row batching, not true packed attention.
- `docs/data/PACKING.md` and the benchmark probe both warn that packed-varlen branch scoring is experimental and should not be assumed better than `smart_batched_exact` without a parity/profiling gate.

References:
- [1] `configs/stage1/set_continuation/rmp_ce.yaml`: `train_forward.branch_runtime.mode: smart_batched_exact`, `branch_batching.max_branch_rows: 8`, `ddp_sync.candidate_padding: none`, `budget_policy.enabled: false`.
- [2] `docs/training/STAGE1_ET_RMP_CE.md`: ET-RMP contract, current metrics, and note that the checked-in production runtime uses `smart_batched_exact`.
- [3] `progress/benchmarks/2026-04-28_stage1_mp_branch_runtime_packing_probe.md`: `smart_batched_exact` fastest in the probe; packed-varlen slower and still experimental.
- [4] `src/trainers/stage1_set_continuation/branch_batcher.py`: `plan_smart_branch_batches(...)` groups rows by volume/rows caps, not by true padding-free packing.
- [5] `src/trainers/stage1_set_continuation/full_suffix.py`: `score_full_suffix_batch_retained(...)` only applies one global trailing crop via `logits_to_keep`.

## Task 2: Prototype a padding-free packed full-suffix runtime and wire schema/tests for it
Outcome: partial

Preference signals:
- The user’s correction “Try to use padding-free packing, not batching” suggests that a new runtime mode is preferable to merely tuning `max_branch_rows` or other batching caps.
- The user’s earlier request to reduce forward propagation implies a design that concatenates multiple rows into one model forward and then unpacks the losses.

Key steps:
- Added a new failing test first in `tests/test_stage1_set_continuation_full_suffix.py` that demanded a single packed forward for two full-suffix rows, with `cu_seq_lens_q/k`, `position_ids`, `text_position_ids`, no `attention_mask`, and numerical equivalence to serial scoring.
- Implemented a new helper in `src/trainers/stage1_set_continuation/full_suffix.py` named `score_full_suffix_batch_padding_free_packed(...)`.
- The new helper concatenates multiple rows into one packed sequence, builds `cu_seq_lens_q`, `cu_seq_lens_k`, `text_position_ids`, and 3-row Qwen-style `position_ids`, and computes each row’s loss by offsetting the full-suffix steps back into the packed coordinate space.
- The new helper intentionally requires `logits_mode == "full"` because the existing trailing `logits_to_keep` crop is not safe for multiple packed suffix windows in one concatenated sequence.
- Exported the new helper via `__all__` in `full_suffix.py`.
- Verified the new test passed: `tests/test_stage1_set_continuation_full_suffix.py::test_padding_free_packed_full_suffix_scores_rows_in_one_forward_without_padding` passed after the helper landed.

Failures and how to do differently:
- A first attempt to import `score_full_suffix_batch_padding_free_packed` failed as expected because the symbol did not yet exist; this confirmed the test was effectively red before implementation.
- The packed path initially targeted the full-suffix scorer only; it still needed trainer integration and config exposure to become usable.
- The packed scorer currently works at the helper level, but it does not yet integrate with the trainer’s runtime-mode telemetry or config validation by itself.

Reusable knowledge:
- For full-suffix ET-RMP, a true packing path must not rely on `logits_to_keep`; it should concatenate rows and score each row by offsets over one packed logits tensor.
- The packed helper can safely pack row-local `input_ids`, synthesize `text_position_ids` as concatenated per-row resets, and produce Qwen-compatible `position_ids` shaped `[3, 1, T]` for the packed sequence.
- The test fixture confirmed the expected shape/contract for packed full-suffix scoring: one packed forward with `input_ids.shape == (1, total_len)`, `cu_seq_lens_q/k` matching segment boundaries, and one `pack_num_samples` entry equal to the number of packed rows.

References:
- [1] Added helper in `src/trainers/stage1_set_continuation/full_suffix.py`: `score_full_suffix_batch_padding_free_packed(...)`.
- [2] Added test in `tests/test_stage1_set_continuation_full_suffix.py` asserting one packed forward and equivalence to serial loss.
- [3] Passing verification: `python -m pytest tests/test_stage1_set_continuation_full_suffix.py::test_padding_free_packed_full_suffix_scores_rows_in_one_forward_without_padding -q` -> `1 passed`.

## Task 3: Expose a new `padding_free_packed` Stage-1 runtime mode in schema and trainer smoke tests
Outcome: partial

Preference signals:
- The user explicitly rejected “batching” in favor of “padding-free packing,” so the runtime mode should be explicit in config rather than hidden behind the old batching label.

Key steps:
- Added `padding_free_packed` to `Stage1SetContinuationBranchRuntimeConfig` in `src/config/schema.py`.
- Added a schema guard requiring `train_forward.logits.mode == "full"` when `branch_runtime.mode == "padding_free_packed"`.
- Reused the existing DDP/candidate-padding guard logic and extended the runtime-mode validation set so the new mode is recognized.
- Added config tests in `tests/test_stage1_set_continuation_train_forward_config.py` for:
  - accepting `branch_runtime.mode: padding_free_packed` with `logits.mode: full` and `candidate_padding: none`
  - rejecting `padding_free_packed` when `logits.mode: supervised_suffix`
- These config tests passed after the schema update.
- Added a smoke test in `tests/test_stage1_set_continuation_trainer_smoke.py` expecting `padding_free_packed` to run ET-RMP full-suffix scoring as one concatenated forward and emit packing telemetry.

Failures and how to do differently:
- The trainer smoke test failed because metrics code had not yet been updated for the new runtime mode: `mp/branch_runtime_mode has no numeric code for value: 'padding_free_packed'`.
- This means the config/runtime surface needs a follow-up update in `src/trainers/stage1_set_continuation/metrics.py` and likely in the trainer’s metric emission paths before the new runtime can be used end-to-end.
- The work was not completed to the point of wiring `padding_free_packed` into `Stage1SetContinuationTrainer._process_full_suffix_batch(...)` and the runtime telemetry payloads.

Reusable knowledge:
- `Stage1SetContinuationBranchRuntimeConfig` currently enumerates `retained_graph`, `checkpointed_exact`, `smart_batched_exact`, and now `padding_free_packed`.
- For the new mode, config validation should enforce `logits.mode: full`; the current retained-logit crop path is not safe for multiple packed suffix windows.
- `tests/test_stage1_set_continuation_train_forward_config.py` now captures the config contract for the new mode and is the right place to extend if more guardrails are added.

References:
- [1] `src/config/schema.py`: `Stage1SetContinuationBranchRuntimeConfig` and the new `padding_free_packed` guard.
- [2] `tests/test_stage1_set_continuation_train_forward_config.py`: new config acceptance/rejection tests for the packed runtime.
- [3] `tests/test_stage1_set_continuation_trainer_smoke.py::test_entry_trie_rmp_ce_padding_free_packed_uses_single_concat_forward` failed at metric coding, not at the forward path.
- [4] Exact failure snippet: `ValueError: mp/branch_runtime_mode has no numeric code for value: 'padding_free_packed'`.

## Task 4: Decide the likely next implementation direction for throughput/memory gains
Outcome: uncertain

Preference signals:
- The user is steering away from “just batching” and toward true “padding-free packing,” which suggests future agents should prioritize a packed-row runtime/attention contract over row-grouping heuristics.

Key steps:
- The code exploration showed the ET-RMP path already has a clean split between retained-graph / smart-batched scoring and the new experimental packed helper.
- The next necessary changes are likely to be:
  - add `padding_free_packed` to metric code maps in `src/trainers/stage1_set_continuation/metrics.py`
  - wire trainer selection to call the packed helper
  - ensure model inputs satisfy packing-position contract in the forward path
  - decide whether `training.packing`/`eval_packing` remain rejected for ET-RMP or whether a dedicated trainer-side packed path should supersede them
  - add smoke/provenance telemetry proving fewer forwards per step and acceptable memory use

Failures and how to do differently:
- The investigation initially focused on `smart_batched_exact`, but the user’s correction made clear that true padding-free packing was the desired direction.
- A future pass should start by mapping the metric/trainer runtime switch points before expanding the packed helper further.

Reusable knowledge:
- `score_full_suffix_batch_padding_free_packed(...)` is now a concrete starting point for packed-row ET-RMP runtime work.
- The existing config/test surface now recognizes the new mode, so the remaining work is mostly runtime dispatch and metric plumbing.

References:
- [1] `src/trainers/stage1_set_continuation/full_suffix.py` new packed scorer helper.
- [2] `src/config/schema.py` new runtime mode validation.
- [3] `tests/test_stage1_set_continuation_trainer_smoke.py` current failure on metric coding indicates the next missing plumbing layer.

