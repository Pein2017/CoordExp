## 1. Config Contract + Validation

- [x] 1.1 Update `src/config/schema.py` Stage2-AB typed config to remove `stage2_ab.channel_b.mode`, `stage2_ab.channel_b.rollouts_per_step`, and `stage2_ab.channel_b.rollout_decode_batch_size` (and any async sub-config), and ensure unknown keys fail fast with actionable errors.
- [x] 1.2 Add config validation for `custom.extra.rollout_matching.decode_batch_size` (int > 0) and fail fast if any removed legacy keys are present:
  - `custom.extra.rollout_matching.rollout_generate_batch_size`
  - `custom.extra.rollout_matching.rollout_infer_batch_size`
  - `custom.extra.rollout_matching.post_rollout_pack_scope`
- [x] 1.3 Stage2-AB only: enforce `training.effective_batch_size` is divisible by `per_device_train_batch_size × learner_world_size` (no ceil overshoot) and fail fast with guidance if not.

## 2. Single Step-Budgeted Channel-B Pathway

- [x] 2.1 Refactor `src/trainers/stage2_ab/scheduler.py` so Channel-B budgeting always uses `rollouts_per_step := training.effective_batch_size` and no longer depends on `stage2_ab.channel_b.mode` or `rollouts_per_step` overrides.
- [x] 2.2 Refactor `src/trainers/stage2_ab_training.py` to remove async actor-learner support:
  - drop `Stage2ABAsyncQueueManagerMixin` usage and async state
  - remove `mode` branching and the `world_size>1` step-mode restriction
  - ensure Channel-B runs only in the step-budgeted “buffer across micro-steps, execute on final micro-step” pathway.
- [x] 2.3 Refactor `src/trainers/stage2_ab/executors.py` to read the unified decode knob (`custom.extra.rollout_matching.decode_batch_size`) and remove Stage2-AB specific decode batch knobs.

## 3. Unified Decode Batching Across HF + vLLM

- [x] 3.1 Update `src/trainers/rollout_matching_sft.py` to use `custom.extra.rollout_matching.decode_batch_size` as the single decode batching knob for HF and vLLM, and delete/stop using `_rollout_generate_batch_size()` and any call sites that read legacy batch knobs.
- [x] 3.2 vLLM server mode: query `${base_url}/get_world_size/` and cache server DP world size; derive per-rank request chunk sizes so that (under multi-learner DDP) per-rollout-GPU decode work per call is bounded by `decode_batch_size`.
- [x] 3.3 Ensure Stage2-AB Channel-B uses the same unified decode batching behavior as rollout-matching (no extra knobs).

## 4. YAML Defaults Under `configs/stage2_ab/**`

- [x] 4.1 Update all `configs/stage2_ab/**/*.yaml` to the standardized pattern:
  - remove any `stage2_ab.channel_b.mode` entries
  - remove `custom.extra.rollout_matching.rollout_generate_batch_size`
  - remove `custom.extra.rollout_matching.rollout_infer_batch_size`
  - remove `custom.extra.rollout_matching.post_rollout_pack_scope`
  - set `custom.extra.rollout_matching.decode_batch_size: 4`
- [x] 4.2 Ensure Stage2-AB prod/smoke configs keep `training.effective_batch_size` as the only raw-rollout demand signal and remain divisible under the default launcher topology.

## 5. Docs + Verification

- [x] 5.1 Update `docs/training/STAGE2_RUNBOOK.md` and `docs/training/METRICS_LOSSES.md` to remove `async` / legacy knobs and document the single step-budgeted pathway + `decode_batch_size`.
- [x] 5.2 Run unit tests:
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_rollout_matching_sft.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py`
  - `PYTHONPATH=. conda run -n ms python -m pytest -q tests/test_stage2_ab_vllm_server_mode_smoke.py`
