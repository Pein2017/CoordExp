thread_id: 019dcab9-8c29-7f81-bb91-2e67e6f9185e
updated_at: 2026-04-26T17:28:46+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/26/rollout-2026-04-26T16-57-30-019dcab9-8c29-7f81-bb91-2e67e6f9185e.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 set-continuation prod hang was diagnosed as a rank-divergent post-eval save-control deadlock, not a simple save_steps gate

Rollout context: The user asked to inspect tmux session `prod` for a Stage-1 training run launched with `scripts/train.sh` and `configs/stage1/set_continuation/production.yaml` on 8 GPUs. The run appeared to hang forever at `step-100` right after `eval_step`. The user later pointed out that GPU memory usage stayed at `100%` and that `save_steps=200`, so the issue should not be explained by a normal step-based save at step 100.

## Task 1: Diagnose the live `prod` hang and identify the root cause

Outcome: success

Preference signals:

- The user asked to "inspect the tmux session `prod`" and diagnose the issue from the live run, which suggests that in similar incidents future agents should inspect the live session, logs, and process tree before theorizing from code alone.
- When the assistant initially leaned toward checkpoint/save-synchronization, the user corrected with "The GPU memory occupation haven't changed at all and showed `100%` usage. Something seems to be idle/dead." -> this suggests the user values distrust of superficial GPU-usage signals and wants the agent to keep looking for a dead/stuck state rather than assume progress from memory occupancy alone.
- When the user later said "I already sent stop signal and indeed nothing helpful is printed. Please be awared that the `save_steps` is `200` and it shouldn't trigger the checkpoint saving yet at `step 100`" -> this indicates future agents should treat config details from the resolved run as authoritative and should re-check the exact save strategy instead of assuming step-based save semantics.

Key steps:

- Captured the live `prod` tmux pane and process tree rather than relying on summaries. The pane showed eval completed (`200/200` samples, COCO metrics printed) and then the trainer remained stuck at `Train: 100/3664`.
- Verified the actual live worker processes and GPU state: eight `src.sft` ranks were still alive and hot on CPU, while `nvidia-smi` showed GPU contexts attributed to stale `[Not Found]` PIDs rather than visible live worker PIDs.
- Checked run artifacts under `output_remote/stage1_2b/set_continuation/.../v2-20260426-114031/`, where timestamps showed eval artifacts completed around `14:20:25` UTC and then no further useful progression.
- Read the resolved config for the stopped run and confirmed `training.save_strategy: "best"`, `save_steps: 200`, `eval_strategy: "steps"`, `eval_steps: 100`, `metric_for_best_model: "eval_det_bbox_AP"`, and `save_delay_steps: 600`.
- Traced code paths in `src/callbacks/stage1_detection_eval.py`, `src/trainers/stage1_set_continuation/trainer.py`, and `src/callbacks/save_delay_callback.py` to locate the post-eval save-control boundary.
- Used TDD on the suspected failure boundary: wrote a regression test first, confirmed it failed, patched the trainer, then re-ran targeted tests.

Failures and how to do differently:

- The first hypothesis that `save_steps=200` itself was blocking the run at step 100 was insufficient; the actual active save mode in the resolved config was `save_strategy="best"`, so step-based save cadence was not the governing condition.
- The shutdown attempt confirmed a collective hang pattern: `torchrun` sent SIGINT to all ranks, waited 30 seconds, then force-killed workers with SIGKILL, but this did not yield a useful worker Python traceback. For this failure class, live artifact/state inspection is more useful than waiting for a clean exception.

Reusable knowledge:

- In this run, the decisive config facts were in `resolved_config.json`, not just the source YAML.
- `save_strategy="best"` means HuggingFace decides save/best behavior right after eval based on `metric_for_best_model`; `save_steps` does not gate that path the way it would under `save_strategy="steps"`.
- `Stage1SetContinuationTrainer.evaluate()` performs callback eval, then broadcasts rank-0 metrics to all ranks, then returns metrics to the HF training loop. That means any callback-based save-delay guard that depends on eval metrics must account for the post-broadcast boundary.
- The likely deadlock shape here was rank divergence after eval: rank 0 had the new metric and save-delay guard state, while worker ranks did not yet have equivalent best-metric state before the trainer returned.

References:

- [1] Live tmux capture showed eval completion followed by stall at `Train: 100/3664` and post-eval metrics output, with no subsequent advancement.
- [2] `nvidia-smi` output showed `100%` SM usage and ~32–36 GiB on all 8 GPUs, but PIDs were stale `[Not Found]` entries, not the live rank PIDs.
- [3] `resolved_config.json` lines around 1478-1487 showed `eval_strategy="steps"`, `save_strategy="best"`, `save_steps=200`, `metric_for_best_model="eval_det_bbox_AP"`, and `save_delay_steps=600`.
- [4] `src/trainers/stage1_set_continuation/trainer.py::evaluate` broadcasts metrics after callback execution: `dist.broadcast_object_list(payload, src=0)` then `metrics.update(payload[0])`, then returns.
- [5] `src/callbacks/save_delay_callback.py::on_evaluate` only runs its best-metric guard when `args.save_strategy == SaveStrategy.BEST` and needs metrics to be present.

## Task 2: Add and verify a rank-symmetric save-delay guard regression

Outcome: success

Preference signals:

- The user interrupted the first checkpoint-divergence hypothesis with `save_steps=200` and asked for a re-check of the resolved training args and exact post-eval control path before editing. That suggests that on similar issues the next agent should re-validate actual runtime config and control flow before touching code.
- The user had already sent the stop signal and there was no useful output; this implies that for similar live hangs the user accepts a code/test-based diagnosis and fix path once the live failure is characterized.

Key steps:

- Added a regression test in `tests/test_stage1_set_continuation_trainer_smoke.py` for the worker-rank case where eval metrics are only visible after broadcast.
- The test initially failed as expected because the trainer had no post-broadcast save-delay guard.
- Added `Stage1SetContinuationTrainer._apply_rank_symmetric_save_delay_best_metric_guard(...)` and invoked it immediately after the trainer broadcasts eval metrics in `evaluate()`.
- Kept the new helper conservative: it only activates when `save_strategy == SaveStrategy.BEST`, iterates over callbacks, finds `SaveDelayCallback`, and replays its `on_evaluate(...)` with the broadcast metrics.
- Adjusted the helper to use `cast(Any, callback)` so it didn’t introduce extra type noise beyond the existing unknown-type debt in this area.

Reusable knowledge:

- The fix point is specifically between metric broadcast and the return from `evaluate()`, not inside the detection callback itself.
- The new helper should remain callback-local and rank-symmetric; it should not rework the save architecture or add new CLI flags.
- `SaveDelayCallback` is already installed centrally in `src/bootstrap/trainer_setup.py`; the trainer only needed to re-apply it after all ranks had the same metrics.

Failures and how to do differently:

- The first regression attempt failed for the intended reason and surfaced the missing helper; the second attempt also failed because the helper returned early when `control` was absent in the test harness. The fix was to give the test a minimal `control` object (`SimpleNamespace(should_save=True)`), which matches how the production trainer actually runs.
- `basedpyright` on these files still reports substantial pre-existing unknown-type debt (`187 errors, 2 warnings`). Future agents should not treat that as evidence that this fix is wrong; use the focused behavioral tests instead.

References:

- [1] `tests/test_stage1_set_continuation_trainer_smoke.py:373-400` added `test_trainer_applies_save_delay_guard_after_broadcasted_eval_metrics()`.
- [2] `src/trainers/stage1_set_continuation/trainer.py:1251-1276` added `_apply_rank_symmetric_save_delay_best_metric_guard(...)`.
- [3] `src/trainers/stage1_set_continuation/trainer.py:1331` now calls the helper after `broadcast_object_list(...)` and before runtime logging/return.
- [4] Targeted verification passed: `pytest tests/test_stage1_set_continuation_trainer_smoke.py` -> `20 passed`; `pytest tests/test_metric_key_lookup.py tests/test_checkpoint_weight_only_policy.py -k 'save_delay or best_save_strategy'` -> `2 passed`; `ruff check` and `ruff format --check` passed on the touched files.
- [5] `basedpyright` still emitted many unrelated unknown-type errors on this area, so static type cleanliness was not achieved even though the behavioral fix and lint/format checks passed.
