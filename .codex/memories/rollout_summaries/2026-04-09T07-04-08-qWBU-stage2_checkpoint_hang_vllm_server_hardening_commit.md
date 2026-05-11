thread_id: 019d710e-3040-7830-a222-325cb8750256
updated_at: 2026-04-09T10:14:28+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/09/rollout-2026-04-09T07-04-08-019d710e-3040-7830-a222-325cb8750256.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-2 checkpoint hang/root-cause investigation, vLLM server-mode validation, launcher hardening, and commit

Rollout context: The user was troubleshooting a long Stage-2 training pause after `eval_step` on `output/stage2_ab/prod/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged`, then asked to continue with vLLM server mode, then asked to harden the launcher against stale vLLM workers, and finally asked to commit the local changes. The work was done in `/data/CoordExp` and used the `ms` conda environment for verification.

## Task 1: Root-cause the apparent post-eval hang and validate checkpoint behavior

Outcome: success

Preference signals:
- The user asked to "dive into the deep root cause" and later to continue with vLLM server mode, indicating they want a genuine postmortem with evidence, not just a superficial guess.
- The user accepted the assistant’s shift from the server-mode path to a direct DDP checkpoint-focused smoke when the earlier server-mode attempt failed for unrelated vLLM startup reasons, suggesting they value targeted reproduction over insisting on a single path when the first path is noisy.
- The user said "Good, do `2`" after being offered options, indicating they are comfortable with concrete actionable next steps once the root cause is identified.

Key steps:
- Inspected the stage-2 runbook and implementation map to trace the likely post-eval save/barrier path.
- Confirmed from the production run directory that `checkpoint-300` already existed, so the slowdown was not raw checkpoint I/O.
- Verified the live run state: two DDP ranks were still alive, and rank 0/1 were both running long after the last eval log.
- Read the installed `transformers.Trainer` code in the `ms` environment and identified the raw `dist.barrier()` in `_save_checkpoint()` as the narrow upstream synchronization point after `save_model(output_dir)` when `save_strategy in {steps, epoch}` and `best_global_step` is set.
- Determined that the Stage-2 wrapper itself was not the primary failure; the apparent hang was actually a checkpoint/save-delay/config interaction plus a barrier-sensitive training path.

Failures and how to do differently:
- The first vLLM server-mode smoke was a poor checkpoint regression because it hit a different startup problem (`Engine core initialization failed`) caused by stale local vLLM workers, not by checkpoint logic.
- The rollout’s inherited `save_delay_steps: 100` from `configs/base.yaml` meant short smokes could look healthy while never actually creating checkpoints. Future checkpoint smokes should override this to `0` explicitly.

Reusable knowledge:
- In this repo, the Stage-2 save path can look like an eval hang when the callback/Trainer save gate and distributed barrier semantics are involved.
- `save_delay_steps` is inherited from `configs/base.yaml` and must be zeroed in short checkpoint-focused smokes if you want to validate actual checkpoint creation.
- For Stage-2 `transformers.Trainer` save cadence, checkpoint creation can be delayed or blocked by `SaveDelayCallback`, while the real synchronization point after `save_model()` is the upstream `dist.barrier()` in `_save_checkpoint()`.

References:
- [1] `output/stage2_ab/prod/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged/.../v1-20260407-053816/logging.jsonl` stopped at the eval payload with `eval/runtime/runtime_s: 602.36759914` and `global_step/max_steps: 300/1221`.
- [2] `output/.../checkpoint-300/` existed and contained `adapter_model.safetensors`, `adapter_config.json`, `additional_config.json`, `trainer_state.json`, `training_args.bin`.
- [3] `resolved_config.json` showed `eval_steps: 300`, `save_strategy: steps`, `save_steps: 300`, `metric_for_best_model: detection/mAP`, `greater_is_better: true`, `save_delay_steps: 100`.
- [4] `transformers.Trainer._save_checkpoint()` in the installed package contains a raw `dist.barrier()` after `save_model(output_dir)` when step/epoch saving and `best_global_step` are in play.

## Task 2: Validate and fix checkpoint smoke configs

Outcome: success

Preference signals:
- The user wanted a concrete way to test the issue, and the assistant’s creation of a direct DDP checkpoint smoke was accepted as the right validation path.
- The user later asked to continue in server mode, which reinforced that future similar work should preserve both a direct DDP regression and a server-mode path.

Key steps:
- Added a direct-DDP smoke config derived from the prod path, but with `rollout_backend: hf` / `eval_rollout_backend: hf` to isolate checkpoint behavior from the six-GPU server startup complexity.
- Added a checkpoint-focused Stage-2 smoke config with `b_ratio: 0.0` and `save_delay_steps: 0`.
- Added a prod-faithful server-mode smoke config that keeps vLLM rollout server mode but also explicitly sets `save_delay_steps: 0`.
- Verified on the direct DDP smoke that it passed the first eval/save boundary and produced both `checkpoint-2` and `checkpoint-4`.
- Verified on the server-mode smoke that it also passed the first eval/save boundary and produced both `checkpoint-2` and `checkpoint-4`.

Failures and how to do differently:
- Short checkpoint smokes will silently fail to create checkpoints if they inherit `save_delay_steps` from the base config. That was the real reason the first smoke attempt looked like it was “healthy” but produced no checkpoint dirs.
- The server-mode smoke is intentionally heavier and slower than the direct DDP smoke; use the direct DDP smoke as the canonical checkpoint regression and the server-mode smoke as the end-to-end parity check.

Reusable knowledge:
- The direct DDP smoke is the cleanest fast regression for checkpoint/save behavior.
- The server-mode smoke is useful once the rollout server is healthy, but it can be confounded by stale vLLM worker state.
- In the successful direct smoke (`v1-20260409-080956`), `checkpoint-2` and `checkpoint-4` were both present and `best_model_checkpoint` stayed at `checkpoint-2`.
- In the successful server-mode smoke (`v0-20260409-091756`), the same checkpoint pattern held under `vllm` server mode.

References:
- [1] `configs/stage2_two_channel/smoke/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged_checkpoint_ddp_4steps.yaml`
- [2] `configs/stage2_two_channel/smoke/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged_checkpoint_ddp_direct_4steps.yaml`
- [3] `configs/stage2_two_channel/smoke/2b_lvis_proxy_pseudo_positive_dup_targeting_ckpt1564_merged_save_eval_6steps.yaml`
- [4] Successful direct smoke run dir: `output/stage2_ab/smoke/.../v1-20260409-080956/`
- [5] Successful server-mode smoke run dir: `output/stage2_ab/smoke/.../v0-20260409-091756/`
- [6] Direct smoke `trainer_state.json` at `checkpoint-2`: `global_step=2`, `best_global_step=2`, `best_model_checkpoint=.../checkpoint-2`.
- [7] Direct smoke `trainer_state.json` at `checkpoint-4`: `global_step=4`, `best_global_step=2`, `best_model_checkpoint=.../checkpoint-2`.
- [8] Server-mode smoke `trainer_state.json` at `checkpoint-2`/`checkpoint-4` showed the same best-checkpoint bookkeeping.

## Task 3: Harden the vLLM server launcher against stale local worker processes

Outcome: success

Preference signals:
- The user said "Good, do `2`" after the assistant proposed launcher hardening, which indicates they prefer a preventive guard instead of repeatedly rediscovering the same stale-process failure.
- The user later asked to commit the local changes, suggesting they value packaging the prevention fix as a clean, reusable repo change rather than leaving it as a one-off diagnosis.

Key steps:
- Inspected `src/launchers/stage2_vllm_server.py` and the launcher preflight helpers to find the best insertion point.
- Added a preflight check that runs before server boot and inspects the chosen rollout GPUs for local vLLM-related compute processes.
- The guard looks for local `vllm` / `swift rollout` / `EngineCore` / `openai.api_server`-style processes by combining `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name` with `/proc/<pid>/cmdline` / `comm` identity checks.
- The guard fails fast with an actionable message listing GPU index, PID, memory usage, and command identity.
- Added focused tests proving the detector and the launcher fail-fast path.
- Verified the launcher test file in the `ms` environment: `conda run -n ms python -m pytest tests/test_stage2_vllm_server_launcher.py` -> `18 passed`.

Failures and how to do differently:
- The initial server-mode failure was caused by orphaned local `VLLM::EngineCore_DP*` workers holding ~70 GiB on GPUs `0-5`; the new guard addresses exactly that failure mode.
- The first attempt to test the new fail-fast path expected `RuntimeError`, but the launcher’s real top-level error path wraps failures in `SystemExit(1)` via `_die()`. The test had to be aligned to the actual launcher contract.

Reusable knowledge:
- `scripts/train_stage2.sh` / `src.launchers.stage2_vllm_server` already has port and world-size sanity checks, but stale local compute processes can still poison startup before those checks matter.
- A good preventive guard for this launcher should detect only local vLLM-related workers on the selected rollout GPUs, not all unrelated GPU usage.
- The real failure signature for a poisoned server launch was six orphaned `VLLM::EngineCore_DP*` workers on GPUs `0-5` plus about `70 GiB` each of memory use.

References:
- [1] `src/launchers/stage2_vllm_server.py` — added `_find_local_vllm_processes_on_gpus()` and `_assert_no_stale_local_vllm_processes()` plus a call before server boot.
- [2] `tests/test_stage2_vllm_server_launcher.py` — new coverage for the detector and `main()` fail-fast behavior.
- [3] Verification command: `conda run -n ms python -m pytest tests/test_stage2_vllm_server_launcher.py` -> `18 passed in 0.27s`.
- [4] Failure evidence before the fix: `ps` showed `VLLM::EngineCore_DP0` through `VLLM::EngineCore_DP5` orphaned and `nvidia-smi` showed ~70 GiB used on GPUs `0-5`.

## Task 4: Commit the local changes

Outcome: success

Preference signals:
- The user asked to "help me commit the local changes", which indicates they want the assistant to package the repo changes cleanly once the validation is done, rather than leaving them as an uncommitted diff.
- The commit was made only after checking `git status` so unrelated changes could be avoided, which is a good default for future similar requests.

Key steps:
- Checked the worktree with `git status --short` and reviewed the diff for the intended files.
- Committed the launcher hardening plus the smoke configs and their tests.
- Final commit hash: `6dac24a`.

Failures and how to do differently:
- The worktree had other unrelated modified files at the time of commit-time inspection, so the exact staged set mattered. Future commit requests should keep the "what to include" scope explicit if there are unrelated local edits.

Reusable knowledge:
- Commit message used: `Harden stage2 checkpoint smoke and vLLM launcher`.
- The worktree was clean after the commit.

References:
- [1] Commit hash: `6dac24a`
- [2] Commit message: `Harden stage2 checkpoint smoke and vLLM launcher`
- [3] Post-commit status: clean worktree.

