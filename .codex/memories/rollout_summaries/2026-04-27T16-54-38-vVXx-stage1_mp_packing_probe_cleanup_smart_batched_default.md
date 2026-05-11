thread_id: 019dcfdd-481a-7e93-ae6d-966242824f07
updated_at: 2026-04-28T10:08:15+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/27/rollout-2026-04-27T16-54-38-019dcfdd-481a-7e93-ae6d-966242824f07.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 MP branch-runtime packing probe was recorded, then the experimental worktree was retired, and `configs/stage1/set_continuation/production.yaml` was confirmed to already default to `smart_batched_exact`.

Rollout context: The user handed off a dirty worktree for Stage-1 MP branch packing experiments, then later asked to (1) record the rough smart-vs-packed comparison in docs, (2) clean up the worktree/branch, and (3) ensure the checked-in production YAML stays on the best packing mechanism. The worktree was expected to contain many uncommitted experimental changes and was not supposed to be merged.

## Task 1: Stage-1 MP packed-varlen / cross-sample packing probe

Outcome: partial

Preference signals:
- The user said the prior effort was “wated” and asked to “record these rough comparison in the docs and manage to cleanup this worktree and branch and stay default to use smart batch mechanism” -> future Stage-1 packing work should preserve benchmark evidence in docs, explicitly call out that rough comparisons are not production proof, and keep the default on smart batching unless a future parity/throughput gate clearly beats it.
- The user later asked “Is the `smart batch` implemented in `main` or the `worktree`?” -> future agents should verify whether a feature already exists on `main` before treating a worktree effort as novel.

Key steps:
- Read the active OpenSpec/preflight/design material and inspected the existing `branch_packing.py`, `branch_scorer.py`, `trainer.py`, `sft.py`, and relevant tests in the worktree.
- Ran a rough 8-GPU production-like benchmark comparing `smart_batched_exact`, `online_rank_microbatch_packed`, and `offline_sample_packed` on the Stage-1 MP COCO coord-token setup.
- Separated trainer `train_runtime` from offline sample-pack construction time, and treated the offline pack manifest build as a one-time preprocessing cost rather than part of the repeated train-loop runtime.
- Observed that the packed experiments improved fill density but did not beat smart batching on the measured logical throughput/step-time surfaces.

Failures and how to do differently:
- The first aggregate report under-read metadata because the benchmark tooling initially looked for `smoke_summary.json` that the benchmark runs did not emit; the fix was to pull from `logging.jsonl`, `effective_runtime.json`, and the sample-pack manifest instead.
- The first interpretation mixed physical packed-envelope counts with logical raw-sample counts; the user explicitly corrected this, and the updated interpretation compared logical raw samples per optimizer update and per second rather than just physical rows.
- The offline packed run’s logical batch geometry differed from the smart baseline, so the benchmark remained a rough comparison rather than a controlled apples-to-apples parity proof.

Reusable knowledge:
- In this repo, Stage-1 set-continuation smart batching is the safe path for exact candidate scoring because candidate branches remain separate batch rows; packed-varlen branch execution is the tricky boundary-sensitive path that still needs real-Qwen no-step parity before production use.
- The rough benchmark showed that offline sample packing can achieve very high fill ratio (`mean_fill=0.981`) but still fail to beat smart batching end to end because packed MP scoring/loss assembly is expensive.
- If comparing packed vs smart Stage-1 MP runs, separate three clocks: process wall time, trainer `train_runtime`, and any offline sample-pack preprocessing time.

References:
- [1] Rough benchmark report committed on `main`: `progress/benchmarks/2026-04-28_stage1_mp_branch_runtime_packing_probe.md`
- [2] Preserved aggregate artifacts: `progress/benchmarks/artifacts/2026-04-28_stage1_mp_branch_runtime_packing_probe_aggregate.md` and `.json`
- [3] Benchmark headline values recorded in the note: `smart_batched_exact` `train_runtime=398.309s`, `train_steps_per_second=0.015`, `train_samples_per_second=1.928`; `online_rank_microbatch_packed` `418.334s`, `0.014`, `1.836`; `offline_sample_packed` `761.400s`, `0.008`, `1.836`; offline fill metrics `raw_samples=2048`, `raw_packs=1418`, `aligned_packs=1424`, `mean_fill=0.981`.

## Task 2: Cleanup / retire the experimental worktree and branch

Outcome: success

Preference signals:
- The user explicitly asked to “cleanup this worktree and branch” after deciding the rough packed comparison was not worth continuing -> future agents should be prepared to retire experimental worktrees once the user decides the path is not promising.
- The user also asked to “stay default to use smart batch mechanism” -> future agents should preserve that default in current docs/configs even when cleaning up abandoned experiments.

Key steps:
- Committed the docs-only benchmark note on `main` with `docs(stage1): record MP packing runtime probe`.
- Removed `/data/CoordExp/.worktrees/stage1-mp-padding-free-branch-packing-spec` with `git worktree remove --force`.
- Deleted the local branch `codex/stage1-mp-padding-free-branch-packing-spec`.
- Verified the worktree path no longer existed and that the branch count was zero.

Failures and how to do differently:
- The worktree contained many unrelated dirty experimental changes; the cleanup had to avoid sweeping those into the docs commit.
- Before removing a worktree, make sure any durable benchmark evidence is copied into `progress/` or another stable repo location; temp files inside the worktree would be lost on cleanup.

Reusable knowledge:
- `git worktree remove --force <path>` plus `git branch -D <branch>` is the cleanup path when the feature branch is explicitly being retired and the worktree is no longer needed.
- `git worktree list` is a quick way to confirm only the intended worktrees remain after cleanup.

References:
- [1] Commit on `main`: `295c484 docs(stage1): record MP packing runtime probe`
- [2] Cleanup evidence: worktree removed, branch deleted, `git worktree list` only showed `/data/CoordExp` on `main` and the unrelated `feat/agent-research-runtime` worktree.
- [3] Verification prints: `WORKTREE_REMOVED`, branch lookup count `0`, and `DOCS_VERIFY_OK`.

## Task 3: Confirm and preserve production default in `configs/stage1/set_continuation/production.yaml`

Outcome: success

Preference signals:
- The user asked, “Please now refer to my `production.yaml` and make sure it uses the best packing mechanism so far” -> future agents should verify the checked-in production config directly rather than assuming it needs rewriting.
- The user’s earlier correction about logical batch size showed they care about the true training-information budget, not just physical row counts -> future config checks should confirm both runtime mode and batch geometry.

Key steps:
- Checked `configs/stage1/set_continuation/production.yaml` on `main`.
- Confirmed the production YAML already uses `train_forward.branch_runtime.mode: smart_batched_exact` with `branch_batching.enabled: true`, `branch_batching.strategy: ms_swift_constant_volume_buckets`, `branch_batching.max_branch_rows: 8`, and `ddp_sync.candidate_padding: none`.
- Ran a config-loader materialization check to confirm inherited/defaulted values: `training.packing=false`, `training.eval_packing=false`, `encoded_sample_cache.enabled=false`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=2`, `effective_batch_size=128`, `branch_runtime.mode=smart_batched_exact`, `branch_batching.enabled=true`, `branch_batching.strategy=ms_swift_constant_volume_buckets`, `branch_batching.max_branch_rows=8`, `branch_batching.max_branch_tokens=None`, `logits.mode=supervised_suffix`, `ddp_sync.candidate_padding=none`, `prefix_reuse.kv_cache.mode=disabled`.

Failures and how to do differently:
- A first attempt to read the config-loader object treated `training` like an attribute object when it was a dict; the fix was to use dict-safe accessors for the materialized config.
- The checked-in production config already matched the desired default, so no edit was necessary.

Reusable knowledge:
- The current production Stage-1 config on `main` already defaults to the safest/current-best mechanism: smart batching with exact selected-candidate scoring, not packed varlen.
- This production config intentionally keeps dataset packing and eval packing off, disables the encoded-sample cache, and uses no DDP candidate padding or KV cache.

References:
- [1] `configs/stage1/set_continuation/production.yaml` lines confirming `branch_runtime.mode: smart_batched_exact` and `branch_batching.strategy: ms_swift_constant_volume_buckets`
- [2] Materialized config values from `ConfigLoader.load_materialized_training_config(...)`:
  - `training.packing False`
  - `training.eval_packing False`
  - `encoded_sample_cache.enabled False`
  - `per_device_train_batch_size 8`
  - `gradient_accumulation_steps 2`
  - `effective_batch_size 128`
  - `branch_runtime.mode smart_batched_exact`
  - `branch_batching.enabled True`
  - `branch_batching.strategy ms_swift_constant_volume_buckets`
  - `branch_batching.max_branch_rows 8`
  - `branch_batching.max_branch_tokens None`
  - `logits.mode supervised_suffix`
  - `ddp_sync.candidate_padding none`
  - `prefix_reuse.kv_cache.mode disabled`
