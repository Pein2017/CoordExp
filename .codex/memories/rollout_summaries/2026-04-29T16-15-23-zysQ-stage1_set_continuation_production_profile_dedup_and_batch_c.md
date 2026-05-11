thread_id: 019dda06-0fd4-7192-a629-bd4c40cc89fc
updated_at: 2026-04-29T16:42:03+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/29/rollout-2026-04-29T16-15-23-019dda06-0fd4-7192-a629-bd4c40cc89fc.jsonl
cwd: /data/CoordExp
git_branch: main

# Stage-1 set-continuation production profile was normalized to the repo’s tested 16/128 contract, with duplicate run identifiers deduplicated via consistent `bsz16` naming.

Rollout context: The user asked to update `configs/stage1/set_continuation/production.yaml` to deduplicate `run_name`, `artifact_subdir`, `experiment`, and `benchmark`, and resolve inconsistency. The repo’s Stage-1 docs/tests initially suggested a conflicting 32/256 variant, but the user explicitly corrected the batch contract to keep `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1`, and `effective_batch_size: 128`.

## Task 1: Inspect and identify the intended canonical Stage-1 set-continuation profile

Outcome: success

Preference signals:
- The user asked to “deduplicate the `run_name`, `artifact_subdir` and `experiment` and `benchmark`, and resolve any inconsistency,” which suggests future edits should look for duplicated identity fields and make the profile internally consistent rather than changing substantive training behavior.
- The user later corrected the batch contract with “keep: per_device_train_batch_size: 16 / gradient_accumulation_steps: 1 / effective_batch_size: 128,” which is strong evidence that this profile should stay on the 16/128 production regime even if other docs/tests mention a different variant.

Key steps:
- Read the target YAML, adjacent Stage-1 configs, docs, and the contract test that pins this profile.
- Traced the config loader behavior for `training.artifact_subdir` / `training.run_name` and the schema for top-level `experiment` and `benchmark` sections.
- Used `git log -p` to identify that the profile had evolved through earlier naming regimes, then aligned the file with the current contract test and docs.

Failures and how to do differently:
- The first edit pass initially drifted toward the newer 32/256 naming and batch contract because the file prose and some docs had that shape. The user’s correction showed that the tested contract was the authoritative one for this profile.
- When a config file’s prose conflicts with the user’s explicit desired settings, prefer the user’s latest correction and update the tests/identifiers to match that contract.

Reusable knowledge:
- `configs/stage1/set_continuation/production.yaml` is validated by `tests/test_stage1_set_continuation_benchmark_profiles.py`, so changes to identifiers or batch sizing should be kept in sync with that test.
- The config loader materializes `training.output_dir` / `training.logging_dir` from `training.output_root` + `training.artifact_subdir`, and it checks `training.run_name` against the output directory name. That means `run_name` and `artifact_subdir` are not merely cosmetic; they participate in resolved path consistency.
- Top-level `experiment` and `benchmark` are first-class typed sections, so deduplication can be done by reusing exact strings or YAML anchors without changing semantics.

References:
- [1] `configs/stage1/set_continuation/production.yaml` originally had conflicting naming/batch text; after normalization it resolved to `artifact_subdir: coco1024_sota1332_setcont_et_rmp_ce_support2_bsz16_v1`, `run_name: setcont-coco1024-sota1332-et-rmp-ce-support2-bsz16-v1`, `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 1`, `effective_batch_size: 128`.
- [2] `tests/test_stage1_set_continuation_benchmark_profiles.py` was updated to assert the same `bsz16` identifiers and `16/128` batch contract.
- [3] Verification command: `conda run -n ms python -m pytest -q tests/test_stage1_set_continuation_benchmark_profiles.py` → `6 passed in 0.89s`.

## Task 2: Deduplicate identifiers and resolve the batch-contract inconsistency in production.yaml

Outcome: success

Preference signals:
- The user’s correction “keep: per_device_train_batch_size: 16 / gradient_accumulation_steps: 1 / effective_batch_size: 128” indicates they want the final config to preserve the existing effective update batch while deduplicating names, not to migrate to a different memory regime.
- The original request to deduplicate `run_name`, `artifact_subdir`, `experiment`, and `benchmark` suggests that repeated identity-like strings should be normalized so the profile is easier to maintain and less contradictory.

Key steps:
- Reworked `configs/stage1/set_continuation/production.yaml` so the model checkpoint and benchmark budget label are anchored once and reused.
- Aligned `artifact_subdir`, `run_name`, `benchmark.group_id`, and `custom.extra.benchmark_report.same_budget_label` / `train_forward_budget` to the `bsz16` production contract.
- Preserved the authored experiment narrative and benchmark metadata, but removed the accidental 32/256 framing that contradicted the kept batch settings.
- Updated the contract test so the repo’s own verification matches the YAML exactly.

Failures and how to do differently:
- The first attempt at deduplication exposed a stale mismatch between YAML and tests; the correct fix was not to leave both versions around but to propagate the chosen contract everywhere it was asserted.
- Anchors are useful for deduplicating exact repeated values, but they should not be used to hide conflicting conceptual versions; only reuse exact strings that are already authoritative.

Reusable knowledge:
- For this profile, the canonical benchmark identity is `stage1_set_continuation_et_rmp_ce_support2_bsz16`, not the older `bsz32` variant that appeared during an intermediate edit.
- The tested production profile now expects `artifact_subdir: coco1024_sota1332_setcont_et_rmp_ce_support2_bsz16_v1`, `run_name: setcont-coco1024-sota1332-et-rmp-ce-support2-bsz16-v1`, and `same_budget_label: smart_batched_exact_full_suffix_rows_no_ddp_padding_et_rmp_ce_support2_bsz16_v1`.
- The profile remains `stage1_set_continuation`, `objective.mode: entry_trie_rmp_ce`, with `branch_support_weight: 2.0` and `branch_balance_weight: 1.0`, `branch_runtime.mode: smart_batched_exact`, `branch_batching.max_branch_rows: 32`, `max_branch_tokens: 65536`, and `budget_policy.enabled: false`.
- The helper test `tests/test_stage1_set_continuation_benchmark_profiles.py` is the right place to pin these expectations and catch any future drift.

References:
- [1] YAML changes: `training.per_device_train_batch_size: 16`, `training.gradient_accumulation_steps: 1`, `training.effective_batch_size: 128`, `benchmark.group_id: stage1_set_continuation_et_rmp_ce_support2_bsz16`.
- [2] YAML deduplication: `baseline` and `same_budget_label` / `train_forward_budget` were reduced to single-source values via YAML anchors.
- [3] Test changes: assertions updated for `bsz16` identifiers and `16/128` batch values.
- [4] Final verification: `conda run -n ms python -m pytest -q tests/test_stage1_set_continuation_benchmark_profiles.py` → `6 passed`.

