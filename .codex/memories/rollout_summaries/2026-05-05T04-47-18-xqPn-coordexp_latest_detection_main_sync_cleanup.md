thread_id: 019df676-42bf-7ea2-b306-a387873a5661
updated_at: 2026-05-05T14:20:11+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/05/rollout-2026-05-05T04-47-18-019df676-42bf-7ea2-b306-a387873a5661.jsonl
cwd: /data/CoordExp
git_branch: main

# Synchronized latest detection runtime/contracts into main, then cleaned up temporary worktree branches and verified current docs were updated.

Rollout context: The user asked for a codebase-level synchronization audit and planning pass for `.worktrees/compact-detection-sequence`, with `src/` as source of truth and a focus on aligning configs/scripts/docs around the refactored architecture. The work was carried out in an isolated integration worktree (`/data/CoordExp/.worktrees/refactor-latest-integration`), then fast-forwarded into `main` and pushed to origin. The user later clarified they wanted direct main integration, not PR-only publication, and also wanted all temporary worktree developer branches cleaned up. The final state was verified on `main`.

## Task 1: Source/config contract alignment for latest compact detection

Outcome: success

Preference signals:
- The user explicitly asked to treat `src/` as the source of truth and to avoid “blindly patch[ing] files one by one,” indicating they want contract-first auditing and careful migration planning before edits.
- The user asked to “spawn multiple subagents” and inspect different perspectives, indicating they prefer parallel, perspective-separated reasoning for complex migration work.
- The user later accepted the idea of broad validation and said “You have the full access to implement and execute whatever you want,” indicating they were comfortable with a full implementation/verification pass after the planning stage.
- When later asked about direct integration, the user said they wanted the latest updated state in `main` and to clean up temporary branches, indicating a preference for direct branch consolidation over lingering integration branches or PR-only publication.
- When asked to be careful with conflicts, the user said to “be extremely cautious and patient” and to “preserve the latest one over the previous older one,” which should be treated as a future default for merge conflict handling in similar repo cleanups.

Key steps:
- Loaded the repo guidance, memory, and parallel-agent workflow instructions first.
- Dispatched separate subagents for: `src/` config/runtime contracts, YAML config mapping, scripts/runtime launchers, training-profile semantics, and broader legacy-risk sweep.
- Inspected current schema/runtime symbols in `src/config/schema.py`, `src/config/loader.py`, `src/sft.py`, `src/training_runtime/plan.py`, and `src/detection/runtime.py`.
- Confirmed the latest compact detection contract is top-level latest-detection config (`data`, `prompt`, `detection_template`, `token_rows`, `objective`, `packing`, `evaluation`, `validation`) and that `LatestDetectionTrainingConfig` rejects `custom`.
- Identified that legacy `TrainingConfig` still supports `custom`, `stage2_ab`, and `rollout_matching`, while latest compact detection is schema-separated.
- Implemented the latest contract changes in `src/config/schema.py` and `src/sft.py`, including typed debug parsing and stricter recursive-detection packing guardrails.
- Added and updated focused tests for latest contract behavior and legacy config migration behavior.
- Converted the old unsupported-packing “smoke” path into an explicit negative contract path, then updated the remaining test reference to the new negative config.
- After a test failure showed the old smoke path now failed too early for the intended assertion, the test was corrected to target the negative contract directly.

Failures and how to do differently:
- The initial latest packing test still referenced the old smoke path, which had become a comments-only stub. The failure happened before the intended runtime guard, so the test had to be retargeted to the new negative config path.
- The `custom.coord_loss` migration coverage initially only scanned `stage2*` configs. A reviewer pointed out that the hard error applies to all legacy `TrainingConfig` paths, so the scan was broadened to all config YAMLs.
- When trying to use Serena symbol tools on the nested worktree, the project root resolution did not line up with the worktree paths. The practical fallback was narrow line-based patching and direct file inspection in the integration worktree.

Reusable knowledge:
- Latest compact detection now uses typed top-level sections and does not accept `custom`.
- `LatestDetectionTrainingConfig.debug` is typed via `DebugConfig`; `src/sft.py` preserves `debug.output_dir` behavior through the typed path.
- Recursive CE/latest compact detection packing is fail-fast at schema/materialization time, not just at runtime.
- `custom.coord_loss` on legacy `TrainingConfig` is now a hard migration error with guidance toward `custom.coord_soft_ce_w1` / the latest objective pipeline.
- The negative contract for unsupported latest recursive-detection packing lives under `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`.

References:
- [1] `src/config/schema.py`: latest detection schema parsing, typed `DebugConfig`, and recursive packing guardrails were updated.
- [2] `src/sft.py`: latest `debug.output_dir` is now handled through typed config only.
- [3] `tests/test_latest_training_config_contract.py` and `tests/test_legacy_config_contract.py`: focused contract coverage added/updated.
- [4] `tests/test_recursive_detection_ce_sft_wiring.py`: stale smoke-path assertion updated to the negative config path.
- [5] `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`: new negative expected-failure config.

## Task 2: Config/docs/scripting synchronization and provenance cleanup

Outcome: success

Preference signals:
- The user repeatedly emphasized that `configs/` needed to be remapped and reorganized according to the new architecture, especially because many YAML knobs had moved or been regrouped.
- The user explicitly asked for a “clean future-facing config organization” and for old conventions to be removed unless still clearly useful.
- The user later asked whether “all relevant documents in the docs/” were updated, indicating they care about routing/inventory documentation staying in sync with the executable config surface.
- The user’s direct instruction to “preserve the latest one over the previous older one” established that stale compatibility stubs should not be kept if they confuse the current contract.

Key steps:
- Added shared latest-detection authoring snippets under `configs/_shared/latest_detection/` for dataset, surface, and objective overlays.
- Added `configs/_shared/stage1_sft/README.md` to document the legacy Stage-1 SFT surface family separately.
- Added `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml` as an explicit negative contract.
- Deleted the old `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml` after reviewers confirmed the comment-only stub was still a stale launch/test hazard.
- Marked `configs/stage1/compact_detection_sequence/smoke/compact_full_tiny.yaml` as a legacy bridge rather than a current latest-schema example.
- Updated docs routing surfaces: `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/training/README.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE1_ET_RMP_CE.md`, and `docs/data/PACKING.md`.
- Reclassified `scripts/run_infer_eval.sh` as a legacy/debug wrapper that refuses official COCO/LVIS/both metrics entirely, so reportable metrics must go through the YAML-first inference -> scoring -> evaluation flow.
- Reclassified `scripts/run_vis.sh` as manual/debug, and `scripts/pipelines/run_rollout_stability_probe.sh` as historical/debug.
- Updated `docs/eval/WORKFLOW.md` to make official metric provenance explicit: COCO/LVIS/both claims must consume `gt_vs_pred_scored.jsonl`.
- Updated the current super-power roadmap/plan document to reflect the actual implementation procedure and the removal of the stale smoke YAML.
- Verified the final docs state by searching the current docs for stale references and checking the changed routing text.

Failures and how to do differently:
- Reviewers flagged that keeping a comment-only `.yaml` under `smoke/` was still risky because old tests and glob-based discovery could hit it. The fix was to delete the stub entirely and retarget the last test.
- Reviewers also flagged that `scripts/run_infer_eval.sh` could mix a fresh raw inference artifact with an unrelated scored artifact. The fix was to refuse official-style metrics entirely in that wrapper instead of trying to consume an external scored file.
- The docs/catalog initially described the new latest-detection overlays as if they were canonical shared overlays. That wording was downgraded to “authoring snippets” because the canonical launch configs still do not `extends` those overlays yet.

Reusable knowledge:
- The canonical current latest compact detection route is `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`.
- The new shared latest-detection overlays currently exist as authoring snippets, not as inherited launch config inputs.
- `docs/catalog.yaml` now uses `authoring_snippets: configs/_shared/latest_detection/` for latest compact detection rather than `shared_overlays`.
- `docs/eval/WORKFLOW.md` and `scripts/run_infer_eval.sh` now align on the rule that official-style metrics must not be produced from the legacy raw-artifact wrapper.
- The current docs explicitly warn against the older `support2_bsz32` / `32/256` set-continuation framing and point to the `bsz16` / `16/128` production identity.

References:
- [1] `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, `docs/training/README.md`, `docs/training/STAGE1_OBJECTIVE.md`, `docs/training/STAGE1_ET_RMP_CE.md`, `docs/data/PACKING.md`, `docs/eval/WORKFLOW.md`: current durable routing/provenance docs updated.
- [2] `scripts/README.md`, `scripts/run_infer_eval.sh`, `scripts/run_vis.sh`, `scripts/pipelines/run_rollout_stability_probe.sh`: wrapper classification and provenance guardrails updated.
- [3] `configs/_shared/latest_detection/README.md` and the new overlay YAMLs: latest-detection authoring snippets added.
- [4] `configs/stage1/recursive_detection_ce_latest/negative/compact_full_static_packing_should_fail.yaml`: explicit negative contract.
- [5] `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_packing_unsupported.yaml`: removed stale smoke YAML.

## Task 3: Validation, merge-to-main, branch/worktree cleanup, and final docs check

Outcome: success

Preference signals:
- The user later said they did not need a PR and wanted direct merging to main because this is their personal repo, which changed the publication path from PR-first to direct main integration.
- The user then clarified the long-term goal: “only keep everything updated latest in the main branch and clean up all the temporal work tree developer branches.” That is a durable workflow preference for this repo: converge to `main`, remove temporary codex branches/worktrees, and avoid leaving integration branches hanging around.
- The user asked to be “extremely cautious and patient about all the conflicts” and to preserve the latest over older changes, which was obeyed during the fast-forward and cleanup process.
- When asked whether all docs were updated, the follow-up request signaled that they care about current durable docs being in sync even after the code merge is done.

Key steps:
- Ran focused validation after the implementation and again after reviewer-driven fixes.
- The first targeted pytest run exposed a single stale-test issue, which was then fixed.
- Re-ran the focused tests and got `164 passed`.
- Ran shell syntax checks for the shell wrappers and a YAML parsing probe for the relevant docs/configs.
- Ran a negative guard check showing `eval_metrics=coco` now exits before `Running inference...` in the legacy wrapper.
- Verified `git diff --check` and staged only the intended implementation set.
- Committed the work on the integration branch, pushed it to origin, then fast-forwarded `main` directly to that commit.
- Hit one safe local-only issue: an older untracked copy of the super-power plan in `/data/CoordExp` would have been overwritten by the committed version already on `main`. That file was removed locally so the newer committed plan could win, matching the user’s “latest over older” preference.
- Verified `main` and `origin/main` were aligned at the new commit.
- Removed the temporary worktrees: `/data/CoordExp/.worktrees/compact-detection-sequence` and `/data/CoordExp/.worktrees/refactor-latest-integration`.
- Deleted local and remote `codex/*` branches tied to the temporary worktrees.
- Confirmed the only remaining local dirt in `/data/CoordExp` is unrelated `.codex/skills/gitnexus-*` deletion state, which was intentionally preserved and not treated as part of this goal.

Failures and how to do differently:
- `git branch -d` refused to delete the local compact branch at first because it was not fully merged into its old remote tracking branch. The branch was nonetheless proven safe via ancestry checks (`main` and `origin/main` were both ancestors of it), so `git branch -D` was used only after verifying containment in `main`.
- `git merge --ff-only` from the root checkout initially refused because of an older untracked draft plan file that would have been overwritten. The resolution was to remove the stale untracked copy and rerun the fast-forward, rather than forcing a manual merge.
- The user asked specifically whether the docs were updated; rather than answering from memory, a current-doc search was used to verify the durable docs in `docs/` were actually updated.

Reusable knowledge:
- The final merged commit on `main` is `e162a1f refactor(training): align latest detection runtime contracts`.
- `main` and `origin/main` are now aligned at that commit.
- Temporary branches removed: `codex/compact-detection-sequence`, `codex/refactor-latest-integration`.
- Remote temporary branches removed: `origin/codex/compact-detection-sequence`, `origin/codex/refactor-latest-integration`.
- The current `docs/` surfaces that matter for this refactor are updated and now point to the latest compact detection route, the legacy bridge route, the negative packing contract, and the legacy/debug eval wrapper classification.

References:
- [1] Validation command: `conda run -n ms python -m pytest tests/test_latest_training_config_contract.py tests/test_legacy_config_contract.py tests/test_training_config_strict_unknown_keys.py tests/test_recursive_detection_ce_sft_wiring.py -q` → `164 passed in 2.06s`.
- [2] Guard command: `eval_metrics=coco output_base_dir=temp/verify_run_infer_eval_guard bash scripts/run_infer_eval.sh` → exit `2` before inference with the expected refusal message.
- [3] `git diff --check` and `git diff --cached --check` both passed.
- [4] `git push origin HEAD:main` fast-forwarded remote `main` to the integration commit.
- [5] `git worktree remove ...`, `git branch -D ...`, and `git push origin --delete ...` cleaned the temporary branches/worktrees.
- [6] Final docs verification: current docs mention the updated latest route, the legacy bridge, authoring-snippet wording, and the official-metric guardrails; no active current-doc route still points at the removed smoke YAML.

## Bottom line

The code/config/docs/scripts migration was completed and merged into `main`, with the latest compact-detection runtime contracts and doc routing synchronized to the refactor. The temporary integration worktrees and codex branches were removed, and the current durable docs relevant to this refactor were updated. The only explicitly preserved local state at the end was unrelated `.codex/skills/gitnexus-*` deletion dirt, which was not part of the merge goal and was left untouched.
