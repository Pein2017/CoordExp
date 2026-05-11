# Task Group: CoordExp repo-local Codex configuration, cleanup, and operator defaults

scope: Current repo-local agent setup, cleanup requests for unsuitable tooling, Serena/Codex launch behavior, and local memory/config migration. Use when the task touches `.codex`, Serena MCP, Codex launch flags, or repo-local cleanup of abandoned tooling.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this checkout's repo-local agent/config surfaces and user operating defaults, but re-check live files before editing because config paths and tool wrappers can drift.

## Task 1: Remove Graphify artifacts and scrub repo-local traces, completed cleanup

### rollout_summary_files

- rollout_summaries/2026-04-20T03-43-50-QEqm-graphify_cleanup_repo_local_traces.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/20/rollout-2026-04-20T03-43-50-019da8fc-c324-7af3-8c2a-9d6c53898f42.jsonl, updated_at=2026-04-20T03:52:24+00:00, thread_id=019da8fc-c324-7af3-8c2a-9d6c53898f42, latest current-state evidence; supersedes older repo-local Graphify setup guidance)

### keywords

- graphify, graphify-out, GRAPH_REPORT, .codex/history.jsonl, .codex/memories/raw_memories.md, .codex/graphify, .graphify_detect.json, cleanup, repo-local traces

## Task 2: Standardize workspace Codex home on `.codex`, update tracking, remove legacy `.codex_config`

### rollout_summary_files

- rollout_summaries/2026-04-06T06-02-55-VmgB-codex_home_migration_to_dot_codex.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/06/rollout-2026-04-06T06-02-55-019d6163-1253-7510-a386-50a2c7dc95b5.jsonl, updated_at=2026-04-06T06:08:45+00:00, thread_id=019d6163-1253-7510-a386-50a2c7dc95b5, canonical repo-home migration)
- rollout_summaries/2026-04-03T07-06-17-0toa-codex_self_improving_port_and_shared_memory.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/03/rollout-2026-04-03T07-06-17-019d522a-00a0-79b1-858d-4b5e08d069ee.jsonl, updated_at=2026-04-03T07:35:18+00:00, thread_id=019d522a-00a0-79b1-858d-4b5e08d069ee, historical context for portable-skill vs shared-memory split)

### keywords

- .codex, .codex_config/pein, CODEX_HOME, .gitignore allowlist, .self-improving, portable skill, shared memory, remove legacy directory

## Task 3: Fix Serena MCP to run in Codex mode and explain tool-surface behavior

### rollout_summary_files

- rollout_summaries/2026-04-03T02-53-44-mrYL-serena_codex_config_update_and_remote_reset.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/03/rollout-2026-04-03T02-53-44-019d5142-c929-76e2-a32d-3e6ab158ea03.jsonl, updated_at=2026-04-03T03:11:08+00:00, thread_id=019d5142-c929-76e2-a32d-3e6ab158ea03, inspect real config before answering)

### keywords

- Serena MCP, --context codex, desktop-app, execute_shell_command, .codex_config/pein/config.toml, uv run, --project, codex.yml

## Task 4: Route subagent capacity by task difficulty instead of a blanket effort setting

### rollout_summary_files

- rollout_summaries/2026-04-03T03-37-41-yGnS-subagent_model_allocation_tiered_routing.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/03/rollout-2026-04-03T03-37-41-019d516b-050a-7962-ba64-9d2580332270.jsonl, updated_at=2026-04-03T03:44:11+00:00, thread_id=019d516b-050a-7962-ba64-9d2580332270, durable user operating preference)

### keywords

- subagents, gpt-5.4-mini, gpt-5.4, medium, high, xhigh, reasoning_effort, AGENTS.md, config.toml, task routing

## Task 5: Answer Codex launch permission questions with docs-backed full-access guidance

### rollout_summary_files

- rollout_summaries/2026-04-03T03-13-38-pUIF-codex_full_access_danger_full_access.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/03/rollout-2026-04-03T03-13-38-019d5154-fed7-7a51-bd21-d7ffd11a64b9.jsonl, updated_at=2026-04-03T03:16:15+00:00, thread_id=019d5154-fed7-7a51-bd21-d7ffd11a64b9, official-Codex answer about sandbox flags)

### keywords

- codex, danger-full-access, yolo, workspace-write, full-auto, ask-for-approval never, sandbox, proxy, CODEX_HOME, bwrap

## User preferences

- when the user says a tool/framework is unsuitable, preserve their wording and treat it as a true removal request: "请帮我清空所有`graphify`和`graphify-out`的相关 artifacts" and "请帮我将其完全“铲除”，包括所有的 references 和 prompt" -> default to deleting generated artifacts plus repo-local references, prompt material, and memory/history traces rather than trying to preserve the old setup [Task 1]
- when standardizing local agent state, the user explicitly asked to migrate from `.codex_config/pein` to `.codex/` "to align with the `standard` official convention" and later said they would set `CODEX_HOME=/data/home/xiaoyan/AIteam/data/CoordExp/.codex` -> prefer `.codex/` as the canonical repo-local Codex home in this checkout [Task 2]
- when troubleshooting Serena/Codex setup, the user asked whether it was configured for "codex, not claude code or others" and pasted the exact config path -> inspect the live config stanza and explain behavior from concrete file/code evidence rather than giving only conceptual advice [Task 3]
- when choosing model effort for subagents, the user corrected the blanket approach with "No, we should optimally allocate the model capacity" and gave a task-tier sketch -> route model/reasoning effort by task difficulty instead of defaulting every subtask to the highest setting [Task 4]
- when asking about Codex permissions, the user said "Please search Web and tell me how to launch the codex with real full access" and "I want the `dangerous` mode" -> answer from current official docs and preserve their proxy / `CODEX_HOME` alias shape while changing only the Codex flags [Task 5]

## Reusable knowledge

- The current repo-local cleanup fact is that Graphify should be treated as removed from this checkout. The validated cleanup surfaces were `graphify-out/`, `.codex/graphify/`, `.codex/history.jsonl`, `.codex/memories/raw_memories.md`, and `./.graphify_detect.json`; final checks used both `find` and `rg` and found no remaining repo-local matches [Task 1]
- The repo now standardizes on `.codex/` rather than `.codex_config/pein/`. Because `.gitignore` is allowlist-based, any home migration needs allowlist updates, not just a new ignore rule. Historical `pein` strings may still appear in archived logs/sessions and should not be mistaken for live config [Task 2]
- Serena’s Codex behavior is controlled by the Serena startup arg `--context codex`, not by a separate Codex model/platform field. Without it, Serena falls back to `desktop-app`, which is why tools like `execute_shell_command` can still appear. The validated working stanza is `uv run --directory ... serena start-mcp-server --project /data/home/xiaoyan/AIteam/data/CoordExp --context codex` [Task 3]
- The user-approved subagent routing heuristic is: `gpt-5.4-mini` for pure information collection only; `gpt-5.4` with `medium` for bounded implementation and mechanical execution; `gpt-5.4` with `high` as the default frontier tier for debugging, review, audit, planning, and ambiguous implementation; `xhigh` only for the hardest or highest-stakes work [Task 4]
- For Codex CLI permissions, `--full-auto` is not full access. The high-permission mode is `--sandbox danger-full-access`, and `--yolo` is the most permissive shortcut if supported. Preserve `HTTP_PROXY`, `HTTPS_PROXY`, and `CODEX_HOME` when suggesting the corrected alias [Task 5]
- The self-improving split that the user wanted was portable skill definition separate from live shared memory: portable skill under the repo-local skill tree, mutable shared memory under `.self-improving/`, and both kept commit-safe because the memory is meant to travel with the branch [Task 2]

## Failures and how to do differently

- Do not keep stale operational guidance for removed tooling. The older Graphify install/scope work is historical only; the current-state memory for this repo should point to the cleanup and absence of Graphify, not to reinstall instructions [Task 1]
- Re-read live files immediately before patching. One cleanup patch failed because the expected Graphify block in `AGENTS.md` had already been removed; stale assumptions about text content are brittle during cleanup passes [Task 1]
- Broad `--hidden --no-ignore` searches across `.codex/log`, `.codex/sessions`, and similar archives create noisy false leads during config migrations. Scope the first pass to active config/tracked files, then decide separately whether logs need cleanup [Task 2]
- If local `codex --help` or repo grep is blocked by `bwrap: Failed to make / slave: Permission denied`, treat that as harness interference and avoid inferring Codex CLI behavior from local probing. Use official docs for launch-mode questions [Task 5]

# Task Group: CoordExp git hygiene, selective commits, worktrees, and merge recovery

scope: How the user wants local changes grouped, merged, pushed, and cleaned up across the main checkout and worktrees. Use for commit grouping, pull/merge conflict resolution, selective staging, or worktree integration.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this repo's git workflow and user preferences, but validate current branch/worktree state before applying because local dirtiness and remotes are time-sensitive.

## Task 1: Split mixed local changes into logical commits, pull `origin/main`, resolve conflicts, and finish with a clean branch

### rollout_summary_files

- rollout_summaries/2026-04-11T12-56-12-nOdN-git_commit_pull_merge_resolve_stage2_and_skill_work.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/11/rollout-2026-04-11T12-56-12-019d7c9d-3d2a-7461-b5e0-816258697ca7.jsonl, updated_at=2026-04-11T13:16:51+00:00, thread_id=019d7c9d-3d2a-7461-b5e0-816258697ca7, strongest grouped-commit plus merge-conflict playbook)

### keywords

- git pull --no-rebase origin main, merge conflict, grouped commits, git add -p, git diff --check, conda run -n ms, origin/main, split into multiple commits

## Task 2: Commit related routing/docs and Serena-memory maintenance while leaving unrelated dirt untouched

### rollout_summary_files

- rollout_summaries/2026-04-03T03-35-26-BLJm-commit_routing_docs_and_serena_memory_refresh.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/03/rollout-2026-04-03T03-35-26-019d5168-f4d9-73a2-8679-6d3c9b8be590.jsonl, updated_at=2026-04-03T03:57:12+00:00, thread_id=019d5168-f4d9-73a2-8679-6d3c9b8be590, selective maintenance commits)
- rollout_summaries/2026-04-01T14-44-28-5Epi-channel_b_cluster_aware_duplicate_targeting.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/01/rollout-2026-04-01T14-44-28-019d4980-c3fd-7691-b82a-ffa36e2c2d68.jsonl, updated_at=2026-04-03T07:32:47+00:00, thread_id=019d4980-c3fd-7691-b82a-ffa36e2c2d68, selective commit of only the OpenSpec folder)

### keywords

- git status --short --branch, unrelated dirty files, selective staging, git add openspec/changes/..., ahead 2, ignore the other dirty changes

## Task 3: Merge a validated worktree feature back to `main`, push, and clean up the worktree/branch

### rollout_summary_files

- rollout_summaries/2026-04-02T08-15-32-HgAt-center_size_bbox_supervision_worktree_merge_smoke.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/02/rollout-2026-04-02T08-15-32-019d4d43-0a53-7602-b578-f420283f738d.jsonl, updated_at=2026-04-03T07:40:50+00:00, thread_id=019d4d43-0a53-7602-b578-f420283f738d, worktree-to-main integration and cleanup)

### keywords

- git worktree, merge, push origin main, cleanup branch, stale index contention, single-GPU smoke, worktree remove, branch delete

## User preferences

- when the worktree contains multiple themes, the user said "Please help me properly commit the local changes and pull the remote and resolve the conflicts. Ask my clarifications when unclear" and later "Split into multiple commits in groups" -> pause for scope clarification when needed and default to grouped commits rather than one mega-commit [Task 1]
- when the user asks to "ignore the other dirty changes" or similar, stage only the intended files and leave unrelated dirt alone, even if those other changes are nearby in the worktree [Task 2]
- when the user says "please help me properly merge this worktree and cleanup the worktree and branch" and confirms `main` is clean, treat that as a request for full integration: merge, push/sync, delete the worktree, and delete the feature branch rather than stopping at a local commit [Task 3]
- if publication was interrupted or not explicitly requested, confirm before retrying `git push`; the user accepted local commits without always wanting an immediate push retry [Task 2]

## Reusable knowledge

- The initial repo-state inspection set that worked was `git status --short --branch`, `git branch --show-current`, `git remote -v`, `git diff --stat`, and `git diff --name-only`. That was enough to see when the branch was behind remote and whether the worktree naturally split into logical groups [Task 1]
- `git add -p` was the practical splitter when files mixed multiple topics. In this repo, Python verification should run via `conda run -n ms python ...` after merge/conflict edits [Task 1]
- A successful Stage-2 merge strategy was to take `origin/main` as the base in conflicted files, then re-apply the local feature carefully on top, followed by `git diff --check` and targeted tests to flush out missing compatibility imports/parameters [Task 1]
- Selective commit hygiene works well here: stage just the intended OpenSpec change folder or just the docs/memory files, keep unrelated files unstaged, and preserve them for later work [Task 2]
- For worktree feature integration, a real but minimal validation bar is acceptable: targeted tests plus one genuine single-GPU smoke can be enough before merging if the user framed the feature as lightweight and initial validation [Task 3]

## Failures and how to do differently

- Do not force one commit before pulling if `git status` shows a logically mixed worktree. Identify unrelated groups early and ask before committing everything together [Task 1]
- Merge conflict resolution in this repo can require reintroducing small compatibility symbols/parameters after taking remote versions. After conflict markers are gone, run the narrowest tests that exercise the touched code path before finalizing the merge [Task 1]
- Avoid parallel git-index activity during merge/cleanup. One worktree merge hit `Unable to write index` / stale index contention; the recovery was to abort stale merge state and retry sequentially with exclusive index access [Task 3]
- If a push was aborted, do not assume publication happened. Re-check `git status --short --branch` and confirm whether the user wants a new push attempt [Task 2]

# Task Group: CoordExp BaiduPCS-Go transfer workflow, skill packaging, and live monitoring

scope: Proven Baidu Netdisk upload/download workflow for this repo, including skill-first execution, tmux launches, progress estimation, and reusable handoff prompts. Use when `bypy` is failing, large folders need upload/download, or the user references the BaiduPCS-Go skill.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for Ubuntu-like environments and this repo's transfer conventions, but re-check cookie availability, remote paths, and skill paths in the active checkout.

## Task 1: Diagnose failing Baidu Netdisk uploads, switch from `bypy` to BaiduPCS-Go, and package the working workflow as a reusable skill

### rollout_summary_files

- rollout_summaries/2026-04-06T05-30-31-gxyA-baidupcsgo_upload_skill_creation_and_download_handoff.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/06/rollout-2026-04-06T05-30-31-019d6145-65b3-73c2-95d6-820cd8040eb4.jsonl, updated_at=2026-04-06T06:11:49+00:00, thread_id=019d6145-65b3-73c2-95d6-820cd8040eb4, base workflow and skill packaging)

### keywords

- bypy, Slice MD5 mismatch, 31064, file is not authorized, BaiduPCS-Go, browser cookies, tmux, upload_dir.sh, .codex/skills/baidupcsgo-upload, download prompt

## Task 2: Use the repo skill, run a detached tmux upload with the same remote relative path, update the skill for parallel upload/download, and estimate ETA from live evidence

### rollout_summary_files

- rollout_summaries/2026-04-11T12-46-43-NbNv-baidupcsgo_upload_tmux_parallel_skill_update.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/11/rollout-2026-04-11T12-46-43-019d7c94-8bc9-7712-85bb-849a374df681.jsonl, updated_at=2026-04-11T13:03:00+00:00, thread_id=019d7c94-8bc9-7712-85bb-849a374df681, latest tmux/progress and skill-parallelism evidence)

### keywords

- baidu_net_cookie.txt, tmux, baidupcs_stage1_upload, temp/baidupcs_stage1_upload.log, parallel upload, parallel download, BAIDUPCS_UPLOAD_FILE_THREADS, BAIDUPCS_DOWNLOAD_THREADS, ETA, remote ls

## User preferences

- when the user says "Please refer to `.codex/skills/baidupcsgo-upload`" -> inspect the skill first instead of improvising a fresh Baidu workflow [Task 2]
- when the user says "尽量上传我的整个文件夹,而不是压缩文件" -> default to directory upload rather than archive upload when the tool can preserve the tree directly [Task 1]
- when the user says "对于完整的上传我需要使用`tmux`" or asks to update a remote `output/` folder "with the same relative path in a `tmux` session" -> launch large transfers in detached tmux and preserve the same relative remote path under `/output` [Task 1][Task 2]
- when the user asks "How long will it take?" or "help me check the uploading process" -> estimate from live logs and remote file state, not from folder size alone [Task 2]
- when the user asks for another server handoff, provide a ready-to-paste prompt with concrete local/remote paths and validation steps, not a vague summary [Task 1]

## Reusable knowledge

- In this environment, `bypy` large-file failures that look like `Slice MD5 mismatch` can mask the real cause: `HTTP 403`, `error_code=31064`, `file is not authorized` during slice upload. The successful pivot was `qjfoidnh/BaiduPCS-Go` with browser-cookie login [Task 1]
- BaiduPCS-Go sees the real Netdisk root `/`, not the `/apps/bypy` sandbox. Remote target paths may need explicit `mkdir` chain creation under `/output/...` before upload [Task 1][Task 2]
- Conservative defaults that worked for large uploads were `--norapid -p 1 -l 1 --retry 8`. The later skill update parameterized concurrency rather than hard-coding it, exposing env knobs for both upload and download while keeping safe defaults [Task 1][Task 2]
- The durable repo skill surfaces are `.codex/skills/baidupcsgo-upload/SKILL.md`, `scripts/upload_dir.sh`, and `scripts/download_dir.sh`. The update added symmetric remote-to-local download support and knobs such as `BAIDUPCS_UPLOAD_FILE_THREADS`, `BAIDUPCS_UPLOAD_PARALLEL_FILES`, `BAIDUPCS_DOWNLOAD_THREADS`, and `BAIDUPCS_DOWNLOAD_PARALLEL_FILES` [Task 2]
- For live ETA, combine local file sizes, current shard progress from the log/tmux pane, and remote `BaiduPCS-Go ls` on the target directory. That produced a grounded ETA for the last shard instead of a hand-wavy whole-folder estimate [Task 2]

## Failures and how to do differently

- Do not stop at the top-level `Slice MD5 mismatch` message. Turn on enough logging to reveal the underlying HTTP/auth error before deciding the upload strategy [Task 1]
- Do not assume `bypy` remote paths equal BaiduPCS-Go remote paths. One tool sees `/apps/bypy`, the other sees `/` [Task 1]
- `tmux capture-pane` and naive `tail` reads can be too noisy for ETA because the log uses carriage-return progress updates. Cross-check the active shard and remote file visibility instead of trusting a single progress line [Task 2]

# Task Group: CoordExp Stage-2 diagnostics, pseudo-positive behavior, and duplicate/failure analysis

scope: Artifact-backed diagnosis of Stage-2 two-channel runs, especially invalid predictions, duplication-control stability, Channel-B FN ordering, and partial-annotation failure modes. Use when the user wants run diagnosis, mechanism-level reasoning, or symptom taxonomy from a specific artifact tree.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this repo's Stage-2 analysis surfaces and terminology, but metrics and conclusions are run-specific and should be revalidated against the exact artifact directory the user names.

## Task 1: Diagnose a Stage-2 pseudo-positive K=4 run where invalid preds are dominated by structural decode failures

### rollout_summary_files

- rollout_summaries/2026-04-11T12-53-20-xEEX-stage2_ab_invalid_preds_diagnosis.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/11/rollout-2026-04-11T12-53-20-019d7c9a-9d0c-7bf1-b7fe-f5490ccef563.jsonl, updated_at=2026-04-11T13:07:57+00:00, thread_id=019d7c9a-9d0c-7bf1-b7fe-f5490ccef563, symptom taxonomy and run-specific diagnosis)

### keywords

- invalid preds, prepare_failures, wrong_arity, missing_desc, unexpected_keys, truncated_rollout, parse_truncated_rate, gating_rejection_rate, pred_objects=0, max_new_tokens=3084

## Task 2: Check whether dup-targeting actually controls duplication and how FN objects are appended in the current Channel-B target

### rollout_summary_files

- rollout_summaries/2026-04-08T13-27-06-FL5P-stage2_lvis_proxy_dup_targeting_artifacts_and_fn_append_orde.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/08/rollout-2026-04-08T13-27-06-019d6d46-717b-7d83-a960-3fd73e3eb13d.jsonl, updated_at=2026-04-08T13:37:04+00:00, thread_id=019d6d46-717b-7d83-a960-3fd73e3eb13d, duplication stability and FN-tail ordering)

### keywords

- duplicate_burst_unlikelihood, near_iou90_pairs_same_desc_count, recovered_ground_truth_rate, pseudo_positive_selected_count, channel_b_prepare_failure, clean_prefix, fn_objs, append_text

## Task 3: Analyze Channel-B pseudo supervision under partial COCO annotation and crowded scenes

### rollout_summary_files

- rollout_summaries/2026-04-01T14-57-50-it4N-stage2_channel_b_pseudo_supervision_partial_annotation_analy.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/01/rollout-2026-04-01T14-57-50-019d498d-007f-7a22-9446-1eb1e1bb3a0d.jsonl, updated_at=2026-04-01T15:14:02+00:00, thread_id=019d498d-007f-7a22-9446-1eb1e1bb3a0d, mechanism-level critique and cautious fixes)

### keywords

- partial annotation, mechanism-level reasoning, Failure Modes, Root Causes, Proposed Fixes, Trade-offs, Hungarian matching, maskiou, unmatched audit, desc-neutral, token_ce, text_gate

## User preferences

- when the user asks to "explore and diagnose this experiment" and points to a concrete run directory, start from that exact artifact tree instead of answering generically from code or memory alone [Task 1]
- when the user asks for the "main symptom" of invalid preds, front-load a symptom taxonomy with representative failure classes and examples rather than broad speculation [Task 1]
- when the user asks whether missed objects were "Added in the last positions?" or similar, answer with the exact ordering rule and the code locations that enforce it, not just a conceptual summary [Task 2]
- when the user asks for mechanism analysis, they prefer "mechanism-level reasoning rather than high-level suggestions" and a structured report shape like "Failure Modes / Root Causes / Proposed Fixes / Trade-offs" [Task 3]

## Reusable knowledge

- In the diagnosed K=4 pseudo-positive run, invalid preds were mostly structural-output failures, not subtle geometry mistakes. Common classes were runaway bracket or punctuation tails, incomplete JSON, wrong bbox arity, missing `desc`, unexpected keys, and invalid coord-slot contents. Many invalid views hit `max_new_tokens=3084`, had `pred_objects=0`, and came from the `monitor_dumps/prepare_failures` surface [Task 1]
- Train-side sampled explorer rollouts and eval-side greedy rollouts are different failure surfaces. In the same run, greedy eval at step 300 remained reasonably healthy while sampled training rollouts were much noisier, so diagnosis must separate those paths [Task 1]
- The current Channel-B construction first forms a `clean_prefix` from retained accepted objects, then appends true unmatched GT objects as a tail fragment. In the historical tail-append behavior, `append_text` serializes `fn_objs` after the prefix rather than interleaving them into it [Task 2]
- Duplicate suppression in the analyzed LVIS-proxy pseudo-positive run was partial and unstable. Metrics such as `dup/near_iou90_pairs_same_desc_count` could drop on some batches but spike later; `pseudo_positive_selected_count` stayed at `0`, and recovered-GT rates were too noisy to claim robust FN improvement [Task 2]
- Under partial annotation, current Channel-B behavior is anchor-centric and can conflate hallucinations, real unlabeled objects, and duplicate-like re-enumerations. The analysis found a mechanism/spec mismatch: broad Channel-B prefix CE and `text_gate` surfaces can leak desc/text supervision onto retained unmatched anchors that were supposed to stay desc-neutral [Task 3]
- The repo already contains the right evidence path for such analysis: exact run artifacts, `logging.jsonl`, `monitor_dumps/prepare_failures`, the Stage-2 trainer/target-builder code, and the relevant OpenSpec/docs for unmatched promotion and LVIS-aware triage [Task 1][Task 3]

## Failures and how to do differently

- Do not summarize a noisy run as "controlled" just because some batches looked better. If late batches still collapse or the last logged duplicate metrics remain high, say the control is intermittent rather than stable [Task 2]
- Avoid claiming recall improvement without a matched baseline artifact. The analyzed dup-targeting run did not provide a clean same-surface comparison, so the right answer was uncertainty, not a positive claim [Task 2]
- Large `prepare_failures` directories should be aggregated before inspection. Looking at a few files without counting or clustering the failure families will miss the dominant symptom surfaces [Task 1]
- Under partial annotation, "unmatched" is not synonymous with hallucination. Future analyses should explicitly separate annotation incompleteness from model error and avoid over-promoting unmatched anchors without audit/calibration evidence [Task 3]

# Task Group: CoordExp Stage-2 config contracts, target-building knobs, and rollout-eval artifact surfaces

scope: Config-first Stage-2 changes that alter objective weights, Channel-B ordering semantics, proxy-dataset readiness, or eval artifact persistence. Use when editing Stage-2 YAML/schema/trainer contracts or explaining what a config really does.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this repo's Stage-2 config surfaces and invariants, but re-check current OpenSpec/docs when changing stable contracts or artifact names.

## Task 1: Audit proxy-dataset readiness, duplicate-targeting behavior, and enabled loss weights in a new 2B prod recipe

### rollout_summary_files

- rollout_summaries/2026-04-06T05-38-34-fMif-stage2_proxy_dataset_config_audit_and_loss_weights.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/06/rollout-2026-04-06T05-38-34-019d614c-c72a-7ef2-b63d-fb4742478e63.jsonl, updated_at=2026-04-06T06:01:17+00:00, thread_id=019d614c-c72a-7ef2-b63d-fb4742478e63, mechanism/weight audit before launch)

### keywords

- proxy_supervision, object_weight_mode, duplicate_burst_unlikelihood, adjacent_repulsion, loss weights, lvis_proxy, pseudo_positive, 2b prod config, train.coord.summary.json

## Task 2: Trace config inheritance and update a prod leaf to a CE+CIoU-only duplication ablation

### rollout_summary_files

- rollout_summaries/2026-04-08T13-22-09-bQJJ-stage2_config_ce_ciou_only_ablation.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/08/rollout-2026-04-08T13-22-09-019d6d41-ea96-7562-a816-fffdae8df4eb.jsonl, updated_at=2026-04-08T13:34:25+00:00, thread_id=019d6d41-ea96-7562-a816-fffdae8df4eb, list-replacement and ablation semantics)

### keywords

- ConfigLoader, extends, list replacement, pure ce, ciou, bbox_geo, coord_reg, loss_gradient_monitor, duplication ablation, stage2_ab.pipeline.objective

## Task 3: Add the Channel-B `insertion_order` knob and document the backward-compatible default

### rollout_summary_files

- rollout_summaries/2026-04-08T13-30-12-gOOZ-stage2_channel_b_insertion_order_knob.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/08/rollout-2026-04-08T13-30-12-019d6d49-46af-7bf2-ba55-e85f9d02a5c1.jsonl, updated_at=2026-04-08T14:49:02+00:00, thread_id=019d6d49-46af-7bf2-ba55-e85f9d02a5c1, actual knob and docs/spec updates)

### keywords

- insertion_order, tail_append, sorted, top-left sort, clean-prefix, FN append, target_builder, Stage2ABChannelBConfig, docs update

## Task 4: Persist eval-step rollout artifacts by default, then expose that as `rollout_matching.eval_detection.materialize_artifacts`

### rollout_summary_files

- rollout_summaries/2026-04-08T14-49-03-BE9Q-stage2_eval_rollout_artifacts_default_on_configurable.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/08/rollout-2026-04-08T14-49-03-019d6d91-7a59-7ac1-8a27-7d09176f1a7a.jsonl, updated_at=2026-04-08T15:46:40+00:00, thread_id=019d6d91-7a59-7ac1-8a27-7d09176f1a7a, current eval-artifact contract)

### keywords

- rollout_matching.eval_detection.materialize_artifacts, gt_vs_pred.jsonl, gt_vs_pred_scored.jsonl, raw_rollouts.jsonl, pred_token_trace.jsonl, raw_output_json, evaluate_and_save

## User preferences

- for Stage-2 config changes, the user explicitly asked: "Please explore my whole config system and inheritance and loss modules first." -> trace inheritance plus the actual loss/monitoring modules before editing the YAML [Task 2]
- when the user says "Only keep the standard CE and CIOU losses" or similar, take that literally and disable every other relevant regression-style subterm rather than stopping at the most obvious weight [Task 2]
- when adding a new behavior knob, the user wanted "current tail" and "sorted mode" while keeping the default compatible "so that we don't need full migration over the pass configs" -> preserve existing behavior by default and avoid broad config migration unless explicitly requested [Task 3]
- when changing training-time eval behavior, the user asked to "fully capture and persist all `rollout` outputs at each evaluation step", make it "consistent in format and granularity" with offline tooling, and later "Revert this as a tunable config and default to be `true`" -> make the surface offline-compatible, configurable, and default-on [Task 4]
- when asked to update docs, the user wanted the behavior docs/specs kept in sync immediately with the code change, not as a later cleanup [Task 3][Task 4]

## Reusable knowledge

- `ConfigLoader.load_yaml_with_extends()` deep-merges dicts but replaces lists wholesale. For Stage-2 leaves, `stage2_ab.pipeline.objective` cannot be partially patched; restate the whole list when changing one objective entry [Task 2]
- The proxy-dataset audit found that current Stage-2 runtime does not yet apply proxy-tier weights even if the dataset carries proxy distinctions. The active objective modules do not consume `object_weight_mode` in the current Stage-2 path, so a launch would not realize the user's intuition about softer proxy weighting without more plumbing [Task 1]
- `adjacent_repulsion` is a coord regularizer, not the canonical duplicate-targeting mechanism. The canonical duplicate control in the current runtime is `loss_duplicate_burst_unlikelihood`; newer cluster-aware duplicate semantics were still at the spec/handoff stage during these runs [Task 1]
- The inserted-order contract now has two modes: `tail_append` preserves the historical clean-prefix plus FN-tail behavior, while `sorted` rebuilds the final Channel-B sequence by top-left sorting retained accepted-clean objects plus FN objects using the stage-1 sorted ordering contract. `tail_append` remains the default [Task 3]
- The current eval-artifact contract lives under `rollout_matching.eval_detection.materialize_artifacts: bool = True`. When enabled, training-time eval persists offline-compatible artifacts under `training.output_dir/eval_detection/step_<global_step>/`, and `raw_output_json` must be preserved for confidence post-op / trace reconstruction parity [Task 4]
- The validated docs/spec surfaces for these behaviors are `docs/ARTIFACTS.md`, `docs/training/STAGE2_RUNBOOK.md`, and the relevant OpenSpec specs. Those should be updated in the same change when behavior or artifact contracts move [Task 3][Task 4]

## Failures and how to do differently

- "Pure CE" is ambiguous in this codebase. It can mean monitoring only, objective changes, or both. Confirm the intended surface instead of assuming the phrase maps to one knob [Task 2]
- A leaf config that changes one inherited objective without restating the whole objective list will silently drop other inherited objectives because list values are replaced, not merged [Task 2]
- `configs/stage2_two_channel/base.yaml` alone can fail some config-load tests because `custom.object_field_order` is missing; use the appropriate leaf/config family when validating Stage-2 settings [Task 3]
- Hardwiring artifact dumping directly into code was not the right endpoint. The durable shape was a config knob in the existing `rollout_matching.eval_detection` hierarchy with explicit default-on semantics [Task 4]

# Task Group: CoordExp inference config authoring and checkpoint-surface routing

scope: Creating or locating runnable inference YAMLs and matching them to the correct checkpoint/dataset family. Use when the user asks for a new infer preset, wants the exact runnable command, or names a precise folder under `configs/infer`.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this repo's infer config conventions, but checkpoint paths and dataset manifests are checkout-specific and must be revalidated.

## Task 1: Create a COCO 1024 inference preset for the 2B LVIS-proxy merged checkpoint

### rollout_summary_files

- rollout_summaries/2026-04-02T07-02-45-e1qC-stage1_2b_coco1024_lvis_proxy_infer_config.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/02/rollout-2026-04-02T07-02-45-019d4d00-666e-7b73-a78a-5e9f1a9f10e6.jsonl, updated_at=2026-04-02T07:16:25+00:00, thread_id=019d4d00-666e-7b73-a78a-5e9f1a9f10e6, exact folder/command answer)

### keywords

- configs/infer/coco_1024, coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged, run_infer.py, val_200_lvis_proxy_merged.yaml, output/stage1_2b, val.coord.jsonl

## User preferences

- when the user names a target folder like "Please create a file under `configs/infer/coco_1024/`", create the file there rather than describing or creating it elsewhere [Task 1]
- when the user says "mimic" an existing preset and "I want to see how it performs", preserve the nearby config shape and minimize workflow drift [Task 1]
- when the user says "only use serena in python search", prefer Serena semantic/file search over broad shell scanning for Python-file navigation [Task 1]
- when the user asks whether they can directly run `scripts/run_infer.py`, answer with the exact runnable command, not just "yes" [Task 1]

## Reusable knowledge

- The closest template for this checkpoint family was `configs/infer/coco_1024/val_200_lvis_proxy_merged.yaml`, and the 2B checkpoint directory `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-merged` was a valid HF-style export, so `infer.model_checkpoint` could point to it directly [Task 1]
- The matching LVIS-proxy 1024 validation JSONL for this family is `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl` [Task 1]
- The established infer knobs for this family were `prompt_variant: coco_80`, `object_field_order: desc_first`, `object_ordering: sorted`, `mode: auto`, `pred_coord_mode: auto`, `generation.temperature: 0.0`, `top_p: 0.9`, `max_new_tokens: 3084`, `repetition_penalty: 1.05`, `batch_size: 4`, `seed: 42`, `limit: 200`, and `detect_samples: 128` [Task 1]
- The resulting YAML is directly runnable with `PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config ...` and is intentionally infer-only (`stages.eval: false`, `stages.vis: false`) rather than a full scoring bundle [Task 1]

## Failures and how to do differently

- Broad repo scans across large `output/` or data trees were too heavy and got interrupted. For similar authoring tasks, narrow to the closest template file, the named checkpoint directory, and the specific dataset manifest [Task 1]
- If the user names the folder, do not first create a file under a different surface such as `configs/bench/`; follow the folder constraint from the start [Task 1]

# Task Group: CoordExp spec-driven change work, lightweight feature implementation, and handoff packaging

scope: Brainstorm-first feature work that moves through OpenSpec, worktrees, targeted implementation, and handoff docs. Use when the user wants a narrow change scoped lightly, asks for a worktree/OpenSpec, or needs a self-contained handoff for another agent.
applies_to: cwd=/data/home/xiaoyan/AIteam/data/CoordExp; reuse_rule=safe for this repo's OpenSpec-first feature workflow, but exact change names, worktree paths, and implementation details are task-specific.

## Task 1: Brainstorm, spec, implement, smoke-test, and merge center-size bbox supervision as an internal loss-space change

### rollout_summary_files

- rollout_summaries/2026-04-02T08-15-32-HgAt-center_size_bbox_supervision_worktree_merge_smoke.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/02/rollout-2026-04-02T08-15-32-019d4d43-0a53-7602-b578-f420283f738d.jsonl, updated_at=2026-04-03T07:40:50+00:00, thread_id=019d4d43-0a53-7602-b578-f420283f738d, full brainstorm-to-merge feature loop)

### keywords

- brainstorm first, worktree, OpenSpec, center_size, xyxy, bbox_geo, smoke, single-GPU, internal parameterization, merge cleanup

## Task 2: Reuse offline duplicate-detector evidence to draft and fast-forward a cluster-aware Channel-B OpenSpec change for handoff

### rollout_summary_files

- rollout_summaries/2026-04-01T14-44-28-5Epi-channel_b_cluster_aware_duplicate_targeting.md (cwd=/data/home/xiaoyan/AIteam/data/CoordExp, rollout_path=/data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/01/rollout-2026-04-01T14-44-28-019d4980-c3fd-7691-b82a-ffa36e2c2d68.jsonl, updated_at=2026-04-03T07:32:47+00:00, thread_id=019d4980-c3fd-7691-b82a-ffa36e2c2d68, detector-backed OpenSpec handoff)

### keywords

- openspec, channel-b-cluster-aware-duplicate-targeting, fastforward those *.md, detector background, temp/dup_postop_eval.py, proposal, design, tasks, commit local change only

## User preferences

- on new ideas, the user asked to "brainstorm and discuss" first and only then "create the worktree and openspec if promising" -> discussion-first gating is the preferred default before implementation [Task 1]
- when scoping a first implementation, the user asked to "Make the implementation easy/light as possible" -> prefer the narrowest viable internal change, not a public-format migration or broad refactor [Task 1]
- when the user says "Fastforward those *.md", they want the OpenSpec artifact set carried through in one pass rather than incremental drafting [Task 2]
- when the user says "Please also add the background about your previous `detector` design and effect. I'll hand over to other agent to continue the spec and hence need the full `picture`." -> handoff docs should preserve empirical background and artifacts, not only the target contract text [Task 2]

## Reusable knowledge

- For bbox supervision changes, the durable invariant is to preserve the canonical outward `bbox_2d` / `xyxy` contract and treat `center_size` as loss-space only. That let the feature stay config-first and low blast radius while still exercising the new geometry path [Task 1]
- A good lightweight validation bar for this repo was targeted tests plus one real single-GPU smoke from the worktree. The successful smoke used `conda run -n ms bash scripts/train.sh` with a Stage-2 smoke config and produced the expected non-zero bbox metrics in a 2-step run [Task 1]
- The OpenSpec workflow that worked here was `openspec new change`, then `openspec status`, then `openspec instructions` for `proposal`, `design`, `specs`, and `tasks`, followed by a selective commit of just the change directory if unrelated dirt exists [Task 2]
- The duplicate-detector study was useful handoff evidence even though it was not the final training solution. It showed cluster-shaped collapse and measured a positive AP shift after stripping duplicate FP mass, which justified a spec change focused on duplicate-target construction rather than a wholesale replacement of `loss_duplicate_burst_unlikelihood` [Task 2]

## Failures and how to do differently

- If running from a worktree, make sure the worktree can actually see the prepared data bundle and checkpoint paths. One smoke failed until the worktree-local symlinks for data/checkpoint were added [Task 1]
- Launch repo training entrypoints through `conda run -n ms`; a direct launch outside `ms` failed with `ModuleNotFoundError: No module named 'yaml'` [Task 1]
- An OpenSpec handoff that lacks the detector background is incomplete for downstream implementers. Preserve the empirical script/artifacts and the precision-recall trade-off, not just the intended future design [Task 2]
