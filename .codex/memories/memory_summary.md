## User Profile

The user is actively developing and evaluating the CoordExp research stack, especially Stage-2 two-channel training, artifact analysis, config contracts, and reproducible experiment workflows. They use Codex as a hands-on collaborator inside the repo and routinely ask for concrete edits, exact commands, artifact-backed diagnosis, and git hygiene rather than high-level advice.

They care a lot about current-state correctness and about not having to repeat operating constraints. Repeated patterns include: asking for mechanism-level reasoning over generic summaries, requesting exact file/path placement when they name a target folder, wanting narrow config-first changes that preserve existing contracts by default, and preferring local repo conventions to be made explicit in tracked instructions or memory. They also use detached/tmux workflows for long-running transfers or jobs, ask for realistic ETAs from live evidence, and expect the agent to preserve unrelated local work unless explicitly told to discard it.

On coding and research tasks, "good" usually means: start from the exact artifact path or config they pointed at, trace the real code/config path first, preserve compatibility unless the user asks for migration, update docs/specs together with behavior changes when contracts move, and validate on the narrowest realistic surface. On meta-tooling tasks, they want concrete answers grounded in the actual config file or official docs, not speculation.

They also appear to prefer efficient, capacity-aware agent behavior. They explicitly rejected blanket maximum reasoning for subagents and instead want model effort routed by task difficulty. When asking for handoffs to other agents or machines, they want ready-to-run prompts or workflows with exact paths and verification steps.

## User preferences

- When they point to an exact run directory and ask to diagnose it, start from that artifact tree and answer from the concrete evidence there.
- When they ask for mechanism analysis, prefer code-path tracing and structured sections like "Failure Modes / Root Causes / Proposed Fixes / Trade-offs" over generic advice.
- When they ask for the "main symptom" of a failure, front-load the symptom taxonomy and representative examples.
- When they ask whether objects are "Added in the last positions?" or similar, answer with the exact ordering rule and the code/config locations that enforce it.
- When they say "Please explore my whole config system and inheritance and loss modules first.", do the inheritance/loss tracing before editing config leaves.
- When they say "Only keep the standard CE and CIOU losses" or equivalent, interpret that literally and disable the other relevant subterms, not just the obvious one.
- When they ask for a new behavior knob, preserve backward-compatible defaults unless they explicitly ask for migration.
- When they request docs updates after behavior changes, update docs/specs in the same change rather than leaving them for later.
- When they name a target folder such as `configs/infer/coco_1024/`, create the file there instead of proposing alternatives.
- When they say "mimic" an existing preset/config, keep the shape close to the nearby template and minimize workflow drift.
- When they ask whether they can directly run a script with a new config, include the exact runnable command.
- When they say "only use serena in python search", prefer Serena semantic/file search over broad shell scanning for Python navigation.
- When the worktree is mixed, default to grouped commits and ask clarifications when scope is unclear instead of making one big commit.
- When they ask to "ignore the other dirty changes", stage only the intended files and leave unrelated dirt untouched.
- When they ask to merge/cleanup a worktree and confirm `main` is clean, carry the task through merge, push/sync if requested, and branch/worktree cleanup.
- If a push was interrupted or not explicitly requested, confirm before retrying.
- When they explicitly ask to drop local changes and use remote latest, show the diff and then prefer the remote version/reset path.
- When they reference `.codex/skills/baidupcsgo-upload`, inspect that skill first instead of improvising a fresh transfer workflow.
- For large Baidu transfers, default to directory upload, detached `tmux`, and preserving the same relative remote path under `/output`.
- When they ask "How long will it take?" for a transfer, estimate from live logs plus remote file state rather than folder size alone.
- When they need a handoff to another machine/agent, give a ready-to-paste prompt with exact paths, workflow constraints, and validation steps.
- When they say a tool/framework is not suitable and ask to "铲除" it, treat that as a true cleanup request: remove artifacts, repo-local references, prompts, and memory/history traces rather than preserving the old setup.
- For local agent setup, prefer `.codex/` as the canonical repo-local Codex home in this checkout.
- For Serena/Codex questions, inspect the real config stanza and explain behavior from concrete file/code evidence.
- For Codex launch-permission questions, answer from current official docs and preserve their proxy / `CODEX_HOME` alias shape while changing only the Codex flags.
- For subagents, route model/reasoning effort by task difficulty instead of defaulting everything to maximum effort.
- On new feature ideas, they prefer to brainstorm/discuss first, then create the worktree/OpenSpec if the idea looks promising.
- For first implementations, default to the lightest viable internal change rather than broad output-format or contract migration.
- When they say "Fastforward those *.md", they want the full OpenSpec artifact set written in one pass.
- When preparing a handoff for another agent, include the empirical background and artifact pointers, not just the intended future design.

## General Tips

- In this repo, `conda run -n ms python ...` is the safe default for Python checks and `conda run -n ms` should wrap training/inference validation too.
- The user consistently values exact path placement and exact cwd applicability. Preserve those details in explanations and when creating files.
- Prefer the narrowest realistic verification surface: targeted tests, config parse checks, or one real smoke run rather than broad suites unless the change surface warrants more.
- For Stage-2 analysis, separate train-side sampled explorer rollouts from eval-side greedy rollouts; they can show very different failure surfaces.
- For Stage-2 config work, remember that config dicts deep-merge but lists replace wholesale; leaf objective edits usually need the full list restated.
- For repo-local agent config questions, distinguish live config from archived `.codex/log`, `.codex/sessions`, or other historical artifacts to avoid noisy false leads.
- For Codex launch-mode questions in this environment, local probing may be distorted by an outer sandbox/wrapper; official docs are more reliable than `codex --help` here.
- For BaiduPCS-Go progress checks, combine the active shard progress with remote `ls`; log lines alone can be misleading because of carriage-return updates.
- Treat removed tooling as removed. Historical rollout summaries about old Graphify setup are no longer the current operating state for this checkout.

## What's in Memory

### /data/home/xiaoyan/AIteam/data/CoordExp

#### 2026-04-20

- graphify cleanup and repo-local trace removal: graphify, graphify-out, .codex/history.jsonl, .codex/graphify, .graphify_detect.json
  - desc: Search this first when a task mentions removing abandoned tooling, cleaning repo-local agent traces, or confirming whether Graphify still exists in the current checkout. Covers the successful removal of repo-local Graphify artifacts/references and the verification pattern that showed no remaining matches in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.
  - learnings: The active state is "Graphify removed", not "Graphify installed". Cleanup had to scrub `.codex` memory/history surfaces in addition to deleting `graphify-out/`, and serial post-delete verification was more reliable than concurrent checks.

### /data/home/xiaoyan/AIteam/data/CoordExp

#### 2026-04-11

- grouped commits, pull, merge conflict resolution, and clean final branch: git pull --no-rebase origin main, git add -p, split into multiple commits, merge conflict, conda run -n ms
  - desc: Search this for user requests to commit mixed local changes "properly", pull remote, resolve conflicts, and leave a clean branch. Covers the grouped-commit workflow, conflict-resolution approach, and the user's preference for asking clarifications when scope is unclear.
  - learnings: Mixed worktrees should be split into logical commits before pulling. The successful conflict strategy was to take `origin/main` as the base in conflicted files, re-apply the local feature, then run narrow verification.

- BaiduPCS-Go tmux upload, parallel skill update, and ETA estimation: baidupcsgo-upload, tmux, baidupcs_stage1_upload, temp/baidupcs_stage1_upload.log, BAIDUPCS_UPLOAD_FILE_THREADS
  - desc: Search this when the user references `.codex/skills/baidupcsgo-upload`, wants a detached `tmux` transfer, asks for parallel upload/download knobs, or wants a live ETA from an ongoing Baidu Netdisk transfer.
  - learnings: The skill should stay symmetric for upload and download. ETA was best estimated from current shard progress plus remote `ls`, not from `tmux capture-pane` alone.

- Stage-2 invalid-pred diagnosis and symptom taxonomy: invalid preds, prepare_failures, wrong_arity, missing_desc, unexpected_keys, max_new_tokens=3084
  - desc: Search this when diagnosing a Stage-2 run with many invalid predictions or when the user asks for the "main symptom" of invalid preds. Covers the sampled-rollout failure classes and the difference between noisy train-side explorer rollouts and healthier eval-side greedy rollouts.
  - learnings: The dominant failure surface was format collapse, not subtle geometry mismatch. Aggregate `monitor_dumps/prepare_failures` first, then classify the dominant symptom families.

### /data/home/xiaoyan/AIteam/data/CoordExp

#### 2026-04-08

- Stage-2 Channel-B insertion-order knob and backward-compatible docs/spec updates: insertion_order, tail_append, sorted, top-left sort, clean-prefix, FN append
  - desc: Search this for Channel-B ordering semantics, whether missed/FN objects stay at the tail, or how to make ordering configurable without migrating every existing config. Includes the real target-builder semantics and the docs/spec surfaces updated alongside the code.
  - learnings: `tail_append` remains the backward-compatible default, while `sorted` does a final top-left sort over retained accepted objects plus FN objects. The user wanted the new knob without forcing a full config migration.

- Stage-2 eval artifact dumping under rollout_matching.eval_detection: materialize_artifacts, gt_vs_pred_scored.jsonl, raw_rollouts.jsonl, pred_token_trace.jsonl, raw_output_json
  - desc: Search this when changing training-time eval persistence or trying to match offline infer/eval artifacts. Covers the default-on config knob `rollout_matching.eval_detection.materialize_artifacts` and the exact artifact contract preserved at eval steps.
  - learnings: The right home is the existing rollout-eval config hierarchy, not an ad hoc trainer flag. `raw_output_json` must be kept for downstream confidence/post-op parity.

- Stage-2 proxy-dataset audit, CE+CIoU ablation, and duplicate-targeting contract checks: proxy_supervision, object_weight_mode, ConfigLoader, list replacement, duplicate_burst_unlikelihood, adjacent_repulsion
  - desc: Search this for "what losses are really enabled", "do proxy weights actually apply", or when editing prod Stage-2 leaves. Covers the mechanism audit, the CE+CIoU-only ablation, and the config-loader/list-replacement pitfall.
  - learnings: Proxy-tier weighting was not yet active in the runtime path despite the dataset carrying proxy distinctions. Leaf objective edits require restating the entire list because lists replace wholesale.

### Older Memory Topics

#### /data/home/xiaoyan/AIteam/data/CoordExp

- Codex/Serena repo-local setup and launch behavior: .codex, .codex_config/pein, --context codex, danger-full-access, gpt-5.4-mini, xhigh
  - desc: Covers repo-local Codex home migration, Serena MCP `--context codex`, docs-backed full-access launch guidance, portable skill vs shared `.self-improving/` memory, and the user's tiered subagent model-allocation policy. Use when working on local agent setup or answering Codex/Serena environment questions in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- BaiduPCS-Go upload troubleshooting and skill creation: bypy, Slice MD5 mismatch, 31064, browser cookies, upload_dir.sh, download prompt
  - desc: Earlier foundational Baidu Netdisk workflow memory: switching from `bypy` to BaiduPCS-Go, packaging the workflow as a reusable skill, and producing a ready-to-run download prompt for another machine. Use when a fresh transfer workflow needs the root-cause/fallback playbook.

- COCO 1024 infer preset for the 2B LVIS-proxy merged checkpoint: configs/infer/coco_1024, run_infer.py, output/stage1_2b, val_200_lvis_proxy_merged.yaml
  - desc: Covers creating the infer YAML under the exact user-requested folder, matching it to the HF export checkpoint, and returning the exact runnable `scripts/run_infer.py` command. Use when the user asks for a new infer preset rather than a full eval bundle.

- Center-size bbox supervision worktree feature loop: brainstorm first, center_size, xyxy, single-GPU smoke, worktree cleanup
  - desc: Covers brainstorm-first scoping, OpenSpec/worktree setup, narrow internal-loss implementation, one real smoke run, and merge/push/worktree cleanup. Use when the user wants a lightweight feature implemented end-to-end with minimal blast radius.

- Channel-B cluster-aware duplicate-targeting OpenSpec handoff: openspec, channel-b-cluster-aware-duplicate-targeting, fastforward those *.md, detector background
  - desc: Covers detector-backed spec drafting, preserving empirical background for another implementer, and selectively committing only the OpenSpec change while leaving unrelated local dirt untouched.

- Stage-2 partial-annotation and crowded-scene pseudo-supervision analysis: partial annotation, mechanism-level reasoning, unmatched audit, desc-neutral, token_ce, text_gate
  - desc: Mechanism-level analysis of how partial COCO annotation interacts with Channel-B matching, triage, pseudo positives, and duplicate suppression. Use when the user wants a structured causal explanation and lightweight SFT-style fixes rather than generic training advice.
