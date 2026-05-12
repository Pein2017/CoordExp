## User Profile

The user is actively developing and evaluating the CoordExp research stack, and they also use Codex to maintain the surrounding operational tooling: data provenance, large-asset sync workflows, repo-local agent setup, experiment diagnosis, and exact config contracts. They want concrete edits, exact paths, exact commands, artifact-backed reasoning, and narrow verification rather than generic advice.

They care strongly about current-state correctness and about not having to restate operating constraints. Repeated patterns include: start from the exact artifact path or config they named, trace the real code or config path before answering, preserve compatibility unless they ask for migration, and update docs or manifests together with behavior changes when contracts move. They routinely ask for commit hygiene, scoped staging, and ready-to-run handoff prompts for other agents or machines.

They prefer workflows that are portable and explicit. When something should travel across environments, they want the policy and semantics encoded in the skill, manifest, or tracked document itself rather than left in chat context. They also distinguish carefully between raw data, processed data, and cache or output churn, and they care about reproducibility across machines without expensive or unsafe whole-tree synchronization.

Memory currently spans two CoordExp checkouts: recent operational work in `cwd=/data/CoordExp` and older research or tooling work in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`. Future agents should preserve those cwd boundaries and avoid mixing checkout-specific facts.

## User preferences

- When they point to an exact run directory and ask to diagnose it, start from that artifact tree and answer from the concrete evidence there.
- When they ask for mechanism analysis, prefer code-path tracing and structured sections like “Failure Modes / Root Causes / Proposed Fixes / Trade-offs” over generic advice.
- When they ask for the “main symptom” of a failure, front-load the symptom taxonomy and representative examples.
- When they ask whether objects were “Added in the last positions?” or similar, answer with the exact ordering rule and the code or config locations that enforce it.
- When they say “Please explore my whole config system and inheritance and loss modules first.”, do the inheritance and loss tracing before editing config leaves.
- When they say “Only keep the standard CE and CIOU losses” or equivalent, interpret that literally and disable the other relevant subterms.
- When they ask for a new behavior knob, preserve backward-compatible defaults unless they explicitly ask for migration.
- When they request docs updates after behavior changes, update docs or specs in the same change rather than leaving them for later.
- When they name a target folder such as `configs/infer/coco_1024/`, create the file there instead of proposing alternatives.
- When they say “mimic” an existing preset or config, keep the shape close to the nearby template and minimize workflow drift.
- When they ask whether they can directly run a script with a new config, include the exact runnable command.
- When they say “only use serena in python search”, prefer Serena semantic or file search over broad shell scanning for Python navigation.
- When the worktree is mixed, default to grouped commits and ask for clarification when scope is unclear instead of making one big commit.
- When they ask to “ignore the other dirty changes”, stage only the intended files and leave unrelated dirt untouched.
- If a push was interrupted or not explicitly requested, confirm before retrying.
- When they explicitly ask to drop local changes and use remote latest, show the diff and then prefer the remote version or reset path.
- When they reference `.codex/skills/baidupcsgo-upload`, inspect that skill first instead of improvising a fresh transfer workflow.
- For large Baidu transfers, default to directory upload, detached `tmux`, and preserving the same relative remote path under `/output`.
- When they ask “How long will it take?” for a transfer, estimate from live logs plus remote file state rather than folder size alone.
- When they need a handoff to another machine or agent, give a ready-to-paste prompt with exact paths, workflow constraints, and validation steps.
- When they say a tool or framework is not suitable and ask to “铲除” it, treat that as a true cleanup request: remove artifacts, repo-local references, prompts, and memory or history traces rather than preserving the old setup.
- For local agent setup in the older checkout, prefer `.codex/` as the canonical repo-local Codex home.
- For Serena or Codex questions, inspect the real config stanza and explain behavior from concrete file or code evidence.
- For Codex launch-permission questions, answer from current official docs and preserve their proxy or `CODEX_HOME` alias shape while changing only the Codex flags.
- For subagents, route model or reasoning effort by task difficulty instead of defaulting everything to maximum effort.
- On new feature ideas, they prefer to brainstorm and discuss first, then create the worktree or OpenSpec if the idea looks promising.
- For first implementations, default to the lightest viable internal change rather than broad output-format or contract migration.
- When they say “Fastforward those *.md”, they want the full OpenSpec artifact set written in one pass.
- When preparing a handoff for another agent, include the empirical background and artifact pointers, not just the intended future design.
- When they say “把暂时不需要的数据给完全删掉以节省磁盘”, they want destructive cleanup, not a report-only audit.
- When they correct “VG raw也可以保留”, treat `public_data/vg/raw` as a keep candidate in similar cleanup runs unless they say otherwise.
- When they say a processed data root is in active use, such as “你先不用动这一个数据集的源数据”, treat that root as no-touch unless they explicitly approve rewriting or regeneration.
- When they ask for a separate `public_data*` record so cross-node exports from raw are identical, treat Git-tracked provenance manifests as the source of truth for processed `public_data`, not disk mirroring.
- When they ask for `*.jsonl` training-sample hashes because they are “比较便宜”, default to per-JSONL checksums rather than whole-folder hashing.
- When they say an unproduced dataset “不用追踪/记录”, do not track unmaterialized variants as current canonical artifacts.
- When they explicitly say “请commit and push这些 changes” and mention another environment that will “先pull再执行校验”, finish the workflow with a push plus a pull-first verification handoff.
- When they say the sync idea is “好像好复杂” and ask to “打包成一个 skills”, move the operational logic into a portable self-contained skill bundle instead of normal repo code.
- When they say another environment should also “领悟到精髓并执行”, encode the workflow semantics and safety policy in `SKILL.md`, references, or manifests rather than relying on chat context.

## General Tips

- Preserve cwd boundaries. Current memory spans both `/data/CoordExp` and `/data/home/xiaoyan/AIteam/data/CoordExp`, and reuse rules are often checkout-sensitive.
- In this repo, `conda run -n ms python ...` is the safe default for targeted Python checks, tests, and validation scripts.
- Prefer the narrowest realistic verification surface: targeted tests, config parse checks, or one real smoke run rather than broad suites unless the change surface warrants more.
- For Stage-2 analysis, separate train-side sampled explorer rollouts from eval-side greedy rollouts; they can show very different failure surfaces.
- For Stage-2 config work, remember that config dicts deep-merge but lists replace wholesale; leaf objective edits usually need the full list restated.
- For repo-local agent config questions, distinguish live config from archived `.codex/log`, `.codex/sessions`, or other historical artifacts to avoid noisy false leads.
- For Codex launch-mode questions in this environment, local probing may be distorted by an outer sandbox or wrapper; official docs are more reliable than local CLI probing.
- For BaiduPCS-Go progress checks, combine active shard progress with remote `ls`; log lines alone can be misleading because of carriage-return updates.
- Treat removed tooling as removed. Historical rollout summaries about old Graphify setup are not the current operating state for that checkout.
- For `public_data` reproducibility, prefer manifest-driven regeneration contracts plus targeted tests over whole-tree sync.
- For processed-data alignment across nodes, JSONL-only checksums are the accepted cheap sentinel; do not expand hashing to raw images or whole directories unless the user asks for that.
- For portable operational workflows, package the semantics and helper scripts inside a self-contained skill bundle rather than scattering logic through the codebase.

## What is in Memory

### /data/CoordExp

#### 2026-05-11

- `public_data` cleanup, provenance manifests, and JSONL checksum contract: public_data, manifests/public_data_provenance, jsonl_training_samples_only, tests/test_public_data_provenance_manifests.py
  - desc: Search this first when the task involves deleting stale `public_data`, defining what processed datasets are currently canonical, adding or validating Git-tracked provenance manifests, or preparing another environment to regenerate and verify the same processed datasets in `cwd=/data/CoordExp`.
  - learnings: The accepted contract is manifest-driven reproducibility for the three materialized COCO1024 processed roots, plus cheap JSONL-only SHA256 checksums. Active training roots are no-touch, and unmaterialized variants should not stay in the current provenance set.

- portable Baidu append-only union sync skill packaging: baidudisk-union-sync, BaiduPCS-Go, append-only union sync, config-template.json, semantics.md
  - desc: Search this when the user wants a reusable cross-machine large-asset sync workflow packaged as a self-contained Codex skill instead of repo code in `cwd=/data/CoordExp`. Covers the portable skill layout, safety semantics, validation steps, and scoped commit behavior.
  - learnings: The durable model is append-only union sync, not mirror sync: upload new files, pull missing files, stop on conflicts, and keep deletes or overwrites manual. The policy should live in the skill bundle itself so another agent can execute it without extra chat context.

### /data/home/xiaoyan/AIteam/data/CoordExp

#### 2026-04-20

- graphify cleanup and repo-local trace removal: graphify, graphify-out, .codex/history.jsonl, .codex/graphify, .graphify_detect.json
  - desc: Search this first when a task mentions removing abandoned tooling, cleaning repo-local agent traces, or confirming whether Graphify still exists in the older checkout. Covers the successful removal of repo-local Graphify artifacts and references plus the verification pattern that showed no remaining matches in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.
  - learnings: The active state is “Graphify removed”, not “Graphify installed”. Cleanup had to scrub `.codex` memory and history surfaces in addition to deleting `graphify-out/`.

### /data/home/xiaoyan/AIteam/data/CoordExp

#### 2026-04-11

- grouped commits, pull, merge conflict resolution, and clean final branch: git pull --no-rebase origin main, git add -p, split into multiple commits, merge conflict, conda run -n ms
  - desc: Search this for user requests to commit mixed local changes properly, pull remote, resolve conflicts, and leave a clean branch in the older checkout. Covers the grouped-commit workflow, conflict-resolution approach, and the preference for asking clarifications when scope is unclear.
  - learnings: Mixed worktrees should be split into logical commits before pulling. The successful conflict strategy was to take `origin/main` as the base in conflicted files, re-apply the local feature, then run narrow verification.

- BaiduPCS-Go tmux upload, parallel skill update, and ETA estimation: baidupcsgo-upload, tmux, baidupcs_stage1_upload, BAIDUPCS_UPLOAD_FILE_THREADS
  - desc: Search this when the user references `.codex/skills/baidupcsgo-upload`, wants a detached `tmux` transfer, asks for parallel upload or download knobs, or wants a live ETA from an ongoing Baidu Netdisk transfer.
  - learnings: The skill should stay symmetric for upload and download. ETA was best estimated from current shard progress plus remote `ls`, not from pane output alone.

- Stage-2 invalid-pred diagnosis and symptom taxonomy: invalid preds, prepare_failures, wrong_arity, missing_desc, unexpected_keys, max_new_tokens=3084
  - desc: Search this when diagnosing a Stage-2 run with many invalid predictions or when the user asks for the main symptom of invalid preds. Covers the sampled-rollout failure classes and the difference between noisy train-side explorer rollouts and healthier eval-side greedy rollouts.
  - learnings: The dominant failure surface was format collapse, not subtle geometry mismatch. Aggregate `monitor_dumps/prepare_failures` first, then classify the dominant symptom families.

### Older Memory Topics

#### /data/home/xiaoyan/AIteam/data/CoordExp

- Stage-2 Channel-B insertion-order knob and backward-compatible docs or spec updates: insertion_order, tail_append, sorted, top-left sort, clean-prefix, FN append
  - desc: Covers Channel-B ordering semantics, whether missed or FN objects stay at the tail, and how to make ordering configurable without migrating every existing config. Use when the user asks for the exact insertion-order rule or a backward-compatible ordering knob in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Stage-2 eval artifact dumping under rollout_matching.eval_detection: materialize_artifacts, gt_vs_pred_scored.jsonl, raw_rollouts.jsonl, pred_token_trace.jsonl, raw_output_json
  - desc: Covers default-on eval artifact persistence, offline-compatible rollout dumps, and the exact eval artifact contract. Use when changing training-time eval persistence or matching offline infer or eval artifacts in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Stage-2 proxy-dataset audit, CE+CIoU ablation, and duplicate-targeting contract checks: proxy_supervision, object_weight_mode, ConfigLoader, list replacement, duplicate_burst_unlikelihood, adjacent_repulsion
  - desc: Covers mechanism audits for enabled losses, proxy weighting gaps, list-replacement pitfalls, and CE+CIoU-only ablations in Stage-2 config leaves. Use when editing or auditing Stage-2 training configs in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Codex and Serena repo-local setup and launch behavior: .codex, .codex_config/pein, --context codex, danger-full-access, gpt-5.4-mini, xhigh
  - desc: Covers repo-local Codex home migration, Serena MCP `--context codex`, docs-backed full-access launch guidance, portable skill vs shared memory, and the tiered subagent model-allocation policy. Use when working on local agent setup or answering Codex or Serena environment questions in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- COCO 1024 infer preset for the 2B LVIS-proxy merged checkpoint: configs/infer/coco_1024, run_infer.py, output/stage1_2b, val_200_lvis_proxy_merged.yaml
  - desc: Covers creating the infer YAML under the exact user-requested folder, matching it to the HF export checkpoint, and returning the exact runnable `scripts/run_infer.py` command. Use when the user asks for a new infer preset rather than a full eval bundle in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Center-size bbox supervision worktree feature loop: brainstorm first, center_size, xyxy, single-GPU smoke, worktree cleanup
  - desc: Covers brainstorm-first scoping, OpenSpec or worktree setup, narrow internal-loss implementation, one real smoke run, and merge or cleanup. Use when the user wants a lightweight feature implemented end to end with minimal blast radius in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Channel-B cluster-aware duplicate-targeting OpenSpec handoff: openspec, channel-b-cluster-aware-duplicate-targeting, fastforward those *.md, detector background
  - desc: Covers detector-backed spec drafting, preserving empirical background for another implementer, and selectively committing only the OpenSpec change while leaving unrelated local dirt untouched in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.

- Stage-2 partial-annotation and crowded-scene pseudo-supervision analysis: partial annotation, mechanism-level reasoning, unmatched audit, desc-neutral, token_ce, text_gate
  - desc: Covers mechanism-level analysis of how partial COCO annotation interacts with Channel-B matching, triage, pseudo positives, and duplicate suppression. Use when the user wants a structured causal explanation and lightweight SFT-style fixes rather than generic training advice in `cwd=/data/home/xiaoyan/AIteam/data/CoordExp`.
