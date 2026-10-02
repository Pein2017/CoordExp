# Opus 5.5 全工作区审阅：阶段性记录

> 状态：额度中断，未完成总报告。本文保存模型中途记录，不能当作其最终建议或已接受的架构结论。

用户要求自由探索 /data/CoordExp 及其 worktrees、历史 Codex sessions、Lead/Worker 协作、目录管理、skills/AGENTS，并允许自行启动子代理。主审使用 Claude Opus 5.5 / max / native_orchestrator；ultra 不在实时目录中，max 已由用户确认。

## 本次执行与中断

- 开始：2026-10-02T15:09:18.484Z；结束：2026-10-02T15:23:07.865Z。
- 主审启动了三个后台子代理：两个 Opus 5.5、一个 Sonnet 5.5。原生转录中的模型字段已核对；子代理 effort 未核验。
- 主审及三个子代理的末尾均出现 `You've hit your session limit`，没有生成最终报告。
- Claude 提示额度重置：2026-10-03 02:30 +07（客户端时区 Asia/Saigon；对应 2026-10-02 19:30 UTC）。
- 主审记录 60 次工具调用；三个子代理分别记录 50、54、41 次，总计 205 次。计数来自原生转录，不等于模型轮数或完成的任务数。
- HarnessDock 的原生计价统计为约 9.70 USD；这是 provider reported cost，不是订阅实际扣费。

## 已关注的线索，尚未形成最终判断

1. **Lead/Worker 与历史 sessions**：子代理已查询线程分布、父子关系、复用首条消息/fork 的情况，并查看历史短等待密度。已有原始聚合输出，但尚未给出方法局限、归因或完整建议；不能据此直接判定用户过度委派或过度审阅。
2. **架构与兼容性**：代码子代理正在比较 main、infras、research-probes 的结构、分支分歧、推理 API 改名、refinement 对推理内部接口的依赖，以及历史测试与当前 pytest 默认收集范围的关系。它进一步追踪了坐标空间转换，但未完成结论。
3. **目录与工程治理**：子代理记录了 main/src 中仅含旧 bytecode 的目录，检查了 reference/legacy_src、根目录报告、配置分布和测试布局。主审正在比较研究代码与 docs/governance/agent tooling 的提交占比，并审视多个 memory/continuity 存储。当前分支在审阅中有其他任务提交，数量必须绑定当时快照。
4. **skills 与 AGENTS**：主审中途指出一个 Claude 同步技能桶引起的跨 harness 列表问题，尚未提交完整解释；另一个明确线索是 infras 的 CLAUDE.md/GEMINI.md 指向缺失的 AGENTS.md。

主代理额外只读核验：`/data/CoordExp/.worktrees/coordexp-infras/CLAUDE.md` 与 `GEMINI.md` 均为指向 `AGENTS.md` 的悬空链接，目标当前不存在。未修复。

## HarnessDock 体验与续接障碍

- 启动、进度观察和长等待均有返回；中途历史读取曾提示没有已验证的原生会话历史，终止后该接口可以读取记录。
- 原生结束消息表明额度耗尽，但公开失败原因是 `harness_incompatible`。具体持久状态为 `compatibility_surface_drift / native_team_surface`，continuation 为 blocked。
- 观察到的原生工具表包含 Agent 与 SendMessage；缺少策略要求的 TaskCreate/TaskGet/TaskList/TaskUpdate，且 teamTransportLiveValidated 为 false。实际转录记录了异步 Agent 启动，没有 SendMessage 调用。
- 这些现象需要区分：额度耗尽导致此次工作无法继续；兼容性判定另外阻止插件自动 resume。不能通过改状态文件或换模型绕过。
- 重置额度后，应先确认并处理兼容性状态，再续接已保留的同一原生 Opus 会话及子代理结果，不从头重复探索。此次未重启、未回退模型、未修改 runtime/代码/配置/sessions。

## 原始阶段性文本

### Opus 主审

来源：`/data/CoordExp/.claude/projects/-data-CoordExp-codex-harnessdock-dev/a10ba1e1-d19b-445a-b2c8-043687c3bee5.jsonl`

- 2026-10-02T15:12:18.620Z，原始第 146 行：

  > Launching wave 1 shortly; verifying the rollout JSONL record format so the session-analysis briefs are accurate.

- 2026-10-02T15:13:56.159Z，原始第 164 行：

  > Wave 1 launched (session-quant on Sonnet, session-qual and code-arch on Opus). While they run, I'm reviewing skills and AGENTS.md myself.

- 2026-10-02T15:15:19.492Z，原始第 211 行：

  > Still in my own skills/AGENTS pass: found a cross-harness skill-listing regression from today's Claude sync bucket; now checking a broken `CLAUDE.md` in the infras worktree.

- 2026-10-02T15:16:38.542Z，原始第 251 行：

  > Measuring commit mix (code vs docs/governance) and how far local branches are ahead of origin.

- 2026-10-02T15:19:08.509Z，原始第 331 行：

  > Checking the multiple memory/continuity stores (Codex memories, project-memory, Serena, Claude) while teammates run.

- 2026-10-02T15:20:59.200Z，原始第 377 行：

  > Teammates still running. Meanwhile quantifying the agent-tooling repos' size and churn (HarnessDock, wake-me-up, usage-ledger, serena-light) as context for the "meta-work" question.

### Opus：会话使用定性审阅

来源：`/data/CoordExp/.claude/projects/-data-CoordExp-codex-harnessdock-dev/a10ba1e1-d19b-445a-b2c8-043687c3bee5/subagents/agent-a42afe7694409fb91.jsonl`

- 2026-10-02T15:13:29.701Z，原始第 15 行：

  > I'll start by reading the policy files and surveying the thread index.

- 2026-10-02T15:13:57.306Z，原始第 53 行：

  > Now let me survey the thread index.

### Opus：代码架构审阅

来源：`/data/CoordExp/.claude/projects/-data-CoordExp-codex-harnessdock-dev/a10ba1e1-d19b-445a-b2c8-043687c3bee5/subagents/agent-a696bd91a60339f3c.jsonl`

- 2026-10-02T15:13:53.537Z，原始第 15 行：

  > I'll start with the worktree topology and branch divergence, using read-only git inspection.

- 2026-10-02T15:14:05.234Z，原始第 27 行：

  > Main has moved since the lead's snapshot (now 9a84fd813, not 7a8cb9456). Next I'll check the worktree states and what each `src/` contains.

- 2026-10-02T15:14:20.485Z，原始第 36 行：

  > Main's `src/` lists both legacy and current packages. I'll separate tracked packages from leftover `__pycache__`-only directories.

- 2026-10-02T15:14:40.708Z，原始第 48 行：

  > Twelve stale bytecode-only directories remain in main's `src/`. Next I'll check whether the architecture documented on main matches the code on each branch.

- 2026-10-02T15:14:54.470Z，原始第 63 行：

  > The infras branch is a slimmed-down tree with its own config layout. I'll read its routing docs and compare them with main's claims.

- 2026-10-02T15:15:01.291Z，原始第 69 行：

  > `CLAUDE.md` on infras appears to be a dangling link. I'll verify it and check the agent-instruction files on each branch.

- 2026-10-02T15:15:11.241Z，原始第 75 行：

  > On infras, `CLAUDE.md` and `GEMINI.md` are tracked symlinks to an `AGENTS.md` that the branch no longer contains. Next I'll read main's branch policy.

- 2026-10-02T15:15:28.656Z，原始第 87 行：

  > 154 of 155 infras commits have no patch-equivalent on main, so promotion is not cherry-pick. I'll look at how main's history absorbs infras work.

- 2026-10-02T15:15:35.958Z，原始第 93 行：

  > Main's history mixes research code with operator tooling (Serena, RTK, browser ports, async commit APIs). I'll quantify what main's 252 commits since the fork actually touch.

- 2026-10-02T15:15:44.909Z，原始第 99 行：

  > Main tracks `.codex/`, `codex-wake-me-up/`, and `codex-usage-ledger/` in the same history as the research code. I'll check main's current tracked top-level composition.

- 2026-10-02T15:15:52.874Z，原始第 105 行：

  > Main tracks 1038 files under `reference/`, oddly named `"research…` paths, and stray root reports. I'll identify these.

- 2026-10-02T15:16:00.736Z，原始第 111 行：

  > Main has no `pyproject.toml` or requirements file, while infras has one; main also vendors 395 legacy `.py` files under `reference/legacy_src`. I'll look at packaging and import style.

- 2026-10-02T15:16:07.747Z，原始第 117 行：

  > Main's `pytest.ini` restricts collection to a subset of `tests/`. I'll measure how many of main's 438 test files fall outside the default test paths.

- 2026-10-02T15:16:16.789Z，原始第 123 行：

  > Roughly 380 of main's test files sit outside `testpaths`. I'll run a read-only AST scan (stdin script, bytecode writing disabled) to see whether those tests import modules that still exist.

- 2026-10-02T15:16:50.251Z，原始第 135 行：

  > Confirmed: these tests import the removed `src.infer.*`, `src.trainers.*`, and `src.data_collators.*`, so they cannot even be collected. Next I'll map the current-route module sizes on each branch.

- 2026-10-02T15:18:16.648Z，原始第 181 行：

  > Research-probes is self-contained with no stated sync path to or from infras. I'll look at its layout and how probe code relates to `src/`.

- 2026-10-02T15:18:50.144Z，原始第 193 行：

  > Infras' import graph is clean (no module cycles), but there is one `eval <-> training` package cycle, and `training/session.py` has fan-out 34. I'll run the same analysis on main and research-probes.

- 2026-10-02T15:19:00.312Z，原始第 199 行：

  > On main, Label Studio refinement tooling imports inference internals 13 times, and `eval -> training` exists on all branches. I'll identify those exact edges.

- 2026-10-02T15:19:20.910Z，原始第 211 行：

  > Main's 32K-LOC refinement tooling depends on 5 inference symbols that infras has removed or renamed. I'll check for renames and examine the infras runtime surface.

- 2026-10-02T15:19:28.959Z，原始第 217 行：

  > Infras renamed the inference API (`assemble_runtime` → `assemble_frontend`, `HFGenerateBackend` → `HFBackendSession`). Next I'll measure config sprawl and inheritance per branch.

- 2026-10-02T15:21:41.437Z，原始第 283 行：

  > Even the "canonical" `build_coco_views.py` imports the legacy encoder. I'll check what the specs define for norm1000 and how the inference/eval/vis side decodes predicted bins.

- 2026-10-02T15:21:50.285Z，原始第 289 行：

  > Vis says prediction boxes are "already pixels". I'll find where inference converts predicted bins to pixels and confirm it shares the same owner function.

### Sonnet：会话使用量化盘点

来源：`/data/CoordExp/.claude/projects/-data-CoordExp-codex-harnessdock-dev/a10ba1e1-d19b-445a-b2c8-043687c3bee5/subagents/agent-aaa1b5033cb6d453b.jsonl`

- 2026-10-02T15:13:02.035Z，原始第 14 行：

  > I'll start by checking the SQLite schema and basic distributions through a read-only immutable connection.

- 2026-10-02T15:13:07.469Z，原始第 26 行：

  > The schema is confirmed. Next I'll read the Model routing policy and run the first-pass DB aggregates.

## 续接资料

- 运行与返回记录：`/data/CoordExp/codex-harnessdock-dev/outputs/opus-coordexp-review-20261002/harnessdock-return.json`
- 阶段性文本与工具调用定位：`/data/CoordExp/codex-harnessdock-dev/outputs/opus-coordexp-review-20261002/native-partial-responses.json`
- 续接清单：`/data/CoordExp/codex-harnessdock-dev/outputs/opus-coordexp-review-20261002/resume-packet.json`
- 原生转录保留在原路径；没有改写。临时量化扫描数据与脚本已校验复制到任务所有的 outputs / .local/scratch，防止 /tmp 清理丢失。
