---
type: investigation
role: handoff
authority: non_normative_research
status: awaiting_implementation_entry
updated: 2026-09-09
summary: 将已获用户认可的研究开发架构交接到 research-probes；等待实验停止后实施。
---

# Research probe 开发流程改造：实施交接

## 目标与第一步

目标：实验代码能够独立演进和退休，研究知识持续累积，公共能力有明确归属；减少新实验继承的历史脚本、跨 worktree 依赖和默认搜索噪声。

接手 cwd：`/data/CoordExp/.worktrees/research-probes`。

**先核对 self-rollout 是否已结束，以及用户是否已确认所有 probe 实验停止、代码可以修改。交接时尚未收到这一确认。** 用户最近明确说：`self-rollout` 当前还在 active using，需要等它结束。不得因为收到本文、创建新任务或观察到某个进程退出，就推断实施条件已满足。等待期间可继续只读检查和准备 OpenSpec；不停止实验、不迁移其代码、不清理依赖目录。

实施条件满足后，按下文顺序自主推进已接受的范围。不要重新做一轮泛化架构调查，也不要要求用户逐个重复批准范围内的常规步骤。若发现新的语义、成本或破坏性范围问题，再交给用户决定。

本文是交接材料，不是科学结果、运行时授权或新的规范 owner。用户在本次对话的明确决定优先于下列尚待更新的旧规则；进入实施时把决定落实到 owning OpenSpec 和现有规范。

## 已接受的决定；不要恢复旧方案

1. `research-probes` 是唯一长期研究主线，承载研究知识、公共代码及新方向的 fork 起点。
2. **退役 `research-probe-infras`。** 用户询问能否直接在 research-probes 修改 infra，并明确接受该简化。公共模块与验收边界保留，长期专属 infra worktree 不再保留。
3. 日常小型 infra 修改直接在 research-probes 完成；较大改造或并行冲突时使用临时改造 worktree，验收合回后退休。不要让这个临时位置再次成为永久分支。
4. 实验专属代码放在 `probes/<direction_name>/`，使用普通 Python 包和显式导入。一个持续方向可包含多个研究单元和运行，不按每个 run 创建一个包或 worktree。
5. 实验依赖现有 `src` 公共 owner；禁止公共模块反向依赖实验，禁止通过另一个临时 worktree 的路径导入代码。数据、checkpoint 和执行产物可使用明确且身份可核对的共享路径。
6. 文档知识回流与公共代码提升分别处理。实验分支不因贡献一个 helper 而整支合并。研究主线的 `src` 改动不自动等于对 CoordExp-Swift 的生产提升。
7. 原始实验代码可保存版本后退出当前工作树；只迁移未来仍有用途的代码，不把全部历史脚本重新包装一遍。
8. `permanent-owner-bridge-cache-validation` 由用户独立退休，完全排除在本次改造、检查及清理范围外。保留 CoordExp-swift 和固定 research-probes；不改根仓库 main 的无关工作、共享 agent/runtime 配置、凭据和远端 refs。

用户实施意图原文：

> 开始实施的条件我会停下所有的.worktrees下的 probe 的实验,完全将代码都`解锁`,可以自由任意地做修改.
> 比如,可以随时cleanup一些 worktree,只要确保文档知识已经合并.

后续用户又明确要求等待 self-rollout 结束。以上许可适用于这项研究工作流改造；不授权丢弃最新成果或扩大到无关基础设施。

## 最小阅读路径与待修正规则

按以下顺序阅读；不要扫描全部历史材料：

1. [用户级 agent contract](/data/CoordExp/.codex/AGENTS.md) 与接手 cwd 的有效本地指令。
2. [分支和 worktree policy](/data/CoordExp/.worktrees/research-probes/docs/BRANCH_AND_WORKTREE_POLICY.md)，尤其 Research-probe routing。
3. [现有 infra 能力边界](/data/CoordExp/.worktrees/research-probes/docs/RESEARCH_PROBE_INFRA_BASE.md)。
4. [研究文档入口](/data/CoordExp/.worktrees/research-probes/research/index.md) 和 [research graph contract](/data/CoordExp/.codex/skills/research-flow/references/research-graph-contract.md)。
5. 仅按当前迁移对象阅读下文的具体代码和研究记录。

旧 policy 的三处内容已被本次用户决定或已验证问题推翻，实施中须在 owner 修正：

- research-probe-infras 永久保留、永不退休及固定集成链路：改为上文单一 research main 方案。该 worktree 当前有 Git lock；满足保存和退休条件后，按这一明确生命周期决定解除该对象的锁。不要解除 research-probes 的保护锁。
- 按整个 research unit 目录 checkout 的 records-only 回流：改为按用途选择的显式文件清单，因为已有目录混入 Python。不能仅凭目录或扩展名判定其是否应回流。
- 对共享文档一概“append, do not overwrite”：原始证据保留；compass 作为当前综合应重写过时部分，不继续堆叠流水账。

此前一轮 Astra `xhigh` 架构把关已完成，支持普通方向包、先保存再选择性迁移；当时保留永久 infra lane 的建议已被随后用户接受的单主线方案取代。不要为相同设计再默认追加一轮评审。

## 接手时重查的工作树快照

以下 HEAD 于 2026-09-09 交接时核对，只用于定位；不是完整工作状态或可直接删除的清单。

| worktree 名称（均位于 `/data/CoordExp/.worktrees/`） | 分支 | HEAD 短值 | 处置方向 |
| --- | --- | --- | --- |
| research-probes | research-probes | 73b8b3cc2 | 固定保留；实施总入口 |
| research-probe-infras | research-probe-infras | ba4fb18d4 | 核对并处理内容后退休 |
| c-anchored-owner-mechanism-audit | codex/c-anchored-owner-mechanism-audit | 2912c769d | 保存、回流后按条件退休 |
| coco-gt-correction-portfolio | probe/coco-gt-correction-portfolio | a4bac643c | 同上 |
| dora-prox-linear-n2 | probe/dora-prox-linear-n2 | e4f932faa | self-rollout 的依赖，不能先行移除 |
| human13-output-qp-identity-generalization | probe/human13-output-qp-identity-generalization | a2c049453 | 保存、回流后按条件退休 |
| image2299-logit-lens | probe/image2299-logit-lens | 269477a3a | 同上 |
| n256-shared-output-qp-norm-scaling | codex/n256-shared-output-qp-norm-scaling | 756947c68 | 同上；保留历史源码身份 |
| self-rollout-behavior | probe/self-rollout-behavior | f8c5f1f51 | 用户报告仍 active；先等待结束 |

此前只读检查还发现：

- DORA 包含 Human13 当前 tip 的历史；C、COCO、logit-lens、self-rollout 包含当前 research main。不要逐分支重复回流相同来源的结果。
- infra tip 是 main 的祖先，落后 2 个提交；这不意味着工作目录无独有内容。
- `git status --porcelain --untracked-files=all` 曾显示 self-rollout 有 29 个未跟踪 research 文件；DORA 有 9 个未跟踪 research 文件、4 个 scripts、2 个 tests、5 个 OpenSpec 文件，另有 7 个修改的 research 文件。全部属于需重查的易变事实。**tag 当前 HEAD 会遗漏这些内容。**
- main 有预先存在的共享 `.codex`/AGENTS 删除状态；infra 另有 repo_lifecycle 相关删除。不要恢复、覆盖或顺手 broad-stage 这些改动。
- DORA 的 `outputs/` 是 worktree 内真实目录。ignored 不等于可丢弃；删除前要保留被引用的必要产物并验证记录里的证据定位仍有效。

## 已验证的设计依据与具体迁移风险

无需重做全量计数：此前以 main 和七个临时分支的 HEAD 检查 scripts/research，得到 1,460 份 Python 文件、241 个不同 Git blob，170 个 blob 在八棵树中相同。main 本身有 172 个此目录下脚本。这些是继承副本计数，不是 1,460 份独立实现，也不是删除依据。

- [self-rollout train.py](/data/CoordExp/.worktrees/self-rollout-behavior/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-08-owner-outcome-autonomous/learning/train.py) 把 DORA 绝对路径加入 sys.path，并导入三个旧脚本及配置。保存其源码/config 依赖闭包，或对未来要继续使用的 producer 做受验收的迁移，然后才能退休 DORA。仅移动 self-rollout 目录不能解决问题。
- [N256 train-scaling runner](/data/CoordExp/.worktrees/n256-shared-output-qp-norm-scaling/scripts/research/run_n256_shared_output_qp_train_scaling.py) 的 `_load_launch` 绑定 runner/core/full/docs 的哈希及 `__file__` 哈希。历史源码、配置、收据保留原样；重构后的实现使用新身份，不能重写旧收据来冒充兼容。源码可恢复与执行重放已验证要分别报告。
- [Human13 runner](/data/CoordExp/.worktrees/human13-output-qp-identity-generalization/scripts/research/run_human13_output_qp_same_panel.py) 与 [N256 runner](/data/CoordExp/.worktrees/n256-shared-output-qp-norm-scaling/scripts/research/run_n256_shared_output_qp_norm_scaling.py) 重复实现 immutable_json；现有 src.artifacts 可作为复用候选，但必须核对 JSON 字节、哈希、occupied-path 和错误语义。
- 两者还导入 [旧 coverage 脚本](/data/CoordExp/.worktrees/research-probes/scripts/research/compare_clean_rollout_owner_coverage.py) 的 `_global_matches`。可考虑抽取小公共模块；保持调用方的阈值、去重顺序、分母及标注语义。不能因名字相近就建设一个统一 QP/training 框架。
- main 的 compass 曾有 1,209 行，当前路由仍停在 C-audit；[后续 C 结果](/data/CoordExp/.worktrees/c-anchored-owner-mechanism-audit/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical/results.md) 和 [DORA frontier synthesis](/data/CoordExp/.worktrees/dora-prox-linear-n2/research/investigations/qwen3-vl-dense-enumeration/2026-09-07-research-frontier-synthesis.md) 在检查时尚未出现在 main 对应位置。它们是回流入口，不是未经审阅即可提升的总体结论。
- [gitignore](/data/CoordExp/.worktrees/research-probes/.gitignore) 使用 allowlist，须显式允许 probes/；[pytest.ini](/data/CoordExp/.worktrees/research-probes/pytest.ini) 不自动发现包内测试。默认测试通过不能代替样板自身验收。

## 实施顺序与退出条件

建议先创建一个 owning OpenSpec change，例如 `restructure-research-probe-development`（尚未创建；先确认没有并行 owner）。它承接已接受设计，不重开泛化架构讨论。规范直接更新既有 owner，避免创建第二套知识或生命周期系统。

| 阶段 | 工作 | 验收/退出条件 |
| --- | --- | --- |
| 0. 进入实施 | 获取用户已停止全部 probe 的确认，核对 live holders、实际工作状态及对象边界 | 无冲突实验活动；self-rollout 等待条件已满足 |
| 1. 保存 | 按范围保存 committed/modified/untracked 的有效内容、源码/config 依赖及必要证据；使用显式路径和可恢复版本 | 抽查源码哈希恢复；被保留结果的证据在目录移除后仍可访问；无遗漏的相关工作状态 |
| 2. 知识回流 | Git 差异和 blob 身份过滤继承副本；显式选取记录；先 result，再 router，最后当前综合和必要 decision | 接受的新结果可从主线找到；分歧与证据边界保留；不夹带 producer 代码 |
| 3. 新路径样板 | 更新 policy/infra guide；加入 probes/ 约定；迁移一个未来还会使用的真实 producer 或完整离线入口 | 独立 checkout 中明确的包测试、import 和 CPU/preflight/保留产物读取通过；无 sibling worktree 代码依赖 |
| 4. 收缩代码 | 按保留调用者检查旧脚本、配置、测试和说明；公共能力分别提交验收；移除已保存的历史 producer | 活跃代码无悬空依赖；每项留下的实验代码有方向归属；历史引用有精确恢复定位 |
| 5. 退休切换 | 逐个复核目标，退休满足条件的临时方向与 research-probe-infras；固定 research-probes 不移除 | 知识已回流、源码可恢复、证据已保留、无未处理内容或 live holders；从新 main 可正常开一个方向 |

阶段 5 可在独立对象满足条件时提前执行，不必等全部代码迁移。但 DORA 需等 self-rollout 依赖保存或迁移验收，infra 需等其有效内容处理及永久保留规则更新。不要为了退休而要求全部历史实验迁移到新包。

先保留再修改；不要使用 reset/clean/broad-stage 掩盖 dirty。不要把 ignored 输出当缓存批量删除。历史重建存在原路径或环境约束时，记录约束；不宣称未执行过的完整重放。Git archive refs 只保存源码，不代替产物保留。

## 目标目录、知识更新与验收尺度

```text
src/<existing owner>/                 公共能力及其稳定测试
probes/<direction_name>/              普通包；一个方向可承载多轮实验
  __init__.py
  <entry>.py                          python -m probes.<direction>.<entry>
  configs/、tests/、README.md          按实际需要增加，README 链接研究 owner
research/investigations/<topic>/
  compass.md                         当前综合，保持精炼
  experiments/index.md               路由
  experiments/<unit-id>/unit.md       问题、范围、对照、成本与 stop rule
  experiments/<unit-id>/results.md    事实和有边界的解释
outputs/research/...                  明确定位的执行证据；退出 worktree 前核对实际根
```

小 probe 可以只有一个模块和简短记录。局部 helper 先留在方向包内；不要创建 probes/common 杂物箱、全局 ProbeContext、插件 registry、事件总线或通用调度器。现有 artifact/journal/admission/inference 按需直接组合，差分训练不强塞进 deterministic inference 接口。

知识按研究问题综合，不能按分支数量计贡献或按最新文件覆盖结论。原始事实保留详细来源，其余位置用摘要和链接；compass 可替换过时综合。decision 只在路线改变时更新；OpenSpec 管实现与兼容契约。观察、科学结论和归档状态分开。不要写入 Codex-managed memory；项目 continuation memory 仅按其有效授权和需要处理。

迁移代码先刻画最近的 caller-visible 行为，保持科学输出、分母、artifact 身份及关键失败语义；不要删除有效代码来制造 RED。文档做链接/结构检查，包做明确测试与真实入口检查。文档归集、源码保存及离线样板不需要 GPU 重跑；若选中的模型运行改动确实需要 GPU smoke，先明确最小范围和成本，不能把整理变成新研究实验。

验收以知识可检索、证据可访问、源码可恢复、实验包可独立开展和默认工作集收缩为准。无需脚本数量归零、统一所有历史科学算法或建设持久迁移数据库。

## 协作与已完成/未完成

可用两个职责明确的实施包并行处理知识归集与代码样板；共享 index/compass 和 Git 集成保持单一 owner。lead 负责保存、边界、依赖、验收及逐个退休。原 Astra xhigh 把关已结束，无需等待旧 subagent 或恢复本次任务中的其他运行时工作。

已完成：两轮只读 scout、Astra xhigh 一轮整体设计把关、用户接受目录/流程设计及随后退役永久 infra lane 的简化。

未完成：OpenSpec、源码/产物保存、记录回流、包迁移、代码清理、worktree 退休和任何实现验收。本次仅创建这份交接文件；没有提交、合并、停止实验或删除目录。该文件为未提交的本地交接材料；接手后纳入自己的范围内提交，不要在清理中遗漏。
