# AGENTS / Skills 定制化优化记录

日期：2026-09-05。范围：当前 `/data/CoordExp` 实际继承的规则和用户维护的 Skills。
结论：完成本轮有限源文件优化；未提交 Git，未启动研究实验或正式记忆写入。

## 盘点与归属

- 当前继承 `/data/CoordExp/.codex/AGENTS.md` 和 `/data/CoordExp/AGENTS.md`；未发现本层覆盖文件或 Skills 子目录中的额外 AGENTS。没有同步其他 worktree。
- 用户 Skills 共 43 个磁盘入口、33 个不同名称：`.codex/skills` 30 个，`.agents/skills` 13 个。10 个 OpenSpec 名称在两处重复且内容不完全相同，不能当作软链接或无差别副本删除。
- 两个显式调用入口是 `grill-me` 和 `workflow-self-distillation`；未改变任何隐式调用 policy。项目配置没有 Skills 禁用条目。所查入口没有软链接。
- 系统和插件缓存仅作为边界/作者规范参考，不修改。未全读所有 Skills 正文；先读描述、长度和引用，再审读相关自定义入口。
- 一个 Luna 只读 scout 提供盘点候选，主代理核对源文件、决定修改和验收；没有附加 reviewer 或多层代理。
- 两份 AGENTS、OpenSpec 变体及 native guide 已有未提交工作。检查时间、差异和近期任务状态后，在现有 AGENTS 上做增量修改；native guide 只修默认调用名，保留正文中的模型路由偏好。OpenSpec 副本不接管。
- 历史审计仅用于选择检查项；本轮计数、路径、哈希与验证均重新采集，未复用旧的通过率。

## 修改清单与字符数

计数采用 UTF-8 解码后的字符数，包含空白；不是字节数或 Token 数。基线为本轮修改前的实际源文件，而非 Git HEAD。

| 源文件 | 修改前 | 修改后 | 内容 |
|---|---:|---:|---|
| [用户级 AGENTS](/data/CoordExp/.codex/AGENTS.md) | 8,315 | 8,642 | 合并委派重复规则；按描述选 skill、按需读引用；记忆先搜索后局部回读，必要时扩展 |
| [项目级 AGENTS](/data/CoordExp/AGENTS.md) | 3,067 | 2,929 | 删除全局已覆盖的权限与验证重复表述，保留项目级科学和测试约束 |
| [project-memory](/data/CoordExp/.codex/skills/project-memory/SKILL.md) | 2,467 | 3,211 | 限定已配置仓库；区分 Codex 托管记忆；写入前检查授权、网关和未闭合事务；冲突只暂停受影响写入 |
| [research-flow](/data/CoordExp/.codex/skills/research-flow/SKILL.md) | 11,321 | 10,643 | 将研究 OpenSpec 收尾流程移入条件触发的参考文件 |
| [codex-usage-ledger](/data/CoordExp/.codex/skills/codex-usage-ledger/SKILL.md) | 5,892 | 4,844 | 收窄描述，避免普通历史回顾误触发；命令示例改为按需读取 |
| [reclaim-code-entropy](/data/CoordExp/.codex/skills/reclaim-code-entropy/SKILL.md) | 9,514 | 9,038 | 精简描述，排除纯文档整理、普通功能开发和纯性能审计；正文能力不删减 |
| [native guide 元数据](/data/CoordExp/.codex/skills/native-subagents-guide/agents/openai.yaml) | 363 | 289 | 将失效的 `$native-depth2-contracting-pilot` 修正为实际名称 `$native-subagents-guidance`；不复制正文中的路由表 |
| **以上入口与元数据合计** | **40,939** | **39,596** | **减少 1,343 字符** |

新增按需参考：

- [研究收尾流程](/data/CoordExp/.codex/skills/research-flow/references/openspec-closeout.md)：990 字符；条款正文与原文逐字一致。
- [Ledger 调用示例](/data/CoordExp/.codex/skills/codex-usage-ledger/references/invocation-examples.md)：1,427 字符。

计入新增参考后，这组指导源文件总量为 **42,013 字符，比原来增加 1,074**。增加来自明确的记忆安全边界、参考入口及示例保护。备份、验证脚本和本报告不计入指导文本。没有实际用量对比，不能宣称 Token、套餐额度或运行效率的节省比例。

## 行为与不变项

- 检索由默认读当前状态入口改为先定位相关内容；一至两篇是初始回读量，不是禁止补证的硬上限。
- 当前根目录没有 `memories/config.yaml`，因此 project-memory 不自动初始化仓库记忆，也不接管 `.codex/memories`。托管记忆仍依注入指令经 add-on-note 网关写入；本轮未启动正式记忆写事务，因此没有尝试接管或恢复事务。
- Ledger 示例使用项目要求的 `conda run -n ms python`，不再覆盖已有 `CODEX_HOME`；缺失必填变量时在执行前报错。实际 ledger 参数不变，默认仍只统计子代理，只有用户要求才计入 root。
- Ledger 的六项交付、strict 与 proxy 区分、研究收尾的四项条件及失败停止规则均保留。Serena/Pi 工具偏好、授权、验收、等待时长、代理深度与模型路由正文均不变。
- 没有改模型配置、业务代码、系统/插件缓存或正式记忆内容，没有删除材料、stage、commit 或 push。

## 备份与验证

备份目录：[guidance-backup-20260905-NtR5VH](/data/CoordExp/.codex/guidance-backup-20260905-NtR5VH)。使用 `cp -a --parents` 保存七份修改前源文件及目录层级；保留权限等元数据。恢复时只比较和选择需要的文件，不整目录覆盖当前工作。

- [原始盘点](/data/CoordExp/.codex/guidance-backup-20260905-NtR5VH/inventory.json)：入口、真实路径、描述、长度和 policy。
- [基线哈希](/data/CoordExp/.codex/guidance-backup-20260905-NtR5VH/baseline.json)：83 个源文件的哈希、字符数、权限和路径身份。
- [验证结果](/data/CoordExp/.codex/guidance-backup-20260905-NtR5VH/validation.json)：76 个未改源文件保持原哈希；修改文件备份身份、权限及路径检查通过。
- 43 个入口的 YAML 均可解析；21 个非 OpenSpec 入口通过 `quick_validate.py`。22 个未改的 OpenSpec 入口有既存的 `compatibility` 字段校验器不兼容，未删字段强行变绿。
- 17 个被检查的入口/新增参考中的本地 Markdown 文件目标均存在。研究收尾逐字迁移、ledger 命令参数等价、两个 shell 示例语法与缺失 CODEX_HOME 拒绝执行检查通过。
- native guide 的默认调用名检查在修改前失败、修改后通过，其他 UI 字段不变。Ledger 真实 wrapper 的 `--help` 成功；未进行未经请求的用量扫描。
- 已回读源文件并审查相对备份的精确差异；`git diff --check` 通过。这些是结构、保全和入口检查，不是新任务中的自动选 skill 行为评测。

复验命令：

```bash
conda run -n ms python /data/CoordExp/.codex/guidance-backup-20260905-NtR5VH/verify.py
```

## 保留项 / 未完成项

1. 10 组 OpenSpec 双源、长 onboarding/explore 教程暂不合并；需要确认两套生成/维护归属后再做，而非覆盖已有未提交工作。
2. 本地 validator 与 OpenSpec `compatibility` 元数据的既存不兼容未修复；不属于本轮用户指导源文件优化。
3. 系统/插件自带的广泛触发描述与强制流程未修改，也未通过禁用插件改变默认能力。
4. 未测优化前后实际 Token、费用、延迟或模型任务质量；源文件变化也不会追溯删除已注入当前上下文的旧文本。
