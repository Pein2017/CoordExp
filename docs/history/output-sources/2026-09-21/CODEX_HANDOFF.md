# 给 Codex 的接手说明：2026-09-21 源码与产物边界重构

请先了解本轮改动，再继续用户另外授权的任务。这份说明不授权新的训练、GPU 实验、commit、push、merge、部署或恢复旧 run。

## 工作区与阅读入口

先重新确认实际 Project、canonical path、HEAD 和 dirty 状态。根目录是 `/data/CoordExp`；研究基座是 `/data/CoordExp/.worktrees/research-probes`，不要创建替代 worktree。此次没有提交代码，不能只看 HEAD 判断是否已取得重构。

在 research-probes 阅读 `docs/OUTPUT_STORAGE_POLICY.md`、短版 `AGENTS.md`、`docs/RESEARCH_PROBE_INFRA_BASE.md`。研究知识和当前方向仍由 `research/CONVENTIONS.md`、`research/index.md` 及相应 unit 的 state/result 管理。实施记录位于 `/data/CoordExp/docs/history/output-sources/2026-09-21/MIGRATION.md`。

## 已完成的变化

两个 outputs 目录里的 Python、Markdown、shell、bytecode 已清理。仍被调用的代码回到了维护包；历史源文件按 hash 归档；两个旧虚拟环境和一个 Label Studio worktree 已移到 `.local/retired-output-runtimes/2026-09-21/`。原始模型、数据、结果与 sealed receipts 没有被本轮重写。

维护代码不再从 outputs 中加载 scorer、untied embedding provider、row-branch producer、endpoint/witness 或 CoDETR profile。当前 owner 包括：
- `probes/training_set_completion/row_scoring.py`：共享 saved-row accounting。
- `src/qwen/input_identity.py`：native input/tensor identity。
- `probes/training_set_completion/artifacts.py`：不同路径及 JSON 编码契约的明确操作。
- `probes/dora_owner_learning/composition.py`：三个 trainer 共享的模型组合验证。
- `src/artifacts/source_provenance.py`：在 outputs 外保存 exact source bytes。
- `src/artifacts/source_archive.py`：显式按 expected SHA 查找历史字节，不执行历史代码。

untied payload 和 row-branch 等有独立科学语义的实现保留独立 owner，不能因为名称相似就替换或合并。新 checkpoint 的自动模型卡存为 `model_card.json`；可视化说明存入 `manifest.json.summary`，Python API 使用 `VisualizationResult.summary`，不再使用 `readme_path`。

## 以后怎么开发

维护实验代码与测试放在 `probes/<direction>/`；跨方向的稳定机制放到恰当的 `src/` owner；`scripts/` 保持薄入口。默认 ordinary imports，不从 outputs、archive、scratch 或另一个 worktree 借执行代码。三方工具通过明确的维护 adapter 接入。

第一次探索允许本地实现。第二个真实消费者出现时，判断复用的是相同机制还是只是相似配方：相同机制提取 owner，cohort、mask、loss denominator、geometry、intervention、metric、decode 和 stop rule 保持显式。不要复制整个 trainer，也不要为了消除相似循环发明通用框架。

一次性查询可以 inline；临时复杂脚本可以放 `.local/scratch/<task>/`，但不能成为维护代码的依赖。需要复现、引用其结果或复用时，先迁入维护源码。检索默认限定 src/probes/scripts/tests；查历史时定向使用 manifest，不把历史快照误当当前实现。

outputs 只放数据、模型 payload、JSON/JSONL receipts、resolved config、日志、指标和渲染产物，不再放 loose .py/.md/.sh、虚拟环境或 vendor checkout。Source snapshot 用 `preserve_source` 放到 `docs/history/run-sources/`，将返回路径写入当前 receipt，并包含新共享依赖。

旧 source hash 不匹配，不代表可以把 expected hash 改成当前 hash。历史 reader 必须明确消费 checksum-pinned 的旧证据；新执行必须绑定当前代码。不可用 archive fallback 让新 run 通过旧 gate，不可恢复旧 outputs 脚本作为兼容捷径。源码可恢复不等于旧上下文可直接重放。

## 验证与并行内容

本轮最终研究测试为 921 passed、5 failed、15 skipped；五个失败的名称和消息与开工前一致，均为旧 Source256 的 source-binding 不匹配。本轮没有新增失败，也没有为绿测改写 sealed hashes。根目录相关测试 51 passed。源码、归档和产物校验的范围及局限见 MIGRATION.md 与 verification-receipt-v2.json。

每次相关工作结束，运行只读边界检查：

```sh
python -B -m src.artifacts.output_layout \
  --root /data/CoordExp/outputs \
  --root /data/CoordExp/.worktrees/research-probes/outputs
```

消费者变化仍需针对性测试和 saved-output parity；不能用目录检查替代科学验证。运行 CPU 测试时隐藏 CUDA、禁用联网下载并限制 CPU 线程。

本轮过程中另外出现 `research/experiments/2026-09-21-spatial-progress-recovery/`，并修改了 research/index.md 和 catalog。它不是维护任务创建或启动的；不要把这些额外改动当成维护 diff 来回滚、归档或统一提交。后续操作先核对该 lane 的真实状态和最新用户授权。

备份与映射：`/data/CoordExp/.local/maintenance/2026-09-21-source-artifact-boundary/` 和 `/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json`。这是同机备份。不要删除它们，也不要整包解压覆盖当前工作树；需要历史文件时按原路径和 expected SHA 做定向恢复/读取。

接手后先概括当前 owner、存放规则以及与当前任务相关的兼容变化，不要自动启动实验或扩大重构范围。

## Human13 的格式兼容补充

`probes/human13/magnitude_finite.py` 新导出使用 materialization v2：`adapter_files` 只绑定实际 adapter 配置和 tensor，模型卡在独立的 `metadata_files` 中，源 adapter 可以没有模型卡。旧 v1 的 README 字节通过 expected SHA 显式从 archive 验证，但真实权重/配置仍必须在当前路径一致。不要改旧 receipt，不要用旧 README 的恢复能力去绕过新 v2 metadata 的校验。相关源文件与测试已更新。
