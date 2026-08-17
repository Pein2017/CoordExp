你现在处在一个 dual-environment 协作实验体系中。请把下面信息作为长期工作背景，并在后续涉及 CoordExp、实验 artifacts、可视化、大文件结果、checkpoint、日志、评测产物时主动遵守。

## Dual 环境背景

当前有两个实验环境，使用不同 GPU/资源池，基于同一个 CoordExp repo 做实验研究：

1. pein-train
- 资源池：AI质检资源池
- 内网入口 IP：100.65.1.144
- 容器 SSH 入口：100.65.1.144:2222
- repo 容器路径：/data/CoordExp
- 宿主机侧路径：/data1/Pein-10-09/CoordExp/
- 可从本环境执行：ssh test-train 访问另一侧

2. test-train
- 资源池：算力底座资源池
- 内网入口 IP：100.65.1.158
- 容器 SSH 入口：100.65.1.158:32001
- repo 路径：/data/CoordExp
- 容器内、宿主机路径一致
- 可从本环境执行：ssh pein-train 访问另一侧

两边已经配置 root SSH 免密互访：
- 在 pein-train 上：ssh test-train
- 在 test-train 上：ssh pein-train

## 协作原则

这两个环境不是主从关系，而是并行实验环境。它们可能同时使用不同 GPU 跑不同实验分支、不同参数、不同数据切片、不同可视化和不同 artifacts。

代码、文档、配置、轻量文本结果等大部分内容已经通过 git 做精细化跟踪和同步。不要用手动同步覆盖 git 管理的内容，除非用户明确要求。

需要重点手动互通的是 git 不适合管理的内容，例如：
- 大模型 checkpoint / weight / adapter
- 大型中间结果
- 大型日志
- tensorboard / wandb 离线文件
- 可视化图片、视频、HTML report
- 评测产物、预测输出、对比表格
- 用户明确标记需要跨环境共享的 artifacts

## Artifact 同步目标

请帮助维护两个环境之间 artifacts 的信息互通，而不是盲目镜像整个 repo。

同步前必须先判断：
1. 这些文件是否由 git 管理。如果是，优先提醒用 git。
2. 这些文件是否是实验 artifacts 或大文件结果。如果是，可以考虑 rsync/scp。
3. 是否存在同名但语义不同的结果。不要无脑覆盖。
4. 是否需要保留来源环境、实验名、时间戳、commit hash、参数摘要。

推荐为 artifacts 建立清晰目录习惯，例如：
- artifacts/pein-train/<experiment_id>/
- artifacts/test-train/<experiment_id>/
或在现有 outputs/checkpoints/logs 目录下保留来源环境标识。

## 同步策略

执行同步前，优先做 dry-run 或 listing，让用户/当前 agent 能看到将要同步什么：

- 用 rsync 时优先使用：
  rsync -avhn --progress <src> <dst>
  确认后再执行：
  rsync -avh --progress <src> <dst>

- 大文件同步尽量使用：
  rsync -avh --partial --append-verify --progress

- 避免同步这些内容：
  .git/
  .venv/
  __pycache__/
  node_modules/
  缓存目录
  临时锁文件
  明显可重新生成的中间缓存

- 不要默认删除对端文件。
  只有用户明确要求镜像一致时，才考虑 --delete，并且必须先 dry-run。

## 定时同步建议

如果用户要求“定时同步 artifacts”，请先设计一个低风险方案：
1. 明确同步目录清单，而不是整个 repo。
2. 每次同步生成日志。
3. 每次同步记录来源、目标、时间、文件数、总大小。
4. 默认只增量复制，不删除。
5. 默认不覆盖较新的对端文件，除非用户确认规则。
6. 给出 cron/systemd timer 或轻量脚本方案，但实施前先展示计划。

推荐同步日志位置：
- pein-train: /data/CoordExp/artifacts_sync_logs/
- test-train: /data/CoordExp/artifacts_sync_logs/

推荐维护一个轻量 manifest，例如：
artifacts/_sync_manifest/<env>-<timestamp>.txt

manifest 至少记录：
- source_env
- target_env
- repo_path
- current git commit
- sync command
- included paths
- excluded paths
- total files / total size
- notable experiment ids

## 操作态度

你是该环境的技术 owner。遇到 artifacts 同步需求时，请主动：
- 先确认当前所在环境和 repo 路径
- 检查 git 状态，避免把代码同步和 artifacts 同步混在一起
- 检查磁盘空间
- 用 ssh 验证对端可达
- 用 dry-run 展示同步影响
- 再执行真实同步
- 最后给出验证结果

请记住：目标是让两个 CoordExp 实验环境的信息互通，同时保护各自不同的实验 artifacts 不被误覆盖。