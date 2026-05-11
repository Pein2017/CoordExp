thread_id: 019da029-8bdf-7762-922e-ffe495d14192
updated_at: 2026-04-18T10:40:45+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/18/rollout-2026-04-18T10-36-10-019da029-8bdf-7762-922e-ffe495d14192.jsonl
cwd: /data/CoordExp
git_branch: codex/raw-text-continuity-probe

# 调研了 Qwen3-VL 在 detection/grounding JSON 训练中更偏 compact 还是 pretty-inline，并用本地 ms-swift 源码与官方资料交叉核实。

Rollout context: 用户要求“帮我调研一下 Qwen3-VL 模型在 detection、json 格式训练时，用的是 compact 的格式还是 pretty-inline 的格式？请联网搜索和浏览本地的 ms-swift library 来一探究竟。” 主要目标是判断 Qwen3-VL / ms-swift 的 detection(grounding) 训练样式，以及是否应采用 compact JSON 还是更偏 pretty-inline 的表示。

## Task 1: 调研 Qwen3-VL grounding/detection JSON 训练格式

Outcome: success

Preference signals:

- 用户明确要求“联网搜索和浏览本地的 `ms-swift` library 来一探究竟” -> 说明这类问题应默认采用“外部资料 + 本地源码”双线核实，而不是只看单一文档或凭经验判断。
- 用户把问题表述为“compact 还是 pretty-inline” -> 说明后续遇到类似格式/序列化问题时，应该直接对比真实生成文本的空白与序列化方式，而不是只看抽象字段名。

Key steps:

- 先读了仓库里的 upstream/导航类文档，定位到 `docs/standards/UPSTREAM.md`，再把查找范围收敛到 ms-swift 的 grounding / template 代码。
- 在 `/data/ms-swift` 中查到 `docs/source_en/Customization/Custom-dataset.md`、`docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md` 和 `docs/source_en/Instruction/Command-line-parameters.md` 都明确提到 `QWENVL_BBOX_FORMAT='new'` 兼容 Qwen3-VL 官方 cookbook，并给出 JSON 数组对象样例。
- 在 `swift/llm/template/template/qwen.py` 中确认 `QWENVL_BBOX_FORMAT` 默认是 `'legacy'`；`new` 模式下 `replace_ref()` 直接返回纯文本 `ref`，`replace_bbox()` 直接返回 `str(bbox)`，没有整体 `json.dumps(..., separators=(',', ':'))` 这种 compact 重写。
- 在 `swift/llm/template/base.py` 中确认模板层只是做占位符替换，不会把整段 assistant 内容统一重写成 compact JSON。
- 联网打开了 Qwen3-VL 官方 cookbook 页面，并将其与 ms-swift 文档中的样例对照，确认“new”格式更接近官方 cookbook 的 pretty-inline 风格。

Failures and how to do differently:

- 一开始从仓库里扫 `rg` 时输出过大，说明针对这类问题更适合先从文档目录和上游包路径收敛，再进源码细看，而不是依赖全仓库粗搜结果。
- `git -C /data/ms-swift ...` 遇到 safe.directory 报错，后续如果需要 git 元信息，应先用 `git -c safe.directory=/data/ms-swift ...` 或先把仓库标记为安全。
- 试图用环境内 `conda run -n ms python -c "import swift; ..."` 时出现了较长等待，后续定位包路径更直接的办法是结合 `find /data -maxdepth ...` 和文档中的源码位置提示。

Reusable knowledge:

- ms-swift 对 Qwen2.5-VL / Qwen3-VL 的 grounding 格式有两条明确路径：`legacy`（`<|object_ref_start|>...<|box_start|>...<|box_end|>`）和 `new`（兼容 Qwen3-VL 官方 cookbook 的 JSON 风格）。
- `QWENVL_BBOX_FORMAT` 默认值在 ms-swift 源码里是 `'legacy'`。
- `new` 模式下，bbox 的替换实现是 `str(bbox)`，所以输出坐标列表会带 Python 列表空格；这意味着它不会主动压成无空格的 compact JSON。
- ms-swift 文档给出的 `new` 样例是多行 JSON 数组对象，形式上更接近 pretty-inline，而不是 compact。
- 用户这次的问题本质上是在问“训练文本的最终表面格式”，所以判断时应关注模板替换后的实际字符串，而不是只看数据集字段结构。

References:

- [1] `/data/ms-swift/docs/source_en/Customization/Custom-dataset.md`：`QWENVL_BBOX_FORMAT='new'` 兼容 Qwen3-VL 官方 cookbook，示例为
  `[{"bbox_2d": <bbox>, "label": "<ref-object>"}, ...]`
- [2] `/data/ms-swift/docs/source_en/BestPractices/Qwen3-VL-Best-Practice.md`：重复给出同样的 grounding JSON 样例，并说明 Qwen3-VL bbox 输出使用 normalized 1000 relative coordinates。
- [3] `/data/ms-swift/docs/source_en/Instruction/Command-line-parameters.md`：`QWENVL_BBOX_FORMAT` 说明中明确写了 `'legacy'` 与 `'new'`，其中 `'new'` 参考 Qwen3-VL cookbook，默认是 `'legacy'`。
- [4] `/data/ms-swift/swift/llm/template/template/qwen.py`：`self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')`；`replace_ref()` / `replace_bbox()` 在 `legacy`/`new` 间切换。
- [5] `/data/ms-swift/swift/llm/template/template/qwen.py`：`replace_bbox()` 的 `new` 分支是 `return [str(bbox)]`，不是 compact JSON 序列化。
- [6] `/data/ms-swift/swift/llm/template/base.py`：`_pre_tokenize()` 只替换 `<ref-object>` / `<bbox>` 占位符，不会把整段 assistant 消息重排成 compact JSON。
- [7] 官方 cookbook 链接：`https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb`
- [8] 联网上下文还打开了 `https://swift.readthedocs.io/en/v3.9/BestPractices/Qwen3-VL-Best-Practice.html`，与本地文档结论一致。
