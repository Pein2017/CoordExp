thread_id: 019d9ee3-e7ed-76a2-81b4-d039583746ea
updated_at: 2026-04-18T04:52:49+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/04/18/rollout-2026-04-18T04-40-29-019d9ee3-e7ed-76a2-81b4-d039583746ea.jsonl
cwd: /data/CoordExp
git_branch: main

# Qwen3-VL tokenizer analysis for detection grounding coordinate text

Rollout context: The user asked for a targeted analysis of `model_cache/models/Qwen/Qwen3-VL-2B-Instruct` tokenizer behavior on detection-grounding coordinate text, using real samples from `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.norm.jsonl`. The requested deliverable was an evidence-based breakdown of how `bbox_2d: [10,100,231,1]` and similar strings tokenize, including token ids/strings, punctuation handling, numeric splitting rules, optional vocabulary coverage stats, and possible implications for grounding accuracy/length.

## Task 1: Qwen3-VL coordinate-text tokenization analysis

Outcome: success

Preference signals:

- The user explicitly required: “使用数据集 ... 作为输入样本来源” and “从数据集中抽取若干真实样本，构造 tokenizer 输入并进行实际编码验证（而非仅基于假设分析）” -> future similar requests should default to real sample verification, not speculative tokenizer commentary.
- The user required the output to include “原始文本 / 对应 token 序列（含 token id 和 token string） / 对关键数值字段的拆分分析” -> future similar analyses should proactively produce concrete token-by-token tables, not just narrative conclusions.
- The user asked for optional but recommended vocabulary coverage and impact analysis -> future similar runs can include lightweight coverage counts and a short implications section when easy to verify.

Key steps:

- Confirmed the local model directory exists and contains a full tokenizer bundle (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.) under `model_cache/models/Qwen/Qwen3-VL-2B-Instruct`.
- Loaded the tokenizer directly with `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` and recorded backend characteristics: `Qwen2TokenizerFast`, fast tokenizer, BPE backend, pre-tokenizer `Sequence(Split(...), ByteLevel(...))`.
- Built a small analysis script in `temp/qwen_tokenizer_analysis.py` to:
  - encode the manual probe string `bbox_2d: [10,100,231,1]`,
  - sample the first few real records from `val.norm.jsonl`,
  - print per-token `id / token / offset / substr / decoded`,
  - probe individual numbers, and
  - compute `0..999` coverage counts.
- Verified the direct tokenizer behavior on the manual string and on real dataset examples, then read back the generated JSON report from `temp/qwen_tokenizer_analysis_output.json`.
- Also checked the tokenizer vocabulary and tokenizer JSON to distinguish “tokenizer can encode digits” from “does it contain full integer tokens.”

Failures and how to do differently:

- The repo helper `scripts/tools/inspect_chat_template.py` failed on this `.norm.jsonl` sample with a CoordJSON validation error: `CoordJSON geometry arrays must contain bare coord tokens like <|coord_123|> or bare norm1000 integers in [0,999]`. This means the helper is not a reliable way to inspect this specific norm surface without adjusting the input/expected mode.
- The raw tokenizer load/inspect commands occasionally produced no immediate stdout in the notebook-like execution path; writing a dedicated script to `temp/` and reading its JSON output was the reliable workaround.
- A large `rg` search initially produced an overwhelming amount of output, so future work should keep searches narrower and prefer targeted docs plus direct tokenizer probes.

Reusable knowledge:

- The Qwen3-VL tokenizer here is `Qwen2TokenizerFast` with BPE backend and a byte-level + split pre-tokenizer. The split regex includes `\p{N}` as a single-digit matcher, so decimal numbers are split digit-by-digit rather than as whole integers.
- `0..999` do **not** exist as a complete single-token numeric vocabulary in this tokenizer surface. Only `0..9` are bare single-token numeric encodings; all larger integers tested were split into separate digit tokens.
- `vocab.json` contains only the pure digit tokens `0`–`9` and no `coord_*` tokens.
- `bbox_2d` tokenizes as `bbox` + `_` + `2` + `d`; punctuation often merges with adjacent whitespace or punctuation into tokens such as `Ġ[`, `":`, `",`, and `]}`.
- Example manual encoding for `bbox_2d: [10,100,231,1]` produced the sequence: `bbox`, `_`, `2`, `d`, `:`, `Ġ[`, `1`, `0`, `,`, `1`, `0`, `0`, `,`, `2`, `3`, `1`, `,`, `1`, `]`.
- Real sample check confirmed the same pattern for dataset-derived bboxes: e.g. `699 -> 6 9 9`, `284 -> 2 8 4`, `722 -> 7 2 2`, `336 -> 3 3 6`.
- Coordinate-style strings like `<|coord_123|>` are not atomic in this tokenizer; the probe showed they split into `<`, `|`, `coord`, `_`, `1`, `2`, `3`, `|`, `>`.
- On `val.norm.jsonl`, the bbox population-wide digit cost was substantial: `40478` boxes, `161912` coordinates, `467405` digit tokens total, averaging `2.886784...` digit tokens per coordinate and `11.547136...` per bbox.
- The repository’s CoordJSON serializer treats geometry arrays specially and only permits bare coord tokens or integers in `[0,999]` inside `bbox_2d`/`poly` contexts.

References:

- [1] `temp/qwen_tokenizer_analysis.py` — ad hoc script used to generate the verification report.
- [2] `temp/qwen_tokenizer_analysis_output.json` — machine-readable evidence, including the manual example, dataset examples, coverage counts, and coord token probe.
- [3] `scripts/tools/inspect_chat_template.py:91-99` — shows this helper hardcodes `coord_mode="coord_tokens"`, `emit_norm="norm1000"`, and `coord_tokens_enabled=True`.
- [4] `src/utils/assistant_json.py:27-29, 45-56` — geometry arrays are rendered only in `bbox_2d`/`poly` contexts, and coord-context values must be bare coord tokens or integers in `[0,999]`.
- [5] Error from the chat-template helper on norm input: `ValueError: CoordJSON geometry arrays must contain bare coord tokens like <|coord_123|> or bare norm1000 integers in [0,999]`.
- [6] Backend tokenizer evidence from JSON report: `tokenizer_class: Qwen2TokenizerFast`, `backend_model_class: BPE`, `pre_tokenizer: Sequence(... Split(... \p{N} ...), ByteLevel(...))`.

