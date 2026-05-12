thread_id: 019e0bd4-0b6d-78e1-939f-6e9eb0905b56
updated_at: 2026-05-09T10:23:59+00:00
rollout_path: /data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/05/09/rollout-2026-05-09T08-21-46-019e0bd4-0b6d-78e1-939f-6e9eb0905b56.jsonl
cwd: /data/home/xiaoyan/AIteam/data/CoordExp
git_branch: main

# Assessing a move from max-object filtering to compact-full token-length budgeting for COCO

Rollout context: The user wanted to reuse an existing COCO 1024-budget pipeline, but reconsidered the old `max_objects=60` cap because the new `compact-full` rendering should reduce sequence length enough to recover images that were previously dropped. The discussion stayed focused on repo facts, current config contracts, and whether a length-based filter is feasible.

## Task 1: Confirm current compact-full / COCO 1024 max60 contract

Outcome: success

Preference signals:
- The user asked to “根据max tokens length来处理每张图片的objects数量” and explicitly contrasted this with the old approach of “控制object count” -> future similar tasks should default to thinking in terms of sequence-budgeted filtering rather than fixed object-count caps.
- The user framed the target as `length(input images + prompt token + assistant sequence in compact form) <= 16k/12k` and said `max object count 应该会比60要多很多` -> future agents should expect the user to prefer a token-budget abstraction when compact-full makes object-count proxies too conservative.

Reusable knowledge:
- Current compact-full Stage-1 configs still point at `public_data/coco/rescale_32_1024_bbox_max60/{train,val}.coord.jsonl`, with `detection_template.id=compact_full`, `coordinate_surface=coord_token`, `bbox_format=xyxy`, and `token_rows` enabled for 1000 coord rows plus `<|object_ref_start|>` / `<|box_start|>`.
- The 1002-row tokenizer contract resolves as expected on the coordexp tokenizer cache: `<|object_ref_start|>=151646`, `<|box_start|>=151648`, `<|coord_0|>=151670`, `<|coord_999|>=152669`, and the coord rows are contiguous.
- `public_data/run.sh` enforces that `PUBLIC_DATA_MAX_OBJECTS` is only supported for the `coord` stage; it refuses to apply that cap to `rescale`, which means any new length-based cap should be implemented as a separate offline stage/preset, not by overloading `rescale`.
- `compact_full` rendering happens in the detection runtime/template path (`src/detection/template.py`, `src/detection/ir.py`, `src/detection/runtime.py`), not by embedding rendered assistant text into raw JSONL.
- The compact-full assistant surface is much shorter per object than the old verbose JSON surface; the measured compact-full assistant token counts on representative COCO outliers were ~762 tokens for a 90-object train image and ~543 tokens for a 62-object val image, roughly 8.5 tokens/object.

Failures and how to do differently:
- The first attempt to replay the dataset pipeline with `rescale_32_1024_bbox` hit the runner’s freshness guard: `RuntimeError: Rescale target preset is not fresh; refusing in-place overwrite.` Future reruns should not try to rescale into an existing preset directory unless the whole directory is intentionally removed first.
- The existing `max60` preset is a proxy for old sequence-length pressure, not a true compact-full budget gate. Future work should avoid treating it as a hard architectural truth once compact-full budgeting is available.
- Assistant-only token counts are informative but not enough for the final gate; the final budgeting rule must include image-token expansion and chat-template prompt overhead.

References:
- [1] Configs and contract surfaces confirmed:
  - `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`
  - `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml`
  - `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_separator2.yaml`
  - `public_data/coco/rescale_32_1024_bbox_max60/{train,val}.coord.jsonl`
- [2] Tokenizer contract check:
  - `object_ref 151646`, `box_start 151648`, `coord_0 151670`, `coord_999 152669`, `coord_contiguous True`, `unique_trainable_rows 1002`
- [3] COCO object-count distribution under the base 1024 preset:
  - `train.jsonl`: `n 117266 max 90 mean 7.248 p95 22 p99 32 gt60 19 objects_gt60 1291`
  - `val.jsonl`: `n 4952 max 62 mean 7.337 p95 22 p99 34 gt60 1 objects_gt60 62`
- [4] Compact-full assistant length spot-checks on worst-case examples:
  - train 90-object sample -> `assistant_tokens 762`, `tokens_per_object 8.47`
  - val 62-object sample -> `assistant_tokens 543`, `tokens_per_object 8.76`

## Task 2: Evaluate whether compact-full can replace max-object filtering with a token-length budget

Outcome: success

Preference signals:
- The user explicitly said the new target is “`length(input images + prompt token + assistatn sequence in compact form) <= 16k/12k`即可” -> future agents should treat the token budget as the governing constraint for this family of datasets.
- The user’s intent was not just to prune less aggressively, but to “将之前无法吃得下的地方数据都恢复过来而不是扔掉” -> future agents should consider length-based filtering policies that preserve dense images rather than defaulting to image-level dropping.

Key steps:
- Checked the current COCO base preset object-count distribution and found only 19 train images and 1 val image above 60 objects; the maximums were 90 (train) and 62 (val).
- Measured compact-full assistant lengths for representative worst cases using the actual tokenizer/rendering path and found the assistant text itself to be well under 1k tokens even on the densest samples.
- Cross-checked the compact-full template/runtime boundary in `src/detection/template.py`, `src/detection/ir.py`, and `src/detection/data.py` to confirm that compact-full rendering is a runtime concern layered over structured JSONL, not a new raw-data format.

Reusable knowledge:
- For COCO 1024-budget data, `max60` is likely over-conservative once compact-full is used; the observed dense examples are far below a 12k/16k total budget even before optimization.
- The right implementation shape is a new offline length-filtered preset or analyzer that computes full sample length using the actual compact-full renderer/tokenizer path, then emits standard structured JSONL for training.
- A length-budgeted preset should count at least: image tokens, system prompt, user prompt, compact-full assistant text, and chat-template overhead.
- The filter should likely record explicit manifest stats such as `budget_tokens`, `surface=compact_full`, `overflow_policy`, `objects_seen`, `objects_written`, and `images_dropped` so downstream training is reproducible.

Failures and how to do differently:
- The attempt to estimate full sample lengths by hand initially produced only assistant-length intuition; future work should compute exact end-to-end lengths with the real tokenizer/processor path before changing the dataset contract.
- One attempt to render via the wrong object shape hit a template validation error (`DetectionObjectEntry` missing `bbox_2d` in the path used). Future code should feed the template the normalized sample object shape expected by `CompactFullTemplate.render_assistant`.
- A raw one-liner to compute COCO object-count stats failed because newline escaping was malformed. For quick stats, use a heredoc Python script instead of inline `-c` when loops are involved.

References:
- [1] `public_data/pipeline/stages.py`:
  - `MaxObjectsFilterStage` enforces object-count filtering only; it is the current place where `max_objects` is applied.
- [2] `public_data/run.sh`:
  - `PUBLIC_DATA_MAX_OBJECTS` is only allowed for `coord`, with explicit errors if used on `rescale` or other stages.
- [3] `src/detection/ir.py` / `src/detection/data.py`:
  - `DetectionDocument.from_normalized_sample(...)` and the normalized/raw detection sample adapters are the relevant runtime bridge for compact-full rendering.
- [4] Spot-check measurements:
  - `public_data/coco/rescale_32_1024_bbox/train.jsonl line 31700 objects 90 assistant_tokens 762 chars 8238 tokens_per_object 8.47`
  - `public_data/coco/rescale_32_1024_bbox/val.jsonl line 3715 objects 62 assistant_tokens 543 chars 5676 tokens_per_object 8.76`
- [5] Current COCO base preset counts:
  - `train.jsonl n 117266 max 90 ... gt60 19 images`
  - `val.jsonl n 4952 max 62 ... gt60 1 image`
