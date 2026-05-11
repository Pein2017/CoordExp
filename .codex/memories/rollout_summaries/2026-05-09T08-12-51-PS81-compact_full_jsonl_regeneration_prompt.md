thread_id: 019e0bcb-e13f-7911-9a63-a301133b7d81
updated_at: 2026-05-09T08:20:36+00:00
rollout_path: /data/CoordExp/.codex/sessions/2026/05/09/rollout-2026-05-09T08-12-51-019e0bcb-e13f-7911-9a63-a301133b7d81.jsonl
cwd: /data/CoordExp
git_branch: main

# User asked for a cross-node prompt to regenerate compact-full / 1002-row COCO JSONL data after a recap of the current codebase and docs

Rollout context: The user requested a quick browse/recap of the current codebase and documentation around the latest `compact-full` data processing, then asked for a prompt that could be given to another Codex agent on a different development environment (with the same codebase pulled) so it could understand the background and re-generate the required `*.jsonl` files. The main target was the dataset for the 1002 special-token rows / compact-full surface.

## Task 1: Recap current compact-full data pipeline and prepare a transferable prompt

Outcome: success

Preference signals:
- The user said: "请参考我最新的`compact-full`相关的文档以及数据处理... 请给我一个`prompt`让另外一个节点的（pull了当前codebase）的codex agent来了解背景并重新生成相应所需要的`*.jsonl`。" -> the user wants future agents to provide an execution-ready handoff prompt for another machine, not just a summary.
- The user added: "主要是`1002`个特殊 tokens 的数据集。请先浏览当前的 codebase 和文档快速recap一下先。" -> future agents should first quickly locate current docs/code and then produce a compact, reproducible recap tied to the current contract.
- The user’s focus on the other node having already pulled the codebase implies the handoff prompt should assume the repo is present and emphasize re-generation/verification steps rather than setup from scratch.

Key steps:
- The agent used the repo’s navigation docs first (`docs/AGENT_INDEX.md`, `docs/catalog.yaml`) and the relevant data/training docs to locate authoritative compact-full sources.
- It identified the latest compact-full surfaces as `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml`, `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml`, and `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_separator2.yaml`.
- It checked the data contract and pipeline docs (`docs/data/CONTRACT.md`, `docs/data/PREPARATION.md`, `public_data/coco/README.md`, `public_data/run.sh`) to confirm the correct COCO preprocessing path and the runner constraints.
- It verified the current local dataset scale with `wc -l` and inspected the preset manifest and filter stats so the final prompt could ask the other node to reproduce the same artifact shape.

Failures and how to do differently:
- One Serena symbol lookup was attempted before properly activating the CoordExp project, which returned a file-not-found error. After activating the correct project, symbol inspection worked. Future similar runs should activate the intended project before using Serena symbol tools.
- The output search over repo docs/code produced a very large amount of irrelevant matches because the query was broad. Future agents should narrow searches earlier around the known compact-full files and data preset names.
- The agent’s first pass was exploratory and broad; for future handoffs, it is more efficient to move quickly from repo-router docs to the exact data preset and manifest files once the target surface is known.

Reusable knowledge:
- The current compact-full Stage-1 data surface for this rollout is `public_data/coco/rescale_32_1024_bbox_max60/{train,val}.coord.jsonl` with `image_root: public_data/coco/rescale_32_1024_bbox_max60`.
- The relevant current compact-full configs are the latest-detection `support2` baseline and the `prefix_rollin` ablations under `configs/stage1/recursive_detection_ce_latest/`.
- The 1002-row compact-full token-row contract is 1000 coord rows plus `<|object_ref_start|>` and `<|box_start|>`, and the agent confirmed the repo’s expected IDs and the contiguous coord-token range from the docs/configs.
- The public-data runner enforces that `PUBLIC_DATA_MAX_OBJECTS` is only supported on the `coord` stage, so a two-step `rescale` then `coord` flow is the correct way to reproduce the max60 dataset directory.
- The local generated dataset size was confirmed as 117,247 train and 4,951 val records, with matching raw/norm/coord counts.

References:
- [1] `docs/AGENT_INDEX.md` and `docs/catalog.yaml` as the repo navigation spine for current docs/routes.
- [2] `docs/data/CONTRACT.md` and `docs/data/PREPARATION.md` as the authoritative JSONL and preprocessing contract.
- [3] `docs/training/STAGE1_OBJECTIVE.md` showing the compact-full prefix-rollin and latest compact detection notes.
- [4] `configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml` and `configs/stage1/recursive_detection_ce_latest/ablation/compact_full_prefix_rollin_balance2.yaml` as the key compact-full training configs.
- [5] `public_data/run.sh` showing the dataset runner modes and the restriction that `PUBLIC_DATA_MAX_OBJECTS` only applies to `coord`.
- [6] `public_data/coco/rescale_32_1024_bbox_max60/pipeline_manifest.json` plus `train.filter_stats.json` / `val.filter_stats.json` as the concrete generated-artifact manifest and filtering evidence.
- [7] `wc -l` output confirming the current dataset scale: `117247` train / `4951` val for both `*.jsonl`, `*.norm.jsonl`, and `*.coord.jsonl`.

## Task 2: Produce a reusable prompt for the other node to regenerate the JSONL artifacts

Outcome: success

Preference signals:
- The user asked for a prompt that another Codex agent could use directly, so the final product needed to be operational and detailed rather than a terse summary.
- The user specifically wanted background plus regeneration instructions for the `*.jsonl` artifacts, indicating they care about a handoff that preserves both context and execution steps.

Key steps:
- The agent synthesized a full prompt that tells the other node to first recap docs, verify the compact-full contract, confirm the tokenizer/token-row assumptions, and then regenerate the COCO raw/rescaled/coord artifacts.
- The prompt explicitly distinguishes raw JSONL, norm1000 JSONL, and coord-token JSONL, and instructs the other node not to add special tokens or resize embeddings.
- It includes the exact expected file paths, the two-stage `rescale` + `coord` workflow, the validation commands, and the expected counts and manifest fields to report back.

Failures and how to do differently:
- None beyond the general note that future agents should keep the prompt focused on exact artifact generation and validation rather than broad repository commentary.

Reusable knowledge:
- The public-data workflow for this case should be treated as a reproducible pipeline: raw COCO convert → 1024-budget rescale → max60 coord-token derivation → validate → inspect chat template.
- The compactor/full contract is anchored by the current docs and configs rather than ad hoc local assumptions; the handoff prompt should always cite those files.

References:
- [1] The final prompt text included the key execution sequence: `./public_data/run.sh coco download`, `./public_data/run.sh coco convert`, `./public_data/run.sh coco rescale --preset rescale_32_1024_bbox -- --image-factor 32 --max-pixels $((32*32*1024))`, then `PUBLIC_DATA_MAX_OBJECTS=60 ./public_data/run.sh coco coord --preset rescale_32_1024_bbox`, followed by `./public_data/run.sh coco validate --preset rescale_32_1024_bbox_max60`.
- [2] The prompt preserved the repository-specific contract notes: `detection_template.id: compact_full`, `coordinate_surface: coord_token`, `bbox_format: xyxy`, and the 1002-row token contract with the expected IDs for `<|object_ref_start|>`, `<|box_start|>`, and the coord row range.
- [3] The prompt also included the exact expected artifact names under `public_data/coco/rescale_32_1024_bbox_max60/`, including `train.jsonl`, `train.norm.jsonl`, `train.coord.jsonl`, `val.jsonl`, `val.norm.jsonl`, `val.coord.jsonl`, `pipeline_manifest.json`, and the filter stats files.
