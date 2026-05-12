thread_id: 019da8ff-951e-7752-884d-a08a7897c7e8
updated_at: 2026-04-22T02:05:35+00:00
rollout_path: /data/home/xiaoyan/AIteam/data/CoordExp/.codex/sessions/2026/04/20/rollout-2026-04-20T03-46-55-019da8ff-951e-7752-884d-a08a7897c7e8.jsonl
cwd: /data/home/xiaoyan/AIteam/data/CoordExp
git_branch: main

# Benchmarked two Stage-1 2B checkpoints on the LVIS-proxy COCO-1024 validation surface, then investigated training setups, path provenance, and a Qwen coord-vocab expansion script.

Rollout context: The user first wanted a two-phase infer-eval benchmark compare on `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.norm.jsonl`, but that exact file did not exist in the repo. After the user corrected the data choice, the workflow switched to `val.coord.jsonl` with explicit non-auto coord settings. The work was done from `/data/home/xiaoyan/AIteam/data/CoordExp` on a clean `main` checkout at commit `7403acdf9367dbac473c8587679946ae6732c006`.

## Task 1: LVIS-proxy benchmark compare (A vs B)

Outcome: success

Preference signals:

- The user corrected the earlier data mismatch with: "对，你忘记考虑到这一点了，请你帮我切换成正确的*.jsonl，不要用`auto`" -> in similar benchmark runs, default to explicit dataset paths and explicit coord-mode settings instead of relying on `auto`.
- The user required a split workflow (10-sample sanity first, then full benchmark only if sanity passed) and insisted on 4 GPUs split 2+2 -> future benchmark work should preserve a staged validation gate and per-checkpoint GPU partitioning when the user asks for reproducible comparisons.
- The user repeatedly asked to "查看进度" / "请查看结果" -> they expect concise status/metric updates from the running logs, not just a final summary.

Key steps:

- Checked the repo’s canonical infer/eval workflow via `coordexp-infer-eval-workflow` and the docs runbook (`docs/eval/WORKFLOW.md`).
- Confirmed the requested `val.norm.jsonl` under the LVIS-proxy directory does not exist; the repo only had `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl` for the proxy surface, while `val.norm.jsonl` existed only under the non-proxy COCO surface.
- Inspected `src/common/coord_standardizer.py` and `src/infer/pipeline.py` to verify that for `mode: coord`, the correct explicit prediction setting is `pred_coord_mode: norm1000` and `bbox_format: xyxy`.
- Wrote explicit YAMLs for sanity and full runs for both checkpoints, with `prompt_variant: coco_80`, `object_field_order: desc_first`, `object_ordering: sorted`, `mode: coord`, `pred_coord_mode: norm1000`, `bbox_format: xyxy`, `seed: 42`, and `generation.batch_size: 4`.
- Launched full benchmark runs using tmux/GPU partitioning with 2 GPUs per checkpoint; both completed successfully.
- Collected final metrics from `proxy_eval_bundle_summary.json` and per-view `metrics.json` files.

Reusable knowledge:

- For this repo, the LVIS-proxy benchmark headline is reported on `coco_real`; `coco_real_strict` and `coco_real_strict_plausible` are additive proxy views.
- For coord-token checkpoints evaluated against `*.coord.jsonl`, `mode: coord` with `pred_coord_mode: norm1000` and `bbox_format: xyxy` is the explicit, non-auto path.
- The infer pipeline can run distributed across local GPUs; the rollout used 2 GPUs per checkpoint and merged shard outputs afterward.
- The benchmark workspace emitted and preserved `commit.txt`, `git_status.txt`, and `workspace_patch.diff` inside the per-run log directories for reproducibility.

Failures and how to do differently:

- The original requested `val.norm.jsonl` path was wrong for the LVIS-proxy directory; future similar tasks should validate the exact input JSONL path before launching.
- `auto` was rejected by the user, and the correct replacement was to spell out the coord semantics explicitly.
- For the full benchmark, the checkpoints were already merge-only full models, so there was no need to rerun training; use the merged dirs directly for inference.

References:

- [1] Canonical benchmark skill: `coordexp-infer-eval-workflow`
- [2] Full benchmark run dirs:
  - `output/infer/coco1024_lvisproxy_valfull_hardce_softce_w1_gate1332`
  - `output/infer/coco1024_lvisproxy_valfull_desc_first_lvis_proxy_merged`
- [3] Final `coco_real` metrics:
  - A: `bbox_AP=0.3947069289742706`, `bbox_AP50=0.5607149411778245`, `bbox_AP75=0.4177241165727422`, `f1ish@0.50_f1_full_micro=0.6629226732702455`
  - B: `bbox_AP=0.38145142718433417`, `bbox_AP50=0.5261004654307864`, `bbox_AP75=0.4091713580393683`, `f1ish@0.50_f1_full_micro=0.5967342032106989`
- [4] Other views for A:
  - `coco_real_strict`: `AP=0.3820744069547984`
  - `coco_real_strict_plausible`: `AP=0.37450540092183554`
- [5] The 10-sample sanity run used the same explicit coord settings and completed before the full run.

## Task 2: Coord vocab expansion / Qwen checkpoint resize

Outcome: partial

Preference signals:

- The user asked for exact source and destination paths for `scripts/tools/expand_coord_vocab.py` and wanted both Qwen 2B and 4B checkpoints moved to `*-coordexp` paths -> in similar setup tasks, use exact paths and verify whether the target already exists before rewriting anything.

Key steps:

- Inspected `scripts/tools/expand_coord_vocab.py` with Serena to read its CLI and save flow.
- Confirmed the script defaults to the 2B base (`_default_2b_dir()`), takes `--src`, `--dst`, `--num-bins` (default 999), and saves a self-contained checkpoint with tokenizer/model plus extra processor files.
- Verified that `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` already exists.
- Found that the local 4B base checkpoint directory `model_cache/models/Qwen/Qwen3-VL-4B-Instruct` does **not** exist in the workspace, while `model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp` already exists.
- Attempted to run the resize command with the missing 4B source path; it failed with Hugging Face path validation because the source path was not a local directory for this workspace.

Reusable knowledge:

- `expand_coord_vocab.py` is intended to be run as a local filesystem resize step that loads a source checkpoint, adds coord tokens, ties weights, and saves a new `*-coordexp` directory.
- The script’s own default `--dst` pattern is `.../Qwen3-VL-2B-Instruct-coordexp`, and it warns that `coord_1000` is not added unless `--num-bins 1000` is used.
- The 4B coordexp target directory already exists locally and contains a complete 4-shard model (`model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`, tokenizer, processor, config, README, coord_tokens, etc.).

Failures and how to do differently:

- The direct resize attempt for 4B failed because the source checkpoint `model_cache/models/Qwen/Qwen3-VL-4B-Instruct` was missing locally; the command treated the path as a Hub repo id and errored.
- Before rerunning the script, future agents should first locate the actual 4B base checkpoint on disk or confirm that the existing `Qwen3-VL-4B-Instruct-coordexp` directory is already the intended serving target.

References:

- [1] Script path: `scripts/tools/expand_coord_vocab.py`
- [2] Script behavior: it loads `--src`, adds coord tokens, resizes embeddings deterministically, ties weights, then writes a self-contained `--dst`
- [3] Existing local 2B coordexp path: `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`
- [4] Existing local 4B coordexp path: `model_cache/models/Qwen/Qwen3-VL-4B-Instruct-coordexp`
- [5] Missing local 4B base path: `model_cache/models/Qwen/Qwen3-VL-4B-Instruct`
- [6] Failed resize attempt error: `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'model_cache/models/Qwen/Qwen3-VL-4B-Instruct'`
