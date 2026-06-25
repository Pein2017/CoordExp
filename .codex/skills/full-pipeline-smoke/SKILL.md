---
name: full-pipeline-smoke
description: Use when validating a CoordExp change through a production-like multi-stage data/train-or-rollout/infer/eval/artifact smoke path, including checkpoint-save and evaluation-step behavior, not for narrow unit or config checks.
metadata:
  short-description: Full-cycle smoke workflow
---

# Full Pipeline Smoke

Smoke configs must be production configs with fewer samples. Preserve model, template, packing, max lengths, geometry, decoding, checkpoint, and artifact contracts unless those knobs are the feature under test.

## What A Full Smoke Proves

Exercise the relevant chain:

- data read and sample contract;
- Qwen3-VL template/multimodal encode;
- packing/position ids when enabled;
- forward/backward/optimizer or rollout step;
- checkpoint save/load-adjacent behavior for the changed checkpoint surface;
- at least one configured evaluation step when the production config evaluates;
- feature-specific decode/matching/scoring/eval path;
- logs/metrics;
- reproducibility artifacts.

Expected artifacts by surface:

- training: `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`, `logging.jsonl`, and the checkpoint directory/file set implied by the production save mode;
- infer/eval: `summary.json`, `resolved_config.json`, `resolved_config.path`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`;
- guarded eval: `metrics_guarded.json`, `per_image_guarded.json`, `duplicate_guard_report.json`.

## YAML Pattern

```yaml
extends:
  - ../prod/<variant>.yaml
  - common_prodlike.yaml
```

Allowed smoke overrides:

- run/output/log dir;
- `training.max_steps` or tightly justified epoch cap, but high enough to cross the first production save and eval boundaries;
- checkpoint cadence shortened to force at least one save, while preserving the same checkpoint mode and saved module/artifact contract;
- eval cadence shortened to force at least one evaluation when production evaluates, while preserving the same eval dataset/template/loss path;
- sample limits;
- dataloader stability knobs: workers `0`, prefetch `null`, persistent workers `false`.

Do not set `training.save_strategy: "no"` for a full smoke unless checkpointing is genuinely out of scope and explicitly called out in the report. Avoid overriding learning rate, optimizer, checkpoint mode, template, packing, max lengths, or decoding unless that is the test.

## Design Checklist

1. Identify the production config.
2. Choose the minimum steps that hit the feature path.
3. Set sample limits high enough for those steps.
4. Force the smoke through at least one checkpoint save when production saves.
5. Force the smoke through at least one eval step when production evaluates.
6. Keep length/packing/geometry contracts production-like.
7. Name the artifacts that prove success before launching.

## Launch Gate

Before declaring a full smoke sufficient for production training:

- Verify the training loop reaches the planned terminal step, not just the first optimizer step.
- Verify at least one checkpoint is written and contains the expected adapter/full-model files, `modules_to_save`, and any feature-specific trainable modules.
- Verify at least one eval step runs successfully when production has eval enabled; check eval metrics are finite and that trainer prediction/eval paths receive normal model outputs.
- Verify `logging.jsonl` contains the feature metrics under the canonical namespace and does not contain retired/debug-only metric namespaces.
- Verify `effective_runtime.json` records the intended world size, per-device batch size, gradient accumulation, effective batch size, save cadence, eval cadence, and packing state.

Run from repo root:

```bash
rtk python <entrypoint> --config <smoke.yaml>
```

Codex shells initialize the `ms` conda environment by default. Use raw `python ...` only when exact stdout matters.

## CoordExp Gotchas

- Worktrees may lack ignored data/model roots. Prefer symlinks at the same relative paths expected by config, not path overrides:
  ```bash
  mkdir -p public_data/coco
  test -e public_data/coco/rescale_32_1024_bbox_max60 || \
    ln -s /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 public_data/coco/rescale_32_1024_bbox_max60
  test -e model_cache || ln -s /data/CoordExp/model_cache model_cache
  ```
- Do not stage runtime symlinks unless explicitly requested.
- For server-mode rollouts, clear local proxy vars and ensure `NO_PROXY` includes `127.0.0.1,localhost`.
- Stage-2 rollout correction uses `stage2_rollout_correction` and rollout-correction configs under `configs/stage2/rollout_correction/`.
- Latest recursive detection sidecars require the current runtime-supported packing policy; do not silently enable unsupported packing.
- Raw-text `xyxy` norm1000 infer/eval must use `infer.mode: text` and `infer.pred_coord_mode: norm1000`.
- `cxcy_logw_logh` or `cxcywh` evidence is valid only for checkpoints trained on that serialization.

## Report

Always label scope: `tiny`, smoke sample count, `val200`, `limit=200`, first-200, full-val, proxy view, raw-text vs coord-token, bbox format, checkpoint id, and launch shape. Verify top-level summaries/manifests, not just process exit.
