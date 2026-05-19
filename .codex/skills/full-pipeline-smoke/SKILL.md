---
name: full-pipeline-smoke
description: Use when validating a CoordExp change through a production-like data/train-or-rollout/infer/eval/artifact smoke path.
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
- feature-specific decode/matching/scoring/eval path;
- logs/metrics;
- reproducibility artifacts.

Expected artifacts by surface:

- training: `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`;
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
- `training.max_steps` or tightly justified epoch cap;
- `training.save_strategy: "no"`;
- sample limits;
- dataloader stability knobs: workers `0`, prefetch `null`, persistent workers `false`.

Avoid overriding learning rate, optimizer, checkpoint, template, packing, max lengths, or decoding unless that is the test.

## Design Checklist

1. Identify the production config.
2. Choose the minimum steps that hit the feature path.
3. Set sample limits high enough for those steps.
4. Keep length/packing/geometry contracts production-like.
5. Name the artifacts that prove success before launching.

Run from repo root:

```bash
rtk conda run -n ms python <entrypoint> --config <smoke.yaml>
```

Use raw `conda run -n ms python ...` only when exact stdout matters.

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
- Stage-2 two-channel uses `stage2_ab.pipeline`; rollout-aligned uses `rollout_matching.pipeline`.
- Latest recursive detection sidecars require the current runtime-supported packing policy; do not silently enable unsupported packing.
- Raw-text `xyxy` norm1000 infer/eval must use `infer.mode: text` and `infer.pred_coord_mode: norm1000`.
- `cxcy_logw_logh` or `cxcywh` evidence is valid only for checkpoints trained on that serialization.

## Report

Always label scope: `tiny`, smoke sample count, `val200`, `limit=200`, first-200, full-val, proxy view, raw-text vs coord-token, bbox format, checkpoint id, and launch shape. Verify top-level summaries/manifests, not just process exit.
