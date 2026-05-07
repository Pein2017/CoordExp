---
name: full-pipeline-smoke
description: Use when validating CoordExp features through a production-like smoke path that exercises data, packing, training or rollout, inference, scoring, evaluation, metrics, and reproducibility artifacts.
metadata:
  short-description: Full-cycle smoke workflow
---

# Full-Cycle Smoke Testing (Production-Like)

Use this skill whenever you add a new feature and want a **real smoke test** that exercises the **entire pipeline path** (not a unit test) while staying close to production conditions.
Start from `docs/IMPLEMENTATION_MAP.md` for the area under test, and keep the smoke aligned with the relevant docs/spec contract.

## Core Principle

**Smoke configs must be production configs with fewer samples.**
Keep all main hyperparameters identical to production; only change runtime knobs.

This prevents “smoke drift” where smoke passes but production fails.

## What “Full Pipeline Cycle” Means

A real smoke test should pass through as many of these as relevant to the feature:

- Data reading + sample contract validation
- Template encode + multimodal handling (images/video, if applicable)
- Packing / position_ids contracts (if packing is enabled)
- Forward + backward + optimizer step (at least 1–3 steps)
- Any feature-specific stage (e.g., rollouts, matching, decoding, post-processing)
- Logging/metrics emission (at least one log line with the key metrics)
- Artifact creation and provenance breadcrumbs:
  - training: `resolved_config.json`, `runtime_env.json`, `effective_runtime.json`, `pipeline_manifest.json`, `experiment_manifest.json`, `run_metadata.json`
  - infer/eval: `summary.json`, `resolved_config.json`, `resolved_config.path`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `metrics.json`
  - guarded eval, when enabled: `metrics_guarded.json`, `per_image_guarded.json`, `duplicate_guard_report.json`

## YAML Hierarchy Pattern (Required)

Use a consistent layout:

- Production configs live in a dedicated folder (e.g. `configs/<area>/prod/`)
  - One `base.yaml` defines shared hyperparameters
  - Variant configs override only the minimal “experiment axis”

- Smoke configs live under `configs/<area>/smoke/`
  - A single `common_prodlike.yaml` centralizes smoke runtime overrides
  - One entrypoint YAML per smoke scenario extends prod + common

Recommended inheritance order:

```yaml
extends:
  - ../prod/<variant>.yaml
  - common_prodlike.yaml
```

Reason: later entries override earlier ones, so smoke can override runtime knobs without touching prod hyperparams.

### Keep legacy paths stable (optional but recommended)

If the repo already has old smoke YAMLs referenced by docs/tests, convert them to 1–3 line wrappers:

```yaml
extends: smoke/<new_smoke_entrypoint>.yaml
```

## Allowed Smoke Overrides

Allowed keys to override in smoke entrypoints (or `common_prodlike.yaml`):

- `training.run_name`
- `training.output_dir`
- `training.logging_dir`
- `training.max_steps` (or `num_train_epochs` only for specific reasons)
- `training.save_strategy: "no"` (and keep `eval_strategy: "no"` unless evaluation is the feature)
- dataset limits:
  - `custom.train_sample_limit`, `custom.val_sample_limit`
  - or feature-equivalent limit knobs in the repo
- dataloader stability knobs:
  - `data.dataloader_num_workers: 0`
  - `data.dataloader_prefetch_factor: null`
  - `data.dataloader_persistent_workers: false`

Avoid overriding (unless the explicit purpose is to test these knobs):

- learning rates / optimizer settings
- packing configuration / max lengths
- model checkpoint / template
- decoding parameters (unless decoding itself is the feature under test)

## Smoke Design Checklist

Before writing any YAML:

1) Identify the **production config** that represents the “real run”
2) Decide the minimum number of steps to hit the feature path
   - e.g., if a schedule has multiple phases, ensure smoke reaches each phase at least once
3) Set sample limits so the dataloader can provide enough samples for those steps
4) Ensure length caps in smoke match prod (so truncation/packing contracts are exercised)
5) Decide which artifacts prove the feature path ran; do not rely on a final log line alone

## Regression Loop: Feature Building -> Smoke -> Guardrail

Recommended loop:

1) Implement feature (code/config)
2) Add or update smoke YAML overlay(s)
3) Run the smoke command(s) and verify:
   - GPU is actually used (no hang)
   - At least one optimizer step completes
   - Key metrics counters/logs are present
   - No silent truncation/invalid parse spikes (feature-dependent)
   - resolved configs and manifests exist and point to the intended inputs/checkpoints
   - infer/eval jobs emit the expected raw, scored, guarded, and summary artifacts
4) Add/adjust a minimal test that runs the smoke config (if repo has such tests)
5) Document the canonical smoke command in the relevant runbook

## CoordExp Verification Surface

Use `conda run -n ms python ...` from the repo root. Wrap noisy commands with `rtk` only when filtered output is acceptable.

Before a smoke run:

- confirm the config extends the intended production config
- confirm only runtime knobs changed (`max_steps`, sample limits, output/log dirs, save/eval cadence)
- confirm prompt, packing, geometry, checkpoint, decoding, and coordinate-surface settings match production unless the smoke explicitly tests them
- for worktrees or `temp/` runs, confirm image roots and JSONL paths resolve to stable shared data, not transient scratch

After a smoke run:

- verify top-level summaries/manifests, not only process exit
- for inference, inspect `summary.json` and `resolved_config.json` instead of reconstructing CLI args
- for downstream eval/vis, verify `resolved_config.path` exists next to `gt_vs_pred.jsonl`
- for COCO/LVIS proxy paths, report the exact view (`coco_real`, `coco_real_strict`, `coco_real_strict_plausible`)
- label scope precisely: smoke sample count, `val200`, `limit=200`, first-200, full-val, raw-text vs coord-token, and checkpoint id when relevant

## Worktree / Branch Smoke Runs

When running smoke tests from a worktree or different branch, keep authored YAML paths stable whenever possible. The preferred fix for missing ignored/heavy runtime roots is to add local worktree symlinks, not to rewrite config paths or create worktree-specific path overrides.

### Preferred Path Strategy: Symlink Heavy Roots

1. **Preserve the config path exactly.**
   If the production or smoke config says:

   ```yaml
   custom:
     train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
     val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
   model:
     model: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
   ```

   keep those paths unchanged.

2. **Create missing local symlinks to shared roots.**
   In isolated git worktrees, ignored data/model folders may not exist even though the main checkout has them. Add symlinks at the same relative locations expected by config:

   ```bash
   # From the worktree root.
   mkdir -p public_data/coco
   test -e public_data/coco/rescale_32_1024_bbox_max60 || \
     ln -s /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 \
       public_data/coco/rescale_32_1024_bbox_max60

   test -e model_cache || \
     ln -s /data/CoordExp/model_cache model_cache
   ```

   Adjust the symlink target only when the shared root on the current host differs. Do not stage these symlinks unless the user explicitly asks; they are runtime conveniences for ignored heavy assets.

3. **Verify the same paths the config will use.**

   ```bash
   test -f public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
   test -f model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp/config.json
   ```

This keeps dataset identity, static-packing fingerprints, model provenance, and run manifests aligned with the production config. It also avoids the common failure where copied JSONL slices under `temp/` make relative image paths resolve against the wrong directory.

### When To Override Paths Instead

Use environment variables or worktree-specific path overrides only when symlinks are impossible or when the experiment is explicitly testing a different dataset/model root. If you do override paths, record that in the run note and artifact interpretation because the resolved path identity changed.

## Runtime Guardrails (CoordExp defaults)

- Prefer YAML-first changes over adding new CLI flags.
- For current Stage-1 baseline work, prefer `configs/stage1/`; treat `configs/fusion/` as the historical/experimental multi-dataset surface.
- For Stage-2 smoke, use `configs/stage2_two_channel/` and preserve the authored `stage2_ab.pipeline.*` or `rollout_matching.pipeline.*` contract.
- For infer/eval smoke, prefer the unified YAML pipeline under `configs/infer/`, `configs/postop/`, `configs/eval/`, and `configs/bench/`.
- For raw-text `xyxy` norm1000 smoke, set `infer.mode: text`, `infer.pred_coord_mode: norm1000`, and keep the repaired confidence-scoring path in the loop.
- For `cxcy_logw_logh` or `cxcywh`, only treat outputs as valid evidence for checkpoints trained on that exact serialization.
- If local HTTP services are involved (server-mode rollouts), ensure proxy is disabled:
  - `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`
  - ensure `NO_PROXY` contains `127.0.0.1,localhost`
- Avoid destructive git commands unless explicitly requested by the user.
