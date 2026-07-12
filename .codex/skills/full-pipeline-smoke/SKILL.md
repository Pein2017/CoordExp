---
name: full-pipeline-smoke
description: Use for a production-like CoordExp data, training, checkpoint, inference, evaluation, and artifact smoke; not for a narrow unit, schema, or config check.
---

# Full Pipeline Smoke

Use the smallest run that crosses every changed production boundary. Keep model,
template, geometry, packing, lengths, loss, optimizer, checkpoint, decode, and
artifact semantics production-like unless that surface is under test.

## Current route

Start from `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, and the relevant stable
`coordexp-swift-*` specs. Current roots and entrypoints are:

- training: `configs/coordexp_swift/prod/`, `configs/coordexp_swift/smoke/`,
  and `python -m src.train`;
- inference: `configs/coordexp_swift/infer/` and `python -m src.infer`;
- evaluation: `scripts/evaluate_detection.py` and
  `src/eval/detection_consumer.py`.

`configs/stage1/`, `configs/stage2/`, `src/sft.py`, `src/trainers/`, and the old
`src/infer/` package are historical MS-Swift routes. Use them only when the user
explicitly requests historical reproduction.

## Design the smoke

1. Name the production config, changed boundaries, and proof artifacts.
2. Select an existing config under `configs/coordexp_swift/smoke/` when it
   preserves the relevant production contract; otherwise derive the smallest
   explicit smoke config from the current production config.
3. Limit samples or steps only enough to keep the changed path real. Force one
   checkpoint save and one eval-forward step when those production surfaces are
   in scope.
4. Launch from the repository root in the `ms` environment.
5. Check artifacts and semantic counters, then run current inference/evaluation
   when the change crosses those boundaries.

Canonical command shapes:

```bash
conda run -n ms python -m src.train --config configs/coordexp_swift/smoke/<smoke>.yaml
conda run -n ms python -m src.infer --config configs/coordexp_swift/infer/<infer>.yaml
conda run -n ms python scripts/evaluate_detection.py \
  --artifact-dir <infer-run-dir> --out-dir <infer-run-dir>/eval
```

## Evidence gate

Require the evidence relevant to the changed surface:

- data/template/packing: accepted sample counts, semantic spans, position or
  packing receipts, and no silent drops or reordering;
- training: terminal planned step, finite losses/gradients, canonical metrics,
  and resolved runtime/config receipts;
- checkpoint: at least one written checkpoint with the expected adapter,
  selected-token, and handoff composition;
- eval-forward: at least one configured step with finite outputs when enabled;
- inference: merged raw/scored row parity, provenance sidecar, selected-token
  score evidence, and run summary;
- evaluation: `metrics.json`, `coco_gt.json`, and `coco_predictions.json` from
  the current Swift evaluator.

Typical training receipts include `resolved_config.json`,
`effective_runtime.json`, `experiment_manifest.json`, `run_metadata.json`, and
`logging.jsonl`; exact files remain contract/config dependent. Current Swift V1
evaluation is aggregate-only. Do not require legacy guarded/F1-ish/LVIS outputs
unless the task explicitly targets that historical evaluator family.

## Guardrails

- Do not disable checkpointing or evaluation merely to make a full smoke pass;
  use a narrower validation label if those surfaces are out of scope.
- Do not silently reduce `global_max_length`, packing, geometry, serialization,
  or distributed shape when the changed behavior depends on them.
- A worktree may need approved local symlinks for ignored data/model roots; do
  not stage them.
- Tiny, two-row, val200, first-200, and full-validation results are different
  evidence scopes. Report the exact config, checkpoint, artifact root, sample
  scope, bbox/serialization mode, and skipped boundaries.
