---
name: coordexp-infer-eval-workflow
description: Use when launching, repairing, auditing, or summarizing CoordExp infer/scoring/eval/Oracle-K/proxy-bundle artifact workflows.
---

# CoordExp Inference And Evaluation Workflow

Use YAML-first production paths. Do not invent stable CLI flags when config already captures the run.
Treat this skill as the stable workflow guide, not a promise that one exact script path will never move.

## Entry Points

- Primary pipeline surfaces:
  - infer entrypoints such as `scripts/run_infer.py`, `src/infer/pipeline.py::run_pipeline`, `src/infer/engine.py::InferenceEngine.infer`
  - confidence / scoring surfaces such as `scripts/postop_confidence.py`, `src/eval/confidence_postop.py`
  - evaluation surfaces such as `scripts/evaluate_detection.py`, `src/eval/detection.py::evaluate_and_save`
  - proxy / bundle surfaces such as `scripts/evaluate_proxy_detection_bundle.py`, `src/eval/proxy_eval_bundle.py`
  - artifact ownership such as `src/infer/artifacts.py`, `src/eval/artifacts.py`
- Workflow references:
  - `docs/eval/WORKFLOW.md`
  - `docs/eval/CONTRACT.md`
  - `docs/ARTIFACTS.md`

When code moves, prefer the current checked-in pipeline/config surfaces over memorized script names. First verify:

1. which config schema currently owns infer, scoring, and eval;
2. which entrypoint actually consumes that schema;
3. where the canonical output artifacts are written;
4. whether the run is coord-token, raw-text, or another coordinate surface.

Commands:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py --config <infer.yaml>
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py --config <postop.yaml>
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py --config <eval.yaml>
PYTHONPATH=. conda run -n ms python scripts/evaluate_oracle_k.py --config <oracle.yaml>
```

Wrap with `rtk` when filtered output is acceptable.

## Default Decode Assumptions

Unless the user explicitly asks otherwise, use:

- `temperature = 0.0`
- `repeat_penalty = 1.10`

Treat these as the default reproducibility settings for ordinary CoordExp infer/eval prep. Override them only for intentional decoding ablations, legacy reproduction, or when a checked-in config already pins different values.

## Coordinate-Surface Rules

- Coord-token `xyxy`: run confidence post-op.
- Raw-text `xyxy` norm1000: set `infer.mode: text`, `infer.pred_coord_mode: norm1000`; confidence post-op must use numeric-text alignment, not coord-token geometry.
- `cxcy_logw_logh` or `cxcywh`: do not run confidence post-op; use deterministic constant-score compatibility only for checkpoints trained on that serialization.

## Proxy Bundle

For COCO + LVIS-proxy runs:

1. infer once;
2. score once;
3. evaluate the same scored artifact under:
   - `coco_real`: benchmark-aligned headline;
   - `coco_real_strict`: COCO plus strict same-extent proxies;
   - `coco_real_strict_plausible`: broad analysis view, not standard COCO.

Do not compare proxy-expanded views against standard COCO baselines without the label.

## Reusable Helper

```bash
HELPER=.codex/skills/coordexp-infer-eval-workflow/scripts/coordexp_infer_eval.py
python "$HELPER" prepare-recursive --repo-root <root> --checkpoint <ckpt> --run-tag <tag> --gpus <ids> --master-port <port>
python "$HELPER" summarize <run_dir> --format markdown
```

The helper defaults to `temperature=0.0` and `repeat_penalty=1.10`. Pass `--rp` only when intentionally overriding the default repetition penalty.

Use `--dry-run` before writing and `--force` only when intentionally reusing an output directory.

## Verification

Before launch, check intended JSONL, image roots, checkpoint/adapter, prompt/order settings, coordinate surface, scope label, decoding knobs, entrypoint ownership, and GPU launch shape.

After infer:

- `summary.json`
- `gt_vs_pred.jsonl`
- `resolved_config.json`
- `resolved_config.path` next to downstream artifacts when needed

After scoring:

- `confidence_postop_summary.json`
- `pred_confidence.jsonl` for confidence-scored paths
- `gt_vs_pred_scored.jsonl`

After eval:

- `metrics.json`, `per_image.json`
- guarded companions when `duplicate_control.enabled`
- proxy bundle summary when used

For sharded runs, trust merged top-level summaries/manifests over shard logs.

## Failure Modes

- `metrics: both` on COCO proxy artifacts can route into LVIS-federated assumptions; inspect `src/eval/detection.py`.
- Missing visualization images usually means `provenance.source_jsonl_dir` or root image provenance is wrong.
- Proxy-expanded GT count surprises should be checked against `metadata.coordexp_proxy_supervision.object_supervision`.
- A scored raw-text collapse usually means the wrong confidence alignment path ran.
- If a familiar script disappeared, do not force the old command shape; trace the current config owner and artifact writer first.
- Do not re-run inference when only eval views changed.
