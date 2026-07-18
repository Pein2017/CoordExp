---
name: coordexp-infer-eval-workflow
description: Use when launching, repairing, validating, or summarizing current CoordExp-Swift inference, scoring, and detection evaluation through HF or vLLM.
---

# CoordExp-Swift Inference And Evaluation

Use YAML-first production paths. Do not invent stable CLI flags when config already captures the run.
Treat this skill as the stable workflow guide, not a promise that one exact script path will never move.

## Entry Points

- Config: `configs/coordexp_swift/infer/`
- Entrypoint: `python -m src.infer --config <config.yaml>`
- Pipeline: `src/inference/pipeline.py`
- Shared request/session contract: `src/inference/backend.py`
- Processor-only frontend and launch projection: `src/inference/runtime.py`
- Backends: `src/inference/hf_backend.py`, `src/inference/vllm_backend.py`
- Execution-model materialization: `src/inference/execution_model.py`
- Artifacts: `src/inference/artifacts.py`, `src/inference/merge.py`
- Evaluator: `scripts/evaluate_detection.py` and
  `src/eval/detection_consumer.py`

Older `scripts/run_infer.py`, `src/infer/`, confidence post-op, Oracle-K,
proxy-bundle, grammar-constrained, and Stage-2 rollout paths are historical for
this worktree unless the user explicitly names one.

When code moves, prefer the current checked-in pipeline/config surfaces over memorized script names. First verify:

1. which config schema currently owns infer, scoring, and eval;
2. which entrypoint actually consumes that schema;
3. where the canonical output artifacts are written;
4. whether the run is coord-token, raw-text, or another coordinate surface.

Commands:

```bash
CUDA_VISIBLE_DEVICES=<ids> conda run -n ms \
  python -m src.infer --config configs/coordexp_swift/infer/<config>.yaml

conda run -n ms python scripts/evaluate_detection.py \
  --artifact-dir <run-dir> \
  --out-dir <run-dir>/evaluation/detection
```

Codex shells initialize the `ms` conda environment by default; do not add `conda run -n ms` unless working outside that initialized environment.

Wrap with `rtk` when filtered output is acceptable.

## Backend And Decode Contract

- `backend.type: hf` dynamically loads the base model, optional DoRA adapter,
  and optional selected-token embedding delta. HF remains first-class.
- `backend.type: vllm` resolves the same composition into a content-addressed
  immutable execution model, then runs one offline vLLM engine per active
  rank-local worker.
- Materialized HF is a composition-fidelity oracle, not a replacement for
  dynamic HF.
- Use FP32 for strict HF/vLLM parity claims. BF16 vLLM is supported for
  throughput runs but must be labeled non-parity evidence.

Canonical scored inference requires `temperature: 0.0`, `top_p: 1.0`, and
`n: 1`. Treat repetition penalty as an explicit config choice, not a hidden
default. `generation.batch_size` is immutable per-device decode concurrency.

## Current Artifact Gate

Before citing a benchmark, require completed summary/manifest status, complete
raw/scored/provenance/trace/diagnostic/image-plan artifacts, exact row/image/GT
binding, and zero unexplained parser/drop/truncation/score failures. Non-smoke
runs must report `benchmark_eligible: true`; the evaluator must also report
`benchmark_metric: true`.

Selected-token scores always use policy likelihood after active generation
processors. Optional `raw_model_logprob` is diagnostic LM-head likelihood only
and must never replace `pred[*].score`.

GT boxes are norm1000 `xyxy`; parser-normalized prediction `bbox` values are
pixel `xyxy`. Prediction `coord_bins` is source evidence and must not be drawn
or evaluated as pixels.

For distributed runs, require complete rank coverage, strict merged order,
backend performance, worker exit, no orphan engine process, and GPU memory
return. Failed runs may publish terminal diagnostics but not benchmark-looking
top-level raw/scored artifacts.

## Historical Addenda

The remaining confidence, proxy, Oracle-K, and Stage-2 notes apply only when
the user explicitly requests those historical/mainline workflows.

## Coordinate-Surface Rules

- Coord-token `xyxy`: run confidence post-op.
- Raw-text `xyxy` norm1000: set `infer.mode: text`, `infer.pred_coord_mode: norm1000`; confidence post-op must use numeric-text alignment, not coord-token geometry.
- `cxcy_logw_logh` or `cxcywh`: do not run confidence post-op; use deterministic constant-score compatibility only for checkpoints trained on that serialization.

## Diagnostic Completion

When the user wants raw rollout behavior inspected, prefer completing the rollout and preserving invalid or non-metric-bearing rows with parser metadata over aborting on the first malformed output, unless the official eval contract requires strict failure. Label diagnostic artifacts as non-metric-bearing unless strict parser, source image identity, dimensions, coordinate surface, and metric-bearing provenance all pass.

Before launch, compare training-side names with infer/eval schema names. Do not pass training-only enum values into infer configs; if translation is needed, record the mapping in the generated config or run note.

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

## Stage-2 Eval Validity Gate

Before treating Stage-2 eval artifacts as metric-bearing, reject or repair runs where:

- a row lacks real source image identity or dimensions;
- strict parser status is replaced by diagnostic fallback or `metric_bearing=false`;
- multi-image inputs were collapsed instead of rejected;
- prompt-token / detection-format provenance or score fingerprints are missing;
- `resolved_config.path` cannot recover the authoritative pipeline config.

Useful debug surfaces:

- `monitor_dumps/eval_phase_trace` for the last completed eval phase;
- source-JSONL provenance and image-root metadata when archived artifacts need geometry recovery;
- `configs/stage2/rollout_correction/smoke/compact_full_vllm_train64_val32_6steps_coco80_evaltrace.yaml` for compact-full COCO-80 evaltrace smoke coverage.

## Failure Modes

- `metrics: both` on COCO proxy artifacts can route into LVIS-federated assumptions; inspect `src/eval/detection.py`.
- Missing visualization images usually means `provenance.source_jsonl_dir` or root image provenance is wrong.
- Proxy-expanded GT count surprises should be checked against `metadata.coordexp_proxy_supervision.object_supervision`.
- A scored raw-text collapse usually means the wrong confidence alignment path ran.
- If a familiar script disappeared, do not force the old command shape; trace the current config owner and artifact writer first.
- Do not re-run inference when only eval views changed.
