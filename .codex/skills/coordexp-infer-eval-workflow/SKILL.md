---
name: coordexp-infer-eval-workflow
description: Use for current CoordExp-Swift inference, selected-token scoring, raw/scored/provenance artifact checks, accepted val200 validation, and detection-evaluator results. Treat older MS-Swift confidence, Oracle-K, proxy, guarded, and Stage-2 paths as historical.
---

# CoordExp-Swift Infer And Eval

Start from `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, and
`docs/eval/README.md`. Read `docs/eval/CONTRACT.md`,
`docs/eval/WORKFLOW.md`, or `docs/ARTIFACTS.md` only when their exact contract is
needed. Use the stable `coordexp-swift-infer-*` or
`coordexp-swift-detection-evaluator` OpenSpec only for compatibility-sensitive
semantics.

## Current route

- Configs: `configs/coordexp_swift/infer/`
- Entrypoint: `src/infer.py`
- Runtime: `src/inference/pipeline.py`, `runtime.py`, `backend.py`
- Artifact owner: `src/inference/artifacts.py`
- Evaluator: `scripts/evaluate_detection.py` ->
  `src/eval/detection_consumer.py`

Do not route current work through `scripts/run_infer.py`, `src/infer/`, legacy
confidence post-op wrappers, proxy bundles, or Stage-2 configs. Use those only
for an explicitly requested historical reproduction after verifying that the
named checkout still supports them.

## Run

Use a checked-in YAML; do not reconstruct its model, adapter, embedding, prompt,
or decode settings as ad hoc flags.

```bash
conda run -n ms python -m src.infer \
  --config configs/coordexp_swift/infer/<config>.yaml
```

The accepted V1 readiness gate is the fixed val200 config recorded by
`docs/catalog.yaml`:

```text
configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_val200.yaml
```

Tiny smoke runs prove implementation only. A full validation-dataset run is
optional unless the user asks for it.

Evaluate the inference artifact directory:

```bash
conda run -n ms python scripts/evaluate_detection.py \
  --artifact-dir <run-dir> \
  --out-dir <run-dir>/eval
```

If the scored path is already known, the supported alias is:

```bash
conda run -n ms python scripts/evaluate_detection.py \
  --pred-jsonl <run-dir>/gt_vs_pred_scored.jsonl \
  --out-dir <run-dir>/eval
```

`--pred-jsonl` must name `gt_vs_pred_scored.jsonl`; the evaluator still loads
the sibling raw artifact and provenance sidecar.

## Artifact contract

Current inference writes:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`
- `gt_vs_pred_scored.jsonl.provenance.json`
- `pred_token_trace.jsonl`
- `parse_diagnostics.jsonl`
- `image_plan.jsonl`
- `summary.json`
- `run_manifest.json`
- resolved config snapshots under the run directory

Before calling a result metric-bearing, verify:

1. raw, scored, and provenance files come from the same artifact directory;
2. raw/scored hashes and ordered row IDs match the provenance binding;
3. row identity, image path/dimensions, and GT are unchanged by scoring;
4. each prediction has finite score plus selected-token source evidence bound to
   its `row_id`, `object_span_id`, and score-policy fingerprint;
5. GT norm1000 `xyxy` is converted once to pixels, while scored prediction
   boxes are already pixel `xyxy`.

The direct Swift evaluator writes only:

- `metrics.json`
- `evaluation_receipt.json`
- `coco_gt.json`
- `coco_predictions.json`

Its COCO sidecars use evaluator-local categories and are not test-server
submission files. V1 is aggregate-only: do not promise `per_image.json`, LVIS,
duplicate guards, overlays, proxy views, or confidence reconstruction.

## Repair and reporting

- Inspect the exact run config, manifest, summary, provenance, receipt, and a
  few raw/scored row pairs before theorizing.
- Preserve malformed diagnostic rows and parser evidence, but do not label
  non-metric-bearing rows as benchmark evidence.
- Re-run evaluation without inference when only evaluator code or output views
  changed and the scored contract still validates.
- Report config, checkpoint and payload identity, row scope, artifact root,
  metric file, representative parser failures, and evidence limitations.
- Keep legacy and current results visibly separated; never compare proxy-expanded
  or reconstructed-score artifacts to the Swift V1 headline as if equivalent.

The bundled `scripts/coordexp_infer_eval.py` remains a legacy recursive-run
compatibility helper because repository tests cover its old free-decode config
surface. It is not a current Swift launcher and should not be used for new runs.
