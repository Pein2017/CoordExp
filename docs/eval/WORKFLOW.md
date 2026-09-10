---
doc_id: docs.eval.workflow
layer: docs
doc_type: runbook
status: canonical
domain: eval
summary: Current CoordExp-Swift inference, selected-token scoring, evaluation, and visualization.
tags: [eval, infer, runbook]
updated: 2026-09-09
---

# Evaluation Workflow

Use this route for current Swift artifacts. Start from the named config/run;
use [CONTRACT.md](CONTRACT.md) for artifact compatibility and
[INTERPRETATION.md](INTERPRETATION.md) for metric or matching meaning.

## Run inference and scoring

```bash
python -m src.infer --config configs/coordexp_swift/infer/<config>.yaml
```

The config selects the data, model composition, output directory and backend.
HF loads the base, adapters and selected-token embedding delta dynamically;
vLLM uses the immutable execution-model composition. Backend/source ownership
is in the [implementation map](../IMPLEMENTATION_MAP.md), with exact support
in the [backend/trace contract](../../openspec/specs/coordexp-swift-infer-backend-trace/spec.md).
A backend or raw-score channel requires its own current support evidence.

```text
input JSONL + model composition + decode config
  -> src.infer -> current inference artifact writer
  -> gt_vs_pred.jsonl + pred_token_trace.jsonl
  -> gt_vs_pred_scored.jsonl + scored provenance
  -> direct COCO bbox consumer -> metrics + evaluation receipt
```

Keep the complete run directory: the evaluator replays selected-token scores
from `pred_token_trace.jsonl` and checks raw/scored provenance. A scored JSONL
copied alone is insufficient. The writer also records `summary.json`,
`run_manifest.json`, `parse_diagnostics.jsonl`, and `image_plan.jsonl`.
See [ARTIFACTS.md](../ARTIFACTS.md) when another artifact's owner is needed.

## Run detection evaluation

```bash
python scripts/evaluate_detection.py \
  --artifact-dir outputs/coordexp-swift/<run>/inference \
  --out-dir outputs/coordexp-swift/<run>/inference/eval
```

Use the actual inference directory selected by the config. Alternatively,
`--pred-jsonl <run-dir>/gt_vs_pred_scored.jsonl` selects its parent directory;
it still requires the sibling evidence. Do not pass both input options.
`--metrics-name` optionally changes the metrics filename. There is no current
`--config` evaluator option.

The consumer writes `metrics.json`, `evaluation_receipt.json`, `coco_gt.json`,
and `coco_predictions.json`. It produces aggregate COCO bbox metrics;
LVIS reduction, duplicate guards, F1 matching and overlays are separate tasks.
Its COCO category IDs are evaluator-local and its prediction sidecar is not an
official test-server submission. For geometry and namespace interpretation,
use [INTERPRETATION.md](INTERPRETATION.md#geometry-labels-and-coco-namespaces).

## Review predictions

The current shared renderer consumes the Swift run directly:

```bash
python scripts/visualize_detection.py gt-vs-pred \
  --run-dir outputs/coordexp-swift/<run>/inference \
  --out-dir outputs/coordexp-swift/<run>/review --limit 2
```

Use `compare` for two run directories; inspect its `--help` for row selection
and labels. Rendering is separate from aggregate evaluation. Its owner is
[`src/vis/`](../../src/vis/), not an implied evaluator overlay option.

## Validation scope

Tiny debug/smoke runs verify implementation only. Current benchmark eligibility
requires `debug.smoke: false` and at least 200 rows; passing those conditions
does not establish a new scientific claim. The accepted fixed val200 run is
sufficient for the existing Swift V1 local validation scope. A full dataset or
official test-dev run is optional and requires its own requested scope.

Before comparing results, bind config, checkpoint/composition, data subset,
score channel and evaluator settings. Check the evaluation receipt and
[artifact contract](CONTRACT.md), then interpret the declared metric through
[INTERPRETATION.md](INTERPRETATION.md).

## Historical workflows

Old confidence post-op, raw-text/non-canonical boxes, proxy/LVIS, Oracle-K and
visualization-sidecar conventions are preserved in the
[legacy evaluation reference](../history/evaluation/2026-09-09-legacy-eval-reference.md).
The [COCO test-dev runbook](COCO_TEST_SUBMISSION.md) is historical; it does not
establish a current Swift submission route. Historical commands must not be
mixed into this artifact pipeline.
