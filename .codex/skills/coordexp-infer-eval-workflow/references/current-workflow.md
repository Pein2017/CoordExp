# Current inference and evaluation workflow

Use this reference for the maintained Swift inference, scoring, detection-evaluation, and row-review commands. The selected checkout's OpenSpec contracts and source code own exact behavior and artifact serialization.

## Choose the output owner

Read the selected checkout's `docs/OUTPUT_STORAGE_POLICY.md` before setting a destination. New runs belong to their producer's physical output root:

- Research Probes: `/data/CoordExp/.worktrees/research-probes/outputs`
- Infrastructure runs: `/data/CoordExp/.worktrees/coordexp-infras/outputs`

`/data/CoordExp/outputs` is a selected shared-asset store, not a generic launch destination. Existing frozen artifacts there may be inputs; that does not make it the default destination for new outputs. For another worktree, use the physical owner named by its current policy.

## Run inference

```bash
python -m src.infer --config <selected-checkout-inference-config.yaml>
```

Resolve config families in the selected checkout: coordexp-infras uses `configs/infer/`; root main uses its `configs/coordexp_infras/infer/` surface. The authored config selects the data, model composition, backend, decode settings, and output directory. Set that output directory under the selected `OUTPUT_ROOT/coordexp-swift/<run>/inference`; use the config family's supported fields rather than inventing command-line overrides.

## Evaluate the completed run

```bash
python -m scripts.evaluate_detection \
  --artifact-dir "$OUTPUT_ROOT/coordexp-swift/$RUN_ID/inference" \
  --out-dir "$OUTPUT_ROOT/coordexp-swift/$RUN_ID/inference/eval"
```

The evaluator can instead take `--pred-jsonl <run-dir>/gt_vs_pred_scored.jsonl`; it still requires sibling evidence. Do not pass both input options. The evaluator has no `--config` option.

Keep the full run directory. Current metric-bearing output includes the raw and scored rows, scored-provenance sidecar, token trace, run manifest, resolved config, summary, parse diagnostics, and image plan. A scored JSONL copied alone is insufficient. The evaluator writes `metrics.json`, `evaluation_receipt.json`, `coco_gt.json`, and `coco_predictions.json`; the sidecars are not an official COCO test-server submission.

## Review rows

```bash
python -m scripts.visualize_detection gt-vs-pred \
  --run-dir "$OUTPUT_ROOT/coordexp-swift/$RUN_ID/inference" \
  --out-dir "$OUTPUT_ROOT/coordexp-swift/$RUN_ID/review" --limit 2
```

Use the command's help for two-run comparison and row-selection options. Rendering is a separate review step and does not change aggregate metrics.

## Scope

Tiny smoke runs establish mechanics only. The current Swift V1 local benchmark gate is the fixed val200 run with `debug.smoke: false`, valid scored provenance, and bbox mAP/mRecall output. Passing that gate supports only its declared validation scope; a full validation dataset or official test-dev run needs a separate request and contract.

For artifact binding and eligibility, consult the selected checkout's evaluation contract and owning specs. For matcher, category, geometry, or physical-owner meaning, consult the selected checkout's interpretation/research owner. LVIS proxy analysis, Oracle-K, historical confidence post-processing, and official COCO submission are separate workflows; do not silently add them to this path.
