## 1. Inference engine scaffolding
- [x] 1.1 Add `src/infer/engine.py` with `InferenceEngine`, `GenerationConfig`, `CoordinateProcessor` placeholders (no code yet).
- [x] 1.2 Define unified output schema (per-object type/points/desc/score, errors list, width/height, image, mode, raw_output) in comments.

## 2. CLI definition
- [x] 2.1 Refactor/create inference CLI to require `--gt_jsonl`, `--model_checkpoint`, `--mode` (no auto-detect), `--out`, `--limit`, `--device`, and generation flags (`--temperature`, `--top_p`, `--max_new_tokens`, `--repetition_penalty`, `--seed`).
- [x] 2.2 Document argument parsing and validation expectations; enforce missing/invalid mode as error.

## 3. Mode-specific processing contract
- [x] 3.1 Specify coord-mode behavior: enforce 0-999, denorm GT+pred to pixels, skip-with-error on violations or GT/pred size missing.
- [x] 3.2 Specify text-mode behavior: keep GT pixels, denorm preds when norm/tokens, accept pixel preds, skip-with-error on violations.
- [x] 3.3 Define mode/GT mismatch handling (e.g., pixel GT with coord mode) as per-sample error `mode_gt_mismatch`.

## 4. Geometry handling & evaluation alignment
- [x] 4.1 Capture requirement to preserve `poly`/`bbox_2d` geometries in outputs; polygons evaluated via COCO segmentation/mask IoU (single ring, clamped, non-degenerate).
- [x] 4.2 Note `line` tolerated structurally but excluded from metrics at this stage.
- [x] 4.3 Plan evaluator update to consume new schema, feed polygons as COCO `segmentation`, and skip lines in metrics while keeping them in reports.

## 5. Error handling, counters, and determinism
- [x] 5.1 Define per-sample `errors` list semantics and skip policy; aggregate counters and emit summary JSON (`pred.summary.json`) with counters, mode, totals, and distinct error codes.
- [x] 5.2 Define seeding expectations (torch + cuda + `torch.Generator` passed to `model.generate`) for reproducible runs.

## 6. Consumer and docs updates
- [x] 6.1 Update `vis_tools/vis_coordexp.py` to consume the new schema; remove reliance on legacy fields.
- [x] 6.2 Update `src/eval/detection.py` to consume the new schema, feed polygons as COCO `segmentation`, and ignore lines for metrics.
- [x] 6.3 Refresh `docs/centralized_infer/*` and evaluator docs to the unified schema and flag set.

## 7. Validation
- [x] 7.1 Run `openspec validate add-centralized-inference-engine --strict` after edits and fix any issues.
