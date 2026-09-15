# Internal Co-DETR context / free-category profile

`run(candidates_path, output_dir)` is an internal batch callable, not a CLI or
service. It uses the existing frozen artifact scripts rather than another
inference framework. Call one invocation at a time, with GPU0 and GPU1 available.
Models run sequentially and exit after the batch.

Input is JSONL with `case_id`, `image_id`, `image_path`, pixel `bbox` in xyxy
order, integer `width`/`height`, and canonical COCO80 `category`.
Optional `image_sha256` is checked. At most100 candidates per invocation;
identifiers must be unique, filename-safe and collision-free after colon
replacement. Images, dimensions and bounds are checked before launching.

The callable owns only a newly created output directory. It retains stage
logs, raw model outputs, exact exit codes, input/model/code fingerprints,
end-to-end elapsed time, and the final `decisions.jsonl`/`summary.json` paths.
Technical failures raise; unresolved, invalid model JSON and capped model
answers remain explicit unknowns. Each stage has a30-minute timeout and
bounded termination grace. Existing outputs are never overwritten.

Runtime: existing Conda `mmdet` for Co-DETR on GPU0, then `ms` for Qwen3-VL8B
on GPU1. The model weights, processor assets, helpers, point prompt and rule
are fingerprinted. The pinned scripts/artifacts under
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-10-autonomous-unmatched-evaluator`
are required dependencies; keep them with this internal profile.

## Evidence boundary

**Screening only.** On a frozen64-image proxy-development holdout,14 candidates
were supported:11 clearly usable,1 box error and2 reference-uncertain after
explicit visual-reference adjudication. Original references/results are kept.
No GT, reference labels or visual-review tool is read/called during inference.
Unknown is not a negative, acceptance is not certified GT, and lower-IoU
cross-prediction duplication is not evaluated. Strict IoU>.95 repeat handling
remains separate. Do not automatically rewrite labels or create hard rewards.

See the [research result](../../../docs/history/research-records/2026-09-15/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-10-autonomous-unmatched-evaluator/results.md)
for original and adjudicated counts, uncertainty, latency and exact evidence.
