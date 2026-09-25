# Model and data identities

This is the input-identity owner, not a run authorization or checkpoint registry.
Locators below are recorded research inputs; external bytes were not modified by
cleanup. Verify their manifests before a new qualification. Original detailed
lineage: Git `108dede0154abfd90a54d18234d9e0bac780a3ba:research/assets.md`.

## Mature model compositions

Base: `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.
The coordinate vocabulary is structured, not a random unordered initialization.
The current assembly owners are `src.qwen.runtime_loading`, `src.adapters.dora`
and `src.qwen.special_token_embeddings` / `src.qwen.untied_embeddings`.

Tied x->y Source step2444:
`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`.
Compose base + `adapter/` + `special_token_embeddings/`; loading only the DoRA
adapter is not equivalent. The1004 selected IDs comprise1000 coordinates and
four wrappers. Input/output deltas are shared.

Mature untied+axis001 step2444:
`/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444`.
Compose the same base with this root's adapter and independent input/output deltas.
`inference_payload_manifest.json` binds its inference payload. It is not an
untie-only causal control: objective and training history differ from tied Source.
Original training source: `365cd55d169e5292b60bb75b672c0a65813df5b0`.

Exact compared recipes live in `configs.tied` / `configs.untied` of
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json`.
The retired convenience loader is recoverable from Git, not a current API.
New execution must explicitly select the current model/input/policy contract.

## COCO processed source pool

`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted/`
contains train/val.coord.jsonl and pipeline_manifest.json:117266/4952 images,
849951/36491 annotated positives. These are source-pool counts, not a new split.
Preserve xyxy norm1000 bins and lexicographic(x1,y1) source-order tie breaking.
Image paths resolve from the JSONL's parent, never the shell cwd. Processed images
live in `/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images`; originals in
`/data/CoordExp/public_data/coco/raw/images`. Recheck geometry on resolution changes.

COCO annotation absence is not a physical negative or exhaustive stopping target.
Historical exposure sets stay in their catalogued source records. No old exclusion
inventory is silently reused as a new experiment's selection policy.

## Frozen Human13, refined5 and sentinel identities

Human13 has392 reviewed positives across val2017 images:
1584:19,2299:46,2685:29,4134:37,5001:23,6040:15,7511:44,10707:19,
13348:15,13923:21,14038:47,14439:27,16228:50. It is development data, not held-out.
The frozen sorted input and matching receipt are
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl`
and adjacent `human-refined-13.geo_sorted_xy.coord.receipt.json`.
Preserve owner/image IDs, including negative human-added annotation IDs.

Refined5 has178 positives in train2017 images7116:5,309264:14,351017:49,417044:63,
477415:47. The original pre-edit source contains83 positives, not the same labels.
Frozen research reference:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl`.
The live Label Studio workspace is
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4`
(Project3 despite its directory suffix); future exports create new versions,
not permission to rewrite old results. Refinement does not establish exhaustive
scene labels or verified negatives.

The mature panel above binds Human13(13/392), refined5(5/178), sentinel128(128/919).
There are145 unique images rather than146: bird309264 has different memberships.
The later six-image/eleven-trajectory recurrence pool is distinct; only417044
overlaps refined5. Exact selection/controls stay in the catalogued unit inputs.
See [physical evaluation](questions/physical-evaluation.md) before comparing them.
