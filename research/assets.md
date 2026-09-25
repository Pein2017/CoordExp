---
doc_id: research.assets
layer: research
doc_type: assets-reference
status: canonical
domain: research
summary: Curated locators for reusable research checkpoints, datasets and fixed panels.
updated: 2026-09-22
---

# Reusable research assets

Start here when a task needs an established checkpoint, dataset or fixed panel.
This is a small retrieval index, not a run registry or permission to launch.
Scientific claims and accepted results stay with their linked research units;
exact payload identity stays with the original manifests and run receipts.

Use the stable IDs below in discussion and task profiles, together with the
resolved manifest/path in a new execution receipt. Avoid ambiguous names such
as “step-2444”, “fresh128”, “latest checkpoint” or “the COCO dataset”.

## Reuse and maintenance

- **Read the index first.** Reuse the named manifest/config/loader. Check only
  the selected asset's existence and relevant bindings; search more broadly only
  when it is missing, incompatible, changed or absent here.
- **Promote selectively.** The research lead adds an asset when it becomes a
  reusable baseline, a stable diagnostic/evaluation panel, or a dataset source
  worth carrying across studies. Workers propose additions at candidate handoff;
  every temporary run/checkpoint does not earn an entry.
- **Keep identity and exposure explicit.** Record exact paths, necessary model
  components, geometry/order, intended role, known confounds or exposure, source
  manifest/result, status and last verification date. New checkpoints or changed
  data contents get a new ID/version; never silently retarget an existing ID.
- **Verify proportionately.** Last checked is not perpetual validity. Cheap
  path/config/manifest checks replace repeated discovery; actual launch still
  verifies required payload hashes and the relevant loader/data contracts.
  Do not rehash all stored checkpoints just to consult this page.
- **Retire visibly.** Mark unavailable/superseded assets and link their replacement;
  preserve provenance. A moved path with identical content is a documented
  location update, not a new scientific result.

Initial entries were checked on **2026-09-22** for path availability and selected
config/manifest metadata. The processed train/val file hashes matched their
manifest. Human13 source/derived hashes, the 392-object multiset and all 13 bound
image hashes were independently checked. This indexing pass did not reload
models or certify every tensor/media payload. Dataset counts below come from
named manifests.

For the user's manually refined references, go directly to
[Human13](#human13-392-geo-sorted-xy-v1) and
[the five burst images](#refined5-178-v1). Their sources, frozen copies and
runtime views are distinct assets; none is replaced by a session recollection.

## Checkpoints and composition

### `qwen3vl2b-coord-natural-adjacent-base`

- Status/role: reusable base for the two mature packages below; not a complete
  trained detector by itself.
- Root: [/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent](/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent).
- Identity/config: root `config.json`, tokenizer/processor and weight index;
  [maintained loading owner](../src/qwen/runtime_loading.py).
- Coordinate-token initialization is already structured. Do not describe this
  base as a random, unordered coordinate vocabulary; see the bounded
  [coordinate-input study](experiments/2026-09-19-coordinate-input-continuity/results.md).

### `mature-tied-xy-step2444`

- Status/role: reusable mature tied baseline; selected for the September 22 pilot.
- Root: [/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444](/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444).
- Required composition: the named base **plus `adapter/` DoRA plus
  `special_token_embeddings/`**. The selected input/output delta is shared.
  Loading only the adapter does not reproduce this checkpoint.
- Resolved recipe: `configs.tied` in the [mature comparison panel](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json).
  Maintained research loader: [mature_tied_untied.py](../probes/model_profiles/mature_tied_untied.py),
  `config_for('tied')` / `load_model('tied', device)` (FP32/SDPA, frozen by default).
- Historical SFT uses `geo_sorted_xy`; exact prompt/processor/decode belongs to
  the named config. The 1004 selected IDs include 1000 coordinates and four
  special tokens; they are not all coordinate bins.

### `mature-untied-axis001-xy-step2444`

- Status/role: reusable comparison package, **not an untie-only causal control**.
- Root: [/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444](/data/CoordExp/outputs/infra_base/train/qwen3-vl-2b-geo-sorted-xy-untied-axis001-ebs24-4epoch/checkpoints/step-2444).
- Required composition: same named base, this root's `adapter/` and
  `special_token_embeddings/`, with independent trained input/output deltas.
  Root `inference_payload_manifest.json` binds the inference payload.
- Resolved recipe/loader: `configs.untied` in the same mature panel and
  `config_for('untied')` / `load_model('untied', device)` in the same loader.
- Axis loss, optimization and training history differ from the tied package.
  [Accepted comparison](experiments/2026-09-18-untied-highconfidence18-natural/results.md)
  owns the evidence and confound boundary; this entry claims no universal winner.

## Dataset and fixed-panel locators

### `coco12k-geo-sorted-xy-v1`

- Status/role: reusable processed source pool, not a frozen experiment split.
- Root: [/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted](/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000_xy_sorted).
  Files: `train.coord.jsonl`, `val.coord.jsonl`, `pipeline_manifest.json`.
- Manifest counts: 117,266 train / 4,952 val images; 849,951 / 36,491 positive
  objects. The manifest owns the full-file hashes and upstream source identity.
- Geometry/order: `xyxy`, `<|coord_0|>` ... `<|coord_999|>`, norm1000;
  `geo_sorted_xy` sorts lexicographically by left-top `(x1,y1)`, with source-order
  tie breaking. Use [data contracts](../docs/data/CONTRACT.md) for consumer conversion.
- Record image paths are relative to the JSONL parent, not the current worktree.
  Processed images: [/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images).
  Original images: [/data/CoordExp/public_data/coco/raw/images](/data/CoordExp/public_data/coco/raw/images).
  Resolve split/file identity and dimensions before using originals for detail
  comparisons; processed-image dimensions are not original-image dimensions.
- Use `(metadata.split, image_id, file_name)` and source bindings when making
  splits. COCO annotations supply known positives; annotation absence is not a
  verified physical negative or an exhaustive EOS/count target.
- Worktree caveat: generic `configs/_shared/datasets/` paths are not this exact
  `_xy_sorted` asset, and `research-probes/public_data/coco/...` was absent at
  this check. Use the verified absolute root above; do not silently substitute a
  similarly named preset or create a data copy to make a relative path work.

### `human13-392-geo-sorted-xy-v1`

- Status/role: the user's carefully human-refined 13-image reference, repeatedly
  used for finite fitting and mechanism studies. Preserve its exact version;
  it is not held-out, and refinement does not prove exhaustive full-scene labels.
- **Default frozen coordinate input:**
  [human-refined-13.geo_sorted_xy.coord.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl).
  Its adjacent
  [coord.receipt.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.receipt.json)
  binds source/derived hashes, image hashes and all 392 source-to-derived objects.
- **Before sorting:**
  [human-refined-13.coord.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/evaluation-inputs/human-refined-13.coord.jsonl)
  and its [admission receipt](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/receipt.json).
  This inserted image2299 (38 persons + 8 ties) into the byte-preserved
  [legacy12 reference](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl)
  (346 objects). The receipt binds the exact then-current source row in
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`.
  Use the frozen snapshot for reproduction, not a re-extraction from a mutable
  dataset. This is recovered lineage, not a claim that an original annotation
  editor/export is still available.
- **Image IDs / positive objects:**
  `1584:19, 2299:46, 2685:29, 4134:37, 5001:23, 6040:15, 7511:44,
  10707:19, 13348:15, 13923:21, 14038:47, 14439:27, 16228:50`.
  All are `val2017`; total 13 images / 392 objects. This image ID list, not the
  label “Human13”, defines membership.
- Preserve `coco_ann_id`, including negative human-added IDs, with image identity,
  descriptions and box extent. Row order changes do not create new owners; IDs
  from different images are not interchangeable. Do not silently merge a part,
  group, overlapping instance or unresolved region into another owner.
- Images are bound in `images_manifest` inside the sorting receipt. Relative
  paths resolve from the JSONL parent to the processed `.../images/val2017/`
  root above. For original detail use corresponding raw `val2017` images with a
  fresh geometry mapping; don't relabel a resized image as original resolution.
- For model-specific trajectories, finite row banks and owner spellings, follow
  the mature panel below; those derived views do not replace the source labels.

### `refined5-178-v1`

- Status/role: the user's five manually refined duplication-burst study images;
  5 `train2017` images / 178 positive annotations. Development reference, not a
  held-out set or a guarantee of exhaustive scene annotation.
- **Live annotation workspace:**
  [label_studio_refinement_4](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4).
  Its [README](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/README.md)
  identifies **Label Studio Project 3**, despite the directory's `_4` suffix.
  [working.norm.jsonl](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/working.norm.jsonl)
  is the current export of the user's additions, deletions, geometry and class
  edits. It may change after a successful editor Update. Reading this index
  neither starts the editor nor authorizes exporting/changing annotations.
- **Bindings:**
  [project_manifest.json](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/project_manifest.json)
  binds the five task IDs `122219–122223`, source rows and original source hash;
  [last_export.json](/data/CoordExp/public_data/coco/rescale_32_1024_bbox/label_studio_refinement_4/last_export.json)
  owns the latest recorded export timestamp, counts and output hash. At this
  check the recorded export was `2026-09-18T06:26:40.454757+00:00`.
- **Default frozen research reference:**
  [annotation-snapshot-v1/working.norm.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-history-rereading-mechanism/human-evaluation/annotation-snapshot-v1/working.norm.jsonl).
  Independently checked byte-identical to the live export on 2026-09-22. This
  equality is dated: a future user edit creates a new annotation version, not
  permission to rewrite past experiment scores or this frozen snapshot.
- **Do not use `source.norm.jsonl` as the refined labels.** It preserves the
  pre-edit five-row baseline (83 positives), with the changes below:

| Image ID | Pre-edit objects | Refined positives |
| --- | ---: | ---: |
| 7116 | 6 | 5 |
| 309264 | 10 | 14 |
| 351017 | 25 | 49 |
| 417044 | 15 | 63 |
| 477415 | 27 | 47 |
| Total | 83 | 178 |

- The snapshot is numeric normalized `xyxy`; the mature18 runtime below is its
  coordinate-token/model input derivative. Preserve image and `coco_ann_id`
  identity, including deterministic negative IDs for new boxes. Exported boxes
  do not supply verified negatives or an explicit ignore mask.
- Corresponding original images exist at
  `/data/CoordExp/public_data/coco/raw/images/train2017/<12-digit-ID>.jpg`.
  Native sizes in the table's order are `640x427`, `640x427`, `640x425`,
  `500x375`, `640x426`; processed counterparts live under the processed image
  root above. Verify the coordinate mapping when switching resolution.
- Selected-event overlays and judgments are in
  [mature comparison physical-review](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/physical-review)
  with [final-review.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/physical-review/final-review.json).
  This is a bounded model-output review, not a replacement annotation set or
  complete visual review of every image; unresolved events remain unresolved.
- The later six-image/eleven-trajectory recurrence pool is different: only
  image417044 overlaps these five. Keep the two populations separate.

### `human13-refined5-sentinel128-v1`

- Status/role: repeatedly used development/calibration/sentinel panels; **not
  fresh held-out evidence**. No permission to turn prior evaluation into training.
- Authoritative composition/config/source bindings:
  [mature panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/panel.json).
- It records Human13 (13 images / 392 positives), refined5 (5 / 178), and
  sentinel128 (128 / 919). There are 145 unique images, not 146: bird309264 has
  distinct refined/sentinel annotation memberships. Preserve both identities.
- Refined18 input: [refined18.runtime.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-18-untied-highconfidence18-natural/refined18.runtime.jsonl).
  Sentinel input: [runtime-records.jsonl](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-17-readout-norm-fresh128/runtime-records.jsonl).
- The panel's `sources`, `memberships`, `teachers`, `refined_banks` and
  `sentinel_banks` preserve annotation/exposure distinctions. Read the
  [accepted result](experiments/2026-09-18-untied-highconfidence18-natural/results.md)
  before using its metrics or obligations.

### `census128-seed19-excl525-v1`

- Status/role: fixed September 19 natural-evaluation panel. Prospectively selected
  for that study; already exposed now, not fresh evaluation for a new study.
- Selection/identity: [eligible-manifest.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/eligible-manifest.json)
  and adjacent `exclusions.json`. The receipt retains source hashes, row/image
  identities, eligibility, sampling and prior-source manifests.
- Frozen selection: 128 images, seed19, from the processed source above after
  excluding the 525-image union of prior train256, dev128 and mature comparison
  images. Exclusions overlap; do not add their counts as disjoint groups.
- This is not an existing 128-train/32-validation calibration split. Its
  `selected_rows` and exclusion evidence can prevent accidental reuse; a new
  experiment still owns its own disjoint admission manifest.

### `mature-recurrence-6img11traj-v1`

- Status/role: fixed mechanism-development pool, not held-out or population data.
- Shared panel: [shared-panel.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census/shared-panel.json).
- Source trajectories and experiment-specific admission:
  [shared-admission.json](/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-21-spatial-progress-gate/selection/shared-admission.json).
- Six images / eleven model-image failure trajectories. A single-model study
  does not inherit eleven executable tied trajectories; interventions have
  their own eligibility/HOLD denominators. The old admission is provenance,
  not a grant to reuse every boundary or candidate in a different contrast.
- [Accepted recovery](experiments/2026-09-21-spatial-progress-recovery/lead-results.md)
  and [visual-binding result](experiments/2026-09-21-visual-instance-binding/lead-results.md)
  bound the conclusions available from this pool.

## Compact identity anchors

These small manifest hashes were read on 2026-09-22. They locate known versions;
their referenced payloads remain subject to the consuming experiment's checks.

| Manifest | SHA256 |
| --- | --- |
| `coco12k-geo-sorted-xy-v1/pipeline_manifest.json` | `f8162b91798fd5ac0e5612b5910b93adeb0cace1d9c12bca3fcd92e500bba378` |
| Mature comparison `panel.json` | `968f6f7e7752ad68deb697d69690704cb94c2cd3d3ebd15c254d9cbb709bf40a` |
| Census `eligible-manifest.json` | `61c742773ea2483693ed55b5ce39251a056953d043e1bfc04f82ab6baeec876b` |
| Spatial-progress `shared-admission.json` | `8863a4eb6d00ed9cb27cf8dd41e32429e7081e3bf363907b23bf4a7a0ba1d7d8` |
| Human13 before sorting | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| Human13 `geo_sorted_xy` | `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23` |
| Refined5 frozen snapshot (also live export at this check) | `1d8d7c6d63e982f2d5060fa96a7cbc825c0314246276b181aaefff9ee80fed85` |
