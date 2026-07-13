---
title: Readiness Amendment for Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement
description: Freezes the scientific and execution contract while recording the materialized-artifact and runtime gaps that still block metric-bearing execution.
type: investigation
role: readiness-amendment
authority: non_normative_research
unit_id: 2026-07-13-spatial-scope-history-disentanglement
status: focused_review_passed
contract_review_status: passed
execution_readiness: blocked
implementation_status: authorized_in_progress
architecture_promotion_status: not_promoted
updated: 2026-07-13
---

# Readiness Amendment for Masked Spatial Policy and Accepted-Row Prefix Policy Disentanglement

## Purpose and Current Gate

This amendment freezes the smallest scientifically useful five-arm experiment
described by [the research unit](unit.md). It incorporates the independent
[review synthesis](review.md) and resolves every design choice that may be
resolved without inspecting new experimental-arm outputs.

The contract passed focused scientific and executable-semantics review. On
2026-07-13 the user explicitly authorized the bounded implementation, the
minimal score-preserving custom batched sampler inside the existing Hugging
Face backend, and gated eight-Graphics-Processing-Unit execution up to the
frozen ceilings. Metric-bearing execution remains blocked because the current
runtime does not yet implement per-request sampled
generation, the spatial and cumulative-prompt input policies, or the declared
object-level merger. The pre-output audit ledger also has not been materialized.
Authorization does not waive any readiness, mechanics, evidence, or stop gate.

The narrow stable-seam implementation is governed by OpenSpec change
`add-request-scoped-inference-controls`. Research orchestration remains owned by
this unit and its research implementation surfaces.

## Terminology and Name Registry

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model family
  under investigation.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed set
  of reportable categories in this unit.
- **Validation-200 (`val200`)**: the first 200 rows, ordered by image identifier,
  of the frozen processed Common Objects in Context validation source. It is a
  convenience scope, not a random or representative validation sample.
- **Dense-Union-51 — Annotation-Derived Dense Union of 51 Images**: the exhaustive
  51-image subset selected from `val200` by the annotation-only Boolean rule
  frozen below. It is the primary manually audited scope.
- **Weight-Decomposed Low-Rank Adaptation (`DoRA`)**: the adapter method used by
  the frozen checkpoint.
- **Hugging Face Transformers backend (`HF backend`)**: the current local
  inference backend wrapping Transformers generation.
- **Red-Green-Blue color space (`RGB`)**: the three-channel source-canvas pixel
  representation.
- **32-bit floating-point arithmetic (`float32`)**: arithmetic precision used
  for mask means and any intervention calculation where 16-bit rounding could
  alter pixels or coordinates.
- **Secure Hash Algorithm 256-bit (`SHA-256`)**: the content digest used for
  immutable identity and seed derivation.
- **JavaScript Object Notation Lines (`JSONL`)**: one JavaScript Object Notation
  record per line.
- **JavaScript Object Notation (`JSON`)**: the structured single-document
  serialization used for manifests and seals.
- **Unicode Transformation Format 8-bit (`UTF-8`)**: the byte encoding used for
  seed-domain strings and hashed identifier lists.
- **K spatial calls (`K`)**: the number of canonical spatial cells; the primary
  four-by-four grid has `K=16`.
- **Raw width-height bounding-box format (`xywh`)**: horizontal origin,
  vertical origin, width, and height in the raw Common Objects in Context image
  coordinate frame.
- **Corner bounding-box format (`xyxy`)**: left, top, right, and bottom corners
  in the processed source-canvas coordinate frame.
- **Bounding-box Intersection over Union (`IoU`)**: intersection area divided by
  union area for two bounding boxes.
- **Non-Maximum Suppression (`NMS`)**: deterministic class-wise greedy selection
  of the highest-scored box followed by suppression of lower-ranked overlapping
  boxes. It is not a transitive connected-component collapse.
- **Prediction-Set Diversity**: one minus mean pairwise Jaccard similarity of
  official reference-object identifier sets detected by sampled calls, averaged
  across images whose call union is nonempty.
- **Brain Floating Point 16-bit (`bfloat16`)**: the model arithmetic format used
  by the frozen inference runtime.
- **Nucleus-sampling cumulative probability cutoff (`top_p`)**: the smallest
  descending-probability token set whose cumulative mass reaches the configured
  value before sampling.
- **Qwen end-of-message token (`qwen_im_end`)**: the executed model token that
  closes an assistant response and realizes the terminal no-more-objects
  decision for one call.
- **Compact Object Selected-Token Score, version 1
  (`compact-object-selected-token-score-v1`)**: exponentiated mean selected-token
  log probability over the four object-schema wrapper tokens and four
  coordinate tokens; description tokens are excluded.
- **Strict duplicate component**: a connected component of final valid
  predictions under identical normalized class and pairwise bounding-box
  Intersection over Union at least `0.85`.
- **Post-Merge Strict Duplicate Rate**: the number of final predictions beyond
  the highest-ranked representative in each strict duplicate component divided
  by all final valid predictions; an empty prediction set returns `not
  applicable`.
- **Minimum meaningful effect**: the smallest absolute paired Local Rescue Rate
  difference treated as scientifically consequential.
- **Terminal no-more-objects decision (`STOP`)**: the model-generated terminal
  action ending one object-enumeration call.
- **Common Objects in Context crowd flag (`iscrowd`)**: the official annotation
  flag giving crowd regions ignore semantics rather than individual-instance
  semantics.
- **Graphics Processing Unit (`GPU`)**: an accelerator used only after separate
  implementation and execution authorization.
- **Compute Unified Device Architecture (`CUDA`)**: the device-runtime interface
  used by sampled-runtime attestation and isolated production workers.

All arm, metric, hypothesis, invariant, ownership, and accepted-row-prefix names
retain the complete declarations in [the research unit](unit.md).

## Frozen Source and Model Identity

### Model-facing validation source

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl
SHA-256: 9b524a8c20f03758e2e3939703ff35a1e35a1108fb095ac6e7af5a1539c8cfc4
row count: 200
```

Its ordered image identifiers and object contents equal:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.first200.coord.jsonl
SHA-256: 9ce25e3d6d4349377c9fb18282408c542671a29dee658e3ca476fdbbf9e5b126
```

The complete processed validation source is:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
SHA-256: 18a3cad3b7ad847ecf39949fe751d963008dbb7707f796c742c3fa23ae3c8e8b
row count: 4,951
```

The experiment source canvas is the existing 1024-budget processed image under:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/
```

Therefore, the no-resize invariant means **no additional geometric resize
relative to this frozen processed source canvas**. It does not claim raw Common
Objects in Context pixel scale.

### Raw official annotation source

```text
/data/CoordExp/public_data/coco/raw/annotations/instances_val2017.json
SHA-256: e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f
```

This source owns original image dimensions, category identifiers, annotation
identifiers, raw `xywh` bounding boxes, and `iscrowd`. The model-facing `val200`
source contains every non-crowd annotation for its 200 images but omits the 16
crowd annotations. Historical evaluator artifacts that force `iscrowd=0` are
not crowd-aware official evidence for this unit.

### Primary checkpoint

```text
/data/CoordExp/.worktrees/CoordExp-swift/outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_llm_12000_accelerate8_ebs24_8epoch_warmup0p1/checkpoints/step-4887/checkpoint.json
SHA-256: 613e5d97f4a7a53d6325b5c1909813d6bb82e72622942b225df9724b5556a536
```

The legacy path token `gaussian_rps` means **Gaussian Soft-Target Coordinate
Cross-Entropy with Ordered Cumulative-Distribution Penalty**. It remains only
for provenance and is not a canonical abbreviation.

Frozen payload identities:

| Payload | SHA-256 or fingerprint |
|---|---|
| Adapter configuration | `a3eead68cd99225606fcf023201f101ad2ed6829c458cb0414fffe620260db54` |
| Adapter tensor | `a2bc1d4bd68edbc884e1a2a84a61fb05cd34d3c06e52b93f2e3d8e10e0208df9` |
| Adapter fingerprint | `ba25619faf51ef7d1eb732e514f250efd37bce3d7f54d1db9eb5e83d8cc2a6c6` |
| Special-token embedding tensor | `1b625b3cb2552e626052200cdd6ef5be73e00bf7feb660d901f9e4df6c253bff` |
| Special-token embedding fingerprint | `c8df057b9e3be1479c24e2272338ba627c19a663d19333a024e558488e64ebeb` |
| Base-model configuration | `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de` |
| Tokenizer | `ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8` |
| Image preprocessor configuration | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| Chat-template configuration | `6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4` |
| Generation configuration | `4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59` |

The base-model path is:

```text
/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

Before the first calibration model call, the readiness preflight must hash the
model index and every weight shard named by that index, sort records by relative
path, and seal the resulting manifest. A missing shard, an unindexed shard, a
path change, or a later digest change stops execution. The hashes are
authorized-preflight-derived values, not unspecified scientific choices.

The primary config lineage is:

```text
configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml
SHA-256: f3000588accbcf1d9ada3b2f3e0b3324d660b4810b75d8f5d050d9f184f9ca80
```

The new execution config must preserve its model, adapter, special-token,
prompt, object schema, and no-resize identities while replacing only the frozen
decode and research-policy factors in this amendment. The executed resolved
config receives a new digest and cannot reuse the historical deterministic-run
identity.

## Frozen Execution and Audit Cohorts

### Broad execution cohort

All five primary arms execute on all 200 ordered `val200` images. No historical
or new model score may rank, select, exclude, or reorder these images.

Broad official metrics are conditional on this fixed first-200 convenience
scope. They are not claims about the complete Common Objects in Context
validation distribution.

### Dense-Union-51 audited cohort

For each `val200` image, join raw annotations by image identifier and count only
non-crowd COCO-80 individual annotations. Select the image if any clause is true:

```text
noncrowd_annotated_object_count >= 12
OR annotated_person_count >= 8
OR annotated_food_tableware_count >= 7
```

The food and tableware set is exactly the following Common Objects in Context
category identifiers and names:

```text
44 bottle
46 wine glass
47 cup
48 fork
49 knife
50 spoon
51 bowl
52 banana
53 apple
54 sandwich
55 orange
56 broccoli
57 carrot
58 hot dog
59 pizza
60 donut
61 cake
```

The rule produces 51 images, 860 non-crowd official annotations, 337 annotated
person instances, 200 annotated food/tableware instances, and 16 source crowd
annotations distributed across 16 images.

Ordered image identifiers:

```text
139, 632, 885, 1000, 1584, 2157, 2299, 2685, 3845, 3934, 4134,
5001, 5586, 6040, 6471, 6771, 7281, 7511, 7574, 7816, 7818, 8277,
8629, 9378, 9400, 9590, 9891, 10707, 11197, 12120, 12576, 12639,
12670, 13348, 13659, 13923, 14038, 14439, 15254, 15335, 15517,
16228, 16451, 17627, 17714, 17959, 18380, 18491, 18575, 19109,
19432
```

The SHA-256 digest of the newline-delimited ordered decimal image identifiers,
including the final newline, is:

```text
e7c6b59950cdef93b59638c1899179ee80319daeda712bfc171bd329978675c8
```

The subset is exhaustive under the rule; it has no selection seed. Images carry
overlapping source-only tags for annotated-count density, people density,
food/tableware density, and source crowd presence. Reviewers may mark ambiguity
but may not remove an image.

### Sampling calibration cohort

Decode calibration uses 12 processed validation images outside `val200`; they
are permanently excluded from every metric-bearing cohort. Six satisfy the
Dense-Union-51 rule and six contain one through three non-crowd annotations with
no source crowd annotation.

```text
dense: 563648, 303713, 529148, 538236, 559099, 276434
sparse: 30494, 269314, 546829, 42889, 79408, 61471
```

The SHA-256 digest of ordered lines formatted as `<stratum>:<image_id>` with a
final newline is:

```text
fe11eb219531328cbd18189696925caf5b75c616f9ffd4eca2e5226911833d39
```

Calibration outputs form a one-time, pre-output, annotation-informed decode
calibration. They are not metric-bearing evidence and cannot contribute
objects, thresholds, images, or estimates to the primary result.

## Pre-Output Reference-Ledger Contract

The metric-bearing run remains blocked until the following immutable artifacts
exist under the declared readiness-artifact root:

```text
cohort-manifest.jsonl
dense-union-51-manifest.jsonl
official-individual-ledger.jsonl
official-crowd-ignore-ledger.jsonl
review-queue.jsonl
reviewer-one-labels.jsonl
reviewer-two-labels.jsonl
adjudication.jsonl
audit-augmented-ledger.jsonl
ledger-seal.json
```

The two reviewers receive only unmodified source-canvas images, the frozen
COCO-80 ontology, reviewer instructions, and immutable review identifiers. They
do not receive official boxes, historical or new predictions, arm names,
scores, crops, failure labels, or each other's labels. Each pass enumerates
visible objects before the adjudicator receives both passes, official
annotations, and raw crowd regions.

### Reviewer instruction contract

Every selected image remains in the ledger. Each reviewer independently labels
every visually separable COCO-80 instance. A candidate is individually
reportable when its category and instance identity are visually distinguishable
and the reviewer can draw a reproducible tight box around its visible extent.
There is no minimum size or visible-percentage threshold. Occlusion or image-edge
truncation does not prevent acceptance when the visible evidence still supports
category, instance identity, and a stable visible-extent box.

Reviewer state meanings are:

- `accepted`: one distinguishable COCO-80 instance with one category and a
  reproducible visible-extent box;
- `ambiguous`: object evidence exists, but category or instance identity is not
  uniquely resolvable; record every plausible COCO-80 category identifier;
- `partial`: a reportable category or instance is plausible, but occlusion,
  truncation, or indistinct boundary prevents a reproducible individual box;
- `crowd`: multiple instances of one category are visible but cannot be
  individuated reliably; draw one group region and record the category;
- `out-of-scope`: visible content does not belong to COCO-80 and therefore
  cannot enter a reportable-object denominator.

The reviewer packet identifier is
`dense-union-51-image-only-review-v1`. Reviewers select only canonical COCO-80
category names; no free-text category alias is accepted. Allowed reason codes
are `none`, `category_not_unique`, `instance_not_separable`,
`boundary_not_reproducible`, `occluded_but_boxable`,
`truncated_but_boxable`, `non_coco80`, `official_duplicate`,
`reviewer_disagreement`, and `adjudicator_override`. An
`adjudicator_override` record must also link the superseded labels and retain a
free-text rationale for audit, but the rationale never enters a metric.

For `accepted`, `ambiguous`, `partial`, and `crowd` records, draw the smallest
axis-aligned integer pixel-edge `xyxy` rectangle judged to contain the visible
evidence. Coordinates use the source canvas, satisfy
`0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`, and never extrapolate a
fully occluded extent. A null box is allowed only when the reviewer records
`partial` and the reason `boundary_not_reproducible`. Overlapping objects remain
separate when distinct visible evidence supports separate identities; otherwise
use `crowd`. Reviewer instructions, ontology, category aliases, reason codes,
and annotation-tool version are a single versioned packet whose content digest
is sealed before either pass.

### Artifact schemas and deterministic assembly

Every JavaScript Object Notation Lines artifact uses UTF-8, one object per line,
lexicographically sorted keys, compact separators, and a final newline. Records
are ordered by frozen image order and then immutable record identifier.

Required fields are:

| Artifact family | Required record fields |
|---|---|
| Cohort manifest | schema version, image identifier, frozen order, source-row index, image path and SHA-256, source and raw dimensions, source hashes, annotation-only counts, cohort membership, and density tags |
| Official individual ledger | ledger version, image identity, `coco-ann:<annotation-id>` object identifier, category identity, raw `xywh`, source-canvas float32 `xyxy`, provenance `official_annotation`, `iscrowd=false`, and state `accepted` |
| Official crowd-ignore ledger | the same provenance fields, `iscrowd=true`, state `crowd`, and the source-canvas crowd-ignore region |
| Review queue | review identifier, image identity and SHA-256, reviewer-packet digest, ontology digest, and reviewer role slot |
| Reviewer labels | review identifier, reviewer role identifier, reviewer-local object identifier, category or candidate categories, integer source-canvas `xyxy` or allowed null, reviewer state, and reason code |
| Adjudication | adjudication identifier, linked reviewer-label and official-object identifiers, final category, final source-canvas box, final state, provenance decision, and reason code |
| Audit-augmented ledger | ledger version, image identity, immutable object identifier, category, source-canvas box, provenance, linked source records, final state, and adjudication identifier |

Pair the two reviewers' same-category non-crowd records by the frozen
maximum-cardinality, then maximum-Intersection-over-Union matcher at threshold
`0.50`. Pairing only forms an adjudication group; it never accepts an object
automatically. The adjudicator resolves every paired record, unpaired record,
official individual, and crowd region under the same instruction packet.

Every non-crowd official annotation remains an `accepted` official individual
and retains its official source-canvas box. A reviewer proposal becomes an
`accepted` audit addition only when the adjudicator confirms that it is a
distinct visible COCO-80 instance not already owned by an official individual.
All other proposals end as `ambiguous`, `partial`, `crowd`, or `out-of-scope`.
For accepted additions, the adjudicator freezes one integer visible-extent box
and category. Stable addition ordinals are assigned only after adjudication by
sorting on category identifier, `y1`, `x1`, `y2`, `x2`, and linked reviewer
identifiers. Official object identifiers use `coco-ann:<annotation-id>`;
additions use `audit:<ledger-version>:<image-id>:<stable-ordinal>`.

Only `accepted` individual objects enter the primary audit-augmented rescue
denominator. The preflight must pass a synthetic two-reviewer fixture containing
agreement, one-sided addition, category disagreement, occlusion, crowd, and an
official duplicate; repeated assembly must produce byte-identical ledger
output.

Raw crowd `xywh` boxes are mapped from raw-image pixels to source-canvas pixels
with independent horizontal and vertical scale factors. The preflight must
record raw and processed dimensions, float32 scale factors, unrounded float32
corners, final clipped float32 `xyxy` corners, and a round-trip receipt. Research
matching retains float32 coordinates; integer rounding is visualization-only.

`ledger-seal.json` records every source and artifact SHA-256 digest, reviewer
role identifiers, reviewer-packet version and digest, creation time, and the future
metric-bearing run's earliest permitted start time. Corrections create a new
ledger version; they never edit a sealed ledger in place.

After execution, pooled unmatched candidates may receive a separate arm-blind
adjudication. That secondary ledger cannot rewrite the sealed primary
denominator or headline Audit-Augmented Local Rescue Rate.

## Frozen Spatial Input Contract

### Common source canvas and visual quantum

The base vision patch size is 16 pixels and the merge size is two, giving a
32-pixel merged-token quantum. The source-canvas images in `val200` are already
divisible by 32 in both dimensions. Therefore:

- no primary source image is padded or resized;
- every full-canvas arm receives byte-identical source-canvas dimensions;
- native tiles use boundaries aligned to the 32-pixel merged-token grid and
  therefore also need no padding;
- any runtime observation contradicting these statements stops the run.

This narrower contract removes padding as a primary-arm confound. Padding
support is not required or authorized for this unit.

### Four-by-four primary partition

Partition the merged-token grid, not raw pixels. For an extent of `M` merged
tokens and cell index `i` in zero through three, the half-open core interval is:

```text
[floor(i * M / 4), floor((i + 1) * M / 4))
```

Convert token boundaries to pixels by multiplication by 32. A center exactly on
an internal boundary belongs to the higher-index cell. Cell identity is
row-major: `cell = 4 * row_index + column_index`, with top-to-bottom rows and
left-to-right columns.

Extend each of the core's left, right, top, and bottom sides by 25 percent of the
corresponding core extent. Round each per-side extension outward in merged-token
units with `ceil(0.25 * core_token_extent)` and require at least one merged-token
cell where the source boundary permits it. Clip the resulting core-plus-halo
rectangle to the source canvas. Halo pixels provide context but never own
predictions.

Reference and prediction ownership uses the global source-canvas bounding-box
center and the same half-open core intervals. A box with a non-finite center, an
empty area, or a center outside the source canvas is invalid and unowned.

### Native tile and masked canvas

`TILE_RESET` crops the core-plus-halo rectangle without resize. For a tile
extent `E` and coordinate bin `c` in zero through 999, decode each local
coordinate with the canonical rule `round(c * E / 1000)`, where `round` uses
nearest-integer ties-to-even semantics. Add the integer tile origin to the
decoded horizontal or vertical coordinate, then clip to the source canvas.
Retain the original coordinate bins, local integer box, tile origin, unclipped
global integer box, and clipped global integer box. `MASK_RESET`,
`MASK_CUMULATIVE`, `FULL_SINGLE`, and `FULL_BAG_K` apply the identical rule
directly with the source-canvas width and height. This preserves the current
canonical parser semantics; no alternate unrounded prediction box is invented.

`MASK_RESET` and `MASK_CUMULATIVE` retain the complete source-canvas dimensions.
Compute the per-image channel mean across every source-canvas RGB unsigned
8-bit pixel in float32, round to nearest with ties to even, clamp to `[0,255]`,
and cast to unsigned 8-bit integers. Fill every pixel outside the current
core-plus-halo with that value. No outline, alpha cue, blur, feathering, or
alternative fill is allowed.

### Secondary grids

The two-by-two and eight-by-eight grids are not part of the first metric-bearing
run. Running either requires a new pre-output budget amendment. An eight-by-eight
amendment is eligible only if every cell has at least four core merged tokens
and nine core-plus-halo merged tokens, and the four-by-four mask-harm guardrail
has passed. A two-by-two amendment is eligible when four-by-four mask harm makes
the spatial-policy result uninterpretable. Neither grid may replace the primary
four-by-four result.

## Frozen Prompt and Accepted-Row State Contract

The system instruction, user instruction, assistant row grammar, category list,
and chat template are byte-for-byte inherited from the frozen primary config.

Every call is reconstructed from scratch and contains exactly one image: the
current full source canvas, current masked canvas, or current native tile. No
previous image, image placeholder, user turn, or visual token is retained.

For `MASK_CUMULATIVE`, construct the ordinary chat-template generation prompt
for the current image and unchanged user instruction. Before generation, append
the concatenated accepted global-coordinate rows directly inside the still-open
assistant turn. The model continues after the final `<|box_end|>` token. Add no
separator, natural-language continuation instruction, terminal no-more-objects
token, end-of-sequence token, end-of-message token, or conversation terminator.

Rows enter this prefix only when they parse under the frozen compact grammar,
normalize to one COCO-80 class, have finite positive-area source-canvas
coordinates, and are owned by the current core. Preserve call order and
generated order. Do not merge or suppress rows before prompt admission. Invalid,
out-of-ontology, non-owning, and terminal content remains in raw artifacts but
does not enter active prefix state.

The next prompt must fit without truncation. Any left truncation, right
truncation, row deletion, summary, or silent reserialization stops the run.
Golden synthetic fixtures must attest the exact prompt bytes, token identifiers,
single-image-placeholder count, and exclusion of prior terminal tokens.

`TILE_CUMULATIVE_EXPLORATORY` remains disabled because tile-local generated
coordinates do not share a native frame with previous tile rows.

## Frozen Decode and Seed Contract

### Primary generation factors

```text
backend: Hugging Face Transformers
model arithmetic dtype: bfloat16
do_sample: true
top_p: 0.95
top_k: 0 (disabled; no inherited top-k token-count cutoff)
repetition_penalty: 1.0
max_new_tokens: 512
stop policy: qwen_im_end
return_dict_in_generate: true
output_scores: true
primary batch size: 4
```

Here `bfloat16` means the 16-bit brain floating-point format used by the frozen
model runtime. Token scores, coordinate transforms, mask construction, matching,
merging, and metric accumulation are promoted to at least float32 before
research analysis.

Physical batching is sealed independently inside 17 execution-wave partitions:
one independent partition containing `FULL_SINGLE`, `FULL_BAG_K`, `TILE_RESET`,
and `MASK_RESET`, followed by one partition for each of the 16 canonical
`MASK_CUMULATIVE` cells. A batch may never cross a partition boundary. Batch
size four (`B4`) is canonical. Batch size three (`B3`) is permitted only as the
single natural final tail inside one partition; it is not an adaptive fallback.

The 200-image primary schedule contains 13,000 calls and exactly 3,250 B4
batches: 2,450 in the independent partition and 50 in each cumulative-cell
partition. The optional Dense-Union-51 second-root schedule contains 3,315
calls, 816 B4 batches, and 17 B3 tails: one B3 tail after 624 B4 batches in the
independent partition, plus one B3 tail after 12 B4 batches in each of the 16
cumulative-cell partitions. Therefore it has 833 physical batches in total.
Seeds, request order, arm membership, call counts, and estimands are unchanged.

The same request-scoped sampled-runtime mechanics attestation MUST cover B4 and
B3 execution. A one-request loop is not a fallback or an experimental arm.
Resume never silently changes batch cardinality: it may run only whole sealed
batches admitted by the dependency-aware resume contract, or it starts a new
immutable run identifier.

Within one fixed batch cardinality, reversing request order MUST preserve each
request's complete sampled trajectory and selected-token score replay. This is
the admission gate for request-owned random streams and row routing. B4-versus-
B3 trajectory equality is not an admission requirement: an exact CUDA probe at
temperature `0.2` and `max_new_tokens = 512` preserved forward/reverse replay
inside both cardinalities but produced a long-horizon trajectory change for one
shared request when cardinality changed. Cross-cardinality agreement is
therefore persisted as a finite-or-null diagnostic, while the executed
cardinality remains part of every sealed physical-batch receipt.

A physical-batch barrier requires exactly one terminal attempt for every member;
it does not require every member to be successful. `completed`, `failed`,
`skipped`, `capped`, and `invalid` are legal scientific terminal statuses.
Infrastructure or artifact-protocol failure still fails closed, while a legal
non-completed status remains available to the dependency-aware resume contract,
which may require an explicit continuation plan for downstream cumulative work.

The eight authorized Graphics Processing Units are non-exclusive: unrelated
small-memory workloads may coexist. Every model-bearing launch records the
visible-device process inventory plus free and used memory, never terminates or
reconfigures another workload, and proceeds only when every planned worker has
at least 24 gibibytes of free device memory immediately before launch. Here one
gibibyte is 1,073,741,824 bytes. Insufficient headroom pauses the launch; it does
not silently reduce per-device batch size, change active-rank count, or alter
the parallel-layout receipt.

The backend now exposes an explicit request-scoped sampled-generation surface
with one pseudo-random generator per request. That code path is not admitted by
configuration alone: the exact installed runtime, model lineage, generation
policy, request-order gate, and executed batch cardinalities must pass the CUDA
attestation described below before metric-bearing execution.

### Temperature mechanics calibration

Exactly one sampled-runtime CUDA attestation invocation loads the complete
frozen runtime once and covers all three exact calibration generation policies:
temperature `0.2`, temperature `0.4`, and temperature `0.6`. For each policy,
that invocation must persist exact B4 execution, exact B3 execution,
request-order reversal within each fixed cardinality, cross-cardinality
diagnostics, same-seed replay, and admitted-production-path evidence inside one
aggregate attestation artifact. One policy cannot borrow another policy's
capability, and an unattested temperature is forbidden.

After the scientific calibration selects the first passing policy in the
predeclared order, the persisted aggregate evidence authorizes each of the eight
production workers to perform an exact-policy process-local capability rebind.
Every rebind validates the live frozen runtime identity and the selected exact
generation-policy identity before metric-bearing execution. It does not reload
the three-policy attestation panel, alter candidate order, or change any
scientific calibration gate.

Candidate temperatures are tested in this fixed order:

```text
0.2, 0.4, 0.6
```

For each temperature in order, first execute one canonical batch-size-four panel
of four domain-separated seeds per calibration image, giving 48 calls. A
candidate fails immediately unless:

1. at least 46 of 48 calls parse without a call-level failure;
2. at least 39 of 48 calls end through natural closure;
3. at least nine of 12 images produce at least two distinct raw outputs across
   four seeds; and
4. Prediction-Set Diversity is at least `0.10` on images with a nonempty
   sampled-call union.

The first candidate passing those four gates becomes the tentative candidate.
Attest it with two additional 48-call panels: an identical batch-size-four
replay and a batch-size-four execution with reversed request scheduling. Every
request must be byte-identical to its same-seed canonical result in both panels.
A replay or scheduling failure invalidates the sampled backend and stops the
unit; a higher temperature cannot repair request-randomness semantics. Batch
size one is neither a fallback nor an experimental arm.

If no candidate passes the initial four gates, metric-bearing execution stops
and the unit is amended. The chosen temperature is written once to the mechanics
receipt and cannot be changed after any primary-arm output exists. Calibration
does not choose a temperature by Average Precision, Local Rescue Rate, or
preferred object count.

### Seed derivation

The root integer is `2026071301` and the namespace is:

```text
coordexp-dense-enumeration-seeds-v1
```

For each request, build the UTF-8 string:

```text
<namespace>\0<root>\0<role>\0<image_id>\0<cell_or_call_label>
```

Take the first eight bytes of its SHA-256 digest as an unsigned big-endian
integer and clear the most significant bit, producing an unsigned 63-bit seed.

- `FULL_SINGLE` uses role `baseline` and label `single`.
- `FULL_BAG_K` and every paired canonical spatial cell use role `paired-cell`
  and the zero-padded labels `cell-00` through `cell-15`.
- Calibration uses role `temperature-calibration` and labels `call-00` through
  `call-03`.
- Bootstrap resampling uses role `image-bootstrap` and label `replicates-10000`.

The baseline seed is therefore disjoint from every paired-cell seed. Each
request receives its own generator; no global process random state may own the
scientific seed. The run retains the fully materialized per-image seed table.

## Frozen Object Score, Merge, and Match Contract

### Row-local score

Use the existing `compact-object-selected-token-score-v1` policy: exponentiate
the mean selected-token log probability over the four schema wrappers and four
coordinate tokens. Description tokens remain excluded exactly as in the frozen
score policy. Missing, non-finite, or provenance-inconsistent scores make the
prediction invalid for the object merger.

### Final object merger

After ownership and global coordinate mapping, apply class-wise greedy NMS:

1. retain every valid prediction; use no confidence threshold;
2. sort by descending row-local score, then immutable image identifier,
   canonical call identifier, generated row order, and immutable prediction
   identifier;
3. keep the next unsuppressed prediction;
4. suppress each later same-class prediction whose bounding-box IoU with that
   kept prediction is at least `0.70`;
5. do not fuse or average boxes;
6. do not let a suppressed box suppress another box;
7. retain parent and suppressor identifiers for every transition.

This is intentionally not connected-component collapse. Raw owning-call, raw
any-call union, pre-merge, post-merge, merge-created match, and merge-destroyed
match artifacts remain mandatory. A diagnostic strict-duplicate view also uses
same-class IoU at least `0.85`; it does not alter NMS. Its connected components
and Post-Merge Strict Duplicate Rate use the definitions in the terminology
registry. The component representative is the first prediction under the same
deterministic ordering used by NMS.

### Reference matching

Match only identical normalized COCO-80 categories. Primary matching requires
IoU at least `0.50`; the predeclared localization sensitivity requires IoU at
least `0.75`. Choose a one-to-one assignment that first maximizes match count,
then maximizes total IoU, then breaks exact ties lexicographically by immutable
prediction and reference identifiers. Empty conditional denominators return
`not applicable`, never zero.

For audit-augmented manual precision, apply operations in this order:

1. match predictions to `accepted` individual objects by the rule above;
2. for every unmatched prediction, test same-category official or adjudicated
   `crowd` regions and ignore the prediction when intersection area divided by
   prediction area is at least `0.50`;
3. for each still-unmatched prediction, test non-null `ambiguous` or `partial`
   regions whose category or candidate-category set contains the prediction
   category, and ignore the prediction when intersection area divided by
   prediction area is at least `0.50`;
4. count every remaining unmatched valid COCO-80 prediction as an unmatched
   prediction in the manual-precision denominator.

Ignored predictions enter neither the manual-precision numerator nor
denominator. Report accepted matches, crowd-ignored predictions,
uncertainty-ignored predictions, and unmatched predictions separately, plus a
required sensitivity that counts every uncertainty-ignored prediction as
unmatched. Crowd and uncertain regions never become ordinary individual rescue
objects and never increase recall.

Official metrics must use the restored raw `iscrowd` annotations and a named
crowd-aware evaluator whose version and inputs are sealed in the run receipt.
The current evaluator that rewrites crowd flags to zero is ineligible. Official,
audit-augmented, and later post-output adjudication metrics remain separately
named.

## Frozen Estimands, Uncertainty, and Decision Rules

### Primary scope

The five arms execute once on all 200 images. The scientific headline uses the
sealed Audit-Augmented Local Rescue Rate on Dense-Union-51. Official-annotation
views on Dense-Union-51 and all 200 images are secondary scope and
generalization checks from the same raw calls.

Both headline spatial views must pass independently:

1. Masked versus Full Owning-Seed Raw Rescue Difference; and
2. Masked-Canvas versus Full-Image Bagging Policy-Utility Rescue Difference.

Sign agreement alone is insufficient. A positive mechanism-adjacent reading is
`unresolved` if either view misses its threshold or uncertainty rule.

### Uncertainty

Use 10,000 image-clustered bootstrap replicates and the percentile 95-percent
confidence interval. Whole images, every reference object, every arm, and every
raw call travel together. Use the domain-separated bootstrap seed above. Do not
reweight strata in the headline; multi-label strata are descriptive. Report
numerator, denominator, point estimate, lower bound, and upper bound.

For a metric whose denominator is zero in a bootstrap replicate, mark that
replicate `not applicable` for that metric rather than inserting zero. Emit the
number of applicable replicates. A confidence interval is valid only when at
least 9,500 of 10,000 replicates are applicable; otherwise that metric and every
decision depending on it are `not applicable`. For Prediction-Count Inflation,
each applicable replicate is the ratio of the summed named-arm final valid
prediction count to the summed `FULL_SINGLE` final valid prediction count over
the resampled images. The same applicable-replicate rule governs its upper
confidence bound.

### Effect and equivalence thresholds

- Minimum meaningful absolute Local Rescue Rate difference: `0.05`.
- Superiority: the complete lower confidence bound is greater than `+0.05`.
- Equivalence margin: `[-0.05, +0.05]`; the complete interval must lie inside
  the margin.
- Minimum fresh-`FULL_SINGLE` missed accepted objects on Dense-Union-51: `150`.
  A smaller denominator is `underpowered`; the cohort is not expanded post hoc.

### Safety guardrails

All safety intervals are image-clustered and paired to the named comparator.
For the masked spatial-policy claim, the candidate is `MASK_RESET` and the
comparator is `FULL_BAG_K`. For the accepted-row prefix-policy contrast, the
primary candidate is `MASK_RESET` and the comparator is `MASK_CUMULATIVE`,
matching the declared Reset-minus-Cumulative rescue estimand. If the observed
confidence interval instead establishes superiority of `MASK_CUMULATIVE`, apply
the identical guardrails again with candidate and comparator swapped before
calling that reverse direction safe. For the native-tile contrast, the candidate
is `TILE_RESET` and the comparator is `MASK_RESET`. A lower-bound difference is
candidate minus comparator; an upper-bound increase is candidate minus
comparator. Every arm being interpreted must pass the applicable absolute rule
even when it is the comparator.

Mask-harm retention applies separately to `MASK_RESET` and
`MASK_CUMULATIVE`. Overall retention applies separately to every multi-call
primary arm, always relative to reference objects detected by `FULL_SINGLE`.
Prediction-count inflation is the ratio of total final valid predictions in the
named arm to total final valid predictions in `FULL_SINGLE` on the identical
scope. Post-Merge Strict Duplicate Rate uses the frozen strict duplicate
component definition above.

| Guardrail | Frozen acceptance rule |
|---|---|
| Mask-harm retention | Lower confidence bound at least `0.80` for `FULL_SINGLE`-detected reference boxes wholly contained in one core and at least one merged-token quantum from every internal core boundary. |
| Overall retention | Lower confidence bound at least `0.85`. |
| Audit manual precision | Absolute point estimate at least `0.70` and lower bound of the paired difference from the named comparator at least `-0.10`. |
| Post-merge strict duplicate rate | Upper bound of the paired increase at most `+0.05` and absolute point estimate at most `0.10`. |
| Invalid-row rate | Absolute point estimate at most `0.05` and upper bound of the paired increase at most `+0.02`. |
| Natural-closure rate | Absolute point estimate at least `0.75` and lower bound of the paired difference at least `-0.10`. |
| Prediction-count inflation | Upper confidence bound of total final valid predictions divided by total `FULL_SINGLE` valid predictions in each bootstrap replicate on the same scope is at most `2.0`. |

Raw multi-call duplicates are expected and reported but do not own the strict
post-merge duplicate guardrail. Any failed guardrail changes a rescue gain to
`unsafe output expansion` rather than improved enumeration.

### Boundary-result replication trigger

Run a second root-seed replication on Dense-Union-51 only if any headline
confidence bound lies within `0.02` of a decision threshold, if raw and
post-merge directions disagree, or if the chosen bagging temperature has
Prediction-Set Diversity below `0.15` in the primary run. The second root is
`2026071302` with otherwise identical derivation. A direction reversal leaves
the result `seed-sensitive and unresolved`.

## Frozen Budgets

For `K=16`, one five-arm primary image uses:

```text
FULL_SINGLE:       1 call
FULL_BAG_K:       16 calls
TILE_RESET:       16 calls
MASK_RESET:       16 calls
MASK_CUMULATIVE:  16 calls
total:            65 calls
```

Primary all-200 maximum:

```text
model calls: 13,000
configured generated-token ceiling: 6,656,000
```

Temperature calibration maximum, comprising three possible 48-call candidate
panels and two additional 48-call replay/order panels for the first tentative
candidate:

```text
model calls: 240
configured generated-token ceiling: 122,880
```

A triggered Dense-Union-51 second-root replication adds at most:

```text
model calls: 3,315
configured generated-token ceiling: 1,697,280
```

No secondary grid, core-only, traversal-order, historical repetition-penalty,
or alternate-checkpoint panel is included. Each requires a separate pre-output
amendment with its own exact call and token budget. Realized prompt tokens,
generated tokens, image tokens, failures, and wall/device time are recorded in
addition to configured ceilings.

## Required Readiness and Run Receipts

The exact readiness-artifact root for this contract version is:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-13-spatial-scope-history-disentanglement/readiness-v2/
```

`readiness-v1` is preserved as immutable superseded provenance. Its first
pre-seal materialization retained raw numeric annotation order within each
image, while this contract requires frozen image order followed by
lexicographic immutable record identifier. `readiness-v2` rematerializes the
two official ledgers with the corrected deterministic ordering, preserves the
two independent reviewer artifacts byte-for-byte, and binds the old/new
artifact comparison in `pre-seal-ordering-correction-receipt.json`. Only
`readiness-v2` may receive the final audit-augmented ledger and ledger seal.

The root is append-only after `ledger-seal.json` is written. Metric-bearing run
roots remain separate immutable `<run-id>` directories under the same unit
directory and reference the readiness seal by hash.

Before metric-bearing execution, immutable receipts must establish:

1. checkpoint, base model, adapter, special-token, tokenizer, processor, prompt,
   ontology, source, and annotation identities;
2. the ordered 200-image cohort, Dense-Union-51, 12-image calibration cohort,
   per-image source hashes, dimensions, and raw-to-source transforms;
3. official individual, official crowd-ignore, reviewer, adjudication, and
   sealed audit-augmented ledger identities;
4. authored and resolved config, exact Git commit, dirty-diff digest,
   dependency versions, device/runtime identity, and parallel layout;
5. exact per-request seed, executed generation keyword arguments, generator
   identity, same-seed replay, different-seed response, and batch-size-four
   scheduling-order invariance;
6. source shape, token-grid shape, core, halo, mask mean, visible-token support,
   image-token count, ownership, and coordinate round trip for every spatial
   input;
7. exact prompt bytes, prompt token identifiers, image-placeholder count,
   admitted rows, excluded rows, and context-budget result for every call;
8. raw output, token trace, parse result, stop reason, row-local score, failure,
   ownership, global box, merge lineage, and immutable identifiers;
9. raw owning-call, raw union, pre-merge, post-merge,
   merge-created/merge-destroyed, matching, and bootstrap primitives sufficient
   to recompute every estimate;
10. attempted and completed calls plus configured and realized call, prompt,
    image-token, generated-token, device-time, and wall-time budgets;
11. terminal status retaining every failed, skipped, capped, and invalid attempt.

## Current Implementation Gaps and Approval Boundary

The focused review must treat the following as real blockers, not documentation
details:

1. the current HF backend hardcodes deterministic generation and does not carry
   temperature, nucleus cutoff, or per-request seed into executed generation;
2. the current prompt builder does not support an open assistant continuation
   prefix containing accepted rows;
3. the current runtime has no native tile, full-canvas mask, core/halo ownership,
   or multi-call policy surface;
4. the current repository object merger is a data-parallel shard merger, not
   the class-wise object-level NMS declared here;
5. the current official evaluator reconstructs crowd flags incorrectly for this
   unit and cannot own crowd-aware metrics without a research-side restoration
   path or stable fix;
6. cohort, image-hash, reviewer, adjudication, and sealed-ledger artifacts are
   not yet materialized.

The authorized implementation must change the smallest stable surface.
A stable inference request/schema change requires a narrow OpenSpec change;
one-time cohort, mask, prompt-policy, merger, and analysis machinery remains
research code under `scripts/research/` and `src/analysis/`. The user-authorized
custom sampler is limited to categorical token selection inside the existing
batch backend; it does not authorize another inference pipeline or a vLLM
backend.

## Stop Conditions

Implementation approval was contingent on the completed focused review. If a
new unresolved critical- or high-severity scientific issue is found during
implementation or re-review, pause implementation and amend the contract before
continuing.

Stop before metric-bearing execution if any required ledger, identity, hash,
sampling, prompt, spatial, ownership, coordinate, merge, matching, score,
context, budget, or artifact receipt is missing or fails.

Stop interpretation when the accepted missed-object denominator is below 150,
mask-harm retention fails, any safety guardrail fails, bagging remains a
weak-diversity control, a headline cannot be recomputed from primitives, or a
triggered second-root result reverses direction.

Every outcome retains `architecture_promotion_status: not_promoted`.

## Amendment Result

Scientific and desired execution semantics passed focused fixed-point review.
Execution readiness remains blocked and evidence status remains `none` while
authorized implementation is in progress. No architecture is authorized.
