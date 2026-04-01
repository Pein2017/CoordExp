## Context

The user wants to inject extra LVIS-derived annotations into the existing COCO
training dataset, but not all recovered objects should be treated as equally
trustworthy.

The desired supervision split is:

- `real` objects:
  - native COCO objects
- `strict` proxy objects:
  - LVIS-derived objects that are safe enough to behave like hard GT
- `plausible` proxy objects:
  - LVIS-derived objects that imply useful objectness / recall signal, but
    whose semantics or spatial extent are too vague to supervise as hard GT

But there is one more layer that needs to be explicit before those supervision
tiers are assigned:

- `same_extent_proxy`:
  - semantically and geometrically close enough to behave like the target COCO
    object
- `cue_only_proxy`:
  - semantically useful enough to indicate objectness or missing recall, but
    with a systematically different annotation extent
- `reject`:
  - too ambiguous or unstable to export as supervision

The most important constraint from the discussion is:

> structure tokens must still be trained globally.

That means the system cannot simply lower CE on an entire object entry.
Instead, it needs to preserve the current token-type semantics:

- structure tokens:
  - punctuation
  - braces / brackets
  - commas / colons
  - quotes
  - field names such as `"desc"` and `"bbox_2d"`
- desc-value tokens:
  - the natural-language content inside each object's `desc`
- coord tokens:
  - the `<|coord_k|>` tokens or equivalent bbox slot groups

The repo already has strong building blocks for this:

- rendered target syntax is stable CoordJSON,
- Stage-2 `token_ce` already separates `struct_ce` and `desc_ce`,
- Stage-2 `bbox_geo` and `coord_reg` already consume weighted bbox groups /
  coord slots,
- target-building code already knows how to recover desc spans and bbox groups
  from teacher-forced sequences.

So the correct design is not "invent a second target language". The correct
design is:

- keep the visible target syntax identical,
- carry proxy supervision information in aligned metadata,
- and project that metadata onto the existing token-type / bbox-group loss
  carriers.

## Goals / Non-Goals

**Goals**

- Define a reproducible offline augmented COCO artifact that can include
  LVIS-derived `strict` and `plausible` objects.
- Define a determined mapping-analysis step that can separate same-extent
  proxies from cue-only proxies before training export.
- Keep the raw JSONL and rendered CoordJSON contract stable.
- Preserve one-to-one alignment between final object order and proxy metadata.
- Apply object-local proxy weights only to:
  - desc-value CE
  - bbox geometry
  - coord-regularization
- Keep structure CE global and unaffected by proxy confidence.
- Land the first implementation with a shared metadata contract that supports:
  - Stage-1 via collator-emitted aligned token-weight tensors
  - Stage-2 via existing weighted supervision carriers

**Non-Goals**

- No change to the visible object entry syntax for this first version.
- No new CLI flags.
- No requirement to inject proxy provenance into the model prompt text.
- No polygon-specific proxy supervision in the first version.

## Decisions

### 1) The augmentation surface is an offline merged JSONL, not a runtime join

The first implementation should materialize a new dataset artifact rather than
joining COCO and LVIS recoveries at fetch time.

Why:

- reproducibility is better when the exact augmented object list is frozen in
  one artifact,
- cache behavior stays predictable,
- per-sample inspection is easier,
- debugging metadata/object alignment is much simpler,
- and ablations become:
  - original COCO JSONL
  - strict-augmented JSONL
  - strict+plausible augmented JSONL

This keeps the augmentation step auditable and lets training stay close to the
existing dataset path.

### 2) Proxy supervision lives in top-level metadata aligned to the final object list

Each final record keeps the existing canonical object list:

```json
{
  "images": ["..."],
  "objects": [
    {"desc": "dining table", "bbox_2d": [...]},
    {"desc": "dining table", "bbox_2d": [...]}
  ],
  "metadata": {
    "coordexp_proxy_supervision": {
      "object_supervision": [
        {"source": "coco", "proxy_tier": "real", "desc_ce_weight": 1.0, "coord_weight": 1.0},
        {"source": "lvis", "proxy_tier": "plausible", "desc_ce_weight": 0.25, "coord_weight": 0.1}
      ]
    }
  }
}
```

Key invariant:

- `len(metadata.coordexp_proxy_supervision.object_supervision) == len(objects)`

The metadata list is ordered exactly like the final sorted `objects` list.

Why top-level metadata:

- it preserves the current assistant rendering contract,
- existing dataset/builder paths already preserve top-level metadata,
- and object-local supervision remains available to the trainer without being
  part of the generated text.

### 3) Mapping class and supervision tier are separate layers

The first version should keep two distinct decisions:

- mapping class:
  - `same_extent_proxy`
  - `cue_only_proxy`
  - `reject`
- supervision tier:
  - `real`
  - `strict`
  - `plausible`

Normative mapping semantics:

- `same_extent_proxy`:
  - the LVIS category usually annotates the same physical object extent that
    COCO intends for the mapped class
- `cue_only_proxy`:
  - the LVIS category is still a useful objectness or presence cue, but its box
    extent is not reliably the same as COCO's intended object extent
- `reject`:
  - the mapping is too weak, too ambiguous, too low-support, or too
    semantically wrong to be used for training

Normative tier semantics:

- `real`:
  - native dataset object, behaves like ordinary GT
- `strict`:
  - proxy object treated as GT-strength supervision
- `plausible`:
  - proxy object retained for soft supervision only

Recommended initial weights:

- `real`:
  - `desc_ce_weight = 1.0`
  - `coord_weight = 1.0`
- `strict`:
  - `desc_ce_weight = 1.0`
  - `coord_weight = 1.0`
- `plausible`:
  - `desc_ce_weight = 0.25`
  - `coord_weight = 0.0`

These values should be encoded in the exported artifact so that the training
run is fully reproducible from the data artifact alone.

### 3a) Stage-1 uses batch extras as the proxy-weight carrier

Stage-1 does not have the Stage-2 target-builder pipeline, so the same metadata
contract should be projected into collator-emitted batch extras:

- `proxy_desc_token_weights`
- `proxy_coord_token_weights`

These tensors are aligned 1:1 with `labels` and MUST:

- assign structure tokens no proxy-specific downweighting,
- assign desc-value tokens according to per-object `desc_ce_weight`,
- assign coord tokens according to per-object `coord_weight`,
- preserve pack concatenation order exactly.

This keeps the rendered target unchanged while letting Stage-1 consume the same
offline augmented dataset contract as Stage-2.

Tier assignment must not be guessed directly from semantic evidence, though.
It should flow from the proxy-mapping strategy:

- `same_extent_proxy -> strict`
- `cue_only_proxy -> plausible`
- `reject -> not exported`

### 4) A determined proxy-mapping analysis is required before export

The current semantic-evidence mining stage is necessary but not sufficient.
The exporter needs one more geometry-compatibility layer, and that decision
needs to be materialized as a standalone artifact before dataset export.

For each candidate LVIS->COCO mapping, the analysis should measure at least:

- `precision_like`
- `coverage_like`
- `mean_iou`
- `median_iou`
- `iou_ge_05_rate`
- `iou_ge_075_rate`
- `intersection_over_lvis`
- `intersection_over_coco`
- `area_ratio`
- `coco_contains_lvis_rate`
- `lvis_contains_coco_rate`
- normalized center offset

Interpretation:

- `same_extent_proxy`:
  - strong semantic evidence
  - high overlap
  - high extent symmetry
  - low one-way containment bias
- `cue_only_proxy`:
  - good evidence that the LVIS category is a useful indicator for the COCO
    class
  - but clear extent asymmetry or unstable same-extent behavior
- `reject`:
  - ambiguous target class, weak support, or poor geometry quality

This is the specific place where mappings like:

- `mug -> cup`
- `tablecloth -> dining table`

should diverge.

`mug -> cup` should be allowed to land in `same_extent_proxy` if the symmetric
geometry metrics stay strong.

`tablecloth -> dining table` should land in `cue_only_proxy` if the evidence
shows that LVIS boxes are typically contained within or otherwise systematically
smaller than the COCO table extent.

The output of this stage should be a ranked determined-mapping artifact that
the export step can consume directly:

- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv`
- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_summary.json`
- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_report.md`

At minimum, the CSV contract should include:

- LVIS category identity:
  - `lvis_category_id`
  - `lvis_category_name`
- mapped COCO category identity:
  - `mapped_coco_category_id`
  - `mapped_coco_category_name`
- source / prior context:
  - `mapping_kind`
  - `candidate_source`
  - `candidate_kind`
  - `prior_kind`
- exported determination:
  - `determination_tier`
  - `mapping_class`
  - `decision_rule_version`
  - `decision_rule`
- supporting evidence:
  - `n_match`
  - `n_images`
  - `precision_like`
  - `coverage_like`
  - `mean_iou`
  - `median_iou`
  - `iou_ge_05_rate`
  - `iou_ge_075_rate`
- ranking / audit helpers:
  - `support_score`
  - `geometry_score`
  - `proxy_score`
  - `determination_reason`
  - `tier_rank`

The dataset exporter should consume this determined-mapping CSV rather than
recomputing strict/plausible policy ad hoc from raw evidence tables during
every export run.

### 5) The rendered target text stays exactly the same

The system should not emit proxy metadata into the visible assistant payload.

Object entries remain ordinary CoordJSON values:

- `{"desc": "...", "bbox_2d": [...]}` for bbox objects

This means:

- the tokenizer/rendering contract does not change,
- prompt variants stay compatible,
- inference and evaluation parsing do not need proxy-specific syntax support,
- and the proxy weighting logic remains entirely a training-time concern.

### 6) Structure supervision remains global; proxy weights apply only to desc and coord families

This is the most important behavioral decision in the change.

Normative behavior:

- structure CE remains fully supervised for all object entries, including
  plausible proxy objects,
- desc-value CE uses the object-local `desc_ce_weight`,
- bbox geometry and coord regularization use the object-local `coord_weight`.

Mapping-class implications:

- `same_extent_proxy` objects may use GT-strength `desc_ce_weight` and
  GT-strength `coord_weight`
- `cue_only_proxy` objects may still use full structure CE, but should use:
  - weak or moderate `desc_ce_weight`
  - strongly downweighted `coord_weight`
  - and `coord_weight = 0.0` is explicitly allowed when the objectness signal is
    useful but the box extent is not trustworthy

Concrete interpretation:

- `"desc"` is a structure token sequence, not desc-value content
- `"bbox_2d"` is a structure token sequence, not coord content
- commas / braces / quotes remain global structure targets
- the natural-language value of the desc field is the only desc-weighted text
  region
- bbox coord slots are the only coord-weighted geometry region

This preserves JSON / prompt competence while still letting plausible objects
contribute weaker semantic and localization signal.

### 7) Module opt-in is explicit via `object_weight_mode: metadata`

To stay config-first and avoid silent behavior changes, Stage-2 modules should
consume proxy metadata only when their authored config enables it.

Recommended module config addition:

- `token_ce.config.object_weight_mode: none | metadata`
- `bbox_geo.config.object_weight_mode: none | metadata`
- `coord_reg.config.object_weight_mode: none | metadata`

Recommended first behavior:

- `none`:
  - ignore proxy metadata and behave as today
- `metadata`:
  - use object-local weights when present
  - fall back to `1.0` weights if the metadata block is absent

This lets us ablate the data artifact independently from the runtime behavior.

### 8) The teacher-forcing context must expose derived object-local views

The modules should not each rediscover proxy/object alignment from raw text.
The shared context / target-builder layer should provide:

- object-local mapping class labels,
- object-local desc spans,
- object-local bbox group indices,
- object-local desc weights,
- object-local bbox/coord weights,
- fail-fast checks for count mismatches.

This keeps the module code simple and avoids duplicated parsing logic.

### 9) Stage-2 is the correct first implementation surface

The first implementation should target:

- `stage2_two_channel`
- `stage2_rollout_aligned`

Why:

- the current Stage-2 path already separates structure vs desc CE,
- rollout and GT teacher forcing already operate through shared modules,
- bbox / coord modules already accept weighted groups and slots.

Stage-1 is a logical follow-up, but it should not block the first proxy
supervision contract.

### 10) Proxy supervision needs explicit observability

Training logs should expose at least:

- how many `real`, `strict`, and `plausible` objects were active,
- the effective desc-weight sum,
- the effective coord-weight sum.

Without this, a run could silently overuse plausible objects or mis-handle the
metadata path.

## Open Questions

- Whether `strict` should always use weight `1.0` or reserve room for
  near-strict weights later.
- Whether `plausible` should later split into:
  - semantic-uncertain
  - extent-uncertain
- What exact threshold family best separates `same_extent_proxy` from
  `cue_only_proxy` across categories beyond the initial evidence sweep.
- Whether Stage-1 should share the same metadata carrier exactly or use a
  thinner adapter.

These are follow-up questions and should not block the v1 contract.
