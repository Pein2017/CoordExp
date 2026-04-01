## Why

The current COCO/LVIS recovery work has already shown that LVIS can improve
COCO-80 recall on shared images, but it also exposed a critical distinction:

- some recovered LVIS instances are reliable enough to behave like ordinary
  COCO GT,
- some recovered LVIS instances are useful objectness cues but are too weak in
  semantics, spatial extent, or both to be treated as hard GT.

The user explicitly wants to accept that tradeoff:

> a small number of proxy mistakes is acceptable if the added recall benefit is
> materially larger.

That means the next change should not be "merge LVIS into COCO blindly".
Instead, it should give the training stack a reproducible way to carry two
different supervision strengths:

- `strict`: reliable recovered objects that can be added directly,
- `plausible`: useful proxy objects that should stay in training but with lower
  `desc_ce` and lower bbox/coord supervision.

The latest analysis also shows that "semantic evidence" alone is still too
coarse. We need one more determined proxy-mapping layer that separates:

- `same_extent_proxy`:
  - LVIS and COCO annotate essentially the same object extent
  - example direction: `mug -> cup`
- `cue_only_proxy`:
  - LVIS is a useful objectness cue for a COCO class, but the annotated extent
    is systematically different
  - example direction: `tablecloth -> dining table`
- `reject`:
  - ambiguous, weakly supported, or geometrically unstable mappings

The current evidence already suggests this split:

- `mug -> cup` has strong matched-instance geometry and behaves like a direct
  proxy,
- `tablecloth -> dining table` has decent co-occurrence evidence but weaker
  extent agreement and behaves more like a cue than a hard replacement box.

The current stack already has the right loss decomposition to support this:

- `token_ce` separates `struct_ce` from `desc_ce`,
- `bbox_geo` and `coord_reg` already consume per-group / per-slot weights,
- Stage-2 target building already has span-aware and group-aware supervision
  carriers.

So the right first change is:

- keep the visible CoordJSON target format unchanged,
- append extra LVIS-derived proxy objects into the same offline COCO training
  dataset artifact,
- record aligned per-object proxy metadata outside the rendered target text,
- and let the trainer apply object-local weights only to the `desc` and coord
  families while leaving structure supervision global.

This is an important contract change because it affects:

- how augmented COCO training data is materialized,
- how object-local supervision metadata aligns with object order,
- how token types are interpreted for proxy supervision,
- and how Stage-2 training consumes weighted object-level targets without
  breaking the canonical prompt / JSON contract.

## What Changes

- Introduce a new capability:
  - `lvis-coco-proxy-supervision`
- Define an offline augmented COCO training artifact that can append LVIS
  recovered objects to the canonical COCO object list while preserving the
  existing CoordJSON contract.
- Define per-object supervision tiers:
  - `real`
  - `strict`
  - `plausible`
- Define a determined proxy-mapping strategy before supervision tiers are
  assigned:
  - `same_extent_proxy`
  - `cue_only_proxy`
  - `reject`
- Require tier assignment to follow the mapping strategy:
  - `same_extent_proxy -> strict`
  - `cue_only_proxy -> plausible`
  - `reject -> not exported for supervision`
- Define aligned top-level metadata for each rendered object entry, including:
  - `source`
  - `proxy_tier`
  - `mapping_class`
  - `desc_ce_weight`
  - `coord_weight`
  - optional LVIS mapping / recovery provenance
- Keep the assistant-visible object syntax unchanged:
  - object entries remain ordinary `{"desc": ..., "bbox_2d": [...]}` values
  - no supervision metadata is injected into rendered object text
- Make structure supervision explicit:
  - punctuation, braces, commas, quotes, and field-name tokens like `"desc"` and
    `"bbox_2d"` remain global `struct_ce` targets
  - proxy weighting applies only to desc-value tokens and bbox/coord groups
- Extend Stage-2 teacher-forcing context / modules so:
  - `token_ce` can consume metadata-driven desc weights,
  - `bbox_geo` can consume metadata-driven bbox group weights,
  - `coord_reg` can consume metadata-driven coord-slot weights,
  - structure CE stays unaffected by proxy weights
- Add canonical config knobs so Stage-2 pipelines can opt into metadata-driven
  object weighting without new CLI flags.
- Add observability for how many proxy objects and how much effective proxy
  weight enter training.
- Add a finer-grained mapping-analysis requirement before export, including
  extent-compatibility metrics such as:
  - `intersection_over_lvis`
  - `intersection_over_coco`
  - `area_ratio`
  - one-way containment rates
  - normalized center offset
- Add an explicit determined-mapping artifact that can be consumed by export
  rather than re-derived informally at training time:
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv`
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_summary.json`
  - `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_report.md`

## Recommended First Version

The recommended v1 is intentionally conservative and Stage-2-first:

- materialize one offline augmented COCO JSONL artifact per split rather than a
  runtime join,
- keep bbox-only supervision for this first change,
- determine proxy strategy from evidence before exporting training tiers,
- keep the rendered target syntax identical to today's CoordJSON,
- carry all proxy-specific information in top-level metadata aligned 1:1 with
  the final object order,
- support three per-object tiers:
  - `real`
  - `strict`
  - `plausible`
- support three mapping strategies:
  - `same_extent_proxy`
  - `cue_only_proxy`
  - `reject`
- preserve deterministic object ordering after merge using the repo's existing
  `(minY, minX)` invariant,
- apply metadata-driven weighting only in Stage-2 pipeline modules first:
  - `token_ce`
  - `bbox_geo`
  - `coord_reg`
- leave Stage-1 integration as a follow-up once the Stage-2 behavior is
  validated.

Recommended initial supervision policy:

- `real`:
  - `desc_ce_weight = 1.0`
  - `coord_weight = 1.0`
- `strict` (`same_extent_proxy`):
  - `desc_ce_weight = 1.0`
  - `coord_weight = 1.0`
- `plausible` (`cue_only_proxy`):
  - `desc_ce_weight = 0.25`
  - `coord_weight = 0.10`

Recommended first-pass mapping-analysis policy:

- `same_extent_proxy` requires:
  - strong semantic evidence,
  - strong overlap quality,
  - strong extent symmetry
- `cue_only_proxy` requires:
  - enough evidence to indicate useful objectness / recall signal,
  - but clear extent mismatch or unstable same-extent compatibility
- `reject` covers:
  - ambiguous mappings,
  - weak support,
  - or mappings whose geometry does not look trustworthy even as a cue

This recommendation is not meant to freeze one single weight choice forever.
It exists to define a clear first implementation and clean ablation surface.

## Assumptions

- The user prefers recall gains over eliminating every proxy-label mistake.
- The current LVIS recovery pipeline can already produce a reproducible split
  between `same_extent_proxy`, `cue_only_proxy`, and `reject` before training
  starts.
- Preserving the existing rendered prompt / object syntax is more important
  than surfacing proxy information in-model as extra text.
- In the first version, Stage-2 is the best landing zone because it already has
  object-local weighted supervision carriers that can be extended cleanly.

## Non-Blocking Follow-Ups

- Add Stage-1 support for metadata-driven proxy weights.
- Allow proxy metadata to carry richer uncertainty subtypes, such as:
  - semantic-uncertain
  - extent-uncertain
- Learn or tune proxy weights from held-out validation rather than fixing them
  manually.
- Support proxy-aware ignore-only or objectness-only modes for highly uncertain
  categories.
- Extend the same pattern to polygon supervision if that becomes important.

## Risks To Validity

- Plausible objects may still bias classification or localization if their
  effective weight mass becomes too large.
- Metadata/object order misalignment would silently corrupt supervision unless
  the builder fails fast.
- Structure tokens must stay globally supervised; if proxy weighting leaks onto
  keys or punctuation, syntax learning could regress.
- Some proxy categories may be useful only for recall and not for box extent,
  so a single `coord_weight` may still be too coarse for later work.
- If `same_extent_proxy` vs `cue_only_proxy` is determined from weak metrics,
  the wrong supervision tier could still be assigned even if the weight plumbing
  is correct.
- Stage-2-only support may leave Stage-1 experiments temporarily behind until a
  follow-up lands.

## Required Evidence

- Evidence that augmented JSONL records preserve the canonical CoordJSON
  rendering contract while adding only top-level metadata.
- Evidence that metadata arrays stay aligned 1:1 with the final sorted object
  order after LVIS recovery injection.
- Evidence that the mapping-analysis pass can separate same-extent proxies from
  cue-only proxies using geometry-compatibility metrics rather than semantic
  evidence alone.
  - minimum expected metrics:
    - `intersection_over_lvis`
    - `intersection_over_coco`
    - `area_ratio`
    - `coco_contains_lvis_rate`
    - `lvis_contains_coco_rate`
    - normalized center offset
- Evidence that Stage-2 desc weighting changes only desc-value supervision and
  not structure CE.
- Evidence that Stage-2 bbox/coord weighting follows the intended per-object
  proxy tiers.
- Evidence from at least one ablation comparing:
  - base COCO
  - COCO + strict
  - COCO + strict + plausible
- Evidence that proxy-weight observability is visible in training logs / debug
  artifacts.
- Evidence that the determined-mapping artifact is explicit and reproducible,
  with enough columns to audit why a mapping landed in `strict`,
  `plausible`, or `reject`.

## Current Baseline Artifact

The current val2017 semantic-proxy baseline has been exported to:

- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017.csv`
- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_summary.json`
- `openspec/changes/add-lvis-coco-proxy-supervision/artifacts/determined_proxy_mappings_val2017_report.md`

This artifact was derived from the existing:

- `temp/coco_lvis_projection_val2017/mapping_evidence.csv`
- `temp/coco_lvis_projection_val2017/learned_mapping.json`

using rule version `v1_semantic_proxy_rank_2026-04-01`.

Current semantic-evidence counts from that export:

- mappings considered: `53`
- `strict`: `8`
- `plausible`: `13`
- `reject`: `32`

The implementation should use this exported mapping file as the baseline
contract for semantic proxy selection, rather than relying on informal
hand-picked examples.

## Capabilities

### New Capabilities

- `lvis-coco-proxy-supervision`: offline augmentation of COCO-style training
  data with LVIS-derived direct and cue-only proxies plus aligned object-local
  supervision metadata.

### Modified Capabilities

- `teacher-forcing-unified-loss-registry`: clarify that structure tokens remain
  global while proxy weights affect only desc and coord families.
- `teacher-forcing-objective-pipeline`: require derived object-local span/group
  carriers for metadata-driven proxy weighting.
- `stage2-ab-training`: extend the Stage-2 AB config / runtime contract so
  `token_ce`, `bbox_geo`, and `coord_reg` can opt into metadata-driven object
  weighting.
- `rollout-matching-sft`: extend rollout-aligned Stage-2 to the same
  metadata-driven object-weight contract.
- `trainer-metrics-components`: add canonical observability for proxy-object
  counts and effective supervision weights.

## Impact

- Immediate impact is OpenSpec / contract definition only.
- Expected implementation surface is likely centered on:
  - `src/analysis/coco_lvis_missing_objects.py`
  - `src/datasets/builders/jsonlines.py`
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/teacher_forcing/rollout_meta.py`
  - `src/trainers/teacher_forcing/modules/token_ce.py`
  - `src/trainers/teacher_forcing/modules/bbox_geo.py`
  - `src/trainers/teacher_forcing/modules/coord_reg.py`
  - Stage-2 config schema / validation surfaces
