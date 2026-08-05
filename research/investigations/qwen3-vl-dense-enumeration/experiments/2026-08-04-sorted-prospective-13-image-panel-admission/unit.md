---
title: Sorted Prospective 13-Image Panel Admission
description: Materializes a prospective 13-image probe input by byte-preserving the frozen human-refined-12 panel and inserting COCO val image 2299 in numeric image_id order, with no completed-result claim.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-08-04-sorted-prospective-13-image-panel-admission
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Sorted Prospective 13-Image Panel Admission

## Decision and claim boundary

This unit is a data-admission seam, not an experiment. It materializes one
new versioned probe input -- a 13-row panel -- by byte-preserving all twelve
rows of the frozen
[human-refined-12 panel](../2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/results.md)
and inserting exactly one additional row, COCO `val2017` image `2299`, read
from the training authority file
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`,
in ascending numeric `image_id` order. It runs no model, scores no checkpoint,
and changes no training data. It does not identify a mechanism, does not
license an intervention, and does not promote an architecture, a training
treatment, or a production contract.

**This is a prospective probe input only.** It admits one new candidate image
into a versioned evaluation-input file; it does not run inference, does not
score image `2299` against any checkpoint, and does not compute or imply any
13-image aggregate metric. The completed human-refined-12 denominator used by
every prior unit that reports against the twelve-image panel is unchanged and
remains the sole completed-result panel until a future unit explicitly
promotes the 13-image panel through its own admission and evidence gate.

**Reporting constraint.** Any future unit that runs inference, scoring, or
analysis against `human-refined-13.coord.jsonl` must report the legacy-12
subset and image `2299` side-by-side (i.e., as two clearly labeled slices)
before it may report any pooled 13-image summary statistic. This constraint
is also embedded in the build receipt's `scope` block so it survives
independently of this document.

The strongest permitted claim from this unit alone is:

> The training-authority row for COCO `val2017` image `2299` (46 objects: 38
> `person` + 8 `tie`) was appended to a byte-preserved copy of the frozen
> twelve-image human-refined panel (346 objects), in ascending `image_id`
> order, with its `images` reference normalized to the new panel's on-disk
> location and bound to a verified SHA-256; the twelve legacy rows are
> unchanged at the byte level; all 13 published rows' image references were
> independently resolved and hash-verified at the new panel's own location
> (not just image `2299`'s); and the full published panel totals 392 objects
> (346 legacy + 46 new).

### Revision note (post-review)

An earlier draft of this unit verified only image `2299`'s path and hash and
reported a single undifferentiated owner-count block. An independent review
found two contract gaps: (1) the other 12 rows' `images` references were
never resolved or hash-verified at the new panel's location, and (2) the
owner-count block did not distinguish the single inserted row's counts from
the full 13-row panel's counts, and did not check the full-panel total. Both
gaps are closed in the current build (schema `v2`, described below); the
draft output root was removed and rebuilt once before this unit was declared
complete, so no consumer ever saw the uncorrected artifact.

## Scope

- CPU only. No model, tokenizer, GPU, inference, or training config is
  loaded or referenced.
- The builder never copies JPEG bytes. The inserted row references the
  canonical image file
  `/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000002299.jpg`
  by a normalized relative path, and the receipt binds its SHA-256. This
  avoids redundant image storage while still making image `2299` a
  first-class, hash-verified probe object.
- Ownership is limited to
  `scripts/research/build_sorted_prospective_13_image_panel.py`,
  `tests/research/test_build_sorted_prospective_13_image_panel.py`, and this
  experiment's `unit.md`/`tasks.md`. No existing 12-image file, config, other
  script, or shared routing document (`index.md`, `compass.md`,
  `memories/current.md`) is touched by this unit.

## Frozen source identities

| Source | Path | SHA-256 |
| --- | --- | --- |
| Legacy 12-image panel | `outputs/research/qwen3-vl-dense-enumeration/2026-07-21-best-sampled-trajectory-positive-row-imitation-screen/evaluation-inputs/human-refined-12.coord.jsonl` | `cfe4f693133287f9e6c561fc094710c642f049aec3d50f44dea98b764ba2aa85` |
| Training authority (whole file) | `public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl` | `81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894` |
| Training authority row for image 2299 (line 28) | same file, exact line | `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b` |
| Canonical image bytes | `public_data/coco/rescale_32_1024_bbox/images/val2017/000000002299.jpg` | `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3` |

Image `2299` carries 46 owners: 38 `person` + 8 `tie`. The frozen legacy
panel carries 346 objects in total. The builder verifies every one of the
four source identities above, the target row's owner-count triple, the
legacy panel's 346-object total, and the full published panel's 392-object
total (346 + 46, cross-checked as an explicit sum) before it writes any
output byte; any mismatch raises `PanelContractError` and no output is
produced.

## Method

`scripts/research/build_sorted_prospective_13_image_panel.py` implements a
single `build_panel(output_root, sources=...)` seam (schema
`sorted_prospective_13_image_panel.v2`):

1. Read the legacy panel's raw bytes, verify its whole-file SHA-256 and exact
   12-line count, confirm its `image_id`s are strictly ascending with no
   duplicates, and verify its aggregate object count (346, `legacy_owner_counts`
   in the receipt).
2. Read the training authority file, verify its whole-file SHA-256, and
   locate exactly one row with `image_id == 2299`; verify that row's exact
   line SHA-256.
3. Verify the row's owner composition (46 objects: 38 `person` + 8 `tie`,
   `target_owner_counts` in the receipt -- explicitly distinct from the
   legacy and full-panel counts).
4. Resolve the row's single `images` reference to the canonical on-disk
   image, verify its SHA-256, and rewrite the reference as a path relative to
   the new panel's `evaluation-inputs/` directory (all other fields --
   `file_name`, `objects`, `width`, `height`, `image_id`, `metadata` -- are
   carried through unchanged).
5. Insert the normalized row into the legacy sequence in ascending
   `image_id` order (index `1`, between images `1584` and `2685`) and assert
   that removing the inserted line reproduces the original 12-line file
   byte-for-byte. Verify the full 13-row panel's aggregate object count (392,
   `full_panel_owner_counts` in the receipt) equals both the expected
   constant and `legacy_owner_counts.total + target_owner_counts.total`.
6. Stage the 13-line JSONL into a temp directory at the exact depth of the
   final output (`.../evaluation-inputs/`), then fail-closed resolve every
   one of the 13 published rows' `images` references from that staged
   location -- not just image `2299`'s -- verifying each one exists as a
   regular file and recording its `image_id`, resolved path, byte size, and
   SHA-256 into an `images_manifest` table plus a table-level digest. This is
   necessary (not cosmetic): `Path.resolve(strict=True)` requires every
   intermediate path component to physically exist before it will collapse a
   `..` traversal, so this check can only run once files are staged at the
   real target depth, and it verifies the coincidence that the new unit
   directory sits at the same depth as the legacy panel's original directory
   -- the twelve unmodified legacy `images` references still resolve
   correctly from the new location.
7. Publish `evaluation-inputs/human-refined-13.coord.jsonl` and a
   self-sealed `receipt.json` (containing its own `receipt_content_sha256`)
   atomically under one unit output root: the staged temp directory is
   renamed into place on first publish, and a rerun against the same inputs
   discards a freshly re-verified, byte-identical staged candidate as a
   no-op; any foreign file or content divergence in an existing output root
   raises an error instead of silently overwriting.

## Artifacts and verification

Immutable output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-04-sorted-prospective-13-image-panel-admission/`

| Artifact | SHA-256 |
| --- | --- |
| `evaluation-inputs/human-refined-13.coord.jsonl` (13 lines) | `01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8` |
| `receipt.json` (schema `v2`) | `ad78c174897509dca07c24c57d12c897fdad42724d83af08147d79c9644c5414` |

The published JSONL bytes are unchanged from the first draft (the panel's
line content did not change); the receipt hash changed because its schema
gained `legacy_owner_counts`, `full_panel_owner_counts`, the renamed
`target_owner_counts`, and the `images_manifest` table.

Verified end to end:

- All 13 rows are valid JSON; the 12 legacy lines are byte-identical to the
  source panel (lines 1 and 3-13 of the output match lines 1-12 of the
  source exactly; the inserted row lands at index 1).
- All 13 rows' `images` references resolve and hash-verify from the
  published `evaluation-inputs/` directory, not just image `2299`'s; the
  `images_manifest` table in the receipt records each row's `image_id`,
  resolved path, byte size, and SHA-256, plus a table-level digest
  (`63ba3d12f8cc842201f469030e5e17a11366d0ce796f47046bc33e907c5b9576`).
- `target_owner_counts` (the single inserted row): 46 total, 38 `person`,
  8 `tie`. `legacy_owner_counts` (the 12 legacy rows): 346 total.
  `full_panel_owner_counts` (all 13 published rows): 392 total (226 `person`,
  12 `tie`, plus 30 other categories), matching `346 + 46` exactly.
- Running the builder a second time against the same output root returns
  `status: identical_existing_output` with byte-identical artifacts (no-op
  proof); the full 13-image resolution and hash verification re-runs on
  every call, including reruns, before the identical-content comparison.
- Focused tests: `tests/research/test_build_sorted_prospective_13_image_panel.py`
  (19 passed) -- fixture-based contract tests for byte preservation, ordering,
  path normalization, hash/count mismatches (including legacy- and
  full-panel-total drift), a broken legacy image path (fail-closed
  rejection, no partial output), duplicate/foreign-file rejection, no-op
  reruns, and one real-data end-to-end test against the actual
  `/data/CoordExp` sources.
- `ruff check` and `python -m py_compile` are clean on both new files.
- The unowned draft output root was removed (`rm -rf` on the exact literal
  path, confirmed unreferenced anywhere else in the repository before
  removal) and rebuilt from the corrected script; the rebuild is a
  clean-room republish, not a mutation of a previously declared-complete
  artifact.

## Stop boundary

This unit stops after publishing the versioned probe input and its receipt.
It does not run inference, scoring, or any pooled analysis over the 13-image
panel. A future unit that wants to use `human-refined-13.coord.jsonl` for
inference or evaluation must open its own unit, cite this one as the input's
provenance, and honor the legacy-12-vs-image-2299 side-by-side reporting
constraint before any pooled summary.
