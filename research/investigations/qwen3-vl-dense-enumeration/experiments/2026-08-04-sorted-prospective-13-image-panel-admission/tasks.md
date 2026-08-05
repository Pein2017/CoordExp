---
title: Sorted Prospective 13-Image Panel Admission - Tasks
description: CPU-only data-admission checklist for materializing the prospective 13-image probe input.
type: investigation
role: research-tasks
authority: non_normative_research
unit_id: 2026-08-04-sorted-prospective-13-image-panel-admission
topic: qwen3-vl-dense-enumeration
status: complete
updated: 2026-08-04
---

# Tasks

## Design

- [x] Bind the four frozen source identities: legacy 12-image panel
  (whole-file SHA-256), training authority file (whole-file SHA-256), the
  exact authority line for image `2299` (line SHA-256), and the canonical
  image bytes (SHA-256).
- [x] Declare the prospective-probe-only boundary: no completed 12-image
  denominator change, no pooled 13-image summary without a future unit's own
  admission and side-by-side reporting.
- [x] Decide not to duplicate JPEG bytes into the output tree; reference the
  canonical image by a normalized relative path and bind its SHA-256 in the
  receipt instead.

## Implementation

- [x] Implement `scripts/research/build_sorted_prospective_13_image_panel.py`
  as one CPU seam that verifies every frozen input identity before writing
  any output byte.
- [x] Byte-preserve all 12 legacy panel lines; insert the normalized image
  `2299` row in ascending `image_id` order; assert removing the inserted
  line reproduces the original file byte-for-byte.
- [x] Verify owner composition on the inserted row (46 objects: 38 `person` +
  8 `tie`) before publishing.
- [x] Publish `evaluation-inputs/human-refined-13.coord.jsonl` and a
  self-sealed `receipt.json` atomically under one unit output root, using a
  create-or-identical commit (temp-directory rename on first publish;
  byte-identical no-op on rerun; error on any foreign or divergent existing
  content).
- [x] Add focused tests covering byte preservation, insertion ordering, path
  normalization, hash/count-mismatch rejection, duplicate/foreign-file
  rejection, no-op reruns, and one real-data end-to-end run.

## Post-review correction (schema v2)

An independent review found two contract gaps in the first draft: only image
`2299`'s path/hash was verified (not all 13 runtime image references), and
the owner-count block did not distinguish the inserted row's counts from the
full-panel counts or check the full-panel total. Both are closed here; the
draft output root was removed and rebuilt once (it was unowned/unreferenced
elsewhere, so this is a clean-room republish, not a mutation of a
declared-complete artifact).

- [x] Add fail-closed resolution/existence/hash verification for all 13
  published rows' `images` references (not just image `2299`'s), staged at
  the exact depth of the final output so `Path.resolve(strict=True)` can
  physically verify each intermediate path component; seal the result as an
  `images_manifest` table (`image_id`, resolved path, byte size, SHA-256) plus
  a table-level digest in the receipt.
- [x] Add `legacy_owner_counts` (346, the 12 legacy rows) and
  `full_panel_owner_counts` (392 = 346 + 46, all 13 published rows,
  cross-checked as an explicit sum) to the receipt; rename the prior
  undifferentiated `owner_counts` block to `target_owner_counts` (the single
  inserted row only).
- [x] Add a test for a broken legacy image path (fail-closed rejection, no
  partial output written) and a test for full-panel/legacy count drift.
- [x] Safely remove the exact, validated, unreferenced draft output root
  (`rm -rf` on the literal path only, after confirming no other file in the
  repository references it) and rebuild once from the corrected script.
- [x] Rebuild twice to reconfirm `created` then `identical_existing_output`,
  with the full 13-image verification re-running before the identical-content
  comparison on every call (not skipped on rerun).

## Execution and verification

- [x] Verify all four frozen source hashes and the 46/38/8 owner-count
  identity against the real `/data/CoordExp` files before running the
  builder.
- [x] Run the builder against real data; capture the receipt and artifact
  hashes.
- [x] Run the builder a second time against the same output root and confirm
  `status: identical_existing_output` with byte-identical artifacts.
- [x] Independently re-verify on disk: line count, byte-for-byte legacy
  preservation, image path resolution, and image SHA-256.
- [x] Run focused tests (`pytest tests/research/test_build_sorted_prospective_13_image_panel.py`,
  19 passed), `ruff check`, `python -m py_compile`, and `git diff --check` on
  the owned files.
- [x] Write `unit.md` stating the prospective-probe-only boundary, the
  legacy-12-vs-image-2299 side-by-side reporting requirement, and the
  post-review schema-`v2` correction.

## Stop boundary

- [x] Stop after publishing the versioned probe input and its receipt. Do
  not run inference, scoring, GPU work, or any pooled 13-image analysis in
  this unit.
