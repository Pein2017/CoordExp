---
title: Research raw-material deletion receipt
type: investigation
role: archival-receipt
authority: non_normative_research
status: complete
updated: 2026-08-29
---

# Deletion receipt

- Exact targets removed: **84** (the paths in `deletion-targets.txt` only).
- Deletion command: `/usr/bin/find <exact-target> -xdev -depth -delete`.
- Manifested file payload removed: **59.44 GiB**.
- Post-delete `/data/CoordExp/outputs/research`: **21,001,461,760 bytes** (~19.56 GiB).
- Post-delete qwen dense root: **20,182,781,952 bytes**; 56 top-level roots remain.
- Post-delete output file count: **7,356**.

## Safety checks

- Every target resolved as a real directory under `/data/CoordExp/outputs/research`.
- qwen targets had no `image2299` descendant.
- No target had an `lsof` handle immediately before deletion.
- No compute process was reported by the GPU snapshot.
- All Image2299-marked roots, both live qwen roots, and the eight-coordinate
  closeout dependency remain present.

## Not deleted

The remaining raw is intentional: current Image2299 work, two active
non-Image2299 qwen roots, and the explicit step-2444 source bundle consumed by
Image2299. Revisit only with a new exact manifest after active runs close or
the Image2299 worktree records a replacement input.

