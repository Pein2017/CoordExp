## Migration Notes: public-data-pipeline-dedup

This change intentionally removes legacy/back-compat behavior to force a single refreshed contract.

### Breaking Changes Summary

| Area | Old | New |
|---|---|---|
| Max-object suffix | `_max_<N>` (example `_max_60`) | `_max{N}` (example `_max60`) |
| Pixel-space preset artifact | `<split>.raw.jsonl` (+ alias `<split>.jsonl`) | `<split>.jsonl` only (no alias copy) |
| Preset resolution authority | runner + planner both resolved effective preset | planner-only resolution (`run.sh` passes base preset + max_objects) |
| Derived preset images | byte-copy fallback allowed | hardlink-only materialization, fail-fast on link errors |

### What You Need To Update

1) Preset directory names and suffixes

- If you have existing preset dirs ending in `_max_60`, rename to `_max60` (or rebuild via the pipeline).
- This change removes “equivalence” and “reuse legacy if exists” resolution; mismatched names fail fast.

2) Any path references to `.raw.jsonl`

- Update scripts/configs/docs/tests to stop referencing:
  - `train.raw.jsonl`, `val.raw.jsonl`
- Use:
  - `train.jsonl`, `val.jsonl`

3) Rebuild behavior when rescale params change

- Preset `images/` are treated as immutable once written by rescale.
- Tools MUST NOT overwrite existing files in-place (hardlinks make in-place overwrite cross-preset dangerous).
- If you need to change rescale parameters (max_pixels/min_pixels/image_factor), rebuild is manual:
  - pick a fresh preset name, or
  - deliberately delete the existing preset directory (destructive).

4) Derived presets now require hardlink support (no copy fallback)

- Max-object derived presets materialize `images/` by creating hardlinks to the base preset's resized images.
- This requires base + derived presets to live on the same filesystem (hardlinks cannot cross devices).
- If hardlink creation fails (for example cross-device link), the pipeline fails fast with a hint to move/recreate the preset under a single storage root.

5) Runner behavior change (effective preset source of truth)

- `public_data/run.sh` no longer pre-resolves effective preset names.
- It forwards `--preset <base>` plus optional `PUBLIC_DATA_MAX_OBJECTS` to the pipeline factory.
- The planner computes and prints the effective preset path.

### Straggler Detection (Expected Zero Matches)

Run these from repo root:

```bash
rg -n '\\.raw\\.jsonl' configs public_data scripts tests docs openspec/specs
rg -n '_max_' configs public_data scripts tests docs openspec/specs
```

Expected result:
- `.raw.jsonl` scan: zero, except intentionally legacy/archive references.
- `_max_` scan: only explicitly documented legacy migration examples (or zero).

### Sanity Commands (Post-Migration)

- Base preset:
  - `./public_data/run.sh coco rescale --preset rescale_32_768_bbox`
- Derived preset (max filter):
  - `PUBLIC_DATA_MAX_OBJECTS=60 ./public_data/run.sh coco coord --preset rescale_32_768_bbox`
- Training preflight:
  - `bash scripts/train.sh config=configs/stage2_ab/prod/desc_first_a_only.yaml`
