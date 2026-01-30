# Public Data Changelog

## 2026-01-28 — Visual Genome (VG) pipeline reproducibility

- Added `public_data/vg/README.md` and `configs/public_data/vg.yaml` to record default VG preparation choices (objects version, split policy, filters, preset).
- Hardened `public_data/scripts/prepare_visual_genome.py`:
  - sha256 checksum verification for VG zips (opt-out via `--no-verify-checksums`)
  - bbox rounding/clipping via `src/datasets/geometry.py` helpers
  - deterministic per-image dedupe of exact (desc, bbox_2d) duplicates (opt-out via `--no-dedupe-objects`)
  - drop high-confidence junk `desc` labels like articles/pronouns (opt-out via `--no-filter-junk-descs`)
  - fail-fast on misaligned VG annotation arrays (`image_data.json` vs objects/regions)
- Added end-to-end smoke coverage:
  - `public_data/vg/smoke_test.py` (no-download synthetic pipeline + validation)
  - `tests/test_vg_pipeline_smoke.py` (pytest coverage for the same flow)
- Added `public_data/vg/collect_junk_descs.py` and `public_data/vg/junk_descs.py` to audit and define junk-label filtering.
