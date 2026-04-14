## 1. Offline Bbox-Format Branch Generator

- [x] 1.1 Add a shared offline bbox-format derivation module in `public_data/scripts/` that runs as a separate `public_data/` workflow branch and emits derived split artifacts under sibling preset roots named `<preset>_<format>/`.
- [x] 1.2 Implement first-pass `cxcy_logw_logh = [cx, cy, logw, logh]` derivation from canonical preset `xyxy` sources, exporting `train.jsonl` / `val.jsonl` plus matching `train.coord.jsonl` / `val.coord.jsonl` when those splits exist.
- [x] 1.3 Stamp derived records with `prepared_bbox_format`, slot-order metadata, canonical source lineage, and conversion-version metadata, and add focused tests that round-trip a known canonical box through the offline branch.

## 2. Public-Data Runner And Validation

- [ ] 2.1 Extend `public_data/run.sh` and the shared pipeline planner to support a first-class `bbox-format` command without adding new runner flags.
- [ ] 2.2 Add validation coverage for sibling derived preset roots such as `<preset>_cxcy_logw_logh/`, including missing-manifest, missing split JSONL outputs, provenance-mismatch, and malformed-geometry failures.
- [ ] 2.3 Keep `public_data/run.sh <dataset> all --preset <preset>` canonical-only, and add runner tests that prove `all` does not silently create non-canonical branches.

## 3. Runtime Training Guard Refactor

- [x] 3.1 Refactor `src/datasets/` so offline-prepared non-canonical bbox branches are consumed directly and the supported online bbox-format conversion path is removed for that workflow.
- [ ] 3.2 Add fail-fast training/config validation for mismatched dataset provenance versus `custom.bbox_format`, including train/val disagreement, slot-order disagreement, and canonical-xyxy inputs authored as non-canonical runs.
- [x] 3.3 Add regression tests covering the real failure mode: derived `cxcy_logw_logh` records render one-pass model-facing bbox slots only and never undergo a second runtime conversion.

## 4. Cache And Provenance Hardening

- [ ] 4.1 Update encoded-sample cache fingerprints and manifests to include prepared bbox-format branch identity, canonical source lineage, and conversion-version metadata.
- [ ] 4.2 Reject or bypass legacy/ambiguous caches for non-canonical bbox-format runs according to `training.encoded_sample_cache.ineligible_policy`, and add tests for both `error` and `bypass`.
- [ ] 4.3 Ensure run metadata and resolved config artifacts record the prepared bbox-format branch used by training so later audits can distinguish `xyxy` and derived `cxcy_logw_logh` runs.

## 5. Config, Docs, And Workflow Migration

- [x] 5.1 Update Stage-1 `cxcy_logw_logh` configs to point at offline-prepared derived branch JSONLs rather than relying on runtime conversion.
- [x] 5.2 Update `docs/data/PREPARATION.md`, `docs/data/CONTRACT.md`, and `docs/training/STAGE1_OBJECTIVE.md` to document the canonical `xyxy` source flow plus the separate offline bbox-format branch workflow.
- [ ] 5.3 Add a reproducible smoke workflow that names the canonical preset, the derived branch path, and the required `train.jsonl` / `val.jsonl` plus `train.coord.jsonl` / `val.coord.jsonl` artifacts needed before launching training.

## 6. Verification And Re-Run

- [ ] 6.1 Add an end-to-end public-data smoke test that runs canonical prep plus offline `cxcy_logw_logh` derivation and verifies the emitted branch manifest, `train.jsonl` / `val.jsonl`, and matching `train.coord.jsonl` / `val.coord.jsonl` outputs.
- [ ] 6.2 Re-run a targeted Stage-1 `cxcy_logw_logh` smoke configuration with an explicit config path, run name, seed, and output directory to confirm the new branch is consumed without runtime conversion.
- [ ] 6.3 Re-train and re-evaluate the `cxcy_logw_logh` experiment from the offline-prepared branch, then compare the new official eval bundle against the previous tainted run artifacts.
