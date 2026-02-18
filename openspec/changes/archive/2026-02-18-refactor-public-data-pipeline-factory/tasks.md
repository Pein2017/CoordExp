## 1. Pipeline Core Scaffolding

- [x] 1.1 Add unified internal pipeline package under `public_data/` with explicit modules for adapter registry, stage interfaces, pipeline planner, and writer utilities.
- [x] 1.2 Define adapter base contract and registry/factory API for dataset ids (`lvis`, `coco`, `vg`) with fail-fast unknown-id handling.
- [x] 1.3 Define shared stage interface and deterministic stage-plan execution order contract.
- [x] 1.4 Add shared preset naming utility that applies optional suffix token `_max_<N>` when max-object filtering is enabled and deduplicates repeated suffixes.

## 2. Adapter Implementations and Migration

- [x] 2.1 Implement COCO adapter using existing download/convert semantics and route through shared stage pipeline.
- [x] 2.2 Implement LVIS adapter using existing conversion semantics and route through shared stage pipeline.
- [x] 2.3 Implement VG adapter using existing conversion semantics and route through shared stage pipeline.
- [x] 2.4 Ensure adapters own only source-ingestion behavior; remove/avoid duplicated shared transforms inside adapter code paths.

## 3. Shared Transform Stage Integration

- [x] 3.1 Integrate shared rescale stage using existing smart-resize behavior and verify geometry/path invariants are preserved.
- [x] 3.2 Integrate optional max-object stage (default off) using shared filtering behavior compatible with existing `filter_jsonl_max_objects.py`.
- [x] 3.3 Integrate explicit norm1000 numeric normalization stage and emit `<split>.norm.jsonl` artifacts as first-class outputs.
- [x] 3.4 Integrate coord-token stage as first-class output mode for train/val artifacts (from normalized numeric stage outputs).
- [x] 3.5 Integrate shared formatter/writer for relative image paths, deterministic output ordering, and consistent stats/manifests across `<split>.raw.jsonl`, `<split>.norm.jsonl`, and `<split>.coord.jsonl`.
- [x] 3.6 Integrate validation as a first-class optional planner stage (not only out-of-band checks), with fail-fast behavior when enabled.

## 4. Always-On Structural Preflight Checks

- [x] 4.1 Implement unconditional structural preflight checks before optional validation stage:
  - required top-level keys (`images`, `objects`, `width`, `height`)
  - exactly-one-geometry invariant (`bbox_2d` xor `poly`)
  - non-empty `desc`
  - geometry list type/arity sanity.
- [x] 4.2 Add tests proving structural preflight runs even when optional validation stage is disabled.
- [x] 4.3 Add tests proving structural preflight fails fast before downstream stages on invariant violations.

## 5. Runner and Plugin Compatibility

- [x] 5.1 Keep `public_data/run.sh` command grammar stable while switching internal execution to unified pipeline entrypoints.
- [x] 5.2 Keep `public_data/datasets/*.sh` external contract stable; ensure existing plugin invocations still work unchanged.
- [x] 5.3 Verify no new runner CLI flags are required for this refactor; preserve backward-compatible defaults.

## 6. Validation, Parity, and Reproducibility

- [x] 6.1 Add parity tests for COCO/LVIS/VG comparing legacy vs unified outputs on deterministic sample slices (record counts, object counts, ordering-sensitive checks).
- [x] 6.2 Add suffix-policy tests for max-object behavior:
  - default (off) has no suffix
  - enabled appends `_max_<N>`
  - repeated enablement does not duplicate suffix
  - legacy `max<N>` naming is recognized as equivalent to `_max_<N>` and does not fork artifact paths.
- [x] 6.3 Run contract validation on unified outputs with `public_data/scripts/validate_jsonl.py` (including coord-token artifacts).
- [x] 6.4 Run end-to-end runner smoke checks (`download/convert/rescale/coord/validate`) for at least one dataset path through unified internals.
- [x] 6.5 Record reproducibility checkpoints in artifacts/logs for each parity run:
  - dataset id,
  - effective preset name,
  - max_objects configuration,
  - output artifact paths and counts.
- [x] 6.6 Define and enforce parity gate pass/fail criteria before cutover:
  - pass requires split-level count parity, deterministic ordering parity, and validator pass on all artifact variants (`.raw.jsonl`, `.norm.jsonl`, `.coord.jsonl`)
  - fail triggers rollback to legacy path via migration-time operator procedure.
- [x] 6.7 Add rollback execution task that documents operator steps for reverting a dataset from unified path to legacy path when parity gate fails (migration-time), then remove runtime fallback toggles after full cutover.

## 7. Documentation and Cutover

- [x] 7.1 Update `public_data/README.md` and dataset READMEs to document unified internal pipeline architecture and stable external interfaces.
- [x] 7.2 Document optional max-object configuration behavior and suffix naming `max_{N}` in public-data docs.
- [x] 7.3 Remove deprecated internal duplicate paths once parity gates pass for all migrated datasets.
