## 1. OpenSpec Updates (Contracts First)

P0 ordering: this section must land before any code changes (specs drive contracts; code follows).

- [x] 1.1 Update `openspec/specs/public-data-pipeline/spec.md`:
  - define the single canonical suffix `_max{N}` and remove `_max_{N}` equivalence rules
  - define canonical preset artifact filenames (pixel-space `<split>.jsonl`)
  - define derived preset image sharing behavior (no byte-copy; derived `images/` via hardlinks to base preset; fail-fast if hardlink cannot be created)
  - define immutability/overwrite semantics for preset `images/` and rebuild behavior when rescale params change
  - define how rescale-param mismatches are detected via `pipeline_manifest.json` (what keys must be recorded; what comparison logic is authoritative)
  - define `PUBLIC_DATA_MAX_OBJECTS` as a supported `public_data/run.sh` interface (max-object filtering) and specify conflict behavior (fail-fast if multiple sources disagree)
- [x] 1.2 Update `openspec/specs/public-data-adapter-factory/spec.md`:
  - update writer artifact naming contract to match canonical layout
  - remove “legacy runner mapping” requirements that imply duplicated raw/alias artifacts

## 2. Canonical Naming + Layout (Remove Legacy)

- [x] 2.1 Remove legacy max-suffix reuse logic in `public_data/pipeline/naming.py`; enforce the single canonical suffix.
- [x] 2.2 Remove `LegacyAliasStage` from `public_data/pipeline/stages.py` and the planner stage lists in `public_data/pipeline/planner.py`.
- [x] 2.3 Update `public_data/pipeline/writer.py`:
  - emit `<split>.jsonl` as the pixel-space preset JSONL
  - remove `legacy_raw_alias` and `ensure_legacy_raw_alias`
  - update pipeline manifest accordingly
- [x] 2.4 Update `public_data/run.sh` and `public_data/scripts/run_pipeline_factory.py` to match the canonical filenames and naming rules.
- [x] 2.5 Remove runner-side preset pre-resolution in `public_data/run.sh` (drop `resolve_effective_preset_name` usage) so the pipeline planner is the single source of truth for `effective_preset`.
- [x] 2.6 Delete `public_data/scripts/resolve_effective_preset.py` (and any remaining call sites) if it becomes unused after runner/pipeline unification.

## 3. Derived Presets Without Byte-Copy

- [x] 3.1 Replace the image-tree copy behavior in `public_data/pipeline/planner.py::_copy_images_if_needed` with hardlink materialization (and remove `_images_tree_needs_materialization` heuristics that treat `st_nlink > 1` as invalid).
- [x] 3.2 Implement derived-preset `images/` materialization via hardlinks to the base preset’s resized images (precheck same-filesystem; fail-fast on hardlink error; no byte-copy fallback).
- [x] 3.3 Add a validation check that derived preset `images/` is a real directory (not a symlink) and spot-check that opened image sizes match JSONL meta.

## 4. Validator Consolidation

- [x] 4.1 Extend `public_data/scripts/validate_jsonl.py` CLI to accept:
  - size-contract expectations (max_pixels + multiple-of),
  - image check controls (`none|exists|open` + spot-check N),
  - the “rescale presets must not have symlinked images/” safety policy that previously lived in a dedicated preflight validator path.
  - deterministic spot-check sampling (first N structurally-parsed JSONL records in line order; no RNG; structurally-parsed excludes image IO checks; do not skip/replace failing sampled records)
- [x] 4.2 Route training scripts to call `public_data/scripts/validate_jsonl.py` (with explicit expected max_pixels + multiple-of) and remove direct calls to the legacy dedicated preflight validator.
- [x] 4.3 Delete the legacy dedicated max-pixels preflight validator wrapper (or keep as a thin wrapper that calls the shared validator, then delete in a follow-up).

## 5. Tests + Verification

- [x] 5.1 Update `public_data/tests/test_pipeline_factory.py` for the new naming/layout behavior.
- [x] 5.2 Add regression coverage for derived preset image sharing:
  - derived `images/` is a real directory (not symlink),
  - derived images are hardlinks (and failure to hardlink is a hard error),
  - validators spot-check that opened image sizes match JSONL meta.
- [x] 5.3 Run focused checks:
  - `conda run -n ms ruff check public_data/pipeline public_data/scripts scripts/tools scripts`
  - `conda run -n ms python -m py_compile public_data/pipeline/*.py public_data/scripts/*.py scripts/tools/*.py`
  - `conda run -n ms pytest -q public_data/tests/test_pipeline_factory.py`

## 6. Repo-Wide Contract Updates (Docs + Call Sites)

- [x] 6.1 Update documentation to remove `.raw.jsonl` as the canonical pixel-space preset artifact:
  - `public_data/README.md`
  - `public_data/*/README.md`
  - `docs/data/INTAKE_PIPELINE.md`
  - `public_data/scripts/README.md`
- [x] 6.2 Update tests and scripts that assume `.raw.jsonl` paths (beyond the pipeline factory tests):
  - `tests/test_public_data_pipeline_parity.py`
  - `tests/test_public_data_runner_smoke.py`
  - `public_data/scripts/lvis_full_pipeline.sh` and any export scripts that treat `*.raw.jsonl` as the stable on-disk name
- [x] 6.3 Add an inventory gate to prevent drift:
  - `rg -n '\\\\.raw\\\\.jsonl' configs public_data scripts tests docs openspec/specs` should match only explicitly-justified legacy fixtures (ideally none).
  - `rg -n '_max_' configs public_data scripts tests docs openspec/specs` should match only explicitly-justified legacy fixtures (ideally none).

## 7. Migration Notes (User-Facing)

- [x] 7.1 Write a concrete migration note at `openspec/changes/public-data-pipeline-dedup/migration.md`:
  - old → new mapping table (`_max_60` → `_max60`, `.raw.jsonl` removal),
  - how to detect stragglers (exact `rg` commands and expected zero matches),
  - how to rebuild safely without in-place overwrite (fresh preset name or deliberate full delete).
