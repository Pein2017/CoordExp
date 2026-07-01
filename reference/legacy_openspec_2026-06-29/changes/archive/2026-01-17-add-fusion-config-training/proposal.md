# Change: Add Fusion Config Training (Qwen3-VL-Compatible Schema)

## Why
CoordExp currently requires a single `custom.train_jsonl` / `custom.val_jsonl` pair, which forces users to pre-merge datasets offline. This blocks flexible experimentation (e.g., LVIS + Visual Genome + other public sources) and diverges from the proven multi-dataset fusion workflow in the upstream Qwen3-VL repository.

## What Changes
- Re-enable `custom.fusion_config` and wire it into `src/sft.py` so training can sample from multiple datasets without pre-merging JSONLs.
- Adopt a fusion-config schema compatible with Qwen3-VL containers (`targets:` + `sources:`), including `extends`-based inheritance.
- In CoordExp v1, treat `targets` and `sources` as the same kind of dataset entry (no target/source semantic split). Each entry contributes a per-epoch quota based on its own pool size and `ratio`.
- Build evaluation data from any dataset entry that provides a non-null `val_jsonl` path (entries without `val_jsonl` are skipped for eval).
- Keep the default single-dataset workflow unchanged when `custom.fusion_config` is not set.
- Validate `template` values in fusion configs (unknown template IDs raise a clear error) while keeping CoordExp's dense-caption encoding path.

## Non-Goals (This Change)
- Implementing new dataset converters (e.g., VG download/convert) beyond what already exists in `public_data/`.
- Changing the global JSONL data contract or prompt contract.
- Reproducing Qwen3-VL's target/source quota semantics (e.g., sources scaled relative to total target quota).
- Adding sampling-policy features like `sample_without_replacement` or a determinism guarantee; these are intentionally out of scope for this v1.

## Impact
- **Affected code**:
  - `src/config/schema.py` (un-deprecate `custom.fusion_config`, adjust validation)
  - `src/sft.py` (instantiate fusion dataset for train/eval when configured)
  - `src/datasets/fusion.py`, `src/datasets/fusion_types.py`, `src/datasets/unified_fusion_dataset.py` (fusion config + runtime mixing)
  - `src/datasets/wrappers/__init__.py` (add wrappers to support VG and generic JSONL sources)
- **Docs**:
  - Add a `docs/data/FUSION_DATASET.md` describing the fusion config format and how it differs from offline JSONL merging.
  - Update any existing docs that still claim fusion is disabled/deprecated.

## Risks / Mitigations
- **Behavior change risk**: previously configs with `custom.fusion_config` hard-failed; enabling fusion could mask user errors.
  - Mitigation: strict fusion-config validation (required fields, unique dataset names, empty pool errors).
- **Packing interaction**: packing is default in CoordExp; fused datasets change length distributions.
  - Mitigation: keep packing supported but add clear documentation and guardrails (recommended defaults + warnings).
