## Refactor Plan for CoordExp

### To-Do Tracker (incremental)
- [x] Add shared common modules for types/helpers (`src/common/{schemas,geometry,io}.py`) with behavior-preserving re-exports.
- [x] Point `datasets/utils.py` and `datasets/fusion.py` at shared `load_jsonl` to remove duplicate loaders; keep backward compatibility.
- [ ] Introduce geometry key normalization shim and adopt it in datasets/eval without changing outputs.
- [ ] Standardize imports to `src.*` across datasets/eval/collators for clarity and circular-safety.
- [ ] Extract shared dataset label/segment helper for collators/metrics; document required batch fields.
- [x] Align docs (`DATA_JSONL_CONTRACT.md`) and config comments with canonical geometry keys and CoordSpace enum.
- [ ] Add alias/legacy schema tests (bbox/polygon -> bbox_2d/poly) and geometry round-trip tests.

### Current State (audit highlights)
- Legacy/deprecated code paths still live alongside active pipeline, e.g., hard-sample mining config and callback plus fusion dataset kept for archival but not wired.  
```619:628:src/config/schema.py
@dataclass(frozen=True)
class HardSampleMiningConfig:
    """Deprecated configuration placeholder for hard sample mining."""
    enabled: bool = False
    start_epoch: int = 0
```
```1:5:src/callbacks/hard_sample_mining.py
"""Deprecated hard-sample mining callback (not used).

This module remains for archival/reference only. The training pipeline no longer
wires HardSampleMiningCallback or its config. Do not enable in new runs.
"""
```
```1:9:src/datasets/unified_fusion_dataset.py
"""Unified fusion dataset that concatenates JSONL files and uses a single template.

Deprecated: fusion-based training is currently disabled while we focus on
single-source LVIS runs. This implementation is retained for potential future
use but is not wired into the active pipeline.
"""
```
- Duplicate type/config definitions for coord offsets live in both config schema and runtime adapter, inviting drift.  
```96:103:src/config/schema.py
@dataclass(frozen=True)
class CoordOffsetConfig:
    enabled: bool = False
    ids: tuple[int, ...] = ()
    embed_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 0.0
```
```21:28:src/coord_tokens/offset_adapter.py
@dataclass
class CoordOffsetConfig:
    enabled: bool = False
    ids: Sequence[int] = ()
    embed_lr: float | None = None
    head_lr: float | None = None
    weight_decay: float = 0.0
```
- Naming is mostly snake_case, but geometric fields mix `bbox_2d`/`poly`/`line` while configs and evaluators also use `bbox`/`polygon` variants; “coord offset” types share names across layers.
- Types are strong in `config/schema.py`, weaker in data/eval paths (heavy `Any` in augmentation, parsing, collators); dataset contracts are only lightly typed.
- Module layout mixes active and deprecated flows in the same namespaces; coord vocabulary, geometry, evaluation, and callbacks are scattered, making dependencies implicit.

### Key Improvement Areas
- Consolidate and clearly separate active vs deprecated modules; avoid importing deprecated code on the hot path.
- Unify coord-offset and coord-token type definitions and naming across config, adapters, datasets, and eval to prevent drift.
- Standardize geometry schemas (object fields, normalization flags) and align dataset ↔ builder ↔ evaluator naming.
- Strengthen type surface in data loaders, builders, augmentation, and evaluation utilities; reduce `Any`.
- Clarify imports to minimize hidden coupling (e.g., callbacks pulling config classes) and to prevent future circulars.

### Phased Refactoring Strategy (backward compatible)
1) **Proposal & scoping (OpenSpec)**: Create a refactor change (e.g., `refactor-coordexp-structure`) covering naming, type contracts, and module moves; ensure Qwen3-VL compatibility and pipeline behavior are explicitly non-breaking.  
2) **Baseline conventions**: Define project-wide naming (snake_case, geometry keys `bbox_2d|poly|line`, coord vocab terms `coord_token`, `coord_offset`), and schema contracts for samples/configs. Publish in `docs/` + `src/README`.  
3) **Type+schema unification**:  
   - Collapse duplicated coord offset configs into one shared dataclass, re-exported where needed.  
   - Promote dataset/eval contracts to TypedDict/dataclasses and reuse across builders, collators, and eval parsing.  
4) **Module reorg (non-breaking paths first)**:  
   - Move deprecated fusion/hard-sample mining into `src/legacy/` (imports kept behind feature flags for compatibility).  
   - Group coord-specific runtime pieces under `src/coord_tokens/` (codec, validator, adapter, loss) with a single config surface.  
   - Isolate geometry helpers (`datasets/geometry.py`) as the canonical source; have augmentation/eval import from it.  
5) **Consistency & cleanup**:  
   - Normalize builder/evaluator naming for geometry and coord tokens; ensure parsing/serialization agree.  
   - Simplify imports so callbacks depend on stable interfaces, not deep config internals.  
6) **Validation**:  
   - Run existing tests (`tests/test_detection_eval.py`, packing/token-type tests) plus targeted smoke runs of `python -m src.sft` with coord tokens and without.  
   - Add a small contract test to assert sample schema (objects + coords) and coord-offset wiring.

### Action Checklist (detailed)
- [ ] Draft OpenSpec change `refactor-coordexp-structure` with scope, risks, and migration notes (no behavior change intent).  
- [ ] Codify naming/style guide in `docs/` and surface in `src/README`.  
- [ ] Merge coord offset configs: pick one definition, re-export for config parsing and runtime adapter; update references in `sft.py` and optimizer wiring.  
- [ ] Formalize geometry/record schema (TypedDict or dataclasses) and reuse in `datasets/contracts.py`, builders, collators, eval parsing.  
- [ ] Mark/relocate deprecated modules (`fusion.py`, `unified_fusion_dataset.py`, `hard_sample_mining.py`) under `legacy/` with shim imports for compatibility; remove default imports on hot paths.  
- [ ] Align coord-token handling across dataset building, validator, and eval parsing (single normalize/denorm helpers, consistent key names).  
- [ ] Tighten type hints in augmentation/apply_augmentations, dataset builders, collators, and eval to reduce `Any` and catch shape/field mismatches early.  
- [ ] Audit imports for optional features; guard deprecated callbacks and fusion helpers behind explicit flags to avoid accidental load.  
- [ ] Add validation/tests: schema contract test, coord-offset hook test, eval parsing round-trip (coord tokens ↔ pixels), and a smoke SFT run with/without packing.  
- [ ] Update docs/config examples to the new naming/schema and ensure Qwen3-VL compatibility notes remain.

This plan keeps the current training/eval pipeline intact while clarifying ownership of coord vocab, geometry handling, and detection evaluation, setting us up for consistent, paper-ready refactors without breaking Qwen3-VL integration.
