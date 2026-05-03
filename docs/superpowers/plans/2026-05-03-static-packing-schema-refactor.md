# Static Packing Schema Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each code slice and verification-before-completion before claiming completion. Use Serena for Python symbol inspection/editing when available, and `rtk conda run -n ms python -m pytest ...` for noisy tests.

**Goal:** Add typed `StaticPackingPlan` and `StaticPackingManifest` wrappers around the existing static-packing raw plan/cache contract while preserving all current JSON artifact keys, checksums, cache filenames, and public dataset attributes.

**Decision Source:** Task 8 in `docs/superpowers/plans/2026-05-03-type-schema-refactor.md` selected:

```markdown
- Static packing decision: implement typed `StaticPackingPlan` / `StaticPackingManifest` in this branch because raw plans cross module boundaries.
```

**Scope Guard:** This plan owns the follow-up implementation only. The Task 8 decision gate did not modify static-packing code, tests, or operator docs.

## Target Files

Modify:

- `src/datasets/wrappers/packed_caption.py`
- `tests/test_packing_wrapper.py`

Inspect only unless test failures prove a contract mismatch:

- `src/sft.py`
- `tests/test_stage1_static_packing_runtime_config.py`
- `docs/data/PACKING.md`

Do not modify:

- upstream HF model files,
- cache artifact key names,
- static-packing filenames such as `lengths.json`, `plan_ws*_drop*.json`, or `INDEX.json`,
- SFT CLI/config surfaces.

## Existing Boundary Evidence

- `src/datasets/wrappers/packed_caption.py:544` builds `raw_plan` as `list[list[int]]`.
- `src/datasets/wrappers/packed_caption.py:613` converts `raw_plan` into DDP-aligned `aligned_plan`.
- `src/datasets/wrappers/packed_caption.py:644` reads plan-cache JSON and returns `dict[str, Any]`.
- `src/datasets/wrappers/packed_caption.py:702` persists plan-cache JSON with raw and aligned plan lists, checksums, DDP alignment fields, and stats.
- `src/datasets/wrappers/packed_caption.py:769` persists setup `INDEX.json` as an ad hoc mapping.
- `src/datasets/wrappers/packed_caption.py:1070` exposes `StaticPackedCaptionDataset.raw_plan`, `.pack_plan`, checksums, DDP alignment fields, and stats.
- `src/sft.py:2589` constructs the static packed train dataset, then `src/sft.py:2620` logs `dataset.raw_plan`, `dataset.raw_plan_checksum`, `dataset.aligned_plan_checksum`, DDP alignment, and stats.
- `src/sft.py:3200` constructs the static packed eval dataset, then `src/sft.py:3227` logs the same plan/checksum fields.

## Representation Target

Add frozen dataclasses near the existing static-packing helpers in `src/datasets/wrappers/packed_caption.py`:

```python
@dataclass(frozen=True)
class StaticPackingPlan:
    raw_plan: tuple[tuple[int, ...], ...]
    aligned_plan: tuple[tuple[int, ...], ...]
    raw_plan_checksum: str
    aligned_plan_checksum: str
    world_size: int
    dataloader_drop_last: bool
    pad_needed: int
    repeated_pack_indices: tuple[int, ...]
    single_long: int
    skipped_long: int
    avg_fill: float

    @classmethod
    def from_parts(
        cls,
        *,
        raw_plan: Sequence[Sequence[int]],
        aligned_plan: Sequence[Sequence[int]],
        world_size: int,
        dataloader_drop_last: bool,
        pad_needed: int,
        repeated_pack_indices: Sequence[int],
        single_long: int,
        skipped_long: int,
        avg_fill: float,
        raw_plan_checksum: str | None = None,
        aligned_plan_checksum: str | None = None,
    ) -> "StaticPackingPlan": ...

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StaticPackingPlan": ...

    def to_mapping(self, *, fingerprint: Mapping[str, Any]) -> dict[str, Any]: ...
```

```python
@dataclass(frozen=True)
class StaticPackingManifest:
    version: int
    fingerprint: dict[str, Any]
    plan: StaticPackingPlan

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        path: Path,
        expected_fingerprint: Mapping[str, Any],
        world_size: int,
        dataloader_drop_last: bool,
    ) -> "StaticPackingManifest": ...

    def to_mapping(self) -> dict[str, Any]: ...
```

Implementation requirements:

- Store plans internally as tuples for immutability.
- Preserve public `StaticPackedCaptionDataset.raw_plan` and `.pack_plan` as `list[list[int]]` for compatibility with existing tests and downstream code.
- Preserve plan-cache JSON keys: `version`, `fingerprint`, `world_size`, `dataloader_drop_last`, `raw_plan`, `aligned_plan`, `raw_plan_checksum`, `aligned_plan_checksum`, `pad_needed`, `repeated_pack_indices`, `single_long`, `skipped_long`, `avg_fill`.
- Keep `_stable_plan_checksum()` behavior unchanged.
- Keep `_validate_or_initialize_setup_index()` key names unchanged; do not type `INDEX.json` unless this can be done without expanding the slice.

## Test-First Steps

- [ ] **Step 1: Add plan dataclass normalization tests**

Add to `tests/test_packing_wrapper.py`:

```python
def test_static_packing_plan_normalizes_and_roundtrips_payload() -> None:
    from src.datasets.wrappers.packed_caption import StaticPackingPlan

    plan = StaticPackingPlan.from_parts(
        raw_plan=[[0, 2], [1]],
        aligned_plan=[[0, 2], [1], [0, 2]],
        world_size=3,
        dataloader_drop_last=False,
        pad_needed=1,
        repeated_pack_indices=[0],
        single_long=0,
        skipped_long=0,
        avg_fill=0.75,
    )

    payload = plan.to_mapping(fingerprint={"cache_schema": "test"})

    assert payload["raw_plan"] == [[0, 2], [1]]
    assert payload["aligned_plan"] == [[0, 2], [1], [0, 2]]
    assert payload["raw_plan_checksum"] == plan.raw_plan_checksum
    assert payload["aligned_plan_checksum"] == plan.aligned_plan_checksum
    assert StaticPackingPlan.from_mapping(payload) == plan
```

Expected red: import fails because `StaticPackingPlan` does not exist.

- [ ] **Step 2: Add manifest validation tests**

Add to `tests/test_packing_wrapper.py`:

```python
def test_static_packing_manifest_rejects_malformed_plan_payload(tmp_path: Path) -> None:
    from src.datasets.wrappers.packed_caption import StaticPackingManifest

    payload = {
        "version": 2,
        "fingerprint": {"cache_schema": "test"},
        "world_size": 1,
        "dataloader_drop_last": False,
        "raw_plan": "not-a-plan",
        "aligned_plan": [[0]],
        "raw_plan_checksum": "bad",
        "aligned_plan_checksum": "bad",
        "pad_needed": 0,
        "repeated_pack_indices": [],
        "single_long": 0,
        "skipped_long": 0,
        "avg_fill": 1.0,
    }

    with pytest.raises(TypeError, match="raw_plan"):
        StaticPackingManifest.from_mapping(
            payload,
            path=tmp_path / "plan.json",
            expected_fingerprint={"cache_schema": "test"},
            world_size=1,
            dataloader_drop_last=False,
        )
```

Expected red: import fails because `StaticPackingManifest` does not exist.

- [ ] **Step 3: Add production-cache roundtrip assertion**

Extend an existing cache test such as `test_static_packing_deterministic_plan` or `test_static_packing_regenerates_stale_plan_cache_version` with a focused artifact parse:

```python
from src.datasets.wrappers.packed_caption import StaticPackingManifest

plan_cache = next(cache_dir.rglob("plan_ws*_drop*.json"))
payload = json.loads(plan_cache.read_text(encoding="utf-8"))
manifest = StaticPackingManifest.from_mapping(
    payload,
    path=plan_cache,
    expected_fingerprint=payload["fingerprint"],
    world_size=payload["world_size"],
    dataloader_drop_last=payload["dataloader_drop_last"],
)

assert manifest.to_mapping() == payload
```

Expected red until the manifest wrapper exists.

## Implementation Steps

- [ ] **Step 4: Implement `StaticPackingPlan`**

Add the dataclass and helpers in `src/datasets/wrappers/packed_caption.py` near `_stable_plan_checksum()` and `_build_raw_pack_plan()`. Keep validation small:

- plan fields must be sequences of non-empty integer sequences,
- checksum fields are computed when absent,
- `pad_needed`, `single_long`, and `skipped_long` are non-negative integers,
- `repeated_pack_indices` is a sequence of integers,
- `avg_fill` is finite.

- [ ] **Step 5: Implement `StaticPackingManifest`**

Route current `_read_plan_cache()` validation through `StaticPackingManifest.from_mapping()` while preserving current error messages where tests assert them. Let the manifest wrapper own:

- `_PLAN_CACHE_VERSION` check,
- fingerprint mapping check and `_validate_cache_fingerprint()` call,
- `world_size` and `dataloader_drop_last` checks,
- `StaticPackingPlan.from_mapping()` parsing.

- [ ] **Step 6: Route plan persistence through typed wrappers**

Change `_persist_plan_cache()` to build `StaticPackingPlan.from_parts(...)` and write `StaticPackingManifest(...).to_mapping()`.

Keep function signatures stable unless the implementation can narrow internally without changing callers.

- [ ] **Step 7: Route dataset construction through typed plan**

At the `build_static_packed_dataset()` read path, replace local raw field extraction with:

```python
plan_manifest = _read_plan_cache(...)
plan = plan_manifest.plan
```

Then pass `plan.raw_plan`, `plan.aligned_plan`, checksums, alignment fields, and stats into `StaticPackedCaptionDataset`. If changing `_read_plan_cache()` return type would create too much churn, use a new private `_read_plan_manifest()` and leave `_read_plan_cache()` as a compatibility shim inside the module.

- [ ] **Step 8: Preserve public dataset attributes**

Do not force downstream code to consume the dataclass yet. `StaticPackedCaptionDataset` should still expose:

- `raw_plan`
- `pack_plan`
- `raw_plan_checksum`
- `aligned_plan_checksum`
- `world_size`
- `dataloader_drop_last`
- `pad_needed`
- `repeated_pack_indices`
- `avg_fill`
- `single_long`
- `skipped_long`

## Verification

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_packing_wrapper.py \
  tests/test_stage1_static_packing_runtime_config.py \
  -q
```

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_packing_cache_fingerprints.py \
  tests/test_packing_template_contracts.py \
  tests/test_sft_preparation_contract.py \
  -q
```

Run:

```bash
git diff --check
```

Expected:

- static-packing tests pass,
- adjacent packing/SFT contract tests pass,
- no whitespace errors,
- plan-cache artifact payloads remain byte-shape compatible at the JSON key level.

## Commit Boundary

Use a focused commit after verification:

```bash
git add src/datasets/wrappers/packed_caption.py tests/test_packing_wrapper.py docs/superpowers/plans/2026-05-03-static-packing-schema-refactor.md
git commit -m "refactor(packing): type static packing cache plan"
```
