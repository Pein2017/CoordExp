# Public Data View Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate COCO public-data artifacts to image-store-relative, norm1000-integer annotation views and prove latest compact Stage-1 can train from the new `coco80/len-12000` view.

**Architecture:** Phase 1 introduces a COCO-local image store, canonical annotation views, view metadata, Git-tracked provenance manifests, and latest compact loader/rendering support for norm1000 integers. Phase 1 preserves all-proxy research metadata but defers calibrated hard-proxy training views to a separate proxy-confidence phase.

**Tech Stack:** Python 3.12, `conda run -n ms`, Qwen3-VL tokenizer/processor under `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp`, existing `public_data` scripts, latest compact detection stack under `src/detection/`, pytest.

---

## Scope And Gates

Do not start implementation until the user explicitly asks.

This plan implements Phase 1 from [the design doc](/data/CoordExp/docs/superpowers/specs/2026-05-17-public-data-view-architecture-design.md):

- prepare/adopt `public_data/coco/images/res-1024` without breaking the old
  runnable image root; any destructive `mv` requires a separate checkpoint;
- create `public_data/coco/views/coco80/full`;
- create `public_data/coco/views/coco80/len-12000`;
- create `public_data/coco/views/coco80/max-60`;
- create `public_data/coco/views/coco80-lvis-proxy/len-12000` as an all-proxy research view;
- update latest compact detection parsing/rendering for norm1000 integer views;
- deprecate runtime `max_objects` filtering;
- update docs and Git-tracked provenance manifests;
- run a real latest compact smoke on `coco80/len-12000`.

This plan does not implement:

- `proxy_confidence` calibration;
- `coco80-lvis-proxy-hard/len-12000`;
- all-proxy weighted loss;
- support-object rendering objectives;
- deletion of old sibling roots;
- production config migration.

Destructive cleanup of old roots is never implied by Phase 1 completion. It
requires a later user-approved cleanup after smoke, manifest/checksum updates,
rollback verification, and an agreed production-run acceptance signal.

## File Responsibility Map

Create:

- `public_data/view_contracts.py`: dataclasses and helpers for image-store metadata, view metadata, image-root resolution, JSON-safe path checks, and checksum-friendly view descriptions.
- `public_data/scripts/build_coco_views.py`: dedicated Phase 1 COCO view factory.
- `public_data/tests/test_view_contracts.py`: unit tests for metadata parsing, path resolution, and image-store consistency checks.
- `public_data/tests/test_build_coco_views.py`: unit tests for view factory behavior on tiny synthetic COCO-style rows.
- `tests/test_latest_detection_norm1000_view.py`: latest compact parser/rendering tests for norm1000 integer rows.
- `tests/test_detection_raw_schema_contract.py`: legacy coord-token parser compatibility tests that must continue to pass while production configs are not migrated.
- `tests/test_detection_training_dataset.py`: legacy dataset loading compatibility tests.
- `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml`: smoke config using the new view path.

Modify:

- `public_data/scripts/build_coco_length_budget_artifacts.py`: keep a compatibility wrapper and route reusable estimator/converter logic into `build_coco_views.py`.
- `public_data/scripts/convert_to_coord_tokens.py`: keep coord-token export as a legacy/debug surface and avoid presenting it as the canonical view writer.
- `public_data/pipeline/stages.py`: keep unified-runner defaults unchanged in Phase 1 except for narrowly scoped compatibility hooks or docs; full default pipeline migration is a later approved phase.
- `public_data/scripts/validate_jsonl.py`: add canonical view validation mode with image-store-relative path resolution and norm1000 integer enforcement.
- `src/detection/data.py`: parse norm1000 integer boxes instead of requiring coord-token boxes for latest compact canonical views.
- `src/detection/template.py`: render norm1000 integers as Qwen coord-token literals in compact assistant sequences.
- `src/detection/dataset.py`: infer image store from view metadata, accept explicit image-root override only when consistent, and remove object-count rejection.
- `src/config/schema.py`: make latest compact `data.image_root` optional and `data.max_objects` deprecated/ignored for the compatibility window.
- `src/sft.py`: pass resolved image-store root into latest compact dataset construction and prevent stale or `None`-derived `ROOT_IMAGE_DIR`.
- `tests/test_public_data_provenance_manifests.py`: accept image-store/view manifest paths and enforce JSONL checksum scope for views.
- `docs/data/CONTRACT.md`: update canonical view JSONL semantics to image-store-relative paths and norm1000 integer geometry.
- `docs/data/PREPARATION.md`: document the COCO view factory and length-budget view generation.
- `public_data/README.md`: document `images/res-1024` and `views/**` layout.
- `manifests/public_data_provenance/README.md`: document image-store and view manifest layout.

Generate during implementation:

- `public_data/coco/images/res-1024/meta.json`;
- `public_data/coco/views/coco80/full/{train,val}.jsonl`;
- `public_data/coco/views/coco80/full/source_comparison.json`;
- `public_data/coco/views/coco80/full/meta.json`;
- `public_data/coco/views/coco80/len-12000/{train,val}.jsonl`;
- `public_data/coco/views/coco80/len-12000/{train,val}.length_stats.json`;
- `public_data/coco/views/coco80/len-12000/source_comparison.json`;
- `public_data/coco/views/coco80/len-12000/meta.json`;
- `public_data/coco/views/coco80/max-60/{train,val}.jsonl`;
- `public_data/coco/views/coco80/max-60/source_comparison.json`;
- `public_data/coco/views/coco80/max-60/meta.json`;
- `public_data/coco/views/coco80-lvis-proxy/len-12000/{train,val}.jsonl`;
- `public_data/coco/views/coco80-lvis-proxy/len-12000/{train,val}.length_stats.json`;
- `public_data/coco/views/coco80-lvis-proxy/len-12000/source_comparison.json`;
- `public_data/coco/views/coco80-lvis-proxy/len-12000/meta.json`;
- `manifests/public_data_provenance/coco/images/res-1024.json`;
- `manifests/public_data_provenance/coco/views/coco80/full.json`;
- `manifests/public_data_provenance/coco/views/coco80/len-12000.json`;
- `manifests/public_data_provenance/coco/views/coco80/max-60.json`;
- `manifests/public_data_provenance/coco/views/coco80-lvis-proxy/len-12000.json`.

## Artifact Tracking Policy

Generated data under `public_data/coco/images/**` and
`public_data/coco/views/**` remains local artifact state and should stay ignored
by Git. Git-tracked outputs for Phase 1 are code, tests, docs, configs, and
provenance manifests under `manifests/public_data_provenance/**`.

Implementation verification must include:

```bash
git check-ignore -v \
  public_data/coco/images/res-1024/meta.json \
  public_data/coco/views/coco80/len-12000/train.jsonl
```

Expected: both generated public-data paths are ignored, while the new manifest
JSON files are not ignored and can be staged.

## Phase 1 Tasks

### Task 1: Add View And Image-Store Contract Helpers

**Files:**

- Create: `public_data/view_contracts.py`
- Create: `public_data/tests/test_view_contracts.py`

- [ ] **Step 1: Write failing tests for image-store-relative path resolution**

Add tests that create a temporary image store and view root:

```python
from pathlib import Path

import pytest

from public_data.view_contracts import (
    ImageStoreMetadata,
    ViewMetadata,
    resolve_view_image_root,
    resolve_image_path,
)


def test_resolve_image_path_is_image_store_relative(tmp_path: Path) -> None:
    repo_root = tmp_path
    image_store_ref = "public_data/coco/images/res-1024"
    image_store = repo_root / image_store_ref
    image_path = image_store / "images" / "train2017" / "000000000001.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake")

    view_root = tmp_path / "public_data" / "coco" / "views" / "coco80" / "len-12000"
    view_root.mkdir(parents=True)
    meta = ViewMetadata(
        schema_version=1,
        kind="annotation_view",
        dataset="coco",
        view="coco80/len-12000",
        image_store=image_store_ref,
        path_anchor="repo_root",
        image_path_semantics="image_store_relative",
        coordinate_space="norm1000",
        coordinate_storage="integer",
        coordinate_range=(0, 999),
        coordinate_chart="xyxy",
        assistant_coordinate_rendering="qwen_coord_tokens",
        primary_jsonl={"train": "train.jsonl", "val": "val.jsonl"},
        sample_policy={"type": "length_budget", "max_total_tokens": 12000},
        length_budget_scope={
            "rendered_families": ["objects"],
            "excluded_sidecars": ["metadata.supervision.support_objects"],
        },
        summary={"records": 1, "rendered_object_count": 1, "support_sidecar_count": 0},
    )

    resolved_root = resolve_view_image_root(meta, view_root=view_root, repo_root=repo_root)
    assert resolved_root == image_store.resolve()
    assert resolve_image_path("images/train2017/000000000001.jpg", image_root=resolved_root) == image_path.resolve()


def test_rejects_escaped_image_path(tmp_path: Path) -> None:
    image_store = tmp_path / "store"
    image_store.mkdir()

    with pytest.raises(ValueError, match="outside image_root"):
        resolve_image_path("../raw/images/leak.jpg", image_root=image_store)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_view_contracts.py -q
```

Expected: fails because `public_data.view_contracts` does not exist.

- [ ] **Step 3: Implement minimal metadata dataclasses and resolvers**

Implement:

```python
@dataclass(frozen=True)
class ImageStoreMetadata:
    schema_version: int
    kind: str
    dataset: str
    image_store: str
    image_path_semantics: str
    max_pixels: int
    visual_token_budget: int
    image_factor: int
    image_root: str
    splits: tuple[str, ...]


@dataclass(frozen=True)
class ViewMetadata:
    schema_version: int
    kind: str
    dataset: str
    view: str
    image_store: str
    path_anchor: str
    image_path_semantics: str
    coordinate_space: str
    coordinate_storage: str
    coordinate_range: tuple[int, int]
    coordinate_chart: str
    assistant_coordinate_rendering: str
    primary_jsonl: Mapping[str, str]
    sample_policy: Mapping[str, Any] | None = None
    length_budget_scope: Mapping[str, Any] | None = None
    annotation_policy: str | None = None
    parent_view: str | None = None
    proxy_policy: Mapping[str, Any] | None = None
    summary: Mapping[str, Any] | None = None
```

Functions:

- `load_view_metadata(path: Path) -> ViewMetadata`;
- `write_view_metadata(path: Path, metadata: Mapping[str, Any]) -> None`;
- `resolve_view_image_root(meta: ViewMetadata, view_root: Path, repo_root: Path | None = None) -> Path`;
- `resolve_image_path(image_ref: str, image_root: Path) -> Path`;
- `safe_relative_image_ref(path: str) -> Path`.

Validation rules:

- `image_path_semantics == "image_store_relative"`;
- committed `image_store` paths are repo-root-relative unless test/debug
  metadata explicitly marks them absolute;
- `coordinate_space == "norm1000"`;
- `coordinate_storage == "integer"`;
- `coordinate_range == (0, 999)`;
- image refs must be relative;
- image refs must not contain `..`;
- image refs must start with `images/`.

- [ ] **Step 4: Run contract helper tests**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_view_contracts.py -q
```

Expected: pass.

### Task 1A: Freeze View Metadata Schema Before Factory Work

**Files:**

- Modify: `public_data/view_contracts.py`
- Modify: `public_data/tests/test_view_contracts.py`

- [ ] **Step 1: Add failing tests for required metadata fields by artifact kind**

Tests must cover:

- image-store metadata requires `kind=image_store`, `dataset`, `image_store`,
  `image_path_semantics`, `max_pixels`, `visual_token_budget`, `image_factor`,
  `image_root`, and `splits`;
- base annotation views require `kind=annotation_view`, repo-root-relative
  `image_store`, `coordinate_space=norm1000`, `coordinate_storage=integer`,
  `coordinate_range=[0, 999]`, `primary_jsonl`, and `summary`;
- `len-*` views additionally require `sample_policy.type=length_budget`,
  `sample_policy.max_total_tokens`, `length_budget_scope.rendered_families`,
  `length_budget_template_id`, and length-stat filenames/checksums where
  available;
- `max-*` views require `sample_policy.type=max_objects_legacy` and the old
  object-count cap;
- all-proxy views require `annotation_policy`, `parent_view`, proxy source
  artifact references, and supervision summary fields.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_view_contracts.py -q
```

Expected: new schema assertions fail until contract helpers understand the full
metadata surface.

- [ ] **Step 3: Implement metadata validation helpers**

Add validation helpers that return actionable errors for missing or inconsistent
metadata fields. Keep the helpers local to `public_data/view_contracts.py` so
the factory, validator, and latest compact loader share one contract.

- [ ] **Step 4: Run contract tests**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_view_contracts.py -q
```

Expected: pass.

### Task 2: Add Norm1000 Integer Latest-Detection Parsing And Rendering Tests

**Files:**

- Create: `tests/test_latest_detection_norm1000_view.py`
- Modify: `tests/test_detection_raw_schema_contract.py`
- Modify: `tests/test_detection_training_dataset.py`
- Modify: `src/detection/data.py`
- Modify: `src/detection/template.py`

- [ ] **Step 1: Write failing parser test for integer boxes**

Add a test with a row:

```python
def _norm1000_row() -> dict:
    return {
        "images": ["images/val2017/000000000139.jpg"],
        "width": 1248,
        "height": 832,
        "image_id": 139,
        "file_name": "000000000139.jpg",
        "metadata": {"source": "coco2017", "split": "val"},
        "objects": [
            {
                "object_id": "coco:ann:1666628",
                "bbox_2d": [699, 284, 722, 336],
                "category_id": 85,
                "category_name": "clock",
                "coco_ann_id": 1666628,
                "desc": "clock",
            }
        ],
    }


def test_parse_raw_detection_row_accepts_norm1000_integer_box() -> None:
    raw = parse_raw_detection_row(_norm1000_row())
    obj = raw.objects[0]
    assert obj.bbox_2d.values == (699, 284, 722, 336)
    assert obj.object_id == "coco:ann:1666628"
```

- [ ] **Step 2: Write failing rendering test for coord-token assistant output**

Assert compact rendering still emits Qwen coord-token literals:

```python
def test_compact_template_renders_norm1000_ints_as_coord_tokens() -> None:
    raw = parse_raw_detection_row(_norm1000_row())
    sample = normalize_detection_row(raw, object_ordering=ObjectOrderingPlan.sorted())
    rendered = get_detection_template("compact_full").render_assistant(sample)
    assert "<|coord_699|>" in rendered.text
    assert "<|coord_336|>" in rendered.text
    assert "[699, 284, 722, 336]" not in rendered.text
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_detection_norm1000_view.py -q
```

Expected: fails because `src/detection/data.py` currently expects coord-token strings.

- [ ] **Step 4: Add legacy coord-token compatibility tests**

Because Phase 1 does not migrate production configs, legacy `.coord.jsonl`
latest-compact paths must keep working during the compatibility window.

Extend existing tests to assert:

- coord-token `bbox_2d` rows still parse for legacy artifacts;
- old latest compact production config paths still parse/load far enough to
  resolve data config without requiring view metadata;
- legacy `data.image_root` remains accepted for old configs;
- no new norm1000 canonical-view behavior breaks
  `tests/test_detection_raw_schema_contract.py` or
  `tests/test_detection_training_dataset.py`.

Run and confirm the compatibility expectation before parser changes:

```bash
conda run -n ms python -m pytest \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  -q
```

Expected: existing legacy tests pass before edits and must continue to pass
after edits.

- [ ] **Step 5: Update typed containers and rendering boundary**

Change `src/detection/data.py`:

- preserve legacy coord-token parsing for `.coord.jsonl` or non-view legacy rows;
- parse canonical view `bbox_2d` as four integers in `0..999`;
- use either a dual-format internal box abstraction or an explicit view-gated
  parser path so old production configs are not broken before production
  migration;
- require non-inverted `x1 <= x2` and `y1 <= y2`;
- keep `object_id` optional for legacy import but required for canonical view validation;
- preserve canonical `object_id` through `RawDetectionObject` and
  `NormalizedDetectionObject`;
- keep any generated historical id separate and clearly named, for example
  `legacy_object_instance_id`, so losses and span sidecars do not accidentally
  join by index-derived ids.

Change `src/detection/template.py`:

- add a helper that renders canonical norm1000 integer values as `<|coord_k|>`;
- keep legacy coord-token rows rendering equivalently;
- keep output text identical to the old compact coord-token rendering.

- [ ] **Step 6: Run focused detection and legacy compatibility tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_norm1000_view.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_detection_compact_full_template.py \
  tests/test_detection_template_ir_contract.py \
  -q
```

Expected: pass.

### Task 2A: Preserve Stable Object IDs Through Rendered Span Sidecars

**Files:**

- Modify: `src/detection/data.py`
- Modify: `src/detection/template.py`
- Modify: `src/detection/dataset.py`
- Create or extend: `tests/test_latest_detection_norm1000_view.py`
- Create or extend: `tests/test_latest_detection_view_metadata.py`

- [ ] **Step 1: Write failing all-proxy provenance tests**

Create a canonical all-proxy row with:

- one real COCO object;
- one renderable LVIS proxy candidate;
- one `metadata.supervision.support_objects` entry;
- `metadata.supervision.object_supervision` keyed by stable `object_id`;
- object order deliberately changed from the metadata insertion order.

Assert:

- every rendered coordinate/description span has the stable `object_id`;
- every rendered `object_id` joins to
  `metadata.supervision.object_supervision[object_id]`;
- generated legacy ids, if present, are not used as the supervision join key;
- Phase 1 `proxy_candidate` records have coordinate/regression weights omitted
  or set to `0.0`, and cannot be consumed as direct hard-bbox supervision;
- `support_objects` are not rendered by the default compact template;
- length stats count only rendered `objects`, not support sidecars.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_norm1000_view.py \
  tests/test_latest_detection_view_metadata.py \
  -q
```

Expected: fails because current render span events do not carry stable
source `object_id` and support-sidecar exclusion is not validated.

- [ ] **Step 3: Implement rendered-span provenance contract**

Required behavior:

- `RenderedObjectEntry` and `RenderSpanEvent` carry canonical `object_id`;
- span metadata includes `span_family`, field name, token start/end, source role,
  and relation snapshot from supervision metadata;
- dataset batch sidecars expose enough span/source metadata for future weighted
  loss without changing the default loss in Phase 1;
- support sidecars remain metadata-only unless a future template explicitly opts
  into rendering them.
- default latest-compact loss ignores `proxy_candidate` coordinate/regression
  spans unless a future approved policy promotes them to `proxy_hard` or a
  separate weighted-loss objective opts in.

- [ ] **Step 4: Run provenance tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_norm1000_view.py \
  tests/test_latest_detection_view_metadata.py \
  -q
```

Expected: pass.

### Task 3: Remove Runtime Object-Count Rejection From Latest Compact Dataset

**Files:**

- Modify: `src/config/schema.py`
- Modify: `src/detection/dataset.py`
- Modify: `tests/test_latest_detection_length_bucketing.py`
- Modify: `tests/test_prefix_rollin_ablation_launch_smoke.py`

- [ ] **Step 1: Add regression test that object count no longer rejects rows**

Add or update a test that builds a latest compact dataset row with more than 60 objects and asserts `encoded_length_for_row()` does not raise because of object count.

Use a synthetic row with 61 simple integer boxes and a temporary image file under an image store.

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_detection_length_bucketing.py -q
```

Expected: existing code raises `data.max_objects=60`.

- [ ] **Step 3: Make `data.max_objects` compatibility-only**

Change schema:

- `DetectionDataConfig.image_root: str | None = None`;
- `DetectionDataConfig.max_objects: int | None = None`;
- accepting `max_objects` emits a compatibility warning during config resolution or dataset build;
- no latest compact code uses it as an admission policy.

Change dataset:

- remove `if len(raw.objects) > self.config.max_objects` checks from `encoded_length_for_row()` and `__getitem__()`;
- remove `config.max_objects <= 0` constructor check;
- keep object-count metadata in `detection_metadata["object_count"]`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_length_bucketing.py \
  tests/test_prefix_rollin_ablation_launch_smoke.py \
  -q
```

Expected: pass after updating expected config fixtures.

### Task 4: Resolve Image Store From View Metadata In Latest Compact Dataset

**Files:**

- Modify: `src/detection/dataset.py`
- Modify: `src/sft.py`
- Create or extend: `tests/test_latest_detection_view_metadata.py`

- [ ] **Step 1: Write failing tests for metadata-derived image root**

Create a temp view:

```text
tmp/public_data/coco/images/res-1024/images/val2017/000000000139.jpg
tmp/public_data/coco/views/coco80/len-12000/meta.json
tmp/public_data/coco/views/coco80/len-12000/val.jsonl
```

Assert:

- dataset loads without explicit `image_root`;
- `data.image_root` override matching meta is accepted;
- mismatched override fails fast.
- when `data.image_root` is omitted, latest compact launch never derives
  `ROOT_IMAGE_DIR` from `None` and never resolves images relative to CWD.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest tests/test_latest_detection_view_metadata.py -q
```

Expected: fails because dataset currently requires explicit image root.

- [ ] **Step 3: Implement view-meta image root resolution**

In `DetectionTrainingDataset.from_jsonl()`:

- find `Path(jsonl_path).parent / "meta.json"`;
- if present, load with `public_data.view_contracts.load_view_metadata`;
- resolve `image_store`;
- if explicit `image_root` is provided, compare resolved paths and fail on mismatch;
- if no `meta.json`, allow explicit legacy `image_root`;
- if neither exists, fail with an actionable message.

In `src/sft.py` latest compact path:

- pass `image_root=None` when config omits it;
- preserve legacy explicit image-root behavior for old configs.
- resolve the view metadata image store before setting `ROOT_IMAGE_DIR`, or skip
  setting that environment variable for latest compact if absolute image paths
  are already passed into the processor/chat messages;
- add a test guard that `ROOT_IMAGE_DIR` is unset or equals the resolved
  image-store path, never a stale config value such as `/data/CoordExp/None`.

- [ ] **Step 4: Run metadata and smoke-launch tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_view_metadata.py \
  tests/test_prefix_rollin_ablation_launch_smoke.py \
  -q
```

Expected: pass.

### Task 4A: Guard Legacy Preprocessing Surfaces Without Migrating Unified Defaults

**Files:**

- Modify: `public_data/scripts/convert_to_coord_tokens.py`
- Modify: `public_data/pipeline/stages.py`
- Create or extend: relevant `public_data/tests/**` coverage for these entrypoints

- [ ] **Step 1: Write tests that preserve Phase 1 scope**

Tests must assert:

- the new COCO view factory is the canonical Phase 1 path for norm1000 view
  generation;
- `convert_to_coord_tokens.py` is labeled and routed as legacy/debug export, not
  as the canonical `views/**/{train,val}.jsonl` writer;
- `public_data/pipeline/stages.py` and `public_data/run.sh` defaults are not
  globally changed in Phase 1;
- no non-COCO or legacy public-data workflow is silently migrated before the
  dedicated COCO factory has passed smoke.

Coord-token export may remain available as an explicit legacy/debug conversion
mode with clear naming.

- [ ] **Step 2: Run tests and confirm failure**

Run the narrow public-data tests that cover these entrypoints. If no targeted
tests exist yet, create them beside the scripts and run:

```bash
conda run -n ms python -m pytest public_data/tests -q
```

Expected: fails only for the new labeling/scope assertions, not because the
unified runner defaults were changed.

- [ ] **Step 3: Implement scope guards and docs**

Required behavior:

- `build_coco_views.py` is documented as the Phase 1 canonical view writer;
- coord-token conversion help text and docs mark it legacy/debug;
- `public_data/pipeline/stages.py` does not switch global defaults in Phase 1;
- any hook added to shared pipeline code is backward-compatible and inert unless
  the new dedicated factory path opts in.

- [ ] **Step 4: Run public-data preprocessing tests**

Run:

```bash
conda run -n ms python -m pytest public_data/tests -q
```

Expected: pass for the targeted preprocessing and view-contract tests.

### Task 5: Build The COCO View Factory

**Files:**

- Create: `public_data/scripts/build_coco_views.py`
- Modify: `public_data/scripts/build_coco_length_budget_artifacts.py`
- Create: `public_data/tests/test_build_coco_views.py`

- [ ] **Step 1: Write tiny factory tests**

Test cases:

- `coco80/full` writes `train.jsonl` with norm1000 integer boxes and `images/...` refs;
- `coco80/len-12000` drops a synthetic over-budget row using a fake estimator;
- `coco80/max-60` preserves historical membership from an existing norm1000 source;
- `coco80-lvis-proxy/len-12000` records `metadata.supervision.object_supervision` keyed by `object_id`.

- [ ] **Step 2: Run factory tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_build_coco_views.py -q
```

Expected: fails because `build_coco_views.py` does not exist.

- [ ] **Step 3: Implement factory architecture**

Implement focused classes or functions:

- `CocoViewFactoryConfig`;
- `ImageStoreAdopter`;
- `Norm1000ViewWriter`;
- `LengthBudgetViewBuilder`;
- `LegacyMaxObjectsViewBuilder`;
- `AllProxyResearchViewBuilder`;
- `ViewStatsWriter`;
- `ViewManifestPayloadBuilder`.

Factory rules:

- prepare `public_data/coco/images/res-1024/images` without breaking
  `public_data/coco/rescale_32_1024_bbox/images` in Phase 1;
- require an explicit `--image-store-mode` with Phase 1 allowed values
  `copy`, `hardlink`, `reflink`, or `reuse-existing`;
- make `move` unavailable in Phase 1 except behind a separately approved
  destructive checkpoint;
- support `--dry-run` that validates paths and planned outputs without moving files;
- support `--dry-run-report path/to/report.json` that records source/target
  image counts, disk estimate, sample old-root and new-root resolution checks,
  and rollback notes;
- fail if target image store exists and is non-empty unless `--reuse-existing-image-store`;
- write JSONL with norm1000 integer coordinates;
- write view `meta.json`;
- do not delete old sibling roots.

- [ ] **Step 4: Reuse length estimator correctly**

Move or import the compact-full estimator from `build_coco_length_budget_artifacts.py` so length accounting still includes:

- post-merge image patch tokens;
- system prompt tokens;
- user/chat-template tokens;
- rendered compact assistant tokens.

The estimator must render norm1000 integer objects to coord-token assistant text before tokenization.

- [ ] **Step 5: Run factory tests**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_build_coco_views.py -q
```

Expected: pass.

### Task 6: Generate Phase 1 COCO Views And Metadata

**Files/Artifacts:**

- Generate: `public_data/coco/images/res-1024/meta.json`
- Generate: `public_data/coco/views/coco80/full/*`
- Generate: `public_data/coco/views/coco80/len-12000/*`
- Generate: `public_data/coco/views/coco80/max-60/*`
- Generate: `public_data/coco/views/coco80-lvis-proxy/len-12000/*`

- [ ] **Step 1: Dry-run the factory**

Run:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/build_coco_views.py \
  --model-path model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp \
  --source-preset public_data/coco/rescale_32_1024_bbox \
  --old-max60-dir public_data/coco/rescale_32_1024_bbox_max60 \
  --old-proxy-dir public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000 \
  --image-store public_data/coco/images/res-1024 \
  --output-root public_data/coco/views \
  --views coco80/full coco80/len-12000 coco80/max-60 coco80-lvis-proxy/len-12000 \
  --max-total-tokens 12000 \
  --splits train val \
  --image-store-mode hardlink \
  --dry-run-report temp/public_data_coco_views_dry_run.json \
  --dry-run
```

Expected: reports planned writes and image-store preflight details, with no file
changes. `hardlink` may be replaced by `copy`, `reflink`, or `reuse-existing` only after
checking local disk and target state. `move` is not a Phase 1 mode.

- [ ] **Step 2: Review image-store adoption policy before any real write**

Before running the real factory, capture and review:

- dry-run planned writes;
- source and target image counts;
- whether the target image store is empty or intentionally reused;
- available disk if using copy/reflink-style non-destructive population;
- old-root and new-root image resolution checks;
- rollback instructions.

If the chosen policy would move files out of the old root, stop and require a
separate explicit user approval before continuing. Phase 1's allowed real modes
are `copy`, `hardlink`, `reflink`, or `reuse-existing`.

- [ ] **Step 3: Run the real factory after the adoption policy is approved**

Run the same command without `--dry-run`, keeping an allowed non-destructive
mode, for example:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/build_coco_views.py \
  --model-path model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp \
  --source-preset public_data/coco/rescale_32_1024_bbox \
  --old-max60-dir public_data/coco/rescale_32_1024_bbox_max60 \
  --old-proxy-dir public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000 \
  --image-store public_data/coco/images/res-1024 \
  --output-root public_data/coco/views \
  --views coco80/full coco80/len-12000 coco80/max-60 coco80-lvis-proxy/len-12000 \
  --max-total-tokens 12000 \
  --splits train val \
  --image-store-mode hardlink \
  --dry-run-report temp/public_data_coco_views_dry_run.json
```

Expected:

- image store exists at `public_data/coco/images/res-1024`;
- all Phase 1 views exist;
- old sibling roots remain on disk and old JSONL image refs still resolve unless
  a separate destructive checkpoint was explicitly approved.

- [ ] **Step 4: Inspect generated stats**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
roots = [
    Path("public_data/coco/views/coco80/full"),
    Path("public_data/coco/views/coco80/len-12000"),
    Path("public_data/coco/views/coco80/max-60"),
    Path("public_data/coco/views/coco80-lvis-proxy/len-12000"),
]
for root in roots:
    print(root)
    print(json.dumps(json.loads((root / "meta.json").read_text())["summary"], indent=2, sort_keys=True))
PY
```

Expected: summaries show record counts, object counts, and policy ids.

### Task 6A: Compare Generated Views Against Source Artifacts

**Files/Artifacts:**

- Generate: `public_data/coco/views/coco80/full/source_comparison.json`
- Generate: `public_data/coco/views/coco80/len-12000/source_comparison.json`
- Generate: `public_data/coco/views/coco80/max-60/source_comparison.json`
- Generate: `public_data/coco/views/coco80-lvis-proxy/len-12000/source_comparison.json`

- [ ] **Step 1: Run deterministic source-membership comparisons**

Compare:

- `coco80/full` against `public_data/coco/rescale_32_1024_bbox`;
- `coco80/len-12000` against the previous length-budget artifact;
- `coco80/max-60` against `public_data/coco/rescale_32_1024_bbox_max60`;
- `coco80-lvis-proxy/len-12000` against
  `public_data/coco/rescale_32_1024_bbox_lvis_proxy_len12000`.

The comparison must check row counts, image ids, image refs, object counts,
length-budget inclusion/exclusion counts, and proxy-object counts where
applicable. It may allow intentional coordinate-storage changes
(`coord-token` -> norm1000 integer) and image-ref rebasing
(old-root-relative -> image-store-relative), but it must fail on unexpected
membership drift.

- [ ] **Step 2: Record comparison artifacts and metadata references**

Each generated `source_comparison.json` should include:

- source artifact path;
- generated view path;
- split-level counts;
- expected intentional deltas;
- unexpected deltas, empty on success;
- command and code version used to compare.

Reference these comparison artifacts from each view `meta.json` summary and from
the Git-tracked provenance manifest.

- [ ] **Step 3: Fail on unexpected drift**

If any comparison reports unexpected sample membership, image-id, or object-count
drift, stop before manifest finalization and inspect the generator source. Do not
paper over drift in provenance unless the user explicitly accepts the semantic
change.

### Task 7: Validate Canonical View JSONL And Length Budget

**Files:**

- Modify: `public_data/scripts/validate_jsonl.py`
- Create or extend: `public_data/tests/test_validate_jsonl.py`
- Create or extend: `public_data/tests/test_coco_length_budget_artifacts.py`

- [ ] **Step 1: Add validator tests for canonical views**

Tests must assert:

- coord-token strings are rejected in canonical view mode;
- pixel-space floats above `999` are rejected;
- coordinate value `1000` is rejected;
- norm1000 ints pass;
- `images[]` resolves through explicit `--image-root` or view `meta.json`.
- `support_objects` remain sidecars and are not accepted inside rendered
  `objects` unless the view policy explicitly declares them rendered.

- [ ] **Step 2: Run validator tests and confirm failure**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_validate_jsonl.py -q
```

Expected: new canonical-view validation tests fail until validator support is
implemented.

- [ ] **Step 3: Implement canonical view validation mode**

Add CLI options:

```text
--view-meta path/to/meta.json
--image-root path/to/image_store
--coordinate-storage integer
```

Rules:

- canonical views require `0 <= coord <= 999` integers;
- `images[]` must be image-store-relative;
- object `object_id` must exist for canonical views;
- if `metadata.supervision.object_supervision` exists, every rendered object id has a metadata entry.

- [ ] **Step 4: Run validator tests**

Run:

```bash
conda run -n ms python -m pytest public_data/tests/test_validate_jsonl.py -q
```

Expected: pass.

- [ ] **Step 5: Validate generated views**

Run:

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py \
  public_data/coco/views/coco80/len-12000/train.jsonl \
  --view-meta public_data/coco/views/coco80/len-12000/meta.json

PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py \
  public_data/coco/views/coco80-lvis-proxy/len-12000/train.jsonl \
  --view-meta public_data/coco/views/coco80-lvis-proxy/len-12000/meta.json
```

Expected: both pass.

### Task 8: Update Provenance Manifests

**Files:**

- Generate: `manifests/public_data_provenance/coco/images/res-1024.json`
- Generate: `manifests/public_data_provenance/coco/views/coco80/full.json`
- Generate: `manifests/public_data_provenance/coco/views/coco80/len-12000.json`
- Generate: `manifests/public_data_provenance/coco/views/coco80/max-60.json`
- Generate: `manifests/public_data_provenance/coco/views/coco80-lvis-proxy/len-12000.json`
- Modify: `manifests/public_data_provenance/schema.json`
- Modify: `manifests/public_data_provenance/README.md`
- Modify: `tests/test_public_data_provenance_manifests.py`

- [ ] **Step 1: Update provenance tests first and confirm nested manifests are discovered**

Change manifest discovery to recurse through nested paths, for example
`MANIFEST_ROOT.rglob("*.json")` excluding `schema.json`.

Add failing tests that:

- corrupt or remove a nested view checksum and expect the test to fail;
- allow `checksums: null` only for `key_params.kind=image_store`;
- require JSONL checksums for every `key_params.kind=annotation_view`.

Run:

```bash
conda run -n ms python -m pytest tests/test_public_data_provenance_manifests.py -q
```

Expected: new tests fail until nested manifest support and generated manifests
exist.

- [ ] **Step 2: Add manifest schema support for images and views**

Schema must allow:

- `relative_path` under `public_data/coco/images/...`;
- `relative_path` under `public_data/coco/views/...`;
- `key_params.kind` equal to `image_store` or `annotation_view`;
- JSONL checksums required for views;
- image-store manifests may use `checksums: null` by default.

- [ ] **Step 3: Generate view manifests with JSONL checksums**

Each view manifest must include:

- producer script and command;
- source views or source presets;
- tokenizer/template for length-budget views;
- `coordinate_space: norm1000`;
- `coordinate_storage: integer`;
- `image_path_semantics: image_store_relative`;
- `sample_policy`;
- `length_budget_scope` for `len-*` views;
- `rendered_object_count`;
- `support_sidecar_count`;
- `length_budget_template_id` for length-budget views;
- source-comparison artifact path and checksum;
- `checksums.scope: jsonl_training_samples_only`;
- `aggregate_sha256`;
- per-JSONL records and file hashes.

- [ ] **Step 4: Run manifest tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_public_data_provenance_manifests.py -q
```

Expected: pass.

### Task 9: Update Latest Compact Smoke Configs

**Files:**

- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml`
- Create: `configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_lvis_proxy_all_len12000_parse.yaml`

- [ ] **Step 1: Add base COCO80 length-budget smoke config**

The config should extend exactly:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_tiny.yaml
```

Override only data paths and the minimum run knobs needed to keep the smoke tiny:

```yaml
data:
  train_jsonl: public_data/coco/views/coco80/len-12000/train.jsonl
  val_jsonl: public_data/coco/views/coco80/len-12000/val.jsonl
```

Do not include `data.max_objects`. Do not include `data.image_root` unless the implementation intentionally keeps a temporary consistency-check override.

- [ ] **Step 2: Add all-proxy parse/span validation config**

The config should also extend exactly:

```text
configs/stage1/recursive_detection_ce_latest/smoke/compact_full_tiny.yaml
```

Use:

```yaml
data:
  train_jsonl: public_data/coco/views/coco80-lvis-proxy/len-12000/train.jsonl
  val_jsonl: public_data/coco/views/coco80-lvis-proxy/len-12000/val.jsonl
```

Set training limits so the config can parse/build one or a few batches without being used as the first weighted-loss production run.

Do not create a `coco80-lvis-proxy-hard` smoke config in Phase 1.

- [ ] **Step 3: Parse configs through the existing config loader**

Run:

```bash
PYTHONPATH=. conda run -n ms python - <<'PY'
from src.config.loader import ConfigLoader

for path in [
    "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml",
    "configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_lvis_proxy_all_len12000_parse.yaml",
    "configs/stage1/recursive_detection_ce_latest/prod/compact_full_support2.yaml",
]:
    cfg = ConfigLoader.load_materialized_training_config(path)
    print(path, type(cfg).__name__)
PY
```

Expected: new smoke configs resolve without `data.max_objects`, and the existing
production config still resolves through the legacy compatibility path because
production migration is out of Phase 1 scope.

### Task 10: Run Required Verification And Smoke

**Files/Artifacts:**

- Output: smoke run directory under `output/` chosen by the smoke config.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
conda run -n ms python -m pytest \
  public_data/tests/test_view_contracts.py \
  public_data/tests/test_build_coco_views.py \
  public_data/tests/test_validate_jsonl.py \
  public_data/tests/test_coco_length_budget_artifacts.py \
  tests/test_latest_detection_norm1000_view.py \
  tests/test_latest_detection_view_metadata.py \
  tests/test_detection_raw_schema_contract.py \
  tests/test_detection_training_dataset.py \
  tests/test_public_data_provenance_manifests.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run latest compact base smoke**

Run the smallest real smoke through `src/sft.py`:

```bash
PYTHONPATH=. conda run -n ms python src/sft.py \
  --config configs/stage1/recursive_detection_ce_latest/smoke/compact_full_coco80_len12000_tiny.yaml
```

Expected:

- the run starts through the real training entrypoint;
- it loads images through view metadata;
- it renders norm1000 integers as coord tokens;
- it completes the configured tiny number of steps;
- no object-count rejection occurs.

- [ ] **Step 3: Re-run all-proxy parse/span sidecar tests**

Re-run the dedicated tests added in Task 2A and confirm:

- every rendered proxy candidate span has a stable `object_id`;
- each rendered `object_id` joins to `metadata.supervision.object_supervision`;
- `support_objects` are not rendered by the default compact template;
- length stats count only rendered `objects`;
- no `coco80-lvis-proxy-hard` config or artifact was created in Phase 1.

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_latest_detection_norm1000_view.py \
  tests/test_latest_detection_view_metadata.py \
  -q
```

Expected: pass.

## Phase 1.5 Follow-Up Plan Boundary

Do not implement in Phase 1. Create a separate design/plan or extend this plan only after user approval.

Required artifacts:

- `public_data/coco/proxy_confidence/lvis_coco80_pair_metrics.jsonl`;
- `public_data/coco/proxy_confidence/policy_proxy_hard_bbox_v0.json`;
- `public_data/coco/proxy_confidence/reports/top_included.md`;
- `public_data/coco/proxy_confidence/reports/top_excluded.md`;
- `public_data/coco/proxy_confidence/reports/borderline_pairs.md`.

Rules:

- confidence is global category-pair-level over `(lvis_category_id, coco_category_id)`;
- policy decisions derive from train evidence;
- val is holdout diagnostics;
- semantic embedding evidence may use `model_cache/all-MiniLM-L6-v2-local` or a user-provided local sentence-embedding model;
- hard-bbox inclusion is primarily geometry/recovery/risk based.

## Phase 2 Follow-Up Plan Boundary

Do not implement in Phase 1. Generate after `proxy_confidence` exists.

Target artifact:

- `public_data/coco/views/coco80-lvis-proxy-hard/len-12000`.

Required gates:

- consume `policy_proxy_hard_bbox_v0.json`;
- record pair-metrics checksum in view metadata;
- validate final 12k budget after hard proxy inclusion;
- run latest compact hard-proxy smoke.

## Self-Review Checklist

- [x] The implementation plan does not require deleting old sibling roots.
- [x] The implementation plan does not migrate production configs in Phase 1.
- [x] The implementation plan does not implement weighted all-proxy loss in Phase 1.
- [x] The implementation plan does not store coord-token strings in canonical view JSONL.
- [x] The implementation plan requires a real latest compact smoke before completion.
- [x] The implementation plan updates Git-tracked provenance manifests.
- [x] The implementation plan keeps `proxy_confidence` and hard-proxy view generation out of Phase 1.

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-05-17-public-data-view-architecture.md`.

Execution must wait for explicit user instruction.

When execution is approved, choose one:

1. **Subagent-Driven (recommended)**: dispatch a fresh worker per task, review between tasks, and keep Phase 1.5/2 out of scope.
2. **Inline Execution**: execute tasks in this session using `superpowers:executing-plans`, with checkpoints after contract helpers, latest compact parser changes, view factory generation, and smoke verification.
