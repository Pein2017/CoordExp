# Pixel-Painted Row Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Stage-1 teacher-forcing feature that paints already-emitted row-state coverage directly onto image pixels before visual encoding.

**Architecture:** Add a strict `pixel_painted_row_coverage` config next to, but mutually exclusive with, latent `row_conditioned_visual_coverage`. Reuse the existing row-state target-IR path, but route pixel-painted PIL images into chat messages before Swift template encoding. Keep loss computation unchanged: ordinary teacher-forced CE sees the same target IR and masks.

**Tech Stack:** Python, dataclasses, PIL/Pillow, existing CoordExp detection dataset/runtime/config modules, pytest.

---

## Scope Lock

Implement the approved design in:

- `docs/superpowers/specs/2026-06-08-pixel-painted-row-coverage-design.md`

Preserve these constraints:

- existing latent `row_conditioned_visual_coverage` behavior is unchanged;
- V1 is training-path first;
- rollout painter reuse is allowed, but full rollout wiring is deferred;
- marker is deterministic optical pixel paint, not a learned residual;
- marker preserves object evidence inside painted regions;
- canonical checked-in pixel-painted configs default to the all-layer LoRA
  comparison surface: visual blocks, projector/merger MLPs, language layers,
  and trainable coordinate token rows/`coord_offset_adapter`;
- no persistent painted-image cache;
- no new objective, loss, EOS forcing, duplicate-unlikelihood, or decode fallback;
- no packing support in V1.
- before full training, run a real Qwen/MS-Swift template encode preflight for
  one in-memory painted PIL image with `do_resize=false`; the unit suite only
  proves the local handoff contract.

## File Structure

Create:

- `src/detection/coverage/pixel_painting.py`
- `src/detection/coverage/pixel_row_state_dataset.py`
- `configs/stage1/detection_teacher_forcing/ablation/pixel_painted_row_coverage_random_sft_smoke.yaml`
- `configs/stage1/detection_teacher_forcing/ablation/pixel_painted_row_coverage_random_sft_ce_cont_from_random_pure_sft_all_layers_1epoch_eval512_8gpu.yaml`
- `tests/detection/coverage/test_pixel_config.py`
- `tests/detection/coverage/test_pixel_painting.py`
- `tests/detection/coverage/test_pixel_row_state_dataset.py`

Modify:

- `src/detection/coverage/config.py`
- `src/detection/coverage/__init__.py`
- `src/config/schema.py`
- `src/config/prompts.py`
- `src/detection/dataset.py`
- `src/detection/runtime.py`
- `tests/detection/coverage/test_config.py`
- `tests/test_teacher_forcing_config_contract.py`

## Task 1: Strict Pixel-Coverage Config

**Files:**
- Modify: `src/detection/coverage/config.py`
- Modify: `src/detection/coverage/__init__.py`
- Modify: `src/config/schema.py`
- Test: `tests/detection/coverage/test_pixel_config.py`
- Test: `tests/test_teacher_forcing_config_contract.py`

- [ ] **Step 1: Write failing config tests**

Add tests that assert:

```python
from src.detection.coverage.config import PixelPaintedRowCoverageConfig


def test_pixel_config_defaults_disabled():
    cfg = PixelPaintedRowCoverageConfig.from_mapping(None)
    assert cfg.enabled is False
    assert cfg.marker.style == "boundary_translucent_tint"
    assert cfg.marker.boundary_rgb == (0, 255, 255)
    assert cfg.marker.interior_rgb == (0, 255, 255)
    assert cfg.marker.interior_alpha == 0.18
    assert cfg.marker.boundary_width_px == 4
    assert cfg.prompt.marker_instruction is True


def test_pixel_config_accepts_minimal_v1_shape():
    cfg = PixelPaintedRowCoverageConfig.from_mapping(
        {
            "enabled": True,
            "version": "v1",
            "row_state_policy": "prefix_expansion",
            "marker": {
                "style": "boundary_translucent_tint",
                "boundary_rgb": [0, 255, 255],
                "interior_rgb": [0, 255, 255],
                "interior_alpha": 0.18,
                "boundary_width_px": 4,
            },
            "prompt": {"marker_instruction": True},
            "training": {
                "unpacked_only": True,
                "include_terminal_state": True,
            },
        }
    )
    assert cfg.enabled is True
    assert cfg.row_state_policy == "prefix_expansion"
```

Add rejection tests for:

```python
{"bad_key": 1}
{"marker": {"style": "hard_mask"}}
{"marker": {"boundary_rgb": [0, 255]}}
{"marker": {"boundary_rgb": [0, 256, 255]}}
{"marker": {"interior_alpha": 1.5}}
{"marker": {"boundary_width_px": 0}}
{"prompt": {"marker_instruction": "yes"}}
{"training": {"unpacked_only": False}}
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_config.py -q -p no:cacheprovider
```

Expected: import or unknown-class failures for `PixelPaintedRowCoverageConfig`.

- [ ] **Step 3: Implement strict config dataclasses**

In `src/detection/coverage/config.py`, add:

```python
_PIXEL_PATH = "pixel_painted_row_coverage"


@dataclass(frozen=True)
class PixelPaintMarkerConfig:
    style: str = "boundary_translucent_tint"
    boundary_rgb: tuple[int, int, int] = (0, 255, 255)
    interior_rgb: tuple[int, int, int] = (0, 255, 255)
    interior_alpha: float = 0.18
    boundary_width_px: int = 4


@dataclass(frozen=True)
class PixelPaintPromptConfig:
    marker_instruction: bool = True


@dataclass(frozen=True)
class PixelPaintTrainingConfig:
    unpacked_only: bool = True
    include_terminal_state: bool = True


@dataclass(frozen=True)
class PixelPaintedRowCoverageConfig:
    enabled: bool = False
    version: str = "v1"
    row_state_policy: str = "prefix_expansion"
    marker: PixelPaintMarkerConfig = field(default_factory=PixelPaintMarkerConfig)
    prompt: PixelPaintPromptConfig = field(default_factory=PixelPaintPromptConfig)
    training: PixelPaintTrainingConfig = field(default_factory=PixelPaintTrainingConfig)
```

Validate exact `version`, `row_state_policy`, fixed style, RGB triples of ints in `[0, 255]`, `0 <= interior_alpha <= 1`, positive integer boundary width, boolean prompt flag, and true training flags. Add `from_mapping()` using `parse_dataclass_strict`.

Export `PixelPaintedRowCoverageConfig` from `src/detection/coverage/__init__.py`.

- [ ] **Step 4: Wire detection schema**

In `src/config/schema.py`:

```python
from src.detection.coverage.config import (
    PixelPaintedRowCoverageConfig,
    RowCoverageConfig,
)
```

Add `"pixel_painted_row_coverage"` to `_DETECTION_OPTIONAL_SECTIONS`.

Parse the optional section in `DetectionTrainingConfig.from_mapping()`, store it on the dataclass, and include it in `to_mapping()`.

Add validation:

```python
if row_conditioned_visual_coverage.enabled and pixel_painted_row_coverage.enabled:
    raise ValueError(
        "pixel_painted_row_coverage is mutually exclusive with "
        "row_conditioned_visual_coverage"
    )
```

For enabled pixel coverage, require `objective.id=teacher_forcing`, `training.group_by_length=false`, `training.packing=false`, and `packing.static_packing=false`.

Let sorted `data.object_ordering` use the same teacher-forcing exception currently allowed for latent row coverage.

- [ ] **Step 5: Verify config tests pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_config.py tests/test_teacher_forcing_config_contract.py -q -p no:cacheprovider
```

Expected: pixel config tests pass and teacher-forcing config contract remains green.

## Task 2: Built-In Marker Prompt Instruction

**Files:**
- Modify: `src/config/prompts.py`
- Modify: `src/detection/runtime.py`
- Test: `tests/detection/coverage/test_pixel_config.py`

- [ ] **Step 1: Write failing prompt test**

Add a test that builds a detection config with:

```python
"pixel_painted_row_coverage": {
    "enabled": True,
    "prompt": {"marker_instruction": True},
}
```

Then assert the prompt used by `build_detection_dataset()` contains:

```text
marked regions correspond to objects already emitted
```

Add a second test with `marker_instruction: false` and assert the phrase is absent.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_config.py -q -p no:cacheprovider
```

Expected: missing prompt augmentation failure.

- [ ] **Step 3: Implement code-owned instruction**

In `src/config/prompts.py`, add:

```python
PIXEL_PAINTED_ROW_COVERAGE_SYSTEM_INSTRUCTION = (
    "Some regions may be marked with a visible boundary and light translucent "
    "tint. Marked regions correspond to objects already emitted in the "
    "assistant prefix. Continue enumerating remaining visible objects; marked "
    "pixels may still contain valid overlapping or contained objects, so do "
    "not treat marked regions as background."
)
```

Add:

```python
def append_pixel_painted_row_coverage_instruction(system_prompt: str | None) -> str:
    base = "" if system_prompt is None else str(system_prompt).rstrip()
    return f"{base}\n{PIXEL_PAINTED_ROW_COVERAGE_SYSTEM_INSTRUCTION}".strip()
```

In `src/detection/runtime.py`, when `pixel_painted_row_coverage.enabled` and `prompt.marker_instruction`, pass the augmented system prompt to `DetectionTrainingDataset.from_jsonl()`.

- [ ] **Step 4: Verify prompt tests pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_config.py -q -p no:cacheprovider
```

Expected: prompt instruction tests pass.

## Task 3: Non-Occluding Pixel Painter

**Files:**
- Create: `src/detection/coverage/pixel_painting.py`
- Test: `tests/detection/coverage/test_pixel_painting.py`

- [ ] **Step 1: Write failing painter tests**

Test:

```python
from PIL import Image

from src.detection.coverage.config import PixelPaintMarkerConfig
from src.detection.coverage.pixel_painting import paint_pixel_coverage_image
from src.detection.coverage.types import CoverageBoxNorm1000


def test_empty_coverage_returns_equivalent_rgb_copy():
    image = Image.new("RGB", (20, 20), (100, 80, 60))
    painted = paint_pixel_coverage_image(image, (), PixelPaintMarkerConfig())
    assert painted.mode == "RGB"
    assert painted.size == image.size
    assert painted.getpixel((10, 10)) == (100, 80, 60)
    assert painted is not image


def test_translucent_fill_preserves_object_evidence_and_marks_region():
    image = Image.new("RGB", (20, 20), (100, 80, 60))
    painted = paint_pixel_coverage_image(
        image,
        (CoverageBoxNorm1000(250, 250, 749, 749),),
        PixelPaintMarkerConfig(boundary_width_px=1, interior_alpha=0.18),
    )
    assert painted.getpixel((10, 10)) != (100, 80, 60)
    assert painted.getpixel((10, 10)) != (0, 255, 255)
    assert painted.getpixel((0, 0)) == (100, 80, 60)


def test_boundary_is_visibly_stronger_than_interior():
    image = Image.new("RGB", (20, 20), (100, 80, 60))
    painted = paint_pixel_coverage_image(
        image,
        (CoverageBoxNorm1000(250, 250, 749, 749),),
        PixelPaintMarkerConfig(boundary_width_px=2, interior_alpha=0.18),
    )
    assert painted.getpixel((5, 5)) == (0, 255, 255)
    assert painted.getpixel((10, 10)) != (0, 255, 255)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_painting.py -q -p no:cacheprovider
```

Expected: module import failure.

- [ ] **Step 3: Implement painter**

Create `src/detection/coverage/pixel_painting.py`:

```python
from PIL import Image, ImageDraw

from src.datasets.geometry import norm1000_xyxy_to_pixel_xyxy
from src.detection.coverage.config import PixelPaintMarkerConfig
from src.detection.coverage.types import CoverageBoxNorm1000


def paint_pixel_coverage_image(
    image: Image.Image,
    coverage_boxes: tuple[CoverageBoxNorm1000, ...],
    marker: PixelPaintMarkerConfig,
) -> Image.Image:
    base = image.convert("RGB")
    if not coverage_boxes:
        return base.copy()

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (*marker.interior_rgb, int(round(marker.interior_alpha * 255)))
    outline = marker.boundary_rgb

    for box in coverage_boxes:
        xyxy = norm1000_xyxy_to_pixel_xyxy(
            box.xyxy,
            width=base.width,
            height=base.height,
        )
        draw.rectangle(xyxy, fill=fill)
        draw.rectangle(xyxy, outline=outline, width=marker.boundary_width_px)

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
```

- [ ] **Step 4: Verify painter tests pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_painting.py -q -p no:cacheprovider
```

Expected: all pixel painter tests pass.

## Task 4: Training Row-State Dataset Wrapper

**Files:**
- Modify: `src/detection/dataset.py`
- Create: `src/detection/coverage/pixel_row_state_dataset.py`
- Modify: `src/detection/runtime.py`
- Test: `tests/detection/coverage/test_pixel_row_state_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Use the fake dataset helpers from `tests/detection/coverage/test_row_state_dataset.py`. Add tests that assert:

```python
dataset = PixelPaintedRowCoverageTrainingDataset(
    base_detection_dataset,
    config=PixelPaintedRowCoverageConfig(enabled=True),
)
sample0 = dataset[0]
sample1 = dataset[1]
assert sample0["pixel_painted_row_coverage"]["box_count"] == 0
assert sample1["pixel_painted_row_coverage"]["box_count"] == 1
assert sample1["row_coverage_state"].coverage_object_indices == (0,)
assert sample1["messages"][1]["content"][0]["type"] == "image"
assert not isinstance(sample1["messages"][1]["content"][0]["image"], str)
```

Assert `sample_id` still matches the row-state sidecar and target IR remains present.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_row_state_dataset.py -q -p no:cacheprovider
```

Expected: missing `PixelPaintedRowCoverageTrainingDataset`.

- [ ] **Step 3: Add image override seam**

In `DetectionTrainingDataset._messages()`, add an optional `image_payloads` keyword:

```python
def _messages(
    self,
    images: Sequence[str],
    *,
    assistant_text: str,
    image_payloads: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    if image_payloads is None:
        payloads = [self._resolve_image(image) for image in images]
    else:
        payloads = list(image_payloads)
        if len(payloads) != len(images):
            raise ValueError("image_payloads length must match images length")
    messages = build_detection_chat_messages(
        system_prompt=self.config.system_prompt,
        user_prompt=self.config.user_prompt,
        images=payloads,
        assistant_text=assistant_text,
    )
    return tuple(messages)
```

In `encode_teacher_forcing_row_state()`, add:

```python
image_payloads: Sequence[Any] | None = None
```

and pass it to `_messages(...)`.

- [ ] **Step 4: Implement wrapper**

Create `src/detection/coverage/pixel_row_state_dataset.py` with:

```python
from PIL import Image
from torch.utils.data import Dataset

from src.detection.coverage.config import PixelPaintedRowCoverageConfig
from src.detection.coverage.pixel_painting import paint_pixel_coverage_image
from src.detection.coverage.row_state_dataset import RowStateIndexEntry
from src.detection.dataset import DetectionTrainingDataset


class PixelPaintedRowCoverageTrainingDataset(Dataset):
    def __init__(
        self,
        base_dataset: DetectionTrainingDataset,
        *,
        config: PixelPaintedRowCoverageConfig,
    ) -> None:
        self.base_dataset = base_dataset
        self.config = config
        self._flat_index = self._build_flat_index()

    def __len__(self) -> int:
        return len(self._flat_index)

    def set_epoch(self, epoch: int) -> None:
        self.base_dataset.set_epoch(epoch)
        self._flat_index = self._build_flat_index()

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self._flat_index[int(index)]
        state = self.base_dataset.row_coverage_state_for_row(
            base_idx=entry.base_idx,
            row_state_k=entry.row_state_k,
        )
        image_path = self.base_dataset._resolve_scene_image_reference(
            self.base_dataset.scene_for_row(entry.base_idx).images
        )
        with Image.open(image_path) as raw_image:
            painted = paint_pixel_coverage_image(
                raw_image,
                tuple(state.coverage_boxes),
                self.config.marker,
            )
        encoded = dict(
            self.base_dataset.encode_teacher_forcing_row_state(
                base_idx=entry.base_idx,
                row_state_k=entry.row_state_k,
                image_payloads=(painted,),
            )
        )
        encoded["row_coverage_state"] = state
        encoded["pixel_painted_row_coverage"] = {
            "enabled": True,
            "style": self.config.marker.style,
            "box_count": len(state.coverage_boxes),
            "source_image": image_path,
        }
        return encoded
```

Implement `_build_flat_index()` equivalent to `RowCoverageTrainingDataset`.

- [ ] **Step 5: Wire runtime**

In `src/detection/runtime.py`, import `PixelPaintedRowCoverageTrainingDataset`. In `maybe_wrap_row_coverage_dataset()`, return:

```python
if training_config.pixel_painted_row_coverage.enabled:
    return PixelPaintedRowCoverageTrainingDataset(
        dataset,
        config=training_config.pixel_painted_row_coverage,
    )
```

Keep the latent row-coverage branch unchanged.

- [ ] **Step 6: Verify dataset tests pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/detection/coverage/test_pixel_row_state_dataset.py tests/detection/coverage/test_row_state_dataset.py -q -p no:cacheprovider
```

Expected: pixel wrapper tests pass and latent wrapper tests remain green.

## Task 5: Final Focused Verification

**Files:**
- All touched implementation and test files.

- [ ] **Step 1: Run coverage-related tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 env -u CODEX_CI python -m pytest tests/detection/coverage tests/test_teacher_forcing_config_contract.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Run schema import smoke**

Run:

```bash
python - <<'PY'
from src.config.schema import DetectionTrainingConfig
from src.detection.coverage.config import PixelPaintedRowCoverageConfig
print(PixelPaintedRowCoverageConfig.from_mapping({"enabled": True}).to_mapping() if hasattr(PixelPaintedRowCoverageConfig.from_mapping({"enabled": True}), "to_mapping") else PixelPaintedRowCoverageConfig.from_mapping({"enabled": True}))
print("ok")
PY
```

Expected: prints a config object and `ok`.

- [ ] **Step 3: Inspect dirty state**

Run:

```bash
rtk git status --short --branch
```

Expected: new pixel-painted files plus previously existing dirty row-coverage files. Do not stage unrelated files unless the user asks.

## Self-Review Notes

- The plan implements all approved design decisions: opt-in config, non-occluding pixel paint, in-memory images, prompt instruction, mutual exclusivity, training-first wrapper, and latent coverage preservation.
- The plan does not implement rollout painting. That is intentionally deferred by design.
- The plan does not add persistent painted-image caches, new losses, or default behavior changes.
