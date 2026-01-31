# Data JSONL Contract (Global)

This document defines the universal JSONL format consumed by all CoordExp training/eval datasets (public detection/grounding and any legacy sources). Every record MUST adhere to this contract so the shared chat-template pipeline can process all sources.

## Top-Level Record
- **Provenance**: Records are typically produced by dataset-specific converters (e.g., `public_data/scripts/convert_lvis.py`) and then optionally resized/tokenized via `public_data/scripts/rescale_jsonl.py` and `public_data/scripts/convert_to_coord_tokens.py` (see `PREPROCESSING.md`). Regardless of source, they MUST match this contract.
- `images` (list[str], required): Relative paths to image files; resolved against the JSONL directory.
- `objects` (list[object], required): Structured annotations (see below).
- `width` (int, required): Image width in pixels (original or post-resize if applied offline).
- `height` (int, required): Image height in pixels.
- `summary` (str, optional): Single-line English summary (if provided by the dataset). When present, it should be built from the raw `desc` strings; identical entries may be merged into `desc xN`. Missing objects or empty `desc` should fail during conversion.
- `metadata` (object, optional): Free-form metadata for provenance (not automatically injected).

## Objects
Each object MUST contain exactly one geometry field plus a non-empty `desc`.
- `desc` (str, required): Plain English description / class string (no hierarchy or slash prefixes required).
- One geometry (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]` pixel coordinates.
  - `poly`: flat list `[x1, y1, x2, y2, ...]` (even length, ≥6 values / ≥3 points). Optional `poly_points` (int) should equal `len(poly)/2` when present.
- No additional geometry fields are allowed on the same object.

Note: only `bbox_2d` and `poly` are supported in CoordExp; `line` geometries are rejected.

### Geometry keys and coordinate space (canonical)
- Accepted geometry keys are **only** `bbox_2d` or `poly` (plus optional `poly_points`). Legacy aliases `bbox` or `polygon` must be converted during preprocessing.
- Coordinate space is either pixel (floating point) or normalized coord tokens `<|coord_k|>` where `k ∈ [0, 999]`. The loader infers pixel vs normalized using image `width`/`height`. Avoid mixing pixel floats and coord tokens in the same object.
- When pre-tokenizing (`custom.coord_tokens.enabled: true`), keep geometry in tokens and set `custom.coord_tokens.skip_bbox_norm: true` to prevent double scaling.

## Invariants
- Coordinates can be pixel-space numbers or pre-tokenized `<|coord_k|>` values (0–999). Width/height must be present so pixel values can be reconstructed for losses.
- Image paths remain relative in JSONL; loaders resolve them to absolute paths.
- Geometry is validated; records with multiple geometry fields per object are rejected.
- Polygon vertices should be canonicalized offline for determinism:
  - drop duplicated closing point if present
  - order vertices clockwise around the centroid (angle sort)
  - rotate so the top-most (then left-most) vertex is first
  This matches the public-data converters (e.g., `public_data/scripts/convert_to_coord_tokens.py`) and the prompt spec.
- Optional fields (e.g., `summary`, `poly_points`, `metadata`) may be absent; templates and preprocessors must tolerate absence.
- **Coord-token mode (opt-in)**: When `custom.coord_tokens.enabled` is true, geometry may be pre-quantized as `<|coord_k|>` tokens (0–999). Set `custom.coord_tokens.skip_bbox_norm: true` to avoid double normalization when feeding tokenized records.

## Example
```json
{
  "images": ["images/0001.jpg"],
  "objects": [
    {"poly": ["<|coord_12|>", "<|coord_34|>", "<|coord_56|>", "<|coord_34|>", "<|coord_56|>", "<|coord_78|>", "<|coord_12|>", "<|coord_78|>"], "poly_points": 4, "desc": "yellow box"},
    {"bbox_2d": ["<|coord_100|>", "<|coord_120|>", "<|coord_180|>", "<|coord_200|>"], "desc": "tool cabinet"}
  ],
  "width": 768,
  "height": 512
}
```

## Current Sources (checked)
- `public_data/*`: LVIS/COCO/Objects365-style exports; polygons include `poly_points`; descriptions are English classes/phrases.
- `old_data/*`: legacy telecom samples that still follow this contract; descriptions may be Chinese but remain flat strings with a single geometry field.

All future domains MUST emit this contract to remain compatible with the shared chat template pipeline.

For an exact view of how a record plus the default prompts are rendered by the Qwen3-VL chat template, run:
```
python scripts/tools/inspect_chat_template.py --jsonl <path/to/data.jsonl> --index 0
```
