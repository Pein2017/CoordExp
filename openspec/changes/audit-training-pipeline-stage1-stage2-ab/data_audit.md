# Data Audit Notes (Stage-1 + Stage-2 AB)

This note captures the evidence-backed data ingestion + processing boundaries for the operational
entrypoints under audit:

- Stage-1: `scripts/train.sh` + `configs/stage1/ablation/geometry_first_coco80.yaml`
- Stage-2 AB: `scripts/train_stage2.sh` + `configs/stage2_ab/prod/ab_mixed.yaml`

It is intentionally written in “data -> transforms -> samples” order so future audits can quickly
locate the owner module for each policy decision.

## 1) Raw -> Cooked -> Training Samples (What Is Actually Consumed)

**Raw datasets** (COCO/LVIS/etc.) are converted offline under `public_data/` into *cooked JSONLs* plus
pre-rescaled images.

**Cooked JSONLs** are what training and eval consume. For the anchored configs:

- Stage-1 JSONL paths (COCO):
  - `custom.train_jsonl`: `public_data/coco/rescale_32_768_bbox_max60/train.coord.jsonl`
  - `custom.val_jsonl`: `public_data/coco/rescale_32_768_bbox_max60/val.coord.jsonl`
- Stage-2 JSONL paths (LVIS bbox-only):
  - `custom.train_jsonl`: `public_data/lvis/rescale_32_768_bbox_max60/train.bbox_only.max60.coord.jsonl`
  - `custom.val_jsonl`: `public_data/lvis/rescale_32_768_bbox_max60/val.bbox_only.max60.coord.jsonl`

**Final training samples** are produced on-the-fly by the learner dataset:
- Dataset: `src/datasets/dense_caption.py` (`BaseCaptionDataset` / alias `DenseCaptionDataset`)
- Builder: `src/datasets/builders/jsonlines.py` (`JSONLinesBuilder`)
- Tokenization/vision processing: upstream ms-swift Qwen3-VL template (`template.encode(...)`)

There is **no intermediate “cooking cache” layer** in the learner path today: the cooked JSONL is read
into memory once per dataset construction (a list of dicts), then encoded per-sample in `__getitem__`.

## 2) Ingestion Path (custom.train_jsonl -> Dataset)

**Config keys**
- `custom.train_jsonl`, `custom.val_jsonl` are resolved by config loading, but the learner ultimately
  reads them as raw strings passed to the dataset factory.

**JSONL load + image-path resolution**
- `BaseCaptionDataset.from_jsonl(...)` calls `load_jsonl(jsonl_path, resolve_relative=True)`:
  - `src/datasets/dense_caption.py:150`
  - which is re-exported from `src/common/io.py:13`
- When `resolve_relative=True`, `images: [<relpath>, ...]` entries are resolved against the JSONL’s
  parent directory and stored as absolute paths in-memory.

Implication:
- The learner does not require `ROOT_IMAGE_DIR` to be set for correct image resolution, because it
  carries absolute image paths after loading.

## 3) Cooked Record Schema Validation (Fail Fast)

At dataset construction time, each base record is validated before training begins:
- `src/datasets/dense_caption.py:98-105` calls `validate_conversation_record(...)`
- Contract helper: `src/datasets/contracts.py:43`

Validation policy for *raw dense-caption records* (records without `"messages"`):
- Require `images` key present and is a list of strings.
- Require `objects` key present and is a list (empty allowed).
- Require positive integer `width` and `height`.
- Validate each object’s geometry *shape* (exactly one of `bbox_2d` or `poly`, correct arity) and
  reject legacy keys (`bbox` / `polygon`).
- Validate `desc` is a string (emptiness is enforced downstream by the builder so error messages stay
  canonical).

Why this matters:
- It prevents “late failures” deep inside packing/rollout code by rejecting malformed JSONLs early,
  with actionable paths like `objects[17].bbox_2d`.

Edge-case policies (explicit):
- Empty `objects: []`: accepted (negative images are allowed).
- Empty `images: []`: accepted by contract validation (text-only records are possible), but **for
  grounding/detection runs it is strongly recommended that cooked JSONLs always include one image**
  per record to avoid silently training text-only examples.
- Missing `width/height`: hard error (required by contract and by `max_pixels` enforcement).

## 4) Geometry + Ordering Invariants (No Drops / No Reorder)

**Object ordering**
- When `custom.object_ordering: sorted` (default), `BaseCaptionDataset` asserts objects are already
  top-left sorted and fails fast if not:
  - `src/datasets/dense_caption.py:236-253`
- When `custom.object_ordering: random`, the dataset shuffles objects *as an explicit ablation*.

**Object payload field order**
- Controlled by `custom.object_field_order` (`desc_first` vs `geometry_first`) and enforced in the
  assistant serialization path:
  - `src/common/object_field_order.py` (payload construction)
  - `src/datasets/builders/jsonlines.py` (builder wiring)

**No runtime resizing**
- CoordExp forbids runtime resizing because it breaks grounding coordinates.
- Enforcement is two-part:
  1. Config hard cap: `configs/base.yaml` sets `template.max_pixels: 786432`.
  2. Learner hard error: `src/datasets/dense_caption.py:_enforce_max_pixels` raises if
     `width*height > template.max_pixels` (no silent rescale).

## 5) Final Training Sample Shape

Per-sample `__getitem__` flow:
- Copy record -> optional preprocess/augment -> enforce `max_pixels` -> optional object ordering ->
  coord token annotation -> build messages -> `template.encode(...)` -> attach metadata keys.

Owner modules:
- `src/datasets/dense_caption.py:395` (`__getitem__`)
- `src/datasets/builders/jsonlines.py:93` (`build_many`)

Outputs:
- The dataset returns a dict with at least `input_ids`, `labels`, and downstream metadata
  (`sample_id`, `dataset`, `base_idx`), plus `messages` and `assistant_payload` snapshots used by
  monitoring/debugging paths.

