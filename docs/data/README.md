# Data & Datasets

This page is a concise roadmap to CoordExp's data pipeline. It intentionally
stays "pointer-first": short, efficient, and aligned with the current codebase.

If you are onboarding a new dataset, follow the links in order and avoid ad-hoc
scripts/flags so runs stay reproducible and paper-ready.

**Source of truth**:
- Docs: [`JSONL_CONTRACT.md`](JSONL_CONTRACT.md), [`INTAKE_PIPELINE.md`](INTAKE_PIPELINE.md)
- Code: `src/datasets/`, `src/datasets/geometry.py`, `src/datasets/builders/jsonlines.py`

---

## Roadmap (Do This In Order)

1) **Pick / produce a JSONL that matches the contract**
   - Contract: [`JSONL_CONTRACT.md`](JSONL_CONTRACT.md)
2) **Prepare data offline (convert -> resize -> normalize -> tokens)**
   - Pipeline: [`INTAKE_PIPELINE.md`](INTAKE_PIPELINE.md)
3) **Train from YAML**
   - Start point: `configs/base.yaml`
   - Packing: [`PACKING.md`](PACKING.md)
4) **(Optional) Multi-dataset fusion**
   - Fusion: [`FUSION_DATASET.md`](FUSION_DATASET.md)

---

## Core Contract (What Training Assumes)

The dataset/trainer stack consumes JSONL records with:
- `images` (paths), `objects` (each has `desc` + exactly one geometry: `bbox_2d` or `poly`), `width`, `height`

See: [`JSONL_CONTRACT.md`](JSONL_CONTRACT.md).

### Coordinate Conventions (Critical)

CoordExp disables runtime normalization:
- You must set `custom.emit_norm: none` (enforced by config validation).
- Numeric coords that appear in JSON must already be **norm1000 integer values** in `0..999`.
- Coord-token mode is opt-in (`<|coord_k|>` where `k in [0,999]`) via `custom.coord_tokens.enabled: true`.

Why this is strict:
- Prevents silent scaling drift between datasets/runs.
- Keeps training/eval comparable across pipelines and makes debugging easier.

### Geometry Preservation (Why We Are Picky)

Geometry is treated as first-class data:
- Augmentations must update images and geometry atomically.
- Do not drop or reorder coordinates in custom code; use `src/datasets/geometry.py`.

---

## Minimal YAML Skeleton (Single Dataset)

Start from `configs/base.yaml` and override only what you need.

```yaml
data:
  dataset: ["dummy"]
  val_dataset: ["dummy"]  # keep eval_strategy stable in ms-swift

custom:
  user_prompt: <prompt_id>           # required (see src/config/prompts.py)
  train_jsonl: /path/to/train.jsonl
  val_jsonl: /path/to/val.jsonl
  emit_norm: none                    # required (runtime normalization is disabled)
  object_ordering: sorted            # sorted|random
  coord_tokens:
    enabled: false                   # true if JSONL uses <|coord_k|> strings
    skip_bbox_norm: true             # keep true if pre-normalized/tokenized
```

Notes:
- `data.dataset: ["dummy"]` is a placeholder to satisfy ms-swift argument validation; training data is loaded from `custom.train_jsonl`.
- Training encodes with `do_resize=False` (images should already be prepared offline).

---

## Implementation Pointers (Where This Lives In Code)

If you need to understand or change behavior, these are the first files to read:

- Dataset orchestration: `src/datasets/dense_caption.py`
- Message / assistant payload formatting: `src/datasets/builders/jsonlines.py`
- Geometry transforms + helpers: `src/datasets/geometry.py`
- Augmentation ops: `src/datasets/augmentation/`
- Preprocessors / validators: `src/datasets/preprocessors/`
- Config validation and invariants (including `custom.emit_norm`): `src/config/schema.py`

---

## Debugging / Verification (Fast, Reproducible)

Validate JSONL structure early (before training):

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/validate_jsonl.py <path.jsonl>
```

Inspect the exact rendered chat text + tokenization for one sample:

```bash
PYTHONPATH=. conda run -n ms python scripts/inspect_chat_template.py --jsonl <path.jsonl> --index 0
```

Smoke a training config (dumps template/prompt/debug artifacts):

```bash
PYTHONPATH=. conda run -n ms python -m src.sft --config <yaml> --debug
```

---

## Packing (Efficiency Lever)

Packing is the primary efficiency lever for long dense JSON outputs.

- Stage-1 / baseline packing: [`PACKING.md`](PACKING.md)
- Stage-2 packing is trainer-internal (post-rollout): [`../training/STAGE2_RUNBOOK.md`](../training/STAGE2_RUNBOOK.md)

---

## See Also

- Intake pipeline: [`INTAKE_PIPELINE.md`](INTAKE_PIPELINE.md)
- Coord objectives + adapters: [`../training/COORD_OBJECTIVE_AND_ADAPTER.md`](../training/COORD_OBJECTIVE_AND_ADAPTER.md)

---

**Last Updated**: 2026-02-01

