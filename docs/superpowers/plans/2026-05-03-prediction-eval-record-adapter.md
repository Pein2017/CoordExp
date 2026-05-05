# Prediction And Evaluation Record Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each code slice and verification-before-completion before claiming completion. Use Serena for Python symbol inspection/editing when available, and `rtk conda run -n ms python -m pytest tests/test_detection_eval_ingestion_diagnostics.py tests/test_detection_eval_output_parity.py tests/test_confidence_postop.py tests/test_proxy_eval_views.py -q` for noisy tests.

**Goal:** Add one canonical read adapter for prediction and evaluation JSONL records so `pred`, legacy `predictions`, and GT `objects` are read consistently without renaming serialized artifact keys.

**Decision Source:** Task 9 in `docs/superpowers/plans/2026-05-03-type-schema-refactor.md` selected:

```markdown
- Prediction/eval decision: implement a canonical read adapter before metric changes; the adapter will preserve serialized record mappings while standardizing read access for `pred`, `predictions`, and `objects`.
```

**Scope Guard:** This plan owns the follow-up implementation only. The Task 9 decision gate did not modify production eval or infer code.

## Target Files

Create:

- `src/eval/prediction_records.py`

Modify:

- `src/eval/detection.py`
- `src/eval/confidence_postop.py`
- `src/eval/artifacts.py`
- `src/eval/proxy_views.py`
- `tests/test_detection_eval_ingestion_diagnostics.py`
- `tests/test_detection_eval_output_parity.py`
- `tests/test_confidence_postop.py`
- `tests/test_proxy_eval_views.py`

Inspect only:

- `src/infer/engine.py`
- `src/infer/pipeline.py`
- `src/eval/proxy_eval_bundle.py`
- `tests/test_unified_infer_pipeline.py`
- `tests/test_proxy_eval_bundle.py`

Do not modify:

- inference artifact filenames,
- `metrics.json` payload keys,
- `gt_vs_pred.jsonl` canonical writer keys,
- `gt_vs_pred_scored.jsonl` canonical writer keys,
- upstream HF model files.

## Existing Boundary Evidence

- `src/infer/engine.py:12` documents canonical `gt_vs_pred.jsonl` lines with `gt`, `pred`, and `raw_output_json`.
- `src/infer/engine.py:439` reads GT from source JSONL `objects` or `gt`.
- `src/infer/engine.py:1586` validates source JSONL `objects`.
- `src/infer/engine.py:1751` writes canonical inference artifact predictions under `pred`.
- `src/eval/detection.py:902` builds inline GT records from `gt` or `objects`.
- `src/eval/detection.py:1046` prepares GT objects from `gt` or `objects`.
- `src/eval/detection.py:1140` reads predictions from `pred`, then legacy `predictions`.
- `src/eval/detection.py:2491` reads duplicate-control objects from `pred`, then legacy `predictions`.
- `src/eval/detection.py:2661` applies duplicate-control output to `pred`, then legacy `predictions`.
- `src/eval/confidence_postop.py:591` reads only `pred`.
- `src/eval/confidence_postop.py:1013` builds scored records from only `pred`.
- `src/eval/artifacts.py:65` adds constant scores from only `pred`.
- `src/eval/proxy_views.py:24` reads GT from `gt`, then `objects`, while preserving prediction fields.
- `src/eval/proxy_eval_bundle.py:134` materializes proxy views from `gt_vs_pred_scored.jsonl` before per-view evaluation.
- `src/eval/artifacts.py:97` writes `metrics.json` as a dynamic metrics/counters artifact, not a prediction-record alias.
- `tests/test_proxy_eval_views.py:62` asserts proxy filtering preserves `pred`.
- `tests/test_confidence_postop.py:313` asserts scored confidence artifacts use `pred`.
- `tests/test_detection_eval_output_parity.py:256` asserts duplicate-control guarded artifacts use `pred`.

## Representation Target

Add `src/eval/prediction_records.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RecordObjectSequence:
    key: str
    objects: tuple

    @property
    def legacy_key_used(self) -> bool:
        return self.key in {"predictions", "objects"}

    def as_list(self) -> list[dict[str, Any]]:
        return [dict(obj) for obj in self.objects]


def _objects_from_value(value: Any) -> tuple | None:
    if not isinstance(value, list):
        return None
    out: list[Mapping[str, Any]] = []
    for obj in value:
        if isinstance(obj, Mapping):
            out.append(obj)
    return tuple(out)


def prediction_objects_from_record(record: Mapping[str, Any]) -> RecordObjectSequence:
    pred_objects = _objects_from_value(record.get("pred"))
    if pred_objects is not None:
        return RecordObjectSequence(key="pred", objects=pred_objects)

    legacy_objects = _objects_from_value(record.get("predictions"))
    if legacy_objects is not None:
        return RecordObjectSequence(key="predictions", objects=legacy_objects)

    return RecordObjectSequence(key="pred", objects=())


def ground_truth_objects_from_record(record: Mapping[str, Any]) -> RecordObjectSequence:
    gt_objects = _objects_from_value(record.get("gt"))
    if gt_objects is not None:
        return RecordObjectSequence(key="gt", objects=gt_objects)

    objects = _objects_from_value(record.get("objects"))
    if objects is not None:
        return RecordObjectSequence(key="objects", objects=objects)

    return RecordObjectSequence(key="gt", objects=())


def write_prediction_objects_to_record(
    record: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    *,
    key: str | None = None,
) -> dict[str, Any]:
    sequence = prediction_objects_from_record(record)
    output_key = key or sequence.key
    if output_key not in {"pred", "predictions"}:
        raise ValueError(f"unsupported prediction object key: {output_key!r}")
    out = dict(record)
    out[output_key] = [dict(obj) for obj in objects]
    return out
```

Implementation requirements:

- `prediction_objects_from_record()` reads `pred` first and legacy `predictions` second.
- `ground_truth_objects_from_record()` reads `gt` first and legacy/source `objects` second.
- Adapter functions must not mutate the input mapping.
- Adapter functions must skip non-mapping list entries when returning object sequences.
- Existing inference writers keep writing canonical `pred`.
- Confidence and constant-score writers keep writing canonical `pred` for newly materialized scored artifacts.
- Duplicate-control guarded output preserves the input prediction key through `write_prediction_objects_to_record(record, kept_objects)`.

## Test-First Steps

- [ ] **Step 1: Add adapter unit tests**

Add to `tests/test_detection_eval_ingestion_diagnostics.py`:

```python
def test_prediction_record_adapter_reads_canonical_and_legacy_prediction_keys() -> None:
    from src.eval.prediction_records import prediction_objects_from_record

    canonical = prediction_objects_from_record(
        {
            "pred": [{"desc": "canonical"}],
            "predictions": [{"desc": "legacy"}],
        }
    )
    legacy = prediction_objects_from_record(
        {
            "predictions": [{"desc": "legacy"}],
        }
    )

    assert canonical.key == "pred"
    assert canonical.as_list() == [{"desc": "canonical"}]
    assert legacy.key == "predictions"
    assert legacy.as_list() == [{"desc": "legacy"}]
```

Add to `tests/test_detection_eval_ingestion_diagnostics.py`:

```python
def test_prediction_record_adapter_reads_canonical_and_legacy_gt_keys() -> None:
    from src.eval.prediction_records import ground_truth_objects_from_record

    canonical = ground_truth_objects_from_record(
        {
            "gt": [{"desc": "canonical"}],
            "objects": [{"desc": "legacy"}],
        }
    )
    legacy = ground_truth_objects_from_record(
        {
            "objects": [{"desc": "legacy"}],
        }
    )

    assert canonical.key == "gt"
    assert canonical.as_list() == [{"desc": "canonical"}]
    assert legacy.key == "objects"
    assert legacy.as_list() == [{"desc": "legacy"}]
```

Expected red: imports fail because `src.eval.prediction_records` does not exist.

- [ ] **Step 2: Add eval and duplicate-control adapter tests**

Add to `tests/test_detection_eval_output_parity.py`:

```python
def test_prepare_pred_objects_reads_legacy_predictions_key() -> None:
    record = _one_record(image="img.png")
    record["predictions"] = record.pop("pred")

    counters = EvalCounters()
    preds, invalid = _prepare_pred_objects(
        record,
        width=64,
        height=48,
        options=EvalOptions(metrics="f1ish"),
        counters=counters,
    )

    assert invalid == []
    assert counters.empty_pred == 0
    assert [pred["desc"] for pred in preds] == ["box"]
```

Add to `tests/test_detection_eval_output_parity.py`:

```python
def test_duplicate_control_preserves_legacy_predictions_key() -> None:
    record = _one_record(image="img.png")
    record["predictions"] = record.pop("pred")
    record["predictions"] = [
        {
            "type": "bbox_2d",
            "points": [0, 0, 63, 47],
            "desc": "box",
            "score": 0.95,
        },
        {
            "type": "bbox_2d",
            "points": [1, 1, 62, 46],
            "desc": "box",
            "score": 0.75,
        },
    ]

    guarded_records, report = _apply_offline_duplicate_control([record])

    assert "pred" not in guarded_records[0]
    assert [obj["score"] for obj in guarded_records[0]["predictions"]] == [0.95]
    assert report["total_predictions_inspected"] == 2
    assert report["total_predictions_suppressed"] == 1
```

Expected red before implementation: the first test may already pass through ad hoc eval fallback, while the second protects key-preservation after routing through the adapter.

- [ ] **Step 3: Add confidence and scoring adapter tests**

Add to `tests/test_confidence_postop.py`:

```python
def test_build_scored_record_reads_legacy_predictions_key() -> None:
    from src.eval.confidence_postop import _build_scored_record

    record = _bbox_record(
        image="legacy.png",
        x1=10,
        y1=20,
        x2=30,
        y2=40,
        raw_output_json=_bbox_raw(10, 20, 30, 40),
    )
    record["predictions"] = record.pop("pred")

    scored = _build_scored_record(
        record=record,
        confidence_objects=[
            {
                "object_idx": 0,
                "kept": True,
                "score": 0.7,
                "confidence": 0.7,
                "confidence_details": {"failure_reason": None},
            }
        ],
    )

    assert scored["pred"][0]["score"] == pytest.approx(0.7)
    assert scored["pred_score_source"] == PRED_SCORE_SOURCE
    assert scored["pred_score_version"] == PRED_SCORE_VERSION
```

Add to `tests/test_detection_eval_output_parity.py`:

```python
def test_with_constant_scores_reads_legacy_predictions_key() -> None:
    from src.eval.artifacts import with_constant_scores

    record = _one_record(image="legacy.png")
    record["predictions"] = record.pop("pred")

    scored = with_constant_scores(
        records=[record],
        pred_score_source="constant_test",
        pred_score_version=7,
        constant_score=0.25,
    )

    assert scored[0]["pred_score_source"] == "constant_test"
    assert scored[0]["pred_score_version"] == 7
    assert scored[0]["pred"][0]["score"] == pytest.approx(0.25)
```

Expected red before implementation: both tests fail because confidence post-op and constant scoring currently read only `pred`.

- [ ] **Step 4: Add proxy GT adapter test**

Add to `tests/test_proxy_eval_views.py`:

```python
def test_filter_proxy_record_reads_legacy_objects_key() -> None:
    record = _record()
    record["objects"] = record.pop("gt")

    filtered = filter_proxy_record(record, view="coco_real")

    assert "gt" not in filtered
    assert [obj["desc"] for obj in filtered["objects"]] == ["clock"]
    assert filtered["metadata"]["coordexp_proxy_eval_view"]["gt_key"] == "objects"
```

Expected red before implementation: this may already pass through local `_gt_key`; it guards the adapter-preserving behavior after refactor.

## Implementation Steps

- [ ] **Step 5: Add `src/eval/prediction_records.py`**

Create the module exactly from the Representation Target block.

- [ ] **Step 6: Route eval preparation through the adapter**

In `src/eval/detection.py`, import:

```python
from src.eval.prediction_records import (
    ground_truth_objects_from_record,
    prediction_objects_from_record,
    write_prediction_objects_to_record,
)
```

Replace GT reads in `preds_to_gt_records()` and `_prepare_gt_record()` with:

```python
raw_gt = ground_truth_objects_from_record(rec).as_list()
```

Replace prediction reads in `_prepare_pred_objects()` with:

```python
objs_raw = prediction_objects_from_record(record).as_list()
```

Replace prediction-key detection in `_duplicate_control_objects_for_record()` with:

```python
prediction_sequence = prediction_objects_from_record(record)
raw_predictions = prediction_sequence.as_list()
pred_key = prediction_sequence.key
if not raw_predictions:
    return [], set()
```

Replace prediction-key detection and guarded assignment in `_apply_offline_duplicate_control()` with:

```python
prediction_sequence = prediction_objects_from_record(guarded_record)
raw_predictions = prediction_sequence.as_list()
pred_key = prediction_sequence.key
if not raw_predictions:
    guarded_records.append(guarded_record)
    empty_result = apply_duplicate_policy(
        anchor_objects=(),
        explorer_objects_by_view=(),
        config=config,
        support_iou_threshold=_OFFLINE_DUPLICATE_CONTROL_SUPPORT_IOU_THRESHOLD,
    )
    results_by_record.append((empty_result, set()))
    continue
```

Use this assignment after computing `kept_indices`:

```python
guarded_record = write_prediction_objects_to_record(
    guarded_record,
    [
        obj
        for pred_index, obj in enumerate(raw_predictions)
        if pred_index not in controlled_indices or pred_index in kept_indices
    ],
    key=pred_key,
)
```

- [ ] **Step 7: Route confidence post-op through the adapter**

In `src/eval/confidence_postop.py`, import:

```python
from src.eval.prediction_records import prediction_objects_from_record
```

Replace each direct `record.get("pred")` sequence read with:

```python
pred_objs = prediction_objects_from_record(record).as_list()
```

Keep `_build_scored_record()` output canonical:

```python
out = dict(record)
out["pred"] = scored_pred
out["pred_score_source"] = PRED_SCORE_SOURCE
out["pred_score_version"] = PRED_SCORE_VERSION
return out
```

- [ ] **Step 8: Route constant scoring through the adapter**

In `src/eval/artifacts.py`, import:

```python
from src.eval.prediction_records import prediction_objects_from_record
```

Replace the direct `row.get("pred")` block in `with_constant_scores()` with:

```python
for pred in prediction_objects_from_record(row).objects:
    pred_out = dict(pred)
    pred_out["score"] = score
    preds_out.append(pred_out)
```

Keep `scored_row["pred"] = preds_out` so scored artifacts remain canonical.

- [ ] **Step 9: Route proxy GT filtering through the adapter**

In `src/eval/proxy_views.py`, import:

```python
from src.eval.prediction_records import ground_truth_objects_from_record
```

Replace `_gt_key()` with:

```python
def _gt_key(record: Mapping[str, Any]) -> str:
    sequence = ground_truth_objects_from_record(record)
    if sequence.objects:
        return sequence.key
    raise ValueError("record must contain list-valued `gt` or `objects`")
```

Keep `filter_proxy_record()` assigning `out[gt_key] = filtered_gt` so legacy `objects` inputs stay legacy `objects` outputs.

## Verification

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_detection_eval_ingestion_diagnostics.py \
  tests/test_detection_eval_output_parity.py \
  tests/test_confidence_postop.py \
  tests/test_proxy_eval_views.py \
  -q
```

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_unified_infer_pipeline.py \
  tests/test_proxy_eval_bundle.py \
  -q
```

Run:

```bash
git diff --check
```

Expected:

- eval ingestion, duplicate control, confidence post-op, proxy-view, infer-pipeline, and proxy-bundle tests pass,
- scored artifacts continue writing canonical `pred`,
- duplicate-control guarded artifacts preserve the input prediction key,
- `metrics.json` keys remain unchanged,
- no whitespace errors.

## Commit Boundary

Use a focused commit after verification:

```bash
git add src/eval/prediction_records.py src/eval/detection.py src/eval/confidence_postop.py src/eval/artifacts.py src/eval/proxy_views.py tests/test_detection_eval_ingestion_diagnostics.py tests/test_detection_eval_output_parity.py tests/test_confidence_postop.py tests/test_proxy_eval_views.py docs/superpowers/plans/2026-05-03-prediction-eval-record-adapter.md
git commit -m "refactor(eval): add prediction record read adapter"
```
