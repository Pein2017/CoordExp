# Raw-Text Coordinate Continuity Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the raw-text coordinate continuity probe study end-to-end: tokenizer/contract audit, multi-token candidate chunk scoring, canonical good-basin probes, upper-bound bad-basin probes, dense-scene failure alignment, and the final artifact bundle under `output/analysis/raw_text_coord_continuity_probe_<date>/`.

**Architecture:** Add one new analysis study surface that reuses existing inference, teacher-forced scoring, duplication mining, prefix perturbation, and shared visualization contracts. Keep the new logic split across focused files: orchestration/config loading, raw-text chunk scoring and lexical features, and metrics/report/plot generation. Extend `TeacherForcedScorer` just enough to support arbitrary span scoring without replacing its current call sites.

**Tech Stack:** Python, YAML configs, Hugging Face Qwen3-VL processor/model loading, existing `InferenceEngine`, pandas/numpy/scipy/matplotlib, repo GT-vs-Pred visualization stack, `pytest`, `ruff`, and `conda run -n ms`.

---

## File Structure

**Create:**

- `src/analysis/raw_text_coord_continuity_probe.py`
  - config dataclasses, stage orchestration, cohort materialization, reproduction wiring, phase runners
- `src/analysis/raw_text_coord_continuity_scoring.py`
  - raw-text bbox serialization helpers, slot replacement, candidate chunk span extraction, lexical features, scorer wrappers
- `src/analysis/raw_text_coord_continuity_report.py`
  - aggregate metrics, regression tables, plot writers, markdown report synthesis
- `scripts/analysis/run_raw_text_coord_continuity_probe.py`
  - CLI entrypoint mirroring existing `scripts/analysis/run_*` study runners
- `configs/analysis/raw_text_coord_continuity/default.yaml`
  - full study config
- `configs/analysis/raw_text_coord_continuity/smoke.yaml`
  - targeted smoke config for audit/pilot and a tiny canonical lane run
- `tests/test_raw_text_coord_continuity_probe.py`
  - config loading, cohort building, artifact writing, aggregate math
- `tests/test_raw_text_coord_continuity_scoring.py`
  - span scoring math, slot replacement, lexical feature extraction, heatmap input shaping

**Modify:**

- `src/analysis/unmatched_proposal_verifier.py`
  - extend `PreparedExample` and `TeacherForcedScorer` with generic span scoring support
- `src/analysis/duplication_collapse_analysis.py`
  - expose a narrow public helper for duplicate-prone case mining instead of copying the current private logic
- `src/analysis/duplication_followup.py`
  - expose or factor prefix-perturbation helpers that the new study can reuse without replaying the whole old report path

**Do not create unless truly needed:**

- new renderer stacks
- ad hoc notebook-only analysis paths
- one-off infer scripts that bypass `scripts/run_infer.py` / `InferenceEngine`

---

### Task 1: Scaffold The Study Surface

**Files:**
- Create: `src/analysis/raw_text_coord_continuity_probe.py`
- Create: `src/analysis/raw_text_coord_continuity_scoring.py`
- Create: `src/analysis/raw_text_coord_continuity_report.py`
- Create: `scripts/analysis/run_raw_text_coord_continuity_probe.py`
- Create: `configs/analysis/raw_text_coord_continuity/default.yaml`
- Create: `configs/analysis/raw_text_coord_continuity/smoke.yaml`
- Test: `tests/test_raw_text_coord_continuity_probe.py`

- [ ] **Step 1: Write the failing scaffold/config tests**

```python
from pathlib import Path

from src.analysis.raw_text_coord_continuity_probe import load_study_config


def test_load_study_config_parses_lanes_and_cohorts(tmp_path: Path) -> None:
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(
        """
run:
  name: raw-text-continuity
  output_dir: output/analysis
  stages: [audit, pilot, canonical, bad_basin, dense_scene, report]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/demo-checkpoint
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 500
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
    sample_count: 1500
    seed: 29
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_study_config(config_path)

    assert cfg.run.stages == (
        "audit",
        "pilot",
        "canonical",
        "bad_basin",
        "dense_scene",
        "report",
    )
    assert cfg.models.base.alias == "base"
    assert cfg.models.pure_ce.prompt_surface == "canonical"
    assert cfg.cohorts.val_headline.sample_count == 500
    assert cfg.cohorts.train_supplemental.seed == 29
```

- [ ] **Step 2: Run the failing scaffold/config tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'src.analysis.raw_text_coord_continuity_probe'
```

- [ ] **Step 3: Add the minimal study module, runner, and configs**

`src/analysis/raw_text_coord_continuity_probe.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: Tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    path: str
    prompt_surface: str


@dataclass(frozen=True)
class CohortConfig:
    jsonl_path: str
    sample_count: int
    seed: int


@dataclass(frozen=True)
class StudyModels:
    base: ModelConfig
    pure_ce: ModelConfig


@dataclass(frozen=True)
class StudyCohorts:
    val_headline: CohortConfig
    train_supplemental: CohortConfig


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    models: StudyModels
    cohorts: StudyCohorts


def load_study_config(config_path: Path) -> StudyConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    run_raw = raw.get("run") or {}
    models_raw = raw.get("models") or {}
    cohorts_raw = raw.get("cohorts") or {}
    return StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=tuple(str(v) for v in run_raw.get("stages") or ()),
        ),
        models=StudyModels(
            base=ModelConfig(**models_raw["base"]),
            pure_ce=ModelConfig(**models_raw["pure_ce"]),
        ),
        cohorts=StudyCohorts(
            val_headline=CohortConfig(**cohorts_raw["val_headline"]),
            train_supplemental=CohortConfig(**cohorts_raw["train_supplemental"]),
        ),
    )


def run_study(config_path: Path) -> dict[str, object]:
    cfg = load_study_config(config_path)
    return {"run_name": cfg.run.name, "stages": list(cfg.run.stages)}
```

`src/analysis/raw_text_coord_continuity_scoring.py`

```python
"""Scoring helpers for raw-text coordinate continuity probes."""

from __future__ import annotations
```

`src/analysis/raw_text_coord_continuity_report.py`

```python
"""Reporting helpers for raw-text coordinate continuity probes."""

from __future__ import annotations
```

`configs/analysis/raw_text_coord_continuity/default.yaml`

```yaml
run:
  name: raw-text-continuity-default
  output_dir: output/analysis
  stages: [audit, pilot, canonical, bad_basin, dense_scene, report]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 500
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
    sample_count: 1500
    seed: 29
```

`scripts/analysis/run_raw_text_coord_continuity_probe.py`

```python
"""Run the raw-text coordinate continuity probe study from a YAML manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.raw_text_coord_continuity_probe import run_study  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_study(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

`configs/analysis/raw_text_coord_continuity/smoke.yaml`

```yaml
run:
  name: raw-text-continuity-smoke
  output_dir: output/analysis
  stages: [audit, pilot]

models:
  base:
    alias: base
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_surface: upper_bound
  pure_ce:
    alias: pure_ce
    path: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
    prompt_surface: canonical

cohorts:
  val_headline:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl
    sample_count: 8
    seed: 17
  train_supplemental:
    jsonl_path: public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
    sample_count: 16
    seed: 29
```

- [ ] **Step 4: Run the scaffold tests and the smoke runner**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
rtk conda run -n ms python scripts/analysis/run_raw_text_coord_continuity_probe.py \
  --config configs/analysis/raw_text_coord_continuity/smoke.yaml
```

Expected:

```text
1 passed
{
  "run_name": "raw-text-continuity-smoke",
  "stages": ["audit", "pilot"]
}
```

- [ ] **Step 5: Commit the scaffold**

```bash
git add \
  src/analysis/raw_text_coord_continuity_probe.py \
  src/analysis/raw_text_coord_continuity_scoring.py \
  src/analysis/raw_text_coord_continuity_report.py \
  scripts/analysis/run_raw_text_coord_continuity_probe.py \
  configs/analysis/raw_text_coord_continuity/default.yaml \
  configs/analysis/raw_text_coord_continuity/smoke.yaml \
  tests/test_raw_text_coord_continuity_probe.py
git commit -m "feat(analysis): scaffold raw-text continuity study"
```

---

### Task 2: Extend TeacherForcedScorer For Arbitrary Span Scoring

**Files:**
- Modify: `src/analysis/unmatched_proposal_verifier.py`
- Modify: `src/analysis/raw_text_coord_continuity_scoring.py`
- Test: `tests/test_raw_text_coord_continuity_scoring.py`

- [ ] **Step 1: Write failing span-scoring tests with synthetic logits**

```python
import torch

from src.analysis.raw_text_coord_continuity_scoring import score_span_logprobs


def test_score_span_logprobs_supports_multi_token_chunk() -> None:
    logits = torch.full((1, 5, 16), -20.0)
    input_ids = torch.tensor([[1, 3, 4, 5, 2]])
    logits[0, 0, 3] = 5.0
    logits[0, 1, 4] = 4.0
    logits[0, 2, 5] = 3.0
    result = score_span_logprobs(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        positions=[1, 2, 3],
    )
    assert result["count"] == 3
    assert result["mean_logprob"] > -0.1
    assert result["sum_logprob"] > -0.3
```

- [ ] **Step 2: Run the failing span-scoring tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_scoring.py -q
```

Expected:

```text
FAIL ... ImportError: cannot import name 'score_span_logprobs'
```

- [ ] **Step 3: Add generic span scoring and thread it through TeacherForcedScorer**

`src/analysis/raw_text_coord_continuity_scoring.py`

```python
from __future__ import annotations

from typing import Iterable, Sequence

import torch


def score_span_logprobs(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    positions: Sequence[int],
) -> dict[str, float]:
    values: list[float] = []
    for pos in positions:
        prev_logits = logits[batch_idx, int(pos) - 1].float()
        target_id = int(input_ids[batch_idx, int(pos)].item())
        token_logprob = float(
            prev_logits[target_id].detach().cpu().item()
            - torch.logsumexp(prev_logits, dim=-1).detach().cpu().item()
        )
        values.append(token_logprob)
    return {
        "count": float(len(values)),
        "sum_logprob": float(sum(values)),
        "mean_logprob": float(sum(values) / len(values)),
    }
```

`src/analysis/unmatched_proposal_verifier.py`

```python
@dataclass(frozen=True)
class PreparedExample:
    full_text: str
    assistant_text: str
    desc_positions: List[int]
    full_input_ids: List[int]
    assistant_start: int
    assistant_input_ids: List[int]


def prepare_example(... ) -> PreparedExample:
    ...
    assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
    start = _find_subsequence(full_ids, assistant_ids, start_hint=len(prompt_ids))
    if start is None:
        raise ValueError("assistant_span_build_failed")
    desc_positions = [int(start + pos) for pos in desc_positions_rel]
    return PreparedExample(
        full_text=str(full_text),
        assistant_text=str(assistant_text),
        desc_positions=desc_positions,
        full_input_ids=[int(v) for v in full_ids],
        assistant_start=int(start),
        assistant_input_ids=[int(v) for v in assistant_ids],
    )


def score_prepared_spans(
    self,
    *,
    prepared: PreparedExample,
    image: Image.Image,
    spans: Sequence[Sequence[int]],
) -> List[dict[str, float]]:
    model_inputs = self.processor(
        text=[prepared.full_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {
        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
        for k, v in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = self.model(**model_inputs, use_cache=False)
    logits = outputs.logits
    input_ids = model_inputs["input_ids"]
    return [
        score_span_logprobs(
            logits=logits,
            input_ids=input_ids,
            batch_idx=0,
            positions=list(span),
        )
        for span in spans
    ]
```

- [ ] **Step 4: Run the scoring tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_scoring.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit the scorer extension**

```bash
git add \
  src/analysis/unmatched_proposal_verifier.py \
  src/analysis/raw_text_coord_continuity_scoring.py \
  tests/test_raw_text_coord_continuity_scoring.py
git commit -m "feat(analysis): add generic teacher-forced span scoring"
```

---

### Task 3: Implement Raw-Text Slot Replacement, Audit, And Lexical Features

**Files:**
- Modify: `src/analysis/raw_text_coord_continuity_scoring.py`
- Modify: `src/analysis/raw_text_coord_continuity_probe.py`
- Test: `tests/test_raw_text_coord_continuity_scoring.py`

- [ ] **Step 1: Write failing tests for slot replacement and lexical features**

```python
from src.analysis.raw_text_coord_continuity_scoring import (
    lexical_features_for_candidate,
    replace_bbox_slot_value,
)


def test_replace_bbox_slot_value_preserves_json_boundaries() -> None:
    assistant_text = '[{"desc":"book","bbox_2d":[199,200,210,250]}]'
    replaced = replace_bbox_slot_value(
        assistant_text=assistant_text,
        slot="x1",
        original_bbox=(199, 200, 210, 250),
        candidate_value=231,
    )
    assert replaced == '[{"desc":"book","bbox_2d":[231,200,210,250]}]'


def test_lexical_features_capture_numeric_and_token_shape() -> None:
    features = lexical_features_for_candidate(
        candidate_value=210,
        center_value=199,
        gt_value=200,
        tokenizer_tokens=["2", "1", "0"],
        center_tokens=["1", "9", "9"],
    )
    assert features["numeric_distance_to_center"] == 11
    assert features["numeric_distance_to_gt"] == 10
    assert features["digit_length_match"] == 1
    assert features["token_count"] == 3
```

- [ ] **Step 2: Run the failing scoring/audit tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_scoring.py -q
```

Expected:

```text
FAIL ... cannot import name 'replace_bbox_slot_value'
```

- [ ] **Step 3: Implement serialization-aware slot replacement and Phase-0 audit helpers**

`src/analysis/raw_text_coord_continuity_scoring.py`

```python
from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence


def replace_bbox_slot_value(
    *,
    assistant_text: str,
    slot: str,
    original_bbox: Sequence[int],
    candidate_value: int,
) -> str:
    payload = json.loads(assistant_text)
    slot_index = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}[slot]
    for row in payload:
        bbox = list(row.get("bbox_2d") or [])
        if bbox == list(original_bbox):
            bbox[slot_index] = int(candidate_value)
            row["bbox_2d"] = bbox
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raise ValueError("original_bbox_not_found")


def lexical_features_for_candidate(
    *,
    candidate_value: int,
    center_value: int,
    gt_value: int,
    tokenizer_tokens: Sequence[str],
    center_tokens: Sequence[str],
) -> dict[str, int]:
    candidate_text = str(candidate_value)
    center_text = str(center_value)
    gt_text = str(gt_value)
    shared_prefix = 0
    for left, right in zip(candidate_text, center_text):
        if left != right:
            break
        shared_prefix += 1
    return {
        "numeric_distance_to_center": abs(int(candidate_value) - int(center_value)),
        "numeric_distance_to_gt": abs(int(candidate_value) - int(gt_value)),
        "char_edit_distance": int(
            round((1.0 - SequenceMatcher(a=candidate_text, b=center_text).ratio()) * max(len(candidate_text), len(center_text)))
        ),
        "token_edit_distance": abs(len(tokenizer_tokens) - len(center_tokens)),
        "digit_length_match": int(len(candidate_text) == len(center_text)),
        "token_count": int(len(tokenizer_tokens)),
        "shared_prefix_length": int(shared_prefix),
        "same_leading_digit": int(candidate_text[:1] == center_text[:1]),
    }
```

`src/analysis/raw_text_coord_continuity_probe.py`

```python
def run_phase0_audit(...) -> dict[str, object]:
    numbers = [0, 1, 9, 10, 99, 100, 199, 200, 210, 999]
    rows = []
    for value in numbers:
        tokens = scorer.tokenizer.tokenize(str(value))
        rows.append(
            {
                "value": value,
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
    return {"numbers": rows}
```

- [ ] **Step 4: Run the updated scoring tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_scoring.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit the raw-text scoring helpers**

```bash
git add \
  src/analysis/raw_text_coord_continuity_scoring.py \
  src/analysis/raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_scoring.py
git commit -m "feat(analysis): add raw-text slot replacement and audit helpers"
```

---

### Task 4: Build Cohorts, Reproduction, And Hard-Case Mining

**Files:**
- Modify: `src/analysis/duplication_collapse_analysis.py`
- Modify: `src/analysis/raw_text_coord_continuity_probe.py`
- Modify: `configs/analysis/raw_text_coord_continuity/default.yaml`
- Modify: `configs/analysis/raw_text_coord_continuity/smoke.yaml`
- Test: `tests/test_raw_text_coord_continuity_probe.py`

- [ ] **Step 1: Write failing tests for deterministic cohort manifests and duplicate-prone mining**

```python
from pathlib import Path

from src.analysis.raw_text_coord_continuity_probe import (
    build_random_cohort,
    build_study_hard_cases,
)


def test_build_random_cohort_is_deterministic(tmp_path: Path) -> None:
    rows = [{"image_id": i, "image": f"img_{i}.jpg"} for i in range(10)]
    left = build_random_cohort(rows, sample_count=4, seed=17)
    right = build_random_cohort(rows, sample_count=4, seed=17)
    assert [row["image_id"] for row in left] == [row["image_id"] for row in right]


def test_build_study_hard_cases_prefers_duplicate_prone_rows(tmp_path: Path) -> None:
    rows = [
        {"image_id": 1, "max_desc_count": 3, "same_desc_duplicate_pair_count": 0},
        {"image_id": 2, "max_desc_count": 17, "same_desc_duplicate_pair_count": 9},
    ]
    hard = build_study_hard_cases(rows, max_cases=1)
    assert hard[0]["image_id"] == 2
```

- [ ] **Step 2: Run the failing cohort/mining tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
FAIL ... cannot import name 'build_random_cohort'
```

- [ ] **Step 3: Expose a narrow duplicate-mining helper and add cohort builders**

`src/analysis/duplication_collapse_analysis.py`

```python
def mine_duplicate_like_rows(
    *,
    gt_vs_pred_path: Path,
    max_cases: int,
    min_pred_objects: int,
    min_duplicate_pairs: int,
    duplicate_iou_threshold: float,
) -> list[dict[str, Any]]:
    cfg = type(
        "CfgStub",
        (),
        {
            "subset": type(
                "SubsetStub",
                (),
                {
                    "min_pred_objects": min_pred_objects,
                    "min_duplicate_pairs": min_duplicate_pairs,
                    "max_cases_total": max_cases,
                    "max_cases_per_checkpoint": max_cases,
                    "duplicate_iou_threshold": duplicate_iou_threshold,
                    "pinned_line_indices": {},
                },
            )(),
            "controls": type(
                "ControlsStub",
                (),
                {"same_desc_iou_threshold": 0.5},
            )(),
        },
    )()
    checkpoint = type(
        "CheckpointStub",
        (),
        {
            "spec": type("SpecStub", (), {"alias": "probe", "bbox_format": "xyxy"})(),
            "paths": {"gt_vs_pred_jsonl": gt_vs_pred_path},
        },
    )()
    rows = []
    for line_idx, row in enumerate(_read_jsonl(gt_vs_pred_path)):
        preds = list(row.get("pred") or [])
        if len(preds) < min_pred_objects:
            continue
        rows.append(
            {
                "line_idx": line_idx,
                "image_id": row.get("image_id"),
                "image": row.get("image"),
                "pred_count": len(preds),
                "max_desc_count": max(
                    Counter(str(obj.get("desc") or "") for obj in preds).values(),
                    default=0,
                ),
                "same_desc_duplicate_pair_count": sum(
                    1
                    for right in range(1, len(preds))
                    for left in range(right)
                    if (_pair_duplicate_metrics(preds[left], preds[right], cfg=cfg) or {}).get("duplicate_like")
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["same_desc_duplicate_pair_count"]),
            int(row["max_desc_count"]),
            int(row["pred_count"]),
        ),
        reverse=True,
    )
    return rows[:max_cases]
```

`src/analysis/raw_text_coord_continuity_probe.py`

```python
import random


def build_random_cohort(rows: list[dict[str, object]], *, sample_count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    sample = list(rows)
    rng.shuffle(sample)
    return sample[:sample_count]


def build_study_hard_cases(rows: list[dict[str, object]], *, max_cases: int) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row.get("same_desc_duplicate_pair_count") or 0),
            int(row.get("max_desc_count") or 0),
            int(row.get("pred_count") or 0),
        ),
        reverse=True,
    )
    return ordered[:max_cases]
```

- [ ] **Step 4: Run the new cohort/mining tests and the smoke config**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
rtk conda run -n ms python scripts/analysis/run_raw_text_coord_continuity_probe.py \
  --config configs/analysis/raw_text_coord_continuity/smoke.yaml
```

Expected:

```text
... passed
{
  "run_name": "raw-text-continuity-smoke",
  ...
}
```

- [ ] **Step 5: Commit the cohort and mining layer**

```bash
git add \
  src/analysis/duplication_collapse_analysis.py \
  src/analysis/raw_text_coord_continuity_probe.py \
  configs/analysis/raw_text_coord_continuity/default.yaml \
  configs/analysis/raw_text_coord_continuity/smoke.yaml \
  tests/test_raw_text_coord_continuity_probe.py
git commit -m "feat(analysis): add continuity cohorts and hard-case mining"
```

---

### Task 5: Implement Canonical Lane Metrics, Controls, And Aggregate Tables

**Files:**
- Modify: `src/analysis/raw_text_coord_continuity_report.py`
- Modify: `src/analysis/raw_text_coord_continuity_probe.py`
- Test: `tests/test_raw_text_coord_continuity_probe.py`

- [ ] **Step 1: Write failing tests for mass@k, basin width, and Vision Lift**

```python
from src.analysis.raw_text_coord_continuity_report import (
    compute_basin_metrics,
    compute_vision_lift_rows,
)


def test_compute_basin_metrics_uses_gt_center() -> None:
    rows = [
        {"candidate_value": 198, "score": -0.5, "gt_value": 200},
        {"candidate_value": 199, "score": -0.2, "gt_value": 200},
        {"candidate_value": 200, "score": -0.1, "gt_value": 200},
        {"candidate_value": 201, "score": -0.2, "gt_value": 200},
    ]
    metrics = compute_basin_metrics(rows, center_key="gt_value")
    assert metrics["mass_at_1"] > 0.2
    assert metrics["mass_at_2"] >= metrics["mass_at_1"]
    assert metrics["local_expected_abs_error"] < 2.0


def test_compute_vision_lift_rows_pairs_correct_and_swapped() -> None:
    rows = [
        {"case_id": "a", "slot": "x1", "image_condition": "correct", "gt_score": -0.1},
        {"case_id": "a", "slot": "x1", "image_condition": "swapped", "gt_score": -0.9},
    ]
    lifted = compute_vision_lift_rows(rows)
    assert lifted[0]["vision_lift"] == 0.8
```

- [ ] **Step 2: Run the failing aggregate-metric tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
FAIL ... cannot import name 'compute_basin_metrics'
```

- [ ] **Step 3: Add canonical-lane aggregate math and regression table builders**

`src/analysis/raw_text_coord_continuity_report.py`

```python
from __future__ import annotations

from collections import defaultdict
from math import exp
from typing import Iterable, Sequence

import numpy as np


def _softmax(values: Sequence[float]) -> list[float]:
    shifted = np.array(values, dtype=float) - np.max(values)
    weights = np.exp(shifted)
    weights /= weights.sum()
    return [float(v) for v in weights]


def compute_basin_metrics(rows: Sequence[dict[str, object]], *, center_key: str) -> dict[str, float]:
    scores = [float(row["score"]) for row in rows]
    weights = _softmax(scores)
    center = int(rows[0][center_key])
    candidates = [int(row["candidate_value"]) for row in rows]

    def neighborhood_mass(radius: int) -> float:
        return float(
            sum(
                weight
                for candidate, weight in zip(candidates, weights)
                if abs(candidate - center) <= radius
            )
        )

    local_eae = float(
        sum(abs(candidate - center) * weight for candidate, weight in zip(candidates, weights))
    )
    peak = max(scores)
    half_height = peak - (peak - min(scores)) / 2.0
    half_width = max(
        abs(candidate - center)
        for candidate, score in zip(candidates, scores)
        if score >= half_height
    )
    return {
        "mass_at_1": neighborhood_mass(1),
        "mass_at_2": neighborhood_mass(2),
        "mass_at_4": neighborhood_mass(4),
        "mass_at_8": neighborhood_mass(8),
        "mass_at_16": neighborhood_mass(16),
        "local_expected_abs_error": local_eae,
        "half_height_width": float(half_width),
    }


def compute_vision_lift_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["slot"]))][str(row["image_condition"])] = float(row["gt_score"])
    return [
        {
            "case_id": case_id,
            "slot": slot,
            "vision_lift": values["correct"] - values["swapped"],
        }
        for (case_id, slot), values in grouped.items()
        if "correct" in values and "swapped" in values
    ]
```

- [ ] **Step 4: Run the aggregate tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit the canonical aggregation layer**

```bash
git add \
  src/analysis/raw_text_coord_continuity_report.py \
  src/analysis/raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_probe.py
git commit -m "feat(analysis): add canonical continuity aggregates"
```

---

### Task 6: Implement Self-Prefix Bad-Basin Probes, Prefix Geometry Interventions, And Dense-Scene Alignment

**Files:**
- Modify: `src/analysis/duplication_followup.py`
- Modify: `src/analysis/raw_text_coord_continuity_probe.py`
- Modify: `src/analysis/raw_text_coord_continuity_report.py`
- Test: `tests/test_raw_text_coord_continuity_probe.py`

- [ ] **Step 1: Write failing tests for wrong-anchor comparisons and heatmap grids**

```python
from src.analysis.raw_text_coord_continuity_report import (
    build_xy_heatmap_grid,
    summarize_wrong_anchor_advantage,
)


def test_summarize_wrong_anchor_advantage_prefers_pred_center_when_requested() -> None:
    rows = [
        {"candidate_value": 101, "score": -0.2, "gt_value": 140, "pred_value": 100},
        {"candidate_value": 140, "score": -0.9, "gt_value": 140, "pred_value": 100},
    ]
    summary = summarize_wrong_anchor_advantage(rows)
    assert summary["pred_center_mass_at_4"] > summary["gt_center_mass_at_4"]


def test_build_xy_heatmap_grid_preserves_cartesian_order() -> None:
    rows = [
        {"candidate_x1": 10, "candidate_y1": 20, "score": -0.1},
        {"candidate_x1": 10, "candidate_y1": 21, "score": -0.2},
        {"candidate_x1": 11, "candidate_y1": 20, "score": -0.3},
        {"candidate_x1": 11, "candidate_y1": 21, "score": -0.4},
    ]
    heatmap = build_xy_heatmap_grid(rows)
    assert heatmap["x_values"] == [10, 11]
    assert heatmap["y_values"] == [20, 21]
    assert heatmap["z_matrix"][0][0] == -0.1
```

- [ ] **Step 2: Run the failing bad-basin tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
FAIL ... cannot import name 'summarize_wrong_anchor_advantage'
```

- [ ] **Step 3: Reuse prefix-perturbation helpers and add bad-basin summaries**

`src/analysis/duplication_followup.py`

```python
def build_prefix_perturbation_variants(
    *,
    prefix_objects: Sequence[Mapping[str, Any]],
    source_index_in_prefix: int,
    gt_next: Optional[Mapping[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    return _perturbation_variants(
        prefix_objects=prefix_objects,
        source_index_in_prefix=source_index_in_prefix,
        gt_next=gt_next,
    )
```

`src/analysis/raw_text_coord_continuity_report.py`

```python
def summarize_wrong_anchor_advantage(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    gt_metrics = compute_basin_metrics(rows, center_key="gt_value")
    pred_metrics = compute_basin_metrics(rows, center_key="pred_value")
    return {
        "gt_center_mass_at_4": gt_metrics["mass_at_4"],
        "pred_center_mass_at_4": pred_metrics["mass_at_4"],
        "wrong_anchor_advantage_at_4": pred_metrics["mass_at_4"] - gt_metrics["mass_at_4"],
    }


def build_xy_heatmap_grid(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    x_values = sorted({int(row["candidate_x1"]) for row in rows})
    y_values = sorted({int(row["candidate_y1"]) for row in rows})
    score_lookup = {
        (int(row["candidate_x1"]), int(row["candidate_y1"])): float(row["score"])
        for row in rows
    }
    return {
        "x_values": x_values,
        "y_values": y_values,
        "z_matrix": [
            [score_lookup[(x, y)] for x in x_values]
            for y in y_values
        ],
    }
```

- [ ] **Step 4: Run the bad-basin tests**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit the bad-basin and dense-scene alignment layer**

```bash
git add \
  src/analysis/duplication_followup.py \
  src/analysis/raw_text_coord_continuity_probe.py \
  src/analysis/raw_text_coord_continuity_report.py \
  tests/test_raw_text_coord_continuity_probe.py
git commit -m "feat(analysis): add bad-basin and dense-scene alignment probes"
```

---

### Task 7: Wire The Full Report, Smoke The Study, And Execute The First Real Run

**Files:**
- Modify: `src/analysis/raw_text_coord_continuity_probe.py`
- Modify: `src/analysis/raw_text_coord_continuity_report.py`
- Modify: `configs/analysis/raw_text_coord_continuity/default.yaml`
- Modify: `configs/analysis/raw_text_coord_continuity/smoke.yaml`
- Test: `tests/test_raw_text_coord_continuity_probe.py`

- [ ] **Step 1: Write a failing test for report synthesis and artifact contract**

```python
from pathlib import Path

from src.analysis.raw_text_coord_continuity_report import write_report_bundle


def test_write_report_bundle_materializes_required_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "probe"
    write_report_bundle(
        out_dir=out_dir,
        summary={"questions": {"q1": "inconclusive"}},
        report_md="# Demo\n",
        per_coord_rows=[{"case_id": "a", "slot": "x1", "candidate_value": 100, "score": -0.1}],
        hard_cases=[{"case_id": "hard-1"}],
    )
    assert (out_dir / "report.md").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_coord_scores.jsonl").exists()
    assert (out_dir / "hard_cases.jsonl").exists()
```

- [ ] **Step 2: Run the failing report-bundle test**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coord_continuity_probe.py -q
```

Expected:

```text
FAIL ... cannot import name 'write_report_bundle'
```

- [ ] **Step 3: Implement artifact writing, the final runner path, and the smoke config**

`src/analysis/raw_text_coord_continuity_report.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def write_report_bundle(
    *,
    out_dir: Path,
    summary: dict[str, object],
    report_md: str,
    per_coord_rows: Sequence[dict[str, object]],
    hard_cases: Sequence[dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "per_coord_scores.jsonl", list(per_coord_rows))
    _write_jsonl(out_dir / "hard_cases.jsonl", list(hard_cases))
```

`src/analysis/raw_text_coord_continuity_probe.py`

```python
def run_study(config_path: Path) -> dict[str, object]:
    cfg = load_study_config(config_path)
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": cfg.run.name,
        "stages": list(cfg.run.stages),
        "questions": {},
    }
    write_report_bundle(
        out_dir=run_dir,
        summary=summary,
        report_md="# Raw-Text Coordinate Continuity Probe\n",
        per_coord_rows=[],
        hard_cases=[],
    )
    return {"run_dir": str(run_dir), "summary": summary}
```

- [ ] **Step 4: Run targeted tests, lint, and the smoke study**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_scoring.py -q
rtk conda run -n ms ruff check \
  src/analysis/raw_text_coord_continuity_probe.py \
  src/analysis/raw_text_coord_continuity_scoring.py \
  src/analysis/raw_text_coord_continuity_report.py \
  scripts/analysis/run_raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_scoring.py
rtk conda run -n ms python scripts/analysis/run_raw_text_coord_continuity_probe.py \
  --config configs/analysis/raw_text_coord_continuity/smoke.yaml
```

Expected:

```text
... passed
All checks passed!
{
  "run_dir": "output/analysis/raw-text-continuity-smoke",
  "summary": {
    "run_name": "raw-text-continuity-smoke",
    "stages": ["audit", "pilot"]
  }
}
```

- [ ] **Step 5: Commit the full runnable study surface**

```bash
git add \
  src/analysis/raw_text_coord_continuity_probe.py \
  src/analysis/raw_text_coord_continuity_scoring.py \
  src/analysis/raw_text_coord_continuity_report.py \
  scripts/analysis/run_raw_text_coord_continuity_probe.py \
  configs/analysis/raw_text_coord_continuity/default.yaml \
  configs/analysis/raw_text_coord_continuity/smoke.yaml \
  tests/test_raw_text_coord_continuity_probe.py \
  tests/test_raw_text_coord_continuity_scoring.py
git commit -m "feat(analysis): add raw-text continuity probe runner"
```

---

## GPU / Execution Notes

- Keep initial audit, scoring-helper tests, and smoke runner work on one GPU.
- Use subagent-driven parallel execution after the scaffold is stable:
  - one worker for scorer extensions and tests
  - one worker for cohort/mining wiring
  - one worker for report/plot generation
  - one worker for config and smoke execution
- Reserve the multi-GPU budget for the actual study runs, not for unit tests.
- For the first real run after smoke:
  - start with `Phase 0 + Phase 1`
  - then launch the canonical lane on a small val slice
  - only after scorer sanity passes should the large `val 500 / train 1500` run start

## Self-Review Checklist

- Does the plan reuse `TeacherForcedScorer`, `InferenceEngine`, duplication-case
  mining, and shared GT-vs-Pred rendering instead of replacing them?
- Does every new output land under the declared artifact contract?
- Are prompt-rescue effects kept out of the canonical lane?
- Are dense-scene failure tails treated as a supportive alignment surface rather
  than headline continuity evidence?
- Are tests present for every new helper that could silently corrupt scoring?

## Execution Recommendation

Recommended execution mode: **Subagent-Driven**.

Why:

- the scoring extension, cohort/reproduction wiring, and reporting/plotting are
  separable once the scaffold lands
- the user explicitly made subagents and multiple GPUs available
- the study is large enough that staged parallel ownership will save time
