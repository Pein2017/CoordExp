# Raw-Text Coordinate Mechanism Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a brand-new, mechanism-first study for the two approved raw-text checkpoint objects, covering confirmatory numeric-composition testing, duplicate-burst onset / pre-burst anchor-collapse, FN mechanism breakdown, and a low-effort review queue.

**Architecture:** Keep one new study surface with a strict stage graph: contract rebuild, immutable case-bank freeze, confirmatory behavioral core, exploratory duplicate/FN branches, representation extraction, review export, and final report synthesis. Reuse existing low-level teacher-forced loading and contact-sheet rendering where those are already correct, but keep the new study state and artifact contract isolated under a new `raw_text_coordinate_mechanism` namespace so none of the old continuity conclusions leak into the new run.

**Tech Stack:** Python 3.12, YAML configs, existing CoordExp inference/eval pipeline, Hugging Face / ms-swift Qwen3-VL runtime, pytest, ruff, JSONL/CSV/Markdown reporting, and existing GT-vs-Pred visualization helpers.

---

## Scope Decision

This spec spans several analysis lanes, but they are not independent products. The case bank, branch runners, representation lane, review queue, and final synthesis all depend on the same checkpoint contract, serializer policy, stable case ids, and artifact layout. Keep this as **one implementation plan** with sharply separated files rather than splitting it into multiple loosely coupled plans.

## File Structure

### New files

- Create: `src/analysis/raw_text_coordinate_mechanism_study.py`
  - Stage orchestration, config loading, execution-cell planning, output-root layout.
- Create: `src/analysis/raw_text_coordinate_case_bank.py`
  - Immutable case-row schema, candidate mining normalization, shortlist freeze logic.
- Create: `src/analysis/raw_text_coordinate_behavior.py`
  - Full changed-chunk coordinate scoring, EOS-vs-object scoring, lexical controls, basin metrics.
- Create: `src/analysis/raw_text_coordinate_exploratory.py`
  - Duplicate-burst intervention matrix, FN taxonomy, exploratory branch materialization.
- Create: `src/analysis/raw_text_coordinate_representation.py`
  - Hidden-state extraction, pooled span states, RSA / CKA / decode metrics.
- Create: `src/analysis/raw_text_coordinate_review_queue.py`
  - Review CSV export, contact-sheet assembly, local snapshot manifest.
- Create: `src/analysis/raw_text_coordinate_mechanism_report.py`
  - Summary tables, verdict derivation, report bundle writing.
- Create: `scripts/analysis/run_raw_text_coordinate_mechanism_study.py`
  - Config-driven CLI entrypoint.
- Create: `configs/analysis/raw_text_coordinate_mechanism/default.yaml`
- Create: `configs/analysis/raw_text_coordinate_mechanism/smoke.yaml`
- Create: `tests/test_raw_text_coordinate_mechanism_study.py`
- Create: `tests/test_raw_text_coordinate_case_bank.py`
- Create: `tests/test_raw_text_coordinate_behavior.py`
- Create: `tests/test_raw_text_coordinate_exploratory.py`
- Create: `tests/test_raw_text_coordinate_representation.py`
- Create: `tests/test_raw_text_coordinate_review_queue.py`
- Create: `tests/test_raw_text_coordinate_mechanism_report.py`

### Existing files to modify

- Modify: `src/analysis/unmatched_proposal_verifier.py`
  - Extend `TeacherForcedScorer` so the new study can request hidden states for prepared examples without replacing the current scoring entrypoints.
- Modify: `tests/test_unmatched_proposal_verifier_scorer.py`
  - Add regression coverage for hidden-state capture and keep the existing adapter-shorthand / left-padding guarantees intact.

### Existing files to reuse without modifying unless a test proves it is necessary

- Reuse: `src/analysis/raw_text_coord_continuity_scoring.py`
  - Source of proven span-logprob math; copy only the narrow logic that is still correct.
- Reuse: `src/analysis/raw_text_coord_manual_review.py`
  - Contact-sheet renderer and bbox-first review presentation.
- Reuse: `src/analysis/rollout_fn_factor_study.py`
  - FN recovery-channel helpers and prefix-mode semantics.
- Reuse: `src/analysis/duplication_followup.py`
  - Prefix-perturbation variant semantics.
- Reuse: `scripts/run_infer.py`
  - Base inference surface when fresh gt-vs-pred artifacts must be rebuilt.

---

### Task 1: Scaffold The New Study Surface

**Files:**
- Create: `src/analysis/raw_text_coordinate_mechanism_study.py`
- Create: `scripts/analysis/run_raw_text_coordinate_mechanism_study.py`
- Create: `configs/analysis/raw_text_coordinate_mechanism/default.yaml`
- Create: `configs/analysis/raw_text_coordinate_mechanism/smoke.yaml`
- Test: `tests/test_raw_text_coordinate_mechanism_study.py`

- [ ] **Step 1: Write the failing config/orchestration tests**

```python
from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_study import (
    load_study_config,
    plan_stage_cells,
)


def test_load_study_config_parses_two_fixed_raw_text_models(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        """
run:
  name: raw-text-coordinate-mechanism
  output_dir: output/analysis
  stages: [contract, case_bank, confirmatory, shortlist, exploratory, representation, review, report]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/demo/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  reuse_existing: true

review:
  fp_budget: 15
  fn_budget: 5
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_study_config(config_path)

    assert cfg.models.base_only.adapter_path is None
    assert cfg.models.base_plus_adapter.adapter_path.endswith("checkpoint-552")
    assert cfg.review.fp_budget == 15
    assert tuple(cfg.run.stages) == (
        "contract",
        "case_bank",
        "confirmatory",
        "shortlist",
        "exploratory",
        "representation",
        "review",
        "report",
    )


def test_plan_stage_cells_splits_models_across_available_gpus() -> None:
    stage_cells = plan_stage_cells(
        stage="exploratory",
        gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        model_aliases=("base_only", "base_plus_adapter"),
        branch_names=("duplicate_fp", "fn", "heatmap", "perturb"),
    )

    assert len(stage_cells) == 8
    assert {cell["gpu_id"] for cell in stage_cells} == set(range(8))
    assert stage_cells[0]["model_alias"] == "base_only"
    assert stage_cells[-1]["branch_name"] == "perturb"
```

- [ ] **Step 2: Run the scaffold tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_mechanism_study.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.analysis.raw_text_coordinate_mechanism_study'`

- [ ] **Step 3: Add the minimal study module, runner, and configs**

`src/analysis/raw_text_coordinate_mechanism_study.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class ModelObjectConfig:
    alias: str
    base_path: str
    adapter_path: str | None
    prompt_variant: str
    object_field_order: str
    serializer_surfaces: tuple[str, ...]


@dataclass(frozen=True)
class DatasetConfig:
    train_jsonl: str
    val_jsonl: str


@dataclass(frozen=True)
class ExecutionConfig:
    gpu_ids: tuple[int, ...]
    reuse_existing: bool


@dataclass(frozen=True)
class ReviewConfig:
    fp_budget: int
    fn_budget: int


@dataclass(frozen=True)
class StudyModels:
    base_only: ModelObjectConfig
    base_plus_adapter: ModelObjectConfig


@dataclass(frozen=True)
class StudyConfig:
    run: RunConfig
    models: StudyModels
    dataset: DatasetConfig
    execution: ExecutionConfig
    review: ReviewConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("study config must be a mapping")
    return payload


def load_study_config(config_path: Path) -> StudyConfig:
    raw = _load_yaml(config_path)
    run_raw = raw["run"]
    models_raw = raw["models"]
    dataset_raw = raw["dataset"]
    execution_raw = raw["execution"]
    review_raw = raw["review"]
    return StudyConfig(
        run=RunConfig(
            name=str(run_raw["name"]),
            output_dir=str(run_raw["output_dir"]),
            stages=tuple(str(value) for value in run_raw["stages"]),
        ),
        models=StudyModels(
            base_only=ModelObjectConfig(
                alias=str(models_raw["base_only"]["alias"]),
                base_path=str(models_raw["base_only"]["base_path"]),
                adapter_path=models_raw["base_only"]["adapter_path"],
                prompt_variant=str(models_raw["base_only"]["prompt_variant"]),
                object_field_order=str(models_raw["base_only"]["object_field_order"]),
                serializer_surfaces=tuple(models_raw["base_only"]["serializer_surfaces"]),
            ),
            base_plus_adapter=ModelObjectConfig(
                alias=str(models_raw["base_plus_adapter"]["alias"]),
                base_path=str(models_raw["base_plus_adapter"]["base_path"]),
                adapter_path=models_raw["base_plus_adapter"]["adapter_path"],
                prompt_variant=str(models_raw["base_plus_adapter"]["prompt_variant"]),
                object_field_order=str(models_raw["base_plus_adapter"]["object_field_order"]),
                serializer_surfaces=tuple(models_raw["base_plus_adapter"]["serializer_surfaces"]),
            ),
        ),
        dataset=DatasetConfig(**dataset_raw),
        execution=ExecutionConfig(
            gpu_ids=tuple(int(value) for value in execution_raw["gpu_ids"]),
            reuse_existing=bool(execution_raw["reuse_existing"]),
        ),
        review=ReviewConfig(
            fp_budget=int(review_raw["fp_budget"]),
            fn_budget=int(review_raw["fn_budget"]),
        ),
    )


def plan_stage_cells(
    *,
    stage: str,
    gpu_ids: Sequence[int],
    model_aliases: Sequence[str],
    branch_names: Sequence[str],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    if stage == "exploratory":
        for gpu_id, (model_alias, branch_name) in zip(
            gpu_ids,
            (
                (model_alias, branch_name)
                for model_alias in model_aliases
                for branch_name in branch_names
            ),
            strict=False,
        ):
            cells.append(
                {
                    "stage": stage,
                    "gpu_id": int(gpu_id),
                    "model_alias": str(model_alias),
                    "branch_name": str(branch_name),
                }
            )
    return cells
```

`scripts/analysis/run_raw_text_coordinate_mechanism_study.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_study import load_study_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_study_config(Path(args.config))
    print(cfg.run.name)


if __name__ == "__main__":
    main()
```

`configs/analysis/raw_text_coordinate_mechanism/default.yaml`

```yaml
run:
  name: raw-text-coordinate-mechanism
  output_dir: output/analysis
  stages: [contract, case_bank, confirmatory, shortlist, exploratory, representation, review, report]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  reuse_existing: true

review:
  fp_budget: 15
  fn_budget: 5
```

`configs/analysis/raw_text_coordinate_mechanism/smoke.yaml`

```yaml
run:
  name: raw-text-coordinate-mechanism-smoke
  output_dir: output/analysis
  stages: [contract, case_bank]

models:
  base_only:
    alias: base_only
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    adapter_path: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    serializer_surfaces: [model_native, pretty_inline]

dataset:
  train_jsonl: public_data/coco/rescale_32_1024_bbox_max60/train.norm.jsonl
  val_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl

execution:
  gpu_ids: [0]
  reuse_existing: true

review:
  fp_budget: 2
  fn_budget: 1
```

- [ ] **Step 4: Run tests and lint to verify the scaffold passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_mechanism_study.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_mechanism_study.py scripts/analysis/run_raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_mechanism_study.py
```

Expected:

- `2 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_mechanism_study.py scripts/analysis/run_raw_text_coordinate_mechanism_study.py configs/analysis/raw_text_coordinate_mechanism/default.yaml configs/analysis/raw_text_coordinate_mechanism/smoke.yaml tests/test_raw_text_coordinate_mechanism_study.py
git commit -m "feat: scaffold raw text coordinate mechanism study"
```

---

### Task 2: Build The Immutable Case Bank And Review Shortlist

**Files:**
- Create: `src/analysis/raw_text_coordinate_case_bank.py`
- Modify: `src/analysis/raw_text_coordinate_mechanism_study.py`
- Test: `tests/test_raw_text_coordinate_case_bank.py`

- [ ] **Step 1: Write the failing case-bank tests**

```python
from src.analysis.raw_text_coordinate_case_bank import (
    build_case_bank_rows,
    freeze_review_shortlist,
)


def test_build_case_bank_rows_emits_required_fields() -> None:
    duplicate_rows = [
        {
            "model_alias": "base_plus_adapter",
            "image_id": 11,
            "line_idx": 0,
            "record_idx": 0,
            "source_object_index": 2,
            "onset_object_index": 3,
            "selection_rank": 1,
            "serializer_surface": "pretty_inline",
        }
    ]
    fn_rows = [
        {
            "model_alias": "base_only",
            "image_id": 22,
            "line_idx": 1,
            "record_idx": 4,
            "gt_idx": 5,
            "selection_rank": 3,
            "serializer_surface": "model_native",
        }
    ]

    rows = build_case_bank_rows(
        duplicate_rows=duplicate_rows,
        fn_rows=fn_rows,
    )

    assert rows[0].bucket == "first_burst_onset"
    assert rows[0].case_uid == "base_plus_adapter:11:0:0:first_burst_onset"
    assert rows[1].bucket == "labeled_fn"
    assert rows[1].serializer_surface == "model_native"


def test_freeze_review_shortlist_respects_fp_and_fn_budgets() -> None:
    rows = build_case_bank_rows(
        duplicate_rows=[
            {
                "model_alias": "base_plus_adapter",
                "image_id": idx,
                "line_idx": 0,
                "record_idx": idx,
                "source_object_index": idx,
                "onset_object_index": idx + 1,
                "selection_rank": idx,
                "serializer_surface": "pretty_inline",
            }
            for idx in range(20)
        ],
        fn_rows=[
            {
                "model_alias": "base_only",
                "image_id": 100 + idx,
                "line_idx": 0,
                "record_idx": idx,
                "gt_idx": idx,
                "selection_rank": idx,
                "serializer_surface": "pretty_inline",
            }
            for idx in range(10)
        ],
    )

    shortlist = freeze_review_shortlist(rows, fp_budget=15, fn_budget=5)

    assert len(shortlist) == 20
    assert sum(row.review_bucket == "FP" for row in shortlist) == 15
    assert sum(row.review_bucket == "FN" for row in shortlist) == 5
```

- [ ] **Step 2: Run the case-bank tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_case_bank.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.analysis.raw_text_coordinate_case_bank'`

- [ ] **Step 3: Implement the case-row schema and shortlist freeze**

`src/analysis/raw_text_coordinate_case_bank.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CaseRow:
    case_uid: str
    model_alias: str
    image_id: int
    line_idx: int
    record_idx: int
    bucket: str
    review_bucket: str
    source_object_index: int | None
    onset_object_index: int | None
    gt_idx: int | None
    selection_rank: int
    serializer_surface: str


def _case_uid(row: dict[str, object], bucket: str) -> str:
    return (
        f"{row['model_alias']}:{row['image_id']}:{row['line_idx']}:"
        f"{row['record_idx']}:{bucket}"
    )


def build_case_bank_rows(
    *,
    duplicate_rows: Iterable[dict[str, object]],
    fn_rows: Iterable[dict[str, object]],
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    for row in duplicate_rows:
        rows.append(
            CaseRow(
                case_uid=_case_uid(row, "first_burst_onset"),
                model_alias=str(row["model_alias"]),
                image_id=int(row["image_id"]),
                line_idx=int(row["line_idx"]),
                record_idx=int(row["record_idx"]),
                bucket="first_burst_onset",
                review_bucket="FP",
                source_object_index=int(row["source_object_index"]),
                onset_object_index=int(row["onset_object_index"]),
                gt_idx=None,
                selection_rank=int(row["selection_rank"]),
                serializer_surface=str(row["serializer_surface"]),
            )
        )
    for row in fn_rows:
        rows.append(
            CaseRow(
                case_uid=_case_uid(row, "labeled_fn"),
                model_alias=str(row["model_alias"]),
                image_id=int(row["image_id"]),
                line_idx=int(row["line_idx"]),
                record_idx=int(row["record_idx"]),
                bucket="labeled_fn",
                review_bucket="FN",
                source_object_index=None,
                onset_object_index=None,
                gt_idx=int(row["gt_idx"]),
                selection_rank=int(row["selection_rank"]),
                serializer_surface=str(row["serializer_surface"]),
            )
        )
    return sorted(rows, key=lambda row: (row.review_bucket, row.selection_rank, row.case_uid))


def freeze_review_shortlist(
    rows: list[CaseRow],
    *,
    fp_budget: int,
    fn_budget: int,
) -> list[CaseRow]:
    fp_rows = [row for row in rows if row.review_bucket == "FP"][:fp_budget]
    fn_rows = [row for row in rows if row.review_bucket == "FN"][:fn_budget]
    return fp_rows + fn_rows
```

`src/analysis/raw_text_coordinate_mechanism_study.py`

```python
from src.analysis.raw_text_coordinate_case_bank import (
    build_case_bank_rows,
    freeze_review_shortlist,
)


def run_case_bank_stage(
    *,
    duplicate_rows: list[dict[str, object]],
    fn_rows: list[dict[str, object]],
    fp_budget: int,
    fn_budget: int,
) -> dict[str, object]:
    case_rows = build_case_bank_rows(
        duplicate_rows=duplicate_rows,
        fn_rows=fn_rows,
    )
    shortlist = freeze_review_shortlist(
        case_rows,
        fp_budget=fp_budget,
        fn_budget=fn_budget,
    )
    return {
        "case_row_count": len(case_rows),
        "shortlist_count": len(shortlist),
    }
```

- [ ] **Step 4: Run tests and lint to verify the case bank passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_case_bank.py tests/test_raw_text_coordinate_mechanism_study.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_case_bank.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_case_bank.py
```

Expected:

- `4 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_case_bank.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_case_bank.py
git commit -m "feat: add immutable raw text mechanism case bank"
```

---

### Task 3: Add Behavioral Scoring Surfaces And Stronger Lexical Controls

**Files:**
- Create: `src/analysis/raw_text_coordinate_behavior.py`
- Test: `tests/test_raw_text_coordinate_behavior.py`

- [ ] **Step 1: Write the failing behavior tests**

```python
from src.analysis.raw_text_coordinate_behavior import (
    lexical_control_features,
    summarize_choice_margin,
)


def test_lexical_control_features_uses_real_token_edit_distance() -> None:
    features = lexical_control_features(
        candidate_value=820,
        center_value=819,
        gt_value=819,
        candidate_tokens=("8", "2", "0"),
        center_tokens=("8", "1", "9"),
    )

    assert features["numeric_distance_to_center"] == 1
    assert features["token_edit_distance"] == 2
    assert features["shared_prefix_length"] == 1


def test_summarize_choice_margin_prefers_higher_score() -> None:
    summary = summarize_choice_margin(
        choice_scores={
            "eos": {"logprob_sum": -4.0},
            "next_object": {"logprob_sum": -1.5},
        }
    )

    assert summary["winner"] == "next_object"
    assert summary["margin"] == 2.5
```

- [ ] **Step 2: Run the behavior tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_behavior.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.analysis.raw_text_coordinate_behavior'`

- [ ] **Step 3: Implement scoring helpers and lexical controls**

`src/analysis/raw_text_coordinate_behavior.py`

```python
from __future__ import annotations

from typing import Mapping, Sequence


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    prev = list(range(len(right) + 1))
    for left_idx, left_value in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_value in enumerate(right, start=1):
            cost = 0 if left_value == right_value else 1
            current.append(
                min(
                    prev[right_idx] + 1,
                    current[right_idx - 1] + 1,
                    prev[right_idx - 1] + cost,
                )
            )
        prev = current
    return int(prev[-1])


def lexical_control_features(
    *,
    candidate_value: int,
    center_value: int,
    gt_value: int,
    candidate_tokens: Sequence[str],
    center_tokens: Sequence[str],
) -> dict[str, int]:
    candidate_text = str(candidate_value)
    center_text = str(center_value)
    shared_prefix_length = 0
    for left, right in zip(candidate_text, center_text):
        if left != right:
            break
        shared_prefix_length += 1
    return {
        "numeric_distance_to_center": abs(candidate_value - center_value),
        "numeric_distance_to_gt": abs(candidate_value - gt_value),
        "char_edit_distance": _edit_distance(tuple(candidate_text), tuple(center_text)),
        "token_edit_distance": _edit_distance(candidate_tokens, center_tokens),
        "digit_length_match": int(len(candidate_text) == len(center_text)),
        "token_count": len(candidate_tokens),
        "shared_prefix_length": shared_prefix_length,
    }


def summarize_choice_margin(
    *,
    choice_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float | str]:
    ranked = sorted(
        (
            (label, float(payload["logprob_sum"]))
            for label, payload in choice_scores.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1]
    return {
        "winner": winner,
        "margin": winner_score - runner_up_score,
    }
```

- [ ] **Step 4: Run tests and lint to verify behavior helpers pass**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_behavior.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_behavior.py tests/test_raw_text_coordinate_behavior.py
```

Expected:

- `2 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_behavior.py tests/test_raw_text_coordinate_behavior.py
git commit -m "feat: add raw text mechanism behavior helpers"
```

---

### Task 4: Build The Confirmatory Core Stage

**Files:**
- Modify: `src/analysis/raw_text_coordinate_behavior.py`
- Modify: `src/analysis/raw_text_coordinate_mechanism_study.py`
- Test: `tests/test_raw_text_coordinate_behavior.py`

- [ ] **Step 1: Add the failing confirmatory-core tests**

```python
from src.analysis.raw_text_coordinate_behavior import summarize_confirmatory_records


def test_summarize_confirmatory_records_separates_serializer_surfaces_and_vision_lift() -> None:
    summary = summarize_confirmatory_records(
        records=[
            {
                "model_alias": "base_only",
                "serializer_surface": "model_native",
                "slot": "x1",
                "candidate_value": 100,
                "gt_value": 100,
                "distance_to_gt": 0,
                "image_condition": "correct",
                "logprob_sum": -1.0,
            },
            {
                "model_alias": "base_only",
                "serializer_surface": "model_native",
                "slot": "x1",
                "candidate_value": 100,
                "gt_value": 100,
                "distance_to_gt": 0,
                "image_condition": "swapped",
                "logprob_sum": -3.5,
            },
            {
                "model_alias": "base_only",
                "serializer_surface": "pretty_inline",
                "slot": "x1",
                "candidate_value": 104,
                "gt_value": 100,
                "distance_to_gt": 4,
                "image_condition": "correct",
                "logprob_sum": -2.0,
            },
        ]
    )

    model_native = next(
        row for row in summary if row["serializer_surface"] == "model_native"
    )
    pretty_inline = next(
        row for row in summary if row["serializer_surface"] == "pretty_inline"
    )

    assert model_native["mass_at_4"] == 1.0
    assert model_native["vision_lift"] == 2.5
    assert pretty_inline["mass_at_4"] == 1.0
```

- [ ] **Step 2: Run the confirmatory-core tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_behavior.py -q
```

Expected:

- FAIL with `ImportError: cannot import name 'summarize_confirmatory_records'`

- [ ] **Step 3: Implement confirmatory summaries and stage wiring**

`src/analysis/raw_text_coordinate_behavior.py`

```python
def summarize_confirmatory_records(
    *,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (
            str(record["model_alias"]),
            str(record["serializer_surface"]),
            str(record["slot"]),
        )
        grouped.setdefault(key, []).append(record)

    summary_rows: list[dict[str, object]] = []
    for (model_alias, serializer_surface, slot), group in grouped.items():
        correct_scores = [
            float(row["logprob_sum"])
            for row in group
            if row["image_condition"] == "correct"
        ]
        swapped_scores = [
            float(row["logprob_sum"])
            for row in group
            if row["image_condition"] == "swapped"
        ]
        mass_at_4 = sum(
            1
            for row in group
            if row["image_condition"] == "correct"
            and int(row["distance_to_gt"]) <= 4
        ) / max(1, sum(1 for row in group if row["image_condition"] == "correct"))
        summary_rows.append(
            {
                "model_alias": model_alias,
                "serializer_surface": serializer_surface,
                "slot": slot,
                "mass_at_4": mass_at_4,
                "vision_lift": (sum(correct_scores) / max(1, len(correct_scores)))
                - (sum(swapped_scores) / max(1, len(swapped_scores))),
            }
        )
    return summary_rows
```

`src/analysis/raw_text_coordinate_mechanism_study.py`

```python
from src.analysis.raw_text_coordinate_behavior import summarize_confirmatory_records


def run_confirmatory_stage(
    *,
    records: list[dict[str, object]],
) -> dict[str, object]:
    summary_rows = summarize_confirmatory_records(records=records)
    return {
        "summary_rows": summary_rows,
        "serializer_surfaces": sorted(
            {row["serializer_surface"] for row in summary_rows}
        ),
    }
```

- [ ] **Step 4: Run tests and lint to verify the confirmatory stage passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_behavior.py tests/test_raw_text_coordinate_mechanism_study.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_behavior.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_behavior.py
```

Expected:

- `5 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_behavior.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_behavior.py
git commit -m "feat: add confirmatory core summaries for raw text study"
```

---

### Task 5: Implement Duplicate-Burst And FN Mechanism Branches

**Files:**
- Create: `src/analysis/raw_text_coordinate_exploratory.py`
- Modify: `src/analysis/raw_text_coordinate_mechanism_study.py`
- Test: `tests/test_raw_text_coordinate_exploratory.py`

- [ ] **Step 1: Write the failing exploratory-branch tests**

```python
from src.analysis.raw_text_coordinate_exploratory import (
    build_prefix_intervention_matrix,
    label_fn_bucket,
)


def test_build_prefix_intervention_matrix_exposes_all_required_variants() -> None:
    variants = build_prefix_intervention_matrix(source_object={"bbox": [1, 2, 3, 4]}, gt_next={"bbox": [4, 5, 6, 7]})

    assert [variant["label"] for variant in variants] == [
        "baseline",
        "drop_previous_object",
        "geometry_only_swap",
        "text_only_swap",
        "x1y1_only",
        "x2y2_only",
        "full_gt_next_geometry",
        "nonlocal_same_desc_geometry",
    ]


def test_label_fn_bucket_prefers_stop_pressure_only_when_other_recovery_is_absent() -> None:
    bucket = label_fn_bucket(
        recovered_by_sampling=False,
        recovered_by_clean_prefix=False,
        recovered_by_stop_probe=True,
        has_teacher_forced_support=True,
        ambiguity_flag=False,
    )

    assert bucket == "stop_pressure_fn"
```

- [ ] **Step 2: Run the exploratory tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_exploratory.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.analysis.raw_text_coordinate_exploratory'`

- [ ] **Step 3: Implement intervention variants and FN bucketing**

`src/analysis/raw_text_coordinate_exploratory.py`

```python
from __future__ import annotations


def build_prefix_intervention_matrix(
    *,
    source_object: dict[str, object],
    gt_next: dict[str, object],
) -> list[dict[str, object]]:
    return [
        {"label": "baseline", "mode": "identity"},
        {"label": "drop_previous_object", "mode": "drop_previous"},
        {"label": "geometry_only_swap", "mode": "replace_bbox_keep_text"},
        {"label": "text_only_swap", "mode": "replace_text_keep_bbox"},
        {"label": "x1y1_only", "mode": "replace_x1y1"},
        {"label": "x2y2_only", "mode": "replace_x2y2"},
        {"label": "full_gt_next_geometry", "mode": "replace_bbox_with_gt_next"},
        {"label": "nonlocal_same_desc_geometry", "mode": "replace_bbox_with_nonlocal_same_desc"},
    ]


def label_fn_bucket(
    *,
    recovered_by_sampling: bool,
    recovered_by_clean_prefix: bool,
    recovered_by_stop_probe: bool,
    has_teacher_forced_support: bool,
    ambiguity_flag: bool,
) -> str:
    if ambiguity_flag:
        return "unlabeled_positive_or_eval_ambiguity"
    if recovered_by_sampling:
        return "decode_selection_fn"
    if recovered_by_clean_prefix:
        return "continuation_blocked_fn"
    if recovered_by_stop_probe and has_teacher_forced_support:
        return "stop_pressure_fn"
    if not has_teacher_forced_support:
        return "never_supported_fn"
    return "never_supported_fn"
```

`src/analysis/raw_text_coordinate_mechanism_study.py`

```python
from src.analysis.raw_text_coordinate_exploratory import (
    build_prefix_intervention_matrix,
    label_fn_bucket,
)


def run_exploratory_stage(*, cases: list[dict[str, object]]) -> dict[str, object]:
    intervention_count = 0
    fn_bucket_counts: dict[str, int] = {}
    for case in cases:
        if case["review_bucket"] == "FP":
            intervention_count += len(
                build_prefix_intervention_matrix(
                    source_object=case["source_object"],
                    gt_next=case["gt_next"],
                )
            )
        else:
            bucket = label_fn_bucket(
                recovered_by_sampling=bool(case["recovered_by_sampling"]),
                recovered_by_clean_prefix=bool(case["recovered_by_clean_prefix"]),
                recovered_by_stop_probe=bool(case["recovered_by_stop_probe"]),
                has_teacher_forced_support=bool(case["has_teacher_forced_support"]),
                ambiguity_flag=bool(case["ambiguity_flag"]),
            )
            fn_bucket_counts[bucket] = fn_bucket_counts.get(bucket, 0) + 1
    return {
        "intervention_count": intervention_count,
        "fn_bucket_counts": fn_bucket_counts,
    }
```

- [ ] **Step 4: Run tests and lint to verify exploratory logic passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_exploratory.py tests/test_raw_text_coordinate_mechanism_study.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_exploratory.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_exploratory.py
```

Expected:

- `4 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_exploratory.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_exploratory.py
git commit -m "feat: add duplicate and fn exploratory branches"
```

---

### Task 6: Add Hidden-State Extraction And Representation Metrics

**Files:**
- Create: `src/analysis/raw_text_coordinate_representation.py`
- Modify: `src/analysis/unmatched_proposal_verifier.py`
- Modify: `tests/test_unmatched_proposal_verifier_scorer.py`
- Test: `tests/test_raw_text_coordinate_representation.py`

- [ ] **Step 1: Write the failing representation tests**

```python
import torch

from src.analysis.raw_text_coordinate_representation import (
    pool_span_hidden_states,
    representation_rsa,
)


def test_pool_span_hidden_states_supports_last_digit_and_mean_digits() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        ]
    )
    pooled = pool_span_hidden_states(hidden_states=hidden, pooling=("last_digit", "mean_digits"))

    assert torch.equal(pooled["last_digit"], torch.tensor([[3.0, 0.0]]))
    assert torch.equal(pooled["mean_digits"], torch.tensor([[2.0, 0.0]]))


def test_representation_rsa_is_one_for_perfectly_ordered_distances() -> None:
    states = torch.tensor([[0.0], [1.0], [3.0]])
    numeric_values = torch.tensor([100.0, 101.0, 103.0])

    assert representation_rsa(states=states, numeric_values=numeric_values) == 1.0
```

- [ ] **Step 2: Run the representation tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_representation.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'src.analysis.raw_text_coordinate_representation'`

- [ ] **Step 3: Implement representation metrics and scorer hidden-state capture**

`src/analysis/raw_text_coordinate_representation.py`

```python
from __future__ import annotations

import torch


def pool_span_hidden_states(
    *,
    hidden_states: torch.Tensor,
    pooling: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    pooled: dict[str, torch.Tensor] = {}
    if "last_digit" in pooling:
        pooled["last_digit"] = hidden_states[:, -1, :]
    if "mean_digits" in pooling:
        pooled["mean_digits"] = hidden_states.mean(dim=1)
    return pooled


def representation_rsa(
    *,
    states: torch.Tensor,
    numeric_values: torch.Tensor,
) -> float:
    state_dist = torch.cdist(states.float(), states.float(), p=2).flatten()
    numeric_dist = torch.cdist(
        numeric_values.float().unsqueeze(1),
        numeric_values.float().unsqueeze(1),
        p=1,
    ).flatten()
    state_centered = state_dist - state_dist.mean()
    numeric_centered = numeric_dist - numeric_dist.mean()
    numerator = torch.sum(state_centered * numeric_centered)
    denominator = torch.sqrt(torch.sum(state_centered ** 2) * torch.sum(numeric_centered ** 2))
    return float((numerator / denominator).item())
```

`src/analysis/unmatched_proposal_verifier.py`

```python
    def score_prepared_batch_hidden_states(
        self,
        *,
        examples: Sequence[PreparedExample],
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        model_inputs = self.processor(
            text=[example.full_text for example in examples],
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("teacher-forced scorer missing hidden states")
        return hidden_states[-1]
```

`tests/test_unmatched_proposal_verifier_scorer.py`

```python
def test_teacher_forced_scorer_score_prepared_batch_hidden_states_returns_last_layer(monkeypatch) -> None:
    scorer = TeacherForcedScorer.__new__(TeacherForcedScorer)
    scorer.device = "cpu"
    scorer.processor = _Processor()
    scorer.model = _Model(hidden_states=True)

    examples = [
        PreparedExample(
            full_text="demo",
            assistant_text="demo",
            desc_positions=[0],
            full_input_ids=[1, 2, 3],
            assistant_start=0,
            assistant_input_ids=[1, 2, 3],
        )
    ]
    hidden = scorer.score_prepared_batch_hidden_states(
        examples=examples,
        images=[Image.new("RGB", (4, 4), color="white")],
    )

    assert tuple(hidden.shape) == (1, 3, 2)
```

- [ ] **Step 4: Run tests and lint to verify representation plumbing passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_representation.py tests/test_unmatched_proposal_verifier_scorer.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_representation.py src/analysis/unmatched_proposal_verifier.py tests/test_raw_text_coordinate_representation.py tests/test_unmatched_proposal_verifier_scorer.py
```

Expected:

- `5 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_representation.py src/analysis/unmatched_proposal_verifier.py tests/test_raw_text_coordinate_representation.py tests/test_unmatched_proposal_verifier_scorer.py
git commit -m "feat: add representation extraction for raw text mechanism study"
```

---

### Task 7: Build The Review Queue And Final Report Bundle

**Files:**
- Create: `src/analysis/raw_text_coordinate_review_queue.py`
- Create: `src/analysis/raw_text_coordinate_mechanism_report.py`
- Modify: `src/analysis/raw_text_coordinate_mechanism_study.py`
- Test: `tests/test_raw_text_coordinate_review_queue.py`
- Test: `tests/test_raw_text_coordinate_mechanism_report.py`

- [ ] **Step 1: Write the failing review/report tests**

```python
from pathlib import Path

from src.analysis.raw_text_coordinate_mechanism_report import write_report_bundle
from src.analysis.raw_text_coordinate_review_queue import build_review_queue_rows


def test_build_review_queue_rows_exposes_notion_friendly_columns() -> None:
    rows = build_review_queue_rows(
        shortlist=[
            {
                "case_uid": "base_only:1:0:0:first_burst_onset",
                "review_bucket": "FP",
                "model_alias": "base_only",
                "selection_rank": 1,
            }
        ]
    )

    assert rows[0]["case_uid"] == "base_only:1:0:0:first_burst_onset"
    assert rows[0]["bucket"] == "FP"
    assert rows[0]["status"] == "unreviewed"
    assert rows[0]["bbox_judgment"] == ""


def test_write_report_bundle_materializes_required_outputs(tmp_path: Path) -> None:
    summary = {"q1": "inconclusive", "q2": "inconclusive"}
    write_report_bundle(
        output_dir=tmp_path,
        summary=summary,
        review_rows=[{"case_uid": "demo"}],
    )

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "review_queue.csv").exists()
```

- [ ] **Step 2: Run the review/report tests to verify they fail**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for the new review/report modules

- [ ] **Step 3: Implement review export and final bundle writers**

`src/analysis/raw_text_coordinate_review_queue.py`

```python
from __future__ import annotations


def build_review_queue_rows(
    *,
    shortlist: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in shortlist:
        rows.append(
            {
                "case_uid": row["case_uid"],
                "bucket": row["review_bucket"],
                "priority": row["selection_rank"],
                "status": "unreviewed",
                "model_focus": row["model_alias"],
                "bbox_judgment": "",
                "mechanism_label": "",
                "best_evidence": "",
                "confidence": "",
                "notes": "",
                "asset_links": "",
            }
        )
    return rows
```

`src/analysis/raw_text_coordinate_mechanism_report.py`

```python
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_report_bundle(
    *,
    output_dir: Path,
    summary: dict[str, object],
    review_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        "# Raw-Text Coordinate Mechanism Report\n\n"
        f"- questions: {len(summary)}\n"
        f"- review_rows: {len(review_rows)}\n",
        encoding="utf-8",
    )
    with (output_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()) if review_rows else ["case_uid"])
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
```

`src/analysis/raw_text_coordinate_mechanism_study.py`

```python
from src.analysis.raw_text_coordinate_mechanism_report import write_report_bundle
from src.analysis.raw_text_coordinate_review_queue import build_review_queue_rows


def run_review_and_report_stage(
    *,
    output_dir: Path,
    shortlist: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    review_rows = build_review_queue_rows(shortlist=shortlist)
    write_report_bundle(
        output_dir=output_dir,
        summary=summary,
        review_rows=review_rows,
    )
```

- [ ] **Step 4: Run tests and lint to verify the review/report surface passes**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py -q
rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_review_queue.py src/analysis/raw_text_coordinate_mechanism_report.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py
```

Expected:

- `4 passed`
- `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/analysis/raw_text_coordinate_review_queue.py src/analysis/raw_text_coordinate_mechanism_report.py src/analysis/raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py
git commit -m "feat: add raw text mechanism review export and report bundle"
```

---

## Operator Runbook

Use these commands only after all seven tasks are implemented and passing.

### Smoke

```bash
rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/smoke.yaml
```

Expected:

- writes `output/analysis/raw-text-coordinate-mechanism-smoke/`
- includes `stage_manifest.json`, `summary.json`, and `case_bank.jsonl`

### Full confirmatory + shortlist build

```bash
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage contract
CUDA_VISIBLE_DEVICES=1 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage case_bank
CUDA_VISIBLE_DEVICES=2 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage confirmatory --model-alias base_only
CUDA_VISIBLE_DEVICES=3 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage confirmatory --model-alias base_plus_adapter
CUDA_VISIBLE_DEVICES=4 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage shortlist
```

### Full exploratory + representation fanout

```bash
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_only --branch duplicate_fp
CUDA_VISIBLE_DEVICES=1 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_plus_adapter --branch duplicate_fp
CUDA_VISIBLE_DEVICES=2 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_only --branch fn
CUDA_VISIBLE_DEVICES=3 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_plus_adapter --branch fn
CUDA_VISIBLE_DEVICES=4 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage representation --model-alias base_only
CUDA_VISIBLE_DEVICES=5 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage representation --model-alias base_plus_adapter
CUDA_VISIBLE_DEVICES=6 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_only --branch heatmap
CUDA_VISIBLE_DEVICES=7 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage exploratory --model-alias base_plus_adapter --branch heatmap
```

### Review and report

```bash
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage review
CUDA_VISIBLE_DEVICES=0 rtk conda run -n ms python scripts/analysis/run_raw_text_coordinate_mechanism_study.py --config configs/analysis/raw_text_coordinate_mechanism/default.yaml --stage report
```

---

## Final Verification Checklist

- `rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_case_bank.py tests/test_raw_text_coordinate_behavior.py tests/test_raw_text_coordinate_exploratory.py tests/test_raw_text_coordinate_representation.py tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py tests/test_unmatched_proposal_verifier_scorer.py -q`
- `rtk conda run -n ms ruff check src/analysis/raw_text_coordinate_mechanism_study.py src/analysis/raw_text_coordinate_case_bank.py src/analysis/raw_text_coordinate_behavior.py src/analysis/raw_text_coordinate_exploratory.py src/analysis/raw_text_coordinate_representation.py src/analysis/raw_text_coordinate_review_queue.py src/analysis/raw_text_coordinate_mechanism_report.py src/analysis/unmatched_proposal_verifier.py tests/test_raw_text_coordinate_mechanism_study.py tests/test_raw_text_coordinate_case_bank.py tests/test_raw_text_coordinate_behavior.py tests/test_raw_text_coordinate_exploratory.py tests/test_raw_text_coordinate_representation.py tests/test_raw_text_coordinate_review_queue.py tests/test_raw_text_coordinate_mechanism_report.py tests/test_unmatched_proposal_verifier_scorer.py`
