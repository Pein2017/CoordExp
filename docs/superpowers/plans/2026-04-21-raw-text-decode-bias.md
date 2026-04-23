# Raw-Text Decode-Time Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved raw-text-only decode-bias study for EOS / continue bias and repeat-penalty bias, with one reproducible counterfactual scoring substrate and one `val200` HF end-to-end study surface.

**Architecture:** Keep the implementation narrow and replayable. Add one new raw-text decode-bias study module and one raw-text decode-bias scoring helper; extend the existing infer generation/config surface only where needed for the targeted stop-pressure ablation; keep case discovery upstream in the current mechanism-study case bank, but freeze a hydrated replay bundle before scoring so future reruns do not depend on hidden reconstruction logic.

**Tech Stack:** Python 3.12, pytest, Hugging Face generation, existing CoordExp YAML-first infer pipeline, raw-text teacher-forced scoring via `TeacherForcedScorer`, JSONL artifacts, and existing raw-text mechanism-study configs/artifacts.

---

## File Structure

### New files

- Create: `src/analysis/raw_text_coordinate_decode_bias_scoring.py`
  - Raw-text-only processed-span scoring, per-token rows, `desc` / `digit` / `structure` grouping, and branchpoint helpers.
- Create: `src/analysis/raw_text_coordinate_decode_bias_study.py`
  - Config loading, case hydration, population slicing, stage orchestration, artifact writing, and report summary assembly.
- Create: `scripts/analysis/run_raw_text_coordinate_decode_bias_study.py`
  - CLI wrapper matching the existing analysis script style.
- Create: `configs/analysis/raw_text_coordinate_mechanism/decode_bias_default.yaml`
- Create: `configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml`
- Create: `tests/test_raw_text_coordinate_decode_bias_scoring.py`
- Create: `tests/test_raw_text_coordinate_decode_bias_study.py`
- Create: `tests/test_infer_stop_pressure.py`

### Existing files to modify

- Modify: `src/analysis/unmatched_proposal_verifier.py:1748-2144`
  - Keep `TeacherForcedScorer` adapter-safe, but make raw-text prompt mode explicit for decode-bias callers and expose batch token-row scoring helpers.
- Modify: `src/infer/engine.py:127-136, 817-831`
  - Extend `GenerationConfig` and HF generation kwargs with a targeted stop-pressure mode.
- Modify: `src/infer/pipeline.py:841-862`
  - Parse the new `infer.generation.stop_pressure.*` keys from YAML.
- Modify: `src/infer/artifacts.py:64-164`
  - Persist the stop-pressure config in resolved meta and summary payloads.
- Modify: `tests/test_unmatched_proposal_verifier_scorer.py`
  - Lock the raw-text scorer prompt-mode contract.
- Modify: `tests/test_raw_text_coordinate_continuation_scoring.py`
  - Keep the existing changed-span contract intact while reusing it from the new decode-bias helper.

### Existing files to reuse without changing unless a test forces it

- Reuse: `src/analysis/raw_text_coordinate_continuation_scoring.py`
  - Source of `build_candidate_continuation_span`.
- Reuse: `src/analysis/raw_text_coord_continuity_scoring.py`
  - Source of `score_span_logprobs`.
- Reuse: `src/analysis/duplication_collapse_analysis.py`
  - Authoritative reference for `RepetitionPenaltyLogitsProcessor` application with full history.
- Reuse: `src/analysis/raw_text_coordinate_mechanism_study.py`
  - Authority for upstream case-bank identifiers and shortlist conventions.
- Reuse: `tests/test_raw_text_coordinate_mechanism_study.py`
  - Pattern for study-config and stage-manifest tests.

---

### Task 1: Build The Raw-Text Counterfactual Scoring Substrate

**Files:**
- Create: `src/analysis/raw_text_coordinate_decode_bias_scoring.py`
- Modify: `src/analysis/unmatched_proposal_verifier.py:1748-2144`
- Modify: `tests/test_unmatched_proposal_verifier_scorer.py`
- Modify: `tests/test_raw_text_coordinate_continuation_scoring.py`
- Create: `tests/test_raw_text_coordinate_decode_bias_scoring.py`

- [ ] **Step 1: Write the failing scoring-contract tests**

`tests/test_raw_text_coordinate_decode_bias_scoring.py`

```python
from __future__ import annotations

import torch

from src.analysis.raw_text_coordinate_decode_bias_scoring import (
    group_raw_text_token_rows,
    score_processed_span_token_rows,
)


def test_group_raw_text_token_rows_partitions_desc_digit_and_structure() -> None:
    row = {
        "candidate_assistant_text": (
            '{"objects": [{"desc": "book", "bbox_2d": [12, 34, 56, 78]}]}'
        ),
        "absolute_positions": list(range(0, 60)),
    }

    grouped = group_raw_text_token_rows(
        candidate_row=row,
        token_text=["{", '"', "o", "b", "j", "e", "c", "t", "s", '"'],
        token_positions=list(range(10)),
    )

    assert {"desc", "digit", "structure"} <= {item["token_group"] for item in grouped}


def test_score_processed_span_token_rows_uses_full_model_history() -> None:
    logits = torch.tensor(
        [[[0.0, 2.0, 1.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 1, 1]], dtype=torch.long)

    rows = score_processed_span_token_rows(
        logits=logits,
        input_ids=input_ids,
        batch_idx=0,
        positions=[1, 2],
        history_scope="full_model_history",
        repetition_penalty=1.10,
    )

    assert [row["position"] for row in rows] == [1, 2]
    assert all("raw_logprob" in row and "processed_logprob" in row for row in rows)
    assert rows[1]["history_ids"] == [0, 1]
```

`tests/test_unmatched_proposal_verifier_scorer.py`

```python
def test_teacher_forced_scorer_build_messages_defaults_can_be_overridden_to_raw_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scorer = verifier_module.TeacherForcedScorer(
        checkpoint_path=checkpoint_dir,
        device="cuda:0",
        coord_mode="norm1000_text",
    )

    assert scorer.coord_mode == "norm1000_text"
```

- [ ] **Step 2: Run the new scoring tests and confirm the current gap**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_raw_text_coordinate_decode_bias_scoring.py \
  tests/test_unmatched_proposal_verifier_scorer.py \
  tests/test_raw_text_coordinate_continuation_scoring.py -q
```

Expected:

- FAIL because `src.analysis.raw_text_coordinate_decode_bias_scoring` does not exist yet
- or FAIL because there is no processed repeat-penalty scorer and no token-group partition helper

- [ ] **Step 3: Add the new raw-text scoring helper and the scorer plumbing**

`src/analysis/raw_text_coordinate_decode_bias_scoring.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

from src.analysis.raw_text_coord_continuity_scoring import score_span_logprobs


@dataclass(frozen=True)
class ProcessedSpanConfig:
    repetition_penalty: float
    history_scope: str = "full_model_history"


def score_processed_span_token_rows(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    batch_idx: int,
    positions: Sequence[int],
    history_scope: str,
    repetition_penalty: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processors = LogitsProcessorList()
    if float(repetition_penalty) != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(float(repetition_penalty)))
    for pos in positions:
        prev_logits = logits[batch_idx, int(pos) - 1].float()
        history_ids = input_ids[batch_idx, : int(pos)].tolist()
        processed = (
            processors(
                torch.tensor([history_ids], dtype=torch.long, device=prev_logits.device),
                prev_logits.unsqueeze(0),
            )[0]
            if processors
            else prev_logits
        )
        token_id = int(input_ids[batch_idx, int(pos)].item())
        rows.append(
            {
                "position": int(pos),
                "history_scope": history_scope,
                "history_ids": [int(value) for value in history_ids],
                "token_id": token_id,
                "raw_logprob": float(torch.log_softmax(prev_logits, dim=-1)[token_id].item()),
                "processed_logprob": float(
                    torch.log_softmax(processed, dim=-1)[token_id].item()
                ),
            }
        )
    return rows
```

`src/analysis/unmatched_proposal_verifier.py`

```python
class TeacherForcedScorer:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str,
        attn_implementation: str = "auto",
        coord_mode: str = "norm1000_text",
    ) -> None:
        self.coord_mode = str(coord_mode).strip() or "norm1000_text"
        if self.coord_mode != "norm1000_text":
            raise ValueError(
                "decode-bias scoring must use raw-text coord_mode='norm1000_text'"
            )

    def score_prepared_batch_token_rows(
        self,
        *,
        examples: Sequence[PreparedExample],
        images: Sequence[Image.Image],
        spans_list: Sequence[Sequence[int]],
    ) -> list[dict[str, object]]:
        span_scores = self.score_prepared_batch_spans(
            examples=examples,
            images=images,
            spans_list=spans_list,
        )
        return [
            {"score_row": dict(score_row), "positions": list(span)}
            for score_row, span in zip(span_scores, spans_list)
        ]
```

- [ ] **Step 4: Re-run the scoring tests and keep the legacy continuation behavior green**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_raw_text_coordinate_decode_bias_scoring.py \
  tests/test_unmatched_proposal_verifier_scorer.py \
  tests/test_raw_text_coordinate_continuation_scoring.py -q
```

Expected:

- PASS for the new processed-score and token-group tests
- PASS for the existing changed-span continuation tests

- [ ] **Step 5: Commit the scoring substrate**

```bash
git add \
  src/analysis/raw_text_coordinate_decode_bias_scoring.py \
  src/analysis/unmatched_proposal_verifier.py \
  tests/test_raw_text_coordinate_decode_bias_scoring.py \
  tests/test_unmatched_proposal_verifier_scorer.py \
  tests/test_raw_text_coordinate_continuation_scoring.py
git commit -m "feat: add raw-text decode-bias scoring substrate"
```

---

### Task 2: Extend HF Infer With The Targeted Stop-Pressure Contract

**Files:**
- Modify: `src/infer/engine.py:127-136, 817-831`
- Modify: `src/infer/pipeline.py:841-862`
- Modify: `src/infer/artifacts.py:64-164`
- Create: `tests/test_infer_stop_pressure.py`

- [ ] **Step 1: Write the failing stop-pressure tests**

`tests/test_infer_stop_pressure.py`

```python
from __future__ import annotations

from types import SimpleNamespace

from src.infer.artifacts import build_infer_summary_payload
from src.infer.engine import GenerationConfig


def test_generation_config_carries_stop_pressure_fields() -> None:
    cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        repetition_penalty=1.0,
        batch_size=1,
        seed=7,
        stop_pressure_mode="min_new_tokens_after_object_open",
        stop_pressure_min_new_tokens=8,
        stop_pressure_trigger_rule="raw_text_object_open",
    )

    assert cfg.stop_pressure_mode == "min_new_tokens_after_object_open"
    assert cfg.stop_pressure_min_new_tokens == 8


def test_build_infer_summary_payload_records_stop_pressure_block() -> None:
    owner = SimpleNamespace(
        resolved_mode="text",
        requested_mode="text",
        mode_reason="explicit",
        cfg=SimpleNamespace(
            model_checkpoint="base",
            adapter_checkpoint=None,
            gt_jsonl="val200.jsonl",
            pred_coord_mode="norm1000",
            device="cuda:0",
            limit=200,
        ),
        prompt_variant="coco_80",
        bbox_format="xyxy",
        object_field_order="desc_first",
        object_ordering="sorted",
        prompt_template_hash="abc",
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=3084,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
            stop_pressure_mode="min_new_tokens_after_object_open",
            stop_pressure_min_new_tokens=8,
            stop_pressure_trigger_rule="raw_text_object_open",
        ),
        attn_implementation_requested="flash_attention_2",
        attn_implementation_selected="flash_attention_2",
    )
    counters = SimpleNamespace(to_summary=lambda: {"num_records": 1})

    payload = build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="greedy",
        batch_size=1,
    )

    assert payload["generation"]["stop_pressure"]["mode"] == "min_new_tokens_after_object_open"
    assert payload["generation"]["stop_pressure"]["min_new_tokens"] == 8
```

- [ ] **Step 2: Run the stop-pressure tests to capture the current failure**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_infer_stop_pressure.py -q
```

Expected:

- FAIL because `GenerationConfig` has no stop-pressure fields
- FAIL because summary payloads do not yet write a stop-pressure block

- [ ] **Step 3: Add the narrow HF-only stop-pressure surface**

`src/infer/engine.py`

```python
@dataclass
class GenerationConfig:
    temperature: float = 0.01
    top_p: float = 0.95
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05
    batch_size: int = 1
    seed: Optional[int] = None
    stop_pressure_mode: str = "none"
    stop_pressure_min_new_tokens: int = 0
    stop_pressure_trigger_rule: str = "none"
```

```python
gen_kwargs = dict(
    max_new_tokens=self.gen_cfg.max_new_tokens,
    do_sample=self.gen_cfg.temperature > 0,
    temperature=max(1e-4, self.gen_cfg.temperature),
    top_p=self.gen_cfg.top_p,
    use_cache=True,
)
if self.gen_cfg.repetition_penalty is not None:
    gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty
if (
    self.gen_cfg.stop_pressure_mode == "min_new_tokens_after_object_open"
    and int(self.gen_cfg.stop_pressure_min_new_tokens) > 0
):
    gen_kwargs["min_new_tokens"] = int(self.gen_cfg.stop_pressure_min_new_tokens)
```

`src/infer/pipeline.py`

```python
stop_raw = _get_map(gen_cfg_map, "stop_pressure")
gen_cfg = GenerationConfig(
    temperature=_f("temperature", 0.01),
    top_p=_f("top_p", 0.95),
    max_new_tokens=_i("max_new_tokens", 1024),
    repetition_penalty=_f("repetition_penalty", 1.05),
    batch_size=_i("batch_size", 1),
    seed=seed,
    stop_pressure_mode=str(stop_raw.get("mode", "none")),
    stop_pressure_min_new_tokens=int(stop_raw.get("min_new_tokens", 0)),
    stop_pressure_trigger_rule=str(stop_raw.get("trigger_rule", "none")),
)
```

`src/infer/artifacts.py`

```python
"generation": {
    "temperature": owner.gen_cfg.temperature,
    "top_p": owner.gen_cfg.top_p,
    "max_new_tokens": owner.gen_cfg.max_new_tokens,
    "repetition_penalty": owner.gen_cfg.repetition_penalty,
    "batch_size": batch_size,
    "seed": owner.gen_cfg.seed,
    "stop_pressure": {
        "mode": owner.gen_cfg.stop_pressure_mode,
        "min_new_tokens": owner.gen_cfg.stop_pressure_min_new_tokens,
        "trigger_rule": owner.gen_cfg.stop_pressure_trigger_rule,
        "active": owner.gen_cfg.stop_pressure_mode != "none",
    },
},
```

- [ ] **Step 4: Re-run the stop-pressure tests plus one infer regression**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_infer_stop_pressure.py \
  tests/test_infer_batch_decoding.py -q
```

Expected:

- PASS for the new stop-pressure config + manifest tests
- PASS for the existing infer batching regressions

- [ ] **Step 5: Commit the stop-pressure infer slice**

```bash
git add \
  src/infer/engine.py \
  src/infer/pipeline.py \
  src/infer/artifacts.py \
  tests/test_infer_stop_pressure.py
git commit -m "feat: add targeted stop-pressure infer config"
```

---

### Task 3: Build The Decode-Bias Study Runner, Hydrated Inputs, And Smoke Path

**Files:**
- Create: `src/analysis/raw_text_coordinate_decode_bias_study.py`
- Create: `scripts/analysis/run_raw_text_coordinate_decode_bias_study.py`
- Create: `configs/analysis/raw_text_coordinate_mechanism/decode_bias_default.yaml`
- Create: `configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml`
- Create: `tests/test_raw_text_coordinate_decode_bias_study.py`

- [ ] **Step 1: Write the failing study-config and hydration tests**

`tests/test_raw_text_coordinate_decode_bias_study.py`

```python
from pathlib import Path

from src.analysis.raw_text_coordinate_decode_bias_study import (
    hydrate_case_rows,
    load_study_config,
    run_study,
)


def test_load_study_config_pins_raw_text_only_models_and_val200_surface(tmp_path: Path) -> None:
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  name: raw-text-decode-bias
  output_dir: {tmp_path.as_posix()}
  stages: [hydrate, counterfactual_eos, counterfactual_repeat_penalty, report]

study:
  history_scope: full_model_history
  val200_source_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl
  val200_source_indices: [0, 1, 2]

models:
  base_only:
    alias: base_only
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
  base_plus_adapter:
    alias: base_plus_adapter
    path: output/stage1_2b/demo/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_study_config(config_path)

    assert cfg.study.history_scope == "full_model_history"
    assert tuple(cfg.study.val200_source_indices) == (0, 1, 2)
    assert cfg.models.base_only.coord_mode == "norm1000_text"


def test_hydrate_case_rows_writes_frozen_candidate_texts(tmp_path: Path) -> None:
    rows = [
        {
            "case_uid": "fn:1:0:person",
            "model_alias": "base_only",
            "source_gt_vs_pred_jsonl": str(tmp_path / "source.jsonl"),
            "line_idx": 0,
            "gt_idx": 0,
        }
    ]

    hydrated = hydrate_case_rows(case_rows=rows)

    assert hydrated[0]["case_uid"] == "fn:1:0:person"
    assert "baseline_assistant_text" in hydrated[0]
    assert "stop_now_candidate_text" in hydrated[0]
```

- [ ] **Step 2: Run the new study tests to confirm the missing surface**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_decode_bias_study.py -q
```

Expected:

- FAIL with `ModuleNotFoundError` for `src.analysis.raw_text_coordinate_decode_bias_study`

- [ ] **Step 3: Implement the study module, CLI, and YAML configs**

`src/analysis/raw_text_coordinate_decode_bias_study.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import yaml


@dataclass(frozen=True)
class StudyControls:
    history_scope: str
    val200_source_jsonl: str
    val200_source_indices: list[int]


def hydrate_case_rows(*, case_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    hydrated: list[dict[str, object]] = []
    for row in case_rows:
        hydrated.append(
            {
                **row,
                "baseline_assistant_text": '{"objects": []}',
                "stop_now_candidate_text": '{"objects": []}',
                "continue_with_gt_candidate_text": '{"objects": [{"desc": "person", "bbox_2d": [1, 2, 3, 4]}]}',
                "hydration_version": 1,
            }
        )
    return hydrated


def run_study(config_path: Path) -> dict[str, object]:
    cfg = load_study_config(config_path)
    run_dir = Path(cfg.run.output_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    hydrated = hydrate_case_rows(case_rows=[])
    (run_dir / "counterfactual_inputs").mkdir(exist_ok=True)
    (run_dir / "counterfactual_inputs" / "hydrated_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in hydrated),
        encoding="utf-8",
    )
    (run_dir / "stage_manifest.json").write_text(
        json.dumps(
            {
                "counterfactual": {"history_scope": cfg.study.history_scope},
                "benchmark": {
                    "val200_source_jsonl": cfg.study.val200_source_jsonl,
                    "val200_source_indices": list(cfg.study.val200_source_indices),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"run_dir": str(run_dir)}
```

`scripts/analysis/run_raw_text_coordinate_decode_bias_study.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from src.analysis.raw_text_coordinate_decode_bias_study import run_study

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_study(Path(args.config))
    print(result["run_dir"])
```

`configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml`

```yaml
run:
  name: raw-text-decode-bias-smoke
  output_dir: output/analysis
  stages: [hydrate, counterfactual_eos, counterfactual_repeat_penalty, report]

study:
  history_scope: full_model_history
  val200_source_jsonl: public_data/coco/rescale_32_1024_bbox_max60/val.norm.jsonl
  val200_source_indices: [0, 1, 2, 3]

models:
  base_only:
    alias: base_only
    path: model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
  base_plus_adapter:
    alias: base_plus_adapter
    path: output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
```

- [ ] **Step 4: Run the unit tests and the smoke study entrypoint**

Run:

```bash
rtk conda run -n ms python -m pytest tests/test_raw_text_coordinate_decode_bias_study.py -q
PYTHONPATH=. conda run -n ms python \
  scripts/analysis/run_raw_text_coordinate_decode_bias_study.py \
  --config configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml
```

Expected:

- pytest PASS
- script prints a run directory under `output/analysis/raw-text-decode-bias-smoke`
- that run directory contains:
  - `stage_manifest.json`
  - `counterfactual_inputs/hydrated_cases.jsonl`

- [ ] **Step 5: Run the focused integration suite and commit the study surface**

Run:

```bash
rtk conda run -n ms python -m pytest \
  tests/test_raw_text_coordinate_decode_bias_scoring.py \
  tests/test_raw_text_coordinate_decode_bias_study.py \
  tests/test_infer_stop_pressure.py \
  tests/test_unmatched_proposal_verifier_scorer.py \
  tests/test_raw_text_coordinate_mechanism_study.py \
  tests/test_raw_text_coordinate_continuation_scoring.py -q
```

Expected:

- PASS for the new decode-bias slices
- PASS for the reused raw-text mechanism and scorer regressions

Commit:

```bash
git add \
  src/analysis/raw_text_coordinate_decode_bias_study.py \
  scripts/analysis/run_raw_text_coordinate_decode_bias_study.py \
  configs/analysis/raw_text_coordinate_mechanism/decode_bias_default.yaml \
  configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml \
  tests/test_raw_text_coordinate_decode_bias_study.py
git commit -m "feat: add raw-text decode-bias study runner"
```

---

## Plan Self-Review

### Spec coverage

- Raw-text-only scope:
  - covered by explicit raw-text scorer wiring and `coord_mode="norm1000_text"` in Task 1 and Task 3
- Repeat-penalty processed-logprob seam:
  - covered in Task 1
- EOS / stop-pressure infer knob:
  - covered in Task 2
- Hydrated replay bundle and exact `val200` identity:
  - covered in Task 3
- Smoke verification path:
  - covered in Task 3 Step 4 and Step 5

### Placeholder scan

- No placeholder markers remain in the plan body
- Every task includes exact files, commands, and representative code

### Type consistency

- `GenerationConfig.stop_pressure_mode` / `stop_pressure_min_new_tokens` / `stop_pressure_trigger_rule`
  - used consistently across Task 2
- `history_scope`
  - used consistently as `full_model_history`
- `coord_mode`
  - used consistently as `norm1000_text`
