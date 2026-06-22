# Autoregressive Binding Mechanism Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audit-adjusted, open-ended mechanism discovery program that first falsifies replay/label artifacts, then recursively searches for the deepest defensible account of how these causal V-LLMs perceive an object, bind it to language and coordinates, emit an object span, shift to the next object, duplicate, drift, or terminate.

**Architecture:** Keep production inference untouched unless a task explicitly creates a bounded experimental surface under the analysis output root. Add focused analysis modules under `src/analysis/autoregressive_binding_template_ablation/`, thin CLIs under `scripts/analysis/`, durable reports under `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted/`, and a hypothesis ledger that can spawn evidence-gated follow-up probes. The first milestone is a deterministic pre-gate over existing artifacts: generation-time repetition-penalty consistency, de-circularized previous-anchor diagnostics, split discipline, family-specific selection events, and non-circular nulls. After that, the program may branch recursively into hidden-state probes, attention/readout interventions, prefix counterfactuals, train-vs-val checks, and limited micro-training assays when a hypothesis needs a causal perturbation.

**Tech Stack:** Python 3.12, PyTorch for model stages only, NumPy, PyYAML, pytest, existing CoordExp detection template utilities, existing Qwen3-VL probe loading code, JSONL/JSON/Markdown artifacts.

---

## Audit-Adjusted Research Contract

This plan supersedes the earlier Phase 3 ordering in this same file. It incorporates the two audit notes:

```text
/data/CoordExp/autoregressive-binding-template-study-audit-2026-06-18.md
/data/CoordExp/autoregressive-audit.md
```

The current evidence baseline remains:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v2_token_embeddings_surface
```

Current config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Primary output root for this adjusted phase:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
```

### Full-Control Discovery Mandate

The user has explicitly authorized a broad, self-directed research loop over the two checkpoint-928 families, train and val datasets, and available GPUs. This plan therefore treats the Phase 3 audit gates as the beginning of a recursive mechanism search, not the whole mission.

Allowed investigation surfaces:

```text
existing train and val JSONL rows, with provenance recorded by path and split
existing free-rollout artifacts and newly generated scaled probes
teacher-forced replay, short free continuation, and prefix counterfactuals
hidden states, attention, logits, token-role traces, and coordinate-token manifolds
8-GPU sharded capture and replay after deterministic manifests are written
limited-scope micro-training or adapter nudges under a new analysis output root
new hypotheses proposed by agents when prior evidence falsifies or narrows a path
```

Bounded intervention rule:

```text
Micro-training is allowed only when the launch config records source checkpoint, train rows, max steps, trainable modules, learning rate, expected artifact root, and rollback path. The first launch for any hypothesis must be a dry-run manifest or a tiny run with max_steps <= 32 unless a later findings note explicitly promotes it.
```

Convergence definition:

```text
A convergence point is reached when the current best mechanism candidate has passed falsification gates, beats strong nulls on held-out rows, has a localized causal behavior readout, and has no unresolved audit objection that could fully explain the result. If every candidate fails, convergence means a clear negative map of which explanations were falsified and which next family of hypotheses is most promising.
```

### Non-Negotiable Corrections

1. Do not claim a "fragile duplicate basin" from raw teacher-forced logits until the same `repetition_penalty=1.10` transform used by rollout is applied to the realized prefix and the effect survives.
2. Do not claim "previous-anchor reuse" from a metric that is mechanically tied to `duplicate_iou70`. Measure immediate previous, nearest previous same-desc, and any-prior anchors separately; use upstream labels for predictive claims.
3. Do not pool desc-first and geometry-first as if `pre_x1` is the same computational event. Use family-specific selection events:

```text
desc_first: pre_desc, desc_end
geometry_first: pre_box_start, pre_x1
```

4. Do not read eval recall semantics into the sequential greedy matcher. Report it as sequential-greedy coverage unless actual eval matching is wired.
5. Do not run hidden-vector capture, 8-GPU causal sweeps, or a final promotion table until the pre-gates pass on held-out rows.
6. Use dynamic adjustment when a promising path appears, but write down the gate result and evidence scope before expanding the path.

### Promotion Gate

Promote a candidate mechanism only if all are true:

```text
rep_penalty_survives: penalty-adjusted logits preserve the signal direction.
upstream_signal: signal is available before the family-specific divergence event.
non_circular: predictor is not defined from the same emitted box identity as the label.
held_out: effect beats nulls on reserve or val200 grouped by image.
within_family: effect direction is visible inside each family separately.
behavioral_causal: localized patch changes the next emitted object under short free continuation.
controls_null: noop, wrong-role, wrong-image, post-commit, shuffled-candidate, and norm-matched-random controls remain null.
```

Demote the current hypothesis if either P0 gate fails:

```text
rep_penalty_collapse: duplicate entropy/rank/p_emitted gap disappears after repetition-penalty correction.
anchor_circularity_collapse: previous-anchor signal does not beat object index or same-class density when measured upstream.
```

---

## File Structure

Create:

```text
src/analysis/autoregressive_binding_template_ablation/phase3_manifest.py
src/analysis/autoregressive_binding_template_ablation/repetition_penalty_gate.py
src/analysis/autoregressive_binding_template_ablation/anchor_circularity_gate.py
src/analysis/autoregressive_binding_template_ablation/candidate_ledger.py
src/analysis/autoregressive_binding_template_ablation/coverage_polarity.py
src/analysis/autoregressive_binding_template_ablation/null_leaderboard.py
src/analysis/autoregressive_binding_template_ablation/identity_posterior.py
src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py
src/analysis/autoregressive_binding_template_ablation/research_loop.py
src/analysis/autoregressive_binding_template_ablation/micro_training.py
scripts/analysis/run_autoregressive_binding_phase3_manifest.py
scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py
scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py
scripts/analysis/run_autoregressive_binding_candidate_ledger.py
scripts/analysis/run_autoregressive_binding_coverage_polarity.py
scripts/analysis/run_autoregressive_binding_null_leaderboard.py
scripts/analysis/run_autoregressive_binding_identity_posterior.py
scripts/analysis/run_autoregressive_binding_behavioral_patch.py
scripts/analysis/run_autoregressive_binding_research_loop.py
scripts/analysis/run_autoregressive_binding_micro_training.py
tests/analysis/test_autoregressive_binding_template_phase3_manifest.py
tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py
tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py
tests/analysis/test_autoregressive_binding_template_candidate_ledger.py
tests/analysis/test_autoregressive_binding_template_coverage_polarity.py
tests/analysis/test_autoregressive_binding_template_null_leaderboard.py
tests/analysis/test_autoregressive_binding_template_identity_posterior.py
tests/analysis/test_autoregressive_binding_template_behavioral_patch.py
tests/analysis/test_autoregressive_binding_template_research_loop.py
tests/analysis/test_autoregressive_binding_template_micro_training.py
```

Modify:

```text
progress/diagnostics/README.md
progress/index.yaml
```

Defer until a gate passes:

```text
src/analysis/autoregressive_binding_template_ablation/hidden_vector_store.py
scripts/analysis/run_autoregressive_binding_hidden_vectors.py
8-GPU hidden-vector capture
8-GPU causal sweep
```

Do not modify:

```text
model_cache/
output/stage1_2b/
outputs/stage1_2b/
upstream HF model files
```

---

### Task 1: Split Manifest And GT Candidate Materialization

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/phase3_manifest.py`
- Create: `scripts/analysis/run_autoregressive_binding_phase3_manifest.py`
- Test: `tests/analysis/test_autoregressive_binding_template_phase3_manifest.py`

- [ ] **Step 1: Write the failing split-manifest test**

Create `tests/analysis/test_autoregressive_binding_template_phase3_manifest.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.autoregressive_binding_template_ablation.phase3_manifest import (
    build_split_manifest,
    materialize_gt_candidates,
)


def test_build_split_manifest_keeps_reserve_out_of_discovery() -> None:
    cohort = {
        "selected_cases": [
            {"image_id": 11, "bucket": "dup"},
            {"image_id": 22, "bucket": "low_recall"},
        ],
        "heldout_reserve": [
            {"image_id": 33, "bucket": "dup"},
        ],
    }
    val200_ids = [11, 22, 33, 44]

    manifest = build_split_manifest(cohort=cohort, val200_image_ids=val200_ids, discovery_limit=2)

    assert manifest["splits"]["discovery"] == [11, 22]
    assert manifest["splits"]["reserve"] == [33]
    assert manifest["splits"]["val200_remainder"] == [44]
    assert manifest["by_image_id"]["33"]["split"] == "reserve"


def test_materialize_gt_candidates_preserves_box_and_density(tmp_path: Path) -> None:
    source = tmp_path / "val.coord.jsonl"
    source.write_text(
        json.dumps(
            {
                "image_id": 7,
                "objects": [
                    {"desc": "cup", "bbox_2d": ["<|coord_10|>", "<|coord_11|>", "<|coord_20|>", "<|coord_21|>"]},
                    {"desc": "cup", "bbox_2d": ["<|coord_40|>", "<|coord_41|>", "<|coord_50|>", "<|coord_51|>"]},
                    {"desc": "bowl", "bbox_2d": ["<|coord_70|>", "<|coord_71|>", "<|coord_80|>", "<|coord_81|>"]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = materialize_gt_candidates(gt_jsonl=source, image_ids=[7])

    assert rows["7"][0]["gt_idx"] == 0
    assert rows["7"][0]["desc"] == "cup"
    assert rows["7"][0]["bbox"] == [10, 11, 20, 21]
    assert rows["7"][0]["same_class_gt_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_phase3_manifest.py -q
```

Expected: fail with missing `phase3_manifest`.

- [ ] **Step 3: Implement manifest and candidate materializer**

Create `src/analysis/autoregressive_binding_template_ablation/phase3_manifest.py` with these public functions:

```python
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


_COORD_RE = re.compile(r"<\|coord_(\d+)\|>")


def build_split_manifest(
    *,
    cohort: Mapping[str, Any],
    val200_image_ids: Sequence[int],
    discovery_limit: int | None = None,
) -> dict[str, Any]:
    selected = [int(item["image_id"]) for item in cohort.get("selected_cases", [])]
    reserve = [int(item["image_id"]) for item in cohort.get("heldout_reserve", [])]
    if discovery_limit is not None:
        selected = selected[:discovery_limit]
    selected_set = set(selected)
    reserve_set = set(reserve)
    overlap = selected_set & reserve_set
    if overlap:
        raise ValueError(f"image ids appear in both discovery and reserve: {sorted(overlap)}")
    remainder = [int(image_id) for image_id in val200_image_ids if int(image_id) not in selected_set | reserve_set]
    by_image_id: dict[str, dict[str, Any]] = {}
    for split, image_ids in {
        "discovery": selected,
        "reserve": reserve,
        "val200_remainder": remainder,
    }.items():
        for image_id in image_ids:
            by_image_id[str(image_id)] = {"image_id": image_id, "split": split}
    return {
        "schema_version": 1,
        "splits": {"discovery": selected, "reserve": reserve, "val200_remainder": remainder},
        "by_image_id": by_image_id,
    }


def materialize_gt_candidates(*, gt_jsonl: str | Path, image_ids: Sequence[int]) -> dict[str, list[dict[str, Any]]]:
    wanted = {int(image_id) for image_id in image_ids}
    output: dict[str, list[dict[str, Any]]] = {}
    with Path(gt_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id not in wanted:
                continue
            gt_rows = row.get("objects") or row.get("gt") or []
            counts = Counter(str(item.get("desc", "")) for item in gt_rows)
            candidates: list[dict[str, Any]] = []
            for gt_idx, item in enumerate(gt_rows):
                desc = str(item.get("desc", ""))
                raw_bbox = item.get("bbox") or item.get("points") or item.get("bbox_2d")
                bbox = []
                for value in raw_bbox:
                    if isinstance(value, str):
                        match = _COORD_RE.fullmatch(value)
                        if match is None:
                            raise ValueError(f"not a coord token: {value}")
                        bbox.append(int(match.group(1)))
                    else:
                        bbox.append(float(value) if isinstance(value, float) else int(value))
                candidates.append(
                    {
                        "gt_idx": gt_idx,
                        "desc": desc,
                        "bbox": bbox,
                        "same_class_gt_count": int(counts[desc]),
                    }
                )
            output[str(image_id)] = candidates
    missing = sorted(wanted - {int(key) for key in output})
    if missing:
        raise ValueError(f"missing gt rows for image ids: {missing}")
    return output
```

- [ ] **Step 4: Add manifest CLI**

Create `scripts/analysis/run_autoregressive_binding_phase3_manifest.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.autoregressive_binding_template_ablation.phase3_manifest import (
    build_split_manifest,
    materialize_gt_candidates,
)


def _read_val200_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(int(json.loads(line)["image_id"]))
            if len(ids) >= 200:
                break
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--discovery-limit", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    val200_ids = _read_val200_ids(Path(cfg["study"]["canonical_gt_jsonl"]))
    split_manifest = build_split_manifest(
        cohort=cohort,
        val200_image_ids=val200_ids,
        discovery_limit=args.discovery_limit,
    )
    all_ids = split_manifest["splits"]["discovery"] + split_manifest["splits"]["reserve"] + split_manifest["splits"]["val200_remainder"]
    gt_candidates = materialize_gt_candidates(gt_jsonl=cfg["study"]["canonical_gt_jsonl"], image_ids=all_ids)
    probe_cfg = dict(cfg)
    probe_cfg["probe"] = dict(cfg.get("probe") or {})
    probe_cfg["probe"]["case_image_ids"] = all_ids
    probe_cfg["probe"]["output_root"] = str(args.output_root / "scaled_probe_val200")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "gt_candidates.json").write_text(json.dumps(gt_candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "phase3_probe_config.yaml").write_text(yaml.safe_dump(probe_cfg, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "split_manifest": str(args.output_root / "split_manifest.json"),
                "gt_candidates": str(args.output_root / "gt_candidates.json"),
                "phase3_probe_config": str(args.output_root / "phase3_probe_config.yaml"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_phase3_manifest.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/phase3_manifest.py scripts/analysis/run_autoregressive_binding_phase3_manifest.py tests/analysis/test_autoregressive_binding_template_phase3_manifest.py
git commit -m "feat: add binding phase3 split manifest"
```

---

### Task 2: Repetition-Penalty Consistency Gate

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/repetition_penalty_gate.py`
- Create: `scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py`
- Test: `tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py`

- [ ] **Step 1: Write the failing repetition-penalty tests**

Create `tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py`:

```python
from __future__ import annotations

import math

from src.analysis.autoregressive_binding_template_ablation.repetition_penalty_gate import (
    apply_repetition_penalty_to_coord_logits,
    summarize_penalty_shift,
)


def test_apply_repetition_penalty_matches_positive_logit_rule() -> None:
    logits = {10: 4.0, 20: -2.0, 30: 1.0}
    adjusted = apply_repetition_penalty_to_coord_logits(logits=logits, repeated_bins={10, 20}, penalty=2.0)

    assert adjusted[10] == 2.0
    assert adjusted[20] == -4.0
    assert adjusted[30] == 1.0


def test_summarize_penalty_shift_reports_rank_change() -> None:
    raw_top_bins = [
        {"bin": 10, "logit": 4.0},
        {"bin": 30, "logit": 3.0},
        {"bin": 40, "logit": 2.0},
    ]

    summary = summarize_penalty_shift(
        raw_top_bins=raw_top_bins,
        emitted_bin=10,
        repeated_bins={10},
        penalty=2.0,
    )

    assert summary["raw_rank_emitted"] == 1
    assert summary["penalty_rank_emitted"] == 3
    assert summary["rank_delta"] == 2
    assert math.isclose(summary["penalty_logits_by_bin"]["10"], 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py -q
```

Expected: fail with missing `repetition_penalty_gate`.

- [ ] **Step 3: Implement the gate helper**

Create `src/analysis/autoregressive_binding_template_ablation/repetition_penalty_gate.py` with:

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def apply_repetition_penalty_to_coord_logits(
    *,
    logits: Mapping[int, float],
    repeated_bins: Iterable[int],
    penalty: float,
) -> dict[int, float]:
    repeated = {int(value) for value in repeated_bins}
    adjusted: dict[int, float] = {}
    for key, value in logits.items():
        bin_id = int(key)
        logit = float(value)
        if bin_id in repeated:
            adjusted[bin_id] = logit / penalty if logit >= 0 else logit * penalty
        else:
            adjusted[bin_id] = logit
    return adjusted


def _rank_of(logits: Mapping[int, float], emitted_bin: int) -> int | None:
    ordered = sorted(((float(v), int(k)) for k, v in logits.items()), reverse=True)
    for rank, (_, bin_id) in enumerate(ordered, start=1):
        if bin_id == int(emitted_bin):
            return rank
    return None


def _softmax_prob(logits: Mapping[int, float], emitted_bin: int) -> float | None:
    if int(emitted_bin) not in logits:
        return None
    max_logit = max(float(value) for value in logits.values())
    denom = sum(math.exp(float(value) - max_logit) for value in logits.values())
    return math.exp(float(logits[int(emitted_bin)]) - max_logit) / denom


def _top_bins_with_probs(logits: Mapping[int, float]) -> list[dict[str, float | int]]:
    max_logit = max(float(value) for value in logits.values())
    denom = sum(math.exp(float(value) - max_logit) for value in logits.values())
    rows = []
    for bin_id, value in sorted(logits.items(), key=lambda item: float(item[1]), reverse=True):
        prob = math.exp(float(value) - max_logit) / denom
        rows.append({"bin": int(bin_id), "logit": float(value), "penalty_prob_cond": prob})
    return rows


def summarize_penalty_shift(
    *,
    raw_top_bins: Sequence[Mapping[str, Any]],
    emitted_bin: int,
    repeated_bins: Iterable[int],
    penalty: float,
) -> dict[str, Any]:
    raw_logits = {int(item["bin"]): float(item["logit"]) for item in raw_top_bins}
    penalty_logits = apply_repetition_penalty_to_coord_logits(
        logits=raw_logits,
        repeated_bins=repeated_bins,
        penalty=penalty,
    )
    raw_rank = _rank_of(raw_logits, emitted_bin)
    penalty_rank = _rank_of(penalty_logits, emitted_bin)
    raw_prob = _softmax_prob(raw_logits, emitted_bin)
    penalty_prob = _softmax_prob(penalty_logits, emitted_bin)
    return {
        "raw_rank_emitted": raw_rank,
        "penalty_rank_emitted": penalty_rank,
        "rank_delta": None if raw_rank is None or penalty_rank is None else penalty_rank - raw_rank,
        "raw_p_emitted_topk": raw_prob,
        "penalty_p_emitted_topk": penalty_prob,
        "p_emitted_delta_topk": None if raw_prob is None or penalty_prob is None else penalty_prob - raw_prob,
        "penalty_logits_by_bin": {str(key): value for key, value in sorted(penalty_logits.items())},
        "penalty_top_bins": _top_bins_with_probs(penalty_logits),
    }


def build_penalty_gate_rows(
    *,
    coord_rows: Sequence[Mapping[str, Any]],
    repeated_bins_by_step: Mapping[tuple[str, int, int, int], set[int]],
    penalty: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in coord_rows:
        if row.get("slot") != "x1":
            continue
        key = (str(row["family"]), int(row["image_id"]), int(row["source_line_idx"]), int(row["object_idx"]))
        summary = summarize_penalty_shift(
            raw_top_bins=row.get("top_bins") or [],
            emitted_bin=int(row["emitted_bin"]),
            repeated_bins=repeated_bins_by_step.get(key, set()),
            penalty=penalty,
        )
        rows.append({"schema_version": 1, **{k: row.get(k) for k in ("family", "image_id", "source_line_idx", "object_idx", "slot")}, **summary})
    return rows
```

- [ ] **Step 4: Add CLI for existing artifacts**

Create `scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py`.

Required behavior:

```text
inputs:
  --probe-root
  --config
  --output-root
reads:
  coord_logit_rows.jsonl
  object_step_rows.jsonl
writes:
  repetition_penalty_gate_rows.jsonl
  repetition_penalty_gate_summary.json
```

The CLI must derive repeated x1 bins from prior parsed object rows in the same `(family, image_id, source_line_idx)` sequence, using `bbox_tokens[0]` when present and rounded `points[0]` only as a fallback. Each row must include `penalty_top_bins` with `penalty_prob_cond` for Task 5. The summary must report duplicate-vs-new raw and penalty-adjusted mean rank, mean top-k probability, and whether the duplicate gap direction survives.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/repetition_penalty_gate.py scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py
git commit -m "feat: add repetition penalty binding gate"
```

---

### Task 3: Anchor Circularity And Upstream Onset Gate

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/anchor_circularity_gate.py`
- Create: `scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py`
- Test: `tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py`

- [ ] **Step 1: Write the failing anchor diagnostics test**

Create `tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.anchor_circularity_gate import (
    anchor_distances_for_step,
    build_upstream_onset_labels,
)


def test_anchor_distances_separate_immediate_nearest_and_any_prior() -> None:
    prior = [
        {"object_idx": 0, "desc": "cup", "x1": 10},
        {"object_idx": 1, "desc": "bowl", "x1": 80},
        {"object_idx": 2, "desc": "cup", "x1": 22},
    ]
    current = {"object_idx": 3, "desc": "cup", "x1": 24}

    out = anchor_distances_for_step(current=current, prior=prior)

    assert out["immediate_prev_x1_delta"] == 2
    assert out["nearest_same_desc_x1_delta"] == 2
    assert out["any_prior_min_x1_delta"] == 2


def test_build_upstream_onset_labels_assigns_label_to_previous_step() -> None:
    rows = [
        {"family": "desc_first", "image_id": 1, "source_line_idx": 0, "object_idx": 0, "prediction_kind": "new_gt"},
        {"family": "desc_first", "image_id": 1, "source_line_idx": 0, "object_idx": 1, "prediction_kind": "duplicate_iou70"},
    ]

    labels = build_upstream_onset_labels(rows)

    assert labels[0]["object_idx"] == 0
    assert labels[0]["next_step_duplicate_onset"] is True
    assert labels[0]["next_object_idx"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py -q
```

Expected: fail with missing `anchor_circularity_gate`.

- [ ] **Step 3: Implement de-circularized anchor helpers**

Create `src/analysis/autoregressive_binding_template_ablation/anchor_circularity_gate.py` with:

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def anchor_distances_for_step(
    *,
    current: Mapping[str, Any],
    prior: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current_x1 = float(current["x1"])
    immediate = prior[-1] if prior else None
    same_desc = [row for row in prior if row.get("desc") == current.get("desc")]
    all_deltas = [abs(current_x1 - float(row["x1"])) for row in prior]
    same_deltas = [abs(current_x1 - float(row["x1"])) for row in same_desc]
    return {
        "immediate_prev_x1_delta": None if immediate is None else abs(current_x1 - float(immediate["x1"])),
        "nearest_same_desc_x1_delta": None if not same_deltas else min(same_deltas),
        "any_prior_min_x1_delta": None if not all_deltas else min(all_deltas),
        "prior_count": len(prior),
        "same_desc_prior_count": len(same_desc),
    }


def build_upstream_onset_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def is_duplicate(row: Mapping[str, Any]) -> bool:
        return (
            row.get("prediction_kind") == "duplicate_iou70"
            or bool(row.get("duplicate_of_previous_iou70"))
            or row.get("sequential_match_kind") == "repeated_gt"
        )

    def is_unmatched(row: Mapping[str, Any]) -> bool:
        return row.get("prediction_kind") == "unmatched" or row.get("sequential_match_kind") == "unmatched"

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), int(row["image_id"]), int(row["source_line_idx"]))].append(row)
    output: list[dict[str, Any]] = []
    for key_rows in grouped.values():
        ordered = sorted(key_rows, key=lambda item: int(item["object_idx"]))
        for idx, row in enumerate(ordered[:-1]):
            nxt = ordered[idx + 1]
            output.append(
                {
                    "schema_version": 1,
                    "family": row["family"],
                    "image_id": int(row["image_id"]),
                    "source_line_idx": int(row["source_line_idx"]),
                    "object_idx": int(row["object_idx"]),
                    "next_object_idx": int(nxt["object_idx"]),
                    "next_step_duplicate_onset": is_duplicate(nxt),
                    "next_step_unmatched_onset": is_unmatched(nxt),
                }
            )
    return output
```

- [ ] **Step 4: Add anchor gate CLI**

Create `scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py`.

Required behavior:

```text
inputs:
  --probe-root
  --split-manifest
  --output-root
reads:
  object_step_rows.jsonl
writes:
  anchor_circularity_rows.jsonl
  upstream_onset_label_rows.jsonl
  anchor_circularity_summary.json
```

The summary must report immediate, nearest same-desc, and any-prior metrics separately for `focus_only=true` and `focus_only=false` when all-object rows are available. It must not score `previous_emitted_mass` directly against `duplicate_iou70`; scoring belongs to Task 6 with upstream labels.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/anchor_circularity_gate.py scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py
git commit -m "feat: add anchor circularity gate"
```

---

### Task 4: Audit-Repaired Candidate Ledger

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/candidate_ledger.py`
- Create: `scripts/analysis/run_autoregressive_binding_candidate_ledger.py`
- Test: `tests/analysis/test_autoregressive_binding_template_candidate_ledger.py`

- [ ] **Step 1: Write the failing candidate ledger test**

Create `tests/analysis/test_autoregressive_binding_template_candidate_ledger.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.candidate_ledger import (
    family_selection_events,
    build_candidate_step_rows,
)


def test_family_selection_events_are_template_specific() -> None:
    assert family_selection_events("desc_first") == ["pre_desc", "desc_end"]
    assert family_selection_events("geometry_first") == ["pre_box_start", "pre_x1"]


def test_candidate_rows_mark_target_density_and_coverage() -> None:
    object_rows = [
        {
            "family": "desc_first",
            "image_id": 1,
            "source_line_idx": 0,
            "object_idx": 0,
            "desc": "cup",
            "sequential_gt_idx": 0,
            "sequential_match_kind": "new_gt",
            "remaining_gt_before": 2,
            "points": [10, 10, 20, 20],
        }
    ]
    gt_by_image = {
        "1": [
            {"gt_idx": 0, "desc": "cup", "bbox": [10, 10, 20, 20], "same_class_gt_count": 2},
            {"gt_idx": 1, "desc": "cup", "bbox": [50, 10, 60, 20], "same_class_gt_count": 2},
        ]
    }
    split_by_image = {"1": {"split": "discovery"}}

    rows = build_candidate_step_rows(
        object_rows=object_rows,
        gt_by_image=gt_by_image,
        split_by_image=split_by_image,
    )

    target = [row for row in rows if row["candidate_gt_idx"] == 0][0]
    remaining = [row for row in rows if row["candidate_gt_idx"] == 1][0]
    assert target["is_target"] is True
    assert remaining["candidate_role"] == "remaining_gt"
    assert remaining["same_class_gt_count"] == 2
    assert target["selection_events"] == ["pre_desc", "desc_end"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_candidate_ledger.py -q
```

Expected: fail with missing `candidate_ledger`.

- [ ] **Step 3: Implement repaired candidate ledger**

Create `src/analysis/autoregressive_binding_template_ablation/candidate_ledger.py`.

Public API:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def family_selection_events(family: str) -> list[str]:
    if family == "desc_first":
        return ["pre_desc", "desc_end"]
    if family == "geometry_first":
        return ["pre_box_start", "pre_x1"]
    raise ValueError(f"unknown family: {family}")


def build_candidate_step_rows(
    *,
    object_rows: Sequence[Mapping[str, Any]],
    gt_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
    split_by_image: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    emitted_gt_by_sequence: dict[tuple[str, int, int], set[int]] = {}
    for obj in sorted(object_rows, key=lambda item: (str(item["family"]), int(item["image_id"]), int(item["source_line_idx"]), int(item["object_idx"]))):
        key = (str(obj["family"]), int(obj["image_id"]), int(obj["source_line_idx"]))
        emitted = emitted_gt_by_sequence.setdefault(key, set())
        image_key = str(int(obj["image_id"]))
        split = split_by_image.get(image_key, {}).get("split", "unknown")
        sequential_gt_idx = obj.get("sequential_gt_idx")
        prediction_kind = "duplicate_iou70" if obj.get("duplicate_of_previous_iou70") else obj.get("prediction_kind") or obj.get("sequential_match_kind")
        for cand in gt_by_image.get(image_key, []):
            gt_idx = int(cand["gt_idx"])
            is_target = sequential_gt_idx is not None and int(sequential_gt_idx) == gt_idx
            already_emitted = gt_idx in emitted
            role = "target_gt" if is_target else "previously_covered_gt" if already_emitted else "remaining_gt"
            x1, y1, x2, y2 = cand["bbox"]
            rows.append(
                {
                    "schema_version": 2,
                    "family": obj["family"],
                    "image_id": int(obj["image_id"]),
                    "source_line_idx": int(obj["source_line_idx"]),
                    "object_idx": int(obj["object_idx"]),
                    "split": split,
                    "selection_events": family_selection_events(str(obj["family"])),
                    "pred_desc": obj.get("desc"),
                    "prediction_kind": prediction_kind,
                    "sequential_match_kind": obj.get("sequential_match_kind"),
                    "sequential_gt_idx": sequential_gt_idx,
                    "candidate_role": role,
                    "candidate_gt_idx": gt_idx,
                    "candidate_desc": cand.get("desc"),
                    "candidate_x1": x1,
                    "candidate_y1": y1,
                    "candidate_x2": x2,
                    "candidate_y2": y2,
                    "same_desc_as_prediction": cand.get("desc") == obj.get("desc"),
                    "same_class_gt_count": cand.get("same_class_gt_count"),
                    "is_target": is_target,
                    "already_emitted_by_sequential_greedy": already_emitted,
                    "remaining_gt_before": obj.get("remaining_gt_before"),
                }
            )
        if sequential_gt_idx is not None and obj.get("sequential_match_kind") in {"new_gt", "repeated_gt"}:
            emitted.add(int(sequential_gt_idx))
    return rows
```

- [ ] **Step 4: Add ledger CLI**

Create `scripts/analysis/run_autoregressive_binding_candidate_ledger.py`.

Required behavior:

```text
inputs:
  --probe-root
  --split-manifest
  --gt-json
  --output-root
reads:
  object_step_rows.jsonl
writes:
  candidate_step_rows.jsonl
  candidate_ledger_summary.json
```

The summary must contain row counts by split, family, prediction kind, candidate role, and `focus_object` when that field exists.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_candidate_ledger.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/candidate_ledger.py scripts/analysis/run_autoregressive_binding_candidate_ledger.py tests/analysis/test_autoregressive_binding_template_candidate_ledger.py
git commit -m "feat: add audit repaired candidate ledger"
```

---

### Task 5: Penalty-Aware Coverage Polarity

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/coverage_polarity.py`
- Create: `scripts/analysis/run_autoregressive_binding_coverage_polarity.py`
- Test: `tests/analysis/test_autoregressive_binding_template_coverage_polarity.py`

- [ ] **Step 1: Write failing mass aggregation test**

Create `tests/analysis/test_autoregressive_binding_template_coverage_polarity.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.coverage_polarity import (
    summarize_coord_candidate_mass,
)


def test_coord_candidate_mass_uses_penalty_adjusted_bins_and_radius() -> None:
    coord_row = {
        "slot": "x1",
        "top_bins": [
            {"bin": 10, "prob_cond": 0.30, "penalty_prob_cond": 0.15},
            {"bin": 11, "prob_cond": 0.10, "penalty_prob_cond": 0.05},
            {"bin": 50, "prob_cond": 0.20, "penalty_prob_cond": 0.40},
            {"bin": 90, "prob_cond": 0.05, "penalty_prob_cond": 0.05},
        ],
    }
    candidates = [
        {"candidate_role": "previously_covered_gt", "candidate_x1": 10, "candidate_desc": "cup"},
        {"candidate_role": "remaining_gt", "candidate_x1": 50, "candidate_desc": "cup"},
    ]

    summary = summarize_coord_candidate_mass(coord_row=coord_row, candidate_rows=candidates, radius=1, prob_key="penalty_prob_cond")

    assert summary["previous_anchor_mass"] == 0.20
    assert summary["remaining_gt_mass"] == 0.40
    assert summary["best_role"] == "remaining_gt"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_coverage_polarity.py -q
```

Expected: fail with missing `coverage_polarity`.

- [ ] **Step 3: Implement penalty-aware mass summary**

Create `src/analysis/autoregressive_binding_template_ablation/coverage_polarity.py`.

Required public API:

```python
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def summarize_coord_candidate_mass(
    *,
    coord_row: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    radius: int,
    prob_key: str,
) -> dict[str, Any]:
    masses = {"previous_anchor_mass": 0.0, "remaining_gt_mass": 0.0, "target_gt_mass": 0.0}
    assigned_bins: set[int] = set()
    for candidate in candidate_rows:
        center = int(round(float(candidate["candidate_x1"])))
        role = str(candidate["candidate_role"])
        out_key = "remaining_gt_mass"
        if role in {"previously_covered_gt", "previous_emitted"}:
            out_key = "previous_anchor_mass"
        elif role == "target_gt":
            out_key = "target_gt_mass"
        for item in coord_row.get("top_bins") or []:
            bin_id = int(item["bin"])
            if abs(bin_id - center) <= radius:
                masses[out_key] += float(item.get(prob_key, item.get("prob_cond", 0.0)))
                assigned_bins.add(bin_id)
    total_top_mass = sum(float(item.get(prob_key, item.get("prob_cond", 0.0))) for item in coord_row.get("top_bins") or [])
    nonzero = [value for value in masses.values() if value > 0]
    entropy = -sum((value / sum(nonzero)) * math.log(value / sum(nonzero)) for value in nonzero) if nonzero else 0.0
    best_role = max(masses.items(), key=lambda item: item[1])[0]
    return {**masses, "unassigned_topk_mass": max(0.0, total_top_mass - sum(masses.values())), "fragmentation_index": entropy, "best_role": best_role}
```

- [ ] **Step 4: Add coverage CLI**

Create `scripts/analysis/run_autoregressive_binding_coverage_polarity.py`.

Required behavior:

```text
inputs:
  --candidate-rows
  --coord-rows
  --penalty-gate-rows
  --output-root
  --radius
writes:
  coverage_polarity_rows.jsonl
  coverage_polarity_summary.json
```

The CLI must write separate rows for raw probabilities and repetition-penalty-adjusted probabilities. Each row must preserve `split`, `family`, `selection_event`, `prediction_kind`, `object_idx`, `remaining_gt_before`, and `same_class_gt_count`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_coverage_polarity.py -q
```

Expected: `1 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/coverage_polarity.py scripts/analysis/run_autoregressive_binding_coverage_polarity.py tests/analysis/test_autoregressive_binding_template_coverage_polarity.py
git commit -m "feat: add penalty aware coverage polarity"
```

---

### Task 6: Held-Out Null Leaderboard

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/null_leaderboard.py`
- Create: `scripts/analysis/run_autoregressive_binding_null_leaderboard.py`
- Test: `tests/analysis/test_autoregressive_binding_template_null_leaderboard.py`

- [ ] **Step 1: Write failing held-out leaderboard test**

Create `tests/analysis/test_autoregressive_binding_template_null_leaderboard.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.null_leaderboard import (
    score_binary_signal,
    split_leaderboard,
)


def test_score_binary_signal_reports_auc_accuracy_and_locked_threshold() -> None:
    rows = [
        {"label": 0, "score": 0.1},
        {"label": 0, "score": 0.2},
        {"label": 1, "score": 0.8},
        {"label": 1, "score": 0.9},
    ]

    result = score_binary_signal(rows, label_key="label", score_key="score")

    assert result["count"] == 4
    assert result["auc"] == 1.0
    assert result["best_accuracy"] == 1.0
    assert result["best_threshold"] == 0.8


def test_split_leaderboard_locks_threshold_on_discovery() -> None:
    rows = [
        {"split": "discovery", "label": 0, "score": 0.1},
        {"split": "discovery", "label": 1, "score": 0.9},
        {"split": "reserve", "label": 0, "score": 0.2},
        {"split": "reserve", "label": 1, "score": 0.95},
    ]

    out = split_leaderboard(rows, label_key="label", score_keys=["score"])

    assert out[0]["signal"] == "score"
    assert out[0]["discovery"]["best_threshold"] == 0.9
    assert out[0]["reserve"]["locked_threshold_accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_null_leaderboard.py -q
```

Expected: fail with missing `null_leaderboard`.

- [ ] **Step 3: Implement leaderboard helpers**

Create `src/analysis/autoregressive_binding_template_ablation/null_leaderboard.py`.

Implementation requirements:

```text
score_binary_signal:
  pairwise AUC without scikit-learn
  best threshold scans unique score values
  returns count, positive_count, negative_count, auc, best_accuracy, best_threshold

split_leaderboard:
  tunes threshold only on discovery rows
  applies the discovery threshold to reserve and val200_remainder rows
  reports each family separately and all-family pooled only as secondary
```

Score keys to include in the CLI:

```text
object_idx
remaining_gt_before
same_class_gt_count
previous_anchor_mass
remaining_gt_mass
fragmentation_index
raw_rank_emitted
penalty_rank_emitted
rank_delta
raw_p_emitted_topk
penalty_p_emitted_topk
```

Targets to include in the CLI:

```text
next_step_duplicate_onset
next_step_unmatched_onset
```

- [ ] **Step 4: Add null leaderboard CLI**

Create `scripts/analysis/run_autoregressive_binding_null_leaderboard.py`.

Required behavior:

```text
inputs:
  --coverage-rows
  --upstream-label-rows
  --penalty-gate-rows
  --output-root
writes:
  null_leaderboard.json
  null_leaderboard_summary.md
```

The CLI must refuse to run if reserve rows are missing:

```text
raise SystemExit("reserve split is required for held-out leaderboard")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_null_leaderboard.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/null_leaderboard.py scripts/analysis/run_autoregressive_binding_null_leaderboard.py tests/analysis/test_autoregressive_binding_template_null_leaderboard.py
git commit -m "feat: add held out binding null leaderboard"
```

---

### Task 7: Same-Class Identity Posterior

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/identity_posterior.py`
- Create: `scripts/analysis/run_autoregressive_binding_identity_posterior.py`
- Test: `tests/analysis/test_autoregressive_binding_template_identity_posterior.py`

- [ ] **Step 1: Write failing identity posterior test**

Create `tests/analysis/test_autoregressive_binding_template_identity_posterior.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.identity_posterior import (
    same_class_candidate_margins,
)


def test_same_class_candidate_margins_identify_target_margin() -> None:
    rows = [
        {"candidate_gt_idx": 0, "candidate_desc": "cup", "mass": 0.60, "is_target": True},
        {"candidate_gt_idx": 1, "candidate_desc": "cup", "mass": 0.25, "is_target": False},
        {"candidate_gt_idx": 2, "candidate_desc": "bowl", "mass": 0.10, "is_target": False},
    ]

    result = same_class_candidate_margins(rows, desc="cup")

    assert result["same_class_count"] == 2
    assert result["target_mass"] == 0.60
    assert result["best_competitor_mass"] == 0.25
    assert result["target_margin"] == 0.35
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_identity_posterior.py -q
```

Expected: fail with missing `identity_posterior`.

- [ ] **Step 3: Implement identity posterior helpers**

Create `src/analysis/autoregressive_binding_template_ablation/identity_posterior.py`.

Public API:

```python
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def same_class_candidate_margins(rows: Sequence[Mapping[str, Any]], *, desc: str) -> dict[str, Any]:
    same = [row for row in rows if row.get("candidate_desc") == desc]
    target = [row for row in same if row.get("is_target")]
    target_mass = max((float(row.get("mass", 0.0)) for row in target), default=0.0)
    competitors = [float(row.get("mass", 0.0)) for row in same if not row.get("is_target")]
    best_competitor = max(competitors, default=0.0)
    masses = [float(row.get("mass", 0.0)) for row in same if float(row.get("mass", 0.0)) > 0]
    total = sum(masses)
    entropy = -sum((value / total) * math.log(value / total) for value in masses) if total > 0 else 0.0
    return {
        "same_class_count": len(same),
        "target_mass": target_mass,
        "best_competitor_mass": best_competitor,
        "target_margin": round(target_mass - best_competitor, 12),
        "same_class_entropy": entropy,
    }
```

- [ ] **Step 4: Add identity posterior CLI**

Create `scripts/analysis/run_autoregressive_binding_identity_posterior.py`.

Required behavior:

```text
inputs:
  --candidate-rows
  --coverage-rows
  --output-root
writes:
  identity_posterior_rows.jsonl
  identity_posterior_summary.json
```

Rows must be keyed by family-specific selection event, not a universal `pre_x1`. Summaries must be split by family and split.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_identity_posterior.py -q
```

Expected: `1 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/identity_posterior.py scripts/analysis/run_autoregressive_binding_identity_posterior.py tests/analysis/test_autoregressive_binding_template_identity_posterior.py
git commit -m "feat: add same class binding posterior"
```

---

### Task 8: Behavioral Causal Patch Smoke

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py`
- Create: `scripts/analysis/run_autoregressive_binding_behavioral_patch.py`
- Test: `tests/analysis/test_autoregressive_binding_template_behavioral_patch.py`

- [ ] **Step 1: Write failing behavioral readout tests**

Create `tests/analysis/test_autoregressive_binding_template_behavioral_patch.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.behavioral_patch import (
    patch_gate_allows_behavioral_smoke,
    summarize_behavior_change,
)


def test_patch_gate_requires_pre_gates_and_heldout_win() -> None:
    gates = {
        "rep_penalty_survives": True,
        "anchor_circularity_cleared": True,
        "held_out_beats_nulls": True,
    }

    assert patch_gate_allows_behavioral_smoke(gates) is True


def test_summarize_behavior_change_requires_next_object_change() -> None:
    before = {"next_prediction_kind": "duplicate_iou70", "next_gt_idx": None, "parse_ok": True}
    after = {"next_prediction_kind": "new_gt", "next_gt_idx": 3, "parse_ok": True}

    result = summarize_behavior_change(before=before, after=after)

    assert result["changed_next_object_behavior"] is True
    assert result["improved_duplicate_to_new_gt"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_behavioral_patch.py -q
```

Expected: fail with missing `behavioral_patch`.

- [ ] **Step 3: Implement behavioral patch gate and readout**

Create `src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py`.

Required public API:

```python
from __future__ import annotations

from typing import Any, Mapping


def patch_gate_allows_behavioral_smoke(gates: Mapping[str, bool]) -> bool:
    required = ["rep_penalty_survives", "anchor_circularity_cleared", "held_out_beats_nulls"]
    return all(bool(gates.get(key)) for key in required)


def summarize_behavior_change(*, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    changed = (
        before.get("next_prediction_kind") != after.get("next_prediction_kind")
        or before.get("next_gt_idx") != after.get("next_gt_idx")
    )
    return {
        "changed_next_object_behavior": changed,
        "improved_duplicate_to_new_gt": before.get("next_prediction_kind") == "duplicate_iou70" and after.get("next_prediction_kind") == "new_gt",
        "parse_preserved": bool(before.get("parse_ok")) and bool(after.get("parse_ok")),
    }
```

- [ ] **Step 4: Add behavioral patch CLI**

Create `scripts/analysis/run_autoregressive_binding_behavioral_patch.py`.

Required behavior:

```text
inputs:
  --config
  --gate-summary
  --patch-cases
  --output-root
  --limit-cases
  --stage plan
  --stage short-rollout
writes for plan:
  behavioral_patch_plan.json
writes for short-rollout:
  behavioral_patch_rows.jsonl
  behavioral_patch_summary.json
```

The CLI must refuse `--stage short-rollout` unless `patch_gate_allows_behavioral_smoke` returns true. The short rollout must score next-object behavior, not only local candidate mass. Controls must include:

```text
self_noop
wrong_role
wrong_image_same_family
post_commit_too_late
same_image_wrong_object_idx
candidate_shuffled_labels
norm_matched_random
rep_penalty_on
rep_penalty_off
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_behavioral_patch.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py scripts/analysis/run_autoregressive_binding_behavioral_patch.py tests/analysis/test_autoregressive_binding_template_behavioral_patch.py
git commit -m "feat: add gated behavioral patch smoke"
```

---

### Task 9: Audit-Adjusted Execution And Findings

**Files:**
- Create: `progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md`
- Modify: `progress/diagnostics/README.md`
- Modify: `progress/index.yaml`

- [ ] **Step 1: Materialize split manifest and scaled observational probe**

Run:

```bash
PHASE3=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
COHORT=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_initial_cohort/cohort_manifest.json

python scripts/analysis/run_autoregressive_binding_phase3_manifest.py \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --cohort-manifest "$COHORT" \
  --output-root "$PHASE3"

PROBE="$PHASE3/scaled_probe_val200"

python scripts/analysis/run_autoregressive_binding_template_deep_probe.py \
  --config "$PHASE3/phase3_probe_config.yaml" \
  --stage surface \
  --output-root "$PROBE"

# Launch the eight replay shard commands concurrently in separate shells or tmux panes.
# Run the merge command only after all eight replay commands exit 0.

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 0 --num-shards 8
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 1 --num-shards 8
CUDA_VISIBLE_DEVICES=2 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 2 --num-shards 8
CUDA_VISIBLE_DEVICES=3 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 3 --num-shards 8
CUDA_VISIBLE_DEVICES=4 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 4 --num-shards 8
CUDA_VISIBLE_DEVICES=5 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 5 --num-shards 8
CUDA_VISIBLE_DEVICES=6 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 6 --num-shards 8
CUDA_VISIBLE_DEVICES=7 python scripts/analysis/run_autoregressive_binding_template_deep_probe.py --config "$PHASE3/phase3_probe_config.yaml" --stage replay --output-root "$PROBE" --shard-index 7 --num-shards 8

python scripts/analysis/run_autoregressive_binding_template_deep_probe.py \
  --config "$PHASE3/phase3_probe_config.yaml" \
  --stage merge \
  --output-root "$PROBE" \
  --num-shards 8

python scripts/analysis/run_autoregressive_binding_template_deep_probe.py \
  --config "$PHASE3/phase3_probe_config.yaml" \
  --stage report \
  --output-root "$PROBE"
```

Expected:

```text
split_manifest.json
gt_candidates.json
phase3_probe_config.yaml
scaled_probe_val200/case_surface_rows.jsonl
scaled_probe_val200/object_step_rows.jsonl
scaled_probe_val200/coord_logit_rows.jsonl
scaled_probe_val200/replay_merge_summary.json
```

- [ ] **Step 2: Run deterministic pre-gates on the scaled probe**

Run:

```bash
PHASE3=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted
PROBE="$PHASE3/scaled_probe_val200"

python scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --probe-root "$PROBE" \
  --output-root "$PHASE3"

python scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py \
  --probe-root "$PROBE" \
  --split-manifest "$PHASE3/split_manifest.json" \
  --output-root "$PHASE3"
```

Expected:

```text
repetition_penalty_gate_rows.jsonl
repetition_penalty_gate_summary.json
anchor_circularity_rows.jsonl
upstream_onset_label_rows.jsonl
anchor_circularity_summary.json
```

- [ ] **Step 3: Run observational ledger and nulls only if P0 gates are not collapsed**

Run:

```bash
python scripts/analysis/run_autoregressive_binding_candidate_ledger.py \
  --probe-root "$PROBE" \
  --split-manifest "$PHASE3/split_manifest.json" \
  --gt-json "$PHASE3/gt_candidates.json" \
  --output-root "$PHASE3"

python scripts/analysis/run_autoregressive_binding_coverage_polarity.py \
  --candidate-rows "$PHASE3/candidate_step_rows.jsonl" \
  --coord-rows "$PROBE/coord_logit_rows.jsonl" \
  --penalty-gate-rows "$PHASE3/repetition_penalty_gate_rows.jsonl" \
  --output-root "$PHASE3" \
  --radius 4

python scripts/analysis/run_autoregressive_binding_null_leaderboard.py \
  --coverage-rows "$PHASE3/coverage_polarity_rows.jsonl" \
  --upstream-label-rows "$PHASE3/upstream_onset_label_rows.jsonl" \
  --penalty-gate-rows "$PHASE3/repetition_penalty_gate_rows.jsonl" \
  --output-root "$PHASE3"

python scripts/analysis/run_autoregressive_binding_identity_posterior.py \
  --candidate-rows "$PHASE3/candidate_step_rows.jsonl" \
  --coverage-rows "$PHASE3/coverage_polarity_rows.jsonl" \
  --output-root "$PHASE3"
```

Expected:

```text
candidate_step_rows.jsonl
candidate_ledger_summary.json
coverage_polarity_rows.jsonl
coverage_polarity_summary.json
null_leaderboard.json
null_leaderboard_summary.md
identity_posterior_rows.jsonl
identity_posterior_summary.json
```

- [ ] **Step 4: Run a gated behavioral patch smoke only if held-out nulls are beaten**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_behavioral_patch.py \
  --stage plan \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --gate-summary "$PHASE3/null_leaderboard.json" \
  --patch-cases "$PHASE3/identity_posterior_rows.jsonl" \
  --output-root "$PHASE3/behavioral_patch" \
  --limit-cases 4

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_behavioral_patch.py \
  --stage short-rollout \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --gate-summary "$PHASE3/null_leaderboard.json" \
  --patch-cases "$PHASE3/behavioral_patch/behavioral_patch_plan.json" \
  --output-root "$PHASE3/behavioral_patch" \
  --limit-cases 4
```

Expected if gate passes:

```text
behavioral_patch_plan.json
behavioral_patch_rows.jsonl
behavioral_patch_summary.json
```

If `--patch-cases` points at `identity_posterior_rows.jsonl`, the plan stage is
candidate selection only. The short-rollout stage must refuse those
candidate-only rows unless a separate realized before/after behavioral patch
case artifact has been materialized. Record that refusal as the honest Task 9
outcome, not as a model-result failure.

Expected if gate fails:

```text
short-rollout exits before model load with a message naming the failed gate
```

- [ ] **Step 5: Write findings note**

Create `progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md`.

Required sections:

```text
Scope
Audit Adjustments
Artifact Roots
Split Manifest
Repetition-Penalty Gate
Anchor Circularity Gate
Candidate Ledger Counts
Penalty-Aware Coverage Polarity
Held-Out Null Leaderboard
Same-Class Identity Posterior
Behavioral Patch Smoke
Promotion Decision
Boundaries
Next Branch
```

Promotion decision table:

```text
promote_to_scaled_behavioral_causal:
  condition: P0 gates survive, upstream signal beats nulls on reserve or val200, within-family signs match, short continuation changes next-object behavior, controls are null
hold_for_more_observation:
  condition: P0 gates survive but held-out effect is one-family-only, weak, or missing behavioral movement
demote_current_hypothesis:
  condition: repetition-penalty correction collapses the duplicate entropy/rank gap, anchor signal is circular, or held-out nulls win
```

- [ ] **Step 6: Link findings**

Add the findings note to:

```text
progress/diagnostics/README.md
progress/index.yaml
```

- [ ] **Step 7: Run verification**

Run:

```bash
python -m pytest \
  tests/analysis/test_autoregressive_binding_template_phase3_manifest.py \
  tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py \
  tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py \
  tests/analysis/test_autoregressive_binding_template_candidate_ledger.py \
  tests/analysis/test_autoregressive_binding_template_coverage_polarity.py \
  tests/analysis/test_autoregressive_binding_template_null_leaderboard.py \
  tests/analysis/test_autoregressive_binding_template_identity_posterior.py \
  tests/analysis/test_autoregressive_binding_template_behavioral_patch.py \
  -q
python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/phase3_manifest.py \
  src/analysis/autoregressive_binding_template_ablation/repetition_penalty_gate.py \
  src/analysis/autoregressive_binding_template_ablation/anchor_circularity_gate.py \
  src/analysis/autoregressive_binding_template_ablation/candidate_ledger.py \
  src/analysis/autoregressive_binding_template_ablation/coverage_polarity.py \
  src/analysis/autoregressive_binding_template_ablation/null_leaderboard.py \
  src/analysis/autoregressive_binding_template_ablation/identity_posterior.py \
  src/analysis/autoregressive_binding_template_ablation/behavioral_patch.py \
  scripts/analysis/run_autoregressive_binding_phase3_manifest.py \
  scripts/analysis/run_autoregressive_binding_repetition_penalty_gate.py \
  scripts/analysis/run_autoregressive_binding_anchor_circularity_gate.py \
  scripts/analysis/run_autoregressive_binding_candidate_ledger.py \
  scripts/analysis/run_autoregressive_binding_coverage_polarity.py \
  scripts/analysis/run_autoregressive_binding_null_leaderboard.py \
  scripts/analysis/run_autoregressive_binding_identity_posterior.py \
  scripts/analysis/run_autoregressive_binding_behavioral_patch.py
git diff --check
```

Expected:

```text
pytest passes
py_compile exits 0
git diff --check exits 0
```

- [ ] **Step 8: Commit**

Commit:

```bash
git add progress/diagnostics/2026-06-18_binding_mechanism_phase3_audit_adjusted_findings.md progress/diagnostics/README.md progress/index.yaml
git commit -m "docs: record audit adjusted binding findings"
```

---

### Task 10: Hypothesis Registry And Recursive Research Loop

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/research_loop.py`
- Create: `scripts/analysis/run_autoregressive_binding_research_loop.py`
- Test: `tests/analysis/test_autoregressive_binding_template_research_loop.py`

- [ ] **Step 1: Write failing hypothesis-registry tests**

Create `tests/analysis/test_autoregressive_binding_template_research_loop.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.research_loop import (
    HypothesisRecord,
    choose_next_research_actions,
    register_hypothesis,
)


def test_register_hypothesis_records_status_and_required_evidence() -> None:
    record = register_hypothesis(
        hypothesis_id="H-anchor-reuse",
        claim="Duplicate onset is driven by previous-anchor reuse.",
        status="active",
        evidence_scope="val200_scaled_probe",
        required_gates=["rep_penalty_survives", "non_circular", "held_out"],
    )

    assert isinstance(record, HypothesisRecord)
    assert record.hypothesis_id == "H-anchor-reuse"
    assert record.required_gates == ["rep_penalty_survives", "non_circular", "held_out"]


def test_choose_next_research_actions_demotes_failed_gate_and_proposes_branch() -> None:
    records = [
        register_hypothesis(
            hypothesis_id="H-anchor-reuse",
            claim="Duplicate onset is driven by previous-anchor reuse.",
            status="active",
            evidence_scope="val200_scaled_probe",
            required_gates=["rep_penalty_survives", "non_circular"],
        )
    ]
    gate_summary = {"rep_penalty_survives": False, "non_circular": True}

    actions = choose_next_research_actions(records=records, gate_summary=gate_summary)

    assert actions[0]["hypothesis_id"] == "H-anchor-reuse"
    assert actions[0]["decision"] == "demote"
    assert "decoder-artifact" in actions[0]["next_hypothesis_family"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_research_loop.py -q
```

Expected: fail with missing `research_loop`.

- [ ] **Step 3: Implement hypothesis registry helpers**

Create `src/analysis/autoregressive_binding_template_ablation/research_loop.py`.

Public API:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HypothesisRecord:
    hypothesis_id: str
    claim: str
    status: str
    evidence_scope: str
    required_gates: list[str]


def register_hypothesis(
    *,
    hypothesis_id: str,
    claim: str,
    status: str,
    evidence_scope: str,
    required_gates: Sequence[str],
) -> HypothesisRecord:
    if status not in {"active", "promoted", "demoted", "paused"}:
        raise ValueError(f"unknown status: {status}")
    return HypothesisRecord(
        hypothesis_id=hypothesis_id,
        claim=claim,
        status=status,
        evidence_scope=evidence_scope,
        required_gates=list(required_gates),
    )


def choose_next_research_actions(
    *,
    records: Sequence[HypothesisRecord],
    gate_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in records:
        failed = [gate for gate in record.required_gates if gate_summary.get(gate) is False]
        missing = [gate for gate in record.required_gates if gate not in gate_summary]
        if failed:
            family = "decoder-artifact" if "rep_penalty_survives" in failed else "coverage-binding"
            decision = "demote"
        elif missing:
            family = "evidence-completion"
            decision = "continue_observation"
        else:
            family = "causal-localization"
            decision = "promote_to_causal_probe"
        actions.append(
            {
                "schema_version": 1,
                "hypothesis_id": record.hypothesis_id,
                "decision": decision,
                "failed_gates": failed,
                "missing_gates": missing,
                "next_hypothesis_family": family,
                "record": asdict(record),
            }
        )
    return actions
```

- [ ] **Step 4: Add research-loop CLI**

Create `scripts/analysis/run_autoregressive_binding_research_loop.py`.

Required behavior:

```text
inputs:
  --hypothesis-json
  --gate-summary-json
  --output-root
writes:
  hypothesis_registry.json
  next_research_actions.json
  next_research_actions.md
```

The initial registry must include at least these hypothesis families:

```text
H-decoder-artifact: repetition penalty or decode processor shapes apparent fragility
H-anchor-reuse: local autoregressive previous-anchor reuse drives duplicates
H-coverage-ledger: model has an internal emitted-vs-remaining coverage state
H-identity-binding: same-class instance identity posterior drives object selection
H-visual-blindness-vs-guidance: false negatives are visual absence vs language-prefix recoverability
H-coordinate-basin: coordinate-token manifold basins attract object spans
H-termination-basin: stop or parse basin competes with next-object transition
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_research_loop.py -q
```

Expected: `2 passed`.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/research_loop.py scripts/analysis/run_autoregressive_binding_research_loop.py tests/analysis/test_autoregressive_binding_template_research_loop.py
git commit -m "feat: add binding research loop registry"
```

---

### Task 11: Limited Micro-Training Planner

**Files:**
- Create: `src/analysis/autoregressive_binding_template_ablation/micro_training.py`
- Create: `scripts/analysis/run_autoregressive_binding_micro_training.py`
- Test: `tests/analysis/test_autoregressive_binding_template_micro_training.py`

- [ ] **Step 1: Write failing micro-training manifest tests**

Create `tests/analysis/test_autoregressive_binding_template_micro_training.py`:

```python
from __future__ import annotations

from src.analysis.autoregressive_binding_template_ablation.micro_training import (
    build_micro_training_manifest,
    micro_training_allowed,
)


def test_micro_training_allowed_requires_bounded_steps_and_root() -> None:
    manifest = {
        "max_steps": 16,
        "output_root": "/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/micro/H-test",
        "train_rows": "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl",
        "source_checkpoint": "/data/CoordExp/outputs/example/checkpoint-928",
    }

    assert micro_training_allowed(manifest) is True


def test_build_micro_training_manifest_records_rollback_and_hypothesis() -> None:
    manifest = build_micro_training_manifest(
        hypothesis_id="H-identity-binding",
        source_checkpoint="/ckpt",
        train_rows="/train.jsonl",
        val_rows="/val.jsonl",
        output_root="/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/micro/H-identity-binding",
        trainable_modules=["token_embeddings_adapter"],
        learning_rate=1e-5,
        max_steps=8,
    )

    assert manifest["hypothesis_id"] == "H-identity-binding"
    assert manifest["max_steps"] == 8
    assert manifest["rollback_path"] == "/ckpt"
    assert manifest["within_safety_limits"] is True
    assert manifest["launch_allowed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_micro_training.py -q
```

Expected: fail with missing `micro_training`.

- [ ] **Step 3: Implement micro-training manifest helpers**

Create `src/analysis/autoregressive_binding_template_ablation/micro_training.py`.

Public API:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def micro_training_allowed(manifest: Mapping[str, Any]) -> bool:
    output_root = str(manifest.get("output_root", ""))
    return (
        int(manifest.get("max_steps", 0)) <= 32
        and int(manifest.get("max_steps", 0)) > 0
        and output_root.startswith(
            "/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/"
        )
        and bool(manifest.get("train_rows"))
        and bool(manifest.get("source_checkpoint"))
    )


def build_micro_training_manifest(
    *,
    hypothesis_id: str,
    source_checkpoint: str,
    train_rows: str,
    val_rows: str,
    output_root: str,
    trainable_modules: Sequence[str],
    learning_rate: float,
    max_steps: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "hypothesis_id": hypothesis_id,
        "source_checkpoint": source_checkpoint,
        "rollback_path": source_checkpoint,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "output_root": output_root,
        "trainable_modules": list(trainable_modules),
        "learning_rate": float(learning_rate),
        "max_steps": int(max_steps),
        "expected_artifacts": [str(Path(output_root) / "checkpoint-final"), str(Path(output_root) / "micro_training_summary.json")],
    }
    manifest["within_safety_limits"] = micro_training_allowed(manifest)
    manifest["launch_allowed"] = False
    return manifest
```

- [ ] **Step 4: Add micro-training CLI**

Create `scripts/analysis/run_autoregressive_binding_micro_training.py`.

Required behavior:

```text
inputs:
  --stage dry-run
  --stage launch
  --hypothesis-id
  --source-checkpoint
  --train-rows
  --val-rows
  --output-root
  --trainable-modules
  --learning-rate
  --max-steps
writes:
  micro_training_manifest.json
  micro_training_launch.sh
```

The `launch` stage must refuse to start when `within_safety_limits` is false, and must also refuse when `launch_allowed` is false because reviewed current training entrypoint/config fields are missing. The generated launch script must include the exact config path or command template that would run the tiny training job, but the planner-generated script itself must refuse by default.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest tests/analysis/test_autoregressive_binding_template_micro_training.py -q
```

Expected: targeted micro-training tests pass.

Commit:

```bash
git add src/analysis/autoregressive_binding_template_ablation/micro_training.py scripts/analysis/run_autoregressive_binding_micro_training.py tests/analysis/test_autoregressive_binding_template_micro_training.py
git commit -m "feat: add bounded binding micro training planner"
```

---

### Task 12: Recursive Convergence Report

**Files:**
- Create: `progress/diagnostics/2026-06-18_binding_mechanism_recursive_convergence.md`
- Modify: `progress/diagnostics/README.md`
- Modify: `progress/index.yaml`

- [ ] **Step 1: Run research-loop synthesis**

Run:

```bash
PHASE3=/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase3_audit_adjusted

python scripts/analysis/run_autoregressive_binding_research_loop.py \
  --hypothesis-json "$PHASE3/hypothesis_registry.json" \
  --gate-summary-json "$PHASE3/null_leaderboard.json" \
  --output-root "$PHASE3"
```

Expected:

```text
hypothesis_registry.json
next_research_actions.json
next_research_actions.md
```

- [ ] **Step 2: Create optional micro-training dry run for the top promoted or unresolved hypothesis**

Run only when `next_research_actions.json` selects a hypothesis family that requires a training perturbation:

```bash
python scripts/analysis/run_autoregressive_binding_micro_training.py \
  --stage dry-run \
  --hypothesis-id H-selected-from-next-actions \
  --source-checkpoint /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/converted_adapters/desc_first_ckpt928_token_embeddings_adapter/checkpoint-928 \
  --train-rows /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --val-rows /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl \
  --output-root "$PHASE3/micro_training/H-selected-from-next-actions" \
  --trainable-modules token_embeddings_adapter \
  --learning-rate 1e-5 \
  --max-steps 16
```

Expected:

```text
micro_training/H-selected-from-next-actions/micro_training_manifest.json
micro_training/H-selected-from-next-actions/micro_training_launch.sh
```

- [ ] **Step 3: Write recursive convergence note**

Create `progress/diagnostics/2026-06-18_binding_mechanism_recursive_convergence.md`.

Required sections:

```text
Scope
Expanded Mandate
Evidence Graph
Hypotheses Promoted
Hypotheses Demoted
Hypotheses Still Alive
Most Plausible Current Mechanism
Alternative Explanations Still Capable Of Explaining The Evidence
Recommended Recursive Branch
Micro-Training Dry Runs
GPU Artifacts
Convergence Status
```

The note must state one of:

```text
converged_candidate: one mechanism candidate deserves scaled causal work
not_converged_continue: evidence narrowed the space but more probes are needed
not_converged_falsified_current_family: current family failed and the next branch is named
blocked: a missing artifact or infrastructure failure prevents meaningful progress
```

- [ ] **Step 4: Link convergence note**

Add the convergence note to:

```text
progress/diagnostics/README.md
progress/index.yaml
```

- [ ] **Step 5: Run final verification**

Run:

```bash
python -m pytest \
  tests/analysis/test_autoregressive_binding_template_phase3_manifest.py \
  tests/analysis/test_autoregressive_binding_template_repetition_penalty_gate.py \
  tests/analysis/test_autoregressive_binding_template_anchor_circularity_gate.py \
  tests/analysis/test_autoregressive_binding_template_candidate_ledger.py \
  tests/analysis/test_autoregressive_binding_template_coverage_polarity.py \
  tests/analysis/test_autoregressive_binding_template_null_leaderboard.py \
  tests/analysis/test_autoregressive_binding_template_identity_posterior.py \
  tests/analysis/test_autoregressive_binding_template_behavioral_patch.py \
  tests/analysis/test_autoregressive_binding_template_research_loop.py \
  tests/analysis/test_autoregressive_binding_template_micro_training.py \
  -q
python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/research_loop.py \
  src/analysis/autoregressive_binding_template_ablation/micro_training.py \
  scripts/analysis/run_autoregressive_binding_research_loop.py \
  scripts/analysis/run_autoregressive_binding_micro_training.py
git diff --check
```

Expected:

```text
pytest passes
py_compile exits 0
git diff --check exits 0
```

- [ ] **Step 6: Commit**

Commit:

```bash
git add progress/diagnostics/2026-06-18_binding_mechanism_recursive_convergence.md progress/diagnostics/README.md progress/index.yaml
git commit -m "docs: record recursive binding convergence"
```

---

## Deferred Scaled Work

These are intentionally outside the first execution path:

```text
hidden-vector store
8-GPU hidden-state capture
8-GPU behavioral patch sweep
production-scale training beyond max_steps=32
```

Enable them only after Task 12 records a `converged_candidate` or `not_converged_continue` decision with a specific non-circular signal that survived the P0 gates. If the decision is `not_converged_falsified_current_family`, start a new branch around the named replacement hypothesis family rather than extending the falsified one.

---

## Self-Review

Spec coverage:

```text
P0-1 repetition penalty mismatch: Task 2 and Task 9.
P0-2 anchor circularity: Task 3 and Task 6.
Cross-family stage mismatch: Task 4, Task 7, and Task 9.
Sequential matcher semantics: Task 4 labels the surface as sequential-greedy coverage.
Reserve and held-out discipline: Task 1, Task 6, and Task 9.
Candidate field and coverage polarity: Task 4 and Task 5.
Same-class identity binding: Task 7.
Behavioral causal readout: Task 8, gated by Task 6.
Expanded full-control mandate: Full-Control Discovery Mandate.
Recursive hypothesis registry: Task 10.
Limited micro-training planning: Task 11.
Convergence reporting: Task 12.
Heavy infrastructure delay: Deferred Scaled Work.
```

Placeholder scan:

```text
No unresolved-detail markers.
No deferred-implementation markers.
No generic edge-case instructions.
Every task names files, tests, commands, expected outputs, and commit scope.
```

Type consistency:

```text
split_manifest.json plus gt_candidates.json feed candidate_step_rows.jsonl.
repetition_penalty_gate_rows.jsonl feeds penalty-aware coverage rows and nulls.
anchor_circularity_rows.jsonl and upstream_onset_label_rows.jsonl feed nulls.
candidate_step_rows.jsonl plus coverage_polarity_rows.jsonl feed identity posterior.
null_leaderboard.json gates behavioral patch short-rollout.
phase3_audit_adjusted_findings.md consumes every produced artifact or records the gate that stopped execution.
hypothesis_registry.json plus gate summaries feed next_research_actions.json.
next_research_actions.json may feed a bounded micro_training_manifest.json.
recursive_convergence.md consumes findings, hypothesis actions, and micro-training dry-run manifests.
```
