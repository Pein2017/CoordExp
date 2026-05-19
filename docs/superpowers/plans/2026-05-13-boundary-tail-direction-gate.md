# Boundary/Tail Direction Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a table-first offline diagnosis harness that uses current A2/A3/A4 artifacts, LVIS/proxy objectness evidence, prior duplicate/confidence thresholds, and selective manual review to recommend the next training/data/objective direction.

**Architecture:** Keep analysis behavior in a narrow `src/analysis/boundary_tail_direction_gate.py` module with typed dataclasses and pure functions. Use `scripts/analysis/diagnose_boundary_tail_direction_gate.py` only as the config-driven CLI wrapper, with YAML config under `configs/analysis/boundary_tail_direction_gate/`. Emit machine-readable tables into `/data/CoordExp/temp/`, then optionally build a small review packet and promote a concise `progress/diagnostics` report after results exist.

**Tech Stack:** Python dataclasses, JSONL/YAML, existing CoordExp eval artifacts, LVIS-proxy JSONL metadata, canonical geometry helpers from `src.datasets.geometry`, coordinate-token parsing from `src.coord_tokens.codec`, existing eval/confidence helpers where practical, pytest via `conda run -n ms python -m pytest`, optional image review rendering through existing visualization helpers.

---

Date: 2026-05-13

Spec: `docs/superpowers/specs/2026-05-13-boundary-tail-direction-gate-design.md`

Status: Ready for Execution: pending explicit execution start.

Do not implement this plan until the user explicitly starts execution. The current document is the constrained execution contract for that future implementation pass.

## Guardrails

- Do not edit `.codex/memories/`.
- Do not launch training or GPU jobs in the initial implementation.
- Do not turn this into a benchmark campaign.
- Do not introduce OpenSpec unless a later change touches stable contracts.
- Do not add config-system complexity beyond one analysis YAML.
- Do not treat duplicate guard, COCO-unmatched status, or LVIS mapping as absolute truth.
- Resolve artifact roots and runtime outputs against `/data/CoordExp`, even when executing from `/data/CoordExp/.worktrees/...`.
- Use canonical geometry and coordinate-token helpers rather than hand-rolled parsing/math unless a parity test proves an adapter is equivalent.
- Use Serena MCP for Python symbol exploration and edits after narrowing files with `rg`.
- Use `conda run -n ms python -m pytest ...` for tests.
- Keep outputs under `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/`.

## Planned File Map

| Path | Role |
|---|---|
| `configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml` | Configures run labels, artifact roots, proxy JSONL sources, thresholds, and output paths. |
| `src/analysis/boundary_tail_direction_gate.py` | Owns dataclasses, artifact loading, matching, labeling, prior-band computation, summary writing, and optional review selection. |
| `scripts/analysis/diagnose_boundary_tail_direction_gate.py` | Thin CLI wrapper that loads YAML and calls the analysis module. |
| `tests/test_boundary_tail_direction_gate.py` | Unit tests for matching, prior bands, proxy metadata alignment, duplicate labeling, boundary features, and output schemas. |
| `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/*.jsonl` | Runtime diagnosis outputs, not committed unless later promoted selectively. |
| `progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md` | Final concise diagnosis report, created only after analysis has been run and interpreted. |

## Binding Implementation Notes

- Inference `gt_vs_pred_scored.jsonl` rows may use `coord_mode: pixel`; LVIS-proxy JSONL uses coord-token/norm1000 boxes. The implementation must emit both `bbox_xyxy_pixel` and `bbox_xyxy_norm1000` where possible and must normalize coordinates before IoU, duplicate, GT, proxy, and cross-run matching.
- A pixel coordinate larger than `999` is valid when the source row is pixel-space. Norm1000 validity rules apply only to normalized coordinates.
- Coordinate conversion must use per-row `width` and `height`, and every object row must preserve `coord_mode_source`, `image_width`, and `image_height`.
- `resolved_config.json` is a required preflight input. The report must disclose that current A2 uses `max_new_tokens=1024` while A3/A4 use `max_new_tokens=3084`; A2 remains a stability anchor, not a perfectly matched decode cap.
- Confidence score geometry in `pred_confidence.jsonl` is an exponentiated mean log-probability. Derive `coord_mean_logprob = log(score_geom)` only when the object is kept and the score is positive; otherwise mark `coord_confidence_missing`.

## Task 0: Preflight And Worktree Setup

**Files:**

- Read: `docs/superpowers/specs/2026-05-13-boundary-tail-direction-gate-design.md`
- Read: `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/experiment_design_checkpoint.md`
- Read: `AGENTS.md` if present in the worktree root

- [ ] **Step 1: Inspect current git/worktree state**

Run:

```bash
git status --short
git worktree list --porcelain
```

Expected: unrelated `.codex/memories/` dirt may exist in the main checkout. Do not revert it.

- [ ] **Step 2: Create isolated worktree**

Run:

```bash
git worktree add -b codex/boundary-tail-direction-gate \
  /data/CoordExp/.worktrees/boundary-tail-direction-gate \
  main
```

Expected: new worktree at `/data/CoordExp/.worktrees/boundary-tail-direction-gate`.

- [ ] **Step 3: Confirm artifact roots exist from the implementation worktree**

Run:

```bash
cd /data/CoordExp/.worktrees/boundary-tail-direction-gate
python - <<'PY'
from pathlib import Path

roots = {
    "A2": "/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu",
    "A3": "/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a3_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu",
    "A4": "/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a4_eos_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu",
}
required = [
    "gt_vs_pred_scored.jsonl",
    "gt_vs_pred_scored_guarded.jsonl",
    "pred_confidence.jsonl",
    "pred_token_trace.jsonl",
    "eval/metrics.json",
    "eval/duplicate_guard_report.json",
    "eval/per_image.json",
    "resolved_config.json",
]
missing = []
for label, root in roots.items():
    for rel in required:
        path = Path(root) / rel
        if not path.exists():
            missing.append(f"{label}:{path}")
if missing:
    raise SystemExit("\\n".join(missing))
print("all primary artifacts present")
PY
```

Expected: prints `all primary artifacts present`.

- [ ] **Step 4: Confirm output root is the main checkout temp directory**

Run:

```bash
python - <<'PY'
from pathlib import Path

out_dir = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513")
print(out_dir)
if not str(out_dir).startswith("/data/CoordExp/temp/"):
    raise SystemExit("output_dir must stay under /data/CoordExp/temp")
PY
```

Expected: prints `/data/CoordExp/temp/boundary_tail_direction_gate_20260513`.

## Task 1: Add Config And Skeleton CLI

**Files:**

- Create: `configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml`
- Create: `scripts/analysis/diagnose_boundary_tail_direction_gate.py`
- Create: `src/analysis/boundary_tail_direction_gate.py`
- Create: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Write a failing config-load test**

Create `tests/test_boundary_tail_direction_gate.py` with:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from src.analysis.boundary_tail_direction_gate import (
    BoundaryTailConfig,
    ManualReviewConfig,
    ProxyConfig,
    ReviewPacketConfig,
    RunConfig,
    load_boundary_tail_config,
)


def test_load_boundary_tail_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "gate.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "runs": [
                    {
                        "label": "A2",
                        "artifact_root": "run_a2",
                        "role": "stability_control",
                    },
                    {
                        "label": "A3",
                        "artifact_root": "run_a3",
                        "role": "prefix_rollin",
                    },
                    {
                        "label": "A4",
                        "artifact_root": "run_a4",
                        "role": "eos_weakened",
                    },
                ],
                "proxy": {
                    "val_coord_jsonl": "proxy/val.coord.jsonl",
                    "val_proxy_summary_json": "proxy/val.proxy_summary.json",
                    "provenance_json": "manifests/proxy.json",
                },
                "output_dir": "/data/CoordExp/temp/boundary_tail_direction_gate_20260513",
                "manual_review": {"initial_case_cap": 32, "expanded_case_cap": 64},
                "review_packet": {"materialize": False},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_boundary_tail_config(cfg_path)

    assert isinstance(cfg, BoundaryTailConfig)
    assert [run.label for run in cfg.runs] == ["A2", "A3", "A4"]
    assert cfg.manual_review.initial_case_cap == 32
    assert cfg.output_dir == Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513")
    assert cfg.review_packet.materialize is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_boundary_tail_direction_gate.py::test_load_boundary_tail_config -q
```

Expected: FAIL because `src.analysis.boundary_tail_direction_gate` does not exist.

- [ ] **Step 3: Implement minimal dataclasses and config loader**

Create `src/analysis/boundary_tail_direction_gate.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Configured A2/A3/A4 artifact root."""

    label: str
    artifact_root: Path
    role: str


@dataclass(frozen=True)
class ProxyConfig:
    """Configured LVIS-proxy objectness sources."""

    val_coord_jsonl: Path
    val_proxy_summary_json: Path
    provenance_json: Path


@dataclass(frozen=True)
class ManualReviewConfig:
    """Manual-review queue limits."""

    initial_case_cap: int = 32
    expanded_case_cap: int = 64


@dataclass(frozen=True)
class ReviewPacketConfig:
    """Review-packet materialization options."""

    materialize: bool = False


@dataclass(frozen=True)
class BoundaryTailConfig:
    """Offline boundary/tail diagnosis configuration."""

    runs: tuple[RunConfig, ...]
    proxy: ProxyConfig
    output_dir: Path
    manual_review: ManualReviewConfig
    review_packet: ReviewPacketConfig


def load_boundary_tail_config(path: Path) -> BoundaryTailConfig:
    """Load the boundary/tail diagnosis config from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TypeError("boundary/tail config must be a YAML mapping")

    runs_raw = payload.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("runs must be a non-empty list")

    runs = tuple(_load_run_config(item) for item in runs_raw)
    proxy = _load_proxy_config(payload.get("proxy"))
    manual_review = _load_manual_review_config(payload.get("manual_review", {}))
    review_packet = _load_review_packet_config(payload.get("review_packet", {}))

    return BoundaryTailConfig(
        runs=runs,
        proxy=proxy,
        output_dir=Path(str(payload["output_dir"])),
        manual_review=manual_review,
        review_packet=review_packet,
    )


def _load_run_config(payload: Any) -> RunConfig:
    """Parse one run config mapping."""

    if not isinstance(payload, dict):
        raise TypeError("runs entries must be mappings")

    return RunConfig(
        label=str(payload["label"]),
        artifact_root=Path(str(payload["artifact_root"])),
        role=str(payload["role"]),
    )


def _load_proxy_config(payload: Any) -> ProxyConfig:
    """Parse the LVIS-proxy source config."""

    if not isinstance(payload, dict):
        raise TypeError("proxy must be a mapping")

    return ProxyConfig(
        val_coord_jsonl=Path(str(payload["val_coord_jsonl"])),
        val_proxy_summary_json=Path(str(payload["val_proxy_summary_json"])),
        provenance_json=Path(str(payload["provenance_json"])),
    )


def _load_manual_review_config(payload: Any) -> ManualReviewConfig:
    """Parse manual-review limits."""

    if not isinstance(payload, dict):
        raise TypeError("manual_review must be a mapping")

    return ManualReviewConfig(
        initial_case_cap=int(payload.get("initial_case_cap", 32)),
        expanded_case_cap=int(payload.get("expanded_case_cap", 64)),
    )


def _load_review_packet_config(payload: Any) -> ReviewPacketConfig:
    """Parse review-packet materialization options."""

    if not isinstance(payload, dict):
        raise TypeError("review_packet must be a mapping")

    return ReviewPacketConfig(materialize=bool(payload.get("materialize", False)))
```

- [ ] **Step 4: Create the real A2/A3/A4 config**

Create `configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml`:

```yaml
runs:
  - label: A2
    role: stability_control
    artifact_root: /data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu
  - label: A3
    role: prefix_rollin
    artifact_root: /data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a3_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
  - label: A4
    role: eos_weakened
    artifact_root: /data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_a4_eos_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
proxy:
  val_coord_jsonl: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl
  val_proxy_summary_json: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.proxy_summary.json
  provenance_json: /data/CoordExp/manifests/public_data_provenance/coco/rescale_32_1024_bbox_max60_lvis_proxy.json
output_dir: /data/CoordExp/temp/boundary_tail_direction_gate_20260513
manual_review:
  initial_case_cap: 32
  expanded_case_cap: 64
review_packet:
  materialize: false
```

- [ ] **Step 5: Create a thin CLI wrapper**

Create `scripts/analysis/diagnose_boundary_tail_direction_gate.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.boundary_tail_direction_gate import load_boundary_tail_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build boundary/tail direction-gate tables.")
    parser.add_argument("--config", required=True, help="Path to boundary/tail diagnosis YAML.")
    return parser.parse_args()


def main() -> None:
    """Load config and print the planned output directory."""

    args = parse_args()
    cfg = load_boundary_tail_config(Path(args.config))
    print(f"loaded {len(cfg.runs)} runs; output_dir={cfg.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run config and CLI smoke**

Run:

```bash
conda run -n ms python -m pytest tests/test_boundary_tail_direction_gate.py::test_load_boundary_tail_config -q
conda run -n ms python scripts/analysis/diagnose_boundary_tail_direction_gate.py \
  --config configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml
```

Expected: test passes and CLI prints `loaded 3 runs`.

## Task 2: Load Artifacts And Proxy Metadata

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add tests for artifact and proxy loading**

Append to `tests/test_boundary_tail_direction_gate.py`:

```python
import json

from src.analysis.boundary_tail_direction_gate import (
    load_jsonl_records,
    load_proxy_records,
)


def test_load_jsonl_records_preserves_rows(tmp_path: Path) -> None:
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(
        json.dumps({"image_id": 1, "pred": [{"desc": "person"}]}) + "\n"
        + json.dumps({"image_id": 2, "pred": []}) + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl_records(jsonl)

    assert [row["image_id"] for row in rows] == [1, 2]
    assert rows[0]["pred"][0]["desc"] == "person"


def test_load_proxy_records_aligns_object_supervision(tmp_path: Path) -> None:
    proxy_jsonl = tmp_path / "proxy.coord.jsonl"
    proxy_jsonl.write_text(
        json.dumps(
            {
                "image_id": 7,
                "image": "images/val2017/000000000007.jpg",
                "objects": [
                    {"desc": "person", "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"]},
                    {"desc": "vase", "bbox_2d": ["<|coord_5|>", "<|coord_6|>", "<|coord_7|>", "<|coord_8|>"]},
                ],
                "metadata": {
                    "coordexp_proxy_supervision": {
                        "object_supervision": [
                            {
                                "source": "coco",
                                "proxy_tier": "real",
                                "mapping_class": "real",
                                "desc_ce_weight": 1.0,
                                "coord_weight": 1.0,
                            },
                            {
                                "source": "lvis",
                                "proxy_tier": "strict",
                                "mapping_class": "same_extent_proxy",
                                "desc_ce_weight": 0.7,
                                "coord_weight": 1.0,
                            },
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proxy = load_proxy_records(proxy_jsonl)

    assert 7 in proxy
    assert proxy[7][0].support_tier == "real"
    assert proxy[7][1].support_tier == "strict"
    assert proxy[7][1].points == [5, 6, 7, 8]
    assert proxy[7][1].desc_ce_weight == 0.7
    assert proxy[7][1].coord_weight == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_load_jsonl_records_preserves_rows \
  tests/test_boundary_tail_direction_gate.py::test_load_proxy_records_aligns_object_supervision -q
```

Expected: FAIL because loader functions are missing.

- [ ] **Step 3: Implement JSONL and proxy loading**

Add to `src/analysis/boundary_tail_direction_gate.py`:

```python
import json
from src.coord_tokens.codec import tokens_to_ints


@dataclass(frozen=True)
class ProxyObject:
    """Proxy or COCO object from the LVIS-proxy validation JSONL."""

    image_id: int
    desc: str
    points: list[int]
    source: str
    support_tier: str
    mapping_class: str
    desc_ce_weight: float
    coord_weight: float
    mapping_kind: str | None = None
    lvis_category_name: str | None = None
    why_recovered: str | None = None


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records preserving original mapping payloads."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise TypeError(f"{path} contains a non-mapping JSONL row")
                rows.append(row)
    return rows


def load_proxy_records(path: Path) -> dict[int, list[ProxyObject]]:
    """Load per-image COCO/LVIS-proxy objects with aligned supervision metadata."""

    proxy_by_image: dict[int, list[ProxyObject]] = {}
    for row in load_jsonl_records(path):
        image_id = int(row["image_id"])
        objects = list(row.get("objects") or [])
        supervision = (
            (row.get("metadata") or {})
            .get("coordexp_proxy_supervision", {})
            .get("object_supervision", [])
        )
        if len(objects) != len(supervision):
            raise ValueError(f"proxy object/supervision length mismatch for image_id={image_id}")

        proxy_by_image[image_id] = [
            _proxy_object_from_payload(image_id, obj, sup)
            for obj, sup in zip(objects, supervision, strict=True)
        ]

    return proxy_by_image


def _proxy_object_from_payload(
    image_id: int,
    obj: dict[str, Any],
    supervision: dict[str, Any],
) -> ProxyObject:
    """Build a typed proxy object from aligned object and supervision payloads."""

    return ProxyObject(
        image_id=image_id,
        desc=str(obj.get("desc") or obj.get("category_name") or ""),
        points=_coord_tokens_to_points(obj.get("bbox_2d")),
        source=str(supervision.get("source") or ""),
        support_tier=str(supervision.get("proxy_tier") or ""),
        mapping_class=str(supervision.get("mapping_class") or ""),
        desc_ce_weight=float(supervision.get("desc_ce_weight", 0.0)),
        coord_weight=float(supervision.get("coord_weight", 0.0)),
        mapping_kind=_optional_str(supervision.get("mapping_kind")),
        lvis_category_name=_optional_str(supervision.get("lvis_category_name")),
        why_recovered=_optional_str(supervision.get("why_recovered")),
    )


def _coord_tokens_to_points(value: Any) -> list[int]:
    """Convert four coordinate tokens into integer norm1000 points."""

    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bbox_2d must contain four coord tokens")
    return list(tokens_to_ints(value, require_even=True))


def _optional_str(value: Any) -> str | None:
    """Return a string when a value is present."""

    if value is None:
        return None
    return str(value)
```

- [ ] **Step 4: Run loading tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_load_jsonl_records_preserves_rows \
  tests/test_boundary_tail_direction_gate.py::test_load_proxy_records_aligns_object_supervision -q
```

Expected: PASS.

## Task 3: Implement Geometry, Matching, And Prior Bands

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add matching and prior-band tests**

Append:

```python
from src.analysis.boundary_tail_direction_gate import (
    bbox_iou_xyxy,
    classify_prior_bands,
    cross_run_match_label,
)


def test_bbox_iou_xyxy() -> None:
    assert bbox_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert round(bbox_iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]), 4) == 0.1429


def test_cross_run_match_label_strict_loose_and_extra() -> None:
    assert cross_run_match_label("person", [0, 0, 10, 10], "person", [1, 1, 11, 11]) == "strict_same_desc"
    assert cross_run_match_label("person", [0, 0, 10, 10], "rider", [0, 0, 10, 10]) == "loose_geometry"
    assert cross_run_match_label("person", [0, 0, 10, 10], "person", [8, 8, 18, 18]) == "extra"


def test_classify_prior_bands() -> None:
    bands = classify_prior_bands(nearest_iou=0.995, coord_mean_logprob=-3.5)

    assert bands["overlap_prior_safe_hard_iou_0999"] is False
    assert bands["overlap_prior_practical_severe_iou_099"] is True
    assert bands["overlap_prior_soft_band_iou_095_099"] is False
    assert bands["coord_prior_low_conf_lt_neg3p4"] is True


def test_classify_prior_bands_keeps_missing_confidence_separate() -> None:
    bands = classify_prior_bands(nearest_iou=0.20, coord_mean_logprob=None)

    assert bands["coord_confidence_missing"] is True
    assert bands["coord_prior_low_conf_lt_neg3p4"] is False
    assert bands["coord_prior_prefixed_matched_p10_lt_neg3p364"] is False
    assert bands["coord_prior_fixed_matched_p10_lt_neg3p209"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_bbox_iou_xyxy \
  tests/test_boundary_tail_direction_gate.py::test_cross_run_match_label_strict_loose_and_extra \
  tests/test_boundary_tail_direction_gate.py::test_classify_prior_bands \
  tests/test_boundary_tail_direction_gate.py::test_classify_prior_bands_keeps_missing_confidence_separate -q
```

Expected: FAIL because functions are missing.

- [ ] **Step 3: Implement geometry and prior bands**

Add a thin adapter around canonical geometry helpers rather than hand-rolling IoU math:

```python
from src.datasets.geometry import aabb_area, intersect_aabb


def bbox_iou_xyxy(a: list[float] | list[int], b: list[float] | list[int]) -> float:
    """Compute IoU for norm1000 ``xyxy`` boxes using canonical AABB helpers."""

    a_box = [float(v) for v in a]
    b_box = [float(v) for v in b]
    inter = intersect_aabb(a_box, b_box)
    intersection = aabb_area(inter)
    union = aabb_area(a_box) + aabb_area(b_box) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def normalize_desc(desc: str) -> str:
    """Normalize a detection description for coarse matching."""

    return " ".join(desc.strip().lower().split())


def cross_run_match_label(
    desc_a: str,
    box_a: list[int],
    desc_b: str,
    box_b: list[int],
) -> str:
    """Classify one candidate cross-run object match."""

    iou = bbox_iou_xyxy(box_a, box_b)
    same_desc = normalize_desc(desc_a) == normalize_desc(desc_b)
    if same_desc and iou >= 0.50:
        return "strict_same_desc"
    if iou >= 0.70:
        return "loose_geometry"
    if (same_desc and iou >= 0.30) or iou >= 0.50:
        return "ambiguous"
    return "extra"


def classify_prior_bands(
    *,
    nearest_iou: float,
    coord_mean_logprob: float | None,
) -> dict[str, bool]:
    """Classify overlap and confidence values into approved prior bands."""

    confidence_missing = coord_mean_logprob is None
    coord = 0.0 if confidence_missing else float(coord_mean_logprob)
    return {
        "overlap_prior_safe_hard_iou_0999": nearest_iou >= 0.999,
        "overlap_prior_practical_severe_iou_099": nearest_iou >= 0.99,
        "overlap_prior_soft_band_iou_095_099": 0.95 <= nearest_iou < 0.99,
        "overlap_prior_broad_suspicious_iou_090_095": 0.90 <= nearest_iou < 0.95,
        "coord_confidence_missing": confidence_missing,
        "coord_prior_low_conf_lt_neg3p4": (not confidence_missing) and coord < -3.4,
        "coord_prior_prefixed_matched_p10_lt_neg3p364": (not confidence_missing) and coord < -3.364,
        "coord_prior_fixed_matched_p10_lt_neg3p209": (not confidence_missing) and coord < -3.209,
        "coord_prior_strict_duplicate_opt_lt_neg2p932": (not confidence_missing) and coord < -2.932,
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_bbox_iou_xyxy \
  tests/test_boundary_tail_direction_gate.py::test_cross_run_match_label_strict_loose_and_extra \
  tests/test_boundary_tail_direction_gate.py::test_classify_prior_bands \
  tests/test_boundary_tail_direction_gate.py::test_classify_prior_bands_keeps_missing_confidence_separate -q
```

Expected: PASS.

## Task 4: Build Object Rows And Labels

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add tests for object row labels**

Append:

```python
from src.analysis.boundary_tail_direction_gate import (
    build_object_row,
    classify_auto_label,
)


def test_classify_auto_label_prefers_gt_match() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.75,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
    ) == "positive_gt_match"


def test_classify_auto_label_protects_proxy_match_before_duplicate_risk() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.10,
        proxy_support_level="strict",
        duplicate_like=True,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
    ) == "positive_proxy_match"


def test_classify_auto_label_marks_invalid_geometry() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.0,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=False,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
    ) == "bad_invalid_geometry"


def test_classify_auto_label_protects_dense_duplicate_ambiguity() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.0,
        proxy_support_level="objectness",
        duplicate_like=True,
        dense_instance_ambiguous=True,
        valid_geometry=True,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
    ) == "ambiguous_dense_instance"


def test_classify_auto_label_keeps_supported_top_left_as_positive() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.90,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=True,
        malformed=False,
        wrong_location_candidate=False,
    ) == "positive_gt_match"


def test_classify_auto_label_marks_unsupported_top_left_as_bad() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.0,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=True,
        malformed=False,
        wrong_location_candidate=False,
    ) == "bad_border_or_top_left"


def test_classify_auto_label_keeps_unsupported_low_confidence_as_unknown() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.0,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
        unmatched_plausible=False,
    ) == "unknown"


def test_classify_auto_label_allows_plausible_high_confidence_unmatched() -> None:
    assert classify_auto_label(
        matched_gt_iou=0.0,
        proxy_support_level="none",
        duplicate_like=False,
        dense_instance_ambiguous=False,
        valid_geometry=True,
        border_or_top_left=False,
        malformed=False,
        wrong_location_candidate=False,
        unmatched_plausible=True,
    ) == "neutral_plausible_unmatched_candidate"


def test_build_object_row_keeps_identity_and_prior_bands() -> None:
    row = build_object_row(
        image_id=5,
        image="images/val2017/000000000005.jpg",
        run_label="A4",
        pred_index=2,
        desc="person",
        points=[0, 0, 10, 10],
        score=0.5,
        coord_mean_logprob=-3.6,
        matched_gt_iou=0.0,
        proxy_support_level="none",
        nearest_same_desc_pred_iou=0.995,
        nearest_any_desc_pred_iou=0.995,
        duplicate_guard_suppressed=True,
        source_gt_vs_pred_jsonl="/tmp/a4/gt_vs_pred_scored.jsonl",
        line_idx=11,
        case_uid="A4:11:2",
        object_index=2,
        image_width=1000,
        image_height=1000,
        proxy_source="",
        mapping_class="",
        objectness_support_level="none",
        semantic_support_level="none",
    )

    assert row["image_id"] == 5
    assert row["auto_label"] == "bad_duplicate_like"
    assert row["overlap_prior_practical_severe_iou_099"] is True
    assert row["coord_prior_low_conf_lt_neg3p4"] is True
    assert row["source_gt_vs_pred_jsonl"] == "/tmp/a4/gt_vs_pred_scored.jsonl"
    assert row["bbox_xyxy_pixel"] == [0, 0, 10, 10]
    assert row["bbox_xyxy_norm1000"] == [0, 0, 10, 10]


def test_build_object_row_marks_same_desc_burst_as_duplicate_like() -> None:
    row = build_object_row(
        image_id=6,
        image="images/val2017/000000000006.jpg",
        run_label="A4",
        pred_index=5,
        desc="person",
        points=[100, 100, 120, 140],
        score=0.4,
        coord_mean_logprob=-2.0,
        matched_gt_iou=0.0,
        proxy_support_level="none",
        nearest_same_desc_pred_iou=0.65,
        nearest_any_desc_pred_iou=0.65,
        duplicate_guard_suppressed=False,
        image_width=1000,
        image_height=1000,
        burst_cluster_size=3,
    )

    assert row["auto_label"] == "bad_duplicate_like"


def test_build_object_row_derives_pixel_and_norm1000_from_source_mode() -> None:
    row = build_object_row(
        image_id=7,
        image="images/val2017/000000000007.jpg",
        run_label="A4",
        pred_index=0,
        desc="truck",
        points=[1100, 10, 1240, 100],
        coord_mode_source="pixel",
        image_width=1248,
        image_height=832,
        score=0.7,
        coord_mean_logprob=-1.0,
        matched_gt_iou=0.0,
        proxy_support_level="none",
        nearest_same_desc_pred_iou=0.0,
        nearest_any_desc_pred_iou=0.0,
        duplicate_guard_suppressed=False,
    )

    assert row["bbox_xyxy_pixel"] == [1100, 10, 1240, 100]
    assert row["bbox_xyxy_norm1000"] == [881, 12, 993, 120]
    assert row["valid_geometry"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_prefers_gt_match \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_protects_proxy_match_before_duplicate_risk \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_marks_invalid_geometry \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_protects_dense_duplicate_ambiguity \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_keeps_supported_top_left_as_positive \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_marks_unsupported_top_left_as_bad \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_keeps_unsupported_low_confidence_as_unknown \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_allows_plausible_high_confidence_unmatched \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_keeps_identity_and_prior_bands \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_marks_same_desc_burst_as_duplicate_like \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_derives_pixel_and_norm1000_from_source_mode -q
```

Expected: FAIL because label helpers are missing.

- [ ] **Step 3: Implement object row and label helpers**

Add:

```python
def classify_auto_label(
    *,
    matched_gt_iou: float,
    proxy_support_level: str,
    duplicate_like: bool,
    dense_instance_ambiguous: bool,
    valid_geometry: bool,
    border_or_top_left: bool,
    malformed: bool,
    wrong_location_candidate: bool,
    unmatched_plausible: bool = False,
) -> str:
    """Classify an object into the approved automatic label taxonomy."""

    if malformed:
        return "bad_malformed"
    if not valid_geometry:
        return "bad_invalid_geometry"
    if matched_gt_iou >= 0.50:
        return "positive_gt_match"
    if proxy_support_level == "strict":
        return "positive_proxy_match"
    if border_or_top_left:
        return "bad_border_or_top_left"
    if proxy_support_level in {"plausible", "objectness"}:
        if duplicate_like and dense_instance_ambiguous:
            return "ambiguous_dense_instance"
        return "neutral_plausible_proxy_match"
    if duplicate_like:
        if dense_instance_ambiguous:
            return "ambiguous_dense_instance"
        return "bad_duplicate_like"
    if wrong_location_candidate:
        return "bad_wrong_location_candidate"
    if unmatched_plausible:
        return "neutral_plausible_unmatched_candidate"
    return "unknown"


def build_object_row(
    *,
    image_id: int,
    image: str,
    run_label: str,
    pred_index: int,
    desc: str,
    points: list[int],
    score: float | None,
    coord_mean_logprob: float | None,
    matched_gt_iou: float,
    proxy_support_level: str,
    nearest_same_desc_pred_iou: float,
    nearest_any_desc_pred_iou: float,
    duplicate_guard_suppressed: bool,
    source_gt_vs_pred_jsonl: str | None = None,
    line_idx: int | None = None,
    case_uid: str | None = None,
    object_index: int | None = None,
    pixel_points: list[int] | None = None,
    norm1000_points: list[int] | None = None,
    coord_mode_source: str = "norm1000",
    image_width: int | None = None,
    image_height: int | None = None,
    matched_gt_index: int | None = None,
    matched_gt_desc: str | None = None,
    nearest_gt_iou: float | None = None,
    nearest_gt_desc: str | None = None,
    coord_confidence_min: float | None = None,
    proxy_support_tier: str = "",
    proxy_support_desc: str | None = None,
    proxy_source: str = "",
    mapping_class: str = "",
    mapping_kind: str | None = None,
    desc_ce_weight: float = 0.0,
    coord_weight: float = 0.0,
    objectness_support_level: str = "none",
    semantic_support_level: str = "none",
    duplicate_cluster_id: str | None = None,
    burst_cluster_size: int = 0,
    dense_instance_ambiguous: bool = False,
    malformed: bool = False,
    wrong_location_candidate: bool = False,
) -> dict[str, Any]:
    """Build one object-delta row with prior bands and automatic labels."""

    if norm1000_points is not None and pixel_points is not None:
        norm_points = norm1000_points
        pixel_box = pixel_points
    elif coord_mode_source == "pixel":
        if image_width is None or image_height is None:
            raise ValueError("pixel coord_mode_source requires image_width and image_height")
        pixel_box = points
        norm_points = normalize_box_to_norm1000(
            points,
            coord_mode="pixel",
            width=image_width,
            height=image_height,
        )
    elif coord_mode_source in {"norm1000", "coord", "coord_token"}:
        if image_width is None or image_height is None:
            raise ValueError("norm1000 coord_mode_source requires image_width and image_height")
        norm_points = points
        pixel_box = normalize_box_to_pixel(
            points,
            coord_mode="norm1000",
            width=image_width,
            height=image_height,
        )
    else:
        raise ValueError(f"unsupported coord_mode_source: {coord_mode_source}")

    valid_geometry = _is_valid_xyxy(norm_points)
    border_or_top_left = _is_border_or_top_left(norm_points)
    duplicate_like = _is_duplicate_like(
        duplicate_guard_suppressed=duplicate_guard_suppressed,
        nearest_same_desc_pred_iou=nearest_same_desc_pred_iou,
        coord_mean_logprob=coord_mean_logprob,
        burst_cluster_size=burst_cluster_size,
    )
    prior_bands = classify_prior_bands(
        nearest_iou=max(nearest_same_desc_pred_iou, nearest_any_desc_pred_iou),
        coord_mean_logprob=coord_mean_logprob,
    )
    auto_label = classify_auto_label(
        matched_gt_iou=matched_gt_iou,
        proxy_support_level=proxy_support_level,
        duplicate_like=duplicate_like,
        dense_instance_ambiguous=dense_instance_ambiguous,
        valid_geometry=valid_geometry,
        border_or_top_left=border_or_top_left,
        malformed=malformed,
        wrong_location_candidate=wrong_location_candidate,
        unmatched_plausible=_is_unmatched_plausible(
            coord_mean_logprob=coord_mean_logprob,
            nearest_any_desc_pred_iou=nearest_any_desc_pred_iou,
            duplicate_guard_suppressed=duplicate_guard_suppressed,
            burst_cluster_size=burst_cluster_size,
        ),
    )

    return {
        "image_id": image_id,
        "image": image,
        "run_label": run_label,
        "pred_index": pred_index,
        "source_gt_vs_pred_jsonl": source_gt_vs_pred_jsonl,
        "line_idx": line_idx,
        "case_uid": case_uid,
        "object_index": object_index if object_index is not None else pred_index,
        "generation_order": pred_index,
        "desc": desc,
        "coord_mode_source": coord_mode_source,
        "bbox_xyxy_pixel": pixel_box,
        "bbox_xyxy_norm1000": norm_points,
        "bbox_area": _bbox_area(norm_points),
        "bbox_center": _bbox_center(norm_points),
        "image_width": image_width,
        "image_height": image_height,
        "valid_geometry": valid_geometry,
        "border_or_top_left_flag": border_or_top_left,
        "score": score,
        "coord_confidence_mean": coord_mean_logprob,
        "coord_confidence_min": coord_confidence_min,
        "matched_gt_iou": matched_gt_iou,
        "matched_gt_index": matched_gt_index,
        "matched_gt_desc": matched_gt_desc,
        "nearest_gt_iou": nearest_gt_iou if nearest_gt_iou is not None else matched_gt_iou,
        "nearest_gt_desc": nearest_gt_desc,
        "proxy_support_level": proxy_support_level,
        "proxy_support_tier": proxy_support_tier,
        "proxy_support_desc": proxy_support_desc,
        "proxy_source": proxy_source,
        "mapping_class": mapping_class,
        "mapping_kind": mapping_kind,
        "desc_ce_weight": desc_ce_weight,
        "coord_weight": coord_weight,
        "objectness_support_level": objectness_support_level,
        "semantic_support_level": semantic_support_level,
        "nearest_same_desc_pred_iou": nearest_same_desc_pred_iou,
        "nearest_any_desc_pred_iou": nearest_any_desc_pred_iou,
        "duplicate_guard_suppressed": duplicate_guard_suppressed,
        "duplicate_cluster_id": duplicate_cluster_id,
        "burst_cluster_size": burst_cluster_size,
        "auto_label": auto_label,
        **prior_bands,
    }


def _is_valid_xyxy(points: list[int]) -> bool:
    """Return whether a box is non-inverted and inside the norm1000 range."""

    if len(points) != 4:
        return False
    x1, y1, x2, y2 = points
    return 0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999


def _is_border_or_top_left(points: list[int]) -> bool:
    """Return whether a box looks like a top-left or border-collapse artifact."""

    x1, y1, x2, y2 = points
    return (x1 <= 3 and y1 <= 3) or x2 <= 3 or y2 <= 3


def _bbox_area(points: list[int]) -> int:
    """Compute integer area for a norm1000 box."""

    return int(aabb_area([float(v) for v in points]))


def _bbox_center(points: list[int]) -> list[float]:
    """Compute box center for a norm1000 box."""

    x1, y1, x2, y2 = points
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _is_duplicate_like(
    *,
    duplicate_guard_suppressed: bool,
    nearest_same_desc_pred_iou: float,
    coord_mean_logprob: float | None,
    burst_cluster_size: int,
) -> bool:
    """Return whether duplicate evidence is strong enough for automatic labeling."""

    low_conf = coord_mean_logprob is not None and coord_mean_logprob < -3.4
    return (
        nearest_same_desc_pred_iou >= 0.80
        or (nearest_same_desc_pred_iou >= 0.70 and low_conf)
        or (duplicate_guard_suppressed and nearest_same_desc_pred_iou >= 0.70)
        or burst_cluster_size >= 3
    )


def _is_unmatched_plausible(
    *,
    coord_mean_logprob: float | None,
    nearest_any_desc_pred_iou: float,
    duplicate_guard_suppressed: bool,
    burst_cluster_size: int,
) -> bool:
    """Return whether unsupported unmatched rows deserve neutral, not unknown, handling."""

    if coord_mean_logprob is None:
        return False
    return (
        coord_mean_logprob >= -2.5
        and nearest_any_desc_pred_iou < 0.70
        and not duplicate_guard_suppressed
        and burst_cluster_size < 3
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_prefers_gt_match \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_protects_proxy_match_before_duplicate_risk \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_marks_invalid_geometry \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_protects_dense_duplicate_ambiguity \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_keeps_supported_top_left_as_positive \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_marks_unsupported_top_left_as_bad \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_keeps_unsupported_low_confidence_as_unknown \
  tests/test_boundary_tail_direction_gate.py::test_classify_auto_label_allows_plausible_high_confidence_unmatched \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_keeps_identity_and_prior_bands \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_marks_same_desc_burst_as_duplicate_like \
  tests/test_boundary_tail_direction_gate.py::test_build_object_row_derives_pixel_and_norm1000_from_source_mode -q
```

Expected: PASS.

## Task 5: Add Real Artifact Schema Adapter

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add tests for real-shaped artifact indexes**

Append:

```python
from src.analysis.boundary_tail_direction_gate import (
    build_confidence_index,
    build_duplicate_guard_index,
    build_per_image_index,
    match_proxy_support,
    normalize_box_to_norm1000,
    normalize_box_to_pixel,
    load_proxy_source_summary,
    ProxyObject,
)


def test_build_confidence_index_derives_mean_logprob_from_score_geom() -> None:
    rows = [
        {
            "image": "images/val2017/000000000001.jpg",
            "objects": [
                {
                    "object_idx": 0,
                    "score_geom": 0.25,
                    "kept": True,
                    "confidence_details": {"failure_reason": None},
                },
                {
                    "object_idx": 1,
                    "score_geom": None,
                    "kept": False,
                    "confidence_details": {"failure_reason": "missing_span"},
                },
            ],
        }
    ]

    index = build_confidence_index(rows)

    assert round(index[("images/val2017/000000000001.jpg", 0)].coord_mean_logprob, 4) == -1.3863
    assert index[("images/val2017/000000000001.jpg", 1)].coord_mean_logprob is None
    assert index[("images/val2017/000000000001.jpg", 1)].missing is True


def test_build_duplicate_guard_index_marks_suppressed_indices() -> None:
    report = {
        "records": [
            {
                "image": "images/val2017/000000000001.jpg",
                "pred_count": 5,
                "suppressed_indices": [2, 4],
                "clusters": [{"cluster_id": 0, "indices": [1, 2, 4]}],
            }
        ]
    }

    index = build_duplicate_guard_index(report)

    assert index[("images/val2017/000000000001.jpg", 2)].suppressed is True
    assert index[("images/val2017/000000000001.jpg", 2)].cluster_id == 0
    assert index[("images/val2017/000000000001.jpg", 0)].suppressed is False


def test_build_per_image_index_reads_f1ish_050() -> None:
    rows = [
        {
            "file_name": "images/val2017/000000000001.jpg",
            "gt_count": 3,
            "pred_count": 4,
            "invalid_pred": [{"idx": 1}],
            "f1ish": {"0.50": {"tp_full": 2, "fp_full": 2, "fn_full": 1}},
        }
    ]

    index = build_per_image_index(rows)

    summary = index["images/val2017/000000000001.jpg"]
    assert summary.gt_count == 3
    assert summary.f1ish_tp50 == 2
    assert summary.invalid_count == 1


def test_load_proxy_source_summary_preserves_counts(tmp_path: Path) -> None:
    summary_path = tmp_path / "val.proxy_summary.json"
    provenance_path = tmp_path / "manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "record_count": 2,
                "accepted_proxy_counts": {"strict": 1, "plausible": 3},
                "metadata_namespace": "coordexp_proxy_supervision",
            }
        ),
        encoding="utf-8",
    )
    provenance_path.write_text(json.dumps({"relative_path": "public_data/example"}), encoding="utf-8")

    summary = load_proxy_source_summary(summary_path, provenance_path)

    assert summary["record_count"] == 2
    assert summary["accepted_proxy_counts"]["strict"] == 1
    assert summary["metadata_namespace"] == "coordexp_proxy_supervision"
    assert summary["provenance_relative_path"] == "public_data/example"


def test_coordinate_normalization_from_pixel_to_norm1000() -> None:
    norm = normalize_box_to_norm1000([873, 236, 901, 279], coord_mode="pixel", width=1248, height=832)

    assert norm == [699, 284, 722, 335]


def test_coordinate_normalization_from_norm1000_to_pixel() -> None:
    pixel = normalize_box_to_pixel([699, 284, 722, 336], coord_mode="norm1000", width=1248, height=832)

    assert pixel == [873, 236, 901, 280]


def test_match_proxy_support_preserves_strict_weights() -> None:
    proxy = ProxyObject(
        image_id=1,
        desc="person",
        points=[0, 0, 100, 100],
        source="lvis",
        support_tier="strict",
        mapping_class="same_extent_proxy",
        desc_ce_weight=0.7,
        coord_weight=1.0,
    )

    match = match_proxy_support([0, 0, 100, 100], desc="person", proxies=[proxy])

    assert match.support_level == "strict"
    assert match.objectness_support_level == "strong"
    assert match.semantic_support_level == "strong"
    assert match.desc_ce_weight == 0.7
    assert match.coord_weight == 1.0


def test_match_proxy_support_keeps_nearby_related_object_as_objectness() -> None:
    proxy = ProxyObject(
        image_id=1,
        desc="bicycle",
        points=[0, 0, 100, 100],
        source="lvis",
        support_tier="plausible",
        mapping_class="related_object",
        desc_ce_weight=0.0,
        coord_weight=0.5,
    )

    match = match_proxy_support([0, 0, 100, 100], desc="person", proxies=[proxy])

    assert match.support_level == "objectness"
    assert match.objectness_support_level == "supported"
    assert match.semantic_support_level == "uncertain"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_build_confidence_index_derives_mean_logprob_from_score_geom \
  tests/test_boundary_tail_direction_gate.py::test_build_duplicate_guard_index_marks_suppressed_indices \
  tests/test_boundary_tail_direction_gate.py::test_build_per_image_index_reads_f1ish_050 \
  tests/test_boundary_tail_direction_gate.py::test_load_proxy_source_summary_preserves_counts \
  tests/test_boundary_tail_direction_gate.py::test_coordinate_normalization_from_pixel_to_norm1000 \
  tests/test_boundary_tail_direction_gate.py::test_coordinate_normalization_from_norm1000_to_pixel \
  tests/test_boundary_tail_direction_gate.py::test_match_proxy_support_preserves_strict_weights \
  tests/test_boundary_tail_direction_gate.py::test_match_proxy_support_keeps_nearby_related_object_as_objectness -q
```

Expected: FAIL because schema adapter helpers are missing.

- [ ] **Step 3: Implement typed artifact adapter dataclasses**

Add:

```python
import math


@dataclass(frozen=True)
class ConfidenceEntry:
    """Coordinate-confidence evidence for one predicted object."""

    coord_mean_logprob: float | None
    score_geom: float | None
    missing: bool
    failure_reason: str | None


@dataclass(frozen=True)
class DuplicateGuardEntry:
    """Duplicate-guard evidence for one predicted object."""

    suppressed: bool
    cluster_id: int | None


@dataclass(frozen=True)
class PerImageEvalEntry:
    """Per-image evaluation counters from evaluator artifacts."""

    gt_count: int
    pred_count: int
    invalid_count: int
    f1ish_tp50: int
    f1ish_fp50: int
    f1ish_fn50: int


@dataclass(frozen=True)
class CoordinateBox:
    """Coordinate box represented on both pixel and norm1000 surfaces."""

    source_points: list[int]
    source_mode: str
    pixel_points: list[int]
    norm1000_points: list[int]
    width: int
    height: int


@dataclass(frozen=True)
class ProxySupportMatch:
    """Prediction-to-proxy objectness and semantic support evidence."""

    support_level: str
    support_tier: str
    support_desc: str | None
    source: str
    mapping_class: str
    mapping_kind: str | None
    desc_ce_weight: float
    coord_weight: float
    objectness_support_level: str
    semantic_support_level: str
    nearest_proxy_iou: float
```

- [ ] **Step 4: Implement schema adapter helpers**

Add:

```python
def build_confidence_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], ConfidenceEntry]:
    """Index confidence post-op rows by image path and object index."""

    index: dict[tuple[str, int], ConfidenceEntry] = {}
    for row in rows:
        image = str(row.get("image") or row.get("file_name") or "")
        for obj in row.get("objects") or []:
            object_idx = int(obj.get("object_idx"))
            score_geom = _positive_float_or_none(obj.get("score_geom") or obj.get("score"))
            failure_reason = _confidence_failure_reason(obj)
            missing = score_geom is None or bool(failure_reason) or obj.get("kept") is False
            coord_mean_logprob = None if missing else math.log(score_geom)
            index[(image, object_idx)] = ConfidenceEntry(
                coord_mean_logprob=coord_mean_logprob,
                score_geom=score_geom,
                missing=missing,
                failure_reason=failure_reason,
            )
    return index


def normalize_box_to_norm1000(
    points: list[int],
    *,
    coord_mode: str,
    width: int,
    height: int,
) -> list[int]:
    """Normalize a pixel or norm1000 ``xyxy`` box onto the norm1000 surface."""

    if coord_mode == "pixel":
        x1, y1, x2, y2 = [float(v) for v in points]
        return [
            round(x1 / float(width) * 999.0),
            round(y1 / float(height) * 999.0),
            round(x2 / float(width) * 999.0),
            round(y2 / float(height) * 999.0),
        ]
    if coord_mode in {"norm1000", "coord", "coord_token"}:
        return [int(v) for v in points]
    raise ValueError(f"unsupported coord_mode: {coord_mode}")


def normalize_box_to_pixel(
    points: list[int],
    *,
    coord_mode: str,
    width: int,
    height: int,
) -> list[int]:
    """Normalize a pixel or norm1000 ``xyxy`` box onto the pixel surface."""

    if coord_mode == "pixel":
        return [int(v) for v in points]
    if coord_mode in {"norm1000", "coord", "coord_token"}:
        x1, y1, x2, y2 = [float(v) for v in points]
        return [
            round(x1 / 999.0 * float(width)),
            round(y1 / 999.0 * float(height)),
            round(x2 / 999.0 * float(width)),
            round(y2 / 999.0 * float(height)),
        ]
    raise ValueError(f"unsupported coord_mode: {coord_mode}")


def build_duplicate_guard_index(report: dict[str, Any]) -> dict[tuple[str, int], DuplicateGuardEntry]:
    """Index duplicate-guard suppression and cluster evidence."""

    index: dict[tuple[str, int], DuplicateGuardEntry] = {}
    for record in report.get("records") or []:
        image = str(record.get("image") or record.get("file_name") or "")
        suppressed_indices = {int(idx) for idx in record.get("suppressed_indices") or []}
        cluster_by_index: dict[int, int] = {}
        for cluster in record.get("clusters") or []:
            cluster_id = int(cluster.get("cluster_id", len(cluster_by_index)))
            for idx in cluster.get("indices") or []:
                cluster_by_index[int(idx)] = cluster_id
        if record.get("pred_count") is not None:
            all_indices = set(range(int(record["pred_count"])))
        else:
            all_indices = set(suppressed_indices) | set(cluster_by_index)
        for idx in all_indices:
            index[(image, idx)] = DuplicateGuardEntry(
                suppressed=idx in suppressed_indices,
                cluster_id=cluster_by_index.get(idx),
            )
    return index


def build_per_image_index(rows: list[dict[str, Any]]) -> dict[str, PerImageEvalEntry]:
    """Index evaluator per-image summaries by image file name."""

    index: dict[str, PerImageEvalEntry] = {}
    for row in rows:
        image = str(row.get("file_name") or row.get("image") or "")
        f1ish_050 = (row.get("f1ish") or {}).get("0.50") or {}
        index[image] = PerImageEvalEntry(
            gt_count=int(row.get("gt_count") or 0),
            pred_count=int(row.get("pred_count") or 0),
            invalid_count=len(row.get("invalid_pred") or []),
            f1ish_tp50=int(f1ish_050.get("tp_full") or 0),
            f1ish_fp50=int(f1ish_050.get("fp_full") or 0),
            f1ish_fn50=int(f1ish_050.get("fn_full") or 0),
        )
    return index


def load_proxy_source_summary(summary_path: Path, provenance_path: Path) -> dict[str, Any]:
    """Load lightweight proxy source and provenance counters for reporting."""

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    return {
        "record_count": summary.get("record_count"),
        "record_count_with_added_proxies": summary.get("record_count_with_added_proxies"),
        "accepted_proxy_counts": summary.get("accepted_proxy_counts", {}),
        "skipped_proxy_counts": summary.get("skipped_proxy_counts", {}),
        "metadata_namespace": summary.get("metadata_namespace"),
        "include_plausible": summary.get("include_plausible"),
        "provenance_relative_path": provenance.get("relative_path"),
        "provenance_manifest_path": str(provenance_path),
    }


def match_proxy_support(
    pred_box_norm1000: list[int],
    *,
    desc: str,
    proxies: list[ProxyObject],
) -> ProxySupportMatch:
    """Match one prediction to COCO/LVIS-proxy objectness evidence."""

    best_proxy: ProxyObject | None = None
    best_iou = 0.0
    for proxy in proxies:
        iou = bbox_iou_xyxy(pred_box_norm1000, proxy.points)
        if iou > best_iou:
            best_iou = iou
            best_proxy = proxy

    if best_proxy is None or best_iou < 0.50:
        return ProxySupportMatch(
            support_level="none",
            support_tier="",
            support_desc=None,
            source="",
            mapping_class="",
            mapping_kind=None,
            desc_ce_weight=0.0,
            coord_weight=0.0,
            objectness_support_level="none",
            semantic_support_level="none",
            nearest_proxy_iou=best_iou,
        )

    same_desc = normalize_desc(desc) == normalize_desc(best_proxy.desc)
    strong_semantic = same_desc and best_proxy.support_tier in {"real", "strict"}
    if strong_semantic:
        support_level = "strict"
        objectness_level = "strong"
        semantic_level = "strong"
    elif best_proxy.support_tier in {"strict", "plausible", "real"}:
        support_level = "plausible" if same_desc else "objectness"
        objectness_level = "supported"
        semantic_level = "soft" if same_desc else "uncertain"
    else:
        support_level = "objectness"
        objectness_level = "supported"
        semantic_level = "uncertain"

    return ProxySupportMatch(
        support_level=support_level,
        support_tier=best_proxy.support_tier,
        support_desc=best_proxy.desc,
        source=best_proxy.source,
        mapping_class=best_proxy.mapping_class,
        mapping_kind=best_proxy.mapping_kind,
        desc_ce_weight=best_proxy.desc_ce_weight,
        coord_weight=best_proxy.coord_weight,
        objectness_support_level=objectness_level,
        semantic_support_level=semantic_level,
        nearest_proxy_iou=best_iou,
    )


def _positive_float_or_none(value: Any) -> float | None:
    """Return a positive finite float when available."""

    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _confidence_failure_reason(obj: dict[str, Any]) -> str | None:
    """Extract confidence failure reason from confidence-postop object payloads."""

    details = obj.get("confidence_details") or {}
    reason = details.get("failure_reason")
    if reason is None:
        return None
    return str(reason)
```

- [ ] **Step 5: Run schema adapter tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_build_confidence_index_derives_mean_logprob_from_score_geom \
  tests/test_boundary_tail_direction_gate.py::test_build_duplicate_guard_index_marks_suppressed_indices \
  tests/test_boundary_tail_direction_gate.py::test_build_per_image_index_reads_f1ish_050 \
  tests/test_boundary_tail_direction_gate.py::test_load_proxy_source_summary_preserves_counts \
  tests/test_boundary_tail_direction_gate.py::test_coordinate_normalization_from_pixel_to_norm1000 \
  tests/test_boundary_tail_direction_gate.py::test_coordinate_normalization_from_norm1000_to_pixel \
  tests/test_boundary_tail_direction_gate.py::test_match_proxy_support_preserves_strict_weights \
  tests/test_boundary_tail_direction_gate.py::test_match_proxy_support_keeps_nearby_related_object_as_objectness -q
```

Expected: PASS.

## Task 6: Build End-To-End Table Writer

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `scripts/analysis/diagnose_boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add a synthetic end-to-end output test**

Append:

```python
from src.analysis.boundary_tail_direction_gate import run_boundary_tail_diagnosis


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\\n".join(json.dumps(row) for row in rows) + "\\n", encoding="utf-8")


def test_run_boundary_tail_diagnosis_writes_core_tables(tmp_path: Path) -> None:
    run_root = tmp_path / "run_a2"
    _write_jsonl(
        run_root / "gt_vs_pred_scored.jsonl",
        [
            {
                "image_id": 1,
                "image": "images/val2017/000000000001.jpg",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [{"desc": "person", "type": "bbox_2d", "points": [0, 0, 10, 10]}],
                "pred": [{"desc": "person", "type": "bbox_2d", "points": [0, 0, 10, 10], "score": 0.9}],
            }
        ],
    )
    _write_jsonl(
        run_root / "gt_vs_pred_scored_guarded.jsonl",
        [
            {
                "image_id": 1,
                "image": "images/val2017/000000000001.jpg",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [{"desc": "person", "type": "bbox_2d", "points": [0, 0, 10, 10]}],
                "pred": [{"desc": "person", "type": "bbox_2d", "points": [0, 0, 10, 10], "score": 0.9}],
            }
        ],
    )
    _write_jsonl(
        run_root / "pred_confidence.jsonl",
        [{"image": "images/val2017/000000000001.jpg", "objects": [{"object_idx": 0, "score": 0.9}]}],
    )
    (run_root / "eval").mkdir(parents=True, exist_ok=True)
    (run_root / "eval" / "duplicate_guard_report.json").write_text(
        json.dumps({"records": [{"image": "images/val2017/000000000001.jpg", "suppressed_indices": []}]}),
        encoding="utf-8",
    )
    (run_root / "eval" / "metrics.json").write_text("{}", encoding="utf-8")
    (run_root / "eval" / "per_image.json").write_text(
        json.dumps(
            [
                {
                    "file_name": "images/val2017/000000000001.jpg",
                    "gt_count": 1,
                    "pred_count": 1,
                    "invalid_pred": [],
                    "f1ish": {"0.50": {"tp_full": 1, "fp_full": 0, "fn_full": 0}},
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_root / "resolved_config.json").write_text(
        json.dumps({"cfg": {"infer": {"generation": {"max_new_tokens": 1024}}}}),
        encoding="utf-8",
    )

    proxy_path = tmp_path / "proxy.coord.jsonl"
    _write_jsonl(
        proxy_path,
        [
            {
                "image_id": 1,
                "image": "images/val2017/000000000001.jpg",
                "objects": [{"desc": "person", "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_10|>", "<|coord_10|>"]}],
                "metadata": {
                    "coordexp_proxy_supervision": {
                        "object_supervision": [{"source": "coco", "proxy_tier": "real", "mapping_class": "real"}]
                    }
                },
            }
        ],
    )
    cfg = BoundaryTailConfig(
        runs=(RunConfig(label="A2", artifact_root=run_root, role="stability_control"),),
        proxy=ProxyConfig(
            val_coord_jsonl=proxy_path,
            val_proxy_summary_json=tmp_path / "proxy_summary.json",
            provenance_json=tmp_path / "manifest.json",
        ),
        output_dir=tmp_path / "out",
        manual_review=ManualReviewConfig(),
        review_packet=ReviewPacketConfig(),
    )

    run_boundary_tail_diagnosis(cfg)

    assert (tmp_path / "out" / "image_run_summary.jsonl").exists()
    assert (tmp_path / "out" / "object_delta_table.jsonl").exists()
    assert (tmp_path / "out" / "manual_audit_queue.jsonl").exists()
    object_row = json.loads((tmp_path / "out" / "object_delta_table.jsonl").read_text(encoding="utf-8").splitlines()[0])
    image_row = json.loads((tmp_path / "out" / "image_run_summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert object_row["bbox_xyxy_pixel"] == [0, 0, 10, 10]
    assert object_row["bbox_xyxy_norm1000"] == [0, 0, 100, 100]
    assert object_row["coord_confidence_mean"] is not None
    assert object_row["duplicate_guard_suppressed"] is False
    assert image_row["f1ish_tp50"] == 1
    assert image_row["pred_count_guarded"] == 1
```

- [ ] **Step 2: Run the synthetic end-to-end test to verify it fails**

Run:

```bash
conda run -n ms python -m pytest tests/test_boundary_tail_direction_gate.py::test_run_boundary_tail_diagnosis_writes_core_tables -q
```

Expected: FAIL because `run_boundary_tail_diagnosis` is missing.

- [ ] **Step 3: Implement minimal table writing**

Implement `run_boundary_tail_diagnosis(cfg)` so it:

- creates `cfg.output_dir`;
- loads proxy records once;
- loads `proxy_source_summary = load_proxy_source_summary(cfg.proxy.val_proxy_summary_json, cfg.proxy.provenance_json)`;
- initializes a `resolved_config_summary` dictionary keyed by run label;
- loops through run roots;
- reads `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored_guarded.jsonl`, `pred_confidence.jsonl`, `eval/duplicate_guard_report.json`, `eval/per_image.json`, and `resolved_config.json`;
- builds at least one object row per prediction;
- builds one image summary per image/run;
- joins confidence, duplicate guard, guarded-count, per-image f1ish, proxy/objectness via `match_proxy_support`, coordinate normalization, and resolved-config preflight evidence;
- writes `image_run_summary.jsonl`, `object_delta_table.jsonl`, and `manual_audit_queue.jsonl`;
- writes empty manual queue when no rows require manual review.

Use a helper:

```python
def write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write mapping rows as UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")
```

- [ ] **Step 4: Wire the CLI to run the diagnosis**

Modify `scripts/analysis/diagnose_boundary_tail_direction_gate.py`:

```python
from src.analysis.boundary_tail_direction_gate import (
    load_boundary_tail_config,
    run_boundary_tail_diagnosis,
)
```

and in `main()`:

```python
cfg = load_boundary_tail_config(Path(args.config))
run_boundary_tail_diagnosis(cfg)
print(f"wrote boundary/tail tables to {cfg.output_dir}")
```

- [ ] **Step 5: Run synthetic end-to-end test**

Run:

```bash
conda run -n ms python -m pytest tests/test_boundary_tail_direction_gate.py::test_run_boundary_tail_diagnosis_writes_core_tables -q
```

Expected: PASS.

## Task 7: Add Real Cross-Run Matching And Boundary Features

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add tests for extras and boundary features**

Append:

```python
from src.analysis.boundary_tail_direction_gate import (
    annotate_boundary_features,
    annotate_cross_run_extras,
    select_cross_run_match_boxes,
)


def test_select_cross_run_match_boxes_prefers_pixel_when_dimensions_match() -> None:
    a2 = {
        "bbox_xyxy_pixel": [1100, 10, 1240, 100],
        "bbox_xyxy_norm1000": [881, 12, 993, 120],
        "image_width": 1248,
        "image_height": 832,
    }
    a4 = {
        "bbox_xyxy_pixel": [1102, 10, 1242, 100],
        "bbox_xyxy_norm1000": [882, 12, 994, 120],
        "image_width": 1248,
        "image_height": 832,
    }

    box_a, box_b, surface = select_cross_run_match_boxes(a2, a4)

    assert surface == "pixel"
    assert box_a == [1100, 10, 1240, 100]
    assert box_b == [1102, 10, 1242, 100]


def test_annotate_cross_run_extras_marks_a4_only_object() -> None:
    rows = [
        {"image_id": 1, "run_label": "A2", "pred_index": 0, "desc": "person", "bbox_xyxy_norm1000": [0, 0, 10, 10]},
        {"image_id": 1, "run_label": "A4", "pred_index": 0, "desc": "person", "bbox_xyxy_norm1000": [0, 0, 10, 10]},
        {"image_id": 1, "run_label": "A4", "pred_index": 1, "desc": "cup", "bbox_xyxy_norm1000": [50, 50, 70, 70]},
    ]

    annotated = annotate_cross_run_extras(rows)
    a4_cup = [row for row in annotated if row["run_label"] == "A4" and row["desc"] == "cup"][0]

    assert a4_cup["is_extra_vs_A2"] is True
    assert a4_cup["cross_run_nearest_A2_iou"] == 0.0


def test_annotate_cross_run_extras_uses_one_to_one_matching() -> None:
    rows = [
        {"image_id": 1, "run_label": "A2", "pred_index": 0, "desc": "person", "bbox_xyxy_norm1000": [0, 0, 10, 10]},
        {"image_id": 1, "run_label": "A4", "pred_index": 0, "desc": "person", "bbox_xyxy_norm1000": [0, 0, 10, 10]},
        {"image_id": 1, "run_label": "A4", "pred_index": 1, "desc": "person", "bbox_xyxy_norm1000": [0, 0, 10, 10]},
    ]

    annotated = annotate_cross_run_extras(rows)
    a4_rows = [row for row in annotated if row["run_label"] == "A4"]

    assert sum(1 for row in a4_rows if row["is_extra_vs_A2"] is False) == 1
    assert sum(1 for row in a4_rows if row["is_extra_vs_A2"] is True) == 1


def test_annotate_boundary_features_marks_after_coco_boundary() -> None:
    rows = [
        {"image_id": 1, "run_label": "A4", "pred_index": 0, "auto_label": "positive_gt_match"},
        {"image_id": 1, "run_label": "A4", "pred_index": 1, "auto_label": "neutral_plausible_unmatched_candidate"},
    ]

    annotated = annotate_boundary_features(rows)

    assert annotated[0]["after_coco_boundary"] is False
    assert annotated[1]["after_coco_boundary"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_select_cross_run_match_boxes_prefers_pixel_when_dimensions_match \
  tests/test_boundary_tail_direction_gate.py::test_annotate_cross_run_extras_marks_a4_only_object \
  tests/test_boundary_tail_direction_gate.py::test_annotate_cross_run_extras_uses_one_to_one_matching \
  tests/test_boundary_tail_direction_gate.py::test_annotate_boundary_features_marks_after_coco_boundary -q
```

Expected: FAIL because annotation helpers are missing.

- [ ] **Step 3: Implement cross-run extras and boundary features**

Implement:

```python
def select_cross_run_match_boxes(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
) -> tuple[list[int], list[int], str]:
    """Select the coordinate surface for cross-run matching."""

    same_size = (
        row_a.get("image_width") is not None
        and row_a.get("image_height") is not None
        and row_a.get("image_width") == row_b.get("image_width")
        and row_a.get("image_height") == row_b.get("image_height")
    )
    if same_size and row_a.get("bbox_xyxy_pixel") and row_b.get("bbox_xyxy_pixel"):
        return list(row_a["bbox_xyxy_pixel"]), list(row_b["bbox_xyxy_pixel"]), "pixel"
    return list(row_a["bbox_xyxy_norm1000"]), list(row_b["bbox_xyxy_norm1000"]), "norm1000"


def annotate_cross_run_extras(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate object rows with nearest cross-run matches and extra flags."""

    # For each image and ordered run pair, greedily assign one-to-one matches.
    # Sort candidate pairs by match tier then IoU:
    # 1. strict same-desc IoU >= 0.50;
    # 2. loose any-desc geometry IoU >= 0.70.
    # Candidate IoU must use `select_cross_run_match_boxes`, which prefers
    # pixel-space when both rows share image dimensions and falls back to norm1000.
    # A source object can be consumed once per run pair. Ambiguous matches are
    # recorded in nearest/ambiguous fields but do not make an object non-extra.
    # This prevents multiple A4 duplicate rows from all matching one A2 object.
    # Record `cross_run_match_surface_A2`/`cross_run_match_surface_A3` fields.
```

and:

```python
def annotate_boundary_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate COCO, coverage-aware, and risk-boundary tail features."""

    # per image/run, find last positive_gt_match, last objectness-supported row,
    # and first risk row, then mark after_* booleans for every object.
```

Implementation requirements:

- use copies of row dictionaries rather than mutating caller-owned rows;
- fill missing cross-run fields for all rows;
- support missing A2/A3/A4 labels in synthetic tests;
- treat `positive_gt_match` as COCO coverage;
- treat `positive_gt_match`, `positive_proxy_match`, and `neutral_plausible_proxy_match` as coverage-aware support;
- treat `bad_duplicate_like`, `bad_invalid_geometry`, and `bad_border_or_top_left` as risk starts.

- [ ] **Step 4: Run tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_select_cross_run_match_boxes_prefers_pixel_when_dimensions_match \
  tests/test_boundary_tail_direction_gate.py::test_annotate_cross_run_extras_marks_a4_only_object \
  tests/test_boundary_tail_direction_gate.py::test_annotate_cross_run_extras_uses_one_to_one_matching \
  tests/test_boundary_tail_direction_gate.py::test_annotate_boundary_features_marks_after_coco_boundary -q
```

Expected: PASS.

## Task 8: Threshold Transfer Summary And Manual Queue Ranking

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `tests/test_boundary_tail_direction_gate.py`

- [ ] **Step 1: Add tests for summary and manual queue**

Append:

```python
from src.analysis.boundary_tail_direction_gate import build_manual_audit_queue, build_threshold_prior_transfer_summary


def test_build_manual_audit_queue_selects_ambiguous_decision_cases() -> None:
    rows = [
        {
            "image_id": 1,
            "image": "images/val2017/000000000001.jpg",
            "run_label": "A4",
            "pred_index": 2,
            "source_gt_vs_pred_jsonl": "/tmp/a4/gt_vs_pred_scored.jsonl",
            "line_idx": 4,
            "case_uid": "A4:4:2",
            "object_index": 2,
            "desc": "person",
            "bbox_xyxy_norm1000": [10, 10, 20, 20],
            "auto_label": "ambiguous_dense_instance",
            "is_extra_vs_A2": True,
            "coord_confidence_mean": -2.0,
        },
        {
            "image_id": 1,
            "run_label": "A4",
            "pred_index": 3,
            "desc": "person",
            "bbox_xyxy_norm1000": [10, 10, 20, 20],
            "auto_label": "bad_invalid_geometry",
            "is_extra_vs_A2": True,
            "coord_confidence_mean": -5.0,
        },
    ]

    queue = build_manual_audit_queue(rows, case_cap=32)

    assert len(queue) == 1
    assert queue[0]["auto_label"] == "ambiguous_dense_instance"
    assert queue[0]["source_gt_vs_pred_jsonl"] == "/tmp/a4/gt_vs_pred_scored.jsonl"
    assert queue[0]["line_idx"] == 4
    assert queue[0]["case_uid"] == "A4:4:2"
    assert queue[0]["object_index"] == 2
    assert queue[0]["recommended_manual_options"] == [
        "real_distinct_object",
        "duplicate_same_instance",
        "duplicate_but_distinct_dense_instance_possible",
        "wrong_location",
        "wrong_category_or_desc",
        "invalid_or_artifact",
        "cannot_tell",
    ]


def test_build_threshold_prior_transfer_summary_counts_by_run() -> None:
    rows = [
        {"run_label": "A2", "auto_label": "positive_gt_match", "coord_prior_low_conf_lt_neg3p4": False},
        {"run_label": "A4", "auto_label": "bad_duplicate_like", "coord_prior_low_conf_lt_neg3p4": True},
    ]

    summary = build_threshold_prior_transfer_summary(rows)

    assert summary["by_run"]["A2"]["positive_gt_match"] == 1
    assert summary["by_run"]["A4"]["bad_duplicate_like"] == 1


def test_build_threshold_prior_transfer_summary_includes_sources() -> None:
    summary = build_threshold_prior_transfer_summary(
        [],
        proxy_source_summary={"record_count": 2, "accepted_proxy_counts": {"strict": 1}},
        resolved_config_summary={"A2": {"max_new_tokens": 1024}},
    )

    assert summary["proxy_source"]["record_count"] == 2
    assert summary["resolved_config"]["A2"]["max_new_tokens"] == 1024


def test_build_threshold_prior_transfer_summary_reports_band_tradeoff() -> None:
    rows = [
        {
            "run_label": "A4",
            "auto_label": "bad_duplicate_like",
            "coord_prior_low_conf_lt_neg3p4": True,
            "coord_confidence_mean": -4.0,
        },
        {
            "run_label": "A4",
            "auto_label": "positive_proxy_match",
            "coord_prior_low_conf_lt_neg3p4": True,
            "coord_confidence_mean": -3.8,
            "objectness_support_level": "strong",
        },
        {
            "run_label": "A4",
            "auto_label": "neutral_plausible_proxy_match",
            "coord_prior_low_conf_lt_neg3p4": False,
            "coord_confidence_mean": -1.0,
            "objectness_support_level": "supported",
        },
    ]

    summary = build_threshold_prior_transfer_summary(rows)
    band = summary["prior_bands"]["coord_prior_low_conf_lt_neg3p4"]

    assert band["flagged_danger_tail"] == 1
    assert band["flagged_valid_or_objectness_supported"] == 1
    assert band["valid_or_objectness_supported_total"] == 2
    assert summary["confidence_separability"]["danger_tail_count"] == 1
    assert summary["confidence_separability"]["valid_or_objectness_supported_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_build_manual_audit_queue_selects_ambiguous_decision_cases \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_counts_by_run \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_includes_sources \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_reports_band_tradeoff -q
```

Expected: FAIL because helpers are missing.

- [ ] **Step 3: Implement manual queue and threshold summary**

Implement:

```python
MANUAL_OPTIONS = [
    "real_distinct_object",
    "duplicate_same_instance",
    "duplicate_but_distinct_dense_instance_possible",
    "wrong_location",
    "wrong_category_or_desc",
    "invalid_or_artifact",
    "cannot_tell",
]


def build_manual_audit_queue(rows: list[dict[str, Any]], *, case_cap: int) -> list[dict[str, Any]]:
    """Select decision-changing ambiguous cases for human audit."""

    candidates = [
        row for row in rows
        if row.get("auto_label") in {"ambiguous_dense_instance", "unknown"}
        or (row.get("auto_label") == "neutral_plausible_unmatched_candidate" and row.get("is_extra_vs_A2"))
    ]
    ranked = sorted(candidates, key=_manual_priority_key)
    return [_manual_case_from_row(row) for row in ranked[:case_cap]]


def _manual_priority_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    """Rank rows by decision impact, extra-object status, and confidence."""

    label = str(row.get("auto_label") or "")
    label_rank = {
        "ambiguous_dense_instance": 0,
        "neutral_plausible_unmatched_candidate": 1,
        "unknown": 2,
    }.get(label, 3)
    extra_rank = 0 if row.get("is_extra_vs_A2") else 1
    confidence = row.get("coord_confidence_mean")
    confidence_rank = -float(confidence) if confidence is not None else 999.0
    return (label_rank, extra_rank, confidence_rank, str(row.get("case_uid") or ""))


def _manual_case_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Create one manual audit case preserving source artifact coordinates."""

    case_uid = str(row.get("case_uid") or f"{row.get('run_label')}:{row.get('image_id')}:{row.get('pred_index')}")
    return {
        "case_id": case_uid,
        "audit_id": case_uid,
        "source_gt_vs_pred_jsonl": row.get("source_gt_vs_pred_jsonl"),
        "line_idx": row.get("line_idx"),
        "case_uid": case_uid,
        "object_index": row.get("object_index", row.get("pred_index")),
        "image_id": row.get("image_id"),
        "image": row.get("image"),
        "run_label": row.get("run_label"),
        "pred_index": row.get("pred_index"),
        "desc": row.get("desc"),
        "bbox_xyxy_norm1000": row.get("bbox_xyxy_norm1000"),
        "auto_label": row.get("auto_label"),
        "why_auto_label_is_uncertain": _manual_uncertainty_reason(row),
        "decision_impact": _manual_decision_impact(row),
        "overlay_path_with_gt": None,
        "overlay_path_no_gt": None,
        "crop_path": None,
        "neighbor_context_path_if_available": None,
        "question_for_user": "Is this prediction a real distinct object, a duplicate/burst, or an invalid/wrong object?",
        "recommended_manual_options": MANUAL_OPTIONS,
        "user_label": None,
        "user_notes": "",
    }


def _manual_uncertainty_reason(row: dict[str, Any]) -> str:
    """Describe why automatic labeling is not enough."""

    label = str(row.get("auto_label") or "")
    if label == "ambiguous_dense_instance":
        return "duplicate-like geometry occurs in a dense object region where distinct instances are plausible"
    if label == "neutral_plausible_unmatched_candidate":
        return "prediction is extra versus A2 and may be a valid unlabeled object"
    return "automatic rules cannot assign a stable positive, neutral, or bad label"


def _manual_decision_impact(row: dict[str, Any]) -> str:
    """Describe the training-direction decision affected by this case."""

    if row.get("is_extra_vs_A2"):
        return "decides whether A4/A3 extra emissions are valid-object recall or dirty tail"
    return "decides whether duplicate suppression would harm dense-instance recall"
```

The manual queue is the analysis-native artifact. If visual review is needed, convert queued rows into a reviewer-compatible CSV or gallery shortlist; do not require in-place edits to `manual_audit_queue.jsonl`.

and:

```python
PRIOR_BAND_KEYS = [
    "overlap_prior_safe_hard_iou_0999",
    "overlap_prior_practical_severe_iou_099",
    "overlap_prior_soft_band_iou_095_099",
    "overlap_prior_broad_suspicious_iou_090_095",
    "coord_prior_low_conf_lt_neg3p4",
    "coord_prior_prefixed_matched_p10_lt_neg3p364",
    "coord_prior_fixed_matched_p10_lt_neg3p209",
    "coord_prior_strict_duplicate_opt_lt_neg2p932",
]

VALID_OR_OBJECTNESS_LABELS = {
    "positive_gt_match",
    "positive_proxy_match",
    "neutral_plausible_proxy_match",
    "neutral_plausible_unmatched_candidate",
}

DANGER_TAIL_LABELS = {
    "bad_duplicate_like",
    "bad_invalid_geometry",
    "bad_border_or_top_left",
    "bad_wrong_location_candidate",
    "bad_malformed",
}


def build_threshold_prior_transfer_summary(
    rows: list[dict[str, Any]],
    *,
    proxy_source_summary: dict[str, Any] | None = None,
    resolved_config_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize how prior confidence/overlap bands transfer to current rows."""

    by_run: dict[str, dict[str, int]] = {}
    for row in rows:
        run = str(row.get("run_label"))
        label = str(row.get("auto_label"))
        by_run.setdefault(run, {})
        by_run[run][label] = by_run[run].get(label, 0) + 1

    return {
        "by_run": by_run,
        "row_count": len(rows),
        "proxy_source": proxy_source_summary or {},
        "resolved_config": resolved_config_summary or {},
        "prior_bands": {
            key: _summarize_prior_band(rows, key)
            for key in PRIOR_BAND_KEYS
        },
        "confidence_separability": _summarize_confidence_separability(rows),
    }


def _summarize_prior_band(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Summarize threshold tradeoff for one prior band."""

    flagged = [row for row in rows if bool(row.get(key))]
    return {
        "flagged_count": len(flagged),
        "flagged_valid_or_objectness_supported": sum(_is_valid_or_objectness_supported(row) for row in flagged),
        "flagged_danger_tail": sum(_is_danger_tail(row) for row in flagged),
        "flagged_unknown": sum(str(row.get("auto_label")) == "unknown" for row in flagged),
        "valid_or_objectness_supported_total": sum(_is_valid_or_objectness_supported(row) for row in rows),
        "danger_tail_total": sum(_is_danger_tail(row) for row in rows),
    }


def _summarize_confidence_separability(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Summarize coordinate-confidence separation for useful vs dangerous rows."""

    danger_values = [_coord_conf(row) for row in rows if _is_danger_tail(row) and _coord_conf(row) is not None]
    supported_values = [
        _coord_conf(row)
        for row in rows
        if _is_valid_or_objectness_supported(row) and _coord_conf(row) is not None
    ]
    return {
        "danger_tail_count": len(danger_values),
        "valid_or_objectness_supported_count": len(supported_values),
        "danger_tail_mean": _mean_or_none(danger_values),
        "valid_or_objectness_supported_mean": _mean_or_none(supported_values),
    }


def _is_valid_or_objectness_supported(row: dict[str, Any]) -> bool:
    """Return whether a row is valid or objectness-supported for retention accounting."""

    return (
        str(row.get("auto_label")) in VALID_OR_OBJECTNESS_LABELS
        or str(row.get("objectness_support_level")) in {"strong", "supported"}
    )


def _is_danger_tail(row: dict[str, Any]) -> bool:
    """Return whether a row is a danger-tail candidate for rejection accounting."""

    return str(row.get("auto_label")) in DANGER_TAIL_LABELS


def _coord_conf(row: dict[str, Any]) -> float | None:
    """Return coordinate confidence when present."""

    value = row.get("coord_confidence_mean")
    return None if value is None else float(value)


def _mean_or_none(values: list[float | None]) -> float | None:
    """Return the mean of present values."""

    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / float(len(present))
```

- [ ] **Step 4: Make `run_boundary_tail_diagnosis` write the threshold summary**

At the end of `run_boundary_tail_diagnosis`, write:

```python
# `proxy_source_summary` is loaded before the per-run loop. `resolved_config_summary`
# is populated while each run's `resolved_config.json` is read.
(cfg.output_dir / "threshold_prior_transfer_summary.json").write_text(
    json.dumps(
        build_threshold_prior_transfer_summary(
            object_rows,
            proxy_source_summary=proxy_source_summary,
            resolved_config_summary=resolved_config_summary,
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
```

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n ms python -m pytest \
  tests/test_boundary_tail_direction_gate.py::test_build_manual_audit_queue_selects_ambiguous_decision_cases \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_counts_by_run \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_includes_sources \
  tests/test_boundary_tail_direction_gate.py::test_build_threshold_prior_transfer_summary_reports_band_tradeoff -q
```

Expected: PASS.

## Task 9: Real Artifact Smoke And Schema Inspection

**Files:**

- No committed output changes expected.
- Runtime outputs under: `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/`.

- [ ] **Step 1: Run the table-first diagnosis on real artifacts**

Run:

```bash
conda run -n ms python scripts/analysis/diagnose_boundary_tail_direction_gate.py \
  --config configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml
```

Expected: writes:

```text
/data/CoordExp/temp/boundary_tail_direction_gate_20260513/image_run_summary.jsonl
/data/CoordExp/temp/boundary_tail_direction_gate_20260513/object_delta_table.jsonl
/data/CoordExp/temp/boundary_tail_direction_gate_20260513/manual_audit_queue.jsonl
/data/CoordExp/temp/boundary_tail_direction_gate_20260513/threshold_prior_transfer_summary.json
```

- [ ] **Step 2: Inspect output sizes**

Run:

```bash
wc -l /data/CoordExp/temp/boundary_tail_direction_gate_20260513/*.jsonl
python - <<'PY'
import json
from pathlib import Path

for name in ["image_run_summary.jsonl", "object_delta_table.jsonl", "manual_audit_queue.jsonl"]:
    path = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513") / name
    first = json.loads(path.open(encoding="utf-8").readline())
    print(name, sorted(first.keys()))
PY
```

Expected: non-empty image/object tables, manual queue may be empty or capped.

- [ ] **Step 3: Inspect threshold summary**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513/threshold_prior_transfer_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps(summary.get("by_run", {}), indent=2, sort_keys=True))
PY
```

Expected: counts for A2/A3/A4.

## Task 10: Optional Selective Review Packet

**Files:**

- Modify: `src/analysis/boundary_tail_direction_gate.py`
- Modify: `scripts/analysis/diagnose_boundary_tail_direction_gate.py`
- Optional output: `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/`

- [ ] **Step 1: Decide whether review packet is needed**

Inspect:

```bash
wc -l /data/CoordExp/temp/boundary_tail_direction_gate_20260513/manual_audit_queue.jsonl
```

Expected:

- If `0`, skip this task and proceed to reporting.
- If `>0`, generate a review packet only for queued cases.

- [ ] **Step 2: Enable review packet through YAML, not a new CLI flag**

Create a local execution copy of the analysis YAML or edit the checked-in YAML only if review materialization should become the default for this diagnosis:

```yaml
review_packet:
  materialize: true
```

The CLI should remain limited to `--config`; review-packet behavior must be reproducible from the YAML.

- [ ] **Step 3: Materialize a reviewer-compatible packet**

Use a concrete adapter instead of passing `manual_audit_queue.jsonl` directly into `src.analysis.raw_text_coordinate_review_queue.materialize_review_gallery`.

Supported paths:

- reviewer CSV path: create `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/manual_audit.csv` with `audit_id`, `display_id`, `overlay_path`, `crop_path`, `desc`, `nearest_gt_desc`, `nearest_gt_iou`, `proposal_uid`, `audit_label`, and `audit_notes`, then launch `scripts/analysis/run_manual_audit_reviewer.py`;
- gallery adapter path: convert queue rows to the existing raw-text gallery shortlist schema with `case_uid`, `review_bucket`, `selection_rank`, `model_alias`, `source_gt_vs_pred_jsonl`, `line_idx`, `object_index`, and `image_id` before calling `src.analysis.raw_text_coordinate_review_queue.materialize_review_gallery`.

Do not assume the analysis-native queue schema is already accepted by either helper.

Analysis-native queue fields to preserve in any review adapter:

- `case_id`;
- `audit_id`;
- `source_gt_vs_pred_jsonl`;
- `line_idx`;
- `case_uid`;
- `object_index`;
- `image_id`;
- `run_label`;
- `pred_index`;
- `desc`;
- `bbox_xyxy_norm1000`;
- `auto_label`;
- `decision_impact`;
- `question_for_user`;
- paths to any generated overlay/crop.

- [ ] **Step 4: Run review packet only if needed**

Run:

```bash
conda run -n ms python scripts/analysis/diagnose_boundary_tail_direction_gate.py \
  --config configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml
```

Expected: if `review_packet.materialize: true`, review packet under `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/`.

- [ ] **Step 5: Launch manual reviewer only after packet exists**

If using the reviewer CSV path, launch:

```bash
conda run -n ms python scripts/analysis/run_manual_audit_reviewer.py \
  --audit-csv /data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/manual_audit.csv \
  --port 8765
```

Expected: reviewer writes labels to `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/manual_audit_labels.jsonl` with `audit_id`, `audit_label`, and `audit_notes`. The report gate must read this labeled companion artifact, not require in-place edits to `manual_audit_queue.jsonl`.

## Task 11: Write Concise Diagnosis Report

**Files:**

- Create: `progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md`

- [ ] **Step 1: Read generated table summaries**

Run:

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

path = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513/object_delta_table.jsonl")
counts = defaultdict(Counter)
with path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        counts[row["run_label"]][row["auto_label"]] += 1
for run, counter in sorted(counts.items()):
    print(run, dict(counter))
PY
```

Expected: per-run label counts.

- [ ] **Step 2: Check whether manual audit blocks a concluded recommendation**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

queue_path = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513/manual_audit_queue.jsonl")
labels_path = Path("/data/CoordExp/temp/boundary_tail_direction_gate_20260513/review_packet/manual_audit_labels.jsonl")
labels_by_id = {}
if labels_path.exists():
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label_row = json.loads(line)
        labels_by_id[str(label_row.get("audit_id") or "")] = label_row
unresolved = []
if queue_path.exists():
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audit_id = str(row.get("audit_id") or row.get("case_id") or row.get("case_uid") or "")
        label_row = labels_by_id.get(audit_id, {})
        label = label_row.get("audit_label") or row.get("user_label")
        if row.get("decision_impact") and not label:
            unresolved.append(row)
print(f"unresolved_decision_critical_manual_cases={len(unresolved)}")
PY
```

Expected: prints the unresolved manual count after applying any labeled companion artifact. If this count is greater than zero, the report must mark the mechanism decision as blocked/provisional and the recommended next action as manual audit or a tiny discriminating probe, not a concluded training direction.

- [ ] **Step 3: Draft report with explicit uncertainty**

Create `progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md` with this structure:

```markdown
---
doc_id: progress.diagnostics.boundary-tail-direction-gate-2026-05-13
layer: progress
doc_type: diagnostic-note
status: Analyzing
domain: compact-detection
summary: Table-first A2/A3/A4 boundary-tail diagnosis for the next training/data/objective direction.
---

# Boundary/Tail Direction Gate

## Scope

## Artifact Inputs

## Method

## COCO-Strict Read

## Coverage-Aware LVIS/Objectness Read

## Duplicate-Collapse Evidence

## Prior Threshold Transfer

## Manual Audit

## Mechanism Decision

## Objective Sketches

## Recommended Next Action

## Falsifier

## Open Questions
```

Fill every section from generated artifacts only. If a section lacks evidence, write the exact missing artifact or inspection needed.

- [ ] **Step 4: Run a placeholder scan**

Run:

```bash
rg -n "TBD|TODO|fill in|placeholder|FIXME" \
  docs/superpowers/specs/2026-05-13-boundary-tail-direction-gate-design.md \
  docs/superpowers/plans/2026-05-13-boundary-tail-direction-gate.md \
  progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md
```

Expected: no matches.

## Task 12: Verification And Handoff

**Files:**

- Modify only files owned by this plan.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
conda run -n ms python -m pytest tests/test_boundary_tail_direction_gate.py -q
```

Expected: PASS.

- [ ] **Step 2: Run real-artifact smoke**

Run:

```bash
conda run -n ms python scripts/analysis/diagnose_boundary_tail_direction_gate.py \
  --config configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml
```

Expected: table outputs under `/data/CoordExp/temp/boundary_tail_direction_gate_20260513/`.

- [ ] **Step 3: Inspect git diff scope**

Run:

```bash
git status --short
git diff -- docs/superpowers/specs/2026-05-13-boundary-tail-direction-gate-design.md \
  docs/superpowers/plans/2026-05-13-boundary-tail-direction-gate.md \
  configs/analysis/boundary_tail_direction_gate/a2_a3_a4_val200.yaml \
  scripts/analysis/diagnose_boundary_tail_direction_gate.py \
  src/analysis/boundary_tail_direction_gate.py \
  tests/test_boundary_tail_direction_gate.py \
  progress/diagnostics/2026-05-13_boundary_tail_direction_gate.md
```

Expected: only intended files are modified or created. Unrelated dirt remains untouched.

- [ ] **Step 4: Final handoff summary**

Report:

- exact command run;
- table output paths;
- whether manual review is needed;
- top mechanism read;
- recommended next action;
- falsifier;
- tests run and pass/fail status.

Do not claim production-readiness unless a later training smoke or production candidate actually runs.

## Self-Review Checklist

- [ ] Spec and plan keep OpenSpec out of scope.
- [ ] Plan starts table-first and does not launch training.
- [ ] LVIS/proxy is objectness/semantic-neighborhood evidence, not exact truth.
- [ ] Duplicate handling protects dense instances.
- [ ] Prior threshold docs are used as baselines.
- [ ] Final report requires a falsifier.
- [ ] All generated docs avoid placeholder text.
