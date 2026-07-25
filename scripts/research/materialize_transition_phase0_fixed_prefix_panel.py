#!/usr/bin/env python3
"""Freeze the seven heldout Source-prefix mechanism cases from persisted traces.

This is a CPU-only materializer for the existing-checkpoint transition Phase
Zero unit.  It slices exact generated token identifiers from the persisted
Source and transition-step36 traces, validates the official owner-change
ledger and canonical owner matching, and writes the literal-token manifest
consumed by ``run_complete_candidate_row_scoring.py``.  It never tokenizes a
decoded string and never loads a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.compare_clean_rollout_owner_coverage import (  # noqa: E402
    _global_matches,
    _gt_objects,
    _pred_objects,
    _read_jsonl,
)
from scripts.research.run_complete_candidate_row_scoring import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    validate_manifest as validate_scoring_manifest,
)


UNIT_ID = "2026-07-25-existing-checkpoint-transition-mechanism-decomposition"
PANEL_SCHEMA_VERSION = "transition_phase0_fixed_prefix_panel.v1"
OBJECT_REF_START = 151646
BOX_END = 151649

OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-25-existing-checkpoint-transition-mechanism-decomposition"
)
TRANSFER_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-24-prefix-local-and-on-policy-owner-set-training/"
    "clean-rollouts-transition-step36-transfer-max3084-matched-b4-v1"
)
CONFIG_ROOT = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "research/qwen3_vl_2b_transition_step36_transfer_max3084_matched_b4_v1"
)
HELDOUT_JSONL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "candidate-pool-v1/heldout.jsonl"
)

DEFAULT_INPUT_IDENTITIES: dict[str, dict[str, str]] = {
    "source_trace": {
        "path": str(
            TRANSFER_ROOT
            / "qwen3-vl-2b-transition-step36-transfer-heldout-source-max3084-b4-hf"
            / "pred_token_trace.jsonl"
        ),
        "sha256": "01976c26b59101edb3e02d1358868b62bdeffe2a15ead7c439f91a3ebc37c1b3",
    },
    "treatment_trace": {
        "path": str(
            TRANSFER_ROOT
            / "qwen3-vl-2b-transition-step36-transfer-heldout-transition-step36-max3084-b4-hf"
            / "pred_token_trace.jsonl"
        ),
        "sha256": "3d0a4482f03713d3d11a4717ed3f22b3271ca525fd485471a153c9ccb7b6f337",
    },
    "source_raw": {
        "path": str(
            TRANSFER_ROOT
            / "qwen3-vl-2b-transition-step36-transfer-heldout-source-max3084-b4-hf"
            / "gt_vs_pred.jsonl"
        ),
        "sha256": "e3221258b4fa07fd6b4706c33036ee21f5e3e999550f4baa6cbdad31b7010ff6",
    },
    "treatment_raw": {
        "path": str(
            TRANSFER_ROOT
            / "qwen3-vl-2b-transition-step36-transfer-heldout-transition-step36-max3084-b4-hf"
            / "gt_vs_pred.jsonl"
        ),
        "sha256": "4001d10644bb75fc339eef8b9003c32adcecec0bdf80d1c755936f28416d72e7",
    },
    "owner_ledger": {
        "path": str(
            OUTPUT_ROOT
            / "heldout-owner-churn-review-v1/private/original_comparison_ledger.json"
        ),
        "sha256": "90e309bdcd3fb7c6543faa6427087fa5a0bfacbaefcbd36f25d9cabcd183f655",
    },
    "heldout_source_jsonl": {
        "path": str(HELDOUT_JSONL),
        "sha256": "8378af4429cc3cf2084da50a34fdf9b163e8d210291b3a787cdc05c1bdbd266e",
    },
}

DEFAULT_CHECKPOINT_CONFIGS: dict[str, dict[str, str]] = {
    "source": {
        "path": str(CONFIG_ROOT / "transition-step36-transfer-heldout-source-hf.yaml"),
        "sha256": "a116bc0a3fdb2d2bec5cd5322df7fa85323c740489124047dfecefad6446f8b3",
    },
    "transition-step36": {
        "path": str(
            CONFIG_ROOT / "transition-step36-transfer-heldout-transition-step36-hf.yaml"
        ),
        "sha256": "b85e5c09c244cefd0419964d51317b59296f875c0eb21384189dff565dcf30fb",
    },
}


def _candidate(
    candidate_id: str,
    *,
    arm: str,
    row_index: int,
    step_span: tuple[int, int],
    token_count: int,
    token_hash: str,
    category: str,
    role: str,
    truth_status: str,
    owner_index: int | None = None,
    owner_id: str | None = None,
    covered: bool | None = None,
    ledger_side: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "row_index": row_index,
        "step_span": list(step_span),
        "token_count": token_count,
        "token_ids_sha256": token_hash,
        "category": category,
        "role": role,
        "truth_status": truth_status,
        "owner_index": owner_index,
        "owner_id": owner_id,
        "covered": covered,
        "ledger_side": ledger_side,
    }


FROZEN_CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "heldout-3442-source-terminal-gain",
        "image_id": "3442",
        "row_id": "coco2017_train_000000003442",
        "case_role": "isolated treatment gain at the Source native terminal boundary",
        "source_prefix_row_count": 7,
        "prefix_step_span": [0, 63],
        "prefix_token_count": 64,
        "prefix_token_ids_sha256": "02180868af701ac29d6b324a6010741267a79df306b2729e43ec0c30a62872d7",
        "candidates": [
            _candidate(
                "gain-chair-1592243",
                arm="treatment",
                row_index=2,
                step_span=(18, 26),
                token_count=9,
                token_hash="d08c3e37dfb0c46cb09a8e430f2baed94db875796191491123587fdd9b7ed954",
                category="chair",
                role="gained_uncovered_owner",
                truth_status="geometry_matched_treatment_gain",
                owner_index=1,
                owner_id="1592243",
                covered=False,
                ledger_side="arm_b_only",
            ),
            _candidate(
                "covered-chair-1938373",
                arm="source",
                row_index=3,
                step_span=(28, 36),
                token_count=9,
                token_hash="ee12f4605abcb792948d6d89ad2932a030016e7138c8c3d4b56a7e84df26df5d",
                category="chair",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=5,
                owner_id="1938373",
                covered=True,
            ),
            _candidate(
                "covered-cup-678421",
                arm="source",
                row_index=5,
                step_span=(46, 54),
                token_count=9,
                token_hash="ba4bbaad83d60a7d598b7318b3c2d278a1a2d55b44a257764a06e2a72d448044",
                category="cup",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=10,
                owner_id="678421",
                covered=True,
            ),
        ],
    },
    {
        "case_id": "heldout-4129-person-owner-exchange",
        "image_id": "4129",
        "row_id": "coco2017_train_000000004129",
        "case_role": "same-category person gain and loss exchange",
        "source_prefix_row_count": 6,
        "prefix_step_span": [0, 53],
        "prefix_token_count": 54,
        "prefix_token_ids_sha256": "ab400c2edb0ecedf1824264965fa54f6e99c8916b3d481d8eed0604b4d750d75",
        "candidates": [
            _candidate(
                "loss-person-265782",
                arm="source",
                row_index=6,
                step_span=(54, 62),
                token_count=9,
                token_hash="2a8be12c3659bba24b356fd558db6e997c6ca81f591daeac3503458bf4503a06",
                category="person",
                role="lost_uncovered_owner",
                truth_status="geometry_matched_source_loss",
                owner_index=3,
                owner_id="265782",
                covered=False,
                ledger_side="arm_a_only",
            ),
            _candidate(
                "gain-person-262892",
                arm="treatment",
                row_index=7,
                step_span=(63, 71),
                token_count=9,
                token_hash="1fbf99b85582b22262031268082868cccdf4c0a0f1538c806491b82caa9c8f9b",
                category="person",
                role="gained_uncovered_owner",
                truth_status="geometry_matched_treatment_gain",
                owner_index=4,
                owner_id="262892",
                covered=False,
                ledger_side="arm_b_only",
            ),
            _candidate(
                "covered-person-1282635",
                arm="source",
                row_index=4,
                step_span=(36, 44),
                token_count=9,
                token_hash="b47950809f36155986b7543de0d8c262541a8c9b0cd1ef7cdc385824a6a6b4bd",
                category="person",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=1,
                owner_id="1282635",
                covered=True,
            ),
        ],
    },
    {
        "case_id": "heldout-15379-clean-car-gain",
        "image_id": "15379",
        "row_id": "coco2017_train_000000015379",
        "case_role": "clean car gain at the Source terminal boundary with a neutral alternative",
        "source_prefix_row_count": 13,
        "prefix_step_span": [0, 117],
        "prefix_token_count": 118,
        "prefix_token_ids_sha256": "74b41a04e63e459b8f339b54146d38b3c149e72a2bf03745829fe19e47ab1e13",
        "candidates": [
            _candidate(
                "gain-car-2209199",
                arm="treatment",
                row_index=8,
                step_span=(72, 80),
                token_count=9,
                token_hash="913c09eefab06369be4f629b78f92842a3e4ebbf63f8619214e23ecc98e027f3",
                category="car",
                role="gained_uncovered_owner",
                truth_status="geometry_matched_treatment_gain",
                owner_index=5,
                owner_id="2209199",
                covered=False,
                ledger_side="arm_b_only",
            ),
            _candidate(
                "covered-car-138661",
                arm="source",
                row_index=7,
                step_span=(63, 71),
                token_count=9,
                token_hash="f4bd3a72d64da6e0cdd4eabe3c879fdb6fdf841ad5f4582a8c01bfcaac0d72c3",
                category="car",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=8,
                owner_id="138661",
                covered=True,
            ),
            _candidate(
                "neutral-car-source-row-4",
                arm="source",
                row_index=4,
                step_span=(36, 44),
                token_count=9,
                token_hash="f2fe5202d9b6b7f8e7498cf3256afa40c91057c41ee0554d786ee9c77931ff3e",
                category="car",
                role="unmatched_neutral_candidate",
                truth_status="unmatched_prediction_neutral",
            ),
        ],
    },
    {
        "case_id": "heldout-28058-person-loss",
        "image_id": "28058",
        "row_id": "coco2017_train_000000028058",
        "case_role": "Source-only person loss with a future retained person alternative",
        "source_prefix_row_count": 6,
        "prefix_step_span": [0, 53],
        "prefix_token_count": 54,
        "prefix_token_ids_sha256": "63d7356188bc593f0b6056d9d2a02a928e9353b1d75a8bfa8d6af957a2f04cc9",
        "candidates": [
            _candidate(
                "loss-person-1212049",
                arm="source",
                row_index=6,
                step_span=(54, 62),
                token_count=9,
                token_hash="2bd809406224d970c4a87b08e2c7615b490914b1f59645b24551824619108985",
                category="person",
                role="lost_uncovered_owner",
                truth_status="geometry_matched_source_loss",
                owner_index=8,
                owner_id="1212049",
                covered=False,
                ledger_side="arm_a_only",
            ),
            _candidate(
                "covered-person-1255172",
                arm="source",
                row_index=5,
                step_span=(45, 53),
                token_count=9,
                token_hash="0c43f3f149588d910841d684d518e50d3dd298a9010f193fa07ed17e46e34a85",
                category="person",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=4,
                owner_id="1255172",
                covered=True,
            ),
            _candidate(
                "future-retained-person-226531",
                arm="source",
                row_index=9,
                step_span=(81, 89),
                token_count=9,
                token_hash="5d909034138cfc3de481acf08d39d8db5491aadd240cfb0ee7c11db4c713a8dc",
                category="person",
                role="future_retained_uncovered_owner",
                truth_status="geometry_matched_retained_owner_uncovered_at_prefix",
                owner_index=11,
                owner_id="226531",
                covered=False,
            ),
        ],
    },
    {
        "case_id": "heldout-65891-multi-car-choice",
        "image_id": "65891",
        "row_id": "coco2017_train_000000065891",
        "case_role": "three-way same-category car choice with one gain and two losses",
        "source_prefix_row_count": 7,
        "prefix_step_span": [0, 68],
        "prefix_token_count": 69,
        "prefix_token_ids_sha256": "d0c66967e686dfa1d8d5223820dd4593c5b6cc28d98025c68f544a3096c3865e",
        "candidates": [
            _candidate(
                "gain-car-1785098",
                arm="treatment",
                row_index=7,
                step_span=(69, 77),
                token_count=9,
                token_hash="8b9940ec47e3d122092a2ec45fd2e71b87b3642a0940223c72177135427b9f46",
                category="car",
                role="gained_uncovered_owner",
                truth_status="geometry_matched_treatment_gain",
                owner_index=8,
                owner_id="1785098",
                covered=False,
                ledger_side="arm_b_only",
            ),
            _candidate(
                "loss-car-361825",
                arm="source",
                row_index=10,
                step_span=(96, 104),
                token_count=9,
                token_hash="61e80ef63f0115dc9b79b11939eb965a3103d7d755bcaf9e4d30caa6662094e5",
                category="car",
                role="lost_uncovered_owner",
                truth_status="geometry_matched_source_loss",
                owner_index=14,
                owner_id="361825",
                covered=False,
                ledger_side="arm_a_only",
            ),
            _candidate(
                "loss-car-2042424",
                arm="source",
                row_index=15,
                step_span=(141, 149),
                token_count=9,
                token_hash="18fdc9db16882db78b0bc782f0a008e80a359c137d29a04821a63bd3e9c2125a",
                category="car",
                role="lost_uncovered_owner",
                truth_status="geometry_matched_source_loss",
                owner_index=13,
                owner_id="2042424",
                covered=False,
                ledger_side="arm_a_only",
            ),
            _candidate(
                "covered-traffic-light-409036",
                arm="source",
                row_index=0,
                step_span=(0, 9),
                token_count=10,
                token_hash="19c9849a3fdead0a328ebcd258d512f808948c7fdb2be616ef96c4807a352b4d",
                category="traffic light",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=0,
                owner_id="409036",
                covered=True,
            ),
        ],
    },
    {
        "case_id": "heldout-70558-unchanged-bird-control",
        "image_id": "70558",
        "row_id": "coco2017_train_000000070558",
        "case_role": "unchanged same-category bird preservation control",
        "source_prefix_row_count": 1,
        "prefix_step_span": [0, 8],
        "prefix_token_count": 9,
        "prefix_token_ids_sha256": "1fde869b0190039f97613d907884974db7dab35b83015565529d69e3569d8b4a",
        "candidates": [
            _candidate(
                "covered-bird-43529",
                arm="source",
                row_index=0,
                step_span=(0, 8),
                token_count=9,
                token_hash="1fde869b0190039f97613d907884974db7dab35b83015565529d69e3569d8b4a",
                category="bird",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=0,
                owner_id="43529",
                covered=True,
            ),
            _candidate(
                "future-retained-bird-43135",
                arm="source",
                row_index=1,
                step_span=(9, 17),
                token_count=9,
                token_hash="e2cea5595a830df998e55d805c56445a153981439486a898b9cb215922407640",
                category="bird",
                role="future_retained_uncovered_owner",
                truth_status="geometry_matched_retained_owner_uncovered_at_prefix",
                owner_index=1,
                owner_id="43135",
                covered=False,
            ),
            _candidate(
                "future-retained-bird-42353",
                arm="source",
                row_index=2,
                step_span=(18, 26),
                token_count=9,
                token_hash="084e069140cd7475bd63c0b80e9e48f7c4cf0dbe850b4ab3fa235bccadfdd3dc",
                category="bird",
                role="future_retained_uncovered_owner",
                truth_status="geometry_matched_retained_owner_uncovered_at_prefix",
                owner_index=2,
                owner_id="42353",
                covered=False,
            ),
        ],
    },
    {
        "case_id": "heldout-355385-treatment-length-stop",
        "image_id": "355385",
        "row_id": "coco2017_train_000000355385",
        "case_role": "treatment length-stop and repeated-fork row-realization control",
        "source_prefix_row_count": 5,
        "prefix_step_span": [0, 46],
        "prefix_token_count": 47,
        "prefix_token_ids_sha256": "068af57934ee02ffa1bfbce93764533058768efb73baec90d119af6955cfef23",
        "candidates": [
            _candidate(
                "neutral-source-fork-row-5",
                arm="source",
                row_index=5,
                step_span=(47, 55),
                token_count=9,
                token_hash="7b66cd22c7088f2336a8c0c659e45388a70ab7eb211dd3b83de7bb4661904049",
                category="fork",
                role="unmatched_neutral_candidate",
                truth_status="unmatched_prediction_neutral",
            ),
            _candidate(
                "neutral-treatment-fork-row-5",
                arm="treatment",
                row_index=5,
                step_span=(47, 55),
                token_count=9,
                token_hash="23cbbcd671dfc04bae0a0b4cf1618ee29f7e07494331a66aafeb5d26ce6e6e2e",
                category="fork",
                role="unmatched_neutral_candidate",
                truth_status="unmatched_prediction_neutral",
            ),
            _candidate(
                "covered-knife-693922",
                arm="source",
                row_index=4,
                step_span=(38, 46),
                token_count=9,
                token_hash="cda06678ad74cf5d2d8d2213359b1a001be3370068f9180205e6550477a733dd",
                category="knife",
                role="covered_owner_control",
                truth_status="geometry_matched_covered_owner",
                owner_index=4,
                owner_id="693922",
                covered=True,
            ),
        ],
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token) for token in token_ids],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_file_identity(reference: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    expected = reference.get("sha256")
    observed = sha256_file(path)
    if not isinstance(expected, str) or observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )
    return path


def _generated_trace_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            if value.get("trace_type") != "generated_token" or value.get("is_pad") is True:
                continue
            row_id = value.get("row_id")
            step = value.get("generated_step_index")
            token_id = value.get("token_id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"{path}:{line_number} generated token lacks row_id")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"{path}:{line_number} has invalid generated step")
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"{path}:{line_number} has invalid token_id")
            grouped[row_id].append(value)
    for row_id, records in grouped.items():
        records.sort(key=lambda record: int(record["generated_step_index"]))
        steps = [int(record["generated_step_index"]) for record in records]
        if steps != list(range(len(steps))):
            raise ValueError(f"trace steps for {row_id} are not contiguous from zero")
    return dict(grouped)


def split_complete_trace_rows(
    records: Sequence[Mapping[str, Any]], *, row_id: str
) -> list[list[Mapping[str, Any]]]:
    rows: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("is_stop") is True:
            break
        current.append(record)
        if int(record["token_id"]) == BOX_END:
            if int(current[0]["token_id"]) != OBJECT_REF_START:
                raise ValueError(f"trace row {len(rows)} for {row_id} lacks canonical opener")
            rows.append(current)
            current = []
    return rows


def _trace_span(row: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(row[0]["generated_step_index"]),
        int(row[-1]["generated_step_index"]),
    ]


def validate_token_slice(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_step_span: Sequence[int],
    expected_token_count: int,
    expected_sha256: str,
    label: str,
) -> list[int]:
    observed_span = _trace_span(records)
    if observed_span != [int(value) for value in expected_step_span]:
        raise ValueError(
            f"{label} step span mismatch: observed {observed_span}, "
            f"expected {list(expected_step_span)}"
        )
    token_ids = [int(record["token_id"]) for record in records]
    if len(token_ids) != int(expected_token_count):
        raise ValueError(f"{label} token count mismatch")
    observed_hash = token_ids_sha256(token_ids)
    if observed_hash != expected_sha256:
        raise ValueError(
            f"{label} token hash mismatch: observed {observed_hash}, expected {expected_sha256}"
        )
    return token_ids


def _owner_matches(row: dict[str, Any], *, row_id: str) -> dict[int, tuple[int, float]]:
    gt = _gt_objects(row, row_id=row_id)
    pred, invalid = _pred_objects(row)
    raw_predictions = row.get("pred")
    if invalid or not isinstance(raw_predictions, list) or len(pred) != len(raw_predictions):
        raise ValueError(f"selected row {row_id} has invalid or index-shifting predictions")
    return {
        int(prediction_index): (int(owner_index), float(overlap))
        for owner_index, prediction_index, overlap in _global_matches(gt, pred, 0.50)
    }


def _owner_ref_set(ledger: Mapping[str, Any], field: str) -> set[tuple[str, int]]:
    geometry = ledger.get("common_owner_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("owner ledger lacks common_owner_geometry")
    refs = geometry.get(field)
    if not isinstance(refs, list):
        raise ValueError(f"owner ledger lacks {field}")
    return {
        (str(reference["row_id"]), int(reference["owner_index"]))
        for reference in refs
        if isinstance(reference, Mapping)
    }


def _candidate_manifest_entry(
    spec: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    trace_rows: Mapping[str, list[list[Mapping[str, Any]]]],
    raw_rows: Mapping[str, Mapping[str, dict[str, Any]]],
    match_maps: Mapping[str, Mapping[int, tuple[int, float]]],
    covered_owner_indices: set[int],
    gain_refs: set[tuple[str, int]],
    loss_refs: set[tuple[str, int]],
    input_identities: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    arm = str(spec["arm"])
    row_id = str(case["row_id"])
    row_index = int(spec["row_index"])
    rows = trace_rows[arm]
    if not 0 <= row_index < len(rows):
        raise ValueError(f"{case['case_id']} candidate row index is absent: {row_index}")
    row_tokens = validate_token_slice(
        rows[row_index],
        expected_step_span=spec["step_span"],
        expected_token_count=int(spec["token_count"]),
        expected_sha256=str(spec["token_ids_sha256"]),
        label=f"{case['case_id']}:{spec['candidate_id']}",
    )
    raw = raw_rows[arm][row_id]
    predictions = raw.get("pred")
    if not isinstance(predictions, list) or not 0 <= row_index < len(predictions):
        raise ValueError(f"{case['case_id']} parsed candidate row is absent")
    prediction = predictions[row_index]
    if not isinstance(prediction, Mapping):
        raise ValueError(f"{case['case_id']} parsed candidate row is malformed")
    category = str(prediction.get("description", ""))
    if category != str(spec["category"]):
        raise ValueError(f"{case['case_id']} candidate category mismatch")

    owner_index = spec.get("owner_index")
    owner_id = spec.get("owner_id")
    match = match_maps[arm].get(row_index)
    owner_match: dict[str, Any] | None = None
    if owner_index is None:
        if match is not None:
            raise ValueError(f"{case['case_id']} neutral candidate became geometry-matched")
        if owner_id is not None or spec.get("covered") is not None:
            raise ValueError(f"{case['case_id']} neutral candidate carries an owner label")
    else:
        expected_owner_index = int(owner_index)
        if match is None or int(match[0]) != expected_owner_index:
            raise ValueError(f"{case['case_id']} candidate owner attribution mismatch")
        gt = raw.get("gt")
        if not isinstance(gt, list) or not 0 <= expected_owner_index < len(gt):
            raise ValueError(f"{case['case_id']} owner index is absent from GT")
        owner = gt[expected_owner_index]
        if not isinstance(owner, Mapping):
            raise ValueError(f"{case['case_id']} owner GT row is malformed")
        observed_owner_id = str(owner.get("object_id", owner.get("id", expected_owner_index)))
        observed_category = str(owner.get("description", owner.get("category", "")))
        if observed_owner_id != str(owner_id) or observed_category != category:
            raise ValueError(f"{case['case_id']} owner identity or category mismatch")
        observed_covered = expected_owner_index in covered_owner_indices
        if observed_covered is not bool(spec.get("covered")):
            raise ValueError(f"{case['case_id']} candidate covered status mismatch")
        owner_ref = (row_id, expected_owner_index)
        if spec.get("ledger_side") == "arm_a_only" and owner_ref not in loss_refs:
            raise ValueError(f"{case['case_id']} declared loss is absent from official ledger")
        if spec.get("ledger_side") == "arm_b_only" and owner_ref not in gain_refs:
            raise ValueError(f"{case['case_id']} declared gain is absent from official ledger")
        if str(spec["role"]).startswith("future_retained"):
            if expected_owner_index not in set(match[0] for match in match_maps["source"].values()):
                raise ValueError(f"{case['case_id']} future owner is not Source-retained")
            if expected_owner_index not in set(match[0] for match in match_maps["treatment"].values()):
                raise ValueError(f"{case['case_id']} future owner is not treatment-retained")
        owner_match = {
            "owner_index": expected_owner_index,
            "owner_id": observed_owner_id,
            "category": observed_category,
            "iou": float(match[1]),
            "covered_at_source_prefix": observed_covered,
        }

    trace_key = "source_trace" if arm == "source" else "treatment_trace"
    return {
        "candidate_id": str(spec["candidate_id"]),
        "row": {
            "token_ids": row_tokens,
            "token_ids_sha256": str(spec["token_ids_sha256"]),
        },
        "owner": None if owner_id is None else str(owner_id),
        "category": category,
        "role": str(spec["role"]),
        "covered": spec.get("covered"),
        "truth_status": str(spec["truth_status"]),
        "source": {
            "arm": arm,
            "trace": dict(input_identities[trace_key]),
            "trace_row_id": row_id,
            "generated_row_index": row_index,
            "generated_step_span_inclusive": list(spec["step_span"]),
            "parsed_prediction": {
                "coord_bins": prediction.get("coord_bins"),
                "raw_span_sha256": prediction.get("raw_span_sha256"),
            },
            "owner_match": owner_match,
            "official_ledger_side": spec.get("ledger_side"),
        },
    }


def build_panel_manifest(
    *,
    input_identities: Mapping[str, Mapping[str, str]] = DEFAULT_INPUT_IDENTITIES,
    checkpoint_configs: Mapping[str, Mapping[str, str]] = DEFAULT_CHECKPOINT_CONFIGS,
    case_specs: Sequence[Mapping[str, Any]] = FROZEN_CASE_SPECS,
) -> dict[str, Any]:
    required_inputs = {
        "source_trace",
        "treatment_trace",
        "source_raw",
        "treatment_raw",
        "owner_ledger",
        "heldout_source_jsonl",
    }
    if set(input_identities) != required_inputs:
        raise ValueError("panel input identities do not match the required frozen surfaces")
    if set(checkpoint_configs) != {"source", "transition-step36"}:
        raise ValueError("panel requires Source and transition-step36 checkpoint configs")
    paths = {
        label: verify_file_identity(reference, label=label)
        for label, reference in input_identities.items()
    }
    for role, reference in checkpoint_configs.items():
        verify_file_identity(reference, label=f"checkpoint_config:{role}")

    traces = {
        "source": _generated_trace_rows(paths["source_trace"]),
        "treatment": _generated_trace_rows(paths["treatment_trace"]),
    }
    raw_rows = {
        "source": _read_jsonl(paths["source_raw"]),
        "treatment": _read_jsonl(paths["treatment_raw"]),
    }
    ledger = _read_json(paths["owner_ledger"])
    geometry = ledger.get("common_owner_geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("owner ledger lacks common_owner_geometry")
    if int(geometry.get("arm_a_only_owner_count", -1)) != 39:
        raise ValueError("official Source-only owner count is not 39")
    if int(geometry.get("arm_b_only_owner_count", -1)) != 46:
        raise ValueError("official treatment-only owner count is not 46")
    loss_refs = _owner_ref_set(ledger, "arm_a_only_owner_refs")
    gain_refs = _owner_ref_set(ledger, "arm_b_only_owner_refs")

    images: list[dict[str, Any]] = []
    for case in case_specs:
        row_id = str(case["row_id"])
        if row_id not in traces["source"] or row_id not in traces["treatment"]:
            raise ValueError(f"case trace row is absent: {row_id}")
        if row_id not in raw_rows["source"] or row_id not in raw_rows["treatment"]:
            raise ValueError(f"case raw row is absent: {row_id}")
        trace_rows = {
            arm: split_complete_trace_rows(traces[arm][row_id], row_id=row_id)
            for arm in ("source", "treatment")
        }
        prefix_row_count = int(case["source_prefix_row_count"])
        if not 1 <= prefix_row_count <= len(trace_rows["source"]):
            raise ValueError(f"{case['case_id']} Source prefix row count is unavailable")
        prefix_records = [
            record
            for row in trace_rows["source"][:prefix_row_count]
            for record in row
        ]
        prefix_tokens = validate_token_slice(
            prefix_records,
            expected_step_span=case["prefix_step_span"],
            expected_token_count=int(case["prefix_token_count"]),
            expected_sha256=str(case["prefix_token_ids_sha256"]),
            label=f"{case['case_id']}:source-prefix",
        )
        match_maps = {
            arm: _owner_matches(raw_rows[arm][row_id], row_id=row_id)
            for arm in ("source", "treatment")
        }
        covered_owner_indices = {
            owner_index
            for prediction_index, (owner_index, _) in match_maps["source"].items()
            if prediction_index < prefix_row_count
        }
        source_gt = raw_rows["source"][row_id].get("gt")
        if not isinstance(source_gt, list):
            raise ValueError(f"{case['case_id']} Source GT is malformed")
        covered_owner_ids = sorted(
            str(source_gt[index].get("object_id", source_gt[index].get("id", index)))
            for index in covered_owner_indices
        )
        candidates = [
            _candidate_manifest_entry(
                candidate,
                case=case,
                trace_rows=trace_rows,
                raw_rows=raw_rows,
                match_maps=match_maps,
                covered_owner_indices=covered_owner_indices,
                gain_refs=gain_refs,
                loss_refs=loss_refs,
                input_identities=input_identities,
            )
            for candidate in case["candidates"]
        ]
        images.append(
            {
                "image_id": str(case["image_id"]),
                "row_id": row_id,
                "case_id": str(case["case_id"]),
                "case_role": str(case["case_role"]),
                "boundaries": [
                    {
                        "boundary_id": str(case["case_id"]),
                        "prefix_mode": "base_prompt_plus_generated",
                        "prefix": {
                            "token_ids": prefix_tokens,
                            "token_ids_sha256": str(case["prefix_token_ids_sha256"]),
                        },
                        "source_completed_row_count": prefix_row_count,
                        "source_generated_step_span_inclusive": list(
                            case["prefix_step_span"]
                        ),
                        "covered_owner_ids": covered_owner_ids,
                        "case_role": str(case["case_role"]),
                        "candidates": candidates,
                    }
                ],
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "panel": {
            "schema_version": PANEL_SCHEMA_VERSION,
            "case_count": len(images),
            "selection": "exact seven heldout Source-produced row-boundary cases",
            "prefix_origin": "Source generated_token trace; terminal token excluded",
            "primary_owner_match_iou": 0.50,
            "official_owner_change_counts": {
                "source_only_losses": 39,
                "treatment_only_gains": 46,
            },
            "inputs": {
                label: dict(reference) for label, reference in input_identities.items()
            },
            "checkpoint_configs": {
                role: dict(reference) for role, reference in checkpoint_configs.items()
            },
            "claim_boundary": (
                "fixed-prefix scoring and forced release are diagnostic proxies, not "
                "free-rollout final-set improvement"
            ),
        },
        "images": images,
    }
    validate_frozen_panel_manifest(manifest, verify_files=False)
    return manifest


def validate_frozen_panel_manifest(
    manifest: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("unit_id") != UNIT_ID:
        raise ValueError(f"expected unit_id={UNIT_ID}")
    panel = manifest.get("panel")
    if not isinstance(panel, Mapping) or panel.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError(f"expected panel schema {PANEL_SCHEMA_VERSION}")
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != len(FROZEN_CASE_SPECS):
        raise ValueError("frozen panel must contain exactly seven images")
    expected_case_ids = [str(case["case_id"]) for case in FROZEN_CASE_SPECS]
    observed_case_ids = [str(image.get("case_id")) for image in images]
    if observed_case_ids != expected_case_ids:
        raise ValueError("frozen panel case identities or order drifted")
    if int(panel.get("case_count", -1)) != len(images):
        raise ValueError("panel case_count does not match images")
    inputs = panel.get("inputs")
    checkpoint_configs = panel.get("checkpoint_configs")
    if not isinstance(inputs, Mapping) or dict(inputs) != DEFAULT_INPUT_IDENTITIES:
        raise ValueError("panel input identities drifted from the frozen contract")
    if (
        not isinstance(checkpoint_configs, Mapping)
        or dict(checkpoint_configs) != DEFAULT_CHECKPOINT_CONFIGS
    ):
        raise ValueError("panel checkpoint configs drifted from the frozen contract")
    if panel.get("official_owner_change_counts") != {
        "source_only_losses": 39,
        "treatment_only_gains": 46,
    }:
        raise ValueError("panel official owner-change counts drifted")

    for image, case in zip(images, FROZEN_CASE_SPECS, strict=True):
        if not isinstance(image, Mapping) or any(
            str(image.get(field)) != str(case[field])
            for field in ("image_id", "row_id", "case_id", "case_role")
        ):
            raise ValueError(f"{case['case_id']} identity or role drifted")
        if "base_prompt" in image or "base_prompt_token_ids" in image:
            raise ValueError("fixed panel must omit base_prompt")
        boundaries = image.get("boundaries")
        if not isinstance(boundaries, list) or len(boundaries) != 1:
            raise ValueError(f"{case['case_id']} must contain one boundary")
        boundary = boundaries[0]
        if not isinstance(boundary, Mapping):
            raise ValueError(f"{case['case_id']} boundary must be an object")
        prefix = boundary.get("prefix")
        observed_prefix = (
            boundary.get("boundary_id"),
            boundary.get("source_completed_row_count"),
            boundary.get("source_generated_step_span_inclusive"),
            boundary.get("case_role"),
            prefix.get("token_ids_sha256") if isinstance(prefix, Mapping) else None,
        )
        expected_prefix = (
            case["case_id"],
            case["source_prefix_row_count"],
            case["prefix_step_span"],
            case["case_role"],
            case["prefix_token_ids_sha256"],
        )
        if observed_prefix != expected_prefix:
            raise ValueError(f"{case['case_id']} prefix contract drifted")
        candidates = boundary.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(case["candidates"]):
            raise ValueError(f"{case['case_id']} candidate count drifted")
        for candidate, spec in zip(candidates, case["candidates"], strict=True):
            row = candidate.get("row") if isinstance(candidate, Mapping) else None
            source = candidate.get("source") if isinstance(candidate, Mapping) else None
            observed = (
                candidate.get("candidate_id") if isinstance(candidate, Mapping) else None,
                candidate.get("role") if isinstance(candidate, Mapping) else None,
                row.get("token_ids_sha256") if isinstance(row, Mapping) else None,
                source.get("arm") if isinstance(source, Mapping) else None,
                source.get("generated_row_index") if isinstance(source, Mapping) else None,
                source.get("generated_step_span_inclusive")
                if isinstance(source, Mapping)
                else None,
            )
            expected = (
                spec["candidate_id"],
                spec["role"],
                spec["token_ids_sha256"],
                spec["arm"],
                spec["row_index"],
                spec["step_span"],
            )
            if observed != expected:
                raise ValueError(
                    f"{case['case_id']}:{spec['candidate_id']} contract drifted"
                )
    if verify_files:
        for label, reference in inputs.items():
            if not isinstance(reference, Mapping):
                raise ValueError(f"panel input {label} is not an object")
            verify_file_identity(reference, label=f"panel.inputs.{label}")
        for role, reference in checkpoint_configs.items():
            if not isinstance(reference, Mapping):
                raise ValueError(f"checkpoint config {role} is not an object")
            verify_file_identity(reference, label=f"panel.checkpoint_configs.{role}")
    normalized = validate_scoring_manifest(
        manifest,
        manifest_path=Path("/manifest/transition-phase0-fixed-prefix-panel.json"),
    )
    for image in normalized["images"]:
        boundaries = image["boundaries"]
        if len(boundaries) != 1 or boundaries[0]["prefix_mode"] != "base_prompt_plus_generated":
            raise ValueError("each fixed panel image must contain one generated-suffix boundary")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable manifest: {output}")
    manifest = build_panel_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "case_count": len(manifest["images"]),
                "schema_version": manifest["schema_version"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
