"""Matched-history mechanism diagnostic for the owner-successor snapshot.

The diagnostic is deliberately narrower than an endpoint evaluator.  CPU
preparation binds two already-admitted images, two literal candidate rows per
image, and the sealed Stable50/N16 row sources.  The optional endpoint path
then replays exactly those literal histories on each frozen adapter.  It does
not search for replacement candidates, train, patch KV state, or award any
credit to rows supplied in a prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from probes.dora_owner_learning.candidate_opportunity import file_hash, require, score
from probes.dora_owner_learning.geometric_dedup_eval import overlap_counts
from probes.native_owner_scale import evaluation as evaluation
from probes.parallel_owner_research.history import complete_rows, continuation_ledger
from src.inference.bound_requests import build_bound_native_requests as build_requests
from src.eval.native_rows import native_detection_record as native_record
from src.data.geometry import iou_xyxy
from src.losses.token_scores import aligned_token_logprobs
from src.qwen.native import prepare_native_inputs, prepare_replay


BASE = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration")
OLD_ROOT = BASE / "2026-09-12-native-owner-scale-and-state"
UNIT_ROOT = BASE / "2026-09-13-owner-successor-scale-throughput"
ROOT = UNIT_ROOT / "history"
PREPARATION = ROOT / "preparation"
INPUTS = OLD_ROOT / "scale/training/preparation/inputs-v2.json"
PANEL = OLD_ROOT / "evaluation/candidate-panel-bound-v11.json"
TRUSTED = OLD_ROOT / "evaluation/candidate-trusted-target-ledger-v10.json"
STABLE_ROWS = OLD_ROOT / "evaluation/candidate-stable-rows-v10.jsonl"
STABLE_PROJECTION = OLD_ROOT / "evaluation/candidate-stable-projection-v10.json"
N16_ROWS = OLD_ROOT / "evaluation/candidate-natural-v11/rows.jsonl"
N16_CONSUMER = OLD_ROOT / "evaluation/candidate-natural-v11-consumer/result.json"
RUNTIME_INPUT_JSONL = BASE / "2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"
CAP = 3084
EOS = 151645
ROW_START = 151646
ROW_END = 151649
GPUS = (6, 7)

STABLE_ADAPTER = (
    BASE
    / "2026-09-10-selective-owner-learning-autonomous/positive7-support50-81/training/adapter"
)
N16_ADAPTER = OLD_ROOT / "scale/training/full-fixedP-N16-v2/adapter"
N16_FINGERPRINT = "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"
STABLE_FINGERPRINT = "b3cf3ce468f9d40c7f6a629cf1a6265b8ba218e31d1078ef32af13ec15e63434"

# This is the complete fixed candidate set.  In particular, a failure below
# is a HOLD and never causes a replacement search.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "image_id": 210457,
        "case_id": "coco2017_train_000000210457:672025",
        "c_record_id": "coco2017_train_000000210457:672025:c",
        "w_record_id": "coco2017_train_000000210457:672025:w_kl",
        "class": "cup",
        "target_owner": "672025",
        "target_bins": [295, 337, 340, 419],
        "registered_bins": [435, 322, 472, 407],
    },
    {
        "image_id": 219546,
        "case_id": "coco2017_train_000000219546:708465",
        "c_record_id": "coco2017_train_000000219546:708465:c",
        "w_record_id": "coco2017_train_000000219546:708465:w_kl",
        "class": "spoon",
        "target_owner": "708465",
        "target_bins": [532, 554, 641, 646],
        "registered_bins": [966, 449, 999, 498],
    },
)


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _binding(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def _publish(path: str | Path, value: Any) -> None:
    evaluation.publish(path, value)


def _require_sources(paths: Sequence[str | Path]) -> None:
    for path in paths:
        require(Path(path).exists(), f"missing history source: {path}")


def _runtime_config(panel_config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep panel semantics while using the historical path root for images.

    The bound panel rows carry a relative image reference with five ``..``
    components.  Its current val JSONL path resolves that reference to
    ``/public_data``; the prior accepted train-input path supplies the correct
    `/data/CoordExp` root.  Only ``data.input_jsonl`` changes; template,
    backend, generation, and model fields remain byte-identical.
    """

    value = copy.deepcopy(dict(panel_config))
    value["data"] = copy.deepcopy(value["data"])
    value["data"]["input_jsonl"] = str(RUNTIME_INPUT_JSONL)
    unchanged = copy.deepcopy(dict(panel_config))
    unchanged["data"] = copy.deepcopy(unchanged["data"])
    unchanged["data"]["input_jsonl"] = str(RUNTIME_INPUT_JSONL)
    value_without_data = copy.deepcopy(value)
    panel_without_data = copy.deepcopy(dict(panel_config))
    value_without_data.pop("data", None)
    panel_without_data.pop("data", None)
    require(value_without_data == panel_without_data, "runtime config changed non-data inference semantics")
    return value


def _without_eos(ids: Sequence[int]) -> list[int]:
    values = [int(value) for value in ids]
    require(values, "empty sealed action IDs")
    if values[-1] == EOS:
        values = values[:-1]
    require(EOS not in values, "EOS appears inside a sealed action sequence")
    return values


def _source_rows(row: Mapping[str, Any]) -> list[list[int]]:
    """Return complete literal rows without decoding or re-tokenizing."""

    return complete_rows(_without_eos(row["action_ids"]))


def _panel_indexes(panel: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records = panel.get("records")
    require(isinstance(records, list), "candidate panel records")
    indexed = {int(record["image_id"]): record for record in records}
    require(len(indexed) == len(records), "candidate panel image identity")
    return indexed


def _row_text(tokenizer: Any, ids: Sequence[int]) -> str:
    text = tokenizer.decode(list(ids), skip_special_tokens=False)
    require(tokenizer.encode(text, add_special_tokens=False) == list(ids), "literal row tokenizer roundtrip")
    return text


def _parsed_literal(
    *, text: str, frozen: Mapping[str, Any], role: str
) -> dict[str, Any]:
    parsed = native_record(text, frozen["case"], frozen["golden"], "supplied_prefix")
    require(
        parsed.get("parse_status") == "accepted"
        and parsed.get("dropped_predictions") == []
        and len(parsed.get("pred", [])) == 1,
        f"{role} literal row parser",
    )
    return parsed["pred"][0]


def _sealed_row_descriptor(
    *,
    row: Mapping[str, Any],
    ordinal: int,
    tokenizer: Any,
    frozen: Mapping[str, Any],
    source_path: str | Path,
    source_kind: str,
) -> dict[str, Any]:
    rows = _source_rows(row)
    require(0 <= ordinal < len(rows), f"{source_kind} row ordinal")
    token_ids = rows[ordinal]
    text = _row_text(tokenizer, token_ids)
    prediction = _parsed_literal(text=text, frozen=frozen, role=f"{source_kind}[{ordinal}]")
    return {
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_row_sha256": _digest(row),
        "source_action_ids_sha256": _digest(row["action_ids"]),
        "source_row_ordinal": ordinal,
        "token_ids": token_ids,
        "token_ids_sha256": _digest(token_ids),
        "text": text,
        "token_count": len(token_ids),
        "description": prediction["description"],
        "bbox": list(prediction["bbox"]),
        "coord_bins": list(prediction["coord_bins"]),
        "image_id": int(row["image_id"]),
        "example_id": str(row["example_id"]),
    }


def _input_row_descriptor(
    *,
    record: Mapping[str, Any],
    frozen: Mapping[str, Any],
    tokenizer: Any,
    role: str,
) -> dict[str, Any]:
    token_ids = [int(value) for value in record["target_token_ids"]]
    text = _row_text(tokenizer, token_ids)
    prediction = _parsed_literal(text=text, frozen=frozen, role=role)
    return {
        "source_kind": "inputs-v2.literal_record",
        "source_record_id": record["record_id"],
        "source_record_sha256": _digest(record),
        "token_ids": token_ids,
        "token_ids_sha256": _digest(token_ids),
        "text": text,
        "token_count": len(token_ids),
        "description": prediction["description"],
        "bbox": list(prediction["bbox"]),
        "coord_bins": list(prediction["coord_bins"]),
        "image_id": int(record["image"]["image_id"]),
        "example_id": str(record["example_id"]),
        "candidate_owner_provenance": copy.deepcopy(record.get("candidate_owner_provenance")),
    }


def _owner_for_prediction(prediction: Mapping[str, Any], golden: Mapping[str, Any]) -> tuple[str, float]:
    # ``native_record`` predictions carry pixel-space ``bbox`` values while
    # the panel's GT records retain native coordinate-bin ``bbox`` values.
    # Candidate ownership must therefore compare the prediction's native
    # ``coord_bins`` to GT, not mix the two spaces.
    observed_box = prediction.get("coord_bins", prediction["bbox"])
    candidates = [
        (str(gt["object_id"]), iou_xyxy(observed_box, gt["bbox"]))
        for gt in golden["gt"]
        if gt["description"] == prediction["description"]
    ]
    require(candidates, "literal candidate has no same-class image owner")
    return max(candidates, key=lambda pair: pair[1])


def _trusted_target_by_id(trusted: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    targets = trusted.get("targets")
    require(isinstance(targets, list), "trusted target ledger")
    indexed = {str(item["record_id"]): item for item in targets}
    require(len(indexed) == len(targets), "trusted target record identity")
    return indexed


def _natural_evidence(
    row: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    registered: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize actual N16 rows; supplied prefixes never enter this read."""

    parsed = row.get("parsed", {})
    candidates = []
    for ordinal, prediction in enumerate(parsed.get("pred", [])):
        if prediction.get("description") not in {
            target["description"],
            registered["description"],
        }:
            continue
        target_iou = iou_xyxy(prediction["bbox"], target["bbox"])
        registered_iou = iou_xyxy(prediction["bbox"], registered["bbox"])
        candidates.append(
            {
                "ordinal": ordinal,
                "description": prediction["description"],
                "bbox": list(prediction["bbox"]),
                "target_iou": target_iou,
                "registered_iou": registered_iou,
            }
        )
    def _for(which: str, threshold: float = 0.50) -> dict[str, Any]:
        key = f"{which}_iou"
        matches = [item for item in candidates if item[key] >= threshold]
        return {
            "threshold": threshold,
            "present": bool(matches),
            "first_ordinal": matches[0]["ordinal"] if matches else None,
            "first_row_1_based": matches[0]["ordinal"] + 1 if matches else None,
            "ordinals": [item["ordinal"] for item in matches],
            "rows_1_based": [item["ordinal"] + 1 for item in matches],
            "max_iou": max((item[key] for item in candidates), default=0.0),
        }
    return {
        "image_id": int(row["image_id"]),
        "example_id": str(row["example_id"]),
        "row_source_sha256": _digest(row),
        "valid_row_count": len(parsed.get("pred", [])),
        "stop_reason": row.get("stop_reason", parsed.get("decode_stop_reason")),
        "target": _for("target"),
        "registered": _for("registered"),
        "candidates": candidates,
    }


def _strict_repeat_first(parsed: Mapping[str, Any]) -> int | None:
    seen: list[Sequence[float]] = []
    for ordinal, prediction in enumerate(parsed.get("pred", [])):
        if any(iou_xyxy(prediction["bbox"], old) > 0.95 for old in seen):
            return ordinal
        seen.append(prediction["bbox"])
    return None


def _validate_component_rows(rows: Mapping[str, Mapping[str, Any]]) -> None:
    require(set(rows) == {"H", "n", "S", "a", "b"}, "history component labels")
    require(rows["a"]["description"] == rows["b"]["description"], "candidate classes differ")
    require(rows["a"]["token_count"] == rows["b"]["token_count"], "candidate token lengths differ")
    require(rows["S"]["token_ids"], "common suffix must be nonempty")
    for name, row in rows.items():
        ids = row["token_ids"]
        require(ids and ids[0] == ROW_START and ids[-1] == ROW_END, f"{name} row grammar")
        require(EOS not in ids, f"{name} row contains EOS")


def _history_record(
    *, image: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]], label: str
) -> dict[str, Any]:
    _validate_component_rows(components)
    order = {
        "H_a_n_S": ("H", "a", "n", "S"),
        "H_b_n_S": ("H", "b", "n", "S"),
        "H_n_a_S": ("H", "n", "a", "S"),
        "H_n_b_S": ("H", "n", "b", "S"),
    }[label]
    token_ids = [token for name in order for token in components[name]["token_ids"]]
    class_counts = Counter(components[name]["description"] for name in order)
    return {
        "history": label,
        "component_order": list(order),
        "token_ids": token_ids,
        "token_ids_sha256": _digest(token_ids),
        "token_count": len(token_ids),
        "row_count": len(order),
        "class_counts": dict(sorted(class_counts.items())),
        "image_id": int(image["image_id"]),
        "example_id": str(image["example_id"]),
    }


def _validate_history_family(
    *, histories: Mapping[str, Mapping[str, Any]], components: Mapping[str, Mapping[str, Any]]
) -> None:
    _validate_component_rows(components)
    require(set(histories) == {"H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S"}, "history family labels")
    values = list(histories.values())
    require(len({item["token_count"] for item in values}) == 1, "matched history token counts")
    require(len({item["row_count"] for item in values}) == 1, "matched history row counts")
    require(len({tuple(sorted(item["class_counts"].items())) for item in values}) == 1, "matched history class counts")
    require(
        histories["H_a_n_S"]["token_ids"] != histories["H_b_n_S"]["token_ids"],
        "candidate fork did not change literal history",
    )
    require(
        histories["H_n_a_S"]["token_ids"] != histories["H_n_b_S"]["token_ids"],
        "late candidate fork did not change literal history",
    )
    expected_orders = {
        "H_a_n_S": ("H", "a", "n", "S"),
        "H_b_n_S": ("H", "b", "n", "S"),
        "H_n_a_S": ("H", "n", "a", "S"),
        "H_n_b_S": ("H", "n", "b", "S"),
    }
    for label, order in expected_orders.items():
        expected_ids = [token for name in order for token in components[name]["token_ids"]]
        require(histories[label]["component_order"] == list(order), f"{label} component order")
        require(histories[label]["token_ids"] == expected_ids, f"{label} literal component identity")
        require(order[-1] == "S", f"{label} common suffix")
    require(max(map(len, (item["token_ids"] for item in values))) < CAP, "history consumes all assistant budget")


def _source_row_for_image(rows: Sequence[Mapping[str, Any]], image_id: int) -> Mapping[str, Any]:
    matches = [row for row in rows if int(row["image_id"]) == image_id]
    require(len(matches) == 1, f"sealed rows image identity {image_id}")
    return matches[0]


def _candidate_records(inputs: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    positive = {str(row["record_id"]): row for row in inputs["positive_records"]}
    conditional = {str(row["record_id"]): row for row in inputs["conditional_records"]}
    require(spec["c_record_id"] in positive, "fixed c record missing from inputs-v2")
    require(spec["w_record_id"] in conditional, "fixed w record missing from inputs-v2")
    return positive[spec["c_record_id"]], conditional[spec["w_record_id"]]


def _make_components(
    *,
    spec: Mapping[str, Any],
    frozen: Mapping[str, Any],
    stable_row: Mapping[str, Any],
    tokenizer: Any,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    # H/n/S are intentionally taken from the same sealed Stable50 literal
    # record for both model arms.  The resulting four histories are identical
    # inputs to Stable50 and N16; N16 natural rows are evidence, not history
    # material for a second, outcome-dependent construction.
    context = {
        name: _sealed_row_descriptor(
            row=stable_row,
            ordinal=ordinal,
            tokenizer=tokenizer,
            frozen=frozen,
            source_path=STABLE_ROWS,
            source_kind="sealed_Stable50_context",
        )
        for name, ordinal in (("H", 0), ("n", 1), ("S", 2))
    }
    c_record, w_record = _candidate_records(inputs, spec)
    context["a"] = _input_row_descriptor(record=c_record, frozen=frozen, tokenizer=tokenizer, role="candidate a")
    context["b"] = _input_row_descriptor(record=w_record, frozen=frozen, tokenizer=tokenizer, role="candidate b")
    require(context["a"]["coord_bins"] == spec["target_bins"], "fixed target a bins changed")
    require(context["b"]["coord_bins"] == spec["registered_bins"], "fixed registered b bins changed")
    require(context["a"]["description"] == spec["class"] and context["b"]["description"] == spec["class"], "fixed candidate class")
    target_owner, target_iou = _owner_for_prediction(context["a"], frozen["golden"])
    registered_owner, registered_iou = _owner_for_prediction(context["b"], frozen["golden"])
    require(target_owner == spec["target_owner"], "target a owner identity")
    require(registered_owner != target_owner and registered_iou >= 0.50, "registered b physical owner")
    histories = {
        label: _history_record(image=frozen, components=context, label=label)
        for label in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S")
    }
    _validate_history_family(histories=histories, components=context)
    return context, {
        "histories": histories,
        "candidate_owners": {
            "a": {"owner_id": target_owner, "target_iou": target_iou},
            "b": {"owner_id": registered_owner, "registered_iou": registered_iou},
        },
    }


def _read_natural_evidence(
    *, image_id: int, frozen: Mapping[str, Any], components: Mapping[str, Mapping[str, Any]], n16_rows: Sequence[Mapping[str, Any]], stable_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    target = components["a"]
    registered = components["b"]
    n16 = _source_row_for_image(n16_rows, image_id)
    stable = _source_row_for_image(stable_rows, image_id)
    n16_record = _natural_evidence(n16, target=target, registered=registered)
    stable_record = _natural_evidence(stable, target=target, registered=registered)
    n16_record["first_strict_repeat_ordinal"] = _strict_repeat_first(n16["parsed"])
    stable_record["first_strict_repeat_ordinal"] = _strict_repeat_first(stable["parsed"])
    n16_record["first_strict_repeat_row_1_based"] = (
        n16_record["first_strict_repeat_ordinal"] + 1
        if n16_record["first_strict_repeat_ordinal"] is not None
        else None
    )
    stable_record["first_strict_repeat_row_1_based"] = (
        stable_record["first_strict_repeat_ordinal"] + 1
        if stable_record["first_strict_repeat_ordinal"] is not None
        else None
    )
    return {"N16": n16_record, "Stable50": stable_record}


def build_packet(*, output: str | Path = PREPARATION) -> dict[str, Any]:
    """Build and verify the no-model CPU packet; never launches CUDA work."""

    output = Path(output)
    _require_sources((INPUTS, PANEL, TRUSTED, STABLE_ROWS, STABLE_PROJECTION, N16_ROWS, N16_CONSUMER, RUNTIME_INPUT_JSONL))
    from transformers import AutoTokenizer

    inputs = _read(INPUTS)
    panel = _read(PANEL)
    trusted = _read(TRUSTED)
    stable_rows = _read_jsonl(STABLE_ROWS)
    n16_rows = _read_jsonl(N16_ROWS)
    require(len(stable_rows) == 640 and len(n16_rows) == 640, "sealed N16/Stable50 row denominator")
    require(_read(STABLE_PROJECTION)["status"] == "sealed_no_rerun_lossless_projection", "Stable50 projection seal")
    require(_read(N16_CONSUMER)["status"] == "cold_verified", "N16 natural consumer seal")
    tokenizer = AutoTokenizer.from_pretrained(
        panel["config"]["model"]["base_model"], local_files_only=True
    )
    panel_by_image = _panel_indexes(panel)
    trusted_by_id = _trusted_target_by_id(trusted)
    inputs_by_id = {
        str(row["record_id"]): row
        for key in ("positive_records", "conditional_records")
        for row in inputs[key]
    }
    cases: list[dict[str, Any]] = []
    hold_reasons: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        image_id = int(spec["image_id"])
        frozen = panel_by_image.get(image_id)
        if frozen is None:
            hold_reasons.append({"image_id": image_id, "reason": "fixed_image_missing_from_panel"})
            continue
        c_record = inputs_by_id.get(spec["c_record_id"])
        w_record = inputs_by_id.get(spec["w_record_id"])
        if c_record is None or w_record is None:
            hold_reasons.append({"image_id": image_id, "reason": "fixed_literal_record_missing_from_inputs_v2"})
            continue
        trusted_target = trusted_by_id.get(spec["c_record_id"])
        if trusted_target is None or trusted_target.get("final_review", {}).get("status") != "accept":
            hold_reasons.append({"image_id": image_id, "reason": "fixed_target_not_trusted_accept"})
            continue
        decision = trusted_target.get("final_review", {}).get("decision", {})
        if not decision.get("c_single_owner_absent_from_h") or not decision.get("w_single_owner_nonduplicate"):
            hold_reasons.append({"image_id": image_id, "reason": "fixed_target_w_physical_review_not_qualified"})
            continue
        if int(c_record["image"]["image_id"]) != image_id or int(w_record["image"]["image_id"]) != image_id:
            hold_reasons.append({"image_id": image_id, "reason": "candidate_image_identity_mismatch"})
            continue
        try:
            stable_row = _source_row_for_image(stable_rows, image_id)
            components, details = _make_components(
                spec=spec,
                frozen=frozen,
                stable_row=stable_row,
                tokenizer=tokenizer,
                inputs=inputs,
            )
        except (AssertionError, ValueError, KeyError) as exc:
            hold_reasons.append({"image_id": image_id, "reason": f"fixed_candidate_invariant:{exc}"})
            continue
        cases.append(
            {
                "image_id": image_id,
                "case_id": spec["case_id"],
                "example_id": frozen["example_id"],
                "class": spec["class"],
                "target_owner": spec["target_owner"],
                "target_bins": spec["target_bins"],
                "registered_bins": spec["registered_bins"],
                "image": {
                    "image_path": frozen["case"]["image_path"],
                    "image_sha256": frozen["case"]["image_plan"]["image_content_sha256"],
                    "executed_media_sha256": frozen["case"]["image_plan"]["executed_media_sha256"],
                    "image_width": frozen["case"]["image_width"],
                    "image_height": frozen["case"]["image_height"],
                    "observed_image_grid_thw": frozen["case"]["image_plan"]["observed_image_grid_thw"],
                },
                "panel_record_sha256": _digest(frozen),
                "panel_prompt_token_ids": list(frozen["prompt_token_ids"]),
                "components": components,
                "histories": details["histories"],
                "candidate_owners": details["candidate_owners"],
                "natural_evidence": _read_natural_evidence(
                    image_id=image_id,
                    frozen=frozen,
                    components=components,
                    n16_rows=n16_rows,
                    stable_rows=stable_rows,
                ),
                "source_records": {
                    "c": {"record_id": c_record["record_id"], "sha256": _digest(c_record)},
                    "w": {"record_id": w_record["record_id"], "sha256": _digest(w_record)},
                },
            }
        )
    require(not hold_reasons, "fixed history candidate HOLD; no replacement search")
    require([case["image_id"] for case in cases] == [210457, 219546], "fixed two-image order")
    all_histories = [history for case in cases for history in case["histories"].values()]
    require(len(all_histories) == 8, "four histories per image")
    packet = {
        "schema": "owner_successor_scale.history_packet.v1",
        "status": "candidate_cpu_verified_no_model_execution",
        "unit_id": "2026-09-13-owner-successor-scale-throughput",
        "question": "On the same identity-matched literal histories, does N16 differ from Stable50 in candidate-row score interaction and free successor burden?",
        "claim_boundary": "Diagnostic only; no latent ledger, training gate, architecture promotion, or natural-quality claim.",
        "design": {
            "histories": ["H+a+n+S", "H+b+n+S", "H+n+a+S", "H+n+b+S"],
            "components": "H/n/S are shared sealed Stable50 context rows; a is fixed accepted c; b is fixed registered w. Stable50 and N16 receive byte-identical history token IDs.",
            "candidate_set": [210457, 219546],
            "candidate_search": "disabled; any fixed-candidate qualification failure is HOLD",
            "common_suffix": "S is the final literal row in all four histories",
            "matching": "candidate a/b same class and exact token count; all four histories match total row/class/token counts",
            "natural_evidence_boundary": "N16/Stable50 natural rows are read-only evidence; forced history rows earn no free continuation credit",
        },
        "sources": {
            "inputs_v2": _binding(INPUTS),
            "candidate_panel": _binding(PANEL),
            "trusted_target_ledger": _binding(TRUSTED),
            "stable_rows": _binding(STABLE_ROWS),
            "stable_projection": _binding(STABLE_PROJECTION),
            "n16_rows": _binding(N16_ROWS),
            "n16_consumer": _binding(N16_CONSUMER),
            "runtime_input_jsonl": _binding(RUNTIME_INPUT_JSONL),
            "producer": _binding(Path(__file__).resolve()),
        },
        "adapter_arms": {
            "Stable50": {"root": str(STABLE_ADAPTER), "fingerprint": STABLE_FINGERPRINT},
            "N16": {"root": str(N16_ADAPTER), "fingerprint": N16_FINGERPRINT},
        },
        "runtime": {
            "physical_gpus": list(GPUS),
            "dtype": "fp32",
            "attention": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "max_assistant_tokens": CAP,
            "budget_rule": "len(supplied_history)+len(free_continuation) <= 3084",
            "kv_patch": False,
            "materialization_input_jsonl": str(RUNTIME_INPUT_JSONL),
        },
        "bounds": {
            "images": 2,
            "histories_per_image": 4,
            "models": 2,
            "full_conditional_continuations": 16,
            "literal_row_scores": "a/b for each model/image/history; no natural rerun",
        },
        "cases": cases,
        "counts": {
            "images": len(cases),
            "histories": len(all_histories),
            "full_continuations": 16,
            "fixed_candidate_rows": 4,
            "natural_source_rows": {"Stable50": len(stable_rows), "N16": len(n16_rows)},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _publish(output / "packet.json", packet)
    return packet


def validate_packet(
    packet: Mapping[str, Any],
    *,
    verify_sources: bool = True,
    allow_consumer_code_drift: bool = False,
) -> dict[str, Any]:
    require(packet.get("schema") == "owner_successor_scale.history_packet.v1", "history packet schema")
    require(packet.get("status") == "candidate_cpu_verified_no_model_execution", "history packet status")
    require(packet["runtime"]["physical_gpus"] == list(GPUS), "history GPU reservation")
    require(packet["runtime"]["max_assistant_tokens"] == CAP, "history cap")
    require(packet["runtime"]["materialization_input_jsonl"] == packet["sources"]["runtime_input_jsonl"]["path"], "runtime input binding")
    require(packet["design"]["candidate_search"] == "disabled; any fixed-candidate qualification failure is HOLD", "candidate replacement rule")
    require([int(case["image_id"]) for case in packet["cases"]] == [210457, 219546], "fixed image identities")
    for case in packet["cases"]:
        components = case["components"]
        _validate_history_family(histories=case["histories"], components=components)
        require(case["histories"]["H_a_n_S"]["component_order"][-1] == "S", "suffix row binding")
        require(case["histories"]["H_n_b_S"]["component_order"][-1] == "S", "suffix row binding")
        require(case["components"]["a"]["token_count"] == case["components"]["b"]["token_count"], "candidate token length parity")
        require(case["components"]["a"]["coord_bins"] == case["target_bins"], "target bins packet parity")
        require(case["components"]["b"]["coord_bins"] == case["registered_bins"], "registered bins packet parity")
    if verify_sources:
        for source_name, source in packet["sources"].items():
            # The endpoint packet intentionally binds the exact producer used
            # before CUDA launch.  A post-run consumer may be strengthened for
            # cold recomputation without relabeling the already-executed
            # producer; all data/model sources remain byte-checked.
            if allow_consumer_code_drift and source_name == "producer":
                continue
            require(file_hash(source["path"]) == source["sha256"], f"changed history source {source['path']}")
    return dict(packet)


def render_visuals(packet: Mapping[str, Any], output: str | Path = ROOT / "visuals") -> list[Path]:
    """Render the fixed candidate/context boxes without changing packet data."""

    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    colors = {"H": (56, 189, 248), "n": (168, 85, 247), "S": (148, 163, 184), "a": (239, 68, 68), "b": (245, 158, 11)}
    paths: list[Path] = []
    for case in packet["cases"]:
        image = Image.open(case["image"]["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for name, component in case["components"].items():
            box = tuple(int(value) for value in component["bbox"])
            color = colors[name]
            draw.rectangle(box, outline=color, width=4 if name in ("a", "b") else 2)
            label = f"{name}: {component['description']}"
            left, top, right, bottom = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle((left, top, right + 2, bottom + 2), fill=(0, 0, 0))
            draw.text((box[0] + 2, box[1] + 1), label, fill=color, font=font)
        legend = " | ".join(f"{name}={case['components'][name]['description']}" for name in ("H", "a", "b", "n", "S"))
        draw.rectangle((0, 0, image.width, 14), fill=(0, 0, 0))
        draw.text((4, 2), f"image {case['image_id']}  {legend}", fill=(255, 255, 255), font=font)
        path = output / f"image-{case['image_id']}-candidates.png"
        image.save(path)
        paths.append(path)
        image.close()
    if paths:
        images = [Image.open(path).convert("RGB") for path in paths]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        contact = Image.new("RGB", (width, height), "white")
        top = 0
        for image in images:
            contact.paste(image, (0, top))
            top += image.height
            image.close()
        contact_path = output / "candidate-history-overview.png"
        contact.save(contact_path)
        contact.close()
        paths.append(contact_path)
    return paths


def _target_interaction(
    *,
    prefix_parsed: Mapping[str, Any],
    free_parsed: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Count only free rows for candidate presence and re-entry."""

    def one(name: str) -> dict[str, Any]:
        target = components[name]
        prefix_matches = [
            ordinal
            for ordinal, prediction in enumerate(prefix_parsed.get("pred", []))
            if prediction.get("description") == target["description"]
            and iou_xyxy(prediction["bbox"], target["bbox"]) >= 0.50
        ]
        free_matches = []
        for ordinal, prediction in enumerate(free_parsed.get("pred", [])):
            if prediction.get("description") == target["description"]:
                overlap = iou_xyxy(prediction["bbox"], target["bbox"])
                if overlap >= 0.50:
                    free_matches.append({"ordinal": ordinal, "iou": overlap, "bbox": list(prediction["bbox"])})
        return {
            "prefix_forced_ordinals": prefix_matches,
            "free_present": bool(free_matches),
            "free_first_ordinal": free_matches[0]["ordinal"] if free_matches else None,
            "free_match_ordinals": [item["ordinal"] for item in free_matches],
            "free_later_reentry_count": max(0, len(free_matches) - 1),
            "free_matches": free_matches,
            "forced_rows_earn_no_credit": True,
        }

    return {"a": one("a"), "b": one("b")}


def _row_logprobs(
    *,
    model: Any,
    batch: Any,
    history_ids: Sequence[int],
    target_ids: Sequence[int],
) -> dict[str, Any]:
    prompt = list(batch.prompt_token_ids[0]) + list(history_ids)
    replay = prepare_replay(
        model,
        batch.inputs,
        prompt_token_ids=prompt,
        continuation_token_ids=list(target_ids),
        compact_logits=True,
    )
    with torch.inference_mode():
        outputs = model(**replay.inputs)
    logits = replay.aligned_logits(outputs.logits)
    target_tensor = replay.target_ids.to(device=logits.device)
    values = aligned_token_logprobs(logits, target_tensor).detach().float().cpu().tolist()
    require(len(values) == len(target_ids), "exact row score alignment")
    return {
        "token_logprobs": [float(value) for value in values],
        "sum_logprob": float(sum(values)),
        "mean_logprob": float(sum(values) / len(values)),
        "first_token_logprob": float(values[0]),
        "token_count": len(values),
    }


def _all_history_jobs(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for case in packet["cases"]:
        for history_name in ("H_a_n_S", "H_b_n_S", "H_n_a_S", "H_n_b_S"):
            jobs.append(
                {
                    "job_id": f"{case['image_id']}:{history_name}",
                    "image_id": int(case["image_id"]),
                    "example_id": case["example_id"],
                    "history": history_name,
                    "prefix_token_ids": case["histories"][history_name]["token_ids"],
                    "target_a_token_ids": case["components"]["a"]["token_ids"],
                    "target_b_token_ids": case["components"]["b"]["token_ids"],
                }
            )
    return jobs


def _history_jobs(packet: Mapping[str, Any], shard: int) -> list[dict[str, Any]]:
    """Return the optional two-way partition used by CPU scheduling tests."""

    require(shard in (0, 1), "history shard")
    return _all_history_jobs(packet)[shard::2]


def _check_model_identity(identity: Mapping[str, Any], arm: str, adapter: Mapping[str, Any]) -> None:
    live = identity["model_identity"]["adapter"]
    require(live["adapter_path"] == adapter["root"] and live["merged_adapters"] == [], f"{arm} adapter identity")
    observed = identity["effective_settings"]
    require(observed["observed_model_dtype"]["parameter_dtype_names"] == ["torch.float32"], f"{arm} FP32")
    require(observed["observed_attn_implementation"] == "sdpa", f"{arm} SDPA")


def endpoint_rank(
    packet_path: str | Path,
    endpoint_path: str | Path,
    output: str | Path,
    shard: int,
    physical_gpu: int,
) -> None:
    """Run one frozen model arm on one assigned GPU; root must invoke explicitly."""

    import torch
    from probes.dora_owner_learning.route_access import checkpoint_config
    from probes.dora_owner_learning.runtime import load_policy
    from src.config.inference import InferConfig
    from src.qwen.generation import NativeGenerationPolicy, generate_continuations

    packet = validate_packet(_read(packet_path))
    endpoint = _read(endpoint_path)
    require(endpoint["schema"] == "owner_successor_scale.history_endpoint.v1", "endpoint schema")
    require(endpoint["packet"] == _binding(packet_path), "endpoint packet binding")
    arm = endpoint["arm"]
    require(arm in ("Stable50", "N16"), "history model arm")
    require(physical_gpu == GPUS[shard], "history physical GPU assignment")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu) and torch.cuda.device_count() == 1, "single assigned GPU")
    output = Path(output)
    require(not output.exists(), "history shard output collision")
    output.mkdir(parents=True)
    started = time.monotonic()
    terminal: dict[str, Any] = {
        "schema": "owner_successor_scale.history_terminal.v1",
        "status": "running",
        "arm": arm,
        "shard": shard,
        "physical_gpu": physical_gpu,
        "jobs": len(_all_history_jobs(packet)),
        "model_loads": 0,
        "continuations": 0,
        "row_score_forwards": 0,
        "model_forwards": 0,
        "image_forwards": 0,
    }
    _publish(output / "launch.json", terminal)
    handles: list[Any] = []
    try:
        config_source = _runtime_config(_read(PANEL)["config"])
        config = checkpoint_config(InferConfig.model_validate(config_source), endpoint["adapter"]["root"])
        torch.cuda.reset_peak_memory_stats()
        qwen, identity = load_policy(config, device=torch.device("cuda:0"))
        _check_model_identity(identity, arm, endpoint["adapter"])
        terminal["model_loads"] = 1
        _publish(output / "model.json", {"arm": arm, "identity": identity, "adapter": endpoint["adapter"]})
        qwen.model.eval()
        for parameter in qwen.model.parameters():
            parameter.requires_grad_(False)

        def _model_forward(*_: Any) -> None:
            terminal["model_forwards"] += 1

        def _image_forward(*_: Any) -> None:
            terminal["image_forwards"] += 1

        handles.append(qwen.model.register_forward_pre_hook(_model_forward))
        visuals = [module for name, module in qwen.model.named_modules() if name.endswith("visual")]
        require(len(visuals) == 1, "single native visual module")
        handles.append(visuals[0].register_forward_pre_hook(_image_forward))
        panel = _read(PANEL)
        panel_by_image = _panel_indexes(panel)
        policy = NativeGenerationPolicy(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_model_defaults=False,
        )
        # Each assigned GPU owns one model arm, so it runs all eight histories
        # for that arm.  The helper's two-way partition is not a second model
        # shard and is intentionally not used here.
        jobs = _all_history_jobs(packet)
        with (output / "rows.jsonl").open("x") as stream:
            for job in jobs:
                case = next(item for item in packet["cases"] if int(item["image_id"]) == job["image_id"])
                frozen = panel_by_image[job["image_id"]]
                requests, _ = build_requests(qwen, config_source, [frozen["case"]])
                batch = prepare_native_inputs(qwen.processor, requests, device=torch.device("cuda:0"), record_media_identity=True)
                plan = frozen["case"]["image_plan"]
                require(list(batch.prompt_token_ids[0]) == frozen["prompt_token_ids"], "prompt identity")
                require(batch.media_sha256[0] == plan["executed_media_sha256"], "media identity")
                require(list(batch.image_grids[0]) == plan["observed_image_grid_thw"], "image grid identity")
                prefix = list(job["prefix_token_ids"])
                require(len(prefix) < CAP, "history cap before continuation")
                require(qwen.tokenizer.decode(prefix, skip_special_tokens=False) == "".join(case["components"][key]["text"] for key in case["histories"][job["history"]]["component_order"]), "history token/text identity")
                budget = CAP - len(prefix)
                with torch.inference_mode():
                    generated = generate_continuations(
                        qwen.model,
                        batch,
                        extensions=[prefix],
                        budgets=[budget],
                        eos_token_id=EOS,
                        pad_token_id=qwen.tokenizer.pad_token_id,
                        policy=policy,
                        trace="none",
                    )[0]
                free_ids = list(generated.token_ids)
                require(free_ids and len(prefix) + len(free_ids) <= CAP, "continuation budget")
                require(
                    (generated.stop_reason == "im_end" and free_ids[-1] == EOS and EOS not in free_ids[:-1])
                    or (generated.stop_reason == "length" and len(free_ids) == budget and EOS not in free_ids),
                    "native terminal identity",
                )
                free_text = qwen.tokenizer.decode(free_ids, skip_special_tokens=False)
                prefix_text = qwen.tokenizer.decode(prefix, skip_special_tokens=False)
                frozen_for_parser = {"case": frozen["case"], "golden": frozen["golden"]}
                ledger = continuation_ledger(prefix_text, free_text, frozen_for_parser, len(prefix), len(free_ids), generated.stop_reason)
                prefix_parsed = native_record(prefix_text, frozen["case"], frozen["golden"], "supplied_prefix")
                interactions = _target_interaction(prefix_parsed=prefix_parsed, free_parsed=ledger["free_parsed"], components=case["components"])
                row_scores = {}
                for candidate_name in ("a", "b"):
                    row_scores[candidate_name] = _row_logprobs(
                        model=qwen.model,
                        batch=batch,
                        history_ids=prefix,
                        target_ids=case["components"][candidate_name]["token_ids"],
                    )
                    terminal["row_score_forwards"] += 1
                row = {
                    "schema": "owner_successor_scale.history_row.v1",
                    "job_id": job["job_id"],
                    "arm": arm,
                    "shard": shard,
                    "physical_gpu": physical_gpu,
                    "packet": _binding(packet_path),
                    "endpoint": _binding(endpoint_path),
                    "image_id": job["image_id"],
                    "example_id": job["example_id"],
                    "history": job["history"],
                    "prefix_token_ids": prefix,
                    "prefix_token_count": len(prefix),
                    "free_token_ids": free_ids,
                    "free_token_ids_sha256": _digest(free_ids),
                    "free_text": free_text,
                    "stop_reason": generated.stop_reason,
                    "remaining_budget": budget,
                    "row_logprobs": row_scores,
                    "score_delta_a_minus_b": {
                        "sum_logprob": row_scores["a"]["sum_logprob"] - row_scores["b"]["sum_logprob"],
                        "mean_logprob": row_scores["a"]["mean_logprob"] - row_scores["b"]["mean_logprob"],
                        "first_token_logprob": row_scores["a"]["first_token_logprob"] - row_scores["b"]["first_token_logprob"],
                    },
                    "interactions": interactions,
                    "free_score": ledger["free_score"],
                    "full_score": ledger["full_score"],
                    "full_overlap_counts": ledger["full_overlap_counts"],
                    "burden": ledger["burden"],
                    "forced_rows_earn_no_credit": True,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                terminal["continuations"] += 1
        require(terminal["continuations"] == len(jobs), "history jobs completed")
        terminal.update(status="completed", exit_code=0)
    except BaseException as exc:
        terminal.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        _publish(output / "terminal.json", terminal)
    require(terminal["status"] == "completed", "history terminal completion")


def prepare_endpoint(packet_path: str | Path, arm: str, output_path: str | Path) -> dict[str, Any]:
    packet = validate_packet(_read(packet_path))
    require(arm in ("Stable50", "N16"), "history endpoint arm")
    adapter = packet["adapter_arms"][arm]
    require(Path(adapter["root"]).exists(), "history adapter root")
    endpoint = {
        "schema": "owner_successor_scale.history_endpoint.v1",
        "status": "ready_for_root_gpu_grant",
        "arm": arm,
        "packet": _binding(packet_path),
        "adapter": adapter,
        "physical_gpus": list(GPUS),
        "jobs_per_arm": len(_all_history_jobs(packet)),
        "bounds": {"max_continuations": 8, "max_row_scores": 16, "max_assistant_tokens": 8 * CAP},
        "no_training": True,
    }
    _publish(output_path, endpoint)
    return endpoint


def _arm_launch_specs() -> list[dict[str, Any]]:
    """Bind each model arm to its physical GPU and endpoint slot."""

    return [
        {"arm": arm, "physical_gpu": gpu, "shard": slot}
        for slot, (arm, gpu) in enumerate(zip(("Stable50", "N16"), GPUS, strict=True))
    ]


def launch(packet_path: str | Path, output: str | Path) -> None:
    """Explicit two-arm launcher; never called by CPU preparation."""

    packet = validate_packet(_read(packet_path))
    output = Path(output)
    require(not output.exists(), "history launch output collision")
    output.mkdir(parents=True)
    launches = []
    processes = []
    # One process per arm, each process owns both image histories on one GPU.
    for spec in _arm_launch_specs():
        arm = spec["arm"]
        gpu = spec["physical_gpu"]
        endpoint_path = output / f"{arm.lower()}-endpoint.json"
        prepare_endpoint(packet_path, arm, endpoint_path)
        arm_output = output / arm.lower()
        log = (output / f"{arm.lower()}.log").open("x")
        command = [
            sys.executable,
            "-m",
            "probes.owner_successor_scale.history",
            "endpoint-rank",
            "--packet",
            str(packet_path),
            "--endpoint",
            str(endpoint_path),
            "--output",
            str(arm_output),
            "--shard",
            str(spec["shard"]),
            "--physical-gpu",
            str(gpu),
        ]
        process = subprocess.Popen(
            command,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        launches.append({"arm": arm, "gpu": gpu, "pid": process.pid, "command": command})
        processes.append((process, log))
    _publish(output / "process-launch.json", {"packet": _binding(packet_path), "launches": launches})
    exits = []
    for launch_item, (process, log) in zip(launches, processes, strict=True):
        exits.append({**launch_item, "exit_code": process.wait()})
        log.close()
    _publish(output / "process-exits.json", {"results": exits})
    require(all(item["exit_code"] == 0 for item in exits), "history arm failure; preserve full outputs")


def merge(packet_path: str | Path, output: str | Path) -> dict[str, Any]:
    packet = validate_packet(_read(packet_path), allow_consumer_code_drift=True)
    output = Path(output)
    process_exits = _read(output / "process-exits.json")["results"]
    require(len(process_exits) == 2 and all(item["exit_code"] == 0 for item in process_exits), "history process exits")
    panel = _read(PANEL)
    panel_by_image = _panel_indexes(panel)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(panel["config"]["model"]["base_model"], local_files_only=True)
    expected_jobs = {job["job_id"]: job for job in _all_history_jobs(packet)}
    results: dict[str, list[dict[str, Any]]] = {}
    cost: dict[str, dict[str, Any]] = {}
    recomputed_rows = 0
    for arm in ("Stable50", "N16"):
        arm_root = output / arm.lower()
        endpoint_path = output / f"{arm.lower()}-endpoint.json"
        endpoint = _read(endpoint_path)
        require(endpoint["arm"] == arm and endpoint["packet"] == _binding(packet_path), f"{arm} endpoint identity")
        terminal = _read(arm_root / "terminal.json")
        require(
            terminal["status"] == "completed"
            and terminal["arm"] == arm
            and terminal["jobs"] == 8
            and terminal["continuations"] == 8
            and terminal["model_loads"] == 1
            and terminal["row_score_forwards"] == 16,
            f"{arm} terminal",
        )
        rows = _read_jsonl(arm_root / "rows.jsonl")
        require(len(rows) == 8, f"{arm} history denominator")
        seen_jobs: set[str] = set()
        for row in rows:
            job_id = row.get("job_id")
            require(job_id in expected_jobs and job_id not in seen_jobs, f"{arm} row job identity")
            seen_jobs.add(job_id)
            job = expected_jobs[job_id]
            require(
                row["arm"] == arm
                and row["image_id"] == job["image_id"]
                and row["example_id"] == job["example_id"]
                and row["history"] == job["history"]
                and row["prefix_token_ids"] == job["prefix_token_ids"]
                and row["prefix_token_count"] == len(job["prefix_token_ids"])
                and row["remaining_budget"] == CAP - len(job["prefix_token_ids"])
                and row["forced_rows_earn_no_credit"],
                f"{arm} literal history identity",
            )
            require(row["packet"] == _binding(packet_path), f"{arm} packet row binding")
            require(row["endpoint"] == _binding(endpoint_path), f"{arm} endpoint row binding")
            free_ids = [int(value) for value in row["free_token_ids"]]
            require(free_ids and len(job["prefix_token_ids"]) + len(free_ids) <= CAP, f"{arm} terminal budget")
            require(
                (row["stop_reason"] == "im_end" and free_ids[-1] == EOS and EOS not in free_ids[:-1])
                or (row["stop_reason"] == "length" and len(free_ids) == row["remaining_budget"] and EOS not in free_ids),
                f"{arm} terminal token identity",
            )
            require(row["free_token_ids_sha256"] == _digest(row["free_token_ids"]), f"{arm} free token digest")
            require(tokenizer.decode(free_ids, skip_special_tokens=False) == row["free_text"], f"{arm} free text identity")
            frozen = panel_by_image[int(row["image_id"])]
            prefix_text = tokenizer.decode(job["prefix_token_ids"], skip_special_tokens=False)
            fresh = continuation_ledger(
                prefix_text,
                row["free_text"],
                {"case": frozen["case"], "golden": frozen["golden"]},
                len(job["prefix_token_ids"]),
                len(free_ids),
                row["stop_reason"],
            )
            require(row["free_score"] == fresh["free_score"], f"{arm} free score consumer recompute")
            require(row["full_score"] == fresh["full_score"], f"{arm} full score consumer recompute")
            require(row["full_overlap_counts"] == fresh["full_overlap_counts"], f"{arm} overlap consumer recompute")
            prefix_parsed = native_record(prefix_text, frozen["case"], frozen["golden"], "supplied_prefix")
            case = next(item for item in packet["cases"] if int(item["image_id"]) == int(row["image_id"]))
            interactions = _target_interaction(
                prefix_parsed=prefix_parsed,
                free_parsed=fresh["free_parsed"],
                components=case["components"],
            )
            require(row["interactions"] == interactions, f"{arm} interaction consumer recompute")
            require(row["burden"] == fresh["burden"], f"{arm} burden consumer recompute")
            scores = row["row_logprobs"]
            for name, target in (("a", job["target_a_token_ids"]), ("b", job["target_b_token_ids"])):
                value = scores[name]
                require(value["token_count"] == len(target) and len(value["token_logprobs"]) == len(target), f"{arm} {name} row score length")
                require(all(math.isfinite(float(item)) for item in value["token_logprobs"]), f"{arm} {name} row score finite")
            require(
                row["score_delta_a_minus_b"]["sum_logprob"]
                == scores["a"]["sum_logprob"] - scores["b"]["sum_logprob"],
                f"{arm} row score delta",
            )
            recomputed_rows += 1
        results[arm] = rows
        cost[arm] = {
            key: terminal[key]
            for key in (
                "elapsed_seconds",
                "model_loads",
                "continuations",
                "row_score_forwards",
                "model_forwards",
                "image_forwards",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
                "peak_rss_bytes",
                "physical_gpu",
            )
        }
    require(recomputed_rows == 16, "exact 16-row consumer recompute")
    merged = {
        "schema": "owner_successor_scale.history_result.v1",
        "status": "merged_cold_verified",
        "packet": _binding(packet_path),
        "consumer_producer": _binding(Path(__file__).resolve()),
        "arms": results,
        "counts": {"arms": 2, "images": 2, "histories_per_arm": 8, "full_continuations": 16, "consumer_recomputed_rows": recomputed_rows},
        "cost": {
            "arms": cost,
            "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in cost.values()),
            "total_gpu_hours": sum(item["elapsed_seconds"] for item in cost.values()) / 3600.0,
            "total_model_forwards": sum(item["model_forwards"] for item in cost.values()),
            "total_image_forwards": sum(item["image_forwards"] for item in cost.values()),
            "total_row_score_forwards": sum(item["row_score_forwards"] for item in cost.values()),
        },
        "claim_boundary": packet["claim_boundary"],
        "reporting": {
            "row_score_interaction": "Use per-token and mean a/b logprobs plus a-minus-b deltas.",
            "presence": "Use free continuation only; supplied candidate rows are listed as forced and earn no credit.",
            "reentry": "Free candidate match ordinals and later reentry count are reported per candidate.",
            "burden": "Use native continuation ledger free/full score, overlap counts, cap/EOS and parser burden.",
        },
    }
    _publish(output / "result.json", merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=PREPARATION)
    verify = sub.add_parser("verify")
    verify.add_argument("--packet", type=Path, default=PREPARATION / "packet.json")
    ep = sub.add_parser("endpoint-prepare")
    ep.add_argument("--packet", type=Path, default=PREPARATION / "packet.json")
    ep.add_argument("--arm", choices=("Stable50", "N16"), required=True)
    ep.add_argument("--output", type=Path, required=True)
    rank = sub.add_parser("endpoint-rank")
    rank.add_argument("--packet", type=Path, required=True)
    rank.add_argument("--endpoint", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--shard", type=int, required=True)
    rank.add_argument("--physical-gpu", type=int, required=True)
    visualize = sub.add_parser("visualize")
    visualize.add_argument("--packet", type=Path, default=PREPARATION / "packet.json")
    visualize.add_argument("--output", type=Path, default=ROOT / "visuals")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--packet", type=Path, default=PREPARATION / "packet.json")
    launch_parser.add_argument("--output", type=Path, required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--packet", type=Path, default=PREPARATION / "packet.json")
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        packet = build_packet(output=args.output)
        visuals = render_visuals(packet)
        print(json.dumps({"status": packet["status"], "packet": str(args.output / "packet.json"), "visuals": [str(path) for path in visuals], "counts": packet["counts"]}, sort_keys=True))
    elif args.command == "verify":
        packet = validate_packet(_read(args.packet))
        print(json.dumps({"status": packet["status"], "packet": str(args.packet)}, sort_keys=True))
    elif args.command == "endpoint-prepare":
        endpoint = prepare_endpoint(args.packet, args.arm, args.output)
        print(json.dumps({"status": endpoint["status"], "arm": endpoint["arm"], "jobs_per_arm": endpoint["jobs_per_arm"]}, sort_keys=True))
    elif args.command == "endpoint-rank":
        endpoint_rank(args.packet, args.endpoint, args.output, args.shard, args.physical_gpu)
    elif args.command == "visualize":
        packet = validate_packet(_read(args.packet))
        paths = render_visuals(packet, args.output)
        print(json.dumps({"status": "visuals_written", "paths": [str(path) for path in paths]}, sort_keys=True))
    elif args.command == "launch":
        launch(args.packet, args.output)
    else:
        result = merge(args.packet, args.output)
        print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
