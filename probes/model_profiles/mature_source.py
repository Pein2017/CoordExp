"""Mature original-policy saved-source adapter for native research assays.

This adapter preserves raw rows, companion order, literal tokens and media
identity. Legacy cases without image plans are enriched in place as before.
Its census-producer file is provenance data, not an executable dependency.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import torch
from src.artifacts.utf8_json import literal_binding as _binding
from src.qwen.input_identity import input_identity as _input_identity
from src.config.inference import InferConfig
from src.data.examples import raw_example_from_jsonl_row
from src.inference.bound_requests import build_bound_native_requests
from src.inference.inputs import plan_examples
from src.qwen.native import prepare_native_inputs
REPO = Path(__file__).resolve().parents[2]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def token_hash(tokens: list[int]) -> str:
    return digest(tokens)


def _group(panel: dict[str, Any], key: str) -> dict[str, Any]:
    groups = panel.get("groups", [])
    matches = [x for x in groups if isinstance(x, dict) and x.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"source runtime group is not unique: {key}")
    return matches[0]


def load_saved_source(boundary: dict[str, Any], model: str, source_panel: dict[str, Any], q: Any, device: torch.device) -> tuple[Any, list[dict[str, Any]], dict[str, Any], Any, dict[str, Any]]:
    group = _group(source_panel, str(boundary["group"]))
    raw_path = Path(boundary["raw_path"])
    trace_path = Path(boundary["trace_path"])
    receipt_path = Path(boundary["receipt_path"])
    for path in (raw_path, trace_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = json.loads(raw_path.read_text())["rows"]
    trace = json.loads(trace_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "candidate_complete":
        raise ValueError("native source receipt is not complete")
    if receipt.get("condition") != f"{model}-original":
        raise ValueError("native source condition differs from declared model")
    if receipt.get("group") != boundary["group"]:
        raise ValueError("native source group differs")
    if receipt.get("raw") != _binding(raw_path) or receipt.get("trace") != _binding(trace_path):
        raise ValueError("native source receipt bindings differ")
    config = dict(source_panel["configs"][model])
    config["data"] = dict(input_jsonl=group["input_jsonl"])
    cases = group["cases"]
    if any("image_plan" not in case for case in cases):
        # New census groups use the same source-owned planning seam as the
        # accepted natural producer.  The panel intentionally stores the raw
        # cases, while the native source receipt binds the resulting input
        # identity.  Replan only this mechanical format seam and require the
        # saved receipt identity below.
        config_model = InferConfig.model_validate(source_panel["configs"][model])
        raw_examples = [
            raw_example_from_jsonl_row(
                case["input_record"],
                jsonl_path=Path(group["input_jsonl"]),
                row_number=int(case["row_index"]) + 1,
                raw_line=json.dumps(case["input_record"]),
            )
            for case in cases
        ]
        planned = plan_examples(
            raw_examples,
            config=config_model,
            components=q,
            row_indices=[int(case["row_index"]) for case in cases],
        )
        for case, item in zip(cases, planned, strict=True):
            case["image_path"] = item.image.image_path
            case["image_plan"] = item.image.to_artifact_dict()
        requests = [item.request for item in planned]
        planning = {
            "replanned_image_plan": True,
            "case_indices": [int(case["row_index"]) for case in cases],
            "image_plan_bindings": [digest(case["image_plan"]) for case in cases],
            "producer_path": str(Path(__file__).resolve().parents[2] / 'probes/recurrence_dynamics/recurrence_census/natural.py'),
            "producer_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / 'probes/recurrence_dynamics/recurrence_census/natural.py').read_bytes()).hexdigest(),
        }
    else:
        requests, _ = build_bound_native_requests(q, config, cases)
        planning = {"replanned_image_plan": False}
    batch = prepare_native_inputs(q.processor, requests, device=device, record_media_identity=True)
    if receipt.get("input_identity") != _input_identity(batch):
        raise ValueError("native prompt/media identity changed")
    index = int(boundary["batch_index"])
    if len(raw) != len(group["cases"]) or int(raw[index]["image_id"]) != int(boundary["image_id"]):
        raise ValueError("native companion or image identity changed")
    native = [int(x) for x in raw[index]["token_ids"]]
    if native != [int(x) for x in boundary["native_tokens"]]:
        raise ValueError("native token identity changed")
    if token_hash(native) != str(boundary["native_token_hash"]):
        raise ValueError("native token hash changed")
    if not isinstance(trace.get("steps"), list):
        raise ValueError("native trace has no steps")
    planning["receipt_input_identity"] = receipt.get("input_identity")
    planning["replayed_input_identity"] = _input_identity(batch)
    return batch, raw, trace, group, planning
